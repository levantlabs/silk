# SPDX-License-Identifier: MIT
"""Nutmeg (SPICE) rawfile reader and writer.

Copyright (c) 2026 Levant Labs. Distributed under silk's MIT LICENSE.

Reads the ``.raw`` output written by ngspice (``write``) and Xyce
(``.print <analysis> format=raw``), in both the binary (``Binary:``) and ASCII
(``Values:``) encodings — validated against real output from ngspice-42 and
Xyce 7.10, which are the reader's scope.

Text (the header and the ASCII ``Values:`` block) is decoded as ISO-8859-1
(latin-1), the nutmeg convention: ngspice and Xyce write latin-1-safe bytes, and
a 1:1 latin-1 map never fails and never mangles a stray non-ASCII byte in a
``Title:`` comment. The ``Binary:`` payload is never text-decoded — it is read
straight from the byte buffer with :func:`numpy.frombuffer`.

:func:`write_rawfile` is the inverse: it serialises :class:`RawPlot` objects
(or a :class:`RawFile`) in either encoding, so a result that did not come from
a SPICE simulator can be read by every consumer of rawfiles.

Format summary
--------------
A rawfile is a sequence of *plots*, each a text header followed by a data
block, concatenated in one file::

    Title: <str>
    Date: <str>
    Plotname: <str>
    Flags: real|complex [padded|unpadded] [...]
    No. Variables: <int>
    No. Points: <int>
    [Offset: <float>]
    [Command: <str>]
    [Dimensions: <int>[,<int>...]]
    Variables:
    <TAB><index><TAB><name><TAB><type>[<TAB><attr>...]
    ... (No. Variables lines)
    Binary:      <- raw little-endian float64, or
    Values:      <- ASCII

Keywords are matched case-insensitively. Data is stored point-major: for each
point, one value per variable, in ``Variables:`` order. A ``real`` plot stores
one float64 per value; a ``complex`` plot stores two (real part then imaginary
part), which is byte-identical to little-endian ``complex128``.

Vector 0 is the independent axis (time / frequency / sweep). Its name varies by
simulator and analysis (``time``, ``frequency``, ``sweep``, ``v(v-sweep)``,
``temp-sweep``, and the upper-cased Xyce spellings), so it is identified
positionally, never by name, and vector order is preserved exactly as written.
In a ``complex`` plot that axis is stored as (re, im) pairs like every other
column, but it is handed back as ``float64`` when its imaginary part is
negligible — see :func:`_realify_scale`, which documents why the test is a
tolerance rather than exact zero and why tightening it would reintroduce a bug.

Multiple plots
--------------
:attr:`RawFile.plots` holds every plot in file order. :attr:`RawFile.current_plot`
is the **last** plot, and the top-level ``nvars`` / ``npoints`` / ``vectors`` /
``real`` / ``padded`` mirror it. "Last plot" is the plot a simulator leaves
current when it finishes, which is what both of silk's rawfile guards assume:
:func:`silk.simulation.backend._has_data_rows` treats a trailing ``constants``
plot as an aborted run, and ``NgspiceSignal.from_rawfile`` reads ``vectors[0]``
as the independent axis.
"""

import hashlib
import os

import numpy as np

from silk.exceptions import SimulationError

__all__ = ['read_rawfile', 'write_rawfile', 'RawFile', 'RawPlot', 'RawVector', 'RawfileError']


class RawfileError(SimulationError):
    """Raised when a file is not a parseable nutmeg rawfile."""


# Header keywords that introduce something other than a plain scalar value.
# How far below the scale's own magnitude the imaginary part must sit before a
# complex plot's independent axis is treated as real. See _realify_scale.
_SCALE_IMAG_RTOL = 1e-12

_KW_VARIABLES = 'variables'
_KW_BINARY = 'binary'
_KW_VALUES = 'values'


class RawVector:
    """One variable (column) of a plot.

    Attributes
    ----------
    name : str
        Verbatim as written in the file — never case-folded, because Xyce
        upper-cases node names and silk matches case-insensitively downstream.
    type : str
        The nutmeg type string, e.g. ``'time'``, ``'voltage'``, ``'current'``.
    attrs : tuple of str
        Trailing tokens on the ``Variables:`` line (``grid=3``, ``min=…``, …).
        Preserved but not interpreted.
    data : numpy.ndarray
        ``float64`` for a real plot, ``complex128`` for a complex one.
    """

    __slots__ = ('name', 'type', 'attrs', '_data', '_store', '_slot', '_length')

    def __init__(self, name, type='', attrs=(), data=None):  # noqa: A002
        self.name = name
        self.type = type
        self.attrs = tuple(attrs)
        self._store = None
        self._slot = None
        self.data = np.empty(0, dtype=np.float64) if data is None else data

    @property
    def data(self):
        """Writable samples, copied from a lazy backing file on first access."""
        if self._data is None:
            self._data = self._store.materialize(self._slot, self.name)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self._length = len(value)
        self._store = None
        self._slot = None

    @property
    def _is_complex(self):
        """Sample format without materializing an unread column.

        Once data exists (including caller replacements), its actual dtype
        wins over the original file format.
        """
        if self._data is not None:
            return np.iscomplexobj(self._data)
        return not self._store.plots[self._slot[0]][3]

    def __len__(self):
        return len(self._data) if self._data is not None else self._length

    def __repr__(self):
        return (f'RawVector(name={self.name!r}, type={self.type!r}, '
                f'npoints={len(self)})')


class RawPlot:
    """One plot: a header plus its variables and their data."""

    def __init__(self, header, vectors, npoints, real, padded):
        self.header = header
        self.vectors = vectors
        self.npoints = npoints
        self.real = real
        self.padded = padded

    # -- header conveniences -------------------------------------------------
    @property
    def title(self):
        return self.header.get('title', '')

    @property
    def date(self):
        return self.header.get('date', '')

    @property
    def plotname(self):
        return self.header.get('plotname', '')

    @property
    def command(self):
        return self.header.get('command', '')

    @property
    def flags(self):
        """The ``Flags:`` tokens, lower-cased."""
        return tuple(self.header.get('flags', '').lower().split())

    @property
    def offset(self):
        """The ``Offset:`` value, already applied to vector 0. 0.0 if absent."""
        return _to_float(self.header.get('offset'), 0.0)

    @property
    def dimensions(self):
        raw = self.header.get('dimensions', '')
        return tuple(int(t) for t in raw.replace(',', ' ').split() if t)

    @property
    def numdims(self):
        """Number of entries in ``Dimensions:``; 0 when the keyword is absent
        (an ordinary one-dimensional plot)."""
        return len(self.dimensions)

    @property
    def nvars(self):
        return len(self.vectors)

    def __repr__(self):
        return (f'RawPlot(plotname={self.plotname!r}, nvars={self.nvars}, '
                f'npoints={self.npoints}, real={self.real})')


class RawFile:
    """The parsed rawfile: every plot, plus a view onto the current one.

    The scalar/vector attributes (:attr:`nvars`, :attr:`npoints`,
    :attr:`vectors`, :attr:`real`, :attr:`padded`) mirror :attr:`current_plot`,
    i.e. the last plot in the file.
    """

    def __init__(self, plots, source_path=None):
        self.plots = list(plots)
        self.source_path = source_path

    @property
    def current_plot(self):
        """The last plot in the file, or ``None`` if the file held none."""
        return self.plots[-1] if self.plots else None

    @property
    def vectors(self):
        plot = self.current_plot
        return plot.vectors if plot is not None else []

    @property
    def nvars(self):
        plot = self.current_plot
        return plot.nvars if plot is not None else 0

    @property
    def npoints(self):
        plot = self.current_plot
        return plot.npoints if plot is not None else 0

    @property
    def numdims(self):
        plot = self.current_plot
        return plot.numdims if plot is not None else 0

    @property
    def real(self):
        plot = self.current_plot
        return plot.real if plot is not None else True

    @property
    def padded(self):
        plot = self.current_plot
        return plot.padded if plot is not None else None

    def get_plots(self):
        """Every plot, in file order."""
        return list(self.plots)

    def __repr__(self):
        return (f'RawFile(plots={len(self.plots)}, nvars={self.nvars}, '
                f'npoints={self.npoints})')


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def read_rawfile(path, *, lazy=False):
    """Parse the nutmeg rawfile at *path*.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a ``.raw`` file written by ngspice or Xyce.
    lazy : bool
        Opt in to indexing without loading waveform arrays. Each vector is
        copied on first access and retained by this reader. Unread vectors
        require the original file to remain unchanged; already copied arrays
        stay writable and valid after file changes. Opening streams the file
        to validate ASCII data and compute a content fingerprint.

    Returns
    -------
    RawFile

    Raises
    ------
    RawfileError
        If the file contains no recognisable plot header.
    """
    if lazy:
        return _read_lazy(path)
    path = str(path)
    with open(path, 'rb') as handle:
        data = handle.read()

    plots = []
    pos = 0
    end = len(data)
    while pos < end:
        plot, pos = _parse_plot(data, pos)
        if plot is None:
            break
        plots.append(plot)

    if not plots:
        raise RawfileError(
            f'{path!r} is not a nutmeg rawfile: no plot header found '
            f'(expected a "Plotname:"/"No. Variables:" header followed by '
            f'"Binary:" or "Values:").'
        )
    from hashlib import sha256
    raw = RawFile(plots, source_path=path)
    raw._content_sha256 = sha256(data).hexdigest()
    return raw


def _readline(data, pos):
    """Return ``(line_text, next_pos)``; *line_text* has no line terminator.

    Text is decoded as ISO-8859-1 (latin-1): the nutmeg convention, a lossless
    1:1 byte map that never raises. Only ever called on header and ASCII-value
    lines — the ``Binary:`` payload is read via :func:`numpy.frombuffer`, not here.
    """
    if isinstance(data, _StreamIndex):
        return data.readline(pos)
    nl = data.find(b'\n', pos)
    if nl == -1:
        return data[pos:].decode('iso-8859-1').rstrip('\r'), len(data)
    return data[pos:nl].decode('iso-8859-1').rstrip('\r'), nl + 1


def _to_int(text, default=0):
    if text is None:
        return default
    # ngspice pads some counts with trailing commentary; take the leading int.
    token = text.strip().split()[0] if text.strip() else ''
    try:
        return int(token)
    except ValueError:
        return default


def _to_float(text, default=0.0):
    if text is None:
        return default
    try:
        return float(text.strip().split()[0])
    except (ValueError, IndexError):
        return default


def _split_fields(line):
    """Split a ``Variables:`` line into fields.

    Tab-delimited when tabs are present (so a name containing a space survives),
    otherwise on arbitrary whitespace.
    """
    if '\t' in line:
        return [f for f in line.split('\t') if f]
    return line.split()


def _parse_variable(line):
    """Parse one ``Variables:`` line into a :class:`RawVector`.

    The line is ``<index> <name> <type> [<attr>...]``. Fields are tab-separated,
    but trailing attributes are *space*-separated and ngspice packs them into
    the same tab-delimited field as the type — a real AC rawfile writes
    ``\\t0\\tfrequency\\tfrequency grid=3``. So the type field is re-split on
    whitespace: its first token is the type, the rest are attributes.
    """
    fields = _split_fields(line)
    name = fields[1] if len(fields) > 1 else ''
    tail = ' '.join(fields[2:]).split()
    vtype = tail[0] if tail else ''
    return RawVector(name, vtype, tail[1:])


def _parse_plot(data, pos):
    """Parse one plot starting at or after *pos*.

    Returns ``(plot, next_pos)``, or ``(None, next_pos)`` when no further plot
    header is present (trailing blank lines, or a simulator footer).
    """
    header = {}
    vectors = []
    end = len(data)

    while pos < end:
        line, pos = _readline(data, pos)
        stripped = line.strip()
        if not stripped:
            continue

        key, sep, value = stripped.partition(':')
        if not sep:
            # Not a header line — a simulator footer banner ("End of Xyce(TM)
            # Simulation") or trailing junk. Ignore it unless we are mid-way
            # through a genuine plot header, in which case the file is broken.
            if _is_plot_header(header):
                raise RawfileError(
                    f'malformed rawfile: expected a "Keyword: value" header '
                    f'line or a data section, got {line!r}'
                )
            header.clear()
            continue

        key = key.strip().lower()
        value = value.strip()

        if key == _KW_BINARY:
            return _finish(header, vectors, data, pos, binary=True)
        if key == _KW_VALUES:
            return _finish(header, vectors, data, pos, binary=False)

        if key == _KW_VARIABLES:
            nvars = _to_int(header.get('no. variables'), -1)
            if nvars < 0:
                raise RawfileError(
                    'malformed rawfile: "Variables:" reached before a valid '
                    '"No. Variables:" count'
                )
            # Some writers put the first variable on the "Variables:" line.
            pending = [value] if value else []
            while len(vectors) < nvars:
                if pending:
                    varline = pending.pop()
                else:
                    if pos >= end:
                        raise RawfileError(
                            f'malformed rawfile: header declares {nvars} '
                            f'variables but only {len(vectors)} are present'
                        )
                    varline, pos = _readline(data, pos)
                if not varline.strip():
                    continue
                vectors.append(_parse_variable(varline))
            continue

        header[key] = value

    # Ran off the end without a data section. Only a genuine plot header makes
    # that an error; anything else is a trailing footer we can ignore.
    if _is_plot_header(header):
        raise RawfileError(
            'malformed rawfile: plot header is not followed by a "Binary:" '
            'or "Values:" data section'
        )
    return None, pos


def _is_plot_header(header):
    """True if the accumulated keywords identify a real plot (as opposed to a
    simulator footer that happens to contain ``Keyword: value`` lines)."""
    return 'plotname' in header or 'no. variables' in header


def _finish(header, vectors, data, pos, binary):
    """Read one plot's data block and assemble the :class:`RawPlot`."""
    if isinstance(data, _StreamIndex):
        return data.finish(header, vectors, pos, binary)
    flags = header.get('flags', '').lower().split()
    real = 'complex' not in flags
    # `padded`/`unpadded` describe multi-dimensional data padded out to a
    # rectangular array. This reports only what the file *declares* — None when
    # neither token is present — because the published format does not define a
    # default and we decline to assume one implementation's. Nothing here
    # branches on it: the data block is sized from `No. Points` x `No. Variables`
    # and cross-checked against the bytes actually present, both observable.
    padded = _declared_padding(flags)
    npoints = _to_int(header.get('no. points'), 0)
    nvars = len(vectors)

    if nvars == 0 or npoints <= 0:
        # An empty data block: "No. Points: 0" is exactly what an aborted
        # analysis leaves behind, and must parse cleanly so the data-presence
        # gate — not a parse error — reports the abort.
        dtype = np.float64 if real else np.complex128
        for vec in vectors:
            vec.data = np.empty(0, dtype=dtype)
        _realify_scale(vectors, real)
        return RawPlot(header, vectors, 0, real, padded), pos

    if binary:
        columns, npoints, pos = _read_binary(data, pos, nvars, npoints, real)
    else:
        columns, npoints, pos = _read_ascii(data, pos, nvars, npoints, real)

    for index, vec in enumerate(vectors):
        vec.data = columns[index]
        if 'dims=0' in vec.attrs:
            # ngspice reserves the column but writes a placeholder value for a
            # vector it could not fill; ``dims=0`` marks it zero-length. Honour
            # the marker so the placeholder never reads back as a real 0.0.
            vec.data = vec.data[:0]

    _realify_scale(vectors, real)
    plot = RawPlot(header, vectors, npoints, real, padded)

    # "Offset:" carries a constant (a transient's tstart) to be added to the
    # independent axis. Vector 0 is that axis.
    offset = plot.offset
    if offset and vectors:
        vectors[0].data = vectors[0].data + offset

    return plot, pos


def _declared_padding(flags):
    """``True``/``False`` when ``Flags:`` declares ``padded``/``unpadded``,
    ``None`` when it declares neither.

    Real ngspice and Xyce rawfiles write a bare ``Flags: real`` and never emit
    ``padded``, so the undeclared case is the common one. No parsing decision
    depends on this value.
    """
    if 'padded' in flags:
        return True
    if 'unpadded' in flags:
        return False
    return None


def _realify_scale(vectors, real):
    """Give a complex plot a real-valued independent axis.

    A ``complex`` plot stores *every* column as an (re, im) pair, the scale
    included — an AC rawfile writes ``1.000e+03,0.000e+00`` for each frequency.
    An analysis scale (time, frequency, sweep variable) is a real physical
    quantity, and silk consumes it as one: the metrics do ``float(sig.x[i])``
    and ``np.interp`` over it, and ``np.interp`` *raises* on complex input. So
    vector 0 is handed back as ``float64`` whenever its imaginary part is
    negligible.

    Negligible, not exactly zero — that distinction is deliberate, please do
    not tighten it. Real ngspice-42 AC rawfiles written through the bare ``-r``
    batch path carry uninitialised *denormal* imaginary noise in the frequency
    scale (~1e-308, differing between invocations; visible as
    ``1.0e+00,4.86e-310`` in the ASCII sibling, so it is what ngspice writes,
    not a parsing artifact). Against frequencies of 1e0–1e9 that detritus is
    ~300 orders of magnitude down, so a relative test separates it from genuine
    complex content with no realistic false positive. An exact-zero test would
    instead hand back a complex scale for such files and blow up in
    ``np.interp`` — a trap that only springs on rawfiles silk did not write
    itself, i.e. on ``backend.load_raw``.

    The *data* vectors always stay complex; only the scale gets this treatment.
    """
    if real or not vectors:
        return
    scale = vectors[0].data
    if scale.size:
        # Absolute floor covers the degenerate all-zero-real case and always
        # admits denormals, which are below the smallest normal float.
        tolerance = max(_SCALE_IMAG_RTOL * float(np.max(np.abs(scale.real))),
                        float(np.finfo(np.float64).tiny))
        if float(np.max(np.abs(scale.imag))) > tolerance:
            return
    vectors[0].data = np.ascontiguousarray(scale.real)


def _read_binary(data, pos, nvars, npoints, real):
    """Little-endian float64, point-major. Complex = interleaved (re, im),
    which is exactly the little-endian ``complex128`` memory layout."""
    dtype = np.dtype('<f8') if real else np.dtype('<c16')
    itemsize = dtype.itemsize
    available = (len(data) - pos) // itemsize
    wanted = npoints * nvars
    if available < wanted:
        # A truncated rawfile (killed simulator): keep the whole points we got
        # rather than failing, so the data-presence gate can report the abort.
        npoints = available // nvars
        wanted = npoints * nvars
    flat = np.frombuffer(data, dtype=dtype, count=wanted, offset=pos)
    table = flat.reshape(npoints, nvars)
    columns = [np.ascontiguousarray(table[:, i]) for i in range(nvars)]
    return columns, npoints, pos + wanted * itemsize


def _read_ascii(data, pos, nvars, npoints, real):
    """ASCII ``Values:`` block.

    Each point starts with a line ``<point-index><TAB><value>``; the remaining
    variables follow on tab-indented continuation lines. A complex value is
    written ``real,imag``.
    """
    dtype = np.float64 if real else np.complex128
    table = np.empty((npoints, nvars), dtype=dtype)
    end = len(data)
    filled = 0

    for point in range(npoints):
        row = []
        first = True
        while len(row) < nvars:
            if pos >= end:
                break
            line, pos = _readline(data, pos)
            stripped = line.strip()
            if not stripped:
                continue
            tokens = _value_tokens(stripped)
            if first:
                # Leading token is the point index; drop it.
                tokens = tokens[1:]
                first = False
            row.extend(tokens)
        if len(row) < nvars:
            break  # truncated file — keep the complete points
        for index in range(nvars):
            table[point, index] = _parse_value(row[index], real)
        filled += 1

    if filled < npoints:
        npoints = filled
        table = table[:npoints]
    columns = [np.ascontiguousarray(table[:, i]) for i in range(nvars)]
    return columns, npoints, pos


def _value_tokens(line):
    """Whitespace-split a values line, keeping ``real,imag`` pairs together
    even when a writer puts a space after the comma."""
    tokens = []
    for token in line.split():
        if tokens and (tokens[-1].endswith(',') or token.startswith(',')):
            tokens[-1] += token
        else:
            tokens.append(token)
    return tokens


def _parse_value(token, real):
    if real:
        return float(token)
    part_re, _, part_im = token.partition(',')
    return complex(float(part_re), float(part_im or 0.0))


# Lazy readers share only immutable index metadata, never cached sample arrays.
# No open file descriptors or mappings survive an indexing/materialization call.
_READ_BLOCK = 1024 * 1024


def _file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, stat.st_ctime_ns)


class _LazyStore:
    def __init__(self, path, identity):
        self.path = path
        self.identity = identity
        self.plots = ()

    def __deepcopy__(self, memo):
        # This descriptor is immutable after indexing. RawVector's ordinary
        # deepcopy copies its populated _data using the caller's memo, so
        # aliases to that same array elsewhere in copied state survive.
        memo[id(self)] = self
        return self

    def error(self, vector, detail):
        return RawfileError(
            f"Rawfile {self.path!r}, vector {vector!r}: {detail}. "
            "Reload the rawfile or preserve the original output file."
        )

    def check(self, handle, vector):
        if (_file_identity(os.stat(self.path)) != self.identity or
                _file_identity(os.fstat(handle.fileno())) != self.identity):
            raise self.error(vector, "backing file changed since it was opened")

    def materialize(self, slot, name):
        generation = getattr(self, 'generation', None)
        if generation is None:
            return self._materialize(slot, name)
        try:
            with generation.read():
                self.path = generation.resolve('rawfile_path')
                self.identity = generation.manifest['rawfile_path'][2]
                return self._materialize(slot, name)
        except SimulationError as exc:
            if isinstance(exc, RawfileError):
                raise
            raise self.error(name, f'generation {generation.ident}: {exc}') from exc

    def _materialize(self, slot, name):
        plot_index, column = slot
        pos, npoints, nvars, real, binary, offset, empty = self.plots[plot_index]
        dtype = np.dtype('<f8' if real else '<c16')
        try:
            with open(self.path, 'rb') as handle:
                self.check(handle, name)
                if column in empty or npoints == 0:
                    values = np.empty(0, dtype=dtype)
                elif binary:
                    values = np.empty(npoints, dtype=dtype)
                    handle.seek(pos)
                    # Fixed-size chunks, even when one row exceeds a chunk.
                    # Select the column by its slot, not its possibly repeated name.
                    total = npoints * nvars
                    start = filled = 0
                    while start < total:
                        count = min(total - start, _READ_BLOCK // dtype.itemsize)
                        block = handle.read(count * dtype.itemsize)
                        if len(block) != count * dtype.itemsize:
                            raise self.error(name, "backing file was truncated during read")
                        flat = np.frombuffer(block, dtype=dtype)
                        selected = flat[(column - start) % nvars::nvars]
                        values[filled:filled + len(selected)] = selected
                        filled += len(selected)
                        start += count
                else:
                    stream = _StreamIndex(handle, self, name)
                    values, count, _ = _scan_ascii(
                        stream, pos, nvars, npoints, real, column)
                    if count != npoints:
                        raise self.error(name, "backing file was truncated during read")
                self.check(handle, name)
        except OSError as exc:
            raise self.error(name, str(exc)) from exc
        if column == 0:
            scale = RawVector(name, data=values)
            _realify_scale([scale], real)
            values = scale.data
            if offset and npoints:
                values = values + offset
        return values


class _StreamIndex:
    """Seekable line source for the shared header parser, with lazy payloads."""

    def __init__(self, handle, store, vector="<index>"):
        self.handle = handle
        self.store = store
        self.vector = vector

    def __len__(self):
        return self.store.identity[2]

    def readline(self, pos):
        self.handle.seek(pos)
        line = self.handle.readline()
        if not line and pos < len(self):
            raise self.store.error(self.vector, 'backing file was truncated during read')
        return line.decode('iso-8859-1').rstrip('\r\n'), self.handle.tell()

    def finish(self, header, vectors, pos, binary):
        flags = header.get('flags', '').lower().split()
        real = 'complex' not in flags
        nvars = len(vectors)
        npoints = max(0, _to_int(header.get('no. points'), 0)) if nvars else 0
        start = pos
        if npoints:
            if binary:
                stride = nvars * (8 if real else 16)
                npoints = min(npoints, (len(self) - pos) // stride)
                pos += npoints * stride
            else:
                _, npoints, pos = _scan_ascii(self, pos, nvars, npoints, real)
        slot = len(self.store.plots)
        empty = tuple(i for i, v in enumerate(vectors) if 'dims=0' in v.attrs)
        offset = _to_float(header.get('offset'), 0.0)
        self.store.plots += ((start, npoints, nvars, real, binary, offset, empty),)
        for i, vector in enumerate(vectors):
            vector._data = None
            vector._length = 0 if i in empty else npoints
            vector._store = self.store
            vector._slot = (slot, i)
        return RawPlot(header, vectors, npoints, real, _declared_padding(flags)), pos


def _scan_ascii(data, pos, nvars, npoints, real, column=None):
    """Validate complete rows with one row of tokens and at most one column.

    Match the eager parser's token, truncation and plot-boundary rules. The
    index pass stores no point offsets and allocates no waveform arrays.
    """
    values = None if column is None else np.empty(
        npoints, dtype=np.float64 if real else np.complex128)
    filled = 0
    for point in range(npoints):
        row = []
        first = True
        while len(row) < nvars:
            if pos >= len(data):
                break
            line, pos = data.readline(pos)
            if not line.strip():
                continue
            tokens = _value_tokens(line.strip())
            if first:
                tokens = tokens[1:]
                first = False
            row.extend(tokens)
        if len(row) < nvars:
            break
        for index in range(nvars):
            try:
                value = _parse_value(row[index], real)
            except ValueError:
                # A concurrent overwrite may first appear as a broken token.
                data.store.check(data.handle, data.vector)
                raise
            if index == column:
                values[point] = value
        filled += 1
    return values, filled, pos


def _read_lazy(path):
    source_path = str(path)
    # Retain the pathname (including symlinks) so replacement of its target
    # is detected, but freeze relative paths against later cwd changes.
    path = os.path.join(os.getcwd(), source_path)
    store = _LazyStore(path, None)
    try:
        with open(path, 'rb') as handle:
            store.identity = _file_identity(os.fstat(handle.fileno()))
            store.check(handle, '<index>')
            digest = hashlib.sha256()
            while block := handle.read(_READ_BLOCK):
                digest.update(block)
            store.check(handle, '<index>')
            data = _StreamIndex(handle, store)
            plots = []
            pos = 0
            try:
                while pos < len(data):
                    plot, pos = _parse_plot(data, pos)
                    if plot is None:
                        break
                    plots.append(plot)
            finally:
                # A concurrent rewrite may otherwise look like malformed input.
                store.check(handle, '<index>')
    except OSError as exc:
        raise store.error('<index>', str(exc)) from exc
    if not plots:
        raise RawfileError(
            f'{source_path!r} is not a nutmeg rawfile: no plot header found '
            f'(expected a "Plotname:"/"No. Variables:" header followed by '
            f'"Binary:" or "Values:").'
        )
    raw = RawFile(plots, source_path=source_path)
    raw._content_sha256 = digest.hexdigest()
    return raw


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

# Header keywords written first, in this order; any others follow, in the
# order the plot holds them. ``offset`` is never written (see write_rawfile).
_HEADER_ORDER = ('title', 'date', 'plotname', 'flags')
_HEADER_COUNTS = ('no. variables', 'no. points', 'variables')


def write_rawfile(path, plots, binary=True):
    """Write *plots* to a nutmeg rawfile at *path*.

    Parameters
    ----------
    path : str or os.PathLike
        Destination; overwritten.
    plots : RawFile, RawPlot or iterable of RawPlot
        What to write, in file order. The last plot is the one a reader treats
        as current.
    binary : bool
        ``True`` (default) writes a ``Binary:`` block of little-endian float64
        (``complex128`` for a complex plot). ``False`` writes ``Values:`` text
        with 17 significant digits, which round-trips every float64 exactly.

    Notes
    -----
    * ``No. Variables``, ``No. Points`` and the ``real``/``complex`` flag are
      derived from the vectors, never from the plot's stale header copy. Other
      header keywords (``Title``, ``Date``, ``Plotname``, ``Command``,
      ``Dimensions``, ...) are written as held.
    * ``Offset:`` is not written. :func:`read_rawfile` has already added it to
      vector 0, so re-emitting it would shift the axis twice on read-back.
    * A vector the reader marked ``dims=0`` holds no data; it is written as a
      zero placeholder column, with its attributes kept, so it reads back
      empty again.

    Raises
    ------
    RawfileError
        If a plot has no vectors, vectors of unequal length, a complex vector
        in a real plot, or a header value containing a line break.
    """
    if isinstance(plots, RawFile):
        plots = plots.plots
    elif isinstance(plots, RawPlot):
        plots = [plots]
    plots = list(plots)
    if not plots:
        raise RawfileError('write_rawfile: no plots to write')

    chunks = [_encode_plot(plot, binary) for plot in plots]
    with open(str(path), 'wb') as handle:
        for chunk in chunks:
            handle.write(chunk)


def _header_text(value):
    text = str(value)
    if '\n' in text or '\r' in text:
        raise RawfileError(
            f'write_rawfile: header value {text!r} contains a line break')
    return text


def _encode_plot(plot, binary):
    vectors = list(plot.vectors)
    if not vectors:
        raise RawfileError(
            f'write_rawfile: plot {plot.plotname!r} has no vectors')
    real = bool(plot.real)
    npoints = max(len(v.data) for v in vectors)
    dtype = np.dtype('<f8') if real else np.dtype('<c16')

    columns = []
    for vec in vectors:
        data = np.asarray(vec.data)
        if data.size == 0 and 'dims=0' in vec.attrs:
            data = np.zeros(npoints, dtype=dtype)
        elif len(data) != npoints:
            raise RawfileError(
                f'write_rawfile: vector {vec.name!r} has {len(data)} points, '
                f'expected {npoints}')
        if real and np.iscomplexobj(data):
            raise RawfileError(
                f'write_rawfile: vector {vec.name!r} is complex but plot '
                f'{plot.plotname!r} is flagged real')
        columns.append(data.astype(dtype, copy=False))

    header = dict(plot.header)
    flags = [t for t in header.get('flags', '').split()
             if t.lower() not in ('real', 'complex')]
    flags.insert(0, 'real' if real else 'complex')
    header['flags'] = ' '.join(flags)
    header.setdefault('title', '')
    header.setdefault('date', '')
    header.setdefault('plotname', '')
    for key in _HEADER_COUNTS + ('offset', 'binary', 'values'):
        header.pop(key, None)

    keys = [k for k in _HEADER_ORDER if k in header]
    keys += [k for k in header if k not in _HEADER_ORDER]
    lines = []
    for key in keys:
        value = _header_text(header[key])
        label = key.title()
        lines.append(f'{label}: {value}')
        if key == 'flags':
            lines.append(f'No. Variables: {len(vectors)}')
            lines.append(f'No. Points: {npoints}')
    lines.append('Variables:')
    for index, vec in enumerate(vectors):
        fields = [str(index), _header_text(vec.name), _header_text(vec.type)]
        if vec.attrs:
            fields[2] += ' ' + ' '.join(_header_text(a) for a in vec.attrs)
        lines.append('\t' + '\t'.join(fields))
    lines.append('Binary:' if binary else 'Values:')
    head = ('\n'.join(lines) + '\n').encode('iso-8859-1', errors='replace')

    if npoints == 0:
        return head
    table = np.stack(columns, axis=1)  # point-major
    if binary:
        return head + np.ascontiguousarray(table).tobytes()
    out = []
    for point in range(npoints):
        row = [_format_value(x, real) for x in table[point]]
        out.append(f'{point}\t{row[0]}\n')
        out.extend(f'\t{r}\n' for r in row[1:])
    return head + ''.join(out).encode('ascii')


def _format_value(x, real):
    if real:
        return repr(float(x))
    return f'{float(x.real)!r},{float(x.imag)!r}'
