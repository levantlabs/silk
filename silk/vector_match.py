"""The one vector matcher: which rows of a plot a signal request names.

Shared by the simulator reader (``silk.signals.spice``, which re-exports
:func:`collect_vector_candidates`) and the waveform view layer, which uses it
to prove that a bracket spelling addresses exactly one row of its source plot. Its JS twin lives in ``waveform-core.mjs``; both are checked
against ``tests/fixtures/vector_match_cases.json``.

A stdlib-only leaf: it imports only :mod:`silk.spice_names`. A *plot* is any
object with ``vectors`` (each with ``name``), optionally ``nvars``, ``plotname``,
``npoints`` and ``real``.
"""
from __future__ import annotations

import re

from silk.spice_names import to_designer_name, to_spice_name

__all__ = ['collect_vector_candidates', 'first_row', 'vector_address']

_OP_PLOTNAME = 'operating point'
_WRAPPED = re.compile(r'([vi])\((.*)\)', re.S | re.I)
_UNSAFE_PATH = re.compile(r'[(),@#\s\'"\\]')


def _request_forms(signal_name, signal_type):
    """The spellings a request may mean: as given, and the simulator spelling
    (bracket instance parts converted), lower-cased, ':' read as '.'."""
    kind = 'branch' if str(signal_type).upper() == 'I' else 'node'
    return {signal_name.lower().replace(':', '.'),
            to_spice_name(signal_name, kind=kind).lower().replace(':', '.')}


def collect_vector_candidates(plot, signal_name, signal_type='V', *, first=1):
    """Indices (identity) of every vector of *plot* that *signal_name* names.

    THE one matcher behind ``NgspiceSignal.from_rawfile`` and so
    ``SimulatorBackend.load_signal``: the union of all rows matching
    the as-given OR the simulator spelling is gathered BEFORE choosing, and a
    row reached by both spellings is one candidate. Matching is case
    insensitive and reads Xyce's ``:`` as ``.``. A bare vector name (noise
    spectra) matches itself; a wrapped ``v(...)``/``i(...)`` row matches when
    its inner path equals a request form and its wrapper equals
    *signal_type*. The caller decides: zero is a missing signal, one is the
    result, several are ambiguous. Rows are counted by identity.
    """
    if getattr(signal_name, '_literal_voltage', False):
        # Exact literal node syntax can itself contain selectors/colons. Keep
        # the shared collector, but never reinterpret the target's characters.
        # The structural walk has already emitted known instance components.
        # Leaf/connected-net text must never gain a hierarchical alias.
        forms = {form.lower() for form in signal_name.voltage_forms()}
        found = []
        for i in range(first, len(plot.vectors)):
            match = _WRAPPED.fullmatch(plot.vectors[i].name)
            if match is not None and match[1].lower() == 'v' and match[2].lower() in forms:
                found.append(i)
            elif match is None and plot.vectors[i].name.lower() in forms:
                # Some external files store voltage rows without v(...).
                # Match literally, without interpreting leaf punctuation.
                found.append(i)
        return found
    forms = _request_forms(signal_name, signal_type)
    wanted = str(signal_type).lower()
    nvars = getattr(plot, 'nvars', None)
    if nvars is None:
        nvars = len(plot.vectors)
    found = []
    for i in range(first, nvars):
        name = plot.vectors[i].name
        if name.lower().replace(':', '.') in forms:
            found.append(i)
            continue
        head, paren, rest = name.partition('(')
        if not paren or '(' in rest:
            continue
        if (rest.rstrip(')').lower().replace(':', '.') in forms
                and head.lower() == wanted):
            found.append(i)
    return found


def first_row(plot):
    """The first row the reader may match: 0 for an operating point (it has no
    independent axis: vector 0 is a node or a placeholder), 1 for every plot
    with an axis. The rule of ``rawfile_analysis_kind``: a complex plot is
    never an operating point; a real plot is one when its plotname is
    ``Operating Point`` or it holds exactly one point."""
    if getattr(plot, 'real', True) is False:
        return 1
    plotname = (getattr(plot, 'plotname', '') or '').lower()
    if plotname == _OP_PLOTNAME or getattr(plot, 'npoints', None) == 1:
        return 0
    return 1


def vector_address(plot, row):
    """The proven bracket address of vector *row* of *plot*, or ``None``.

    The address is the codec's bracket alias of the raw name, in the wrapped
    ``v(...)``/``i(...)`` form, and is attached only when the reader's own
    matcher resolves the request ``tb.probe`` would send for it to exactly
    this row of the whole source plot (so a hidden row, a duplicate row, a
    bare or colon/case twin all withhold it). Anything unrecognised returns
    ``None``: omission is always safe.
    """
    name = plot.vectors[row].name
    alias = to_designer_name(name)
    if alias == name:
        return None
    match = _WRAPPED.fullmatch(alias)
    if not match:
        return None
    inner = match.group(2).strip()
    if not inner or _UNSAFE_PATH.search(inner):
        return None
    kind = 'I' if match.group(1).lower() == 'i' else 'V'
    found = collect_vector_candidates(plot, inner, kind, first=first_row(plot))
    return alias if found == [row] else None
