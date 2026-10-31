"""silk.simulation.backend — simulator backends.

A backend owns the full produce-a-Signal pipeline for one simulator —
rendering analyses, emitting the netlist, running, and loading results into
:class:`~silk.signals.base.Signal` objects — selected by name from the
``SimulatorBackend`` registry. Backend selection is the single dispatch point;
there is no separate per-simulator Signal registry.
"""
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from silk.utils._utils import _fmt
from silk.simulation.analysis import Transient, AC, DC, Noise, Op
from silk.simulation.run_files import RunFiles
from silk.exceptions import NotSupportedError


# Terminal-current vector suffixes ngspice emits under `savecurrents`
# (spec 2026-07-03 §6.2). MOS: id/ig/is/ib; two-terminal / branch: i/current/branch.
_CURRENT_SUFFIXES = ('id', 'ig', 'is', 'ib', 'i', 'current', 'branch')

#: The fixed 17-parameter op-point device vector list,
#: reused by the emission side (``spice_netlister.generate_control_block``,
#: which emits ``save`` for exactly these names) and the read side
#: (:meth:`silk.simulation.backend.SpiceBackend.device_op_vectors` below), so the two can never
#: drift apart. Order is authorial, not semantic.
DEVICE_OP_PARAMS = (
    'id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm', 'gds', 'gmbs',
    'cgg', 'cgs', 'cgd', 'cgso', 'cgdo', 'cgbo', 'capbd', 'capbs',
)

#: The fixed subset, verified on Xyce 7.10, of :data:`DEVICE_OP_PARAMS` Xyce's
#: ``N(<dev>:PARAM)``/``ID(<dev>)`` forms expose at all. ``gmbs`` and every
#: capacitance (``cgg``/``cgs``/``cgd``/``cgso``/``cgdo``/``cgbo``) are
#: confirmed UNAVAILABLE ("Function or variable ... " refusal, measured
#: directly) — not a documentation gap, a real absence. Order is the
#: :data:`DEVICE_OP_PARAMS` subsequence, not semantic.
XYCE_OP_CAPABILITY = tuple(
    p for p in DEVICE_OP_PARAMS
    if p in {'id', 'vgs', 'vds', 'vbs', 'vth', 'vdsat', 'gm', 'gds'}
)


def _contiguous_sublist(needle, haystack) -> bool:
    """True if the *needle* segment list appears as a contiguous run in
    *haystack*. Used to match a device's ``path.leaf`` token inside a rawfile
    ``@``-vector's dotted body regardless of the internal device name."""
    n = len(needle)
    if n == 0:
        return False
    return any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def _identity(name: str) -> str:
    """Return *name* unchanged: the default ``EmitContext.resolve_source``.

    Parameters
    ----------
    name : str
        A logical silk source name.

    Returns
    -------
    str
        The same name, so the logical name is emitted as the instance name.
    """
    return name


_CHECKVALID_WARNING = re.compile(
    r'^Warning from checkvalid: vector (\S+) is not available or has zero length\.',
    re.MULTILINE)


def _checkvalid_warned(raw, vector_name: str) -> bool:
    """True if THIS run's own ngspice log recorded a ``Warning from
    checkvalid: vector <vector_name> is not available or has zero
    length.`` line.

    Build-time finding: ngspice can populate such a vector with a filler
    ``0.`` (length 1, never actually zero-length) while still printing
    this exact warning — vector *length* alone cannot tell a genuine 0
    apart from this filler, so every read checks the log too. Silent
    (returns ``False``) when there is no log to check — an externally
    supplied :class:`RawResult`, or a backend/launch mode with no log
    file — so a probe never fails *because* it cannot find a log; it
    only ever fails because the log it DOES have says so.
    """
    from silk.simulation.generation import artifact_read
    with artifact_read(raw, 'logfile_path') as log_path:
        if log_path is None:
            return False
        try:
            text = Path(log_path).read_text(errors='replace')
        except OSError as exc:
            if getattr(raw, '_generation', None) is not None:
                from silk.exceptions import SimulationError
                raise SimulationError(f'Cannot read retained log {log_path!r}: {exc}') from exc
            return False
    target = vector_name.lower()
    return any(m.group(1).lower() == target
              for m in _CHECKVALID_WARNING.finditer(text))


def _ngspice_provided_device_params(raw, plot, address, exclude_param) -> list:
    """Every :data:`DEVICE_OP_PARAMS` name (other than *exclude_param*)
    this device's *plot* genuinely provides — a real, populated,
    non-``checkvalid``-warned vector — for the "this run DID provide"
    half of a device-parameter availability diagnostic. Shared by
    :meth:`NgspiceBackend._device_probe_read_error` and
    :meth:`NgspiceBackend.preflight_unavailable_error`."""
    leaf = address.leaf.lower()
    path = address.instance_path.lower().strip('.')
    token = (path.split('.') if path else []) + [leaf]
    provided = []
    for candidate in DEVICE_OP_PARAMS:
        if candidate == exclude_param:
            continue
        ctail = f'[{candidate}]'
        cmatches = []
        for vec in plot.vectors:
            inner = vec.name.lower()
            if (inner.startswith('i(') or inner.startswith('v(')) and inner.endswith(')'):
                inner = inner[2:-1]
            if not inner.startswith('@') or not inner.endswith(ctail):
                continue
            body = inner[1:-len(ctail)]
            if _contiguous_sublist(token, body.split('.')):
                cmatches.append(vec)
        if (len(cmatches) == 1 and len(cmatches[0].data)
                and not _checkvalid_warned(raw, cmatches[0].name)):
            provided.append(candidate)
    return provided


def _probe_analysis_label(analysis, plot) -> str:
    """The word a device-probe diagnostic shows for
    ``analysis=...``: a
    probe that never passed ``analysis=`` explicitly used to print the
    literal ``analysis=None``, even though silk itself always resolves
    an actual plot to read (a single-plot rawfile needs no ``analysis=``
    at all). Falls back, in order, to the silk analysis word the plot
    itself was recognised as (``'tran'``/``'ac'``/``'dc'``/``'noise'``/``'op'``),
    then to the plot's own simulator-written name when it carries no such word — never the bare ``None`` a
    caller simply didn't have to supply."""
    if analysis is not None:
        return analysis
    from silk.signals.analysis_kind import plot_silk_analysis
    word = plot_silk_analysis(plot)
    if word is not None:
        return word
    plotname = (getattr(plot, 'plotname', '') or '').strip()
    return plotname or '(unknown analysis)'


# Canonical noise-signal names -> port ('out' = output-referred, 'in' =
# input-referred). Accepts the ngspice-native and terse aliases too, so existing
# code (and `onoise_spectrum`/`inoise_spectrum`) keeps resolving. The canonical
# convention is amplitude spectral density, V/√Hz.
_NOISE_PORT = {
    'noise_out': 'out', 'onoise_spectrum': 'out', 'onoise': 'out',
    'noise_in': 'in', 'inoise_spectrum': 'in', 'inoise': 'in',
}


@dataclass
class RawResult:
    """The run-to-load handoff returned by :meth:`SimulatorBackend.run`.

    Only ``load_signal(raw, ...)`` is shared across backends; the shape of
    ``native`` is backend-private.

    Parameters
    ----------
    native : object
        The backend's native result payload. For the SPICE backends this is
        the :class:`~silk.signals.rawfile.RawFile` read from the run's
        rawfile.
    source_path : str
        The file ``native`` was read from, or the input deck for an
        in-memory backend.
    run_files : RunFiles, optional
        The prepared companion paths, retained for later loading. ``None``
        when the result did not come from a prepared run.
    ambient_temperatures : tuple of float, optional
        Ambient temperatures the launch reported solving at, in degrees
        Celsius. Filled in by :meth:`SimulatorBackend.run`; ``()`` means the
        backend produced no evidence.
    simulator_version : str, optional
        The simulator binary's own version as reported by this run; see the
        field note below.

    Examples
    --------
    >>> from silk.simulation.backend import RawResult
    >>> raw = RawResult(native=None, source_path='rawfile.raw')
    >>> raw.ambient_temperatures, raw.simulator_version
    ((), None)
    """
    #: The backend's native result payload; for the SPICE backends a
    #: :class:`~silk.signals.rawfile.RawFile`. Its shape is backend-private.
    native: Any
    #: The file ``native`` was read from, or the input deck for an in-memory
    #: backend.
    source_path: str
    #: The prepared companion paths retained for later loading; ``None`` when
    #: the result did not come from a prepared run.
    run_files: RunFiles | None = None
    #: Ambient temperatures, in degrees Celsius, that the launch reported
    #: having solved at, as read by
    #: :meth:`SimulatorBackend.read_ambient_temperatures`. Empty when the
    #: backend produced no evidence.
    ambient_temperatures: tuple[float, ...] = ()
    #: This simulator BINARY's own version, as reported by THIS run's log/
    #: transcript — e.g. ``'ngspice-47'``, ``'Xyce Release
    #: 7.10.0-opensource'``. ``None`` when unreadable (an externally
    #: supplied result, or a backend that has not implemented
    #: ``read_simulator_version``). Never obtained by invoking the binary
    #: separately — a device-
    #: parameter probe's availability diagnostic needs the version that
    #: actually PRODUCED the rawfile now being read, not the version on
    #: this machine's PATH right now.
    simulator_version: Optional[str] = None


#: Method name a **non-nutmeg** native result may implement to answer the
#: data-presence question about itself. ``_has_data_rows`` checks it
#: before the nutmeg shape, so a backend whose reader returns something other
#: than a :class:`~silk.signals.rawfile.RawFile` gets abort detection by
#: defining one method — without leaking its native shape into this shared
#: layer, and without the alternative seam (overriding
#: :meth:`SimulatorBackend.has_data`) if it would rather not.
DATA_PRESENCE_HOOK = 'silk_has_data'


def _has_data_rows(native) -> bool:
    """True if a loaded native result carries at least one data row.

    Resolution order:

    1. **The protocol hook.** If *native*'s class defines
       :data:`DATA_PRESENCE_HOOK` (``silk_has_data()``), its bool answer wins.
       This is the seam a non-nutmeg native answers through. Like Python's own
       protocols it is looked up on the **type**, not the instance — a stand-in
       that answers every ``getattr`` (a ``MagicMock``) must not be read as
       having implemented it.
    2. **The nutmeg shape** (``.npoints`` / ``.vectors``), which both
       SPICE-family backends share because both read their rawfile through
       silk's nutmeg reader.
    3. **Neither** — raise. silk cannot invent a verdict for a result shape it
       does not know: guessing "no data" would flag every run from such a
       backend as an abort (a false *red*), and guessing "data" would
       reintroduce the false green the gate exists to kill.

    ngspice's write-with-no-analysis fallback: when an analysis aborts, the run
    can leave the synthetic ``constants`` plot (pi/boltz/echarge/…) as the only
    (or last) plot in the rawfile. It reports ``npoints >= 1`` but is not
    analysis data. ``current_plot`` is the reader's view of the *last* plot in
    the rawfile, so the last/only plot's name lives on
    ``native.current_plot.plotname`` (NOT on ``native`` itself). Only a
    constants plot that is last/only is a false
    positive — a real analysis followed by constants cannot occur (the current
    plot after a successful analysis is that analysis).
    """
    if callable(getattr(type(native), DATA_PRESENCE_HOOK, None)):
        answer = getattr(native, DATA_PRESENCE_HOOK)()
        if not isinstance(answer, bool):
            raise TypeError(
                f"{type(native).__name__}.{DATA_PRESENCE_HOOK}() must return a "
                f"bool (True = this result carries data rows); it returned "
                f"{type(answer).__name__}."
            )
        return answer
    plot = getattr(native, 'current_plot', None)
    if plot is not None and getattr(plot, 'plotname', '').strip().lower() == 'constants':
        return False
    npoints = getattr(native, 'npoints', None)
    if npoints is not None:
        return npoints > 0
    vectors = getattr(native, 'vectors', None)
    if vectors is not None:
        return any(len(getattr(v, 'data', ())) > 0 for v in vectors)
    raise NotSupportedError(
        f"silk cannot tell whether a {type(native).__name__} result carries "
        f"data rows: it is not the nutmeg shape (no .npoints and no .vectors) "
        f"and it does not implement {DATA_PRESENCE_HOOK}(). A backend whose "
        f"native result is not a nutmeg rawfile must answer the data-presence "
        f"question itself — define {DATA_PRESENCE_HOOK}() on the native "
        f"result's class, or override has_data() on the backend."
    )


def _matching_plots(plots, analysis):
    """Plots matching one kind under the existing analysis classifier."""
    from silk.signals.analysis_kind import plot_silk_analysis
    return [plot for plot in plots if plot_silk_analysis(plot) == analysis]


def select_plot(native, analysis):
    """THE plot-selection rule for a nutmeg result.

    Parameters
    ----------
    native
        The backend's native result — for SPICE backends a
        :class:`~silk.signals.rawfile.RawFile`. Anything without a ``plots``
        sequence (an in-memory backend's own result object) is handed back as
        ``None`` = "no selection", which is what keeps this function safe to
        call on every load path.
    analysis : str or None
        ``'tran'`` / ``'ac'`` / ``'dc'`` / ``'noise'`` / ``'op'``, or ``None``
        for "the file holds one analysis, read it".

    Returns
    -------
    RawPlot or None
        The plot to read, or ``None`` meaning "the reader's own default"
        (``RawFile.current_plot``) — returned for a single-plot file with no
        *analysis*, so that every rawfile silk has ever written is loaded by
        exactly the path it was loaded by before.

    Raises
    ------
    SimulationError
        If *analysis* is not one of the five analysis words; if the file holds
        no plot or multiple plots of the named analysis; or if the file holds
        MORE THAN ONE plot and no *analysis* was named. Selection among
        multiple plots of the same kind is not currently supported by silk
        and is a future extension; save the intended result in a separate rawfile.

    **silk never guesses the plot.** The last-plot fallback is withdrawn for
    multi-plot files: a bare ``write`` used to save whichever analysis ngspice
    ran last, so a metric that read "the" plot was reading an accident of the
    simulator's scheduling — the AC metric errored on a time axis and the
    time-domain metric returned a real number from the wrong analysis
    (measured repro). A single-plot file is unambiguous and is left
    exactly as it was.
    """
    from silk.exceptions import SimulationError
    from silk.signals.analysis_kind import SELECTABLE_ANALYSES

    # A real sequence, not merely something that answers getattr: the same
    # trap DATA_PRESENCE_HOOK documents above. A MagicMock stand-in for a
    # native result is truthy AND len() == 0, so a duck-typed check would read
    # it as an empty multi-plot file and refuse every load through a mock.
    plots = getattr(native, 'plots', None)
    if not isinstance(plots, (list, tuple)) or not plots:
        return None
    if analysis is None:
        if len(plots) == 1:
            return None
        raise SimulationError(
            f"this rawfile holds {len(plots)} analyses, so a signal cannot be "
            f"loaded without naming one: "
            f"{_describe_plots(plots)}. Pass analysis='<kind>' to "
            f"load_signal (or to add_metric) — silk does not guess which "
            f"analysis a metric meant.")
    if analysis not in SELECTABLE_ANALYSES:
        raise SimulationError(
            f"analysis={analysis!r} is not a silk analysis; expected one of "
            f"{list(SELECTABLE_ANALYSES)}.")
    matches = _matching_plots(plots, analysis)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SimulationError(
            f"this rawfile holds {len(matches)} {analysis!r} plots. "
            "Selection among multiple plots of the same kind is not currently "
            "supported by silk and is a future extension. "
            "Save the intended result in a separate rawfile.")
    raise SimulationError(
        f"this rawfile holds no {analysis!r} plot: {_describe_plots(plots)}. "
        f"Either the deck declared no {analysis!r} analysis, or the run "
        f"produced no data for it.")


def _describe_plots(plots) -> str:
    """``'Transient Analysis' (tran), 'AC Analysis' (ac)`` — what a rawfile
    holds, named as the simulator wrote it AND in the word a caller would pass
    as ``analysis=``. An unclassified plot (for example an aborted run's
    ``constants``) shows its plotname with no word, since it cannot be selected."""
    from silk.signals.analysis_kind import plot_silk_analysis

    parts = []
    for plot in plots:
        name = getattr(plot, 'plotname', '') or '(unnamed)'
        kind = plot_silk_analysis(plot)
        parts.append(f"{name!r}" + (f" ({kind})" if kind else ""))
    return ', '.join(parts)


def _require_data(raw, *, simulator, log_path=None, reason=None,
                  supplied=False, has_data=None, diagnostic_text=None):
    """Layer 2 — the agnostic data-presence gate. Raise SimulationAborted if a
    run produced a rawfile with no data rows. A None native is a legitimate
    Xyce noise-only run (CSV, validated separately) and is left alone.

    Called by :meth:`SimulatorBackend.run` (the template method — every backend
    passes through it, none opts in) and by :meth:`SpiceBackend.load_raw`.

    A pre-computed *reason* (e.g. scraped from Xyce's captured stdout) takes
    precedence; otherwise the ngspice *log_path* is scanned for a reason.

    *has_data* is the data-presence predicate to apply, normally the calling
    backend's :meth:`SimulatorBackend.has_data`; it defaults to the shared
    :func:`_has_data_rows` so a direct call needs nothing extra.

    *supplied* is set by the load path (:meth:`SpiceBackend.load_raw`, i.e.
    ``evaluate(rawfile=…)`` / ``Module.read_raw_file()``), where the rawfile was
    handed to silk rather than produced by a run here. It only
    changes the wording: the simulation was still aborted, wherever it ran.

    *diagnostic_text* is the failing run's captured output when the simulator
    has no log file (Xyce stdout); with a *log_path* the log is read instead.
    Either is matched against the convergence patterns, and a hit raises
    :class:`~silk.exceptions.ConvergenceError` (a ``SimulationAborted``)
    rather than the plain abort.
    """
    from silk.exceptions import SimulationAborted
    if raw.native is None:
        return
    if (has_data or _has_data_rows)(raw.native):
        return
    if reason is None and log_path is not None:
        from silk.simulation.runner import abort_reason_from_log
        reason = abort_reason_from_log(log_path)
    message = (
        f"{simulator} produced no data rows (aborted analysis). "
        f"{'Supplied rawfile' if supplied else 'Rawfile'}: {raw.source_path}"
        + (f". Reason: {reason}" if reason else "")
    )
    from silk.simulation.convergence import convergence_error, read_log_text
    conv = convergence_error(
        simulator, diagnostic_text or read_log_text(log_path), message,
        reason=reason, data_rows=0, log_path=log_path)
    if conv is not None:
        raise conv
    raise SimulationAborted(
        message, simulator=simulator, reason=reason, data_rows=0,
        log_path=log_path,
    )


def _normalize_launch(launched, backend_name):
    """Unpack what a backend's ``_launch`` returned into
    ``(raw, log_path, reason)``.

    A backend with neither a log file nor a pre-scraped reason may return
    the bare :class:`RawResult`; one that has either returns the three-tuple.
    Anything else is a backend bug and says so here rather than failing
    obscurely inside the gate.
    """
    if isinstance(launched, RawResult):
        return launched, None, None
    try:
        raw, log_path, reason = launched
    except (TypeError, ValueError):
        raise TypeError(
            f"{backend_name}._launch() must return a RawResult, or a "
            f"(RawResult, log_path, reason) tuple; it returned "
            f"{type(launched).__name__}."
        ) from None
    if not isinstance(raw, RawResult):
        raise TypeError(
            f"{backend_name}._launch() returned a 3-tuple whose first item is "
            f"a {type(raw).__name__}, not a RawResult."
        )
    return raw, log_path, reason


@dataclass
class EmitContext:
    """Per-emission context handed to a backend's renderers.

    All fields are optional; the defaults suit a standalone
    ``analysis.to_spice()`` call that has no circuit to draw from.

    Examples
    --------
    >>> from silk.simulation.backend import EmitContext
    >>> ctx = EmitContext(resolve_source=lambda name: 'V' + name)
    >>> ctx.resolve_source('vin'), ctx.fmt(1e-9)
    ('Vvin', '1n')

    Parameters
    ----------
    resolve_source : callable, optional
        Maps a logical silk source name to its emitted netlist instance name
        (for example ``'vinp'`` to ``'Vvinp'``). The default is the identity.
    fmt : callable, optional
        silk's number formatter, carried verbatim so emission stays
        byte-identical. The default formats a value as an SPICE number with
        an SI suffix.
    sources : dict, optional
        The deck's logical-name to source-object map. Empty by default;
        the Xyce backend consults it for standalone emission of an
        operating-point analysis.
    emitted_deck : str, optional
        The body the netlister has already emitted, used to keep the private
        source and node an operating-point analysis adds on Xyce from
        colliding with names in the deck. Empty by default.
    """
    #: Maps a logical source name to its emitted instance name; identity by
    #: default.
    resolve_source: Callable[[str], str] = _identity
    #: silk's SPICE number formatter, carried verbatim.
    fmt: Callable[[float], str] = field(default=_fmt)
    #: Logical source name to source object map; empty by default.
    sources: dict = field(default_factory=dict)
    #: The netlist body already emitted; empty by default.
    emitted_deck: str = ''


class SimulatorBackend(ABC):
    """Base class and name-keyed registry for simulator backends.

    A backend owns the whole produce-a-signal pipeline for one simulator:
    rendering analyses, building the netlister, running, and loading
    results. Backends are selected by registry name (for example
    ``'ngspice'`` or ``'xyce'``); see :meth:`register` and :meth:`get`.

    :meth:`run` is a **template method**: the base class owns the run
    orchestration and applies silk's data-presence and analysis-completion
    gates to every result, and concrete backends implement ``_launch``
    instead. The gates are therefore part of the framework rather than a
    per-backend courtesy. A subclass that defines ``run`` or ``run_module``
    is rejected with ``TypeError`` when the class is defined.

    A concrete backend implements :meth:`render_analysis`,
    :meth:`make_netlister` and ``_launch``.

    Examples
    --------
    >>> from silk.simulation.backend import SimulatorBackend
    >>> backend = SimulatorBackend.get('ngspice')
    >>> backend.simulator_name
    'ngspice'
    >>> type(backend).__name__
    'NgspiceBackend'
    """

    _registry: dict = {}

    #: Registry name, stamped on the class by :meth:`register`. It is the
    #: ``simulator`` field on a :class:`~silk.exceptions.SimulationAborted`
    #: raised by :meth:`run`. ``None`` on a backend class never registered.
    name = None

    #: ``True`` on a backend that runs a :class:`~silk.circuit.modules.Module`
    #: directly instead of a SPICE deck: ``Module.simulate()``
    #: then calls :meth:`run_module`, and ``generate_netlist()`` /
    #: ``run_simulator()`` refuse, because there is no deck. The dispatch is
    #: on this capability, never on the backend's name.
    runs_module = False

    def __init_subclass__(cls, **kwargs):
        """Refuse a subclass that defines ``run`` or ``run_module``.

        The data-presence gate lives in those two template methods; a backend
        that overrode one would silently regain the trusted-exit-code behavior
        the gate exists to kill, with no failing test to say so (its happy-path
        tests would still pass). Failing here — at ``class`` statement time,
        before anyone can run that backend — is the enforcement.
        """
        super().__init_subclass__(**kwargs)
        if 'run' in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} defines run(), which simulator backends do "
                f"not own: SimulatorBackend.run is a template method that "
                f"applies silk's data-presence gate to every backend's result "
                f"(silk does not trust simulator exit codes). Implement "
                f"_launch(netlist_path, results_dir, *, debug, timeout) "
                f"instead — return the RawResult (or a "
                f"(RawResult, log_path, reason) tuple) and run() gates it."
            )
        if 'run_module' in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} defines run_module(), which simulator "
                f"backends do not own: SimulatorBackend.run_module is a "
                f"template method that applies silk's data-presence gate to "
                f"every module-level run's result. Implement "
                f"_launch_module(module, *, run_files) instead — return the "
                f"RawResult (or a (RawResult, log_path, reason) tuple) and "
                f"run_module() gates it."
            )

    @classmethod
    def register(cls, name: str):
        """Class decorator registering a backend subclass under *name*.

        The name is stamped on the class (``name``) so :meth:`run` can
        attribute an abort to the simulator the user selected. Registering a
        name that is already taken replaces the earlier registration.

        Parameters
        ----------
        name : str
            Registry name used to select the backend, for example
            ``'ngspice'``.

        Returns
        -------
        callable
            A decorator that registers the class it is applied to and
            returns that class.
        """
        def decorator(subclass):
            cls._registry[name] = subclass
            subclass.name = name
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> "SimulatorBackend":
        """Return a fresh instance of the backend registered under *name*.

        Parameters
        ----------
        name : str
            A registered backend name.

        Returns
        -------
        SimulatorBackend
            A new instance of the registered backend class.

        Raises
        ------
        ValueError
            If no backend is registered under *name*; the message lists the
            registered names.
        """
        backend_cls = cls._registry.get(name)
        if backend_cls is None:
            raise ValueError(
                f"No simulator backend registered for {name!r}. "
                f"Registered: {sorted(cls._registry)}"
            )
        return backend_cls()

    @abstractmethod
    def render_analysis(self, analysis, ctx: EmitContext) -> str:
        """Render one analysis object to this simulator's command line(s).

        Parameters
        ----------
        analysis : AnalysisBase
            The analysis request (:class:`~silk.simulation.Transient`,
            :class:`~silk.simulation.AC`, and so on).
        ctx : EmitContext
            Source-name resolution and number formatting for this emission.

        Returns
        -------
        str
            The simulator's command text, newline-terminated.

        Raises
        ------
        NotImplementedError
            In the abstract base; a concrete backend overrides it.
        """
        raise NotImplementedError

    @property
    def simulator_name(self) -> str:
        """str: The simulator label used in abort messages, normally the registry name.

        Falls back to the class name for a backend that was never registered.
        """
        return self.name or type(self).__name__

    def has_data(self, native) -> bool:
        """Report whether this backend's *native* result carries at least one data row.

        The seam through which a backend owns the data-presence verdict. The
        default answers for the nutmeg shape both SPICE backends share, and
        for any native whose class defines a ``silk_has_data()`` method; a
        backend whose native is neither overrides this. The native result
        stays backend-private either way: nothing in the shared layer reaches
        into it beyond this predicate.

        Parameters
        ----------
        native : object
            The ``native`` payload of a :class:`RawResult`.

        Returns
        -------
        bool
            True if the result holds at least one data row. An aborted
            ngspice run whose only plot is the synthetic ``constants`` plot
            counts as no data.

        Raises
        ------
        NotSupportedError
            If *native* is not the nutmeg shape and does not define
            ``silk_has_data()``.
        TypeError
            If ``silk_has_data()`` returns something other than a ``bool``.
        """
        return _has_data_rows(native)

    def sample_args(self, sample) -> tuple:
        """Return the command-line arguments that carry a Monte Carlo sample's seed.

        Parameters
        ----------
        sample : object
            The Monte Carlo sample; backends that pass a seed read its
            ``seed`` attribute.

        Returns
        -------
        tuple of str
            Extra arguments for the simulator command line. The base class
            (and ngspice, which takes the seed from the deck) returns ``()``;
            Xyce returns ``('-randseed', '<seed>')``.
        """
        return ()

    def run(self, netlist_path: str, results_dir: str, *,
            debug: bool = False, timeout=None, run_files=None,
            extra_args: tuple = ()) -> RawResult:
        """Run this backend's simulator on *netlist_path* and load the result.

        **Template method: do not override** (a subclass that defines ``run``
        is rejected when the class is defined). It calls this backend's
        ``_launch``, then applies the data-presence gate to what came back, so
        every backend fails loudly on a run that completed but produced
        nothing, and then the analysis-completion check. silk does not trust
        the simulator's exit code. Called by ``Module.run_simulator`` and the
        fault-campaign runner.

        Parameters
        ----------
        netlist_path : str
            Path of the SPICE deck to simulate.
        results_dir : str
            Directory where the simulator's output files are written.
        debug : bool, optional
            Let simulator output flow to the terminal and print progress,
            where the backend supports it. Defaults to false.
        timeout : float, optional
            Seconds to wait for the simulator. ``None`` (the default) waits
            indefinitely.
        run_files : RunFiles, optional
            An already-prepared deck's file locations (internal). Without
            it, the explicit deck and output paths use the original launch
            interface. Consuming a prepared deck never allocates a new
            location.
        extra_args : tuple of str, optional
            Extra simulator command-line arguments, for example those from
            :meth:`sample_args`. Only forwarded, and only used by backends
            that accept them (Xyce), when *run_files* is given.

        Returns
        -------
        RawResult
            The native output and the path it was read from, with
            ``ambient_temperatures`` and ``simulator_version`` filled in from
            this run when the backend can report them.

        Raises
        ------
        SimulationAborted
            The run produced no data rows, or a certified silk-generated
            SPICE analysis stopped short of its final grid point or count.
            A failure whose log names a convergence problem raises the
            ``ConvergenceError`` subclass.
        SimulationError
            Raised by the backend's launch when the simulator cannot be
            found or started, times out, or exits with an error.
        TypeError
            If a backend's ``_launch`` returns something other than a
            :class:`RawResult` or a ``(RawResult, log_path, reason)`` tuple.
        """
        generation = None
        explicit_paths = run_files is None
        if (type(self) in (NgspiceBackend, XyceBackend)
                and getattr(self._launch, '__func__', None) is _BUILTIN_LAUNCH[type(self)][0]
                and getattr(self._launch_prepared, '__func__', None) is _BUILTIN_LAUNCH[type(self)][1]):
            from silk.simulation.generation import begin
            files = run_files or self._resolve_run_files(netlist_path, results_dir)
            generation = begin(files)
            run_files = generation.staged
        from silk.simulation.generation import _active
        generation_token = _active.set(generation)
        try:
            if generation is None:
                from silk.simulation.generation import _replacing
                if _replacing.get() is not None:
                    _replacing.get()()
            if run_files is None:
                # Preserve the explicit-path launch seam for existing backends.
                launched = self._launch(netlist_path, results_dir, debug=debug,
                                        timeout=timeout)
            else:
                # Only a backend whose sample_args() is non-empty gets the keyword.
                extra = {'extra_args': extra_args} if extra_args else {}
                if generation is not None and explicit_paths and type(self) is XyceBackend:
                    extra['require_all_outputs'] = False
                launched = self._launch_prepared(run_files, debug=debug,
                                                 timeout=timeout, **extra)
            raw, log_path, reason = _normalize_launch(launched, type(self).__name__)
            if raw.run_files is None:
                raw.run_files = run_files
            _require_data(raw, simulator=self.simulator_name, log_path=log_path,
                          reason=reason, has_data=self.has_data)
            from silk.simulation.completion import require_complete
            deck_path = (raw.run_files.netlist_path
                         if raw.run_files and raw.run_files.netlist_path else netlist_path)
            require_complete(raw, deck_path, simulator=self.simulator_name,
                             log_path=log_path, reason=reason)
            raw.ambient_temperatures = self.read_ambient_temperatures(raw, log_path)
            raw.simulator_version = self.read_simulator_version(raw, log_path)
            if generation is not None:
                from silk.simulation.generation import offer, attach, accept
                raw = attach(raw, offer(generation))
                accept(raw)
            return raw
        except BaseException as exc:
            from silk.simulation.generation import abandon, publish_failure_log
            publish_failure_log(generation, exc)
            abandon(generation)
            raise
        finally:
            _active.reset(generation_token)

    def run_module(self, module, *, run_files=None) -> RawResult:
        """Run *module* itself, with no SPICE deck, and load the result.

        The module-level twin of :meth:`run`, used by a backend whose
        ``runs_module`` is true. **Template method: do not override** (a
        subclass that defines ``run_module`` is rejected when the class is
        defined). It calls this backend's ``_launch_module``, then applies the
        same data-presence gate as :meth:`run`, and fills in
        ``ambient_temperatures`` and ``simulator_version``. Called by
        ``Module.simulate()``.

        Parameters
        ----------
        module : Module
            The circuit module to run directly.
        run_files : RunFiles, optional
            File locations for the run, retained on the result.

        Returns
        -------
        RawResult
            The native output of the module-level run.

        Raises
        ------
        SimulationAborted
            The run produced no data rows.
        NotSupportedError
            If this backend runs SPICE decks and does not implement
            module-level runs.
        """
        launched = self._launch_module(module, run_files=run_files)
        raw, log_path, reason = _normalize_launch(launched, type(self).__name__)
        if raw.run_files is None:
            raw.run_files = run_files
        _require_data(raw, simulator=self.simulator_name, log_path=log_path,
                      reason=reason, has_data=self.has_data)
        raw.ambient_temperatures = self.read_ambient_temperatures(raw, log_path)
        raw.simulator_version = self.read_simulator_version(raw, log_path)
        return raw

    def _launch_module(self, module, *, run_files=None):
        """Run *module* directly — the per-backend half of :meth:`run_module`.

        Only a backend with :attr:`runs_module` implements it; the default
        refuses, because a deck-based backend has nothing to run without a
        netlist. Return what :meth:`_launch` returns.
        """
        raise NotSupportedError(
            f"the {self.simulator_name!r} backend runs SPICE decks, not "
            f"modules: generate a netlist and call run() instead.")

    def read_ambient_temperatures(self, raw, log_path):
        """Return the ambient temperatures this launch reports having solved at.

        Independent runtime evidence, not an echo of the request: a backend
        that cannot produce any returns ``()``, and the operating-conditions
        record then marks the temperature as unknown rather than assuming the
        configured value was applied.

        Parameters
        ----------
        raw : RawResult
            The result of the launch.
        log_path : str or None
            The simulator's log file, when the backend wrote one.

        Returns
        -------
        tuple of float
            Temperatures in degrees Celsius; empty when there is no evidence.
            The base implementation returns ``()``.
        """
        return ()

    def read_simulator_version(self, raw, log_path):
        """Return this simulator binary's own version, as reported by this run.

        Read from the run's own log or transcript rather than by invoking the
        binary separately, so it is the version that produced the result
        being read. See :attr:`RawResult.simulator_version`. The base
        implementation reports it as unknown; the ngspice backend reads the
        log file and the Xyce backend reads the captured standard output.

        Parameters
        ----------
        raw : RawResult
            The result of the launch.
        log_path : str or None
            The simulator's log file, when the backend wrote one.

        Returns
        -------
        str or None
            The version string, or ``None`` when it cannot be read (the base
            implementation always returns ``None``).
        """
        return None

    def _resolve_run_files(self, netlist_path, results_dir, *, analyses=None,
                           temperature=False):
        """Pure layout resolution; backends declare their output companions here.

        *temperature* asks for a backend-written runtime temperature record;
        a backend without one ignores it (ngspice scrapes its own log).
        """
        return RunFiles.resolve(netlist_path, results_dir)

    def _launch_prepared(self, run_files, *, debug=False, timeout=None):
        """Adapt prepared decks to backends implementing the original launch seam."""
        return self._launch(run_files.netlist_path, run_files.directory,
                            debug=debug, timeout=timeout)

    @abstractmethod
    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        """Run the simulator and load its result — the per-backend half of
        :meth:`run`.

        Return either a bare :class:`RawResult`, or a
        ``(RawResult, log_path, reason)`` tuple when this backend has a log file
        to scrape a reason from (ngspice) or has already scraped one (Xyce).
        The exit-code / process check (Layer 1) lives here, because it is
        per-simulator; the data-presence check (Layer 2) does not — :meth:`run`
        applies it to every backend.
        """
        raise NotImplementedError

    @abstractmethod
    def make_netlister(self, pdk_details):
        """Return a netlister for this backend's simulator.

        The netlister turns the circuit into this simulator's netlist text
        (header, includes, devices and sources, analyses, control block).
        Called when a ``Module`` builds its netlister.

        Parameters
        ----------
        pdk_details : object
            The PDK device details the module was built with, handed to the
            netlister constructor.

        Returns
        -------
        object
            The simulator-specific netlister.

        Raises
        ------
        NotImplementedError
            In the abstract base; a concrete backend overrides it.
        """
        raise NotImplementedError


class SpiceBackend(SimulatorBackend):
    """Shared SPICE/nutmeg analysis rendering and result loading for SPICE-family simulators.

    Provides the ``.tran``/``.ac``/``.dc``/``.noise``/``.op`` renderer, the
    nutmeg-rawfile loaders, and the device-probe seams that the ngspice and
    Xyce backends build on. The device-probe seams
    (:meth:`device_current_vectors`, :meth:`terminal_current_vector`,
    :meth:`device_op_vectors`, :meth:`load_device_probe`,
    :meth:`preflight_unavailable_error`) raise
    :class:`~silk.exceptions.NotSupportedError` here and are overridden by
    the backends that support them.

    Examples
    --------
    >>> from silk.simulation import DC
    >>> from silk.simulation.backend import EmitContext, SimulatorBackend
    >>> ctx = EmitContext(resolve_source=lambda name: 'V' + name)
    >>> SimulatorBackend.get('ngspice').render_analysis(DC('vin', 0, 1, 0.5), ctx)
    '.dc Vvin 0 1 0.5\\n'
    """

    #: The fields silk itself writes into each built-in analysis card.
    _CARD_FIELDS = (
        (Transient, ('step', 'stop', 'start', 'maxstep')),
        (AC, ('variation', 'points', 'fstart', 'fstop')),
        (DC, ('source', 'start', 'stop', 'step')),
        (Noise, ('output', 'ref', 'source', 'variation', 'points',
                 'fstart', 'fstop')),
    )

    def render_analysis(self, analysis, ctx: EmitContext) -> str:
        """Render an analysis object as SPICE command text.

        Parameters
        ----------
        analysis : AnalysisBase
            A :class:`~silk.simulation.Transient`, :class:`~silk.simulation.AC`,
            :class:`~silk.simulation.DC`, :class:`~silk.simulation.Noise` or
            :class:`~silk.simulation.Op`.
        ctx : EmitContext
            Source-name resolution and number formatting.

        Returns
        -------
        str
            ``.tran``, ``.ac``, ``.dc``, ``.noise`` or ``.op`` text,
            newline-terminated. A ``DC`` sweep of ``temp`` is emitted
            unchanged; other sweep sources, and the noise input source, pass
            through ``ctx.resolve_source``.

        Raises
        ------
        TypeError
            If *analysis* is not one of the five analysis types.
        silk.exceptions.DeckTextError
            If a field of a ``Transient``, ``AC``, ``DC`` or ``Noise``
            analysis, or the rendered card, contains a line break. Numeric
            fields are also checked when the analysis is built; ``DC.source``
            and other text fields are checked when the card is rendered, and
            the final card text (excluding its terminating newline) is
            checked before it is returned.
        """
        # Fields are checked here as well as at construction: an attribute
        # set later never passed the constructor (TASK-593).
        from silk.utils.deck_text import refuse_line_breaks
        for kind, fields in self._CARD_FIELDS:
            if isinstance(analysis, kind):
                for field in fields:
                    refuse_line_breaks(
                        getattr(analysis, field, None),
                        who=f'{type(analysis).__name__}.{field}')
        text = self._render_card(analysis, ctx)
        # Each built-in card is one line: check the text actually written,
        # after formatting, so a non-string field whose text holds a break
        # is refused as well.
        from silk.utils.deck_text import card_text
        card_text(text[:-1] if text.endswith('\n') else text,
                  who=f'{type(analysis).__name__} analysis card')
        return text

    def _render_card(self, analysis, ctx: EmitContext) -> str:
        fmt = ctx.fmt
        if isinstance(analysis, Transient):
            parts = ['.tran', fmt(analysis.step), fmt(analysis.stop)]
            if analysis.start != 0 or analysis.maxstep is not None:
                parts.append(fmt(analysis.start))
            if analysis.maxstep is not None:
                parts.append(fmt(analysis.maxstep))
            if analysis.uic:
                parts.append('uic')
            return ' '.join(parts) + '\n'
        if isinstance(analysis, AC):
            return (f'.ac {analysis.variation} {analysis.points} '
                    f'{fmt(analysis.fstart)} {fmt(analysis.fstop)}\n')
        if isinstance(analysis, DC):
            src = analysis.source if str(analysis.source).lower() == 'temp' \
                else ctx.resolve_source(analysis.source)
            return (f'.dc {src} {fmt(analysis.start)} '
                    f'{fmt(analysis.stop)} {fmt(analysis.step)}\n')
        if isinstance(analysis, Noise):
            src = ctx.resolve_source(analysis.source)
            return (f'.noise v({analysis.output},{analysis.ref}) {src} '
                    f'{analysis.variation} {analysis.points} '
                    f'{fmt(analysis.fstart)} {fmt(analysis.fstop)} 1\n')
        if isinstance(analysis, Op):
            return '.op\n'
        raise TypeError(
            f"{type(self).__name__} cannot render analysis "
            f"{type(analysis).__name__}"
        )

    def load_raw(self, source_path, *, simulator=None, log_path=None,
                 reason=None) -> RawResult:
        """Load an existing rawfile by path, without running a simulator.

        SPICE-family rawfiles (ngspice's ``write`` output and Xyce's
        ``format=raw`` output alike) are read lazily by silk's own nutmeg
        reader. Used when results are loaded without a fresh run in this
        session.

        This is a load, not a run, so it is outside :meth:`SimulatorBackend.run`,
        but it applies the same data-presence gate: a rawfile that carries no
        analysis data (for example an aborted run's synthetic ``constants``
        plot) raises instead of being measured as real data downstream. This
        is the one gate on the load path; ``ModuleTestbench.evaluate(rawfile=...)``,
        ``Module.read_raw_file()`` and every other offline load reach it.

        Parameters
        ----------
        source_path : str
            Path of the rawfile.
        simulator : str, optional
            The simulator that produced the file, used to attribute an
            abort. With no hint the abort is attributed to ``'rawfile'``.
        log_path : str, optional
            The producing run's log file, used to enrich an abort message.
        reason : str, optional
            A pre-scraped abort reason; takes precedence over the log.

        Returns
        -------
        RawResult
            The lazily-read rawfile and ``source_path``.

        Raises
        ------
        SimulationAborted
            If the rawfile carries no analysis data rows (the
            ``ConvergenceError`` subclass when the log names a convergence
            failure).
        """
        from silk.signals.rawfile import read_rawfile
        raw = RawResult(native=read_rawfile(source_path, lazy=True), source_path=source_path)
        _require_data(raw, simulator=simulator or 'rawfile', log_path=log_path,
                      reason=reason, supplied=True, has_data=self.has_data)
        return raw

    def load_signal(self, raw: RawResult, name: str, *, type: str = 'V',
                    analysis=None):
        """Extract one signal from a run result.

        A request for a canonical noise signal (``noise_out``/``noise_in``,
        or the ngspice-native and terse aliases such as ``onoise_spectrum``)
        is routed to :meth:`load_noise`; everything else is a node signal read
        from the rawfile, with case-insensitive node matching so Xyce's
        upper-cased names resolve.

        Parameters
        ----------
        raw : RawResult
            The run result to read from.
        name : str
            Signal (vector) name.
        type : str, optional
            Vector kind, ``'V'`` (voltage, the default) or ``'I'`` (current).
        analysis : {'tran', 'ac', 'dc', 'noise', 'op'}, optional
            Which plot of a multi-analysis rawfile to read. A single-plot
            file needs none.

        Returns
        -------
        silk.signals.Signal
            The loaded signal.

        Raises
        ------
        SimulationError
            If *analysis* is not one of the selectable analysis words, if the file
            holds no plot of that analysis, or if the file holds more than
            one plot and no *analysis* was named, or if multiple plots match
            *analysis*. Selection among multiple plots of the same kind is
            not currently supported by silk and is a future extension; save
            the intended result in a separate rawfile. silk never guesses the plot.
        """
        port = None if getattr(name, '_literal_voltage', False) else _NOISE_PORT.get(name.lower())
        if port is not None:
            return self.load_noise(raw, port, analysis=analysis)
        from silk.signals.spice import NgspiceSignal
        return NgspiceSignal.from_rawfile(raw.native, name, type,
                                          plot=select_plot(raw.native, analysis))

    def load_noise(self, raw: RawResult, port: str, *, analysis=None):
        """Load a canonical noise spectral-density signal in V/sqrt(Hz).

        The default implementation (ngspice) reads the noise plot from the
        rawfile, where ``onoise_spectrum`` and ``inoise_spectrum`` are already
        amplitude spectral densities; only the name is canonicalized. Multiple
        noise plots, including spectral and integrated noise together, are
        ambiguous under the same selection rule as :meth:`load_signal`. The Xyce
        backend overrides this because its noise output is a separate CSV.

        Parameters
        ----------
        raw : RawResult
            The run result to read from.
        port : {'out', 'in'}
            Output-referred or input-referred noise.
        analysis : {'tran', 'ac', 'dc', 'noise', 'op'}, optional
            Selects the plot in a multi-analysis rawfile, exactly as in
            :meth:`load_signal`. A noise signal in a file that also holds a
            transient or AC plot needs ``analysis='noise'``.

        Returns
        -------
        silk.signals.Signal
            A signal named ``noise_out`` or ``noise_in``.

        Raises
        ------
        KeyError
            If *port* is neither ``'out'`` nor ``'in'``.
        SimulationError
            Under the plot-selection conditions of :meth:`load_signal`.
        """
        from silk.signals.spice import NgspiceSignal
        native_name = {'out': 'onoise_spectrum', 'in': 'inoise_spectrum'}[port]
        sig = NgspiceSignal.from_rawfile(raw.native, native_name, 'V',
                                         plot=select_plot(raw.native, analysis))
        sig.name = f'noise_{port}'
        sig.ylabel = f'noise_{port}'
        return sig

    def device_current_vectors(self, raw, instance_path, leaf_name) -> dict:
        """Map each terminal-current suffix to the rawfile vector name for one element instance.

        Suffixes are ``'id'``, ``'ig'``, ``'is'``, ``'ib'`` (MOS) and ``'i'``,
        ``'current'``, ``'branch'`` (two-terminal and branch). The seam is
        kind-agnostic: mapping an element kind to a port and sign belongs to
        the current-annotation code.

        Parameters
        ----------
        raw : RawResult
            The run result whose vectors are searched.
        instance_path : str
            Emitted container path of the element.
        leaf_name : str
            Element's leaf instance name.

        Returns
        -------
        dict
            Suffix to vector name. A suffix with no matching vector is
            absent; a value of ``None`` marks an ambiguous suffix (more than
            one vector matched).

        Raises
        ------
        NotSupportedError
            In the base implementation (inherited by Xyce); the ngspice
            backend overrides it.
        """
        raise NotSupportedError(
            f"{type(self).__name__} does not expose device currents; current "
            f"annotation is unsupported on this backend.")

    def terminal_current_vector(self, raw, instance_path, leaf_name, port):
        """Return the rawfile vector name for one device terminal's current in ammeter-wrapper mode.

        The vector is the branch current of the 0 V ammeter inside the
        device's wrapper subcircuit, ``i(v.<path>.xm<dev>.vamm_<port>)``. The
        sign is uniform (current into the net is the negated value) and the
        form works for every terminal and device model.

        Parameters
        ----------
        raw : RawResult
            The run result whose vectors are searched.
        instance_path : str
            Emitted container path of the device.
        leaf_name : str
            Device's leaf instance name.
        port : str
            Terminal name.

        Returns
        -------
        str or None
            The vector name, or ``None`` if the rawfile has no such vector.

        Raises
        ------
        NotSupportedError
            In the base implementation (inherited by Xyce); the ngspice
            backend overrides it.
        """
        raise NotSupportedError(
            f"{type(self).__name__} does not support ammeter terminal currents.")

    def device_op_vectors(self, raw, instance_path, leaf_name, polarity=None) -> dict:
        """Map each operating-point parameter to its value for one emitted MOS device.

        Values are read from the rawfile's dedicated operating-point plot
        only (``Module.save_device_params``). Vectors ngspice also writes
        into a co-declared AC, DC or transient plot are never consulted.

        Parameters
        ----------
        raw : RawResult
            The run result to read.
        instance_path : str
            Emitted container path of the device.
        leaf_name : str
            Device's leaf instance name.
        polarity : {'n', 'p'}, optional
            The device's declared channel polarity, for a backend whose raw
            ``id`` reading is not already sign-normalised to the positive
            magnitude convention. Ignored by ngspice, whose readings are
            already normalised.

        Returns
        -------
        dict
            ``{}`` when the rawfile holds no operating-point plot. Otherwise
            every name in :data:`DEVICE_OP_PARAMS` is a key, mapped to a
            ``float``; a parameter whose vector is missing, ambiguous or
            zero-length is ``nan``, never a missing key and never a silent 0.

        Raises
        ------
        NotSupportedError
            In the base implementation; the ngspice and Xyce backends
            override it.
        SimulationError
            If the ngspice rawfile holds multiple operating-point plots.
            Save the intended result in a separate rawfile.

        Warns
        -----
        silk.metrics.diagnostics.MetricWarning
            ngspice only: when the device matches no operating-point vector
            or more than one for some parameter, a diagnostic is emitted
            under the ``op_table`` metric name.
        """
        raise NotSupportedError(
            f"{type(self).__name__} does not expose device operating-point "
            f"parameters; save_device_params is unsupported on this backend.")

    def load_device_probe(self, raw, address, param, *, analysis=None):
        """Read one device-parameter probe (``tb.probe('<name>:<param>')``) back as a signal.

        Reads over the full selected plot, every point of *analysis*, unlike
        :meth:`device_op_vectors`, which reads a single operating-point
        value. It never returns 0 or a silent nan: a vector that came back
        missing, ambiguous or zero-length raises, with a message naming the
        device, the parameter, the analysis, this simulator's own version
        (read from this run's log or transcript) and what the run did provide
        for the same device in the same plot.

        Parameters
        ----------
        raw : RawResult
            The run result to read.
        address : object
            The resolved device-probe address (designer name, emitted
            instance path and leaf, polarity and model).
        param : str
            Device parameter name; must be in :data:`DEVICE_OP_PARAMS`.
        analysis : {'tran', 'ac', 'dc', 'noise', 'op'}, optional
            Which plot to read, as for :meth:`load_signal`. Ignored by the
            Xyce backend, whose rawfile holds a single analysis.

        Returns
        -------
        silk.signals.Signal
            The probe signal named ``<designer_name>:<param>``. On ngspice an
            AC probe is returned real-valued.

        Raises
        ------
        NotSupportedError
            In the base implementation; the ngspice and Xyce backends
            override it.
        ValueError
            ngspice: if *param* is not in :data:`DEVICE_OP_PARAMS`.
        SimulationError
            If the plot cannot be selected, or the probe's vector is
            missing, ambiguous, zero-length or marked unavailable by the
            simulator's own log.
        """
        raise NotSupportedError(
            f"{type(self).__name__} does not support device-parameter "
            f"probes.")

    def preflight_unavailable_error(self, raw, address, param, analysis, result):
        """Build the error for a probe parameter the ngspice preflight found unavailable.

        The preflight asks the resolved ngspice binary, with a minimal
        single-device deck, whether it provides a device parameter; a
        parameter it does not provide never has a save line emitted. This
        method builds the read-time error for such a parameter. It is not
        reached for a parameter the preflight found available (that reads
        through :meth:`load_device_probe`) nor on any backend other than
        ngspice.

        Parameters
        ----------
        raw : RawResult
            The main run's result, scanned for what it did provide.
        address : object
            The resolved device-probe address.
        param : str
            The unavailable device parameter.
        analysis : str or None
            The analysis the probe was declared on, if named.
        result : silk.circuit.modules_results.PreflightResult
            The preflight's answer, carrying the binary's version and the
            preflight log path.

        Returns
        -------
        SimulationError
            The exception to raise (it is returned, not raised, by the
            ngspice implementation). Its message names the device, the
            parameter, the analysis, the binary version, the preflight log
            and the parameters the main run did provide for the device.

        Raises
        ------
        NotSupportedError
            In the base implementation; only the ngspice backend overrides
            it.
        """
        raise NotSupportedError(
            f"{type(self).__name__} does not support the ngspice preflight "
            f"mechanism.")

    def op_param_capability(self, declared) -> tuple:
        """Return this backend's readable subset of the declared operating-point parameters.

        The seam between "the model declares this parameter" and "this
        simulator can actually read it back". The default is the identity:
        every SPICE-family backend except Xyce reads whatever the model
        declares. The Xyce backend intersects with the parameters it can
        expose, so a parameter it cannot read (``gmbs`` and the
        capacitances) is not mislabeled as simulated.

        Parameters
        ----------
        declared : iterable of str
            A subsequence of :data:`DEVICE_OP_PARAMS`, typically a device's
            own declared operating-point parameters.

        Returns
        -------
        tuple of str
            The readable parameters, in the order of *declared*.
        """
        return tuple(declared)


@SimulatorBackend.register('ngspice')
class NgspiceBackend(SpiceBackend):
    """ngspice backend."""

    def _resolve_run_files(self, netlist_path, results_dir, *, analyses=None,
                           temperature=False):
        # ngspice prints the ambient TEMP it solved at into its own log on
        # every run, so it needs no separate temperature record.
        return RunFiles.resolve(netlist_path, results_dir, log=True)

    def read_ambient_temperatures(self, raw, log_path):
        from silk.simulation.conditions import read_ngspice_temperatures
        return read_ngspice_temperatures(log_path)

    def read_simulator_version(self, raw, log_path):
        from silk.simulation.conditions import read_ngspice_version
        return read_ngspice_version(log_path)

    def device_current_vectors(self, raw, instance_path, leaf_name) -> dict:
        names = [raw.native.vectors[i].name for i in range(raw.native.nvars)]
        leaf = leaf_name.lower()
        path = instance_path.lower().strip('.')
        token = (path.split('.') if path else []) + [leaf]
        out = {}
        for suffix in _CURRENT_SUFFIXES:
            tail = f'[{suffix}]'
            matches = []
            for name in names:
                inner = name.lower()
                if inner.startswith('i(') and inner.endswith(')'):
                    inner = inner[2:-1]
                if not inner.startswith('@') or not inner.endswith(tail):
                    continue
                body = inner[1:-len(tail)]          # strip '@' and '[suffix]'
                if _contiguous_sublist(token, body.split('.')):
                    matches.append(name)
            if len(matches) == 1:
                out[suffix] = matches[0]
            elif len(matches) > 1:
                out[suffix] = None                  # ambiguous (§6.2)
        # Plain branch form (no '@', no '[suffix]') — voltage sources emit
        # `i(v<name>)` verbatim (verified P2a, ngspice-42). Keyed under 'branch'.
        target = '.'.join(token)
        branch_matches = []
        for name in names:
            inner = name.lower()
            if not (inner.startswith('i(') and inner.endswith(')')):
                continue
            inner = inner[2:-1]
            if inner.startswith('@') or '[' in inner:
                continue
            if inner == target:
                branch_matches.append(name)
        if len(branch_matches) == 1:
            out.setdefault('branch', branch_matches[0])
        elif len(branch_matches) > 1:
            out['branch'] = None
        return out

    def terminal_current_vector(self, raw, instance_path, leaf_name, port):
        path = instance_path.lower().strip('.')
        leaf = leaf_name.lower()
        inner = 'v.' + (path + '.' if path else '') + leaf + '.vamm_' + port.lower()
        target = f'i({inner})'
        names = {raw.native.vectors[i].name.lower() for i in range(raw.native.nvars)}
        return target if target in names else None

    def device_op_vectors(self, raw, instance_path, leaf_name, polarity=None) -> dict:
        # polarity is accepted for interface parity with the base signature
        # but unused: ngspice's own '@' vectors already read PMOS id/vgs/vds
        # as positive source-referenced magnitudes (TASK-330.1's spike), so
        # there is nothing here to correct.
        plots = getattr(raw.native, 'plots', None)
        if isinstance(plots, (list, tuple)) and not _matching_plots(plots, 'op'):
            return {}                                # switch was off / no op data
        plot = select_plot(raw.native, 'op')
        leaf = leaf_name.lower()
        path = instance_path.lower().strip('.')
        token = (path.split('.') if path else []) + [leaf]
        out = {}
        match_counts = {}
        candidates = set()
        for param in DEVICE_OP_PARAMS:
            tail = f'[{param}]'
            matches = []
            for vec in plot.vectors:
                inner = vec.name.lower()
                # Wrapper spelling is param-type-dependent (TASK-330.3 probe):
                # 'id' comes back i(@...), five voltage-type params
                # (vgs/vds/vbs/vdsat/vth) come back v(@...), the rest (gm/gds/
                # gmbs, the capacitances) stay bare '@...'. Strip either wrapper
                # before matching; a param never wears both.
                if (inner.startswith('i(') or inner.startswith('v(')) and inner.endswith(')'):
                    inner = inner[2:-1]
                if not inner.startswith('@') or not inner.endswith(tail):
                    continue
                body = inner[1:-len(tail)]           # strip '@' and '[param]'
                if body.startswith('m.') and body.split('.')[1:1 + len(token)] == token:
                    matches.append(vec)
            match_counts[param] = len(matches)
            if len(matches) > 1:
                candidates.update(v.name for v in matches)
            if len(matches) == 1 and len(matches[0].data):
                out[param] = float(matches[0].data[0].real)
            else:
                # 0 matches (absent), >1 (ambiguous), or a zero-length vector
                # (measured for sky130 BSIM4's cgdo/cgso, TASK-330.1) — all
                # "unreadable", never a missing key and never a silent 0.
                out[param] = float('nan')
        # TASK-439: a declared device the matcher cannot find at all (every
        # parameter has 0 matches) or finds ambiguously (any parameter has
        # >1) yields nan rows; say so instead of failing silently. A single
        # absent parameter (cgdo/cgso) is normal and stays quiet.
        ambiguous = sorted(p for p, n in match_counts.items() if n > 1)
        if ambiguous or not any(match_counts.values()):
            from silk.metrics import diagnostics
            device = '.'.join(token)
            if ambiguous:
                message = (f'device {device!r} matches more than one operating-point '
                           f'vector for {", ".join(ambiguous)}; those readings are nan')
                code = 'op_device_ambiguous'
            else:
                message = (f'device {device!r} matches no operating-point vector; '
                           f'its row is all nan')
                code = 'op_device_not_found'
            diagnostics.emit(
                metric='op_table', code=code, message=message,
                details={'device': device, 'candidates': sorted(candidates)},
                suggestion='Check for duplicate vectors at this rooted device path.')
        return out

    def load_device_probe(self, raw, address, param, *, analysis=None):
        """:meth:`SimulatorBackend.load_device_probe` on
        ngspice. Matching is the SAME wrapper-spelling rule
        :meth:`device_op_vectors` uses (id-like -> ``i(@...)``, voltage-
        like -> ``v(@...)``, the rest bare ``@...``), scanned over *analysis*'s
        own plot in full — not read-only-point[0], and not restricted to
        ``'op'`` — so a transient/DC-sweep/AC probe reads every point.

        Checks *param* against :data:`DEVICE_OP_PARAMS` BEFORE touching the
        rawfile at all — defense in depth: the PRIMARY check is now in
        ``ModuleTestbench.probe()`` itself, before the run. See
        :func:`~silk.circuit.modules_results.unrecognized_device_param_error`
        for the refusal reason. Kept here too for any other construction
        path that might reach this method directly. Discovered empirically
        while building this: an unrecognised ``'@'`` name saved ALONGSIDE
        at least one genuinely readable vector can come back as a real,
        populated, all-zero reading with no ``checkvalid`` warning at all
        on ngspice-47 — indistinguishable from a real 0 by inspecting the
        rawfile alone; probed ALONE it aborts the whole write instead, so
        the two failure modes are not even consistent with each other.
        """
        import numpy as np
        from silk.signals.spice import NgspiceSignal
        from silk.circuit.modules_results import unrecognized_device_param_error
        if param not in DEVICE_OP_PARAMS:
            raise unrecognized_device_param_error(address.designer_name, param)
        # select_plot returns None for "the file holds one plot, read it" —
        # the reader's own default (RawFile.current_plot), same fallback
        # NgspiceSignal.from_rawfile applies internally.
        plot = select_plot(raw.native, analysis) or raw.native
        leaf = address.leaf.lower()
        path = address.instance_path.lower().strip('.')
        token = (path.split('.') if path else []) + [leaf]
        tail = f'[{param}]'
        matches = []
        for vec in plot.vectors:
            inner = vec.name.lower()
            if (inner.startswith('i(') or inner.startswith('v(')) and inner.endswith(')'):
                inner = inner[2:-1]
            if not inner.startswith('@') or not inner.endswith(tail):
                continue
            body = inner[1:-len(tail)]
            if _contiguous_sublist(token, body.split('.')):
                matches.append(vec)
        checkvalid_warned = (len(matches) == 1
                            and _checkvalid_warned(raw, matches[0].name))
        if len(matches) == 1 and len(matches[0].data) and not checkvalid_warned:
            sig = NgspiceSignal.from_rawfile(plot, matches[0].name, 'V', plot=None)
            # ngspice writes '@' vectors in an AC plot as complex with a 0
            # imaginary part (measured, TASK-330.8's spike + reviewer's
            # MISSING 3) — an operating-point parameter is not meaningfully
            # complex, so this always returns real values, same xlabel as
            # any other probe on this analysis.
            sig.y = np.asarray(sig.y).real
            pretty = f'{address.designer_name}:{param}'
            sig.name = pretty
            sig.ylabel = pretty
            return sig
        raise self._device_probe_read_error(raw, plot, address, param, analysis,
                                            matches, checkvalid_warned)

    def _device_probe_read_error(self, raw, plot, address, param, analysis,
                                 matches, checkvalid_warned=False):
        """Device-parameter availability diagnostic: names the device, the parameter, the
        analysis, this simulator's own VERSION as read from THIS run's log
        (never a hardcoded per-version table — availability is decided by
        what the run actually produced), and what the run DID provide for
        this SAME device in this SAME plot.

        *checkvalid_warned*: this vector DID come back with data (length
        >= 1, often a bare ``0.``), but ngspice's own log recorded a
        ``Warning from checkvalid: vector ... is not available or has
        zero length`` line naming it: that populated value is a filler,
        not a measurement, on a run saving enough OTHER device-parameter
        vectors alongside it to avoid the whole-write abort a smaller
        save set triggers for the exact same missing parameter. Vector
        *length* alone cannot tell this apart from a genuine 0 — only the
        log's own warning can, so every candidate in the "DID provide"
        enumeration below is checked the same way, not just *param*.
        """
        from silk.exceptions import SimulationError
        from silk.circuit.modules_results import device_probe_spelling
        import difflib
        version = raw.simulator_version or 'ngspice (version unknown)'
        provided = _ngspice_provided_device_params(raw, plot, address, param)
        state = ('ambiguous (multiple matching vectors)' if len(matches) > 1
                else 'not available on this run (ngspice checkvalid warning — '
                     'the value in the rawfile, if any, is a filler, not a '
                     'measurement)' if checkvalid_warned
                else 'zero-length' if matches else 'not found')
        hint = difflib.get_close_matches(param, DEVICE_OP_PARAMS, n=3, cutoff=0.6)
        return SimulationError(
            f"probe({device_probe_spelling(address.designer_name, param)}) "
            f"on {version}, analysis={_probe_analysis_label(analysis, plot)!r}: "
            f"{param!r} came back {state} for this device. This run DID "
            f"provide "
            f"{provided or '(no other device-parameter vectors for this device)'} "
            f"for {address.designer_name!r} in this plot."
            + (f" Did you mean {hint}?" if hint else "")
        )

    def preflight_unavailable_error(self, raw, address, param, analysis, result):
        """Availability diagnostic for a param the ngspice PREFLIGHT already found
        unavailable  — its save
        line was never emitted, so the main run's rawfile is never even
        consulted for it. Names the device, the param, the analysis, the
        binary version (the main run's own — the same binary the
        preflight itself used), and what the main run DID provide for
        this device (scanned from the actual rawfile, same as the
        ordinary read-time message)."""
        from silk.exceptions import SimulationError
        from silk.circuit.modules_results import device_probe_spelling
        version = raw.simulator_version or result.version or 'ngspice (version unknown)'
        plot = select_plot(raw.native, analysis) or raw.native
        provided = _ngspice_provided_device_params(raw, plot, address, param)
        return SimulationError(
            f"probe({device_probe_spelling(address.designer_name, param)}) "
            f"on {version}, analysis={_probe_analysis_label(analysis, plot)!r}: "
            f"this simulator does not provide "
            f"{param!r} for {address.model!r} (checked before the run — "
            f"preflight log: {result.log_path}). This run DID provide "
            f"{provided or '(no other device-parameter vectors for this device)'} "
            f"for {address.designer_name!r} in this plot."
        )

    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        files = self._resolve_run_files(netlist_path, results_dir)
        return self._launch_prepared(files, debug=debug, timeout=timeout)

    def _launch_prepared(self, run_files, *, debug=False, timeout=None):
        """Run ngspice and read its rawfile back.

        Hands the runner's log path up so :meth:`SimulatorBackend.run`'s gate
        can scrape an abort reason out of it.
        """
        from silk.simulation.mc_chunk import current as _current_chunk
        chunk = _current_chunk()
        if chunk is not None:
            return chunk.launch_member(run_files, debug=debug, timeout=timeout)
        from silk.simulation.runner import NgspiceRunner
        runner = NgspiceRunner(
            netlist_path=run_files.netlist_path,
            results_dir=run_files.directory,
            debug=debug,
            timeout=timeout,
            run_files=run_files,
        )
        runner._generation_managed = True
        runner.run()
        native = runner.load()
        raw = RawResult(native=native, source_path=run_files.rawfile_path,
                        run_files=run_files)
        return raw, runner.logfile_path, None

    def make_netlister(self, pdk_details):
        from silk.circuit.spice_netlister import PySpiceNetlister
        return PySpiceNetlister(pdk_details)


@SimulatorBackend.register('xyce')
class XyceBackend(SpiceBackend):
    """Xyce backend.

    Shares SPICE analysis rendering with ngspice (via :class:`SpiceBackend`)
    and emits the Xyce dialect through :class:`XyceNetlister`. ``run`` invokes
    the Xyce CLI (the deck's ``.print … format=raw file=…`` directive writes the
    rawfile) and reads it back through the shared SPICE reader.
    """

    def render_analysis(self, analysis, ctx: EmitContext) -> str:
        """Use a private zero-volt source for Xyce's one-point Op sweep.

        Xyce has no ``.print op``. Sweeping a user source overrides its DC
        clause and couples Op to source-emission details (including multi-card
        sources). This independent, grounded branch leaves every user source
        untouched. Collision checking covers the emitted body only, not the
        contents of .include files. Explicit .print targets keep the private
        branch out of saved signals.
        """
        if isinstance(analysis, Op):
            occupied = ctx.emitted_deck.lower()
            if not occupied:
                # Standalone calls need only identities, never source values
                # or waveform callbacks, to allocate a private name.
                names = []
                for source in ctx.sources.values():
                    names.extend(source.physical_instance_names())
                    names.extend(getattr(source, terminal)
                                 for terminal in source._node_terminals)
                occupied = '\n'.join(names).lower()
            node = '__silk_op'
            suffix = 0
            while node in occupied:
                suffix += 1
                node = f'__silk_op_{suffix}'
            name = 'V' + node
            value = '0'
            # Nonzero step avoids Xyce's "no steps taken" warning at 0 V.
            return (f'{name} {node} 0 DC {value}\n'
                    f'.dc {name} {value} {value} 1\n')
        return super().render_analysis(analysis, ctx)

    def _resolve_run_files(self, netlist_path, results_dir, *, analyses=None,
                           temperature=False):
        # Explicit-path callers have no analysis inventory; retain both legacy
        # candidates. Prepared Module decks name only applicable outputs.
        return RunFiles.resolve(
            netlist_path, results_dir,
            log=True,
            raw=analyses is None or any(not isinstance(a, Noise) for a in analyses),
            noise=analyses is None or any(isinstance(a, Noise) for a in analyses),
            temperature=temperature,
        )

    def read_ambient_temperatures(self, raw, log_path):
        """Read the TEMP probe this deck asked Xyce to print.

        Xyce writes no log silk can scrape, so a temperature-configured deck
        carries a second ``.print`` of the ``{TEMP}`` expression — the
        simulator's own evaluation of the circuit temperature at every solve
        point. No probe (an unconfigured run, or an externally supplied deck)
        means no evidence, hence no claim.
        """
        from silk.simulation.conditions import read_xyce_temperatures
        files = raw.run_files
        return read_xyce_temperatures(
            None if files is None else files.temperature_path)

    def read_simulator_version(self, raw, log_path):
        """The Xyce RELEASE this launch reported, from the captured stdout
        ``_run_runner`` stashed on *raw* — Xyce writes no log file, so
        *log_path* is always ``None`` here."""
        from silk.simulation.conditions import read_xyce_version_from_text
        return read_xyce_version_from_text(getattr(raw, '_xyce_stdout', None))

    def make_netlister(self, pdk_details):
        from silk.circuit.xyce_netlister import XyceNetlister
        return XyceNetlister(pdk_details)

    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        from silk.simulation.xyce_runner import XyceRunner
        files = self._resolve_run_files(netlist_path, results_dir)
        runner = XyceRunner(netlist_path, results_dir, debug=debug, timeout=timeout,
                            run_files=files, require_all_outputs=False)
        return self._run_runner(runner)

    def sample_args(self, sample) -> tuple:
        return ('-randseed', str(sample.seed))

    def _launch_prepared(self, run_files, *, debug=False, timeout=None,
                         extra_args=(), require_all_outputs=True):
        """Run Xyce and read its rawfile back.

        Xyce writes no log file, so the abort reason is scraped here from the
        captured stdout and handed up for :meth:`SimulatorBackend.run`'s gate
        to attach.
        """
        from silk.simulation.xyce_runner import XyceRunner
        runner = XyceRunner(
            netlist_path=run_files.netlist_path,
            results_dir=run_files.directory,
            debug=debug,
            timeout=timeout,
            run_files=run_files,
            extra_args=extra_args,
            require_all_outputs=require_all_outputs,
        )
        return self._run_runner(runner)

    def _run_runner(self, runner):
        run_files = runner.run_files
        runner._generation_managed = True
        runner.run()
        native = runner.load()   # None for a noise-only run (CSV, not rawfile)
        source_path = (run_files.rawfile_path if native is not None
                       else run_files.noise_path or run_files.rawfile_path)
        raw = RawResult(native=native, source_path=source_path,
                        run_files=run_files)
        # Xyce writes no log file — its version banner only ever exists in
        # the captured stdout, so it is stashed here (TASK-330.8) for
        # read_simulator_version to pick up. Private: not part of RawResult's
        # documented contract, just this backend's own handoff.
        raw._xyce_stdout = getattr(runner, 'stdout', '') or ''
        from silk.simulation.xyce_runner import xyce_abort_reason
        reason = xyce_abort_reason(raw._xyce_stdout)
        return raw, None, reason

    def op_param_capability(self, declared) -> tuple:
        """Intersect *declared* with ``XYCE_OP_CAPABILITY``, preserving
        *declared*'s own order."""
        return tuple(p for p in declared if p in XYCE_OP_CAPABILITY)

    def device_op_vectors(self, raw, instance_path, leaf_name, polarity=None) -> dict:
        """Xyce's op-point vectors for one device, read from its ``N()``/
        ``ID()`` ``.print`` columns.

        Match the complete emitted placement path and leaf from the root
        of the colon-separated hierarchy, after removing N()/ID()
        decoration. Internal-device and wrapper suffixes are allowed;
        a later subpath in another placement is never a match.

        Only ``XYCE_OP_CAPABILITY`` names are ever looked up; every other
        :data:`DEVICE_OP_PARAMS` name comes back ``nan`` unconditionally —
        Xyce never had a column for it to find (``gmbs``, the capacitances),
        confirmed unavailable by a real run.

        Saved operating-point output is classified as ``dc`` on Xyce.
        Return ``{}`` if no DC plot exists; otherwise select that plot through
        the shared rule, refusing multiple DC plots.

        ``ID()`` reads the plain SPICE branch-current sign (current INTO the
        drain terminal is positive) — negative for a PMOS in normal
        conduction. This is a DIFFERENT convention from ngspice's own ``'@'``
        vectors, which come back already sign-normalised to op_table()'s
        positive-magnitude convention for both polarities (``NgspiceBackend.device_op_vectors`` needs no correction for
        this reason). Verified on a
        real Xyce run: an uncorrected PMOS read `id = -1.98358e-05` where
        ngspice's own reading of the identical device was `+1.98357e-05`,
        and the downstream `gm_id = gm/id` derived column went negative with
        it. Corrected here using the device's OWN DECLARED polarity — never
        ``abs()``, which would silently hide a genuinely reversed current
        (e.g. a mis-biased device conducting backwards) behind a
        magnitude — flipping only the sign a `'p'`-declared device is known
        to need.
        """
        plots = getattr(raw.native, 'plots', None)
        if isinstance(plots, (list, tuple)) and not _matching_plots(plots, 'dc'):
            return {}                                # switch was off / no op data
        leaf = leaf_name.upper()
        path = instance_path.upper().strip('.')
        token = (path.split('.') if path else []) + [leaf]
        plot = select_plot(raw.native, 'dc') or raw.native
        vectors = plot.vectors
        out = {}
        for param in DEVICE_OP_PARAMS:
            if param not in XYCE_OP_CAPABILITY:
                out[param] = float('nan')
                continue
            matches = []
            for vec in vectors:
                inner = vec.name.upper()
                if param == 'id':
                    if not (inner.startswith('ID(') and inner.endswith(')')):
                        continue
                    body = inner[len('ID('):-1]
                else:
                    tail = f':{param.upper()})'
                    if not (inner.startswith('N(') and inner.endswith(tail)):
                        continue
                    body = inner[len('N('):-len(tail)]
                if body.split(':')[:len(token)] == token:
                    matches.append(vec)
            if len(matches) == 1 and len(matches[0].data):
                out[param] = float(matches[0].data[0].real)
            else:
                # 0 matches (not printed — undeclared/uncapable/off-Op), >1
                # (ambiguous) — both "unreadable", never a missing key.
                out[param] = float('nan')
        if polarity == 'p' and not math.isnan(out['id']):
            out['id'] = -out['id']
        return out

    def load_device_probe(self, raw, address, param, *, analysis=None):
        """:meth:`SimulatorBackend.load_device_probe` on Xyce.
        PATH+LEAF contiguous-sublist matching for parameter probes (the
        internal-name segment remains opaque to a reader), over the FULL ``.print`` column (every
        point), with the same PMOS ``ID()`` sign correction. Select the plot
        using *analysis* and the shared :func:`select_plot` rule. Repeated
        step plots of the same kind are ambiguous; save the intended result
        in a separate rawfile. Xyce operating-point output classifies as ``dc``.
        """
        from silk.signals.spice import NgspiceSignal
        plot = select_plot(raw.native, analysis) or raw.native
        leaf = address.leaf.upper()
        path = address.instance_path.upper().strip('.')
        token = (path.split('.') if path else []) + [leaf]
        matches = self._xyce_probe_matches(plot.vectors, token, param)
        if len(matches) == 1 and len(matches[0].data):
            sig = NgspiceSignal.from_rawfile(raw.native, matches[0].name, 'V', plot=plot)
            if param == 'id' and address.polarity == 'p':
                sig.y = -sig.y
            pretty = f'{address.designer_name}:{param}'
            sig.name = pretty
            sig.ylabel = pretty
            return sig
        raise self._device_probe_read_error(raw, address, param, analysis, matches)

    @staticmethod
    def _xyce_probe_matches(vectors, token, param):
        matches = []
        for vec in vectors:
            inner = vec.name.upper()
            if param == 'id':
                if not (inner.startswith('ID(') and inner.endswith(')')):
                    continue
                body = inner[len('ID('):-1]
            else:
                tail = f':{param.upper()})'
                if not (inner.startswith('N(') and inner.endswith(tail)):
                    continue
                body = inner[len('N('):-len(tail)]
            if _contiguous_sublist(token, body.split(':')):
                matches.append(vec)
        return matches

    def _device_probe_read_error(self, raw, address, param, analysis, matches):
        """Device-parameter availability diagnostic on Xyce: names the device, the
        parameter, the analysis, this Xyce RELEASE as read from THIS run's
        own transcript, and what the run DID provide (within
        ``XYCE_OP_CAPABILITY`` — Xyce can never provide anything
        outside it, whatever the device declares)."""
        import difflib
        from silk.exceptions import SimulationError
        from silk.circuit.modules_results import device_probe_spelling
        version = raw.simulator_version or 'Xyce (version unknown)'
        plot = select_plot(raw.native, analysis) or raw.native
        leaf = address.leaf.upper()
        path = address.instance_path.upper().strip('.')
        token = (path.split('.') if path else []) + [leaf]
        provided = []
        for candidate in XYCE_OP_CAPABILITY:
            if candidate == param:
                continue
            cmatches = self._xyce_probe_matches(plot.vectors, token, candidate)
            if len(cmatches) == 1 and len(cmatches[0].data):
                provided.append(candidate)
        state = ('ambiguous (multiple matching vectors)' if len(matches) > 1
                else 'zero-length' if matches else 'not found')
        hint = difflib.get_close_matches(param, XYCE_OP_CAPABILITY, n=3, cutoff=0.6)
        return SimulationError(
            f"probe({device_probe_spelling(address.designer_name, param)}) "
            f"on {version}, analysis="
            f"{_probe_analysis_label(analysis, plot)!r}: {param!r} "
            f"came back {state} for this "
            f"device. This run DID provide "
            f"{provided or '(no other device-parameter vectors for this device)'} "
            f"for {address.designer_name!r} in this plot."
            + (f" Did you mean {hint}?" if hint else "")
        )

    def load_noise(self, raw, port, *, analysis=None):
        """Load canonical noise (V/√Hz) from Xyce's noise CSV.

        Xyce writes noise to ``noise.csv`` (it cannot use a rawfile) with
        ``ONOISE``/``INOISE`` in **V²/Hz**; convert to amplitude density (V/√Hz)
        with a sqrt so the result matches the ngspice/canonical convention.

        *analysis* is accepted and IGNORED: Xyce's noise output is a separate
        file, not one plot among several, so there is nothing here to select
        between. The parameter exists so the shared ``load_signal`` can pass
        it without asking which backend it is talking to.
        """
        import numpy as np
        from silk.signals.spice import NgspiceSignal
        from silk.simulation.xyce_runner import read_xyce_noise_csv

        files = raw.run_files or RunFiles.for_raw_source(raw.source_path)
        if files.noise_path is None:
            raise NotSupportedError('This prepared Xyce run has no noise output.')
        from silk.simulation.generation import artifact_read
        with artifact_read(raw, 'noise_path') as noise_path:
            if noise_path is None and getattr(raw, '_generation', None) is not None:
                raise NotSupportedError('This generation produced no noise output.')
            cols = read_xyce_noise_csv(noise_path or files.noise_path)
        column = {'out': 'ONOISE', 'in': 'INOISE'}[port]
        freq = cols['FREQ']
        amp = np.sqrt(cols[column])   # V²/Hz -> V/√Hz
        return NgspiceSignal(
            name=f'noise_{port}', type='voltage-density',
            xlabel='frequency', ylabel=f'noise_{port}',
            xvalues=freq, yvalues=amp,
        )


# Register the numpy engine alongside the SPICE backends.
from silk.simulation import behavioral_backend  # noqa: E402,F401


# Explicit provider identity: overridden/custom launchers own their artifacts.
_BUILTIN_LAUNCH = {cls: (cls._launch, cls._launch_prepared)
                   for cls in (NgspiceBackend, XyceBackend)}
