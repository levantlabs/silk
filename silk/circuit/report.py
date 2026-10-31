"""silk.circuit.report — circuit summary report.

Assembles topology, port annotation coverage, interface check results,
and behavioral simulation stats into a CircuitReport that can be printed
or saved as a formatted Markdown file.
"""
from __future__ import annotations

import base64
import datetime
import html as _html
import io
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from silk._json_codec import decode_json, encode_json
from silk.lint.severity import Severity
from silk.lint.reports.metadata import ReportMetadata
from silk.lint.reports.lint_report import (
    LintReport, _location_to_dict, _location_from_dict,
)
from silk.lint.reports.coverage_report import CoverageReport
from silk.lint.stress_coverage import stress_coverage_text
from silk.specs import SpecReconciliationReport
from silk.signals.analysis_kind import (
    ANALYSIS_OP as _ANALYSIS_OP,
    ANALYSIS_TRANSIENT as _ANALYSIS_TRANSIENT,
    SMALL_SIGNAL_KINDS as _SMALL_SIGNAL_KINDS,
    rawfile_analysis_kind,
)

if TYPE_CHECKING:
    from silk.circuit.interface_check import AnnotationCoverage, InterfaceViolation
    from silk.circuit.junction_check import JunctionViolation


# ``save_json``'s on-disk schema version. Bump only on a breaking change to
# the JSON shape; ``load_json`` accepts any file with no 'schema_version' key
# (pre-versioning files) or an older supported version, and raises on anything
# newer that this silk version does not know how to read.
REPORT_JSON_SCHEMA_VERSION = 5

# Sections CircuitReport.save_json does not (yet) serialize. Written into
# every JSON file's 'omitted_sections' key so the file is honest about being
# a partial snapshot rather than silently dropping lint/coverage data.
_OMITTED_JSON_SECTIONS = ['lint', 'coverage']

# Display-only seconds -> ns / ps factors for the rendered simulation
# summary (AUDIT-02: named so no bare 1e9/1e12 sits in an f-string).
_S_TO_NS = 1e9
_S_TO_PS = 1e12


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@dataclass
class InstanceSummary:
    """One subcircuit instance in the hierarchy.

    Parameters
    ----------
    instance : str
        Instance name, the key in the parent's ``module_dict`` (for example
        ``'comp1'``).
    module_name : str
        Subcircuit definition name (for example ``'comp'``); the instance
        name for an array-instantiated entry.
    port_count : int
        Number of pins in the subcircuit's ``pin_list``.
    """
    instance: str       #: Name in ``module_dict`` (e.g. ``'comp1'``).
    module_name: str    #: Subcircuit definition name (e.g. ``'comp'``).
    port_count: int     #: Number of ports declared on the subcircuit.


@dataclass
class TopologySummary:
    """High-level view of a module's subcircuit hierarchy.

    Parameters
    ----------
    module_name : str
        Name of the module; ``'unnamed'`` for a module without one.
    instances : list of InstanceSummary
        Subcircuit instances in ``module_dict`` order.
    total_nets : int
        Number of nets in the module's ``net_dict``.
    """
    module_name: str                   #: Module name; ``'unnamed'`` when the module has none.
    instances: List[InstanceSummary]   #: Subcircuit instances, in ``module_dict`` order.
    total_nets: int                    #: Number of nets in ``net_dict``.

    @property
    def total_instances(self) -> int:
        """Number of entries in :attr:`instances`."""
        return len(self.instances)


def _build_topology(module) -> TopologySummary:
    """Extract topology information from a Module."""
    instances = []
    for inst_name, entry in module.module_dict.items():
        submodule = entry.get('device')
        if submodule is None:
            continue
        pin_list = getattr(submodule, 'pin_list', None)
        if pin_list is None:
            pin_list = {}
        sub_name = (inst_name if 'numberInstantiated' in entry else
                    getattr(submodule, 'module_name', inst_name) or inst_name)
        instances.append(InstanceSummary(
            instance=inst_name,
            module_name=sub_name,
            port_count=len(pin_list),
        ))
    return TopologySummary(
        module_name=getattr(module, 'module_name', None) or 'unnamed',
        instances=instances,
        total_nets=len(getattr(module, 'net_dict', {})),
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SignalStats:
    """Per-signal statistics of a value-domain run (transient, DC sweep):
    the plain min / max / mean of the saved vector, in its own unit (volts
    for ``v(...)``, amps for ``i(...)``).

    Parameters
    ----------
    name : str
        Signal label with a ``v(...)`` wrapper stripped (``'vout'``); a
        current keeps its tag (``'i(vsupply)'``).
    min, max, mean : float
        Minimum, maximum and mean of the saved vector, in its own unit.
    """
    name: str     #: Signal label (``'vout'``, ``'i(vsupply)'``).
    min: float    #: Minimum of the vector, in its own unit.
    max: float    #: Maximum of the vector, in its own unit.
    mean: float   #: Mean of the vector, in its own unit.


@dataclass
class MagnitudeStats:
    """Per-signal statistics of a small-signal run (AC, noise): the min /
    max / mean of the vector's MAGNITUDE ``|x|``.

    Named so no consumer can mistake them for voltages: an AC vector is a
    complex quantity and a noise vector a spectral density. Magnitudes are
    taken of the raw vector, with no normalisation by the stimulus.
    The magnitude is the only reduction that is honest for a complex
    vector — numpy's ``min``/``max`` on complex input are lexicographic
    (real part, then imaginary), and ``float()`` of one keeps the real part
    silently.

    Parameters
    ----------
    name : str
        Signal label, as in :class:`SignalStats`.
    mag_min, mag_max, mag_mean : float
        Minimum, maximum and mean of ``|x|``, in the vector's own unit
        (V or A for an AC vector, V/sqrt(Hz) or A/sqrt(Hz) for a noise
        density).
    """
    name: str         #: Signal label.
    mag_min: float    #: Minimum of ``|x|``.
    mag_max: float    #: Maximum of ``|x|``.
    mag_mean: float   #: Mean of ``|x|``.


@dataclass
class SimulationSummary:
    """Key facts about a completed simulation, whichever analysis ran.

    ``analysis`` says which kind (``'transient'``, ``'ac'``, ``'noise'``,
    ``'dc'``, ``'op'`` — the vocabulary of
    ``silk.signals.analysis_kind.rawfile_analysis_kind``) and the axis fields are named for the axis
    actually present: ``t_start``/``t_stop`` are set only when the axis is
    time, ``f_start``/``f_stop`` only when it is frequency, and
    ``sweep_start``/``sweep_stop`` whenever there is an axis (the generic
    bounds, with ``sweep_name`` the axis vector's own name — ``'time'``,
    ``'frequency'``, ``'v(v-sweep)'`` on an ngspice DC sweep, ``'sweep'``
    on a Xyce one). An operating point has no axis at all: ``sweep_name``
    is ``''``, every bound is ``None``, ``n_points`` is 1 and ``signals``
    lists EVERY vector of the plot with ``min == max == mean`` = its value
    (vector 0 is a node on ngspice and Xyce's placeholder ``sweep`` — it is
    listed, never dropped as an axis). ``signals`` holds value
    statistics for a value-domain run and is empty for a small-signal one,
    whose vectors are summarised by magnitude in ``magnitudes`` instead;
    the two lists never both hold entries.

    The transient field names predate the other kinds and are kept so
    every existing JSON consumer reads unchanged; the rest are additive.

    Parameters
    ----------
    t_start, t_stop : float or None
        First and last time point in seconds; ``None`` unless the axis is
        time.
    n_points : int
        Number of points on the axis (1 for an operating point).
    signals : list of SignalStats
        Value statistics for transient, DC and operating-point runs; empty
        for a small-signal run.
    analysis : str, default 'transient'
        Analysis kind: ``'transient'``, ``'ac'``, ``'noise'``, ``'dc'`` or
        ``'op'``.
    sweep_name : str, default 'time'
        Name of the axis vector; ``''`` for an operating point.
    sweep_start, sweep_stop : float or None
        First and last axis value, in the axis's own unit (s, Hz or the
        swept source's unit). When omitted they mirror ``t_start``/``t_stop``
        or, failing that, ``f_start``/``f_stop``.
    f_start, f_stop : float or None
        First and last frequency in Hz; ``None`` unless the axis is
        frequency.
    magnitudes : list of MagnitudeStats
        Magnitude statistics for AC and noise runs; empty otherwise.
    """
    t_start: Optional[float]   #: First time point (s); ``None`` unless the axis is time.
    t_stop: Optional[float]    #: Last time point (s); ``None`` unless the axis is time.
    n_points: int              #: Number of axis points (1 for an operating point).
    signals: List[SignalStats]   #: Value statistics (transient, DC, op); empty for AC/noise.
    analysis: str = 'transient'   #: Analysis kind, one of ``'transient'``, ``'ac'``, ``'noise'``, ``'dc'`` or ``'op'``.
    sweep_name: str = 'time'      #: Name of the axis vector (``'time'``, ``'frequency'``, ``'v(v-sweep)'``, ``'sweep'``); ``''`` for an operating point.
    sweep_start: Optional[float] = None   #: First axis value in the axis's own unit; ``None`` for an operating point.
    sweep_stop: Optional[float] = None    #: Last axis value in the axis's own unit; ``None`` for an operating point.
    f_start: Optional[float] = None   #: First frequency (Hz); ``None`` unless the axis is frequency.
    f_stop: Optional[float] = None    #: Last frequency (Hz); ``None`` unless the axis is frequency.
    magnitudes: List[MagnitudeStats] = field(default_factory=list)   #: Magnitude statistics (AC, noise); empty otherwise.

    def __post_init__(self):
        # The generic bounds mirror whichever named axis was given, so a
        # constructor call (or a pre-TASK-82 JSON file) that knows only
        # ``t_start``/``t_stop`` still yields a fully populated summary.
        if self.sweep_start is None:
            self.sweep_start = self.t_start if self.t_start is not None else self.f_start
        if self.sweep_stop is None:
            self.sweep_stop = self.t_stop if self.t_stop is not None else self.f_stop

    @property
    def t_step(self) -> Optional[float]:
        """Mean time-axis spacing, ``None`` when the axis is not time.

        This is ``(t_stop - t_start) / (n_points - 1)``, the MEAN spacing —
        a simulator's transient grid is adaptive, so it is not the step any
        two points are apart: the ngspice fixture (``.tran 20n 50u``,
        2629 points) reports 19.0 ns here. ``0.0`` when there are fewer than
        two points.
        """
        if self.t_start is None or self.t_stop is None:
            return None
        if self.n_points < 2:
            return 0.0
        return (self.t_stop - self.t_start) / (self.n_points - 1)

    @property
    def sweep_step(self) -> Optional[float]:
        """Mean axis spacing in the axis's own unit, for the linearly swept
        kinds (transient, DC); ``None`` for AC/noise, whose grids are
        logarithmic and have no meaningful single step. Same mean-spacing
        caveat as :attr:`t_step`."""
        if self.analysis in _SMALL_SIGNAL_KINDS:
            return None
        if self.sweep_start is None or self.sweep_stop is None:
            return None
        if self.n_points < 2:
            return 0.0
        return (self.sweep_stop - self.sweep_start) / (self.n_points - 1)

    @property
    def n_signals(self) -> int:
        """How many vectors were summarised, whichever list holds them."""
        return len(self.signals) + len(self.magnitudes)


def _summary_for_axis(analysis: str, axis_name: str, axis, n_points: int,
                      signals: List[SignalStats],
                      magnitudes: List[MagnitudeStats]) -> SimulationSummary:
    """Build a summary with the axis fields that match *analysis*."""
    start, stop = float(axis[0]), float(axis[-1])
    time_axis = analysis == _ANALYSIS_TRANSIENT
    freq_axis = analysis in _SMALL_SIGNAL_KINDS
    return SimulationSummary(
        t_start=start if time_axis else None,
        t_stop=stop if time_axis else None,
        n_points=n_points,
        signals=signals,
        analysis=analysis,
        sweep_name=axis_name,
        sweep_start=start,
        sweep_stop=stop,
        f_start=start if freq_axis else None,
        f_stop=stop if freq_axis else None,
        magnitudes=magnitudes,
    )


def _summary_for_op(signals: List[SignalStats]) -> SimulationSummary:
    """An operating point: one point, no axis, every vector a value."""
    return SimulationSummary(
        t_start=None,
        t_stop=None,
        n_points=1,
        signals=signals,
        analysis=_ANALYSIS_OP,
        sweep_name='',
        sweep_start=None,
        sweep_stop=None,
    )


def _signal_label(sig_name: str) -> str:
    """``'v(vout)'`` -> ``'vout'``; ``'i(vsupply)'`` -> ``'i(vsupply)'``
    (a current keeps its tag); a bare name passes through."""
    if '(' in sig_name:
        lhs, _, rhs = sig_name.partition('(')
        inner = rhs.rstrip(')')
        return inner if lhs.lower() == 'v' else sig_name
    return sig_name


def _net_match_key(label: str) -> str:
    """Case-folded, wrapper-stripped form of a rendered signal label, used
    ONLY to test membership in ``module.net_dict`` — never to change what
    is stored or printed. ``_signal_label`` already strips a ``v(...)``
    label down to the bare net name but leaves its case, and Xyce's raw
    vectors are upper-cased (``'V(IN)'`` -> ``'IN'``), so a bare label is
    folded to lower-case; a label that still carries an ``i(...)`` tag
    (currents keep theirs) is folded the same way its inner name would be.
    Case-folded like ``NgspiceSignal.from_rawfile``'s ``xlabel`` fold and ``_build_simulation``'s ``sweep_name`` fold, so
    the membership test agrees with both backends."""
    if '(' in label:
        _, _, rhs = label.partition('(')
        return rhs.rstrip(')').lower()
    return label.lower()


@dataclass
class OpTableSection:
    """One :class:`~silk.circuit.spec_report.SpecReport`'s operating-point
    table, paired with the identity used to heading and order it in the
    report.

    Built by ``_build_op_tables`` from the ``spec_reports=`` passed to
    ``generate_report`` — one section per SpecReport whose
    ``op_devices`` was actually captured at ``evaluate()``/``run()`` time
    (``save_device_params=True`` capture). A SpecReport with
    ``op_devices is None`` (no capture) contributes no section at all,
    never an empty placeholder one — this is what makes passing a
    no-capture report byte-identical to omitting ``spec_reports=``
    entirely.

    Parameters
    ----------
    test_name : str
        Name of the test whose SpecReport produced the table.
    corner : str or None
        Corner the SpecReport ran at; ``None`` when it ran at none.
    table : OpTable
        The :class:`~silk.circuit.op_table.OpTable` returned by the
        SpecReport's ``op_table()``.
    """
    test_name: str                          #: Test whose SpecReport produced the table.
    corner: Optional[str]                   #: Corner of that SpecReport; ``None`` when none.
    table: Any                              #: The :class:`~silk.circuit.op_table.OpTable` for that SpecReport.


@dataclass
class MetricWarningSection:
    """The MetricWarnings one SpecReport recorded for one output.

    ``output`` is the output (metric) name the warnings are attributed to,
    ``warnings`` are ``MetricWarningRecord`` objects in emission order.

    Parameters
    ----------
    test_name : str
        Test whose SpecReport recorded the warnings.
    owner_path : str
        Placement path of the SpecReport's owner (``'$'`` for the top level).
    corner : str or None
        Corner the SpecReport ran at; ``None`` when none.
    output : str
        Output (metric) name the warnings are attributed to.
    warnings : tuple
        Warnings in emission order.
    spec_keys : tuple of str, default ``()``
        Keys of the Specs the SpecReport checked against this output's
        metric; a reconciliation row shows these warnings only when its
        ``spec.key`` is here.
    """
    test_name: str            #: Test whose SpecReport recorded the warnings.
    owner_path: str           #: Placement path of the SpecReport's owner (``'$'`` at top level).
    corner: Optional[str]     #: Corner of the SpecReport; ``None`` when none.
    output: str               #: Output (metric) name the warnings are attributed to.
    warnings: tuple           #: ``MetricWarningRecord`` objects in emission order.
    #: Keys of the Specs the SpecReport checked against this output's metric:
    #: a reconciliation row shows these warnings only when its spec.key is here.
    spec_keys: tuple = ()

    def to_dict(self) -> dict:
        """Return a JSON-ready dict; ``spec_keys`` and ``warnings`` become lists.

        Returns
        -------
        dict
            Keys ``test_name``, ``owner_path``, ``corner``, ``output``,
            ``spec_keys`` and ``warnings`` (each warning as its own
            ``to_dict()``). Inverse of :meth:`from_dict`.
        """
        return {'test_name': self.test_name, 'owner_path': self.owner_path,
                'corner': self.corner, 'output': self.output,
                'spec_keys': list(self.spec_keys),
                'warnings': [w.to_dict() for w in self.warnings]}

    @classmethod
    def from_dict(cls, d: dict) -> 'MetricWarningSection':
        """Rebuild a section from :meth:`to_dict` output.

        Parameters
        ----------
        d : dict
            Dict with ``test_name``, ``output`` and ``warnings``; ``owner_path``
            defaults to ``'$'``, ``corner`` to ``None`` and ``spec_keys`` to
            empty.

        Returns
        -------
        MetricWarningSection

        Raises
        ------
        KeyError
            If ``test_name``, ``output`` or ``warnings`` is missing.
        """
        from silk.metrics.diagnostics import MetricWarningRecord
        return cls(d['test_name'], d.get('owner_path', '$'), d.get('corner'),
                   d['output'],
                   tuple(MetricWarningRecord.from_dict(w) for w in d['warnings']),
                   tuple(d.get('spec_keys') or ()))


def _build_metric_warnings(spec_reports) -> List[MetricWarningSection]:
    """One section per (SpecReport, output) that recorded warnings, in the
    caller's order. Reports with no warnings contribute nothing, so a clean
    run is byte-identical to omitting ``spec_reports=``."""
    sections = []
    for sr in spec_reports or []:
        for output, recs in (getattr(sr, 'metric_warnings', None) or {}).items():
            if recs:
                sections.append(MetricWarningSection(
                    test_name=sr.test_name,
                    owner_path=getattr(sr, 'owner_path', '$'),
                    corner=getattr(sr, 'corner', None),
                    output=output, warnings=tuple(recs),
                    spec_keys=tuple(
                        r.spec_key for r in getattr(sr, '_results', ())
                        if r.metric_name == output and r.spec_key is not None)))
    return sections


def _warning_notes(sections, outcome, value) -> List[str]:
    """Display lines for the warnings of the reconciliation row *outcome*:
    ``unavailable`` when *value* is nan, ``note`` otherwise. A section counts
    only when its SpecReport checked this spec at the same known owner (same
    metric and immutable key). Unknown ownership never matches. When
    several sections match, each line names its test, owner path and corner.
    Never affects a verdict."""
    from silk.circuit.spec_report import metric_warning_text
    unavailable = ((isinstance(value, float) and value != value)
                   or (outcome.latest_attempt is not None
                       and outcome.latest_attempt.kind == 'unavailable'))
    label = 'unavailable' if unavailable else 'note'
    lines = []
    sections = [sec for sec in sections
                if outcome.owner_path is not None
                and sec.owner_path == outcome.owner_path
                and sec.output == outcome.spec.metric
                and outcome.spec.key in sec.spec_keys]
    for sec in sections:
        prefix = ''
        if len(sections) > 1:
            prefix = f'[{sec.test_name} {sec.owner_path}'
            prefix += f' {sec.corner}] ' if sec.corner is not None else '] '
        for rec in sec.warnings:
            lines.append(f'{label}: {prefix}{metric_warning_text(rec)}')
    return lines


def _build_op_tables(spec_reports, region_thresholds=None) -> List[OpTableSection]:
    """Build the report's op-table sections from caller-supplied SpecReports.

    ``spec_reports`` is taken in the caller's own order and kept in that
    order — the report does not re-sort or group by
    corner. Each SpecReport's ``op_table()`` (operating-point capture) is called exactly
    once; a SpecReport with no capture (``op_devices is None``) is skipped.
    """
    if not spec_reports:
        return []
    sections = []
    for sr in spec_reports:
        if getattr(sr, 'op_devices', None) is None:
            continue
        sections.append(OpTableSection(
            test_name=sr.test_name,
            corner=sr.corner,
            table=(sr.op_table() if region_thresholds is None
                   else sr.op_table(region_thresholds=region_thresholds)),
        ))
    return sections


def _build_simulation(module) -> Optional[SimulationSummary]:
    """Extract simulation stats from a Module that has run ``simulate()``
    (ngspice / Xyce) or ``simulate_behavioral()``.

    Returns ``None`` if no simulation has been run. The legacy behavioral
    shortcut reads ``Net.voltage``; backend runs (behavioral or SPICE) read
    the module's rawfile (``module.simResultsAll``), summarising every saved
    vector.

    The SPICE path follows the reader's own rule for the independent axis:
    vector 0 IS the axis, whatever it is called (``time``, ``frequency``,
    ``v(v-sweep)``, ``sweep``) — the name whitelist this used to apply left
    every DC sweep with no summary at all. Which statistics are
    taken is decided by the shared analysis-kind classifier, so this and
    STR-001 cannot disagree about the same plot: a value-domain run gets
    plain min/max/mean, a small-signal run gets magnitude statistics under
    names that say so, and nothing is ever real-cast. The one plot with no
    axis is an operating point: there vector 0 is the first node (ngspice)
    or a placeholder (Xyce), so every vector is listed and none is taken
    as the axis — the vector-0 rule used to report ``v(in)`` as
    ``sweep_name`` and drop it from the table.
    """
    # Legacy behavioral path only; backend rawfiles use the shared reader
    # below so a DC sweep is never described as elapsed time.
    t = getattr(module, '_behavioral_t', None)
    if t is not None and getattr(module, '_raw', None) is None:
        signals = []
        net_dict = getattr(module, 'net_dict', {})
        for name in sorted(net_dict.keys()):
            net = net_dict[name]
            arr = getattr(net, 'voltage', None)
            if arr is None:
                continue
            signals.append(SignalStats(
                name=name,
                min=float(arr.min()),
                max=float(arr.max()),
                mean=float(arr.mean()),
            ))
        if not signals:
            return None
        return _summary_for_axis(_ANALYSIS_TRANSIENT, 'time', t, len(t),
                                 signals, [])

    # SPICE path — read from simResultsAll rawfile object
    if getattr(module, 'sim_results_loaded', False):
        raw = getattr(module, 'simResultsAll', None)
        if raw is None or not hasattr(raw, 'vectors') or raw.nvars == 0:
            return None
        analysis = rawfile_analysis_kind(raw)
        if analysis is None:
            return None

        if analysis == _ANALYSIS_OP:
            values: List[SignalStats] = []
            for vec in raw.vectors:
                arr = vec.data
                if arr is None or len(arr) == 0:
                    continue
                value = float(arr[0])
                values.append(SignalStats(name=_signal_label(vec.name),
                                          min=value, max=value, mean=value))
            if not values:
                return None
            values.sort(key=lambda s: s.name)
            return _summary_for_op(values)

        axis_vec = raw.vectors[0]
        axis = axis_vec.data
        if axis is None or len(axis) == 0:
            return None
        # A complex plot stores its axis as (re, im) pairs; the reader hands
        # it back realified, but a hand-built result may not.
        axis = np.real(axis) if np.iscomplexobj(axis) else axis

        signals: List[SignalStats] = []
        magnitudes: List[MagnitudeStats] = []
        small_signal = analysis in _SMALL_SIGNAL_KINDS
        for i in range(1, raw.nvars):
            label = _signal_label(raw.vectors[i].name)
            arr = raw.vectors[i].data
            if arr is None or len(arr) == 0:
                continue
            if small_signal:
                mag = np.abs(arr)
                magnitudes.append(MagnitudeStats(
                    name=label,
                    mag_min=float(mag.min()),
                    mag_max=float(mag.max()),
                    mag_mean=float(mag.mean()),
                ))
            else:
                signals.append(SignalStats(
                    name=label,
                    min=float(arr.min()),
                    max=float(arr.max()),
                    mean=float(arr.mean()),
                ))
        signals.sort(key=lambda s: s.name)
        magnitudes.sort(key=lambda s: s.name)
        if not signals and not magnitudes:
            return None
        # Case-folded like NgspiceSignal.from_rawfile's xlabel (TASK-87), so
        # a Xyce report says 'time'/'frequency' exactly as an ngspice one.
        return _summary_for_axis(analysis, axis_vec.name.lower(), axis,
                                 len(axis), signals, magnitudes)

    return None


# ---------------------------------------------------------------------------
# Simulation summary rendering (shared by terminal / Markdown / HTML)
# ---------------------------------------------------------------------------

def _sim_axis_text(sim: SimulationSummary) -> Tuple[str, str]:
    """``(label, text)`` for the summary's swept axis, in the axis's own
    unit: nanoseconds for time, hertz for frequency, and the raw sweep
    values under the axis vector's name for a DC sweep (whose unit the name
    already carries — ``v(v-sweep)`` is volts, ``temp-sweep`` is degrees).
    An operating point has no axis to describe and says so."""
    if sim.analysis == _ANALYSIS_OP:
        return ('Operating point', 'one point, no swept axis')
    if sim.t_start is not None and sim.t_stop is not None:
        return ('Time range',
                f'{sim.t_start * _S_TO_NS:.1f} ns → {sim.t_stop * _S_TO_NS:.1f} ns')
    if sim.f_start is not None and sim.f_stop is not None:
        return ('Frequency range', f'{sim.f_start:g} Hz → {sim.f_stop:g} Hz')
    return ('Sweep range',
            f'{sim.sweep_name} {sim.sweep_start:g} → {sim.sweep_stop:g}')


def _sim_points_text(sim: SimulationSummary) -> str:
    """``'<n> (step: ...)'`` — the step is given only where a mean spacing
    means something (see :attr:`SimulationSummary.sweep_step`)."""
    if sim.t_step is not None:
        return f'{sim.n_points} (step: {sim.t_step * _S_TO_PS:.1f} ps)'
    if sim.sweep_step is not None:
        return f'{sim.n_points} (step: {sim.sweep_step:g})'
    return f'{sim.n_points}'


def _sim_table_rows(sim: SimulationSummary) -> Tuple[Tuple[str, ...],
                                                     List[Tuple]]:
    """``(header, rows)`` for the per-signal table: value statistics with
    the familiar ``(V)`` headers, magnitude statistics labelled as such, or
    — for an operating point, where min/max/mean would print one number
    three times — a single ``Value`` column. Every row is ``(name, *nums)``
    with ``len(header) - 1`` numbers."""
    if sim.analysis == _ANALYSIS_OP:
        header = ('Signal', 'Value (V)')
        rows = [(s.name, s.mean) for s in sim.signals]
    elif sim.magnitudes:
        header = ('Signal', 'Magnitude min', 'Magnitude max', 'Magnitude mean')
        rows = [(m.name, m.mag_min, m.mag_max, m.mag_mean) for m in sim.magnitudes]
    else:
        header = ('Signal', 'Min (V)', 'Max (V)', 'Mean (V)')
        rows = [(s.name, s.min, s.max, s.mean) for s in sim.signals]
    return header, rows


# ---------------------------------------------------------------------------
# Waveform-view href helper
# ---------------------------------------------------------------------------

def _relativize_href(path: str, base_dir) -> str:
    """Relative path from *base_dir* to *path* when it is a sibling/descendant
    of *base_dir*; the absolute *path* otherwise."""
    if not base_dir:
        return path
    try:
        rel = os.path.relpath(path, base_dir)
    except ValueError:
        return path  # e.g. different drive on Windows
    return path if rel.startswith('..') else rel


# ---------------------------------------------------------------------------
# Severity compatibility helpers
# ---------------------------------------------------------------------------


def _severity_of(v) -> Severity:
    """Return the Severity of a violation, tolerating legacy string values.

    silk.lint.violations.Violation carries
    a Severity enum in .severity. Older persisted reports and a handful of
    legacy constructors still use the string 'ERROR' / 'WARNING'.
    """
    s = v.severity
    if isinstance(s, Severity):
        return s
    if isinstance(s, str):
        try:
            return Severity.from_string(s)
        except ValueError:
            return Severity.INFO
    return Severity.INFO


def _severity_name(v) -> str:
    """Return an uppercase severity label for rendering."""
    return _severity_of(v).name


# ---------------------------------------------------------------------------
# JSON reconstruction helpers
# ---------------------------------------------------------------------------

def _topology_from_dict(d: dict) -> TopologySummary:
    return TopologySummary(
        module_name=d['module_name'],
        instances=[InstanceSummary(**inst) for inst in d['instances']],
        total_nets=d['total_nets'],
    )


def _simulation_from_dict(d: dict) -> SimulationSummary:
    """Rebuild a summary from ``asdict`` output. Legacy files containing only
    ``t_start``, ``t_stop``, ``n_points`` and ``signals`` use transient defaults
    for the additive fields. ``__post_init__`` fills the generic bounds from
    ``t_start`` and ``t_stop``."""
    return SimulationSummary(
        t_start=d['t_start'],
        t_stop=d['t_stop'],
        n_points=d['n_points'],
        signals=[SignalStats(**s) for s in d['signals']],
        analysis=d.get('analysis', 'transient'),
        sweep_name=d.get('sweep_name', 'time'),
        sweep_start=d.get('sweep_start'),
        sweep_stop=d.get('sweep_stop'),
        f_start=d.get('f_start'),
        f_stop=d.get('f_stop'),
        magnitudes=[MagnitudeStats(**m) for m in d.get('magnitudes', [])],
    )


def _annotation_from_dict(d: dict) -> 'AnnotationCoverage':
    from silk.circuit.interface_check import AnnotationCoverage, UnannotatedPort
    return AnnotationCoverage(
        total=d['total'],
        annotated=d['annotated'],
        unannotated_ports=[UnannotatedPort(**u) for u in d['unannotated_ports']],
    )


def _report_location_from_dict(data, legacy_type, path):
    """Read tagged locations or the complete structured legacy shape."""
    try:
        if not isinstance(data, dict):
            raise ValueError("location must be an object")
        if '__type' in data:
            return _location_from_dict(data)
        if 'describe' in data:
            raise ValueError("location was saved as text only and cannot be restored; "
                             "regenerate the report")
        keys = ({'net_name', 'port_names'} if legacy_type == 'net' else
                {'instance_path', 'terminal', 'terminal_pair'})
        if set(data) != keys:
            raise ValueError("incomplete or malformed legacy location")
        return _location_from_dict({'__type': legacy_type, **data})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _violation_from_dict(d: dict, path: str = 'violations') -> 'InterfaceViolation':
    """Rebuild an InterfaceViolation with its tagged or legacy net location."""
    from silk.lint.violations import InterfaceViolation

    location = _report_location_from_dict(d.get('location'), 'net', f'{path}.location')
    sev = d['severity']
    if isinstance(sev, str):
        sev = Severity.from_string(sev)
    else:
        sev = Severity(int(sev))
    return InterfaceViolation(
        rule_id=d['rule_id'],
        severity=sev,
        message=d['message'],
        location=location,
        context=d.get('context', {}),
    )


def _violation_to_dict(v) -> dict:
    """Serialize an InterfaceViolation (new lint type) to a JSON-safe dict."""
    loc_dict = _location_to_dict(v.location)
    return {
        'rule_id':  v.rule_id,
        'severity': _severity_name(v),
        'message':  v.message,
        'location': loc_dict,
        'context':  dict(getattr(v, 'context', {}) or {}),
    }


def _junction_violation_to_dict(v) -> dict:
    """Serialize a JunctionViolation (new lint type) to a JSON-safe dict."""
    loc_dict = _location_to_dict(v.location)
    return {
        'rule_id':      v.rule_id,
        'severity':     _severity_name(v),
        'message':      v.message,
        'location':     loc_dict,
        'context':      dict(getattr(v, 'context', {}) or {}),
        'device_type':  getattr(v, 'device_type', ''),
        'measured_v':   float(getattr(v, 'measured_v', 0.0)),
        'rated_max':    float(getattr(v, 'rated_max', 0.0)),
        'worst_time':   float(getattr(v, 'worst_time', 0.0)),
    }


def _spec_from_dict(d: dict) -> 'Spec':
    """Rebuild a (detached) Spec from ``SpecOutcome.as_dict()``'s ``'spec'``
    sub-dict. ``module`` is not part of that serialization (it is a back-
    reference to a live ``Module``, not data), so the reconstructed Spec's
    ``module`` is ``None`` — fine for report rendering/diffing, which only
    reads ``name``/``metric``/``op``/``target``/``units``/``description``.
    """
    from silk.specs import Spec
    return Spec(
        name=d['name'],
        key=d.get('key', d['name']),
        module=None,
        metric=d['metric'],
        op=d['op'],
        target=d['target'],
        units=d.get('units', ''),
        description=d.get('description', ''),
    )


def _measurement_from_dict(d: Optional[dict]) -> Optional['Measurement']:
    if d is None:
        return None
    from silk.specs import Measurement
    from silk.specs.aggregate import decode_aggregate
    from silk.simulation.conditions import decode_condition
    return Measurement(
        source=d['source'],
        value=d['value'],
        passes=d.get('passes'),
        corner=decode_condition('corner', d.get('corner')),
        timestamp=d.get('timestamp', 0.0),
        sim_run_id=d.get('sim_run_id'),
        notes=d.get('notes', ''),
        temperature=decode_condition('temperature', d.get('temperature')),
        voltage=decode_condition('supply', d.get('voltage')),
        sim_type=d.get('sim_type'),
        tool_name=d.get('tool_name'),
        origin_version_id=d.get('origin_version_id'),
        aggregate=decode_aggregate(d.get('aggregate')),
    )


def _spec_outcome_from_dict(d: dict) -> 'SpecOutcome':
    from silk.specs import SpecOutcome, SpecAttempt
    from silk.specs.aggregate import validate_attribution
    spec = _spec_from_dict(d['spec'])
    try:
        measurement = _measurement_from_dict(d.get('measurement'))
        latest_attempt = SpecAttempt.from_dict(d['latest_attempt']) if d.get('latest_attempt') else None
        last_successful_attempt = SpecAttempt.from_dict(d['last_successful_attempt']) if d.get('last_successful_attempt') else None
        for attempt in (latest_attempt, last_successful_attempt):
            if attempt is not None and attempt.aggregate is not None:
                try:
                    validate_attribution(attempt.aggregate, spec.metric,
                                         attempt.aggregate.campaign_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'Attempt {attempt.attempt_id!r}: {exc}') from exc
        evidence = [measurement]
        evidence.extend(attempt.measurement for attempt in
                        (latest_attempt, last_successful_attempt) if attempt is not None)
        for item in evidence:
            if item is not None:
                validate_attribution(item.aggregate, spec.metric, item.sim_run_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Spec outcome {spec.name!r} (owner {d.get("owner_path")!r}): {exc}') from exc
    return SpecOutcome(
        spec=spec,
        latest_attempt=latest_attempt,
        last_successful_attempt=last_successful_attempt,
        owner_path=d.get('owner_path'),
        report_key=d.get('report_key'),
        measurement=measurement,
        status=d['status'],
        margin=d.get('margin'),
        source_used=d.get('source_used'),
        calibrated=d.get('calibrated', False),
        notes=d.get('notes', ''),
    )


class _ModuleNameStub:
    """Minimal module-name carrier for a round-tripped SpecReconciliationReport.

    ``SpecReconciliationReport.module`` is a back-reference to a live
    ``Module`` (used only via ``getattr(module, 'module_name', None)`` in
    ``as_dict``/rendering). A loaded report has no live Module, so this
    stub carries just enough — the name that was already serialized — to
    keep that attribute access working after ``load_json``.
    """
    def __init__(self, module_name: Optional[str]):
        self.module_name = module_name


def _spec_reconciliation_from_dict(d: dict) -> 'SpecReconciliationReport':
    """Rebuild a SpecReconciliationReport from ``SpecReconciliationReport.as_dict()``.

    ``save_json`` always writes a (possibly empty) dict for this section
    — never ``null`` — so this always returns an object, never ``None``.
    That is behaviorally equivalent to the in-memory ``None`` case for
    every consumer (``report_diff._diff_spec_reconciliation`` and the
    HTML/Markdown/terminal renderers all guard on ``not outcomes``, which
    an empty-outcomes reconstruction also satisfies).
    """
    from silk.specs import SpecReconciliationReport
    module_name = d.get('module_name')
    return SpecReconciliationReport(
        module=_ModuleNameStub(module_name) if module_name is not None else None,
        outcomes=[_spec_outcome_from_dict(o) for o in d.get('outcomes', [])],
        calibrations_applied=list(d.get('calibrations_applied', [])),
        timestamp=d.get('timestamp', 0.0),
    )


def _layout_from_dict(d: Optional[dict]) -> Optional[Any]:
    """Rebuild a LayoutEstimate from the ``'layout'`` section ``save_json``
    writes. ``equivalence_classes`` is not part of that serialization (it
    is derived matching-run data, not needed by any report/diff consumer —
    ``report_diff._diff_layout`` reads only ``nets``/``bounding_box``, and
    the report renderers read only ``nets``/``devices``/``warnings``/
    ``bounding_box``/``routing_model_id``) so it round-trips as ``{}``.
    """
    if d is None:
        return None
    from silk.layout.estimator import LayoutEstimate, NetLayout, DeviceLayout
    from silk.layout._types import LayoutWarning
    nets = {
        name: NetLayout(
            length=nl['length'],
            width=nl['width'],
            r=nl['r'],
            c=nl['c'],
            equivalence_class=nl.get('equivalence_class'),
        )
        for name, nl in d.get('nets', {}).items()
    }
    devices = {
        name: DeviceLayout(area=dl['area'])
        for name, dl in d.get('devices', {}).items()
    }
    warnings = tuple(
        LayoutWarning(
            net_name=w.get('net_name'),
            reason=w['reason'],
            severity=w['severity'],
            detail=w['detail'],
        )
        for w in d.get('warnings', [])
    )
    return LayoutEstimate(
        computation_mode=d['computation_mode'],
        routing_model_id=d['routing_model_id'],
        timestamp=d['timestamp'],
        bounding_box=tuple(d['bounding_box']),
        nets=nets,
        devices=devices,
        equivalence_classes={},
        warnings=warnings,
        # Heuristic provenance; absent from files written before it existed.
        packing_factor=d.get('packing_factor'),
        routing_overhead=d.get('routing_overhead'),
    )


def _junction_violation_from_dict(d: dict, path: str = 'junction_violations') -> 'JunctionViolation':
    from silk.lint.violations import JunctionViolation

    location = _report_location_from_dict(d.get('location'), 'terminal', f'{path}.location')
    sev = d['severity']
    if isinstance(sev, str):
        sev = Severity.from_string(sev)
    else:
        sev = Severity(int(sev))
    return JunctionViolation(
        rule_id=d['rule_id'],
        severity=sev,
        message=d['message'],
        location=location,
        context=d.get('context', {}),
        device_type=d.get('device_type', ''),
        measured_v=d.get('measured_v', 0.0),
        rated_max=d.get('rated_max', 0.0),
        worst_time=d.get('worst_time', 0.0),
    )


# ---------------------------------------------------------------------------
# Waveform-view auto-context —
# best-effort, per-field-guarded discovery of run provenance from whatever
# `source` (and, for the spec summary, `self`) happen to expose. Provenance
# is *passed in* to silk.signals.view, never discovered there — the view
# package's import boundary forbids importing Module/spec objects at all
# — so this lives here, where a Module is actually in scope.
# ---------------------------------------------------------------------------

def _safe(fn):
    """Run *fn*, returning its result or ``None`` on any exception —
    a missing attribute omits an auto-populated field rather than raising."""
    try:
        return fn()
    except Exception:
        return None


def _format_one_analysis(analysis) -> str:
    """Human-readable summary of one ``AnalysisBase`` (AC/Transient/DC/Noise
    from :mod:`silk.simulation.analysis`), e.g.
    ``"AC(variation=dec, points=20, fstart=1.0, fstop=1000000000.0)"``.
    Generic over the object's own public attributes rather than a per-type
    formatter, so a new analysis type needs no change here."""
    kind = type(analysis).__name__
    try:
        attrs = vars(analysis)
    except TypeError:
        return kind
    parts = [f'{k}={v}' for k, v in attrs.items() if v is not None and v is not False]
    return f'{kind}({", ".join(parts)})' if parts else kind


def _format_sim_analysis_dict(d) -> Optional[str]:
    """``module.sim_analysis_dict`` (populated by ``add_sim_type()``) can
    hold more than one registered analysis; join them so a single 'analysis'
    context field covers a testbench that runs, say, both an operating-point
    and a transient."""
    if not d:
        return None
    parts = [p for p in (_format_one_analysis(a) for a in d.values()) if p]
    return '; '.join(parts) if parts else None


def _extract_pdk_name(module_like) -> Optional[str]:
    """Best-effort *friendly* PDK name (e.g. ``"sky130"``), read from the
    PDK details object's own ``pdk_name``/``name`` first. Verified against a
    real ``load_pdk('sky130')``-backed ``Module``: ``module.pdk_details.
    pdk_name == 'sky130'``, while ``module.pdk_name`` is a different thing
    entirely — the netlist *class* name (``"PDK_skywater130"``, the first
    positional argument to ``Module()``) — so that is only the fallback, not
    the first choice, despite the more obvious-looking attribute name.
    Deliberately distinct from this module's ``_detect_pdk_name`` (used by
    ``ReportMetadata``, which always returns the sentinel string
    ``"unknown"``): a context field must be *omitted* on a miss, not filled
    with a fabricated-looking placeholder, so this returns ``None`` instead.
    """
    pdk = getattr(module_like, 'pdk_details', None) or getattr(module_like, 'pdk', None)
    if pdk is not None:
        for attr in ('pdk_name', 'name'):
            val = getattr(pdk, attr, None)
            if isinstance(val, str) and val:
                return val
    name = getattr(module_like, 'pdk_name', None)
    if isinstance(name, str) and name:
        return name
    return None


def _build_auto_context_fields(source, module_name) -> List[dict]:
    """Best-effort provenance fields for ``add_waveform_view``, attempted
    in this order: module name, test name, analysis + its parameters, corner, PDK name. (The spec pass/fail summary and the silk
    version are appended by the caller, which alone has ``self`` — the
    report's own ``spec_reconciliation`` — in scope.) Each field is
    independently guarded via :func:`_safe`: a missing attribute on
    *source* just omits that field.
    """
    from silk.simulation.conditions import ConditionState
    fields = []

    def add(label, value):
        if value is not None and value != '':
            fields.append({'label': label, 'value': value})

    add('module', module_name)
    add('test', _safe(lambda: getattr(source, 'test_name', None)))
    add('analysis', _safe(lambda: _format_sim_analysis_dict(
        getattr(source, 'sim_analysis_dict', None))))
    corner = _safe(lambda: getattr(source, '_last_sim_corner', None))
    # none is a fact worth printing; unknown (None) stays absent.
    add('corner', 'none' if corner is ConditionState.NONE else corner)
    add('PDK', _safe(lambda: _extract_pdk_name(source)))
    return fields


def _synthesize_context_title(fields) -> Optional[str]:
    """A page title from the provenance fields: ``'<module> — <ANALYSIS>'``.

    Falls back to the module alone when no analysis resolved, and to ``None``
    when neither did (in which case the caller leaves the title unset and
    :func:`emit_view`'s own fallback chain applies). The analysis field's value
    looks like ``'AC(variation=dec, points=20, …)'``; only the leading type name
    is used, since the parameters are already listed in the popover.
    """
    if not fields:
        return None
    by_label = {f.get('label'): f.get('value') for f in fields if isinstance(f, dict)}
    module = by_label.get('module')
    analysis = by_label.get('analysis') or ''
    kind = analysis.split('(', 1)[0].strip() if analysis else ''
    if module and kind:
        return f'{module} — {kind}'
    return module or None


def _spec_summary_context_field(recon: 'SpecReconciliationReport') -> dict:
    """One context field summarizing spec reconciliation pass/fail/unmeasured
    counts, severity-colored so it reads as a status at a glance in the
    page's info popover."""
    n_unmeasured = len(recon.unmeasured)
    parts = [f'{recon.pass_count} PASS', f'{recon.fail_count} FAIL']
    if n_unmeasured:
        parts.append(f'{n_unmeasured} UNMEASURED')
    severity = 'error' if recon.fail_count else ('warn' if n_unmeasured else 'info')
    return {'label': 'specs', 'value': ' / '.join(parts), 'severity': severity}


def _merge_context(auto_context, caller_context):
    """Merge an auto-populated context with a caller-supplied one, field by
    field, caller wins. ``title``/``subtitle`` are never auto-populated (see
    ``add_waveform_view``'s docstring), so they simply come from whichever
    side supplies them, caller preferred over auto (there is none) — in
    practice, from the caller only.

    Neither side is validated here: the merged dict is handed straight to
    ``emit_view(context=...)``, where context validation rules are enforced.
    A malformed caller ``context`` still raises :class:`~silk.signals.view.ViewError` there.
    """
    if not caller_context:
        return auto_context
    if not auto_context:
        return caller_context
    merged_fields = list(auto_context.get('fields') or [])
    index_by_label = {
        f.get('label'): i for i, f in enumerate(merged_fields) if isinstance(f, dict)
    }
    for cf in caller_context.get('fields') or []:
        label = cf.get('label') if isinstance(cf, dict) else None
        if label is not None and label in index_by_label:
            merged_fields[index_by_label[label]] = cf
        else:
            merged_fields.append(cf)
            if label is not None:
                index_by_label[label] = len(merged_fields) - 1
    merged = dict(auto_context)
    merged.update({k: v for k, v in caller_context.items() if k != 'fields'})
    if merged_fields:
        merged['fields'] = merged_fields
    return merged


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _monte_carlo_from_list(items):
    from silk.circuit.monte_carlo_report import MonteCarloSection
    return [MonteCarloSection.from_dict(d) for d in (items or [])]


@dataclass
class CircuitReport:
    """Aggregated circuit summary report.

    Built by :func:`generate_report`; print it with :meth:`print_summary`,
    write it with :meth:`save`, :meth:`save_html` or :meth:`save_json`, and
    compare two with :func:`silk.circuit.diff_reports`.

    Parameters
    ----------
    module_name : str
        Name of the reported module.
    generated_at : str
        ISO timestamp (seconds resolution, local time).
    topology : TopologySummary
        Subcircuit hierarchy summary.
    annotation : AnnotationCoverage
        Port-annotation coverage.
    violations : list of InterfaceViolation
        Interface-check findings (rule ids ``INT-*``) left after waivers and
        scope exclusions.
    simulation : SimulationSummary or None
        Statistics of the module's loaded results; ``None`` if no simulation
        data is loaded.
    domain_violations : list of InterfaceViolation
        Voltage-domain findings (``DOM-*``).
    junction_violations : list of JunctionViolation
        Junction-stress findings (``STR-*``).
    figures : list of tuple
        ``(matplotlib Figure, title)`` pairs from :meth:`add_figure`; held in
        memory only, excluded from ``repr``, comparison and JSON.
    waveform_views : list of tuple of str
        ``(absolute html path, label)`` pairs from :meth:`add_waveform_view`;
        excluded from ``repr`` and comparison.
    composition_mermaid : str or None
        Mermaid text from :meth:`add_composition_diagram`.
    composition_module_name : str or None
        Name of the module the diagram shows.
    metadata : ReportMetadata or None
        silk and PDK versions the report was built with.
    lint : LintReport or None
        Full lint result; not written by :meth:`save_json`.
    coverage : CoverageReport or None
        Lint-rule coverage; not written by :meth:`save_json`.
    spec_reconciliation : SpecReconciliationReport or None
        Spec outcomes from ``module.reconcile_specs()``.
    layout_estimate : LayoutEstimate or None
        Layout estimate passed to :func:`generate_report`.
    op_tables : list of OpTableSection
        Operating-point tables, one per SpecReport that captured them.
    provenance : list of RunProvenance
        Distinct provenance records of the SpecReports, in caller order.
    metric_warnings : list of MetricWarningSection
        Metric warnings the SpecReports recorded.
    monte_carlo : list
        Sections from :meth:`add_monte_carlo`.
    stress_coverage : dict
        Serialized junction-stress coverage (STR-001); empty when none.
    test_results : object or None
        Complete evidence passed to :func:`generate_report` (including
        failures); ``None`` means no evidence was supplied.
    """
    module_name: str                        #: Name of the reported module.
    generated_at: str                       #: ISO timestamp string (local time, seconds).
    topology: TopologySummary               #: Subcircuit hierarchy summary.
    annotation: AnnotationCoverage          #: Port-annotation coverage.
    violations: List[InterfaceViolation]    #: Interface (``INT-*``) findings after waivers and exclusions.
    simulation: Optional[SimulationSummary]   #: Loaded-results statistics; ``None`` when no data is loaded.
    domain_violations: List['InterfaceViolation'] = field(default_factory=list)   #: Voltage-domain (``DOM-*``) findings.
    junction_violations: List['JunctionViolation'] = field(default_factory=list)   #: Junction-stress (``STR-*``) findings.
    #: ``(matplotlib Figure, title)`` pairs from :meth:`add_figure`; in memory
    #: only, never serialized.
    figures: List[Tuple[Any, str]] = field(default_factory=list, repr=False, compare=False)

    # Waveform explorer links — populated by add_waveform_view(). Each entry
    # is (absolute html path, label); repr/compare-excluded like figures so
    # adding a view never perturbs report equality or golden reprs.
    #: ``(absolute html path, label)`` per view from :meth:`add_waveform_view`.
    waveform_views: List[Tuple[str, str]] = field(default_factory=list, repr=False, compare=False)

    # Composition diagram — populated by add_composition_diagram(). Mermaid
    # text plus the source module's name; repr/compare-excluded like figures so
    # adding a diagram never perturbs report equality or golden reprs.
    #: Mermaid text from :meth:`add_composition_diagram`; ``None`` until added.
    composition_mermaid: Optional[str] = field(default=None, repr=False, compare=False)
    #: Name of the module the composition diagram shows; ``None`` until added.
    composition_module_name: Optional[str] = field(default=None, repr=False, compare=False)

    # Optional embedded reports — populated by generate_report().
    # Direct constructors (for testing) can leave these as None.
    metadata: Optional[ReportMetadata] = None   #: silk and PDK versions; ``None`` on a hand-built report.
    lint: Optional[LintReport] = None           #: Full lint result; not part of the JSON file, ``None`` after :meth:`load_json`.
    coverage: Optional[CoverageReport] = None   #: Lint-rule coverage; not part of the JSON file, ``None`` after :meth:`load_json`.
    spec_reconciliation: Optional['SpecReconciliationReport'] = None   #: Spec outcomes from ``module.reconcile_specs()``.
    layout_estimate: Optional[Any] = None   #: ``LayoutEstimate`` passed to :func:`generate_report`; ``None`` when absent.

    # Operating-point tables (TASK-330.6) — populated by generate_report()
    # from its spec_reports= kwarg via _build_op_tables(). Empty by default:
    # a report built with no spec_reports= (or none of them captured device
    # params) renders no Operating Point Table section at all.
    #: One :class:`~silk.circuit.report.OpTableSection` per SpecReport that captured device
    #: parameters; empty renders no section.
    op_tables: List[OpTableSection] = field(default_factory=list)

    # Provenance records (TASK-306) — silk.simulation.provenance.RunProvenance, one per
    # distinct record among generate_report()'s spec_reports= (deduped by
    # digest). Empty when none carries one, and on a directly constructed or
    # pre-TASK-306 report; an empty list renders no block at all.
    #: ``RunProvenance`` records, deduplicated by digest, in caller order;
    #: empty renders no block.
    provenance: List[Any] = field(default_factory=list)

    # MetricWarnings recorded by generate_report()'s spec_reports= (TASK-421).
    # Empty when none warned, and on a report built without spec_reports=;
    # an empty list adds nothing to the JSON, HTML or markdown.
    #: Metric warnings recorded by the SpecReports; empty adds nothing to the
    #: output.
    metric_warnings: List[MetricWarningSection] = field(default_factory=list)

    # Monte Carlo sections (TASK-431) — populated by add_monte_carlo().
    # Empty by default: no section in the HTML, no key in the JSON.
    #: Sections added by :meth:`add_monte_carlo`; empty renders no section and
    #: writes no JSON key.
    monte_carlo: List[Any] = field(default_factory=list, repr=False, compare=False)

    #: Serialized junction-stress coverage (STR-001); ``{}`` when none.
    stress_coverage: dict = field(default_factory=dict)

    # Caller-supplied complete evidence; None means no evidence was supplied.
    #: Complete evidence (including failures) passed to :func:`generate_report`;
    #: ``None`` means none was supplied.
    test_results: Any = None

    @property
    def interface_check_passed(self) -> bool:
        """True when there are no ERROR-severity violations in any checker."""
        all_violations = list(self.violations) + list(self.domain_violations) + list(self.junction_violations)
        return all(_severity_of(v) != Severity.ERROR for v in all_violations)

    def print_summary(self) -> None:
        """Print a condensed one-screen summary to stdout."""
        warn_count  = sum(1 for v in self.violations if _severity_of(v) == Severity.WARNING)
        error_count = sum(1 for v in self.violations if _severity_of(v) == Severity.ERROR)

        iface_str = 'PASS'
        if error_count:
            iface_str = f'FAIL ({error_count} error(s), {warn_count} warning(s))'
        elif warn_count:
            iface_str = f'PASS with {warn_count} warning(s)'

        sim_str = 'not run'
        if self.simulation:
            sim = self.simulation
            _, axis_text = _sim_axis_text(sim)
            sim_str = (
                f'{sim.analysis}, {axis_text}, {sim.n_points} pts, '
                f'{sim.n_signals} signals'
            )

        print(f"Circuit:     {self.module_name}")
        print(f"  Instances: {self.topology.total_instances}"
              f"  |  Nets: {self.topology.total_nets}")
        if self.annotation.total == 0:
            print("  Annotation coverage: No ports to annotate")
        else:
            print(f"  Annotation coverage: {self.annotation.annotated}/{self.annotation.total}"
                  f" ports ({self.annotation.coverage_pct:.1f}%)"
                  f"  |  Unannotated: {self.annotation.unannotated}")
        dom_errors  = sum(1 for v in self.domain_violations if _severity_of(v) == Severity.ERROR)
        dom_warns   = sum(1 for v in self.domain_violations if _severity_of(v) == Severity.WARNING)
        junc_errors = len(self.junction_violations)

        dom_str = 'PASS'
        if dom_errors:
            dom_str = f'FAIL ({dom_errors} error(s), {dom_warns} warning(s))'
        elif dom_warns:
            dom_str = f'PASS with {dom_warns} warning(s)'

        junc_str = 'not run' if not getattr(self, 'simulation', None) else (
            f'FAIL ({junc_errors} violation(s))' if junc_errors else 'PASS'
        )

        print(f"  Interface: {iface_str}")
        print(f"  Domains:    {dom_str}")
        print(f"  Junctions:  {self._stress_summary() or junc_str}")
        print(f"  Simulation: {sim_str}")

        sig_block = self._render_signals_terminal()
        if sig_block:
            print(sig_block)
        grid_block = self._render_grid('terminal')
        if grid_block:
            print(grid_block)
        met_block = self._render_metrics_terminal()
        if met_block:
            print(met_block)

    def _stress_summary(self) -> str:
        return stress_coverage_text(self.stress_coverage)

    def add_figure(self, fig, title: str) -> None:
        """Attach a matplotlib Figure to the report for HTML embedding.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Stored in memory only; excluded from JSON serialization.
        title : str
            Caption shown above the figure in the HTML report.
        """
        self.figures.append((fig, title))

    def add_monte_carlo(self, mc, *, metrics=None, top: int = 3) -> None:
        """Attach a Monte Carlo section built from ``mc`` (MonteCarloResults).

        The section holds the yield per spec and corner with its 95% Wilson
        interval and samples run / failed / used, one histogram per metric
        with its spec limits, and the failed samples with their seeds. When
        ``mc.draws`` was recorded (``record_draws=True``) it adds the
        sensitivity table per metric and corner in share order and scatter
        plots of the metric against its ``top`` parameters; without draws
        there is no sensitivity section. ``metrics`` defaults to the
        metrics some spec checks. ``save_json`` writes the same numbers
        under ``'monte_carlo'`` without the images.
        """
        from silk.circuit.monte_carlo_report import build_section
        self.monte_carlo.append(build_section(mc, metrics=metrics, top=top))

    def add_composition_diagram(self, module, depth: int = 1) -> None:
        """Attach a composition block diagram of *module* to the report.

        Renders the module one level deep (by default) into Mermaid text via
        ``silk.circuit.composition`` and stores it so ``save()`` emits a
        fenced ```mermaid block and ``save_html()`` emits a
        ``<pre class="mermaid">`` block. Pure — reads the module's existing
        connectivity and any recorded operating points / spec outcomes; mutates
        nothing.

        Parameters
        ----------
        module : Module
            The module to diagram (typically the report's own top module).
        depth : int, default 1
            Levels of hierarchy to draw as boxes; deeper structure is
            summarized as a device count. See
            ``silk.circuit.composition.module_graph``.
        """
        from silk.circuit.composition import module_graph, graph_to_mermaid
        graph = module_graph(module, depth=depth)
        self.composition_mermaid = graph_to_mermaid(graph)
        self.composition_module_name = graph.get('name')

    def add_waveform_view(self, source, *, out_path: str = None, label: str = None,
                          annotations: dict = None, context: dict = None,
                          title: str = None, inline: bool = False) -> None:
        """Emit an interactive waveform explorer and link it from this report.

        Parameters
        ----------
        source : str or os.PathLike or object
            A rawfile path, or any object carrying ``.results_file`` (a
            simulated Module/testbench) — duck-typed, never imported, so this
            method never pulls in ``silk.circuit.modules``.
        out_path : str, optional
            Destination ``.html`` path. Defaults to
            ``<rawfile dir>/<module_name>_view.html`` (module name from
            ``source.module_name`` when available, else the rawfile's own
            basename).
        label : str, optional
            Link text in the rendered report. Defaults to the module name,
            else the output file's basename.
        context : dict, optional
            Extra provenance fields, merged
            field-by-field over the auto-populated context below —
            *context*'s fields win on a matching ``label``, and any others
            are appended. Passing ``context={'title': ..., 'subtitle':
            ...}`` sets the page's displayed title/subtitle, which are never
            auto-populated (so an explicit *title* below keeps working the
            way it always has: auto-population never overrides it — see
            Notes).
        annotations, title, inline
            Passed through to ``silk.signals.view.emit_view``.

        Notes
        -----
        Imports ``silk.signals.view`` lazily, so a report that never calls
        this pulls in none of it.

        Best-effort, per-field-guarded auto-population:
        attempts, in order, the module name, test name (``source.test_name``
        when *source* is testbench-like), registered analysis/parameters
        (``source.sim_analysis_dict``), the resolved corner
        (``source._last_sim_corner``, set by ``Module.simulate()``), the PDK
        name, this report's spec pass/fail/unmeasured summary (from
        ``self.spec_reconciliation``, when populated), and the silk version.
        A missing attribute just omits that field — this method never raises
        because provenance could not be discovered. Only ``fields`` are
        auto-populated, never ``title``/``subtitle``: this keeps this
        method's own *title* parameter (and the rawfile-title/basename
        fallback) in control of the displayed title, matching the
        precedence rule in ``silk.signals.view.emit_view``
        (``context['title']`` beats ``title=``) — an auto-derived title
        would otherwise silently outrank an explicit one.
        """
        from silk.signals.view import emit_view, ViewError

        module_name = None
        rawfile_source = source
        results_file = getattr(source, 'results_file', None)
        if results_file is not None:
            rawfile_source = results_file
            module_name = getattr(source, 'module_name', None) or None
        elif not isinstance(source, (str, os.PathLike)):
            # source is neither a path itself nor an object with a usable
            # (non-None) .results_file — e.g. a testbench that hasn't been
            # run() yet, whose .results_file is legitimately None. Without
            # this, rawfile_source stays `source` and os.fspath() below
            # raises a raw, unhelpful TypeError (2026-08-25 review, B1).
            raise ViewError(
                f"add_waveform_view(): {type(source).__name__} has no rawfile "
                "to show — its .results_file is None or missing. Pass a "
                "rawfile path directly, or a simulated Module/testbench "
                "(call tb.run() / module.simulate() first)."
            )

        if out_path is None:
            rawfile_str = os.fspath(rawfile_source)
            rawfile_dir = os.path.dirname(os.path.abspath(rawfile_str)) or '.'
            base = module_name or os.path.splitext(os.path.basename(rawfile_str))[0]
            out_path = os.path.join(rawfile_dir, f'{base}_view.html')

        # The report always knows its own module; the *source* object usually
        # does not (ModuleTestbench proxies no such attribute, and the common
        # top-level-testbench pattern sets none), which would drop the single
        # most useful provenance field. Fall back to it for the context only —
        # never for out_path above, which would silently rename existing
        # default outputs. 'unnamed' is the report builder's own placeholder,
        # and a placeholder must be omitted rather than displayed.
        context_module = module_name
        if not context_module:
            own = _safe(lambda: self.module_name)
            if own and own != 'unnamed':
                context_module = own

        auto_fields = _safe(lambda: _build_auto_context_fields(source, context_module)) or []
        recon = self.spec_reconciliation
        if recon is not None and _safe(lambda: bool(recon.outcomes)):
            spec_field = _safe(lambda: _spec_summary_context_field(recon))
            if spec_field is not None:
                auto_fields.append(spec_field)
        silk_version = _safe(_detect_silk_version)
        if silk_version and silk_version != 'unknown':
            auto_fields.append({'label': 'silk', 'value': silk_version})
        auto_context = {'fields': auto_fields} if auto_fields else None
        final_context = _merge_context(auto_context, context)

        # Without this, a report-emitted page is titled from the rawfile's own
        # `Title:` line — which for a silk netlist is the boilerplate comment
        # '*** automated netlist generator', i.e. no information at all. Only
        # synthesize when nothing else supplies a title, so an explicit
        # `title=` or `context['title']` is never clobbered.
        if final_context is not None and not final_context.get('title') and title is None:
            synthesized = _safe(lambda: _synthesize_context_title(final_context.get('fields')))
            if synthesized:
                final_context = dict(final_context, title=synthesized)

        nodes = None
        if results_file is not None:
            from silk.simulation.generation import read_retained
            from silk.circuit.view_nodes import _view_node_context
            rawfile_source = read_retained(source)
            # Generic path-carrying objects still work without circuit proof.
            from silk.circuit.modules import Module
            from silk.circuit.module_test import ModuleTestbench
            from silk.circuit.test_placement import SelectedTestbench
            if isinstance(source, (Module, ModuleTestbench, SelectedTestbench)):
                nodes = _view_node_context(source, rawfile_source)
        result = emit_view(rawfile_source, out_path, title=title,
                           annotations=annotations, context=final_context,
                           inline=inline, labels='designer',
                           **({'node_context': nodes} if nodes else {}))

        label_final = (
            label or module_name
            or os.path.splitext(os.path.basename(result.html_path))[0]
        )
        self.waveform_views.append((result.html_path, label_final))

    def save_json(self, path: str) -> None:
        """Persist the report to a JSON file for cross-session diffing.

        The ``figures`` list is excluded — matplotlib Figure objects are not
        JSON-serializable. ``lint``/``coverage`` are also excluded (see
        ``omitted_sections`` in the written file); everything else —
        including ``metadata``, ``spec_reconciliation``, and
        ``layout_estimate`` — round-trips through ``CircuitReport.load_json()``.

        Version 3 embeds complete test evidence and tags nonfinite/complex
        numbers once across the whole tree. Older readers refuse version 3.
        Version 5 tags all five location types with their complete fields;
        readers supporting only versions 1–4 refuse version 5.

        Parameters
        ----------
        path : str
            File to write (overwritten, UTF-8).

        Raises
        ------
        TypeError
            If ``test_results`` holds an unsupported type, or the report
            holds a value that has no JSON form or an unsupported location type.
        ValueError
            If a value cannot be encoded (for example a circular container).
        OSError
            If *path* cannot be written.
        """
        # Intentionally NOT using dataclasses.asdict(self) — figures contains
        # non-dataclass, non-serializable objects (matplotlib Figure).
        from dataclasses import asdict
        from silk.circuit._results_json import results_data
        from silk.circuit._spec_report_json import plain_data
        from silk.simulation.provenance import _report_field_names
        d = {
            'schema_version':    REPORT_JSON_SCHEMA_VERSION,
            'omitted_sections':  list(_OMITTED_JSON_SECTIONS),
            'module_name':       self.module_name,
            'generated_at':      self.generated_at,
            'topology':          asdict(self.topology),
            'annotation':        asdict(self.annotation),
            'violations':        [_violation_to_dict(v) for v in self.violations],
            'simulation':        asdict(self.simulation) if self.simulation else None,
            'domain_violations': [_violation_to_dict(v) for v in self.domain_violations],
            **({'stress_coverage': self.stress_coverage} if self.stress_coverage else {}),
            'junction_violations': [_junction_violation_to_dict(v) for v in self.junction_violations],
            'metadata':          self.metadata.to_dict() if self.metadata is not None else None,
            'provenance':        [{name: getattr(p, name)
                                   for name in _report_field_names(p)} for p in self.provenance],
            'test_results':      (None if self.test_results is None
                                  else results_data(self.test_results)),
        }
        if self.metric_warnings:
            d['metric_warnings'] = [m.to_dict() for m in self.metric_warnings]
        if self.monte_carlo:
            d['monte_carlo'] = [m.to_dict() for m in self.monte_carlo]
        d['spec_reconciliation'] = (
            self.spec_reconciliation.as_dict()
            if self.spec_reconciliation is not None
            else {
                'module_name': None,
                'timestamp': 0.0,
                'outcomes': [],
                'calibrations_applied': [],
            }
        )
        le = self.layout_estimate
        if le is not None:
            d['layout'] = {
                'computation_mode': le.computation_mode,
                'routing_model_id': le.routing_model_id,
                'timestamp': le.timestamp,
                'bounding_box': list(le.bounding_box),
                'packing_factor': le.packing_factor,
                'routing_overhead': le.routing_overhead,
                'nets': {
                    name: {
                        'length': nl.length,
                        'width':  nl.width,
                        'r':      nl.r,
                        'c':      nl.c,
                        'equivalence_class': nl.equivalence_class,
                    }
                    for name, nl in le.nets.items()
                },
                'devices': {
                    name: {'area': dl.area}
                    for name, dl in le.devices.items()
                },
                'warnings': [
                    {
                        'net_name': w.net_name,
                        'reason':   w.reason,
                        'severity': w.severity,
                        'detail':   w.detail,
                    }
                    for w in le.warnings
                ],
            }
        else:
            d['layout'] = None
        text = json.dumps(encode_json(plain_data(d), complex_values=True),
                          allow_nan=False, indent=2)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    @classmethod
    def load_json(cls, path: str) -> 'CircuitReport':
        """Reconstruct a CircuitReport from a JSON file saved by ``save_json``.

        Restores ``metadata``, ``spec_reconciliation``, and
        ``layout_estimate`` alongside the fields ``save_json`` has always
        written. ``lint``/``coverage`` are not part of the on-disk format
        (see ``omitted_sections`` — its absence in an older file is
        tolerated) and stay ``None`` after loading.

        Files with no top-level ``'schema_version'`` key (written before
        versioning) are read as version 1. A file whose
        ``'schema_version'`` is newer than this silk's
        ``REPORT_JSON_SCHEMA_VERSION`` raises ``ValueError`` naming both
        versions, rather than failing later with a confusing ``TypeError``
        from mismatched fields.

        Parameters
        ----------
        path : str
            Report JSON file written by :meth:`save_json`.

        Returns
        -------
        CircuitReport
            The reconstructed report; ``lint`` and ``coverage`` are ``None``
            and ``figures`` is empty.

        Raises
        ------
        ValueError
            If the file is not valid JSON or not a JSON object, its
            ``schema_version`` is not a positive integer or is newer than
            ``REPORT_JSON_SCHEMA_VERSION``, a required field is missing, or a
            violation location is malformed or cannot be restored.
        OSError
            If *path* cannot be opened.
        """
        with open(path, encoding='utf-8') as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in report file '{path}': {exc}") from exc

        if not isinstance(d, dict):
            raise ValueError(f"Report file '{path}': expected a JSON object")
        file_version = d.get('schema_version')
        if file_version is not None and (type(file_version) is not int or file_version < 1):
            raise ValueError(f"Report file '{path}': schema_version must be a positive integer")
        if file_version is not None and file_version > REPORT_JSON_SCHEMA_VERSION:
            raise ValueError(
                f"Report file '{path}' has schema_version {file_version}, "
                f"but this silk version only supports report schema up to "
                f"version {REPORT_JSON_SCHEMA_VERSION}"
            )

        from silk.circuit._results_json import restore_results
        from silk.circuit._spec_report_json import plain_data
        if file_version is not None and file_version >= 3:
            try:
                d = decode_json(plain_data(d, wire=True), complex_values=True)
            except ValueError as exc:
                raise ValueError(f"Report file '{path}': {exc}") from exc
        elif file_version is not None and file_version >= 2:
            d = decode_json(d)

        try:
            return cls(
                test_results=(restore_results(d['test_results'], '$.test_results')
                              if file_version is not None and file_version >= 3 and d.get('test_results') is not None
                              else None),
                module_name=d['module_name'],
                generated_at=d['generated_at'],
                topology=_topology_from_dict(d['topology']),
                annotation=_annotation_from_dict(d['annotation']),
                violations=[
                    _violation_from_dict(v, f'violations[{i}]')
                    for i, v in enumerate(d['violations'])
                ],
                simulation=_simulation_from_dict(d['simulation']) if d['simulation'] else None,
                domain_violations=[
                    _violation_from_dict(v, f'domain_violations[{i}]')
                    for i, v in enumerate(d.get('domain_violations', []))
                ],
                stress_coverage=d.get('stress_coverage', {}),
                junction_violations=[
                    _junction_violation_from_dict(v, f'junction_violations[{i}]')
                    for i, v in enumerate(d.get('junction_violations', []))
                ],
                metadata=(
                    ReportMetadata.from_dict(d['metadata'])
                    if d.get('metadata') is not None else None
                ),
                spec_reconciliation=(
                    _spec_reconciliation_from_dict(d['spec_reconciliation'])
                    if d.get('spec_reconciliation') is not None else None
                ),
                layout_estimate=_layout_from_dict(d.get('layout')),
                provenance=_provenance_from_list(d.get('provenance')),
                metric_warnings=[MetricWarningSection.from_dict(m)
                                 for m in d.get('metric_warnings') or []],
                monte_carlo=_monte_carlo_from_list(d.get('monte_carlo')),
            )
        except KeyError as exc:
            raise ValueError(
                f"Report file '{path}' is missing required field {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"Report file '{path}': {exc}") from exc

    def save_html(self, path: str) -> None:
        """Write a self-contained HTML report to *path*.

        All figures attached via ``add_figure()`` are base64-encoded and
        embedded inline — no external image files needed. Waveform-explorer
        links (``add_waveform_view()``) are written relative to *path*'s
        directory when the view sits underneath it, absolute otherwise.
        """
        base_dir = os.path.dirname(os.path.abspath(path))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._to_html(base_dir=base_dir))

    def _to_html(self, base_dir: str = None) -> str:
        def _esc(s: str) -> str:
            import html as _html_mod
            return _html_mod.escape(str(s))

        warn_count  = sum(1 for v in self.violations if _severity_of(v) == Severity.WARNING)
        error_count = sum(1 for v in self.violations if _severity_of(v) == Severity.ERROR)
        if error_count:
            badge_cls  = 'fail'
            badge_txt  = f'FAIL ({error_count} error(s), {warn_count} warning(s))'
        elif warn_count:
            badge_cls  = 'warn'
            badge_txt  = f'PASS with {warn_count} warning(s)'
        else:
            badge_cls  = 'pass'
            badge_txt  = 'PASS'

        ann = self.annotation
        lines = []
        lines.append('<!DOCTYPE html>')
        lines.append('<html>')
        lines.append('<head>')
        lines.append('<meta charset="utf-8">')
        lines.append(f'<title>Circuit Report: {_esc(self.module_name)}</title>')
        lines.append("""<style>
  body { font-family: system-ui, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { border-bottom: 1px solid #ddd; margin-top: 2em; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }
  th, td { border: 1px solid #ddd; padding: 6px 12px; text-align: left; }
  th { background: #f5f5f5; }
  .badge { display: inline-block; padding: 4px 12px; border-radius: 4px; font-weight: bold; }
  .pass { background: #d4edda; color: #155724; }
  .fail { background: #f8d7da; color: #721c24; }
  .warn { background: #fff3cd; color: #856404; }
  tr.spec-pass td.status { background: #d4edda; color: #155724; font-weight: bold; }
  tr.spec-fail td.status { background: #f8d7da; color: #721c24; font-weight: bold; }
  tr.spec-unmeasured td.status { background: #e2e3e5; color: #383d41; font-weight: bold; }
  details summary { cursor: pointer; color: #555; padding: 4px 0; }
  img { max-width: 100%; margin: 1em 0; border: 1px solid #eee; }
  .fig-title { font-weight: bold; margin-top: 1.5em; }
</style>""")
        lines.append('</head>')
        lines.append('<body>')
        lines.append(f'<h1>Circuit Report: {_esc(self.module_name)}</h1>')
        lines.append(f'<p>Generated: {_esc(self.generated_at)}</p>')
        lines.extend(_provenance_html(self.provenance, _esc))

        # Topology
        lines.append('<h2>Circuit Topology</h2>')
        lines.append(f'<p><strong>Module:</strong> {_esc(self.module_name)}<br>')
        lines.append(f'<strong>Subcircuit instances:</strong> {self.topology.total_instances}<br>')
        lines.append(f'<strong>Total nets:</strong> {self.topology.total_nets}</p>')
        if self.topology.instances:
            lines.append('<table><thead><tr><th>Instance</th><th>Subcircuit</th><th>Ports</th></tr></thead><tbody>')
            for inst in self.topology.instances:
                lines.append(
                    f'<tr><td>{_esc(inst.instance)}</td>'
                    f'<td>{_esc(inst.module_name)}</td>'
                    f'<td>{inst.port_count}</td></tr>'
                )
            lines.append('</tbody></table>')

        # Composition diagram
        comp_html = self._render_composition_html()
        if comp_html:
            lines.append(comp_html)

        # Annotation coverage
        lines.append('<h2>Port Annotation Coverage</h2>')
        if ann.total == 0:
            lines.append('<p><strong>No ports to annotate</strong></p>')
        else:
            lines.append(
                f'<p><strong>{ann.annotated}/{ann.total} ports annotated '
                f'({ann.coverage_pct:.1f}%)</strong></p>'
            )
        if ann.unannotated_ports:
            lines.append(
                f'<details><summary>{ann.unannotated} unannotated port(s) '
                f'— click to expand</summary>'
            )
            lines.append('<table><thead><tr><th>Instance</th><th>Port</th>'
                         '<th>Direction</th><th>Block</th></tr></thead><tbody>')
            for u in ann.unannotated_ports:
                lines.append(
                    f'<tr><td>{_esc(u.instance)}</td><td>{_esc(u.port)}</td>'
                    f'<td>{_esc(u.direction or "—")}</td>'
                    f'<td>{_esc(u.module_name)}</td></tr>'
                )
            lines.append('</tbody></table></details>')
        elif ann.total:
            lines.append('<p>All ports have a <code>PortRole</code> declared. ✅</p>')

        # Interface check
        lines.append('<h2>Interface Check</h2>')
        lines.append(f'<p><span class="badge {badge_cls}">{_esc(badge_txt)}</span></p>')
        if self.violations:
            lines.append('<table><thead><tr><th>Severity</th><th>Rule</th>'
                         '<th>Net</th><th>Message</th></tr></thead><tbody>')
            for v in self.violations:
                lines.append(
                    f'<tr><td>{_esc(_severity_name(v))}</td><td>{_esc(v.rule_id)}</td>'
                    f'<td>{_esc(getattr(v, "net", ""))}</td><td>{_esc(v.message)}</td></tr>'
                )
            lines.append('</tbody></table>')

        # Domain violations
        if self.domain_violations:
            lines.append('<h2>Voltage Domain Violations</h2>')
            lines.append('<table><thead><tr><th>Severity</th><th>Rule</th><th>Net</th><th>Message</th></tr></thead><tbody>')
            for v in self.domain_violations:
                lines.append(
                    f'<tr><td>{_esc(_severity_name(v))}</td><td>{_esc(v.rule_id)}</td>'
                    f'<td><code>{_esc(getattr(v, "net", ""))}</code></td>'
                    f'<td>{_esc(v.message)}</td></tr>'
                )
            lines.append('</tbody></table>')

        if self._stress_summary():
            lines.append('<h2>Junction Stress Coverage</h2>')
            lines.append(f'<p>{_esc(self._stress_summary())}</p>')

        # Junction violations
        if self.junction_violations:
            lines.append('<h2>Junction Overrange Violations</h2>')
            lines.append('<table><thead><tr><th>Instance</th><th>Device</th><th>Terminal</th><th>Measured</th><th>Rated Max</th><th>Time</th></tr></thead><tbody>')
            for v in self.junction_violations:
                terminal = getattr(v.location, 'terminal', '') if hasattr(v, 'location') else ''
                lines.append(
                    f'<tr><td>{_esc(v.instance)}</td><td>{_esc(v.device_type)}</td>'
                    f'<td>{_esc(terminal)}</td><td>{v.measured_v:.3f}V</td>'
                    f'<td>{v.rated_max}V</td><td>{v.worst_time*_S_TO_NS:.2f}ns</td></tr>'
                )
            lines.append('</tbody></table>')

        grid_html = self._render_grid('html')
        if grid_html:
            lines.append(grid_html)

        # Spec Reconciliation
        spec_html = self._render_spec_reconciliation_html()
        if spec_html:
            lines.append(spec_html)

        # Simulation
        if self.simulation is not None:
            sim = self.simulation
            axis_label, axis_text = _sim_axis_text(sim)
            header, rows = _sim_table_rows(sim)
            lines.append('<h2>Behavioral Simulation</h2>')
            lines.append(
                f'<p><strong>Analysis:</strong> {_esc(sim.analysis)}<br>'
                f'<strong>{_esc(axis_label)}:</strong> {_esc(axis_text)}<br>'
            )
            lines.append(
                f'<strong>Points:</strong> {_esc(_sim_points_text(sim))}</p>'
            )
            lines.append('<table><thead><tr>'
                         + ''.join(f'<th>{_esc(h)}</th>' for h in header)
                         + '</tr></thead><tbody>')
            for name, *nums in rows:
                lines.append(
                    f'<tr><td>{_esc(name)}</td>'
                    + ''.join(f'<td>{v:.4f}</td>' for v in nums)
                    + '</tr>'
                )
            lines.append('</tbody></table>')

        # Operating Point Tables
        op_html = self._render_op_tables_html()
        if op_html:
            lines.append(op_html)

        for mc_section in self.monte_carlo:
            from silk.circuit.monte_carlo_report import render_html
            lines.append(render_html(mc_section))

        # Figures
        if self.figures:
            lines.append('<h2>Figures</h2>')
            for fig, title in self.figures:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                lines.append(f'<p class="fig-title">{_esc(title)}</p>')
                lines.append(
                    f'<img src="data:image/png;base64,{b64}" alt="{_esc(title)}">'
                )

        # Layout estimate
        layout_html = self._render_layout_html()
        if layout_html:
            lines.append(layout_html)

        # Waveform explorer links
        if self.waveform_views:
            lines.append('<section class="waveform-views">')
            lines.append('<h2>Waveform Explorer</h2>')
            lines.append('<ul>')
            for view_path, view_label in self.waveform_views:
                href = _relativize_href(view_path, base_dir)
                lines.append(f'<li><a href="{_esc(href)}">{_esc(view_label)}</a></li>')
            lines.append('</ul>')
            lines.append('</section>')

        lines.append('</body>')
        lines.append('</html>')
        return '\n'.join(lines)

    def _render_layout_html(self) -> str:
        """HTML twin of the Markdown ``## Layout Estimate`` section."""
        le = self.layout_estimate
        if le is None:
            return ''
        esc = _html.escape
        w, h = le.bounding_box
        out = ['<section class="layout-estimate">', '<h2>Layout Estimate</h2>']
        out.append(
            f'<p><strong>Bounding box:</strong> {w:.2f} × {h:.2f} μm '
            f'(routing model: <code>{esc(str(le.routing_model_id))}</code>)</p>'
        )
        if le.packing_factor is not None or le.routing_overhead is not None:
            heuristics = []
            if le.packing_factor is not None:
                heuristics.append(
                    f'packing factor {le.packing_factor:g} '
                    f'(device area = W·L·M × packing factor)'
                )
            if le.routing_overhead is not None:
                heuristics.append(f'routing overhead {le.routing_overhead:g}')
            out.append(
                f'<p><strong>Area heuristics:</strong> {esc(", ".join(heuristics))}'
                f' — adjustable estimates, not layout truth.</p>'
            )
        if le.nets:
            out.append('<table><thead><tr><th>Net</th><th>Length (μm)</th>'
                       '<th>Width (μm)</th><th>R (Ω)</th><th>C (fF)</th>'
                       '</tr></thead><tbody>')
            for net_name, nl in sorted(le.nets.items()):
                out.append(
                    f'<tr><td>{esc(str(net_name))}</td><td>{nl.length:.2f}</td>'
                    f'<td>{nl.width:.3f}</td><td>{nl.r:.3f}</td>'
                    f'<td>{nl.c:.3f}</td></tr>'
                )
            out.append('</tbody></table>')
        if le.warnings:
            out.append('<h3>Layout Warnings</h3>')
            out.append('<ul>')
            for wo in le.warnings:
                net_tag = (f' (net: <code>{esc(str(wo.net_name))}</code>)'
                           if wo.net_name else '')
                out.append(
                    f'<li><strong>[{esc(str(wo.severity))}] {esc(str(wo.reason))}'
                    f'</strong>{net_tag}: {esc(str(wo.detail))}</li>'
                )
            out.append('</ul>')
        out.append('</section>')
        return '\n'.join(out)

    def _grid_results(self):
        """Collections with declared operating axes, in supplied test order."""
        from silk.circuit.spec_report import CornerResults
        evidence = self.test_results
        values = evidence.items() if type(evidence) is dict else [('', evidence)]
        return [(name, value) for name, value in values
                if isinstance(value, CornerResults) and value._is_grid()]

    def _reconciliation_title(self):
        return 'Spec Reconciliation'

    def _reconciliation_outcomes(self):
        """Uncovered outcomes for display, retaining the recorded reconciliation."""
        outcomes = getattr(self.spec_reconciliation, 'outcomes', ())
        covered = {(declaration['key'], declaration.get('owner_path'))
                   for _, collection in self._grid_results()
                   for declaration in collection._check_declarations}
        return [outcome for outcome in outcomes
                if (outcome.spec.key, outcome.owner_path) not in covered]

    def _render_grid_spec_outcomes(self, outcomes, mode):
        """Separate recorded single-run measurements from unmeasured declarations."""
        measured = [o for o in outcomes if o.status in ('PASS', 'FAIL')]
        unmeasured = [o for o in outcomes if o.status == 'UNMEASURED']
        parts = []
        if measured:
            parts.append(getattr(self, '_reconciliation_' + mode)(measured, grid=True))
        if unmeasured:
            title = 'Specs not measured'
            note = f'These specs are declared on {self.module_name}, but no test checked them.'
            table = _pvt_table(['Spec', 'Op', 'Target'],
                               [[o.display_label, o.spec.op, f'{o.spec.target:g}']
                                for o in unmeasured], 'markdown' if mode == 'md' else mode)
            if mode == 'html':
                parts.append('<section class="spec-reconciliation"><h2>' + title
                             + '</h2><p>' + _html.escape(note) + '</p>' + table + '</section>')
            else:
                parts.append('\n'.join(['', ('## ' if mode == 'md' else '') + title, '',
                                          _pvt_text(note, 'markdown' if mode == 'md' else mode), '', table, '']))
        return '\n'.join(parts)

    def _render_grid(self, mode):
        """Render recorded deterministic and Monte Carlo grids in test order."""
        from silk.circuit.monte_carlo import MonteCarloResults
        from silk.circuit.monte_carlo_report import build_grid_section, render_grid_section
        parts = []
        for name, collection in self._grid_results():
            if isinstance(collection, MonteCarloResults):
                parts.append(render_grid_section(build_grid_section(collection), name=name, mode=mode))
            else:
                parts.append(self._render_pvt_grid(name, collection, mode))
        return '\n'.join(parts)

    def _render_pvt_grid(self, name, collection, mode):
        """Render recorded checks and runs without changing collection evidence."""
        from copy import copy
        from silk.circuit.spec_report import CornerResults
        reports = collection.reports
        if not name:
            name = next((report.test_name for report in reports.values()), 'unnamed')
        heading = f"PVT Grid: {name} — {'PASS' if collection.passed else 'FAIL'}"
        rows = collection.table()
        # Counts remain owned by the result collection; only their presentation changes.
        import re
        counts = re.fullmatch(r'planned=(\d+): (\d+) reports \((\d+) PASS, (\d+) FAIL\), '
                              r'(\d+) failed and produced no report', collection._counts())
        planned, _, passed, failed, absent = counts.groups()
        axes = []
        for field, unit in (('run.corner', ''), ('run.temperature', ' °C'), ('run.supply', ' V')):
            values = list(dict.fromkeys(row[field] for row in rows))
            axes.append(', '.join(_pvt_number(value) for value in values) + unit)
        summary = (f"Summary: {planned} runs ({' × '.join(axes)}): {passed} PASS, {failed} FAIL, "
                   f'{absent} failed and produced no report')
        if collection._plan_summary():
            summary = 'Summary: ' + collection._plan_summary() + f': {passed} PASS, {failed} FAIL, {absent} failed and produced no report'
        metrics = collection._table_metric_names
        extrema = {}
        for metric in metrics:
            try:
                extrema[metric] = [collection.stat(metric, how) for how in ('min', 'max')]
            except (TypeError, ValueError):
                extrema[metric] = [None, None]
        spec_rows = []
        checked_metrics = set()
        for declaration in collection._check_declarations:
            metric = declaration['metric']
            checked_metrics.add(metric)
            key = declaration['key']
            cells = [row.get('check.' + key, 'UNAVAILABLE') for row in rows]
            status = ('FAIL' if 'FAIL' in cells else
                      'PASS' if cells and all(cell == 'PASS' for cell in cells) else 'UNAVAILABLE')
            # A per-check view lets worst() retain its own direction and stable identity,
            # including two checks with opposite limits on the same metric.
            selected = {}
            for point, report in reports.items():
                view = copy(report)
                view._results = [result for result in report.results
                    if (result.spec_key or result.name) == key and
                    (declaration.get('owner_path') is None or
                     result.owner_path == declaration['owner_path'])]
                selected[point] = view
            view = CornerResults(selected, failed=collection.failed,
                                 skip_unavailable=collection._skip_unavailable)
            try:
                worst = view.worst(metric)
            except ValueError:
                worst = None
            value = worst.value if worst is not None and worst.corner is not None else None
            margin = abs(value - declaration['target']) if value is not None else None
            low, high = extrema.get(metric, (None, None))
            spec_rows.append([status, declaration['name'], declaration['op'],
                _pvt_number(declaration['target']), _pvt_number(value),
                _pvt_point(worst.corner) if worst is not None else '—',
                _pvt_number(margin), _pvt_number(low), _pvt_number(high)])
        for metric in metrics:
            if metric not in checked_metrics:
                spec_rows.append(['—', metric, '—', '—', '—', '—', '—',
                                  *(_pvt_number(value) for value in extrema[metric])])
        spec_columns = ['Status', 'Spec', 'Op', 'Target', 'Worst', 'Worst at', 'Margin', 'Min', 'Max']
        variables = collection._variable_names()
        run_columns = ['Corner', 'Temp (°C)', 'Supply (V)', *('run.' + name for name in variables), 'Status', *metrics]
        run_rows = [[_pvt_number(row['run.corner']), _pvt_number(row['run.temperature']),
                     _pvt_number(row['run.supply']),
                     *(repr(row['run.' + name]) for name in variables),
                     'NO REPORT' if row['run.status'] == 'failed' else row['run.status'],
                     *(_pvt_number(row['metric.' + metric], missing='') for metric in metrics)]
                    for row in rows]
        errors = [_pvt_point((row['run.corner'], row['run.temperature'], row['run.supply'])
                             + ((tuple((name, row['run.' + name]) for name in variables),) if variables else ()))
                  + ': ' + (str(row['run.error'] or '').splitlines() or [''])[0]
                  for row in rows if row['run.status'] == 'failed']
        if mode == 'html':
            esc = _html.escape
            parts = ['<section class="pvt-grid"><h2>' + esc(heading) + '</h2>',
                     '<p>' + esc(summary) + '</p>',
                     _pvt_table(spec_columns, spec_rows, mode, specs=True),
                     f'<details><summary>All {planned} runs</summary>',
                     _pvt_table(run_columns, run_rows, mode)]
            parts.extend('<p>' + esc(error) + '</p>' for error in errors)
            parts.extend(['</details>', '</section>'])
        else:
            parts = [('## ' if mode == 'markdown' else '\n') + _pvt_text(heading, mode),
                     '', _pvt_text(summary, mode), '',
                     _pvt_table(spec_columns, spec_rows, mode), '',
                     _pvt_table(run_columns, run_rows, mode), '']
            parts.extend(_pvt_text(error, mode) for error in errors)
        return '\n'.join(parts)

    def _render_spec_reconciliation_html(self) -> str:
        outcomes = self._reconciliation_outcomes()
        if self._grid_results():
            return self._render_grid_spec_outcomes(outcomes, 'html')
        return self._reconciliation_html(outcomes)

    def _reconciliation_html(self, outcomes, *, grid=False):
        if not outcomes:
            return ''

        pass_n = sum(o.status == 'PASS' for o in outcomes)
        fail_n = sum(o.status == 'FAIL' for o in outcomes)
        unm_n  = sum(o.status == 'UNMEASURED' for o in outcomes)

        rows = []
        for o in outcomes:
            s = o.spec
            if o.measurement is None:
                meas = '&mdash;'
                source = '&mdash;'
                margin = '&mdash;'
            else:
                meas = f'{o.measurement.value:g}'
                source = o.measurement.source
                margin = f'{o.margin:g}' if o.margin is not None else '&mdash;'
            if o.attempt_description:
                meas = _html.escape(o.attempt_description)
            status_class = o.status.lower()
            if o.measurement is not None or o.latest_attempt is not None:
                notes = _warning_notes(self.metric_warnings, o,
                                       o.measurement.value if o.measurement else None)
                if notes:
                    meas += ''.join(
                        f'<br><small class="metric-warning">{_html.escape(n)}</small>'
                        for n in notes)
            rows.append(
                f'<tr class="spec-{status_class}">'
                f'<td class="status">{o.status}</td>'
                f'<td>{_html.escape(o.display_label)}</td>'
                f'<td>{s.op}</td>'
                f'<td>{s.target:g}</td>'
                f'<td>{meas}</td>'
                f'<td>{source}</td>'
                f'<td>{s.units or "&mdash;"}</td>'
                f'<td>{margin}</td>'
                f'</tr>'
            )

        disclosure = ''
        if any(o.measurement is not None for o in outcomes):
            disclosure = (
                '<p>Results from the last run. '
                'Rerun the test after changing the circuit or stimulus to update these results.</p>')

        table = (
            '<table class="spec-reconciliation">'
            '<thead><tr>'
            '<th>Status</th><th>Spec</th><th>Op</th><th>Target</th>'
            '<th>Measured</th><th>Source</th><th>Units</th><th>Margin</th>'
            '</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody>'
            '</table>'
        )

        calib = ''
        if self.spec_reconciliation.calibrations_applied:
            items = ''.join(
                f'<li><code>{c}</code></li>'
                for c in self.spec_reconciliation.calibrations_applied
            )
            calib = f'<p><strong>Calibrations applied:</strong></p><ul>{items}</ul>'

        grid_note = ('<p>Specs the PVT grid measures are in its table above.</p>'
                     if grid else '')
        counts = f'{pass_n} PASS, {fail_n} FAIL' + ('' if grid else f', {unm_n} UNMEASURED')
        title = 'Specs measured in a single run' if grid else self._reconciliation_title()
        return (
            '<section class="spec-reconciliation">'
            f'<h2>{_html.escape(title)}</h2>'
            f'{grid_note}'
            f'{disclosure}'
            f'<p><strong>Summary:</strong> {counts}</p>'
            f'{table}'
            f'{calib}'
            '</section>'
        )

    def _render_op_tables_html(self) -> str:
        """HTML for the Operating Point Table sections.

        One section per ``self.op_tables`` entry, in the order given to
        ``generate_report(spec_reports=...)``. Returns ``''`` when
        ``op_tables`` is empty — either ``spec_reports`` was omitted, or
        none of the SpecReports given had captured op-point data — so a
        report with no capture renders no section at all.

        Cell text comes from ``op_table._cell_text``, the single renderer
        shared with the console/Markdown table,
        so a hole ('—') vs. a real reading that happens to be nan ('nan')
        never diverges between surfaces. A hole cell (this row's
        ``sources[col] is None``) additionally gets ``class="op-hole"``
        here, since HTML is the only surface with per-cell styling. The
        region cell gets ``class="op-region-<region>"``; a row that is
        ``saturation_weak`` OR whose ``region`` is ``'cutoff'`` gets
        ``class="op-flag"`` on its ``<tr>``. A
        ``ft`` cell sourced ``'derived_estimate'`` gets ``class="op-
        estimate"`` plus a tooltip, and the section gets a one-line note.

        Every piece of text that can carry an arbitrary caller/device
        string — the section heading (``test_name``/``corner``), the
        column headers, and every cell's rendered text — is HTML-escaped
        the same way every other section in this file escapes text
        (``html.escape``, matching ``_to_html``'s local ``_esc`` and
        ``_render_composition_html``'s ``_html_mod.escape``). The ``ft``
        tooltip text is a fixed constant, not caller data, so it is not
        escaped.
        """
        if not self.op_tables:
            return ''

        import html as _html_mod
        from silk.circuit.op_table import _COLUMNS, _cell_text

        def _esc(s) -> str:
            return _html_mod.escape(str(s))

        _NON_SOURCED_COLS = ('name', 'path', 'model', 'polarity', 'w_um', 'l_um', 'm', 'region')

        parts = []
        for section in self.op_tables:
            heading = f'Operating Point Table — {_esc(section.test_name)}'
            if section.corner:
                heading += f' ({_esc(section.corner)})'
            parts.append(f'<h2>{heading}</h2>')

            table = section.table
            if not table.rows:
                parts.append('<p><em>(no MOS devices)</em></p>')
                continue

            has_estimate = any(
                row.sources.get('ft') == 'derived_estimate' for row in table.rows
            )

            parts.append(
                '<table class="op-table"><thead><tr>'
                + ''.join(f'<th>{_esc(c)}</th>' for c in _COLUMNS)
                + '</tr></thead><tbody>'
            )
            for row in table.rows:
                flagged = row.saturation_weak or row.region == 'cutoff'
                tr_class = ' class="op-flag"' if flagged else ''
                parts.append(f'<tr{tr_class}>')
                for col in _COLUMNS:
                    text = _esc(_cell_text(row, col))
                    classes = []
                    attrs = ''
                    if col == 'region':
                        if row.region:
                            classes.append(f'op-region-{row.region}')
                    elif col not in _NON_SOURCED_COLS and row.sources.get(col) is None:
                        classes.append('op-hole')
                    if col == 'ft' and row.sources.get('ft') == 'derived_estimate':
                        classes.append('op-estimate')
                        attrs = (
                            ' title="Estimated: gm / (2π·cgg) '
                            '— not a directly measured transit frequency"'
                        )
                    class_attr = f' class="{" ".join(classes)}"' if classes else ''
                    parts.append(f'<td{class_attr}{attrs}>{text}</td>')
                parts.append('</tr>')
            parts.append('</tbody></table>')

            if has_estimate:
                parts.append(
                    '<p class="op-note"><em>ft (marked estimated) is '
                    'gm / (2π·cgg), a small-signal estimate, not '
                    'a directly measured transit frequency.</em></p>'
                )
        return '\n'.join(parts)

    def _render_composition_md(self) -> str:
        """Markdown for the composition diagram — a fenced ```mermaid block."""
        if not self.composition_mermaid:
            return ''
        return '\n'.join([
            '## Composition Diagram',
            '',
            '```mermaid',
            self.composition_mermaid,
            '```',
            '',
        ])

    def _render_composition_html(self) -> str:
        """HTML for the composition diagram.

        The self-contained report bundles no Mermaid runtime, so the block
        displays as text here and renders as a diagram only in Mermaid-enabled
        viewers (e.g. GitHub). The source is HTML-escaped, which is exactly what
        Mermaid's ``<pre class="mermaid">`` reader wants (it consumes
        ``textContent``).
        """
        if not self.composition_mermaid:
            return ''
        import html as _html_mod
        src = _html_mod.escape(self.composition_mermaid)
        return (
            '<h2>Composition Diagram</h2>'
            f'<pre class="mermaid">{src}</pre>'
            '<p><em>Renders as a block diagram in Mermaid-enabled viewers '
            '(e.g. GitHub); shown as text otherwise.</em></p>'
        )

    def _render_signals_terminal(self) -> str:
        """Op-point table for the terminal summary.

        Returns '' when no simulation data is present. Filters to the
        top-level module's nets when the back-ref is available so that
        hierarchical ngspice subnodes don't flood the output; falls back
        to all signals otherwise. The membership test is case-folded and
        wrapper-stripped (:func:`_net_match_key`) so it agrees for both
        backends — Xyce's rawfile vectors are upper-cased ('V(IN)' ->
        'IN'), which a bare, case-sensitive comparison against
        ``module.net_dict`` never matched, emptying this table for every
        Xyce report of every analysis kind.
        """
        sim = self.simulation
        if sim is None or not sim.n_signals:
            return ''

        module = getattr(self, '_source_module', None)
        module_nets = ({n.lower() for n in module.net_dict}
                       if module is not None else None)
        _, all_rows = _sim_table_rows(sim)
        rows = [r for r in all_rows
                if module_nets is None or _net_match_key(r[0]) in module_nets]
        if not rows:
            return ''

        name_w = max(len(r[0]) for r in rows)
        # A real .op prints one value per vector — there is no range. Of the
        # swept kinds only a time-domain run has an operating point to show.
        # A DC sweep's rows are statistics over the swept axis, and a
        # small-signal run's are magnitudes of a transfer ratio or a noise
        # density — the heading and the unit column say which, instead of
        # printing them as volts.
        if sim.analysis == _ANALYSIS_OP:
            lines = ['', 'Operating point (.op, module nets):']
            for name, value in sorted(rows, key=lambda r: r[0]):
                lines.append(f'  {name:<{name_w}}  {value:7.3f} V')
            return '\n'.join(lines)
        if sim.magnitudes:
            lines = ['', f'Signal magnitude ({sim.analysis}, module nets):']
            unit = '  '
        elif sim.analysis == _ANALYSIS_TRANSIENT:
            lines = ['', 'Operating point (module nets):']
            unit = ' V'
        else:
            lines = ['', f'Sweep statistics ({sim.analysis}, module nets):']
            unit = ' V'
        for name, lo, hi, mean in sorted(rows, key=lambda r: r[0]):
            lines.append(
                f'  {name:<{name_w}}  {mean:7.3f}{unit}   '
                f'({lo:.3f} – {hi:.3f})'
            )
        return '\n'.join(lines)

    def _render_metrics_terminal(self) -> str:
        """Spec reconciliation table for the terminal summary.

        Returns '' when no specs are declared.
        """
        outcomes = self._reconciliation_outcomes()
        if self._grid_results():
            return self._render_grid_spec_outcomes(outcomes, 'terminal')
        return self._reconciliation_terminal(outcomes)

    def _reconciliation_terminal(self, outcomes, *, grid=False):
        sr = self.spec_reconciliation
        if not outcomes:
            return ''

        pass_n = sum(o.status == 'PASS' for o in outcomes)
        fail_n = sum(o.status == 'FAIL' for o in outcomes)
        unm_n = sum(o.status == 'UNMEASURED' for o in outcomes)
        lines = ['']
        if grid:
            lines.extend(['Specs measured in a single run',
                          'Specs the PVT grid measures are in its table above.',
                          f'{pass_n} PASS, {fail_n} FAIL'])
        else:
            lines.append(f'Metrics: {pass_n} PASS, {fail_n} FAIL, {unm_n} UNMEASURED')
        if any(o.measurement is not None for o in outcomes):
            lines.append(
                'Results from the last run. '
                'Rerun the test after changing the circuit or stimulus to update these results.')
        name_w = max(len(o.display_label) for o in outcomes)
        for o in outcomes:
            units = o.spec.units or ''
            target = f'{o.spec.op} {o.spec.target:g} {units}'.rstrip()
            meas = (f'measured {o.measurement.value:g} {units}'.rstrip()
                    if o.measurement is not None else 'no measurement')
            if o.attempt_description:
                lines.append(f'  {o.display_label:<{name_w}}  {target}  {o.status}  {o.attempt_description}')
            else:
                lines.append(
                    f'  {o.display_label:<{name_w}}  {target:<18}  {meas:<26}  {o.status}'
                )
            if o.measurement is not None or o.latest_attempt is not None:
                lines.extend('    ' + note for note in _warning_notes(
                    self.metric_warnings, o,
                    o.measurement.value if o.measurement else None))
        if sr.calibrations_applied:
            lines.append(f'  Calibrations applied: {len(sr.calibrations_applied)}')
        return '\n'.join(lines)

    def _render_spec_reconciliation_md(self) -> str:
        """Render the Spec Reconciliation section as markdown.

        Returns the empty string when there are no specs, so the
        section header is omitted rather than shown as empty.
        """
        outcomes = self._reconciliation_outcomes()
        if self._grid_results():
            return self._render_grid_spec_outcomes(outcomes, 'md')
        return self._reconciliation_md(outcomes)

    def _reconciliation_md(self, outcomes, *, grid=False):
        if not outcomes:
            return ''

        title = 'Specs measured in a single run' if grid else self._reconciliation_title()
        lines = ['', '## ' + _pvt_text(title, 'markdown'), '']
        if grid:
            lines.extend(['Specs the PVT grid measures are in its table above.', ''])
        pass_n = sum(o.status == 'PASS' for o in outcomes)
        fail_n = sum(o.status == 'FAIL' for o in outcomes)
        unm_n  = sum(o.status == 'UNMEASURED' for o in outcomes)
        counts = f'{pass_n} PASS, {fail_n} FAIL' + ('' if grid else f', {unm_n} UNMEASURED')
        lines.append(f'**Summary:** {counts}')
        lines.append('')
        if any(o.measurement is not None for o in outcomes):
            lines.extend([
                'Results from the last run. '
                'Rerun the test after changing the circuit or stimulus to update these results.',
                '',
            ])
        lines.append('| Status | Spec | Op | Target | Measured | Source | Units | Margin |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for o in outcomes:
            s = o.spec
            if o.measurement is None:
                meas = '—'
                source = '—'
                margin = '—'
            else:
                meas = f'{o.measurement.value:g}'
                source = o.measurement.source
                margin = f'{o.margin:g}' if o.margin is not None else '—'
            if o.attempt_description:
                meas = _html.escape(o.attempt_description, quote=False).replace('|', '\\|')
            if o.measurement is not None or o.latest_attempt is not None:
                for n in _warning_notes(self.metric_warnings, o,
                                        o.measurement.value if o.measurement else None):
                    meas += ' <br>' + _html.escape(n, quote=False).replace('|', '\\|')
            label = _html.escape(o.display_label, quote=False).replace('|', '\\|')
            lines.append(
                f'| {o.status} | {label} | {s.op} | {s.target:g} | '
                f'{meas} | {source} | {s.units or "—"} | {margin} |'
            )
        if self.spec_reconciliation.calibrations_applied:
            lines.append('')
            lines.append('**Calibrations applied:**')
            for entry in self.spec_reconciliation.calibrations_applied:
                lines.append(f'- `{entry}`')
        lines.append('')
        return '\n'.join(lines)

    def save(self, path: str) -> None:
        """Write the report as a Markdown file to *path*."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._to_markdown())

    def _to_markdown(self) -> str:
        lines = []
        lines.append(f"# Circuit Report: {self.module_name}")
        lines.append(f"Generated: {self.generated_at}")
        lines.append("")
        lines.extend(_provenance_markdown(self.provenance))
        lines.append("---")
        lines.append("")

        # --- Topology ---
        lines.append("## Circuit Topology")
        lines.append("")
        lines.append(f"**Module:** {self.module_name}  ")
        lines.append(f"**Subcircuit instances:** {self.topology.total_instances}  ")
        lines.append(f"**Total nets:** {self.topology.total_nets}")
        lines.append("")
        if self.topology.instances:
            lines.append("| Instance | Subcircuit | Ports |")
            lines.append("|---|---|---|")
            for inst in self.topology.instances:
                lines.append(f"| {inst.instance} | {inst.module_name} | {inst.port_count} |")
        lines.append("")
        lines.append("---")
        lines.append("")

        # --- Composition diagram ---
        comp_md = self._render_composition_md()
        if comp_md:
            lines.append(comp_md)
            lines.append("---")
            lines.append("")

        # --- Annotation coverage ---
        lines.append("## Port Annotation Coverage")
        lines.append("")
        ann = self.annotation
        if ann.total == 0:
            lines.append("**No ports to annotate**")
        else:
            lines.append(
                f"**{ann.annotated}/{ann.total} ports annotated "
                f"({ann.coverage_pct:.1f}%)**"
            )
        lines.append("")
        if ann.unannotated_ports:
            lines.append("### Unannotated ports")
            lines.append("")
            lines.append("| Instance | Port | Direction | Block |")
            lines.append("|---|---|---|---|")
            for u in ann.unannotated_ports:
                lines.append(
                    f"| {u.instance} | {u.port} | {u.direction or '—'} | {u.module_name} |"
                )
        elif ann.total:
            lines.append("All ports have a `PortRole` declared. ✅")
        lines.append("")
        lines.append("---")
        lines.append("")

        # --- Interface check ---
        lines.append("## Interface Check")
        lines.append("")
        if not self.violations:
            lines.append("✅ **PASS** — no port conflicts")
        else:
            errors   = [v for v in self.violations if _severity_of(v) == Severity.ERROR]
            warnings = [v for v in self.violations if _severity_of(v) == Severity.WARNING]
            if errors:
                lines.append(f"❌ **FAIL** — {len(errors)} error(s), {len(warnings)} warning(s)")
            else:
                lines.append(f"⚠️ **PASS with warnings** — {len(warnings)} warning(s)")
            lines.append("")
            lines.append("| Severity | Rule | Net | Message |")
            lines.append("|---|---|---|---|")
            for v in self.violations:
                lines.append(
                    f"| {_severity_name(v)} | {v.rule_id} | "
                    f"{getattr(v, 'net', '')} | {v.message} |"
                )
        lines.append("")
        lines.append("---")
        lines.append("")

        # --- Domain violations ---
        if self.domain_violations:
            lines.append("## Voltage Domain Violations")
            lines.append("")
            for v in self.domain_violations:
                lines.append(
                    f"**[{_severity_name(v)}] {v.rule_id}** "
                    f"(net: `{getattr(v, 'net', '')}`)"
                )
                lines.append(f"> {v.message}")
                lines.append("")

        if self._stress_summary():
            lines.extend(['## Junction Stress Coverage', '', self._stress_summary(), ''])

        # --- Junction violations ---
        if self.junction_violations:
            lines.append("## Junction Overrange Violations")
            lines.append("")
            for v in self.junction_violations:
                lines.append(f"**[{_severity_name(v)}]** {v.message}")
                lines.append("")

        grid_md = self._render_grid('markdown')
        if grid_md:
            lines.append(grid_md)

        # --- Spec Reconciliation ---
        spec_md = self._render_spec_reconciliation_md()
        if spec_md:
            lines.append(spec_md)
            lines.append("---")
            lines.append("")

        # --- Simulation ---
        if self.simulation is not None:
            sim = self.simulation
            axis_label, axis_text = _sim_axis_text(sim)
            header, rows = _sim_table_rows(sim)
            lines.append("## Behavioral Simulation")
            lines.append("")
            lines.append(f"**Analysis:** {sim.analysis}  ")
            lines.append(f"**{axis_label}:** {axis_text}  ")
            lines.append(f"**Points:** {_sim_points_text(sim)}")
            lines.append("")
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "---|" * len(header))
            for name, *nums in rows:
                lines.append(
                    f"| {name} | " + " | ".join(f"{v:.4f}" for v in nums) + " |"
                )
            lines.append("")
            lines.append("---")
            lines.append("")

        # --- Layout estimate ---
        if self.layout_estimate is not None:
            le = self.layout_estimate
            lines.append("## Layout Estimate")
            lines.append("")
            w, h = le.bounding_box
            lines.append(
                f"**Bounding box:** {w:.2f} × {h:.2f} μm  "
                f"(routing model: `{le.routing_model_id}`)"
            )
            if le.packing_factor is not None or le.routing_overhead is not None:
                # Each provenance field is recorded independently (a captured
                # or hand-built estimate may carry one without the other), so
                # each is formatted only when present — an f-string ':g' on a
                # None value raises TypeError.
                heuristics = []
                if le.packing_factor is not None:
                    heuristics.append(
                        f"packing factor {le.packing_factor:g} "
                        f"(device area = W·L·M × packing factor)"
                    )
                if le.routing_overhead is not None:
                    heuristics.append(f"routing overhead {le.routing_overhead:g}")
                lines.append("")
                lines.append(
                    f"**Area heuristics:** {', '.join(heuristics)} — adjustable "
                    f"estimates, not layout truth."
                )
            lines.append("")
            if le.nets:
                lines.append("| Net | Length (μm) | Width (μm) | R (Ω) | C (fF) |")
                lines.append("|---|---|---|---|---|")
                for net_name, nl in sorted(le.nets.items()):
                    lines.append(
                        f"| {net_name} | {nl.length:.2f} | {nl.width:.3f} "
                        f"| {nl.r:.3f} | {nl.c:.3f} |"
                    )
            lines.append("")
            if le.warnings:
                lines.append("### Layout Warnings")
                lines.append("")
                for w_obj in le.warnings:
                    net_tag = f" (net: `{w_obj.net_name}`)" if w_obj.net_name else ""
                    lines.append(
                        f"- **[{w_obj.severity}] {w_obj.reason}**{net_tag}: {w_obj.detail}"
                    )
                lines.append("")
            lines.append("---")
            lines.append("")

        # --- Waveform explorer links ---
        if self.waveform_views:
            lines.append("## Waveform Explorer")
            lines.append("")
            for view_path, view_label in self.waveform_views:
                lines.append(f"- [{view_label}]({view_path})")
            lines.append("")
            lines.append("---")
            lines.append("")

        md = "\n".join(lines).rstrip()
        # Strip trailing horizontal rule if simulation section was omitted
        if md.endswith("---"):
            md = md[:-3].rstrip()
        return md + "\n"


# ---------------------------------------------------------------------------
# Bus plot helpers
# ---------------------------------------------------------------------------

def resolve_signal_path(module, path: str, *, behavioral_results: dict = None):
    """Resolve a signal-path string to a width-aware Signal.

    The ``path`` may be:
    - A bus name on the module (e.g., ``'d'`` if ``d`` is a Port.digital_output(bits=4))
      → returns a width-aware bus Signal aggregating per-bit Signals.
    - A scalar pin name or an explicit bit name (e.g., ``'clk'`` or ``'d[3]'``)
      → returns a scalar Signal.

    Parameters
    ----------
    module : Module
        The module whose buses and behavioral results are queried.
    path : str
        Signal name (without hierarchy prefix for v1).
    behavioral_results : dict, optional
        A mapping of net names to arrays from ``BehavioralEngine.run`` or
        an equivalent rawfile reader. Required for v1; without it, raises
        ``KeyError``.
    """
    import numpy as np
    from silk.signals.base import Signal

    behavioral_results = behavioral_results or {}
    name = path
    buses = getattr(module, '_buses', {}) or {}

    if name in buses:
        spec = buses[name]
        bit_signals = []
        for bit_name in spec.expanded_names():
            arr = behavioral_results.get(bit_name)
            if arr is None:
                raise KeyError(f"no behavioral result for bit {bit_name!r}")
            bit_signals.append(Signal(
                name=bit_name, type='voltage', xlabel='time', ylabel='V',
                xvalues=np.arange(len(arr)), yvalues=np.asarray(arr, dtype=float),
            ))
        return Signal._make_bus_handle(name=name, underlying=tuple(bit_signals), indices=spec.indices)

    # Scalar / per-bit lookup
    if name in behavioral_results:
        arr = behavioral_results[name]
        return Signal(name=name, type='voltage', xlabel='time', ylabel='V',
                      xvalues=np.arange(len(arr)), yvalues=np.asarray(arr, dtype=float))

    raise KeyError(f"no signal named {path!r}")


def render_bus_trace(ax, module, path: str, *, behavioral_results=None,
                     expand_bits: bool = False, threshold: float = 0.5):
    """Render a bus signal on a matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    module : Module
    path : str
        Signal path (bus name, scalar pin, or explicit bit).
    behavioral_results : dict
        ``{name: ndarray}`` of behavioral simulation outputs.
    expand_bits : bool, default False
        If False (default), render a single integer-decoded trace.
        If True, render a stacked bit-axis view (one trace per bit, vertically
        offset). For widths >16, emits a slow-render warning.
    threshold : float, default 0.5
        Logical-1 threshold for integer decoding.
    """
    import warnings

    handle = resolve_signal_path(module, path, behavioral_results=behavioral_results)
    if handle._underlying_signals is None:
        # Scalar path — single trace
        ax.plot(handle.x, handle.y, label=handle.name)
        return

    if expand_bits:
        if handle.width > 16:
            warnings.warn(
                f"render_bus_trace(expand_bits=True, width={handle.width}) "
                f"may render slowly; consider integer-decode mode (default).",
                stacklevel=2,
            )
        for i, bit in enumerate(handle):
            ax.plot(bit.x, bit.y + 1.5 * i, label=bit.name)
        return

    # Default: integer-decoded trace
    int_sig = handle.as_int(threshold=threshold)
    ax.plot(int_sig.x, int_sig.y, label=f'{handle.name} (int)')


def render_bus_html_row(module, bus_name: str, *, behavioral_results: dict) -> str:
    """Render a single HTML <tr> for a bus signal.

    v1 ships the single-row form: bus name, width, integer min/max, transition
    count. No interactive expand-on-click — users inspect bits via
    ``tb.plot('inst.d', expand_bits=True)`` or ``tb.plot('inst.d[i]')``.

    Falls back to a minimal scalar row when ``bus_name`` resolves to a scalar.
    """
    import numpy as np
    from html import escape

    handle = resolve_signal_path(module, bus_name,
                                 behavioral_results=behavioral_results)
    if handle._underlying_signals is None:
        # Scalar — minimal row (delegate-style; full scalar HTML is in
        # other report code)
        arr = behavioral_results.get(bus_name)
        sample_count = len(arr) if arr is not None else 0
        return (
            f'<tr><td>{escape(bus_name)}</td>'
            f'<td>scalar, {sample_count} samples</td></tr>'
        )

    # Bus row — integer-decoded summary
    int_sig = handle.as_int()
    int_y = int_sig.y.astype(int)
    transitions = int(np.sum(np.diff(int_y) != 0))
    return (
        f'<tr class="bus-row">'
        f'<td class="bus-name">{escape(handle.name)} ({handle.width}-bit bus)</td>'
        f'<td class="bus-summary">'
        f'min={int_y.min()}, max={int_y.max()}, '
        f'transitions={transitions}'
        f'</td>'
        f'</tr>'
    )


def generate_report(module, registry=None, auto_discover_config: bool = True,
                    layout_estimate=None, spec_reports=None, test_results=None, *,
                    region_thresholds=None) -> CircuitReport:
    """Assemble a CircuitReport from a Module.

    Reads topology from ``module.module_dict``/``module.net_dict``,
    annotation coverage from the existing metric helpers, and runs the
    lint registry once to populate both the LintReport and the
    CoverageReport. The lint and coverage objects are embedded on the
    returned CircuitReport.

    Parameters
    ----------
    module : Module
        The top-level module to report on.
    registry : RuleRegistry, optional
        Pre-built registry. Use this to inject rules with custom
        severity overrides, pre-loaded waivers, or pre-loaded
        exclusions. If omitted, a fresh ``RuleRegistry.default()`` is
        constructed.
    auto_discover_config : bool, default True
        When ``registry`` is omitted, look in the current working
        directory for ``silk_waivers.yaml`` and
        ``silk_lint_exclusions.yaml`` and load them automatically.
        Pass ``False`` to suppress this behavior in test environments.
    spec_reports : list of SpecReport, optional
        Already-evaluated records to snapshot as a records collection,
        preserving duplicates and order. Successful records also source
        operating point tables, metric warnings and distinct provenance.
    region_thresholds : RegionThresholds, optional
        Explicit operating-region policy for every included report. If omitted,
        each report resolves SILK_REGION_THRESHOLDS and its shipped PDK default
        using its own captured PDK identity and temperature.
    test_results : dict or CornerResults, optional
        Already-evaluated runner evidence, including failures. Alternative to
        spec_reports; supplying both non-None inputs raises ValueError. Neither
        supplied means null evidence; explicit empty input stays empty. This
        function never runs tests or discovers evidence from live benches.
    layout_estimate : LayoutEstimate, optional
        Layout estimate to embed (and to diff or serialize with the report).

    Returns
    -------
    CircuitReport
        With ``lint``, ``coverage``, ``metadata`` and ``spec_reconciliation``
        populated.

    Raises
    ------
    ValueError
        If both *spec_reports* and *test_results* are supplied.
    TypeError
        If *spec_reports* is not a ``list``, or the evidence has an
        unsupported type.
    """
    from silk.circuit.interface_check import annotation_coverage
    from silk.lint.registry import RuleRegistry, module_has_simulation_data
    from silk.lint.waivers import apply_waivers
    from silk.lint_markers import find_stale_markers

    from silk.circuit._results_json import results_data, restore_results, successful_reports

    if spec_reports is not None and test_results is not None:
        raise ValueError('Supply only one of spec_reports and test_results')
    if spec_reports is not None and type(spec_reports) is not list:
        raise TypeError('spec_reports must be a list of SpecReports')
    evidence = spec_reports if spec_reports is not None else test_results
    if evidence is not None:
        evidence = restore_results(results_data(evidence))
        spec_reports = list(successful_reports(evidence))

    module_name = getattr(module, 'module_name', None) or 'unnamed'

    metadata = ReportMetadata(
        silk_version=_detect_silk_version(),
        pdk_name=_detect_pdk_name(module),
        pdk_version=_detect_pdk_version(module),
        module_name=module_name,
    )

    if registry is None:
        registry = RuleRegistry.default()
        if auto_discover_config:
            _maybe_load_local_config(registry)

    # Behavioral OR SPICE — shared with check_junction_overrange so the two
    # entry points cannot drift again (TASK-47).
    has_sim = module_has_simulation_data(module)
    run_result = registry.run(module, has_simulation_data=has_sim)

    # Apply waivers from the registry
    unsuppressed, suppressed = apply_waivers(run_result.violations, registry._waivers)

    # Filter out violations whose location matches a scope exclusion
    unsuppressed_after_exclusion = []
    excluded = []
    for v in unsuppressed:
        instance_path = _location_instance_path(v.location)
        if instance_path and registry.is_excluded(instance_path) is not None:
            excluded.append(v)
        else:
            unsuppressed_after_exclusion.append(v)

    # Partition by category for the legacy CircuitReport fields
    interface_violations = [v for v in unsuppressed_after_exclusion if v.rule_id.startswith("INT-")]
    domain_violations = [v for v in unsuppressed_after_exclusion if v.rule_id.startswith("DOM-")]
    junction_violations = [v for v in unsuppressed_after_exclusion if v.rule_id.startswith("STR-")]

    lint_report = LintReport(
        metadata=metadata,
        violations=unsuppressed_after_exclusion,
        waivers_applied=suppressed,
        waivers_matched=[w for w in registry._waivers
                         if any(w.matches(v) for v in suppressed)],
        no_connects=list(getattr(module, '_no_connects', [])),
        no_connect_ports=list(getattr(module, '_no_connect_ports', [])),
        stale_markers=find_stale_markers(module),
    )

    coverage_report = CoverageReport.from_visit_log(
        metadata=metadata,
        visit_log=run_result.visit_log,
        module_name=module_name,
    )

    lint_report.stress_coverage = coverage_report.stress_coverage

    report = CircuitReport(
        module_name=module_name,
        generated_at=datetime.datetime.now().isoformat(timespec='seconds'),
        topology=_build_topology(module),
        annotation=annotation_coverage(module),
        violations=interface_violations,
        simulation=_build_simulation(module),
        domain_violations=domain_violations,
        junction_violations=junction_violations,
        metadata=metadata,
        lint=lint_report,
        coverage=coverage_report,
        stress_coverage=coverage_report.stress_coverage,
        layout_estimate=layout_estimate,
        test_results=evidence,
        op_tables=_build_op_tables(spec_reports, region_thresholds),
        provenance=_collect_provenance(spec_reports),
        metric_warnings=_build_metric_warnings(spec_reports),
    )

    # Spec reconciliation — always attach, even when empty, so downstream
    # renderers have a uniform contract.
    report.spec_reconciliation = module.reconcile_specs()

    # Runtime-only back-ref used by print_summary to filter the op-point
    # table to module-level nets. Not serialized.
    report._source_module = module

    return report


def _collect_provenance(spec_reports) -> list:
    """Distinct provenance records of ``spec_reports`` in the caller's order
    (deduped by digest). Empty when none carries one, so a report built
    without ``spec_reports=`` renders exactly as before."""
    records, seen = [], set()
    for sr in spec_reports or []:
        rec = getattr(sr, 'provenance', None)
        if rec is None:
            continue
        key = rec.digest()
        if key not in seen:
            seen.add(key)
            records.append(rec)
    return records


def _provenance_from_list(items) -> list:
    from silk.simulation.provenance import RunProvenance
    return [RunProvenance.from_dict(d) for d in (items or [])]


def _provenance_rows(rec) -> List[Tuple[str, str]]:
    """(field, display value) for every non-None field of one record."""
    from silk.simulation.provenance import FIELDS
    rows = []
    d = rec.to_dict()
    for name in FIELDS:
        value = d.get(name)
        if value is None or value == [] or value == {}:
            continue
        if name == 'pdk_libraries':
            value = '; '.join(
                f"{e.get('file')} [{e.get('section')}] file_sha256 "
                f"{(e.get('file_sha256') or '?')[:16]} section_text_sha256 "
                f"{(e.get('section_text_sha256') or '?')[:16]}" for e in value)
        elif isinstance(value, (dict, list)):
            value = json.dumps(encode_json(value), sort_keys=True,
                               allow_nan=False)
        rows.append((name, str(value)))
    return rows


def _run_detail_lines(records):
    """Display shared tool and library facts once, in first-record order."""
    import os
    lines = []
    for rec in records:
        silk = 'silk ' + (rec.silk_version or '?')
        if rec.run_origin == 'external-rawfile':
            simulator = f'external rawfile ({rec.simulator or "simulator unknown"})'
        else:
            simulator = rec.simulator_version or rec.simulator or 'simulator unknown'
        parts = [silk, simulator]
        libraries = ', '.join(os.path.basename(item.get('file') or '?') + ' '
                              + (item.get('file_sha256') or '?')[:10]
                              for item in rec.pdk_libraries or ())
        if rec.pdk_name or libraries:
            parts.append((rec.pdk_name or 'PDK') + (f' ({libraries})' if libraries else ''))
        lines.append(', '.join(parts))
    return list(dict.fromkeys(lines))


def _pvt_number(value, *, missing='—'):
    from numbers import Real
    if value is None:
        return missing
    return f'{value:g}' if isinstance(value, Real) else str(value)


def _pvt_point(point):
    if point is None:
        return '—'
    if not isinstance(point, tuple):
        return str(point)
    corner, temperature, supply = point[:3]
    return (f'{corner}, ' + _pvt_number(temperature) + ' °C, '
            + _pvt_number(supply) + ' V'
            + (''.join(f', {name}={value!r}' for name, value in point[3]) if len(point) == 4 else ''))


def _pvt_text(value, mode):
    if mode != 'markdown':
        return str(value).replace('\r', '').replace('\n', ' ')
    return _html.escape(str(value)).replace('\\', '\\\\').replace('|', '\\|').replace('\n', '<br>').replace('\r', '')


def _pvt_table(columns, rows, mode, *, specs=False):
    if mode == 'html':
        out = ['<table' + (' class="spec-reconciliation"' if specs else '') + '><thead><tr>'
               + ''.join('<th>' + _html.escape(str(c)) + '</th>' for c in columns)
               + '</tr></thead><tbody>']
        for row in rows:
            status = row[0].lower() if specs else None
            css = ('spec-unmeasured' if status == 'unavailable' else 'spec-' + status) if specs else ''
            out.append(('<tr class="' + css + '">' if specs else '<tr>') + ''.join(
                '<td' + (' class="status"' if specs and i == 0 else '') + '>'
                + _html.escape(str(cell)) + '</td>' for i, cell in enumerate(row)) + '</tr>')
        out.append('</tbody></table>')
        return '\n'.join(out)
    return '\n'.join(['| ' + ' | '.join(_pvt_text(c, mode) for c in columns) + ' |',
                       '|' + '---|' * len(columns), *(
                       '| ' + ' | '.join(_pvt_text(cell, mode) for cell in row) + ' |' for row in rows)])


def _provenance_html(records, esc) -> List[str]:
    if not records:
        return []
    return ['<section class="provenance">', *(
        '<p>Run details: ' + esc(line) + '</p>' for line in _run_detail_lines(records)), '</section>']


def _provenance_markdown(records) -> List[str]:
    return [item for line in _run_detail_lines(records)
            for item in ['Run details: ' + _pvt_text(line, 'markdown'), '']]


def _location_instance_path(loc) -> str:
    """Best-effort extraction of an instance_path from a violation location."""
    return (
        getattr(loc, 'instance_path', None)
        or getattr(loc, 'module_name', None)
        or ''
    )


def _maybe_load_local_config(registry) -> None:
    """Auto-load silk_waivers.yaml and silk_lint_exclusions.yaml from cwd."""
    import os
    cwd = os.getcwd()
    waiver_file = os.path.join(cwd, 'silk_waivers.yaml')
    if os.path.isfile(waiver_file):
        try:
            registry.load_waivers(waiver_file)
        except Exception:
            pass  # Don't crash report generation on a malformed config file
    exclusion_file = os.path.join(cwd, 'silk_lint_exclusions.yaml')
    if os.path.isfile(exclusion_file):
        try:
            registry.load_exclusions(exclusion_file)
        except Exception:
            pass


def _detect_silk_version() -> str:
    try:
        from silk import __version__
        return __version__
    except (ImportError, AttributeError):
        return "unknown"


def _detect_pdk_name(module) -> str:
    pdk = getattr(module, "pdk_details", None) or getattr(module, "pdk", None)
    if pdk is None:
        return "unknown"
    for attr in ("pdk_name", "name"):
        val = getattr(pdk, attr, None)
        if isinstance(val, str) and val:
            return val
    return "unknown"


def _detect_pdk_version(module) -> str:
    pdk = getattr(module, "pdk_details", None) or getattr(module, "pdk", None)
    if pdk is None:
        return "unknown"
    val = getattr(pdk, "version", None)
    if isinstance(val, str) and val:
        return val
    return "unknown"
