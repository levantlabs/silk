"""silk.circuit.report_diff — cross-session circuit report diffing.

Compare two CircuitReports (in-memory or loaded from JSON) and produce a
ReportDiff that can be printed to the terminal or saved as a Markdown file.

Usage::

    from silk import diff_reports

    diff = diff_reports('sim_schematic.json', 'sim_extracted.json',
                        label_a='schematic', label_b='extracted')
    diff.print_summary()
    diff.save('schematic_vs_extracted.md')
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union, TYPE_CHECKING

from silk.signals.analysis_kind import SMALL_SIGNAL_KINDS
from silk.lint.locations import NetLocation

if TYPE_CHECKING:
    from silk.circuit.report import CircuitReport
    from silk.circuit.interface_check import InterfaceViolation


# ---------------------------------------------------------------------------
# Diff dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SignalDelta:
    """Statistics of one signal in report A and report B.

    Built by :func:`silk.circuit.diff_reports` for a signal present in both transient or
    DC reports whose minimum, maximum or mean differ exactly. Each ``*_a`` /
    ``*_b`` pair is the statistic from report A and report B, in the signal's
    own unit (volts for ``v(...)``, amps for ``i(...)``).

    Parameters
    ----------
    name : str
        Signal label, for example ``'vout'`` or ``'i(vsupply)'``.
    min_a, min_b : float
        Minimum of the saved vector in report A and report B.
    max_a, max_b : float
        Maximum of the saved vector in report A and report B.
    mean_a, mean_b : float
        Mean of the saved vector in report A and report B.
    """
    name:   str     #: Signal label, as in ``SignalStats.name``.
    min_a:  float   #: Minimum in report A (signal unit).
    min_b:  float   #: Minimum in report B (signal unit).
    max_a:  float   #: Maximum in report A (signal unit).
    max_b:  float   #: Maximum in report B (signal unit).
    mean_a: float   #: Mean in report A (signal unit).
    mean_b: float   #: Mean in report B (signal unit).

    @property
    def delta_mean(self) -> float:
        """``mean_b - mean_a``, in the signal's unit."""
        return self.mean_b - self.mean_a

    @property
    def delta_max(self) -> float:
        """``max_b - max_a``, in the signal's unit."""
        return self.max_b - self.max_a

    @property
    def delta_min(self) -> float:
        """``min_b - min_a``, in the signal's unit."""
        return self.min_b - self.min_a


@dataclass
class SignalDiff:
    """Per-signal diff of two transient or DC reports, by signal name.

    Parameters
    ----------
    changed : list of SignalDelta
        Signals in both reports whose min, max or mean differ (exact float
        comparison), sorted by name.
    unchanged : list of str
        Names present in both with identical statistics, sorted.
    added : list of str
        Names present in report B only, sorted.
    dropped : list of str
        Names present in report A only, sorted.
    """
    changed:   List[SignalDelta]   #: Present in both, statistics differ.
    unchanged: List[str]           #: Present in both, identical (exact float equality).
    added:     List[str]           #: In B only.
    dropped:   List[str]           #: In A only.


@dataclass
class MagnitudeDelta:
    """Magnitude statistics of one node in report A and report B.

    Built for AC and noise reports, whose ``SimulationSummary.magnitudes``
    hold ``|x|`` statistics (:class:`~silk.circuit.report.MagnitudeStats`),
    never voltages. Values are in the vector's own unit: a dimensionless
    transfer ratio for AC, V/sqrt(Hz) for noise density.

    Parameters
    ----------
    name : str
        Signal label.
    mag_min_a, mag_min_b : float
        Minimum magnitude in report A and report B.
    mag_max_a, mag_max_b : float
        Maximum magnitude in report A and report B.
    mag_mean_a, mag_mean_b : float
        Mean magnitude in report A and report B.
    """
    name:        str     #: Signal label, as in ``MagnitudeStats.name``.
    mag_min_a:   float   #: Minimum of ``|x|`` in report A.
    mag_min_b:   float   #: Minimum of ``|x|`` in report B.
    mag_max_a:   float   #: Maximum of ``|x|`` in report A.
    mag_max_b:   float   #: Maximum of ``|x|`` in report B.
    mag_mean_a:  float   #: Mean of ``|x|`` in report A.
    mag_mean_b:  float   #: Mean of ``|x|`` in report B.

    @property
    def delta_mag_mean(self) -> float:
        """``mag_mean_b - mag_mean_a``."""
        return self.mag_mean_b - self.mag_mean_a

    @property
    def delta_mag_max(self) -> float:
        """``mag_max_b - mag_max_a``."""
        return self.mag_max_b - self.mag_max_a

    @property
    def delta_mag_min(self) -> float:
        """``mag_min_b - mag_min_a``."""
        return self.mag_min_b - self.mag_min_a


@dataclass
class MagnitudeDiff:
    """Node-magnitude diff for a matching pair of AC or noise reports.

    Counterpart to :class:`SignalDiff`: AC and noise runs keep their
    statistics in ``SimulationSummary.magnitudes`` rather than ``.signals``,
    so they need their own shape instead of being forced through
    :class:`SignalDelta`'s volt-flavoured field names.

    Parameters
    ----------
    changed : list of MagnitudeDelta
        Nodes in both reports whose magnitude statistics differ (exact float
        comparison), sorted by name.
    unchanged : list of str
        Names present in both with identical statistics, sorted.
    added : list of str
        Names present in report B only, sorted.
    dropped : list of str
        Names present in report A only, sorted.
    """
    changed:   List[MagnitudeDelta]   #: Present in both, statistics differ.
    unchanged: List[str]              #: Present in both, identical (exact float equality).
    added:     List[str]              #: In B only.
    dropped:   List[str]              #: In A only.


@dataclass
class KindMismatch:
    """The two reports ran different analyses (for example transient vs AC).

    Their signal statistics live under different fields and are not
    node-for-node comparable, so :func:`silk.circuit.diff_reports` reports this as an
    explicit structural difference rather than an empty :class:`SignalDiff`,
    which would read as "no change".

    Parameters
    ----------
    analysis_a : str
        Analysis kind of report A (``'transient'``, ``'ac'``, ``'noise'``,
        ``'dc'`` or ``'op'``).
    analysis_b : str
        Analysis kind of report B, same vocabulary.
    """
    analysis_a: str   #: Analysis kind of report A.
    analysis_b: str   #: Analysis kind of report B.


@dataclass
class TopologyDiff:
    """Change in a module's subcircuit hierarchy between two reports.

    Parameters
    ----------
    instances_added : list of str
        Instance names in report B but not A, sorted.
    instances_removed : list of str
        Instance names in report A but not B, sorted.
    nets_delta : int
        ``total_nets`` of B minus that of A.
    """
    instances_added:   List[str]   #: Instance names new in B.
    instances_removed: List[str]   #: Instance names gone from B.
    nets_delta:        int         #: ``total_nets`` in B minus ``total_nets`` in A.


@dataclass
class AnnotationDiff:
    """Change in port-annotation coverage between two reports.

    Parameters
    ----------
    coverage_a : float
        ``AnnotationCoverage.coverage_pct`` of report A, in percent (0 to
        100).
    coverage_b : float
        Coverage percentage of report B.
    ports_resolved : list of str
        ``'instance.port'`` entries unannotated in A and not unannotated in
        B, sorted.
    ports_added : list of str
        ``'instance.port'`` entries unannotated in B and not unannotated in
        A, sorted.
    """
    coverage_a:     float          #: Coverage percentage in A (0 to 100).
    coverage_b:     float          #: Coverage percentage in B (0 to 100).
    ports_resolved: List[str]      #: ``'instance.port'`` unannotated in A, annotated in B.
    ports_added:    List[str]      #: ``'instance.port'`` unannotated in B but not in A.


@dataclass
class ViolationDiff:
    """Interface-violation change between two reports.

    Violations are matched by ``(rule_id, location.describe())``, ignoring
    message, severity and context changes. Repeated findings at the same
    rule and location produce one entry.

    Parameters
    ----------
    resolved : list of InterfaceViolation
        Violations in A and not in B (report A's objects).
    introduced : list of InterfaceViolation
        Violations in B and not in A (report B's objects).
    unchanged : list of InterfaceViolation
        Violations in both, matched by rule and location (report A's objects).
    """
    resolved:   List['InterfaceViolation']   #: In A, not in B.
    introduced: List['InterfaceViolation']   #: In B, not in A.
    unchanged:  List['InterfaceViolation']   #: In both (matched by rule and location).


@dataclass
class LayoutNetDelta:
    """Per-net layout scalars of two layout estimates, side by side.

    Values come from :class:`~silk.layout.estimator.NetLayout` of report A
    and report B.

    Parameters
    ----------
    name : str
        Net name.
    length_a, length_b : float
        Estimated wire length in μm.
    width_a, width_b : float
        Estimated wire width in μm.
    r_a, r_b : float
        Estimated total wire resistance in Ω.
    c_a, c_b : float
        Estimated total wire capacitance in fF.
    """
    name:     str     #: Net name.
    length_a: float   #: Wire length in A (μm).
    length_b: float   #: Wire length in B (μm).
    width_a:  float   #: Wire width in A (μm).
    width_b:  float   #: Wire width in B (μm).
    r_a:      float   #: Total wire resistance in A (Ω).
    r_b:      float   #: Total wire resistance in B (Ω).
    c_a:      float   #: Total wire capacitance in A (fF).
    c_b:      float   #: Total wire capacitance in B (fF).

    @property
    def delta_r(self) -> float:
        """``r_b - r_a`` (Ω)."""
        return self.r_b - self.r_a

    @property
    def delta_c(self) -> float:
        """``c_b - c_a`` (fF)."""
        return self.c_b - self.c_a

    @property
    def delta_length(self) -> float:
        """``length_b - length_a`` (μm)."""
        return self.length_b - self.length_a

    @property
    def delta_width(self) -> float:
        """``width_b - width_a`` (μm)."""
        return self.width_b - self.width_a


@dataclass
class LayoutDiff:
    """Layout-estimate delta between two reports.

    ``ReportDiff.layout_diff`` is ``None`` unless both reports carry a
    layout estimate. A net counts as unchanged when its length, width,
    resistance and capacitance each differ by less than 1e-12.

    Parameters
    ----------
    changed : list of LayoutNetDelta
        Nets in both estimates whose scalars differ, sorted by name.
    unchanged : list of str
        Nets in both with matching scalars, sorted.
    nets_added : list of str
        Nets in B only, sorted.
    nets_dropped : list of str
        Nets in A only, sorted.
    bbox_a : tuple of float
        ``(width, height)`` of the module bounding box in A, in μm.
    bbox_b : tuple of float
        ``(width, height)`` of the module bounding box in B, in μm.
    """
    changed:      List[LayoutNetDelta]   #: Nets in both, scalars differ.
    unchanged:    List[str]              #: Nets in both, identical.
    nets_added:   List[str]              #: Nets in B only.
    nets_dropped: List[str]              #: Nets in A only.
    bbox_a:       tuple                  #: ``(width, height)`` in μm from A.
    bbox_b:       tuple                  #: ``(width, height)`` in μm from B.

    @property
    def bbox_area_delta(self) -> float:
        """Change in bounding-box area (μm²) from A to B."""
        return self.bbox_b[0] * self.bbox_b[1] - self.bbox_a[0] * self.bbox_a[1]


# ---------------------------------------------------------------------------
# ReportDiff
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceDiff:
    """Differing provenance fields between two reports.

    Records are paired by position (a report built from one SpecReport has
    one). ``changed`` holds ``{'record', 'field', 'a', 'b'}`` rows in
    ``silk.simulation.provenance.FIELDS`` order; ``timestamp_utc`` is never
    compared. A record present on one side only is one row with field
    ``'record'`` and that record's digest on its side.

    Parameters
    ----------
    records_a : int
        Number of provenance records in report A.
    records_b : int
        Number of provenance records in report B.
    changed : list of dict
        One ``{'record', 'field', 'a', 'b'}`` row per differing field.
    """
    records_a: int   #: Number of provenance records in report A.
    records_b: int   #: Number of provenance records in report B.
    changed: List[dict] = field(default_factory=list)   #: Differing-field rows ``{'record', 'field', 'a', 'b'}``; empty when the records agree.


@dataclass
class ReportDiff:
    """Difference between two CircuitReports.

    Built by :func:`silk.circuit.diff_reports`. Mirrors :class:`~silk.circuit.report.CircuitReport`:
    use ``print_summary()`` for terminal output, ``save(path)`` for a
    Markdown file, and ``str()`` for a text summary.

    Parameters
    ----------
    label_a, label_b : str
        Human-readable names of the two reports.
    generated_at : str
        Local-time ISO timestamp (seconds resolution) of when the diff was
        built.
    sections_compared : set of str
        Sections that were requested: any of ``'signals'``, ``'topology'``,
        ``'annotation'``, ``'violations'``, ``'provenance'``.
    signal_diff : SignalDiff, MagnitudeDiff, KindMismatch or None
        Signal statistics diff; ``None`` when ``'signals'`` was not
        requested or either report has no simulation.
    topology_diff : TopologyDiff or None
        ``None`` when ``'topology'`` was not requested.
    annotation_diff : AnnotationDiff or None
        ``None`` when ``'annotation'`` was not requested.
    violation_diff : ViolationDiff or None
        ``None`` when ``'violations'`` was not requested.
    layout_diff : LayoutDiff or None
        ``None`` unless both reports carry a layout estimate. Computed
        regardless of *sections*.
    spec_transitions : list of dict
        One row per spec whose reconciled status or value differs between
        the reports, with keys ``spec_name``, ``owner_path``, ``spec_key``,
        ``report_key``, ``before_status``, ``after_status``,
        ``before_value``, ``after_value`` and ``source_used``. Computed
        regardless of *sections*.
    provenance_diff : ProvenanceDiff or None
        ``None`` when ``'provenance'`` was not requested.
    """
    label_a:           str   #: Name of report A.
    label_b:           str   #: Name of report B.
    generated_at:      str   #: Local ISO timestamp (seconds) of when the diff was built.
    sections_compared: Set[str]   #: Sections requested from :func:`silk.circuit.diff_reports`.
    signal_diff:       Optional[Union[SignalDiff, MagnitudeDiff, KindMismatch]]   #: Signal diff; ``None`` if not requested or either report has no simulation.
    topology_diff:     Optional[TopologyDiff]   #: Topology diff; ``None`` if not requested.
    annotation_diff:   Optional[AnnotationDiff]   #: Annotation diff; ``None`` if not requested.
    violation_diff:    Optional[ViolationDiff]   #: Interface-violation diff; ``None`` if not requested.
    layout_diff:       Optional['LayoutDiff'] = None   #: Layout diff; ``None`` unless both reports have a layout estimate.
    spec_transitions:  List[dict] = field(default_factory=list)   #: Spec status/value changes, one dict per spec.
    provenance_diff:   Optional[ProvenanceDiff] = None   #: Provenance diff; ``None`` if not requested.

    def _provenance_lines(self, indent: str) -> List[str]:
        pd = self.provenance_diff
        if pd is None or (pd.records_a == 0 and pd.records_b == 0):
            return []
        if not pd.changed:
            return [f"{indent}Provenance: unchanged"]
        out = [f"{indent}Provenance ({len(pd.changed)} field(s) differ):"]
        for row in pd.changed:
            tag = f"[{row['record']}] " if max(pd.records_a, pd.records_b) > 1 else ''
            out.append(f"{indent}  {tag}{row['field']}: {row['a']!r} -> {row['b']!r}")
        return out

    def __str__(self) -> str:
        """Human-readable summary including any spec transitions."""
        lines = [f"Diff: {self.label_a} -> {self.label_b}"]

        if self.signal_diff is not None:
            sd = self.signal_diff
            if isinstance(sd, KindMismatch):
                lines.append(
                    f"  Signals: kind mismatch ({sd.analysis_a} vs {sd.analysis_b})"
                )
            else:
                label = 'Signals' if isinstance(sd, SignalDiff) else 'Magnitudes'
                n = len(sd.changed) + len(sd.unchanged) + len(sd.added) + len(sd.dropped)
                if sd.changed or sd.added or sd.dropped:
                    lines.append(
                        f"  {label} ({n} compared, {len(sd.changed)} changed, "
                        f"{len(sd.added)} added, {len(sd.dropped)} dropped)"
                    )
                else:
                    lines.append(f"  {label}: unchanged ({n} compared)")

        if self.topology_diff is not None:
            td = self.topology_diff
            if td.instances_added or td.instances_removed or td.nets_delta != 0:
                lines.append(
                    f"  Topology: {len(td.instances_added)} added, "
                    f"{len(td.instances_removed)} removed, "
                    f"nets {td.nets_delta:+d}"
                )
            else:
                lines.append("  Topology: unchanged")

        if self.annotation_diff is not None:
            ad = self.annotation_diff
            if abs(ad.coverage_b - ad.coverage_a) > 0.01:
                lines.append(
                    f"  Annotation: {ad.coverage_a:.1f}% -> {ad.coverage_b:.1f}%"
                )
            else:
                lines.append(f"  Annotation: unchanged ({ad.coverage_a:.1f}%)")

        if self.violation_diff is not None:
            vd = self.violation_diff
            if vd.introduced or vd.resolved:
                lines.append(
                    f"  Violations: {len(vd.introduced)} introduced, "
                    f"{len(vd.resolved)} resolved, {len(vd.unchanged)} unchanged"
                )
            else:
                lines.append(f"  Violations: unchanged ({len(vd.unchanged)})")

        lines.extend(self._provenance_lines('  '))

        if self.spec_transitions:
            lines.append('')
            lines.append('Spec transitions:')
            for t in self.spec_transitions:
                bv = f'{t["before_value"]:g}' if t['before_value'] is not None else '-'
                av = f'{t["after_value"]:g}' if t['after_value'] is not None else '-'
                bs = t['before_status'] or '-'
                as_ = t['after_status'] or '-'
                lines.append(
                    f'  {_spec_transition_label(t)}: {bs} ({bv}) -> {as_} ({av})'
                )

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print a condensed diff to stdout."""
        print(f"Diff: {self.label_a} → {self.label_b}")

        if self.signal_diff is not None:
            sd = self.signal_diff
            if isinstance(sd, KindMismatch):
                print(f"  Signals: kind mismatch ({sd.analysis_a} vs {sd.analysis_b})")
            elif isinstance(sd, MagnitudeDiff):
                n = len(sd.changed) + len(sd.unchanged) + len(sd.added) + len(sd.dropped)
                if sd.changed or sd.added or sd.dropped:
                    print(f"  Magnitudes ({n} compared, {len(sd.changed)} changed, "
                          f"{len(sd.added)} added, {len(sd.dropped)} dropped):")
                    for d in sd.changed:
                        print(f"    {d.name:<12} |mean| {d.mag_mean_a:.6g} → {d.mag_mean_b:.6g}"
                              f"  (Δ {d.delta_mag_mean:+.3g})")
                    for name in sd.added:
                        print(f"    {name:<12} [added]")
                    for name in sd.dropped:
                        print(f"    {name:<12} [dropped]")
                else:
                    print(f"  Magnitudes: unchanged ({n} compared)")
            else:
                n = len(sd.changed) + len(sd.unchanged) + len(sd.added) + len(sd.dropped)
                if sd.changed or sd.added or sd.dropped:
                    print(f"  Signals ({n} compared, {len(sd.changed)} changed, "
                          f"{len(sd.added)} added, {len(sd.dropped)} dropped):")
                    for d in sd.changed:
                        print(f"    {d.name:<12} mean  {d.mean_a:.3f} → {d.mean_b:.3f} V"
                              f"  (Δ {d.delta_mean:+.3f} V)")
                    for name in sd.added:
                        print(f"    {name:<12} [added]")
                    for name in sd.dropped:
                        print(f"    {name:<12} [dropped]")
                else:
                    print(f"  Signals: unchanged ({n} compared)")

        if self.topology_diff is not None:
            td = self.topology_diff
            if td.instances_added or td.instances_removed or td.nets_delta != 0:
                print(f"  Topology: {len(td.instances_added)} added, "
                      f"{len(td.instances_removed)} removed, "
                      f"nets Δ{td.nets_delta:+d}")
            else:
                print("  Topology: unchanged")

        if self.annotation_diff is not None:
            ad = self.annotation_diff
            if abs(ad.coverage_b - ad.coverage_a) > 0.01:
                print(f"  Annotation: {ad.coverage_a:.1f}% → {ad.coverage_b:.1f}%")
            else:
                print(f"  Annotation: unchanged ({ad.coverage_a:.1f}%)")

        if self.violation_diff is not None:
            vd = self.violation_diff
            if vd.introduced or vd.resolved:
                print(f"  Violations: {len(vd.introduced)} introduced, "
                      f"{len(vd.resolved)} resolved, "
                      f"{len(vd.unchanged)} unchanged")
            else:
                print(f"  Violations: unchanged ({len(vd.unchanged)})")

        for line in self._provenance_lines('  '):
            print(line)

        if self.spec_transitions:
            print("  Spec transitions:")
            for t in self.spec_transitions:
                bv = f'{t["before_value"]:g}' if t['before_value'] is not None else '-'
                av = f'{t["after_value"]:g}' if t['after_value'] is not None else '-'
                bs = t['before_status'] or '-'
                as_ = t['after_status'] or '-'
                print(f"    {_spec_transition_label(t)}: {bs} ({bv}) → {as_} ({av})")

    def save(self, path: str) -> None:
        """Write the diff as a Markdown file to *path*."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self._to_markdown())

    def _to_markdown(self) -> str:
        lines = []
        lines.append(f"# Circuit Report Diff: {self.label_a} → {self.label_b}")
        lines.append(f"Generated: {self.generated_at}")
        lines.append("")
        lines.append("---")
        lines.append("")

        if self.signal_diff is not None:
            sd = self.signal_diff
            if isinstance(sd, KindMismatch):
                lines.append("## Signal Comparison")
                lines.append("")
                lines.append(
                    f"**Kind mismatch:** {self.label_a} ran `{sd.analysis_a}`, "
                    f"{self.label_b} ran `{sd.analysis_b}` — not node-for-node "
                    f"comparable."
                )
                lines.append("")
                lines.append("---")
                lines.append("")
            elif isinstance(sd, MagnitudeDiff):
                n = len(sd.changed) + len(sd.unchanged) + len(sd.added) + len(sd.dropped)
                lines.append("## Magnitude Comparison")
                lines.append("")
                lines.append(
                    f"**{n} nodes compared — {len(sd.changed)} changed, "
                    f"{len(sd.added)} added, {len(sd.dropped)} dropped**"
                )
                lines.append("")
                if sd.changed:
                    lines.append("### Changed")
                    lines.append("")
                    lines.append(
                        f"| Node | |mean| {self.label_a} | |mean| {self.label_b} | Δ |mean| |"
                        f" |max| {self.label_a} | |max| {self.label_b} | Δ |max| |"
                    )
                    lines.append("|---|---|---|---|---|---|---|")
                    for d in sd.changed:
                        lines.append(
                            f"| {d.name} | {d.mag_mean_a:.6g} | {d.mag_mean_b:.6g} |"
                            f" {d.delta_mag_mean:+.3g} | {d.mag_max_a:.6g} |"
                            f" {d.mag_max_b:.6g} | {d.delta_mag_max:+.3g} |"
                        )
                    lines.append("")
                if sd.added:
                    lines.append("### Added (in B only)")
                    lines.append("")
                    for name in sd.added:
                        lines.append(f"- `{name}`")
                    lines.append("")
                if sd.dropped:
                    lines.append("### Dropped (in A only)")
                    lines.append("")
                    for name in sd.dropped:
                        lines.append(f"- `{name}`")
                    lines.append("")
                lines.append("---")
                lines.append("")
            else:
                n = len(sd.changed) + len(sd.unchanged) + len(sd.added) + len(sd.dropped)
                lines.append("## Signal Comparison")
                lines.append("")
                lines.append(
                    f"**{n} signals compared — {len(sd.changed)} changed, "
                    f"{len(sd.added)} added, {len(sd.dropped)} dropped**"
                )
                lines.append("")
                if sd.changed:
                    lines.append("### Changed")
                    lines.append("")
                    lines.append(
                        f"| Signal | Mean {self.label_a} | Mean {self.label_b} | Δ Mean |"
                        f" Max {self.label_a} | Max {self.label_b} | Δ Max |"
                    )
                    lines.append("|---|---|---|---|---|---|---|")
                    for d in sd.changed:
                        lines.append(
                            f"| {d.name} | {d.mean_a:.4f} | {d.mean_b:.4f} | {d.delta_mean:+.4f} |"
                            f" {d.max_a:.4f} | {d.max_b:.4f} | {d.delta_max:+.4f} |"
                        )
                    lines.append("")
                if sd.added:
                    lines.append("### Added (in B only)")
                    lines.append("")
                    for name in sd.added:
                        lines.append(f"- `{name}`")
                    lines.append("")
                if sd.dropped:
                    lines.append("### Dropped (in A only)")
                    lines.append("")
                    for name in sd.dropped:
                        lines.append(f"- `{name}`")
                    lines.append("")
                lines.append("---")
                lines.append("")

        if self.topology_diff is not None:
            td = self.topology_diff
            lines.append("## Topology")
            lines.append("")
            if td.instances_added or td.instances_removed or td.nets_delta != 0:
                lines.append(f"**Net count delta:** {td.nets_delta:+d}")
                lines.append("")
                if td.instances_added:
                    lines.append(f"**Instances added:** {', '.join(td.instances_added)}")
                if td.instances_removed:
                    lines.append(f"**Instances removed:** {', '.join(td.instances_removed)}")
            else:
                lines.append("No topology changes.")
            lines.append("")
            lines.append("---")
            lines.append("")

        if self.annotation_diff is not None:
            ad = self.annotation_diff
            lines.append("## Annotation Coverage")
            lines.append("")
            lines.append(f"**{self.label_a}:** {ad.coverage_a:.1f}%  ")
            lines.append(f"**{self.label_b}:** {ad.coverage_b:.1f}%  ")
            lines.append(f"**Delta:** {ad.coverage_b - ad.coverage_a:+.1f}%")
            lines.append("")
            lines.append("---")
            lines.append("")

        if self.violation_diff is not None:
            vd = self.violation_diff
            lines.append("## Interface Violations")
            lines.append("")
            if vd.introduced:
                lines.append(f"### Introduced ({len(vd.introduced)})")
                lines.append("")
                net_only = all(isinstance(v.location, NetLocation) for v in vd.introduced)
                lines.append(f"| Severity | Kind | {'Net' if net_only else 'Location'} | Message |")
                lines.append("|---|---|---|---|")
                for v in vd.introduced:
                    location = v.location.net_name if net_only else v.location.describe()
                    lines.append(f"| {v.severity} | {v.rule_id} | {location} | {v.message} |")
                lines.append("")
            if vd.resolved:
                lines.append(f"### Resolved ({len(vd.resolved)})")
                lines.append("")
                net_only = all(isinstance(v.location, NetLocation) for v in vd.resolved)
                lines.append(f"| Severity | Kind | {'Net' if net_only else 'Location'} | Message |")
                lines.append("|---|---|---|---|")
                for v in vd.resolved:
                    location = v.location.net_name if net_only else v.location.describe()
                    lines.append(f"| {v.severity} | {v.rule_id} | {location} | {v.message} |")
                lines.append("")
            if not vd.introduced and not vd.resolved:
                lines.append(f"No violation changes ({len(vd.unchanged)} unchanged).")
            lines.append("")

        if self.layout_diff is not None:
            ld = self.layout_diff
            lines.append("## Layout Diff")
            lines.append("")
            wa, ha = ld.bbox_a
            wb, hb = ld.bbox_b
            lines.append(
                f"**Bounding box:** {wa:.2f}×{ha:.2f} μm → {wb:.2f}×{hb:.2f} μm  "
                f"(area Δ {ld.bbox_area_delta:+.1f} μm²)"
            )
            lines.append("")
            if ld.changed:
                lines.append(f"### Changed nets ({len(ld.changed)})")
                lines.append("")
                lines.append(
                    f"| Net | ΔR (Ω) | ΔC (fF) | ΔLength (μm) | ΔWidth (μm) |"
                )
                lines.append("|---|---|---|---|---|")
                for d in ld.changed:
                    lines.append(
                        f"| {d.name} | {d.delta_r:+.3f} | {d.delta_c:+.3f} "
                        f"| {d.delta_length:+.2f} | {d.delta_width:+.3f} |"
                    )
                lines.append("")
            if ld.nets_added:
                lines.append(f"### Nets added: {', '.join(f'`{n}`' for n in ld.nets_added)}")
                lines.append("")
            if ld.nets_dropped:
                lines.append(f"### Nets dropped: {', '.join(f'`{n}`' for n in ld.nets_dropped)}")
                lines.append("")
            if not ld.changed and not ld.nets_added and not ld.nets_dropped:
                lines.append(f"No layout changes ({len(ld.unchanged)} nets unchanged).")
            lines.append("")

        pd = self.provenance_diff
        if pd is not None and (pd.records_a or pd.records_b):
            lines.append("## Provenance")
            lines.append("")
            if pd.changed:
                lines.append(f"| Record | Field | {self.label_a} | {self.label_b} |")
                lines.append("|---|---|---|---|")
                for row in pd.changed:
                    lines.append(f"| {row['record']} | {row['field']} "
                                 f"| {row['a']!r} | {row['b']!r} |")
            else:
                lines.append("No provenance changes.")
            lines.append("")

        md = "\n".join(lines).rstrip()
        return md + "\n"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def diff_reports(
    a: Union['CircuitReport', str],
    b: Union['CircuitReport', str],
    label_a: str = 'a',
    label_b: str = 'b',
    sections: Optional[Set[str]] = None,
) -> ReportDiff:
    """Compare two CircuitReports and return a ReportDiff.

    Parameters
    ----------
    a, b : CircuitReport or str
        Reports to compare. If str, treated as a JSON file path and loaded
        via ``CircuitReport.load_json()``.
    label_a, label_b : str
        Human-readable names for the two reports (e.g. ``'schematic'``, ``'extracted'``).
    sections : set of str, optional
        Which sections to compare. Any subset of:
        ``{'signals', 'topology', 'annotation', 'violations',
        'provenance'}``. Defaults to all five. ``'provenance'`` lists every
        :class:`silk.simulation.provenance.RunProvenance` field that differs
        (``timestamp_utc`` excluded) in ``ReportDiff.provenance_diff``.

    Returns
    -------
    ReportDiff
        The comparison. ``layout_diff`` and ``spec_transitions`` are filled
        regardless of *sections*.
        Deltas are ``b - a`` in each quantity's own unit: volts or amps for
        transient and DC signals, μm for layout lengths and widths, percent
        for annotation coverage.

    Raises
    ------
    ValueError
        If a report path cannot be parsed, or its schema is unsupported
        (see :meth:`silk.circuit.CircuitReport.load_json`).
    OSError
        If a report path cannot be opened.
    silk.exceptions.ConfigurationError
        If either report contains duplicate ``(owner_path, spec.key)``
        reconciliation identities.

    Notes
    -----
    Unknown owners match only unknown owners; they never stand in for a
    known placement.

    """
    from silk.circuit.report import CircuitReport

    if isinstance(a, str):
        a = CircuitReport.load_json(a)
    if isinstance(b, str):
        b = CircuitReport.load_json(b)

    if sections is None:
        sections = {'signals', 'topology', 'annotation', 'violations',
                    'provenance'}

    diff = ReportDiff(
        label_a=label_a,
        label_b=label_b,
        generated_at=datetime.datetime.now().isoformat(timespec='seconds'),
        sections_compared=set(sections),
        signal_diff=_diff_signals(a, b)        if 'signals'    in sections else None,
        topology_diff=_diff_topology(a, b)     if 'topology'   in sections else None,
        annotation_diff=_diff_annotation(a, b) if 'annotation' in sections else None,
        violation_diff=_diff_violations(a, b)  if 'violations' in sections else None,
        layout_diff=_diff_layout(a, b),
    )
    diff.spec_transitions = _diff_spec_reconciliation(a, b)
    if 'provenance' in sections:
        diff.provenance_diff = _diff_provenance(a, b)
    return diff


def _diff_provenance(a, b) -> ProvenanceDiff:
    """Pair the two reports' provenance records by position and list every
    differing field."""
    recs_a = list(getattr(a, 'provenance', None) or [])
    recs_b = list(getattr(b, 'provenance', None) or [])
    changed = []
    for i in range(max(len(recs_a), len(recs_b))):
        ra = recs_a[i] if i < len(recs_a) else None
        rb = recs_b[i] if i < len(recs_b) else None
        if ra is None or rb is None:
            changed.append({'record': i, 'field': 'record',
                            'a': None if ra is None else ra.digest(),
                            'b': None if rb is None else rb.digest()})
            continue
        for name, va, vb in ra.diff(rb):
            changed.append({'record': i, 'field': name, 'a': va, 'b': vb})
    return ProvenanceDiff(records_a=len(recs_a), records_b=len(recs_b),
                          changed=changed)


def _spec_transition_label(transition: dict) -> str:
    """One placement-qualified label for both diff text renderers."""
    from silk.specs.report import _single_line
    owner = transition.get('owner_path')
    placement = 'unknown placement' if owner is None else owner
    return _single_line(
        f"{transition['spec_name']} [{placement}; key={transition.get('spec_key')}]")


def _diff_spec_reconciliation(a, b) -> List[dict]:
    """Join unique placement/spec identities; refuse ambiguous input rows."""
    from silk.exceptions import ConfigurationError

    def _collect(report, side) -> dict:
        state: dict = {}
        recon = getattr(report, 'spec_reconciliation', None)
        if recon is None:
            return state
        for position, outcome in enumerate(recon.outcomes, 1):
            key = (outcome.owner_path, outcome.spec.key)
            if key in state:
                prior, _, prior_position = state[key]
                owner = 'unknown' if key[0] is None else repr(key[0])
                raise ConfigurationError(
                    f"Ambiguous {side} reconciliation: duplicate identity "
                    f"owner={owner}, spec key={key[1]!r}; "
                    f"row {prior_position} ({prior.spec.name!r}, "
                    f"report_key={prior.report_key!r}) and row {position} "
                    f"({outcome.spec.name!r}, report_key={outcome.report_key!r}). "
                    "Supply distinct owner paths/spec keys or remove an unintended "
                    "duplicate; regenerate legacy reports with placement identity.")
            state[key] = (outcome, {
                'status': outcome.status,
                'value': (outcome.measurement.value
                          if outcome.measurement is not None else None),
                'source_used': outcome.source_used,
                'latest_attempt': outcome.latest_attempt.as_dict() if outcome.latest_attempt else None,
                'last_successful_attempt': outcome.last_successful_attempt.as_dict() if outcome.last_successful_attempt else None,
            }, position)
        return state

    before = _collect(a, 'before')
    after = _collect(b, 'after')
    identities = sorted(before.keys() | after.keys(),
                        key=lambda key: (key[0] is not None,
                                         key[0] if key[0] is not None else '', key[1]))
    transitions: List[dict] = []
    for owner, spec_key in identities:
        key = (owner, spec_key)
        bs = before[key][1] if key in before else None
        as_ = after[key][1] if key in after else None
        if bs == as_:
            continue
        outcome = (after[key] if key in after else before[key])[0]
        transitions.append({
            'spec_name': outcome.spec.name,
            'owner_path': owner,
            'spec_key': spec_key,
            'report_key': outcome.report_key,
            'before_status': bs['status'] if bs else None,
            'after_status':  as_['status'] if as_ else None,
            'before_value':  bs['value'] if bs else None,
            'after_value':   as_['value'] if as_ else None,
            'source_used':   (as_ or bs)['source_used'],
        })
    return transitions


# ---------------------------------------------------------------------------
# Private diff builders
# ---------------------------------------------------------------------------

def _diff_layout(a, b) -> Optional[LayoutDiff]:
    """Compute LayoutDiff between two reports. Returns None if either has no layout."""
    le_a = getattr(a, 'layout_estimate', None)
    le_b = getattr(b, 'layout_estimate', None)
    if le_a is None or le_b is None:
        return None

    nets_a = le_a.nets
    nets_b = le_b.nets
    names_a = set(nets_a)
    names_b = set(nets_b)

    changed:   List[LayoutNetDelta] = []
    unchanged: List[str] = []

    _EPS = 1e-12
    for name in sorted(names_a & names_b):
        na, nb = nets_a[name], nets_b[name]
        if (abs(na.r - nb.r) < _EPS and abs(na.c - nb.c) < _EPS
                and abs(na.length - nb.length) < _EPS
                and abs(na.width - nb.width) < _EPS):
            unchanged.append(name)
        else:
            changed.append(LayoutNetDelta(
                name=name,
                length_a=na.length, length_b=nb.length,
                width_a=na.width,   width_b=nb.width,
                r_a=na.r,           r_b=nb.r,
                c_a=na.c,           c_b=nb.c,
            ))

    return LayoutDiff(
        changed=changed,
        unchanged=unchanged,
        nets_added=sorted(names_b - names_a),
        nets_dropped=sorted(names_a - names_b),
        bbox_a=le_a.bounding_box,
        bbox_b=le_b.bounding_box,
    )


def _diff_signals(a, b) -> Optional[Union[SignalDiff, MagnitudeDiff, KindMismatch]]:
    """Diff a run's per-node statistics, wherever ``SimulationSummary`` keeps
    them for its analysis kind (Silk split it: ``.signals`` for
    transient/dc, ``.magnitudes`` for ac/noise — the two never both hold
    entries).

    Returns ``None`` when either report has no simulation at all (unchanged
    from before). Returns a :class:`KindMismatch` when the two
    simulations ran different analyses — comparing a transient run's
    voltages against an AC run's transfer-function magnitudes node-for-node
    is meaningless, so this is reported as a structural difference rather
    than an empty diff. Otherwise returns a :class:`SignalDiff` (matching
    transient/dc kinds, unchanged shape) or a :class:`MagnitudeDiff`
    (matching ac/noise kinds).

    Previously, this walked only ``.signals``: moved
    AC/noise statistics to ``.magnitudes``, diffing two AC (or two noise)
    reports always produced an empty ``SignalDiff`` — ``changed``,
    ``added`` and ``dropped`` all ``[]`` — so a before/after AC
    characterization that DID change silently rendered as "Signals:
    unchanged". (Previously, it instead diffed the real-part garbage
    then stored under ``.signals``, which was wrong in a different way.)
    """
    if a.simulation is None or b.simulation is None:
        return None

    kind_a = a.simulation.analysis
    kind_b = b.simulation.analysis
    if kind_a != kind_b:
        return KindMismatch(analysis_a=kind_a, analysis_b=kind_b)

    if kind_a in SMALL_SIGNAL_KINDS:
        return _diff_magnitudes(a.simulation.magnitudes, b.simulation.magnitudes)

    stats_a = {s.name: s for s in a.simulation.signals}
    stats_b = {s.name: s for s in b.simulation.signals}
    names_a = set(stats_a)
    names_b = set(stats_b)

    changed:   List[SignalDelta] = []
    unchanged: List[str]         = []

    for name in sorted(names_a & names_b):
        sa, sb = stats_a[name], stats_b[name]
        if sa.min == sb.min and sa.max == sb.max and sa.mean == sb.mean:
            unchanged.append(name)
        else:
            changed.append(SignalDelta(
                name=name,
                min_a=sa.min,   min_b=sb.min,
                max_a=sa.max,   max_b=sb.max,
                mean_a=sa.mean, mean_b=sb.mean,
            ))

    return SignalDiff(
        changed=changed,
        unchanged=unchanged,
        added=sorted(names_b - names_a),
        dropped=sorted(names_a - names_b),
    )


def _diff_magnitudes(mags_a, mags_b) -> MagnitudeDiff:
    """Diff two ``SimulationSummary.magnitudes`` lists (AC/noise runs) by
    node name, the same present/changed/added/dropped shape as
    :func:`_diff_signals`'s transient/dc branch but over
    mag_min/mag_max/mag_mean rather than min/max/mean."""
    stats_a = {m.name: m for m in mags_a}
    stats_b = {m.name: m for m in mags_b}
    names_a = set(stats_a)
    names_b = set(stats_b)

    changed:   List[MagnitudeDelta] = []
    unchanged: List[str]            = []

    for name in sorted(names_a & names_b):
        ma, mb = stats_a[name], stats_b[name]
        if (ma.mag_min == mb.mag_min and ma.mag_max == mb.mag_max
                and ma.mag_mean == mb.mag_mean):
            unchanged.append(name)
        else:
            changed.append(MagnitudeDelta(
                name=name,
                mag_min_a=ma.mag_min,   mag_min_b=mb.mag_min,
                mag_max_a=ma.mag_max,   mag_max_b=mb.mag_max,
                mag_mean_a=ma.mag_mean, mag_mean_b=mb.mag_mean,
            ))

    return MagnitudeDiff(
        changed=changed,
        unchanged=unchanged,
        added=sorted(names_b - names_a),
        dropped=sorted(names_a - names_b),
    )


def _diff_topology(a, b) -> TopologyDiff:
    inst_a = {i.instance for i in a.topology.instances}
    inst_b = {i.instance for i in b.topology.instances}
    return TopologyDiff(
        instances_added=sorted(inst_b - inst_a),
        instances_removed=sorted(inst_a - inst_b),
        nets_delta=b.topology.total_nets - a.topology.total_nets,
    )


def _diff_annotation(a, b) -> AnnotationDiff:
    unanno_a = {f"{u.instance}.{u.port}" for u in a.annotation.unannotated_ports}
    unanno_b = {f"{u.instance}.{u.port}" for u in b.annotation.unannotated_ports}
    return AnnotationDiff(
        coverage_a=a.annotation.coverage_pct,
        coverage_b=b.annotation.coverage_pct,
        ports_resolved=sorted(unanno_a - unanno_b),
        ports_added=sorted(unanno_b - unanno_a),
    )


def _diff_violations(a, b) -> ViolationDiff:
    def _key(v):
        return (v.rule_id, v.location.describe())

    # Keep one entry per rule/location identity, ignoring message and other details.
    viol_a = {_key(v): v for v in a.violations}
    viol_b = {_key(v): v for v in b.violations}
    keys_a, keys_b = set(viol_a), set(viol_b)

    return ViolationDiff(
        resolved=[viol_a[k]   for k in sorted(keys_a - keys_b)],
        introduced=[viol_b[k] for k in sorted(keys_b - keys_a)],
        unchanged=[viol_a[k]  for k in sorted(keys_a & keys_b)],
    )
