"""silk.circuit.spec_report — SpecReport: structured pass/fail spec checking.

Usage::

    report = testbench.run(corner='tt')
    print(report.summary())
    assert report.passed
"""
import json
import csv
import math
from copy import deepcopy
import os
from dataclasses import dataclass
from numbers import Complex, Real
from typing import Any, NamedTuple

from silk._json_codec import encode_json, decode_json
from silk.exceptions import NotSupportedError, SimulationError
from silk.circuit.metric_handle import MetricValues, _metric_key
from silk.metrics.diagnostics import MetricWarningRecord
from silk.utils._utils import _actual_values


_OPS = {
    '>':  lambda v, t: v > t,
    '>=': lambda v, t: v >= t,
    '<':  lambda v, t: v < t,
    '<=': lambda v, t: v <= t,
    '==': lambda v, t: v == t,
}


@dataclass
class SpecResult:
    """Result of a single spec check.

    Parameters
    ----------
    name : str
        The spec's name.
    metric_name : str
        Name of the metric that was checked.
    op : {'<', '<=', '>', '>=', '=='}
        Comparison applied as ``value <op> target``.
    target : float
        Threshold, in the metric's units.
    value : float or complex or str
        The measured value, in the metric's units: a real or complex
        measurement, or the error text when extraction failed.
    passed : bool
        Whether the check passed. A complex or error value fails.
    spec_key : str, optional
        The spec's stable key (defaults to its name).
    owner_path : str, optional
        Canonical path of the module that declared the spec.
    """
    name: str
    metric_name: str
    op: str
    target: float
    value: Any       # actual measured value; real/complex measurement, str on extraction error
    passed: bool
    spec_key: str | None = None
    owner_path: str | None = None


def _report_observation(result, report):
    """Validate a captured observation against its report and declaration."""
    if not hasattr(result, '_observation'):
        return None
    captured = result._observation
    from silk.specs.attempt import SpecAttempt
    if not isinstance(captured, tuple) or len(captured) != 6:
        raise ValueError('Malformed report observation')
    owner, key, metric, op, target, attempt = captured
    if ((owner, key, metric, op) != (result.owner_path, result.spec_key,
                                    result.metric_name, result.op)
            or not isinstance(target, Real) or isinstance(target, bool)
            or float(target).hex() != float(result.target).hex()
            or not isinstance(attempt, SpecAttempt)):
        raise ValueError('Report observation conflicts with spec identity')
    m = attempt.measurement
    metric_value = report.metrics.get(result.metric_name)
    if m is None:
        from silk.specs.attempt import classify_metric
        kind, _, _ = classify_metric(
            metric_value, metric=result.metric_name, metric_errors=report.metric_errors,
            metric_warnings=report.metric_warnings)
        matches = (attempt.kind == kind
                   and result.passed is False
                   and (repr(result.value) == repr(metric_value)))
    else:
        if (isinstance(m.timestamp, bool) or not isinstance(m.timestamp, Real)
                or not isinstance(m.notes, str)
                or any(item is not None and not isinstance(item, str)
                       for item in (m.sim_run_id, m.sim_type, m.tool_name))):
            raise ValueError('Malformed report observation metadata')
        matches = not (m.aggregate is not None
            or not isinstance(result.value, Real) or isinstance(result.value, bool)
            or not isinstance(metric_value, Real) or isinstance(metric_value, bool)
            or float(metric_value).hex() != float(result.value).hex()
            or float(m.value).hex() != float(result.value).hex()
            or m.passes != result.passed
            or bool(_OPS[result.op](result.value, result.target)) != result.passed)
    if (not matches or attempt.test_name not in (None, report.test_name)
            or attempt.report_key not in (None, report.report_key)):
        raise ValueError('Report observation conflicts with result evidence')
    if (attempt.proved_conditions is not None
            and (report.operating_conditions is None
                 or attempt.proved_conditions != report.operating_conditions.to_dict())):
        raise ValueError('Report observation conflicts with operating conditions')
    conditions = report.operating_conditions
    if conditions is not None:
        source = 'behavioral' if conditions.backend == 'behavioral' else 'spice'
        if (attempt.source != source or (m is not None and any(
                getattr(m, name) != conditions.scalar(fact)
                for name, fact in (('corner', 'corner'), ('temperature', 'temperature'),
                                   ('voltage', 'supply'))))):
            raise ValueError('Report observation conflicts with source/conditions')
    return attempt


def _real_metric_values(metric: str, vals: dict) -> dict:
    # Every caller of this reduction helper requires real scalars. Keep
    # raw report metrics intact for tabulation and lossless export.
    for corner, value in vals.items():
        if isinstance(value, Complex) and not isinstance(value, Real):
            raise ValueError(
                f"metric {metric!r} at {corner!r}: complex values require "
                "a real projection for reduction; choose result.real or abs(result).")
    return vals


class SpecReport:
    """Structured pass/fail report from a ModuleTestbench run.

    Created by testbench evaluation, or restored offline with from_dict/load_json.

    Parameters
    ----------
    test_name : str
        Name of the test (testbench) that produced the report.
    module_name : str
        Name of the module under test.
    metrics : dict
        Extracted metric values ``{name: value}``.
    owner_path : str, default '$'
        Canonical path of the owning module.
    parameters : dict, optional
        This run's declared test parameters (values, never Variables).
    characterization_context : CharacterizationContext, optional
        Runtime descriptor of temperature, corner and supply; ``None`` marks
        the report unscoped.
    corner : str, optional
        The run's corner label.
    operating_conditions : OperatingConditions, optional
        Immutable snapshot of the run's ambient conditions.
    rawfile_path : str or os.PathLike, optional
        Rawfile the metrics were extracted from; stored as a string.
    metric_warnings : dict, optional
        ``{output name: (MetricWarningRecord, ...)}``.
    op_devices : list of dict, optional
        Testbench-wide operating-point inventory captured at evaluate time.
    op_unavailable_reason : str, optional
        Why ``op_devices`` is absent, shown by :meth:`op_table`.
    provenance : RunProvenance, optional
        What produced the report (versions, simulator, deck hash, seed).
    behavioral_elements : tuple of str, optional
        Ignored loads and wired series elements of a behavioral run.
    report_key : str, optional
        Snapshotted runner identity.
    metric_types : dict, optional
        Output name to catalog type (``None`` for custom outputs).
    metric_errors : dict, optional
        Extraction failures by output.

    Attributes
    ----------
    test_name : str
    module_name : str
    owner_path : str
        Canonical root/instance-path of the live module that owns this test
        (``$``, ``$/amp1``, ...) — the ``owner_path`` half of
        characterization identity, distinct from ``module_name``. Defaults to
        ``'$'`` when not supplied.
    parameters : dict
        This run's declared test-parameter map (``{}`` if none were
        declared). Metadata only. A Variable or Variable expression is
        recorded as its value when the report was built, never the object.
    characterization_context : CharacterizationContext or None
        The full runtime descriptor derived from known fixed temperature,
        corner and declared supply (or external-file declarations).
        ``None`` means this report is explicitly unscoped.
    operating_conditions : OperatingConditions or None
        Immutable snapshot of ambient TEMP and its basis, origin, backend,
        corner and caller-declared primary supply. Unknown/sweep temperature
        remains non-scalar. Historical/manual reports may omit the snapshot.
    corner : str or None
        The run's corner label, which may differ from the physical process
        corner. ``operating_conditions`` is authoritative for the physical
        corner and whether it is none or unknown; external evidence may have
        an unknown corner. ``None`` is also possible on a hand-built report.
        See :meth:`store_report <silk.db.CircuitTracker.store_report>` for how
        corner evidence is stored.
    rawfile_path : str or None
        The rawfile these metrics were extracted from, as the path the run
        (or ``evaluate(rawfile=...)``) actually read -- ``Module.
        results_file`` at extraction time. Behavioral runs also
        write a rawfile. ``None`` when no rawfile path was recorded,
        including on a hand-built report without one. This is what makes a
        corner's waveforms findable after
        ``run_corners()`` has moved on to the next corner; the testbench
        keeps the same paths in ``sweep_corners``.
    metrics : dict
        All extracted metric values {name: value}, preserving complex scalars.
        Specs bound to complex values fail; :meth:`summary` explains the
        required ``.real`` or ``abs`` projection. Legacy real-only consumers
        refuse complex values at their own boundaries. Warnings are unchanged.
    report_key : str or None
        Snapshotted runner identity, including duplicate-aware owner paths.
        None for a manual report without execution identity.
    metric_types : dict
        Output to catalog type, or None for custom/derived/unknown outputs.
    metric_errors : dict
        Extraction exceptions by output: message, exception_type, and the
        loaded available_signals list (None when unavailable).
    metric_warnings : dict
        ``{output name: (MetricWarningRecord, ...)}`` -- every
        ``silk.metrics.diagnostics.MetricWarning`` raised while that
        output's extractor ran, attributed by the output's own name (not
        the metric type), so two outputs sharing one metric type keep
        separate warnings. Only outputs that actually warned appear as
        keys; a clean report has an empty dict. Warnings that are not
        ``MetricWarning`` are never recorded here.
    behavioral_elements : tuple of str
        Ignored loads and wired series elements, snapshotted from this run.
        Included in summary(), including after a behavioral rawfile reload.
    storage : dict or None
        The storage receipt from the most recent
        ``Module.run_test(..., store=True)`` that persisted this report:
        ``store_report()``'s ``metrics_stored`` / ``specs_stored`` /
        ``skipped`` plus ``cached`` (True when ``run()`` returned this report
        from its cache instead of simulating again). ``None`` until such a
        store succeeds -- including after a failed one (see
        ``silk.exceptions.ReportStorageError``) and on every report that
        was only ever stored through an explicit ``store_report(report)``
        call, which leaves this attribute alone.
    passed : bool
        True if all checked specs passed (or no specs were checked).
    results : list of SpecResult
        Individual spec check results.
    op_devices : list of dict or None
        the plain-data (JSON/pickle-safe) per-device
        operating-point inventory of the DUT and every helper placement,
        with names and emitted paths rooted at the testbench. Captured by
        ``silk.circuit.modules_results.capture_op_device_records``
        at evaluate() time, when ``save_device_params`` was set on
        the device under test. ``op_table()`` builds only from this stored
        data -- never a live ``Module`` reference, never a re-read of the
        rawfile (reports are pickled across the parallel test executor, and the rawfile can be gone by the time ``op_table()`` is
        called). ``None`` when the switch was off, or when the run's
        backend does not support operating-point capture. Xyce capture is
        limited to device parameters declared and verified for that backend.
    provenance : RunProvenance or None
        what produced this report: silk/Python/numpy versions,
        who ran the simulator (``run_origin``) and which binary, the PDK
        ``.lib`` file and corner section the prepared deck named (with
        hashes), options, save list, prepared-deck sha256, seed and a UTC
        timestamp. See ``silk.simulation.provenance`` for the field contract.
        Set on every testbench-built report; ``None`` on a hand-built one.
    run : object or None
        a ``RunRecord`` (``silk.circuit.RunRecord``): the files of the simulator launch that produced this
        report: ``deck_path`` (the prepared deck), ``log_path``,
        ``rawfile_path``, ``noise_path`` and ``directory``, all absolute, plus
        ``launch_id``. ``None`` for a cache hit, an external rawfile or a
        hand-built report. ``to_dict``/``from_dict`` carry it as the optional
        ``run`` and ``launch_id`` keys (absent when there is no record; the
        wire version stays 1, and files without them load with ``run=None``). The files are
        kept in a fresh folder by default for ordinary runs. Explicit
        ``set_output_options(mode='overwrite')`` replaces them on the next
        run into the same folder.
    launch_id : str or None
        ``run.launch_id``: equal for every report that shares one simulator
        launch (placements deduplicated by ``run_all_tests``), different for
        separate launches.

    Examples
    --------
    ::

        report = tb.run(corner='tt')
        print(report.summary())
        if not report.passed:
            for r in report.results:
                if not r.passed:
                    print(f"FAIL: {r.name} — got {r.value}, need {r.op} {r.target}")
    """

    def __init__(
        self,
        test_name: str,
        module_name: str,
        metrics: dict,
        *,
        owner_path: str = '$',
        parameters: dict | None = None,
        characterization_context: Any = None,
        corner: str | None = None,
        operating_conditions: Any = None,
        rawfile_path: str | None = None,
        metric_warnings: dict | None = None,
        op_devices: list | None = None,
        op_unavailable_reason: str | None = None,
        provenance: Any = None,
        behavioral_elements: tuple = (),
        report_key: str | None = None,
        metric_types: dict | None = None,
        metric_errors: dict | None = None,
    ):
        self.report_key = report_key
        self.metric_types = {key: (metric_types or {}).get(key) for key in metrics}
        self.metric_errors = {} if metric_errors is None else metric_errors
        self.test_name   = test_name
        self.module_name = module_name
        self.metrics = MetricValues(metrics)
        self.metric_warnings = (
            {} if metric_warnings is None else dict(metric_warnings)
        )
        self.owner_path  = owner_path
        # The values the run used, never a live Variable (TASK-402).
        self.parameters  = (_actual_values(dict(parameters))
                            if parameters is not None else {})
        self.characterization_context = characterization_context
        self.operating_conditions = operating_conditions
        self.corner = None if corner is None else str(corner)
        self.rawfile_path = None if rawfile_path is None else os.fspath(rawfile_path)
        self.storage = None
        self.op_devices = op_devices
        self._op_unavailable_reason = op_unavailable_reason
        self.provenance = provenance
        self.behavioral_elements = tuple(behavioral_elements)
        self._results: list[SpecResult] = []

    def to_dict(self) -> dict:
        """Return a fresh version-1 strict JSON wire record of all evidence.

        Nonfinite and complex numbers use numeric tags. No live circuit,
        database, or rawfile is consulted. See docs/reports.md for the schema.

        Returns
        -------
        dict
            A JSON-serialisable dict (version 1 wire format); the report is
            not modified.
        """
        from silk.circuit._spec_report_json import report_data
        return encode_json(report_data(self), complex_values=True)

    def save_json(self, path: str | os.PathLike) -> None:
        """Save standalone evidence; validate before opening the destination.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file (UTF-8, indented JSON); overwritten.
        """
        from silk.circuit._spec_report_json import report_data
        text = json.dumps(encode_json(report_data(self), complex_values=True),
                          indent=2, allow_nan=False)
        with open(path, 'w', encoding='utf-8') as stream:
            stream.write(text + '\n')

    @classmethod
    def from_dict(cls, payload: dict) -> 'SpecReport':
        """Load version-1 wire data without rerunning checks or opening paths.

        Malformed or unsupported records raise ValueError naming the field.
        Unknown additive keys are ignored.

        Parameters
        ----------
        payload : dict
            A record produced by :meth:`to_dict`, possibly after a JSON
            round trip.

        Returns
        -------
        SpecReport
            The restored report; stored values are used as they are.

        Raises
        ------
        ValueError
            If the record is malformed or of an unsupported version.
        """
        from silk.circuit._spec_report_json import plain_data, restore_report
        data = decode_json(plain_data(payload, wire=True), complex_values=True)
        return restore_report(data, cls)

    @classmethod
    def load_json(cls, path: str | os.PathLike) -> 'SpecReport':
        """Read standalone evidence offline; errors identify file and field.

        Parameters
        ----------
        path : str or os.PathLike
            A file written by :meth:`save_json`.

        Returns
        -------
        SpecReport
            The restored report.

        Raises
        ------
        ValueError
            If the file is not valid JSON or the record is malformed; the
            message names the file and the field.
        OSError
            If the file cannot be read.
        """
        try:
            with open(path, encoding='utf-8') as stream:
                payload = json.load(stream)
            return cls.from_dict(payload)
        except ValueError as exc:
            raise ValueError(f'{path}: {exc}') from exc

    def op_table(self, region_thresholds=None):
        """Return the captured DUT and helper operating-point table.

        Parameters
        ----------
        region_thresholds : RegionThresholds, optional
            Overrides user-file and shipped per-PDK policy. ``None`` resolves
            SILK_REGION_THRESHOLDS for the captured provenance PDK (or the
            explicit default entry without PDK identity). A thermal policy
            uses the report's temperature, falling back to 27 C when unknown;
            an explicitly absent temperature requires a fixed-volts policy
            or an explicit override. Malformed policy raises
            RegionThresholdsError naming the file and key.

        Returns
        -------
        silk.circuit.op_table.OpTable

        Raises
        ------
        ValueError
            The report's temperature is none and the resolved policy needs kT/q.
        NotSupportedError
            A behavioral run has no electrical operating-point solve.
        SimulationError
            ``save_device_params`` was off for the run that produced this
            report, or its backend does not support operating-point
            capture -- the specific reason is named.
        """
        if getattr(self.operating_conditions, 'backend', None) == 'behavioral':
            raise NotSupportedError(
                "op_table() is unavailable for a behavioral run: operating-point "
                "device parameters require an electrical solve. Use a SPICE backend.")
        if self.op_devices is None:
            raise SimulationError(
                self._op_unavailable_reason
                or "op_table() has no operating-point data for this report. "
                   "Set save_device_params = True on the device under test "
                   "before run()/evaluate()."
            )
        from silk.circuit.op_table import build_op_table
        return build_op_table(
            self.op_devices, region_thresholds, self.operating_conditions,
            pdk_name=getattr(self.provenance, 'pdk_name', None))

    def check(
        self,
        name: str,
        metric_name: str,
        op: str,
        target: float,
        *,
        spec_key: str | None = None,
        owner_path: str | None = None,
    ) -> 'SpecReport':
        """Evaluate one spec against the measured metrics.

        Complex values are preserved but always fail; ``summary()`` names
        the required ``.real`` or ``abs`` projection, without emitting a
        metric warning or attempting a lossy real cast.

        Parameters
        ----------
        name : str
            Human-readable spec name, e.g. ``'DC gain > 60 dB'``.
        metric_name : str
            Key into self.metrics dict.
        op : str
            Comparison operator: ``'>'``, ``'>='``, ``'<'``, ``'<='``, ``'=='``.
        target : float
            Target value.
        spec_key : str, optional
            Stable declaration key used for persistence matching.
        owner_path : str, optional
            Canonical owner path used for persistence matching.

        Returns
        -------
        SpecReport
            self, for chaining.
        """
        value = self.metrics.get(metric_name)
        if (value is None or isinstance(value, str)
                or (isinstance(value, Complex) and not isinstance(value, Real))):
            passed = False
        else:
            fn = _OPS.get(op)
            if fn is None:
                raise ValueError(f"Unknown operator {op!r}. Valid operators: {list(_OPS)}")
            try:
                passed = fn(float(value), float(target))
            except (TypeError, ValueError):
                passed = False
        self._results.append(
            SpecResult(
                name,
                metric_name,
                op,
                target,
                value,
                passed,
                spec_key=spec_key,
                owner_path=owner_path,
            )
        )
        return self

    @property
    def passed(self) -> bool:
        """True if all checked specs passed (or no specs were checked)."""
        return all(r.passed for r in self._results)

    @property
    def results(self) -> list:
        """List of SpecResult objects, one per check() call."""
        return list(self._results)

    #: the launch record; ``None`` unless a testbench run set it.
    run = None

    @property
    def launch_id(self):
        """``run.launch_id``, or ``None`` when no launch is known."""
        return None if self.run is None else self.run.launch_id

    def summary(self) -> str:
        """Return a human-readable pass/fail table string."""
        status = 'PASS' if self.passed else 'FAIL'
        lines = [
            f"SpecReport [{self.test_name}] — {status}",
            f"  Module: {self.module_name}",
            "",
        ]
        if self.operating_conditions is not None:
            c = self.operating_conditions
            lines.insert(2, f'  Conditions: {c.origin}, {c.backend}; '
                         f'corner {_fact_text(c, "corner")}; '
                         f'TEMP {_fact_text(c, "temperature")}; '
                         f'supply {_fact_text(c, "supply")}')
        provenance = getattr(self, 'provenance', None)
        if provenance is not None:
            lines.insert(len(lines) - 1,
                         f'  Run details: {provenance.summary_line()}')
        for note in getattr(self, 'behavioral_elements', ()):
            lines.append(f'  Behavioral approximation: {note}')
        if not self._results:
            lines.append("  (no specs checked)")
        else:
            for r in self._results:
                mark = 'PASS' if r.passed else 'FAIL'
                lines.append(
                    f"  [{mark}]  {r.name}: "
                    f"{r.metric_name} {r.op} {r.target} "
                    f"(measured: {r.value!r})"
                )
                if isinstance(r.value, Complex) and not isinstance(r.value, Real):
                    lines.append(
                        f"        {r.metric_name}: complex values cannot be compared "
                        "with spec limits; choose result.real or abs(result).")
                unavailable = isinstance(r.value, float) and r.value != r.value
                for rec in self.metric_warnings.get(r.metric_name, ()):
                    label = 'unavailable' if unavailable else 'note'
                    lines.append(f"        {label}: {metric_warning_text(rec)}")
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            f"SpecReport(test={self.test_name!r}, "
            f"passed={self.passed}, "
            f"specs={len(self._results)})"
        )


def metric_warning_text(rec) -> str:
    """One ``MetricWarningRecord`` as a single line of plain text.

    ``code: message (k=v, ...) suggestion``. Control characters (newlines,
    ANSI escapes) are replaced by spaces so a hostile string cannot break a
    terminal line; HTML callers escape the result at their own boundary.
    """
    text = f'{rec.code}: {rec.message}'
    if rec.details:
        text += ' (' + ', '.join(f'{k}={v!r}' for k, v in rec.details.items()) + ')'
    if rec.suggestion:
        text += f' {rec.suggestion}'
    return ''.join(' ' if (ord(c) < 32 or 0x7f <= ord(c) < 0xa0) else c
                   for c in text)


def _fact_text(c, fact: str) -> str:
    """One condition fact as summary text: the value with its basis, ``none``
    with its basis as the reason, ``sweep`` or ``unknown`` spelled out."""
    kind = c.state(fact).value
    basis = getattr(c, f'{fact}_basis', None)
    scalar = c.scalar(fact)
    if kind == 'fixed':
        text = (f'{scalar:g} C' if fact == 'temperature'
                else f'{scalar:g} V' if fact == 'supply' else str(scalar))
        return f'{text} ({basis})' if basis and basis != 'unknown' else text
    if kind in ('none', 'sweep') and basis and basis != 'unknown':
        return f'{kind} ({basis})'
    return kind


class WorstCase(NamedTuple):
    """``CornerResults.worst(metric)`` result: value and its corner.

    Parameters
    ----------
    value : float
        The worst corner's metric value, in the metric's units; ``nan`` when
        unavailable.
    corner : str or tuple or None
        Corner string, or full (corner, temperature, supply) key for a grid;
        ``None`` when no run has a usable value.
    """
    value: float
    corner: str | tuple | None


class CornerResults:
    """Collection of SpecReports from a multi-corner run.

    Parameters
    ----------
    reports : dict
        Run key (a corner name) to :class:`SpecReport`, in run order.
    failed : dict, optional
        Run key to formatted error for each run that produced no report.
    skip_unavailable : bool, default False
        When True, :meth:`stat` and :meth:`worst` skip failed and
        unavailable runs with one ``MetricWarning`` instead of returning
        ``nan``.

    Attributes
    ----------
    reports : dict of str to SpecReport
        Mapping of corner name → SpecReport.
    failed : dict of str to str
        Corner → formatted error for each run that produced no report.
        Always ``{}`` for ``run_corners()`` results (that call raises
        ``CornerRunError`` instead); filled for a declared corner list
        run by ``run()``/``run_all_tests`` (failure handling).
    passed : bool
        True only if no run failed and every corner passed all its specs.

    Examples
    --------
    ::

        results = tb.run_corners()
        if not results.passed:
            for corner, report in results.items():
                if not report.passed:
                    print(f"{corner}: {report.summary()}")
    """

    def __init__(self, reports: dict, *, failed: dict | None = None,
                 skip_unavailable: bool = False, run_inventory=None,
                 metric_names=None, check_declarations=None):
        """
        Parameters
        ----------
        reports : dict
            Run key (a corner name) → :class:`SpecReport`, in run order.
        failed : dict, optional
            Run key → formatted error for each run that produced no report. Those keys are not in *reports*. ``None`` means none.
        skip_unavailable : bool
            ``False`` (the default, and what ``run_corners()`` builds):
            :meth:`stat` and :meth:`worst` return ``nan`` when any run's value
            is unavailable. ``True`` (results built from declared settings,
            run aggregation): they skip failed runs and unavailable values
            and say how many with one ``MetricWarning``.
        """
        self.reports = dict(reports)
        self.failed = {} if failed is None else dict(failed)
        self._skip_unavailable = bool(skip_unavailable)
        self._run_inventory = None if run_inventory is None else deepcopy(tuple(run_inventory))
        self._table_metric_names = tuple(dict.fromkeys(
            metric_names if metric_names is not None else
            (m for report in self.reports.values() for m in report.metrics)))
        if check_declarations is None:
            checks = {}
            for report in self.reports.values():
                for result in report.results:
                    key = result.spec_key or result.name
                    checks.setdefault(key, dict(key=key, name=result.name,
                        owner_path=result.owner_path, metric=result.metric_name,
                        op=result.op, target=result.target))
            check_declarations = checks.values()
        self._check_declarations = deepcopy(tuple(check_declarations))


    @property
    def passed(self) -> bool:
        """True if every run produced a report and every report passed."""
        if self._run_inventory is not None:
            return bool(self._run_inventory) and not self.failed and all(
                row['run.status'] == 'PASS' and all(
                    row['check.' + declaration['key']] == 'PASS'
                    for declaration in self._check_declarations)
                for row in self.table())
        return not self.failed and all(r.passed for r in self.reports.values())

    def summary(self) -> str:
        """One-line per corner pass/fail table; failed runs are counted and
        listed after it.

        Returns
        -------
        str
            A header with the overall verdict, one line per corner, then the
            failed runs with the first line of each error.
        """
        if self._is_grid():
            return self._grid_summary()
        overall = 'PASS' if self.passed else 'FAIL'
        lines = [f'CornerResults — {overall}']
        for corner, report in self.reports.items():
            mark = 'PASS' if report.passed else 'FAIL'
            lines.append(f'  [{mark}]  corner={corner}')
        if self.failed:
            total = len(self.reports) + len(self.failed)
            lines.append(f'  {len(self.failed)} of {total} runs failed and '
                         'produced no report:')
            for key, error in self.failed.items():
                first = str(error).splitlines()[0] if error else ''
                lines.append(f'  [FAILED]  corner={key}: {first}')
        return '\n'.join(lines)

    def _ordered_keys(self):
        if self._run_inventory is not None:
            return [run['key'] for run in self._run_inventory]
        return list(dict.fromkeys((*self.reports, *self.failed)))

    def _point_key(self, key):
        return key

    def _is_grid(self):
        return any(isinstance(self._point_key(key), tuple)
                   for key in self._ordered_keys())

    @staticmethod
    def _check_cell(result):
        if result is None:
            return 'UNAVAILABLE'
        value = result.value
        if isinstance(value, (str, bool)) or not isinstance(value, Real):
            return 'UNAVAILABLE'
        if math.isnan(float(value)):
            return 'UNAVAILABLE'
        return 'PASS' if result.passed else 'FAIL'

    def _variable_names(self):
        for key in self._ordered_keys():
            point = self._point_key(key)
            if isinstance(point, tuple) and len(point) == 4:
                return [name for name, value in point[3]]
        return []

    def _plan_summary(self):
        groups = next((run.get('groups') for run in self._run_inventory or () if run.get('groups')), ())
        if not groups:
            return None
        rows = self.table()
        labels = ['corner', 'temperature', 'supply', *self._variable_names()]
        grouped = {label for group in groups for label in group}
        factors = list(groups) + [[label] for label in labels if label not in grouped]
        factors.sort(key=lambda group: min(labels.index(label) for label in group))
        text = []
        for group in factors:
            count = len({tuple(row['run.' + label] for label in group) for row in rows})
            if count > 1 or group in groups:
                text.append(f'{count} ' + ('paired ' if len(group) > 1 else '')
                            + '(' + ', '.join(group) + ') rows')
        count = len({self._point_key(key) for key in self._ordered_keys()})
        return f'{count} planned points: ' + ' × '.join(text)

    def table(self) -> list:
        """Ordered per-run rows, namespaced coordinates, evidence and checks.

        Observations come only from captured OperatingConditions. A run with
        no report has blank observations and UNAVAILABLE checks. Values are
        retained without coercing structured or complex metric outputs.
        """
        inventory = {run['key']: run for run in self._run_inventory or ()}
        rows = []
        for key in self._ordered_keys():
            point = self._point_key(key)
            requested = inventory.get(key, {})
            coords = point if isinstance(point, tuple) else (point, None, None)
            row = {'run.corner': requested.get('corner', coords[0]),
                   'run.temperature': requested.get('temperature', coords[1]),
                   'run.supply': requested.get('supply', coords[2]),
                   'run.supply_source': requested.get('supply_source')}
            if isinstance(point, tuple) and len(point) == 4:
                row.update({'run.' + name: value for name, value in point[3]})
            if hasattr(self, 'seeds'):
                row.update({'run.sample': key[1], 'run.seed': self.seeds.get(key)})
            report = self.reports.get(key)
            conditions = getattr(report, 'operating_conditions', None)
            for fact, value_field in (('corner', 'corner'), ('temperature', 'temperature_c'),
                                      ('supply', 'supply_voltage_v')):
                row['observed.' + fact] = getattr(conditions, value_field, None)
                row['observed.' + fact + '_state'] = getattr(conditions, fact + '_kind', None)
                row['observed.' + fact + '_basis'] = getattr(conditions, fact + '_basis', None)
                if fact != 'corner':
                    sweep = getattr(conditions, fact + '_sweep', None)
                    for field in ('start', 'stop', 'step'):
                        row['observed.' + fact + '_sweep.' + field] = getattr(sweep, field, None)
            row['observed.supply_source'] = getattr(conditions, 'supply_source', None)
            row['run.status'] = ('PASS' if report.passed else 'FAIL') if report else 'failed'
            failure = self.failed.get(key)
            row['run.error'] = getattr(failure, 'error', failure)
            row.update({'metric.' + name: report.metrics.get(name) if report else None
                        for name in self._table_metric_names})
            checks = {r.spec_key or r.name: r for r in report.results} if report else {}
            row.update({'check.' + declaration['key']: self._check_cell(checks.get(declaration['key']))
                        for declaration in self._check_declarations})
            rows.append(row)
        return rows

    def _table_columns(self):
        columns = ['run.corner', 'run.temperature', 'run.supply', 'run.supply_source']
        columns += ['run.' + name for name in self._variable_names()]
        if hasattr(self, 'seeds'):
            columns += ['run.sample', 'run.seed']
        for fact in ('corner', 'temperature', 'supply'):
            columns += ['observed.' + fact, 'observed.' + fact + '_state',
                        'observed.' + fact + '_basis']
            if fact != 'corner':
                columns += ['observed.' + fact + '_sweep.' + field
                            for field in ('start', 'stop', 'step')]
        return columns + ['observed.supply_source', 'run.status', 'run.error'] + [
            'metric.' + name for name in self._table_metric_names] + [
            'check.' + declaration['key'] for declaration in self._check_declarations]

    def to_csv(self, path) -> None:
        """Write the namespaced per-run table, including failed runs."""
        rows = self.table()
        with open(path, 'w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=self._table_columns(),
                                    lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)

    def _counts(self):
        keys = self._ordered_keys()
        reports = [self.reports[key] for key in keys if key in self.reports]
        passed = sum(report.passed for report in reports)
        return (f'planned={len(keys)}: {len(reports)} reports '
                f'({passed} PASS, {len(reports) - passed} FAIL), '
                f'{len(keys) - len(reports)} failed and produced no report')

    @staticmethod
    def _point_text(point):
        if isinstance(point, tuple):
            def display(value):
                text = repr(float(value)) if isinstance(value, Real) else str(value)
                return text[:-2] if text.endswith('.0') else text
            return (f'corner={point[0]} temperature={display(point[1])} '
                    f'supply={display(point[2])}'
                    + (''.join(f' {name}={value!r}' for name, value in point[3]) if len(point) == 4 else ''))
        return f'corner={point}'

    def _grid_summary(self):
        lines = [f"{type(self).__name__} — {'PASS' if self.passed else 'FAIL'}",
                 '  ' + self._counts()]
        if self._plan_summary():
            lines.append('  ' + self._plan_summary())
        lines.extend(self._metric_summary())
        for key in self._ordered_keys():
            if key not in self.reports:
                error = self.failed.get(key, 'no report')
                error = getattr(error, 'error', error)
                first = (str(error).splitlines() or [''])[0]
                lines.append(f'  [failed] {key!r}: {first}')
        return '\n'.join(lines)

    def _metric_summary(self, *, aggregate=False):
        lines = []
        for name in self._table_metric_names:
            raw = [report.metrics.get(name) for report in self.reports.values()]
            if any(value is not None and not isinstance(value, (Real, str)) for value in raw):
                lines.append(f'  {name}: not summarized (non-real or structured output)')
                continue
            used = [float(value) for value in raw if isinstance(value, Real) and not math.isnan(float(value))]
            prefix = 'aggregate ' if aggregate else ''
            if not used:
                lines.append(f'  {prefix}{name}: used=0 unavailable={len(self._ordered_keys())}')
                continue
            text = (f'  {prefix}{name}: used={len(used)} '
                    f'unavailable={len(self._ordered_keys()) - len(used)} '
                    f"min={self.stat(name, 'min'):g} max={self.stat(name, 'max'):g}")
            if aggregate:
                text += f" mean={self.stat(name, 'mean'):g} std={self.stat(name, 'std'):g}"
            try:
                worst = self.worst(name)
            except ValueError:
                text += ' worst=unavailable (no declared check)'
            else:
                text += f' worst={worst.value:g} at {self._point_text(worst.corner)}'
                if hasattr(worst, 'sample'):
                    text += f' sample={worst.sample}'
            lines.append(text)
        return lines

    #: Valid ``how`` values for :meth:`stat`.
    STAT_KINDS = ('mean', 'rms', 'std', 'min', 'max')

    def _metric_values(self, metric: str) -> dict:
        known = sorted({m for r in self.reports.values() for m in r.metrics})
        if metric not in known:
            raise ValueError(
                f"unknown metric {metric!r}. Valid: {known}")
        vals = {c: r.metrics.get(metric) for c, r in self.reports.items()}
        return _real_metric_values(metric, vals)

    @staticmethod
    def _unavailable(vals: dict) -> list:
        return [c for c, v in vals.items()
                if v is None or isinstance(v, (str, bool)) or v != v]

    @staticmethod
    def _warn_nan(metric: str, what: str, message: str, corners: list):
        from silk.metrics.diagnostics import emit
        emit(metric=metric, code='stat_unavailable_corners',
             message=f"{what} is nan: {message}",
             details={'corners': corners},
             suggestion="Inspect those corners' reports.")

    def _usable_values(self, metric: str, what: str, vals: dict) -> dict:
        """*vals* without unavailable values, after one counted
        ``MetricWarning`` naming the failed runs and the skipped values
        (``skip_unavailable`` results only)."""
        bad = self._unavailable(vals)
        if self.failed or bad:
            from silk.metrics.diagnostics import emit
            total = len(self.reports) + len(self.failed)
            parts = []
            if self.failed:
                parts.append(f"{len(self.failed)} of {total} runs failed to "
                             f"simulate ({', '.join(map(repr, self.failed))})")
            if bad:
                parts.append(f"{len(bad)} of {total} runs gave nan or no "
                             f"value for {metric} "
                             f"({', '.join(map(repr, bad))})")
            emit(metric=metric, code='stat_skipped_runs',
                 message=f"{what} skips {'; '.join(parts)}",
                 details={'failed': list(self.failed), 'unavailable': bad,
                          'runs': total},
                 suggestion="Inspect .failed and those runs' reports.")
        return {c: v for c, v in vals.items() if c not in bad}

    def stat(self, metric, how: str) -> float:
        """Aggregate one scalar metric across every run in this collection.

        Parameters
        ----------
        metric : str or MetricHandle
            A metric name (or registered handle) present in the reports' ``metrics``.
        how : str
            ``'mean'``, ``'rms'`` (root mean square), ``'std'`` (sample
            standard deviation, ``ddof=1``),
            ``'min'`` or ``'max'``.

        Returns
        -------
        float
            Always a float. ``nan`` (with one ``silk.metrics.diagnostics.MetricWarning``
            naming the corners) when any run's
            value is a failed metric (string), missing or ``nan``, and for
            ``'std'`` with fewer than two runs. Results built from declared
            settings (``set_corners``; ``skip_unavailable=True``) instead skip
            failed runs and unavailable values, with one ``MetricWarning``
            (``code='stat_skipped_runs'``) giving both counts and the keys;
            ``nan`` only when no usable value is left.

        Raises
        ------
        ValueError
            Unknown ``metric`` or ``how``, or complex metric values requiring
            ``.real`` or ``abs`` before real-valued reduction.
        """
        metric = _metric_key(metric)
        if how not in self.STAT_KINDS:
            raise ValueError(
                f"stat(): unknown statistic {how!r}. "
                f"Valid: {list(self.STAT_KINDS)}")
        try:
            vals = self._metric_values(metric)
        except ValueError as e:
            raise ValueError(f"stat(): {e}") from None
        what = f'stat({metric!r}, {how!r})'
        if self._skip_unavailable:
            vals = self._usable_values(metric, what, vals)
            if not vals:
                return float('nan')
        bad = self._unavailable(vals)
        if bad:
            self._warn_nan(
                metric, what,
                f"the metric is failed, missing or nan in {len(bad)} run(s)",
                bad)
            return float('nan')
        import numpy as np
        a = np.array([float(v) for v in vals.values()])
        if how == 'std':
            if len(a) < 2:
                self._warn_nan(metric, what,
                               "the sample standard deviation needs at "
                               f"least 2 runs, got {len(a)}",
                               list(vals))
                return float('nan')
            return float(a.std(ddof=1))
        if how == 'mean':
            return float(a.mean())
        if how == 'rms':
            return float(np.sqrt(np.mean(a * a)))
        return float(a.min() if how == 'min' else a.max())

    def worst(self, metric) -> 'WorstCase':
        """The run closest to failing (or furthest past) a spec on ``metric``.

        Ranked by signed margin to the spec target (``>``/``>=``: value -
        target; ``<``/``<=``: target - value; ``==``: -``|value - target|``);
        with several specs on one metric the smallest margin over all of
        them wins.

        Returns
        -------
        WorstCase
            ``(value, corner)``. ``WorstCase(nan, None)`` plus a
            ``silk.metrics.diagnostics.MetricWarning`` naming the
            corners when any run's value is failed, missing or ``nan``.
            With ``skip_unavailable=True`` those runs (and the runs in
            :attr:`failed`) are skipped with one counted ``MetricWarning``
            instead, as in :meth:`stat`.

        Raises
        ------
        ValueError
            Unknown ``metric`` (valid names listed), or no report checks a
            spec on it (there is no failing direction to rank by), or complex
            values requiring ``.real`` or ``abs`` before ranking.
        """
        metric = _metric_key(metric)
        try:
            vals = self._metric_values(metric)
        except ValueError as e:
            raise ValueError(f"worst(): {e}") from None
        specs = [(c, res) for c, r in self.reports.items()
                 for res in r.results if res.metric_name == metric]
        if not specs:
            raise ValueError(
                f"worst({metric!r}): no spec checks this metric, so there "
                f"is no failing direction to rank by. Use stat() with "
                f"'min' or 'max', or declare a spec on it.")
        if self._skip_unavailable:
            vals = self._usable_values(metric, f'worst({metric!r})', vals)
            specs = [(c, res) for c, res in specs if c in vals]
            if not specs:
                return WorstCase(float('nan'), None)
        bad = self._unavailable(vals)
        if bad:
            self._warn_nan(
                metric, f'worst({metric!r})',
                f"the metric is failed, missing or nan in {len(bad)} run(s)",
                bad)
            return WorstCase(float('nan'), None)

        def margin(cr):
            c, res = cr
            v, t = float(vals[c]), float(res.target)
            if res.op in ('>', '>='):
                return v - t
            if res.op in ('<', '<='):
                return t - v
            return -abs(v - t)
        c, _ = min(specs, key=margin)
        return WorstCase(float(vals[c]), c)

    def __repr__(self) -> str:
        failed = f', failed={list(self.failed)}' if self.failed else ''
        return (f'CornerResults(corners={list(self.reports)}{failed}, '
                f'passed={self.passed})')

    def __getitem__(self, corner: str) -> 'SpecReport':
        return self.reports[corner]

    def __iter__(self):
        return iter(self.reports)

    def items(self):
        """Corner/report pairs, as for a dict.

        Returns
        -------
        collections.abc.ItemsView
            ``(corner, SpecReport)`` pairs in run order; failed runs are not
            included.
        """
        return self.reports.items()

    def keys(self):
        """Corner names, as for a dict.

        Returns
        -------
        collections.abc.KeysView
            The corners that produced a report, in run order.
        """
        return self.reports.keys()

    def values(self):
        """Reports, as for a dict.

        Returns
        -------
        collections.abc.ValuesView
            The :class:`SpecReport` of each corner, in run order.
        """
        return self.reports.values()


class CornerRunError(SimulationError):
    """One or more corners of a :meth:`~silk.circuit.ModuleTestbench.run_corners` sweep failed.

    Every corner is still attempted — a failing corner never stops the
    others from running, simulating and evaluating — so ``results`` is a
    full :class:`CornerResults` for every corner that succeeded, and
    ``errors`` maps each corner that did not to the formatted exception its
    run raised. Raised only after every corner has been attempted, so a
    caller that catches this can still read the successful corners' reports
    from ``.results``.

    Attributes
    ----------
    errors : dict of str to str
        ``{corner name: formatted error}`` for every corner that failed.
    results : CornerResults
        Reports from every corner that succeeded (never includes a failed
        corner's key).
    """

    def __init__(self, errors: dict, results: 'CornerResults'):
        self.errors = dict(errors)
        self.results = results
        corners = ', '.join(repr(str(c)) for c in self.errors)
        detail = '\n'.join(f'  {c}: {e}' for c, e in self.errors.items())
        super().__init__(
            f"run_corners(): {len(self.errors)} corner(s) failed "
            f"({corners}); the rest completed — see .results/.errors.\n{detail}"
        )


def results_to_dict(results) -> dict:
    """Export tests, corner/grid or Monte Carlo collections, or a record list.

    Exact keys, ordering, failures, inventories and collection policy are
    retained. Each node uses version 1, 2 or 3 as its evidence requires; a
    version-1/2 tests container may have version-3 grid/Monte Carlo children.
    Custom containers are unsupported. No tests are run.
    """
    from silk.circuit._results_json import results_data
    return encode_json(results_data(results), complex_values=True)


def save_results_json(results, path: str | os.PathLike) -> None:
    """Save a collection, validating before opening the destination."""
    from silk.circuit._results_json import results_data
    text = json.dumps(encode_json(results_data(results), complex_values=True),
                      indent=2, allow_nan=False)
    with open(path, 'w', encoding='utf-8') as stream:
        stream.write(text + '\n')


def results_from_dict(payload: dict):
    """Restore detached tests, corner/grid, Monte Carlo or record collections.

    Versions 1, 2 and 3 are read per collection node, including version-3
    children in version-1/2 tests containers. No bench or rawfile is needed.
    """
    from silk.circuit._results_json import restore_results
    from silk.circuit._spec_report_json import plain_data
    return restore_results(decode_json(plain_data(payload, wire=True), complex_values=True))


def load_results_json(path: str | os.PathLike):
    """Load collection evidence offline; invalid data names the file and field."""
    try:
        with open(path, encoding='utf-8') as stream:
            return results_from_dict(json.load(stream))
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
