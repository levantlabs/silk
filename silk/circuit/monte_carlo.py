"""Monte Carlo settings, seeds, results and sample decks.

Pure maths over :class:`~silk.circuit.spec_report.SpecReport` objects, plus
the per-sample deck description: a sample is a private ``_mc_sample``
attribute on the netlist Module; the netlister reads it to pick the
statistical library section and to seed the simulator. Nothing here
simulates. The run wiring builds a :class:`MonteCarloResults` from the
reports its samples produced.
"""

from __future__ import annotations

import csv
import math
import os
import time
import uuid
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

from silk.circuit.metric_handle import _metric_key
from silk.circuit.spec_report import CornerResults, _real_metric_values
from silk.exceptions import MonteCarloError
from silk.faults.campaign import _derive_run_seed

if TYPE_CHECKING:
    from silk.pdk.statistical import StatisticalModels

VARIATIONS = ('mismatch', 'process', 'both')


@dataclass(frozen=True)
class MonteCarloSettings:
    """Validated Monte Carlo request; raises :class:`MonteCarloError <silk.exceptions.MonteCarloError>`.

    Parameters
    ----------
    n : int
        Number of samples per corner; a positive ``int`` (not a ``bool``).
    variation : {'mismatch', 'process', 'both'}, default 'mismatch'
        Which statistical variation to draw.
    seed : int, default 0
        Base seed; each sample's simulator seed is derived from it and the
        sample index, so a run is reproducible.
    record_draws : bool, default False
        Record every sample's drawn model parameters (needed by
        :meth:`silk.circuit.MonteCarloResults.sensitivity`).
    correlate : tuple, default ``()``
        Operating-point quantities to correlate with the metric; converted
        to a tuple.
    min_yield : float, optional
        Required yield per spec, in (0, 1]; ``None`` requires every sample
        to pass.
    output_mode : {'overwrite', 'new'}, default 'overwrite'
        Overwrite campaign folders or keep a new folder per corner per campaign,
        independently of the module's output policy. Invalid values raise ValueError.
    samples_per_process : int, default 1
        At most this many consecutive samples of one corner share one ngspice
        process; a positive ``int`` (not a ``bool``). It changes how a
        campaign runs, not what it computes.

    Raises
    ------
    MonteCarloError
        If a sampling value is out of range or of the wrong type.
    ValueError
        If output_mode is neither 'overwrite' nor 'new'.
    """
    n: int
    variation: str = 'mismatch'
    seed: int = 0
    record_draws: bool = False
    correlate: tuple = ()
    min_yield: float | None = None
    #: At most this many consecutive samples of one corner run in one ngspice
    #: process; 1 (the default) is one process per sample. It changes how a
    #: campaign runs, not what it computes.
    samples_per_process: int = 1
    output_mode: str = 'overwrite'

    def __post_init__(self):
        if not isinstance(self.output_mode, str) or self.output_mode not in ('overwrite', 'new'):
            raise ValueError(
                f"output_mode must be 'overwrite' or 'new'; got {self.output_mode!r}")
        def is_int(v):
            return isinstance(v, int) and not isinstance(v, bool)
        if not is_int(self.n) or self.n < 1:
            raise MonteCarloError(
                f"Monte Carlo n must be a positive int, got {self.n!r}")
        if not is_int(self.samples_per_process) or self.samples_per_process < 1:
            raise MonteCarloError(
                f"Monte Carlo samples_per_process must be a positive int, "
                f"got {self.samples_per_process!r}")
        if self.variation not in VARIATIONS:
            raise MonteCarloError(
                f"Monte Carlo variation must be one of {list(VARIATIONS)}, "
                f"got {self.variation!r}")
        if not is_int(self.seed):
            raise MonteCarloError(
                f"Monte Carlo seed must be an int, got {self.seed!r}")
        if not isinstance(self.record_draws, bool):
            raise MonteCarloError(
                f"Monte Carlo record_draws must be a bool, "
                f"got {self.record_draws!r}")
        object.__setattr__(self, 'correlate', tuple(self.correlate))
        if self.min_yield is not None:
            ok = (isinstance(self.min_yield, (int, float))
                  and not isinstance(self.min_yield, bool)
                  and 0 < self.min_yield <= 1)
            if not ok:
                raise MonteCarloError(
                    f"Monte Carlo min_yield must be in (0, 1], "
                    f"got {self.min_yield!r}")


def derive_sample_seed(base_seed: int, index: int) -> int:
    """Positive 31-bit simulator seed for sample ``index``; reproducible.

    Parameters
    ----------
    base_seed : int
        The run's base seed (``MonteCarloSettings.seed``).
    index : int
        Sample index.

    Returns
    -------
    int
        Seed in ``1 .. 2**31 - 1``; the same inputs always give the same
        seed.
    """
    return 1 + _derive_run_seed(base_seed, index) % (2 ** 31 - 1)


def wilson_interval(k: int, n: int,
                    z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (95% default).

    ``n == 0`` has no information: ``(nan, nan)``. ``k`` outside ``0..n``
    raises :class:`MonteCarloError <silk.exceptions.MonteCarloError>`.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    z : float, default 1.959963984540054
        Normal quantile of the interval (the default is two-sided 95%).

    Returns
    -------
    tuple of float
        ``(low, high)`` bounds of the success fraction, both within [0, 1];
        ``(nan, nan)`` when ``n == 0``.

    Raises
    ------
    MonteCarloError
        If ``k`` is not within ``0..n``.
    """
    if not (0 <= k <= n):
        raise MonteCarloError(
            f"wilson_interval needs 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


class SampleFailure(NamedTuple):
    """A sample that produced no report: the error text and its seed."""
    error: str
    seed: int


class SampleWorstCase(NamedTuple):
    """``MonteCarloResults.worst(metric)``: value, corner, sample index.

    Parameters
    ----------
    value : float
        The metric value of the worst usable sample, in the metric's units;
        ``nan`` when there is none.
    corner : str or tuple or None
        Corner string, or full (corner, temperature, supply) key for a grid;
        ``None`` when there is no sample.
    sample : int or None
        Sample index within the corner; ``None`` when there is no sample.
    """
    value: float
    corner: str | tuple | None
    sample: int | None


def _usable(v) -> bool:
    return not (v is None or isinstance(v, (str, bool)) or v != v)


class _SpecDeclaration(NamedTuple):
    """A check's definition, independent of any sample's measured value."""
    name: str
    metric: str
    op: str
    target: float


def _spec_inventory(reports, declarations):
    """Snapshot declarations; reject contradictory definitions from any source."""
    known = {}

    def comparison_key(spec):
        # NumPy mixed-scalar equality can round an integer to a float.
        # Compare exact Python values without changing the stored targets.
        target = spec.target
        if isinstance(target, np.integer):
            target = int(target)
        elif isinstance(target, np.floating) and np.isfinite(target):
            target = Fraction(*target.as_integer_ratio())
        return (*spec[:3], target)

    def add(spec):
        previous = known.get(spec.name)
        if previous is not None and comparison_key(previous) != comparison_key(spec):
            raise ValueError(
                f"spec {spec.name!r} has conflicting definitions: "
                f"{previous!r} and {spec!r}; give different checks distinct "
                "names or make their metric, op and target agree.")
        known[spec.name] = spec

    if declarations is not None:
        for d in declarations:
            add(_SpecDeclaration(d['name'], _metric_key(d['metric']),
                                 d['op'], d['target']))
        declared = tuple(known.values())
    for report in reports.values():
        for res in report.results:
            add(_SpecDeclaration(res.name, res.metric_name, res.op,
                                 res.target))
    return tuple(known.values()) if declarations is None else declared


class MonteCarloResults(CornerResults):
    """Monte Carlo reports keyed ``(point, sample index)``.

    A point is a corner string without new axes, or a full immutable
    ``(corner, temperature, supply)`` tuple for a declared operating grid.

    Failed samples are kept (``failed``), counted against yield, and skipped
    by :meth:`stat`, :meth:`percentile` and :meth:`worst` with one
    :class:`silk.metrics.diagnostics.MetricWarning` giving both counts.

    Parameters
    ----------
    reports : dict
        Successful samples' :class:`~silk.circuit.SpecReport`, keyed
        ``(corner, sample index)``.
    settings : MonteCarloSettings
        The request that produced the run.
    seeds : dict
        Simulator seed of each sample, keyed like ``reports``.
    failed : dict, optional
        Samples that produced no report (``SampleFailure`` records), same keys.
    structure_hash : str, optional
        Structure hash of the circuit that was run.
    rerun_hook : callable, optional
        ``rerun_hook(key, seed)`` returning a fresh report; enables
        :meth:`rerun`.
    draws : object, optional
        Recorded drawn parameters; ``None`` unless ``record_draws`` was set.
    metric_names : iterable of str, optional
        Registered outputs, including dotted fields; inferred from the
        reports when omitted.
    spec_declarations : iterable of dict, optional
        Declared checks with ``name``, ``metric``, ``op`` and ``target``;
        inferred from the reports when omitted.
    campaign_source : str, optional
        Fallback source attribution (``'behavioral'``, ``'spice'``,
        ``'extraction'`` or ``'user'``) for :meth:`measurement`.
    draw_scheme : optional
        Identifier of the behavioural draw scheme, kept in aggregate evidence.
    _campaign_id : str, optional
        Internal: campaign identity carried to corner views; generated when
        omitted.
    _conditions : tuple, optional
        Internal: captured per-corner conditions carried to corner views;
        derived from the reports when omitted.

    Raises
    ------
    ValueError
        If two sources give conflicting definitions for one spec name.
    """

    def __init__(self, reports: dict, *, settings: MonteCarloSettings,
                 seeds: dict, failed: dict | None = None,
                 structure_hash: str | None = None, rerun_hook=None,
                 draws=None, metric_names=None, spec_declarations=None, campaign_source=None, draw_scheme=None,
                 rerun_samples=(), _campaign_id=None, _conditions=None,
                 run_inventory=None, check_declarations=None):
        """Retain a campaign inventory independently of its observations.

        ``metric_names`` lists registered outputs (including dotted fields).
        ``spec_declarations`` contains mappings with ``name``, ``metric``,
        ``op`` and ``target``. Both are copied at construction. Conflicting
        definitions for one spec name raise ``ValueError`` naming both.
        Omitted inventories are inferred once from the supplied reports for
        compatibility with hand-built results; an all-failed manual root
        needs explicit declarations. Corner views retain these snapshots.
        """
        super().__init__(reports, failed=failed, run_inventory=run_inventory,
                         metric_names=metric_names, check_declarations=check_declarations)
        self._spec_declarations = _spec_inventory(self.reports, spec_declarations)
        if metric_names is None:
            metric_names = (m for r in self.reports.values() for m in r.metrics)
        self._metric_names = tuple(sorted(
            {_metric_key(m) for m in metric_names}
            | {s.metric for s in self._spec_declarations}))
        if run_inventory is not None:
            self._metric_names = tuple(dict.fromkeys((*self._table_metric_names, *self._metric_names)))
        self._table_metric_names = self._metric_names
        observed_names = {declaration['name'] for declaration in self._check_declarations}
        self._check_declarations += tuple(dict(key=d.name, name=d.name,
            metric=d.metric, op=d.op, target=d.target) for d in self._spec_declarations
            if d.name not in observed_names)
        self.settings = settings
        self.seeds = dict(seeds)
        self.failed = dict(failed or {})
        #: ``(corner, index)`` keys of samples that did not finish in a
        #: multi-sample process and that silk then ran one per process
        #: (``samples_per_process`` > 1), including retries that failed:
        #: a crashed or timed-out process left them unfinished, or their
        #: deck could not join one (a fallback before any process launched,
        #: its reason in the run's warning); empty when nothing was rerun.
        self.rerun_samples = tuple(rerun_samples)
        self.structure_hash = structure_hash
        self._rerun_hook = rerun_hook
        #: Each sample's drawn model parameters
        #: (:class:`~silk.circuit.monte_carlo_draws.DrawTable`), or ``None``
        #: when the run did not set ``record_draws=True``.
        self.draws = draws
        self._campaign_id = _campaign_id or str(uuid.uuid4())
        self._campaign_source = campaign_source
        self._draw_scheme = draw_scheme
        # Capture report evidence once, never consult live simulator settings.
        if _conditions is None:
            conditions = {}
            for (corner, _), report in self.reports.items():
                snapshot = getattr(report, 'operating_conditions', None)
                backend = None if snapshot is None else snapshot.backend
                source = ('behavioral' if backend == 'behavioral' else
                          'spice' if backend in ('ngspice', 'xyce') else None)
                values = tuple(snapshot.scalar(f) for f in ('corner', 'temperature', 'supply')) if snapshot else (None, None, None)
                provenance = getattr(report, 'provenance', None)
                tool = getattr(provenance, 'simulator_version', None) or backend
                record = (source, *values, backend, tool)
                records = conditions.setdefault(corner, [])
                if record not in records:
                    records.append(record)
            self._conditions = tuple((corner, tuple(records))
                                     for corner, records in conditions.items())
        else:
            self._conditions = tuple(_conditions)


    # -- selection -------------------------------------------------------
    def corner(self, name: str) -> 'MonteCarloResults':
        """The samples of one corner as a MonteCarloResults.

        Parameters
        ----------
        name : str
            Corner name.

        Returns
        -------
        MonteCarloResults
            A new results object holding only that corner's samples,
            failures, seeds and draws.

        Raises
        ------
        MonteCarloError
            If no sample (successful or failed) ran at that corner.
        """
        keys = [key for key in self._keys() if self._physical_corner(key[0]) == name]
        if not keys:
            raise MonteCarloError(f'no samples for corner {name!r}. Corners: {self.corners}')
        return self._view(keys)

    @staticmethod
    def _physical_corner(point):
        return point[0] if isinstance(point, tuple) else point

    def _point_key(self, key):
        return key[0]

    def _points(self):
        return list(dict.fromkeys(key[0] for key in self._keys()))

    def point(self, key) -> 'MonteCarloResults':
        """Select one full operating point, including failed samples.

        Corner-only collections use a corner string; grids use a tuple of
        physical corner, requested Celsius and requested supply volts.
        """
        keys = [sample for sample in self._keys() if sample[0] == key]
        if not keys:
            raise MonteCarloError(f'Unknown point {key!r}. Available points: {self._points()!r}')
        return self._view(keys)

    def _view(self, keys):
        selected = set(keys)
        def sub(data):
            return {key: data[key] for key in keys if key in data}
        return MonteCarloResults(
            sub(self.reports), settings=self.settings, seeds=sub(self.seeds),
            failed=sub(self.failed), structure_hash=self.structure_hash,
            rerun_hook=self._rerun_hook,
            campaign_source=self._campaign_source, draw_scheme=self._draw_scheme,
            _campaign_id=self._campaign_id, _conditions=self._conditions,
            rerun_samples=[k for k in self.rerun_samples if k in selected],
            metric_names=self._metric_names,
            spec_declarations=[d._asdict() for d in self._spec_declarations],
            run_inventory=None if self._run_inventory is None else
                [run for run in self._run_inventory if run['key'] in selected],
            check_declarations=self._check_declarations,
            draws=None if self.draws is None else replace(self.draws, values=sub(self.draws.values)))

    @property
    def corners(self) -> list:
        """Physical process corner strings in run order, failed points included."""
        if self._run_inventory is not None:
            return list(dict.fromkeys(self._physical_corner(k[0]) for k in self._keys()))
        order = dict.fromkeys(self._physical_corner(k[0]) for k in self.seeds)
        order.update(dict.fromkeys(self._physical_corner(k[0]) for k in self._keys()))
        return list(order)

    def _keys(self) -> list:
        if self._run_inventory is not None:
            return [run['key'] for run in self._run_inventory]
        return sorted({*self.reports, *self.failed})

    def _ordered_keys(self):
        return self._keys()

    def _metric_values(self, metric: str) -> dict:
        if metric not in self._metric_names:
            raise ValueError(
                f"unknown metric {metric!r}. Valid: {list(self._metric_names)}")
        vals = {k: r.metrics.get(metric) for k, r in self.reports.items()}
        return _real_metric_values(metric, vals)

    def _spec_limits(self, metric: str) -> tuple:
        """Distinct declared (operator, target) limits in declaration order."""
        return tuple(dict.fromkeys((s.op, s.target)
                                   for s in self._spec_declarations
                                   if s.metric == metric))

    def _spec_pass_counts(self) -> dict:
        """Count successes against the retained inventory, including zeros."""
        passes = dict.fromkeys((s.name for s in self._spec_declarations), 0)
        for report in self.reports.values():
            for res in report.results:
                if res.name in passes:
                    passes[res.name] += self._check_cell(res) == 'PASS'
        return passes

    # -- skipping --------------------------------------------------------
    def _used(self, metric: str, what: str, need: int = 0) -> dict:
        """{key: float} of usable values; warns once with both counts."""
        metric = _metric_key(metric)
        try:
            vals = self._metric_values(metric)
        except ValueError as e:
            raise ValueError(f"{what}: {e}") from None
        used = {k: float(v) for k, v in vals.items() if _usable(v)}
        total = len(self.reports) + len(self.failed)
        n_nan = len(vals) - len(used)
        short = len(used) < need
        if self.failed or n_nan or short or not used:
            from silk.metrics.diagnostics import emit
            emit(metric=metric, code='sample_skipped',
                 message=(f"{what}: {len(self.failed)} of {total} samples "
                          f"failed and {n_nan} gave nan or no value; "
                          f"{len(used)} used"
                          + ("; every sample failed" if total and
                             len(self.failed) == total else "")
                          + (f" (need at least {need})" if short else "")),
                 details={'failed': len(self.failed), 'unavailable': n_nan,
                          'total': total, 'used': len(used)},
                 suggestion="See .failed and summary().")
        return used

    # -- statistics ------------------------------------------------------
    def stat(self, metric: str, how: str) -> float:
        """Aggregate usable samples; no observations gives nan and MetricWarning.

        Unknown metric/statistic names raise ValueError. ``std`` also needs
        at least two usable observations.

        Parameters
        ----------
        metric : str
            Registered metric name.
        how : {'mean', 'rms', 'std', 'min', 'max'}
            Statistic; ``std`` is the sample standard deviation (ddof=1).

        Returns
        -------
        float
            The statistic over usable samples, in the metric's units; ``nan``
            when there are none (or fewer than two for ``std``).

        Raises
        ------
        ValueError
            If ``metric`` or ``how`` is unknown.
        """
        return self._statistic(metric, how)[0]

    def _statistic(self, metric, how):
        """The shared reduction and its exact used-sample inventory."""
        if how not in self.STAT_KINDS:
            raise ValueError(f"stat(): unknown statistic {how!r}. "
                             f"Valid: {list(self.STAT_KINDS)}")
        metric = _metric_key(metric)
        used = self._used(metric, f'stat({metric!r}, {how!r})',
                          need=2 if how == 'std' else 0)
        a = np.array(list(used.values()))
        if a.size == 0 or (how == 'std' and a.size < 2):
            return math.nan, used
        if how == 'std':
            return float(a.std(ddof=1)), used
        if how == 'mean':
            return float(a.mean()), used
        if how == 'rms':
            return float(np.sqrt(np.mean(a * a))), used
        return float(a.min() if how == 'min' else a.max()), used

    def measurement(self, metric, how='std', *, corner=None):
        """Return an explicitly attributed statistic for one full operating point.

        A physical corner with multiple grid points requires point(...)
        selection, even if surviving reports have equal observed conditions.

        Insufficient usable data returns nan with the stat() diagnostic.
        Missing source attribution or an ambiguous corner raises MonteCarloError.
        This creates evidence; it neither records a Spec nor writes a database.

        Parameters
        ----------
        metric : str
            Registered metric name.
        how : {'mean', 'rms', 'std', 'min', 'max'}, default 'std'
            Statistic, as :meth:`stat`.
        corner : str, optional
            Physical corner to select. The selected collection must contain
            exactly one full point; use point(...) for a grid.

        Returns
        -------
        Measurement
            Value of the statistic with an attached aggregate record (sample
            counts, seeds, campaign id), attributed to the captured source.

        Raises
        ------
        MonteCarloError
            If several corners ran and none was named, the source attribution
            is missing or inconsistent, or the seeds do not cover the
            selected samples.
        ValueError
            If ``metric`` or ``how`` is unknown.
        """
        from silk.specs import AggregateMeasurement, Measurement
        selected = self if corner is None else self.corner(corner)
        points = selected._points()
        if len(points) > 1:
            raise MonteCarloError('measurement() needs one selected corner/point; use .point(...)')
        point = points[0] if points else None
        label = self._physical_corner(point)
        records = dict(self._conditions).get(point, ())
        if len(records) > 1:
            raise MonteCarloError(f'inconsistent campaign source or conditions at corner {label!r}')
        record = records[0] if records else (None, None, None, None, None, None)
        source = record[0] or self._campaign_source
        if source not in ('behavioral', 'spice', 'extraction', 'user'):
            raise MonteCarloError('measurement() has no captured campaign source attribution; '
                                  'use results from a recorded campaign or provide campaign_source at construction')
        if record[0] is not None and self._campaign_source is not None and source != self._campaign_source:
            raise MonteCarloError('captured campaign source conflicts with report attribution')
        metric = _metric_key(metric)
        value, used = selected._statistic(metric, how)
        keys = selected._keys()
        if set(keys) != set(selected.seeds):
            raise MonteCarloError('measurement() sample seeds must cover exactly the selected observations')
        aggregate = AggregateMeasurement(
            metric=metric, statistic=how, variation=self.settings.variation,
            selected_corner=label, samples_total=len(keys), samples_used=len(used),
            samples_failed=len(selected.failed), samples_unavailable=len(selected.reports)-len(used),
            used_indices=tuple(k[1] for k in used),
            sample_seeds=tuple((k[1], selected.seeds[k]) for k in keys),
            campaign_id=self._campaign_id, base_seed=self.settings.seed,
            requested_samples=self.settings.n, structure_hash=self.structure_hash,
            draw_scheme=self._draw_scheme)
        return Measurement(source=source, value=value, corner=record[1],
                           temperature=record[2], voltage=record[3], tool_name=record[5],
                           timestamp=time.time(),
                           sim_run_id=self._campaign_id, aggregate=aggregate)

    def percentile(self, metric: str, q: float) -> float:
        """The ``q``-th percentile (0-100) over the usable samples.

        No observations gives nan with MetricWarning; an unknown metric or
        out-of-range percentile raises ValueError.

        Parameters
        ----------
        metric : str
            Registered metric name.
        q : float
            Percentile in [0, 100].

        Returns
        -------
        float
            The percentile of the usable values, in the metric's units;
            ``nan`` when there are none.

        Raises
        ------
        ValueError
            If ``q`` is outside [0, 100] or ``metric`` is unknown.
        """
        if not 0 <= q <= 100:
            raise ValueError(f"percentile(): q must be in [0, 100], got {q!r}")
        metric = _metric_key(metric)
        used = self._used(metric, f'percentile({metric!r}, {q})')
        if not used:
            return math.nan
        return float(np.percentile(list(used.values()), q))

    def worst(self, metric: str) -> SampleWorstCase:
        """The usable sample closest to failing a spec on ``metric``.

        No observations gives SampleWorstCase(nan, None, None) with
        MetricWarning. A metric without a declared spec raises ValueError.

        Parameters
        ----------
        metric : str
            Registered metric name that at least one declared spec checks.

        Returns
        -------
        SampleWorstCase
            Value, corner and sample index of the sample with the smallest
            margin to a spec on the metric (largest violation if any fail).

        Raises
        ------
        ValueError
            If no spec checks ``metric`` or the metric is unknown.
        """
        metric = _metric_key(metric)
        declarations = {s.name for s in self._spec_declarations if s.metric == metric}
        if not declarations:
            raise ValueError(
                f"worst({metric!r}): no spec checks this metric, so there "
                f"is no failing direction to rank by.")
        used = self._used(metric, f'worst({metric!r})')
        specs = [(k, res) for k, report in self.reports.items() if k in used
                 for res in report.results if res.name in declarations]
        if not specs:
            return SampleWorstCase(math.nan, None, None)

        def margin(kr):
            k, res = kr
            v, t = used[k], float(res.target)
            if res.op in ('>', '>='):
                return v - t
            if res.op in ('<', '<='):
                return t - v
            return -abs(v - t)
        k, _ = min(specs, key=margin)
        return SampleWorstCase(used[k], k[0], k[1])

    def yield_fraction(self) -> dict:
        """Declared spec yields and Wilson intervals; failures fail every spec.

        A declared spec without successful checks has zero successes. With
        zero observations its fraction and interval are nan.
        """
        total = len(self._keys())
        return {name: (k / total if total else math.nan, wilson_interval(k, total))
                for name, k in self._spec_pass_counts().items()}

    def yield_by_point(self) -> dict:
        """Per-point spec yields and Wilson intervals in retained plan order.

        Missing reports count against yield. Corner-only runs use physical
        corner strings; grids use full point tuples.
        """
        return {point: self.point(point).yield_fraction() for point in self._points()}

    @property
    def passed(self) -> bool:
        """All samples pass; with ``min_yield``, every spec's yield meets it."""
        if not self.reports:
            return False
        floor = self.settings.min_yield
        if floor is None:
            return len(self.reports) == len(self._keys()) and not self.failed and all(
                row['run.status'] == 'PASS' and all(
                    row['check.' + declaration['key']] == 'PASS'
                    for declaration in self._check_declarations)
                for row in self.table())
        # Each operating point must meet the floor; pooled yield is informational.
        return bool(self._spec_declarations) and all(
            bool(yields) and all(f >= floor for f, _ in yields.values())
            for yields in self.yield_by_point().values())

    # -- tabulation ------------------------------------------------------
    def table(self) -> list:
        """Namespaced per-run evidence table, with sample and seed columns."""
        return super().table()

    def to_csv(self, path) -> None:
        """Write the namespaced per-run table including failed samples."""
        super().to_csv(path)

    def plot_histogram(self, metric: str, ax=None):
        """Histogram of the usable values with each spec limit drawn.

        Parameters
        ----------
        metric : str
            Registered metric name.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on; a new figure is created when omitted.

        Returns
        -------
        matplotlib.axes.Axes
            The axes, x labelled with the metric and y with the sample count.
        """
        import matplotlib.pyplot as plt
        metric = _metric_key(metric)
        used = self._used(metric, f'plot_histogram({metric!r})')
        if ax is None:
            _, ax = plt.subplots()
        ax.hist(list(used.values()), bins='auto', color='0.7')
        seen = set()
        for op, target in self._spec_limits(metric):
            if target not in seen:
                seen.add(target)
                ax.axvline(float(target), color='r', linestyle='--',
                           label=f'{op} {target:g}')
        ax.set_xlabel(metric)
        ax.set_ylabel('samples')
        if seen:
            ax.legend()
        return ax

    def summary(self) -> str:
        """Per-point verdicts and informational pooled yield.

        Grids include planned/report/failure counts and scalar statistics;
        corner-only collections retain run/failed/used lines.
        """
        lines = [f"MonteCarloResults — {'PASS' if self.passed else 'FAIL'}"]
        if self._is_grid():
            lines.append('  ' + self._counts())
        for point in self._points():
            selected = self.point(point)
            if self._is_grid():
                lines.append(f"  {self._point_text(point)}: {'PASS' if selected.passed else 'FAIL'} "
                             + selected._counts())
            else:
                lines.append(f'  corner={point}: {len(selected._keys())} run, '
                             f'{len(selected.failed)} failed, {len(selected.reports)} used '
                             f"— {'PASS' if selected.passed else 'FAIL'}")
            for name, (fraction, (lo, hi)) in selected.yield_fraction().items():
                lines.append(f'    yield {name}: {fraction:.3f} [{lo:.3f}, {hi:.3f}]')
        for name, (fraction, (lo, hi)) in self.yield_fraction().items():
            lines.append(f'  aggregate yield {name}: {fraction:.3f} [{lo:.3f}, {hi:.3f}] (informational)')
        if self._is_grid():
            lines.extend(self._metric_summary(aggregate=True))
        if self.rerun_samples:
            lines.append(f'  rerun one per process (not finished in a '
                         f'multi-sample process, or fell back before one '
                         f'launched): {len(self.rerun_samples)} '
                         f'({", ".join(f"{c}/{i}" for c, i in self.rerun_samples)})')
        return '\n'.join(lines)

    def sensitivity(self, metric: str, *, by: str = 'parameter'):
        """Rank *metric* against each device's drawn model parameters.

        Needs a run with ``record_draws=True``. Uses the usable samples
        (failed and nan ones are skipped with one counted
        :class:`silk.metrics.diagnostics.MetricWarning`) and fits the
        metric on the standardised draws by least squares
        (``silk.circuit.monte_carlo_draws.rank_sensitivity``): per
        parameter the effect per sigma, its share of the variance and its
        correlation, sorted by share, with the fit's R squared. ``by='device'``
        sums each device's parameters.

        Select exactly one planned point, even if other points have no usable
        samples: use ``mc.point(key).sensitivity(...)`` for a grid or
        ``mc.corner(name).sensitivity(...)`` for a corner-only campaign.
        Pooling operating points mixes their shifts with the mismatch spread
        and can rank wrongly while the fit's R squared stays high.

        Parameters
        ----------
        metric : str
            Registered metric name.
        by : {'parameter', 'device'}, default 'parameter'
            Rank each drawn parameter, or sum the parameters of each device.

        Returns
        -------
        object
            Rows sorted by variance share, with the fit's R squared.

        Raises
        ------
        ValueError
            If the run did not record draws (set ``record_draws=True``), the
            campaign contains more than one planned point, even when only
            one has usable samples, or on a caller error from
            ``rank_sensitivity`` (unknown ``by``, no usable sample).
        NotSupportedError
            If the run declared ``correlate`` quantities, which are not
            ranked yet.
        """
        from silk.circuit.monte_carlo_draws import rank_sensitivity
        metric = _metric_key(metric)
        if self.draws is None:
            raise ValueError(
                f"sensitivity({metric!r}): this Monte Carlo run recorded no "
                f"drawn parameters; run it with record_draws=True "
                f"(tb.set_monte_carlo(n, record_draws=True)).")
        if self.settings.correlate:
            from silk.exceptions import NotSupportedError
            raise NotSupportedError(
                f"sensitivity({metric!r}): ranking the correlate= "
                f"quantities is not implemented yet; declare the run "
                f"without correlate to rank the drawn parameters.")
        if len(self._points()) != 1:
            if not self._is_grid():
                corners = sorted(self._points())
                raise MonteCarloError(f'sensitivity(): several corners {corners}; use mc.corner({corners[0]!r}).sensitivity(...)')
            raise MonteCarloError('sensitivity() needs one full point; use .point(...)')
        used = self._used(metric, f'sensitivity({metric!r})')
        values = {k: v for k, v in used.items() if k in self.draws.values}
        corners = sorted({k[0] for k in values}, key=str)
        if len(corners) > 1:
            raise ValueError(
                f"sensitivity({metric!r}): the usable samples span several "
                f"corners {corners}; pooled, the process shift between "
                f"corners swamps the mismatch spread and the ranking is "
                f"wrong. Rank one corner at a time, e.g. "
                f"mc.corner({corners[0]!r}).sensitivity({metric!r}).")
        return rank_sensitivity(metric, values, self.draws, by=by)

    def rerun(self, key):
        """Re-run one sample with its recorded seed.

        Parameters
        ----------
        key : tuple
            ``(corner, sample index)`` of a sample in ``seeds``.

        Returns
        -------
        SpecReport
            The fresh report produced by the run's rerun hook.

        Raises
        ------
        MonteCarloError
            If the results were built without a rerun hook or ``key`` is
            unknown.
        """
        if self._rerun_hook is None:
            raise MonteCarloError(
                "rerun() needs a run that recorded how to re-simulate; "
                "these results were built without a rerun hook.")
        if key not in self.seeds:
            raise MonteCarloError(f"unknown sample {key!r}")
        return self._rerun_hook(key, self.seeds[key])

    def __repr__(self) -> str:
        return (f'MonteCarloResults(samples={len(self._keys())}, '
                f'failed={len(self.failed)}, passed={self.passed})')


@dataclass(frozen=True)
class _SampleSpec:
    index: int
    seed: int
    variation: str                      # 'mismatch' | 'process' | 'both'
    section: Optional[str]              # library section; None when wrapper_path is set
    wrapper_path: Optional[str]         # absolute path of the both-wrapper
    record_draws: bool = False          # emit and read the drawn parameters

    base_seed: Optional[int] = None     # full behavioural draw seed
    behavioral_records: Optional[tuple] = None  # resolved at campaign preparation

    #: section name the both-wrapper declares
    WRAPPER_SECTION = 'mc_both'


def write_both_wrapper(path: str, statistical: StatisticalModels) -> str:
    """Write the wrapper library whose one section turns on both variation
    switches and pulls in every process include (absolute paths).

    Returns *path*. Idempotent: rewriting yields the same bytes.
    """
    # Every token below is PDK statistical data written into a library the
    # deck loads: checked as written, before the file is created (TASK-593).
    from silk.utils.deck_text import card_text
    card_text(os.path.abspath(path), who='Monte Carlo wrapper file path')
    lines = [
        '* silk Monte Carlo: mismatch and process variation together',
        f'.lib {_SampleSpec.WRAPPER_SECTION}',
        f".param {card_text(statistical.mismatch_switch, who='PDK mismatch switch')}=1",
        f".param {card_text(statistical.process_switch, who='PDK process switch')}=1",
    ]
    lines += [f'.include "{card_text(os.path.abspath(p), who="PDK process include path")}"'
              for p in statistical.process_includes]
    lines.append(f'.endl {_SampleSpec.WRAPPER_SECTION}')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path


class DrawDevice(NamedTuple):
    """One device whose drawn parameters a sample records (``record_draws``).

    ``path`` is the emitted instance path (``'xdut.XMm1'``), ``internal``
    the model's own element-instance name inside that device
    (``op_internal_name``), ``name`` the designer name (``'dut.m1'``) and
    ``params`` the manifest's ``record`` entry for its device type.
    """
    path: str
    internal: str
    name: str
    params: tuple


def draw_devices(elem_dict, statistical: StatisticalModels,
                 ) -> tuple[list[DrawDevice], tuple[str, ...]]:
    """The devices a ``record_draws`` sample records, and the skipped types.

    Walks every MOS-shaped device below *elem_dict* in emission order (the
    order the ``showmod`` lines are emitted and parsed in). A device whose
    catalog type has a manifest ``record`` entry is recorded; the other
    types are returned, sorted, as skipped.

    Raises :class:`MonteCarloError` for a recorded device whose catalog
    entry declares no ``op_internal_name``: the drawn values are read
    through that name, and it is not derivable from the model name.
    """
    from silk.circuit.emitted_names import emitted_instance_name
    from silk.circuit.modules_results import (
        _device_op_internal_name, _device_op_prefix, _iter_mos_device_paths)
    devices, skipped = [], set()
    for path, dpath, leaf, device in _iter_mos_device_paths(elem_dict):
        kind = getattr(device, 'device_name', None)
        params = statistical.record.get(kind) if kind else None
        if not params:
            skipped.add(str(kind))
            continue
        name = f'{dpath}.{leaf}' if dpath else leaf
        internal = _device_op_internal_name(device)
        if not internal:
            raise MonteCarloError(
                f"record_draws: device {name!r} ({kind}) has no "
                f"op_internal_name in its PDK catalog entry, so its drawn "
                f"parameters cannot be addressed; declare it, or drop "
                f"{kind!r} from the manifest's statistical record.")
        devices.append(DrawDevice(
            _device_op_prefix(path, emitted_instance_name(device, leaf)),
            internal, name, tuple(params)))
    return devices, tuple(sorted(skipped))


def showmod_requests(devices: list[DrawDevice]) -> tuple[list[str], list]:
    """``(control lines, parse requests)`` for *devices*, in one order.

    The lines go into the sample's control block; the requests are what
    :func:`~silk.circuit.monte_carlo_draws.parse_showmod` matches the log's
    blocks against, by position.
    """
    from silk.circuit.monte_carlo_draws import showmod_lines
    # A MOS element's name starts with 'm' in SPICE, and showmod's address
    # spells that letter itself: 'm.<path>.m<rest>'.
    for d in devices:
        if not d.internal.lower().startswith('m'):
            raise MonteCarloError(
                f"record_draws: device {d.name!r} has internal instance "
                f"name {d.internal!r}, which is not a MOS element name "
                f"(it must start with 'm').")
    lines = showmod_lines([(d.path, d.internal[1:], d.params)
                           for d in devices])
    return lines, [(d.name, d.params) for d in devices]


def build_draw_table(draws: dict, skipped_types=(), *, raw: bool = True):
    """A :class:`~silk.circuit.monte_carlo_draws.DrawTable` from
    ``{sample key: {(device, param): value}}``; *raw* is True for actual
    values (ngspice), False for normalised draws (Xyce).

    Columns are the first sample's keys in their recorded order; every
    sample of one plan records the same devices.
    """
    from silk.circuit.monte_carlo_draws import DrawTable
    columns = next(iter(draws.values())).keys() if draws else ()
    columns = tuple(columns)
    return DrawTable(columns=columns,
                     values={k: [v[c] for c in columns]
                             for k, v in draws.items()},
                     raw=raw, skipped_types=tuple(skipped_types))


def write_draws_csv(path, table, seeds: dict) -> None:
    """Write *table*'s samples as CSV: ``corner, sample, seed``, then one
    ``<device>:<param>`` column per drawn parameter, one row per sample."""
    grid = any(isinstance(key[0], tuple) for key in seeds)
    first = next(iter(seeds), None)
    variables = [name for name, value in first[0][3]] if grid and first and len(first[0]) == 4 else []
    head = (['corner', 'temperature', 'supply', *variables, 'sample', 'seed'] if grid else
            ['corner', 'sample', 'seed']) + [f'{d}:{p}' for d, p in table.columns]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(head)
        for key in (table.values if grid else sorted(table.values)):
            coordinates = list(key[0][:3]) + ([v for n, v in key[0][3]] if len(key[0]) == 4 else []) if grid else [key[0]]
            w.writerow(coordinates + [key[1], seeds.get(key)]
                       + [repr(float(v)) for v in table.values[key]])
