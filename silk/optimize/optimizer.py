"""Optimizer — evaluation harness for circuit design optimization.

Optimizer wraps a ModuleTestbench and a VariableSpace into a single
callable ``evaluate(point) → record``.  Any optimizer — scipy, scikit-optimize,
CMA-ES, or an AI agent — drives the search by calling evaluate() and reading
the returned record.

Design principles
-----------------
- **Thin glue layer**: no optimizer logic lives here; this class only defines
  the evaluation protocol and failure logging.
- **Every call is logged**: successes and failures alike, each with a unique
  status tag so they can be filtered and analysed separately later.
- **Objectives and constraints are declared separately** from metrics.
  add_metric() (on ModuleTestbench) measures things; add_objective() and
  add_constraint() (on Optimizer) tell the optimizer what to optimise
  and what counts as feasible.

Failure taxonomy
----------------
``'parameter_error'``   space.apply() failed — bad point, out of bounds.
``'netlist_error'``     CircuitError raised during netlist generation.
``'simulation_failure'``SimulationError raised — ngspice exited non-zero.
``'metric_error'``      Simulation succeeded but one or more metric
                        extractors raised (stored as "Error: ..." strings
                        in the metrics dict by ModuleTestbench).
``'unexpected_error'``  Any other exception from test.run() — a bug or an
                        environment failure, not a design-point failure.
``'success'``           All steps completed; ``feasible`` flag set by
                        evaluating constraints.
"""

import math
import operator
import pickle
import time
from numbers import Complex, Real
from typing import Any, Callable, Dict, List, Optional, Union

from silk.optimize.parameter_space import VariableSpace
from silk.circuit.variable import Variable
from silk.circuit.module_test import (
    ModuleTestbench, _RunSpec, _simulate_run_outcome, _planned_outcome,
)
from silk.circuit.spec_report import SpecReport
from silk.circuit.test_placement import authored_test
from silk.exceptions import CircuitError, SimulationError
from silk.simulation.executor import RunTask, resolve_n_workers, run_tasks


# ---------------------------------------------------------------------------
# n_workers support (TASK-301.5): prepare in the parent, simulate through the
# shared executor, evaluate in the parent, in point order.
# ---------------------------------------------------------------------------
#
# Identity-preserving adoption of a worker's simulated state onto the live
# netlist lives in ModuleTestbench._evaluate_run itself now (TASK-301.3's
# module_test._SIMULATED_STATE_ATTRS) -- there is one shared list, not one
# per caller. _evaluate_run() already no-ops the copy when its result IS
# the live netlist (the serial, no-pickling case here), so this module
# just hands the worker's _RunSimResult straight to it.


def _run_design_point(run_dir, spec):
    """Picklable worker adapter carrying the shared attempt outcome.

    Error classification belongs to the parent optimizer; replacement state
    belongs to the testbench's shared outcome consumer.
    """
    return _simulate_run_outcome(spec)


class Optimizer:
    """Evaluation harness for circuit design optimization.

    Parameters
    ----------
    test : ModuleTestbench or object
        The testbench that runs simulations and returns a ``SpecReport``;
        its ``.metrics`` are consumed directly. Must have
        ``add_module_under_test()`` already called. Custom tests returning ``{'metrics': {...}}`` from
        ``run(corner=...)`` remain supported.
    space : VariableSpace
        The joint parameter space.  Variables in the space must be
        linked to the circuit's device parameters via
        ``update_element()`` before calling ``evaluate()``.

    Examples
    --------
    >>> from silk import Variable, VariableSpace, Optimizer
    >>> class Bench:
    ...     def run(self, corner=None):
    ...         return {'metrics': {'gain': width.value}}
    >>> width = Variable('w', value=2.0, bounds=(1.0, 4.0))
    >>> opt = Optimizer(Bench(), VariableSpace([width]))
    >>> _ = opt.add_objective('gain', minimize=False)
    >>> record = opt.evaluate({'w': 3.0})
    >>> record['status'], record['metrics']['gain']
    ('success', 3.0)
    """

    def __init__(self, test: ModuleTestbench, space: VariableSpace) -> None:
        self.test  = test
        self.space = space

        self._objectives:  Dict[str, Dict]   = {}
        self._constraints: Dict[str, Dict]   = {}
        self._history:     List[Dict]        = []
        self._objective_sources: Dict[str, str] = {}
        self._constraint_sources: Dict[str, str] = {}
        self._margins: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_objective(
        self,
        name: Union[str, Variable],
        minimize: Optional[bool] = None,
        maximize: Optional[bool] = None,
    ) -> 'Optimizer':
        """Designate a measured metric or VariableSpace variable as an objective.

        For multi-objective optimization, all declared objectives are
        combined using equal weights in the ``best`` property via weighted
        linear combination. A NaN in any declared objective excludes the
        record from ``best`` without changing constraint feasibility.

        Parameters
        ----------
        name : str or Variable
            Registered metric name, or a variable in this VariableSpace. Variable
            values are read from recorded samples, never from live state.
        minimize : bool, optional
            Set to ``True`` to minimize (e.g. power, settling time).
            Mutually exclusive with ``maximize``.
        maximize : bool, optional
            Set to ``True`` to maximize (e.g. gain, bandwidth).
            Mutually exclusive with ``minimize``.

        Returns
        -------
        Optimizer
            ``self``, for method chaining.

        Notes
        -----
        Exactly one of ``minimize`` or ``maximize`` should be ``True``.
        For backwards compatibility, if neither is specified, defaults to
        ``minimize=True``.

        Examples
        --------
        Single-objective optimization::

            opt.add_objective('power_uW', minimize=True)

        Multi-objective optimization (equally weighted)::

            opt.add_objective('gain_dB', maximize=True)
            opt.add_objective('power_uW', minimize=True)
            best = opt.best  # Combined via linear equation
        """
        # Handle backwards compatibility
        # minimize=False historically meant "don't minimize" → maximize
        if minimize is False and maximize is None:
            maximize = True
        # Default: if neither specified, minimize
        elif minimize is None and maximize is None:
            minimize = True

        # Validation: can't specify both
        if minimize is True and maximize is True:
            raise ValueError("Cannot specify both minimize=True and maximize=True")

        name, source = self._resolve_target(name)
        if name in self._objective_sources and self._objective_sources[name] != source:
            raise ValueError(f"Objective {name!r} already refers to a different value source")
        self._objective_sources[name] = source
        self._objectives[name] = {
            'minimize': minimize is True,
            'maximize': maximize is True,
        }
        return self

    def add_constraint(
        self,
        name: Union[str, Variable],
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> 'Optimizer':
        """Add an explicit feasibility constraint on a metric or VariableSpace variable.

        A point is feasible only when *all* constraints are satisfied.
        A NaN value cannot satisfy a declared constraint.
        Constraints are checked after a successful simulation; failed
        evaluations are marked ``feasible=None``.

        Parameters
        ----------
        name : str or Variable
            Metric name or variable in this VariableSpace. An explicit metric
            constraint replaces all spec-derived limits on that metric.
        min : float, optional
            Lower bound (inclusive).  ``None`` means unbounded below.
        max : float, optional
            Upper bound (inclusive).  ``None`` means unbounded above.

        Returns
        -------
        Optimizer
            ``self``, for method chaining.
        """
        name, source = self._resolve_target(name)
        if name in self._constraint_sources and self._constraint_sources[name] != source:
            raise ValueError(f"Constraint {name!r} already refers to a different value source")
        self._constraint_sources[name] = source
        self._constraints[name] = {'min': min, 'max': max}
        return self

    def _resolve_target(self, target):
        """Keep metric strings distinct from identity-checked VariableSpace variables."""
        if isinstance(target, str):
            return target, 'metric'
        if isinstance(target, Variable):
            names = [name for name, var in self.space._vars.items() if var is target]
            if len(names) == 1:
                return names[0], 'variable'
            raise ValueError(
                f"Variable {target.name!r} must appear exactly once in this VariableSpace; "
                "pass the Variable used to construct the VariableSpace."
            )
        raise TypeError("Expected a metric name or a Variable from this VariableSpace")

    def _test_specs(self):
        # Custom run-only test objects need not expose spec declarations.
        specs = getattr(self.test, '_specs', ())
        return specs if isinstance(specs, (list, tuple)) else ()

    def add_margin(self, spec_name: str, amount: float) -> 'Optimizer':
        """Tighten one named spec by an absolute, finite, nonnegative amount.

        Use the same numeric unit as the spec target (no parsing of its
        label): 3 for a dB target, 1e-3 for a voltage target stored in volts.
        Lower limits increase; upper limits decrease. Equality specs cannot
        have margins. Repeated calls replace the margin. An explicit metric
        constraint still overrides the spec, regardless of call order.
        """
        specs = [s for s in self._test_specs() if s['name'] == spec_name]
        if len(specs) != 1:
            available = [s['name'] for s in self._test_specs()]
            raise ValueError(
                f"Unknown or ambiguous spec name {spec_name!r}; "
                f"use a unique testbench spec name from {available!r}."
            )
        if specs[0]['op'] == '==':
            raise ValueError(
                f"Cannot add a margin to == spec {spec_name!r}; "
                "use an explicit add_constraint with min/max bounds."
            )
        if not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError("Margin must be a finite, nonnegative number in the spec target's unit")
        self._margins[spec_name] = amount
        return self

    def _effective_constraints(self):
        """Fresh serializable limit descriptions, also used for evaluation."""
        specs = self._test_specs()
        explicit_metrics = {name for name in self._constraints
                            if self._constraint_sources[name] == 'metric'}
        limits = []
        for spec in specs:
            if spec['metric'] in explicit_metrics:
                continue
            margin = self._margins.get(spec['name'], 0)
            target = spec['target']
            target += margin if spec['op'] in ('>', '>=') else -margin
            declaration = spec.get('spec')
            limits.append({
                'name': spec['metric'], 'value_source': 'metric',
                'op': spec['op'], 'target': target,
                'source': 'spec+margin' if spec['name'] in self._margins else 'spec',
                'spec_name': spec['name'], 'spec_target': spec['target'],
                'margin': margin, 'units': getattr(declaration, 'units', ''),
            })
        for name, bounds in self._constraints.items():
            source = self._constraint_sources[name]
            limits.append({
                'name': name, 'value_source': source, **bounds,
                'source': 'explicit constraint',
                'overrides': [s['name'] for s in specs
                              if source == 'metric' and s['metric'] == name],
            })
        return limits

    def _applied_params(self):
        """Freeze the complete sample after resolution snapping, before running."""
        return {name: var.value for name, var in self.space._vars.items()}

    @staticmethod
    def _record_params(record):
        """Older persisted records only captured the requested parameters."""
        return record.get('applied_params', record['params'])

    @staticmethod
    def _value(name, source, metrics, params):
        value = (params if source == 'variable' else metrics).get(name)
        if isinstance(value, Complex) and not isinstance(value, Real):
            raise ValueError(
                f"optimizer value {name!r}: complex values cannot be ordered; "
                "choose result.real or abs(result).")
        return value

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate(self, point: Dict[str, Any], corner: str = 'tt') -> Dict:
        """Apply *point*, run the testbench, and return a logged record.

        After scalar-run preflight, appends to :attr:`history` regardless
        of success or failure, with a ``status`` tag for later filtering.
        Declared Monte Carlo/corner collections raise NotSupportedError
        before applying parameters or preparing a deck.
        Reads metrics from a ``SpecReport`` or a legacy ``{'metrics': {...}}``
        result. Testbench specs constrain feasibility by default. Explicit
        metric constraints override their specs; ``SpecReport.passed`` is
        not consulted because optimizer overrides and margins may differ.
        NaN measurements remain in history with ``status='success'`` when
        extraction completes. NaN fails a constraint on that metric; NaN in
        any declared objective excludes the record from ``best``. Unrelated
        NaN diagnostics affect neither feasibility nor ranking. Infinity
        retains its usual bound comparisons and objective scoring.

        Unlike ``ModuleTestbench.evaluate(rawfile=...)``, which measures
        existing simulation data, this method applies a design point and
        calls ``test.run(corner=...)`` once.

        Parameters
        ----------
        point : dict
            ``{variable_name: value}`` mapping.  Will be applied to the
            VariableSpace (and through it, to the circuit's device
            parameters).
        corner : str
            Process corner forwarded to ``ModuleTestbench.run()``.

        Returns
        -------
        dict
            Record with keys:

            - ``id``: Monotonically increasing evaluation index.
            - ``params``: Copy of *point*.
            - ``corner``: Corner used.
            - ``status``: One of the failure-taxonomy strings above.
            - ``metrics``: Dict of metric values, or ``None`` on early failure.
            - ``feasible``: ``True``/``False`` for successes; ``None`` otherwise.
            - ``error``: Error message string, or ``None``.
            - ``error_type``: Exception class name, or ``None``.
            - ``timestamp``: Unix timestamp at evaluation start.
            - ``applied_params``: Complete applied sample after resolution
              snapping (None if applying the point failed).
            - ``constraints``: Effective-limit provenance snapshot for this
              point.
        """
        if isinstance(authored_test(self.test), ModuleTestbench):
            self.test._require_single_run()

        record: Dict[str, Any] = {
            'id':         len(self._history),
            'params':     dict(point),
            'applied_params': None,
            'corner':     corner,
            'status':     None,
            'metrics':    None,
            'feasible':   None,
            'error':      None,
            'error_type': None,
            'timestamp':  time.time(),
            'constraints': self._effective_constraints(),
        }

        # Step 1 — apply parameters
        try:
            self.space.apply(point)
            record['applied_params'] = self._applied_params()
        except (KeyError, ValueError) as exc:
            record['status']     = 'parameter_error'
            record['error']      = str(exc)
            record['error_type'] = type(exc).__name__
            self._history.append(record)
            return record

        # Step 2 — run simulation and extract metrics
        try:
            result = self.test.run(corner=corner)
            metrics = (result.metrics if isinstance(result, SpecReport)
                       else result['metrics'])
        except SimulationError as exc:
            record['status']     = 'simulation_failure'
            record['error']      = str(exc)
            record['error_type'] = 'SimulationError'
            self._history.append(record)
            return record
        except CircuitError as exc:
            record['status']     = 'netlist_error'
            record['error']      = str(exc)
            record['error_type'] = type(exc).__name__
            self._history.append(record)
            return record
        except Exception as exc:
            # Unexpected error — still log it uniquely
            record['status']     = 'unexpected_error'
            record['error']      = str(exc)
            record['error_type'] = type(exc).__name__
            self._history.append(record)
            return record

        # Step 3 — detect metric extraction failures
        # ModuleTestbench stores "Error: ..." strings for failed extractors
        failed = {k: v for k, v in metrics.items()
                  if isinstance(v, str) and v.startswith('Error:')}
        if failed:
            record['status']     = 'metric_error'
            record['metrics']    = metrics
            record['error']      = '; '.join(f"{k}: {v}" for k, v in failed.items())
            record['error_type'] = 'MetricExtractionError'
            self._history.append(record)
            return record

        # Step 4 — success: check feasibility
        record['status']   = 'success'
        record['metrics']  = metrics
        record['feasible'] = self._check_feasible(
            metrics, record['applied_params'], record['constraints'])
        self._history.append(record)
        return record

    # ------------------------------------------------------------------
    # Batch sweep
    # ------------------------------------------------------------------

    def sweep(
        self,
        n: int,
        method: str = 'lhs',
        corner: str = 'tt',
        max_failures: Optional[int] = None,
        n_workers: Optional[Any] = None,
        *,
        seed: Optional[int] = 0,
    ) -> List[Dict]:
        """Evaluate *n* jointly sampled design points.

        Continues through failures by default.  Set *max_failures* to
        abort early if too many cumulative problems occur — useful when
        a bad region of parameter space causes systematic ngspice crashes.

        Parameters
        ----------
        n : int
            Number of design points (for ``method='grid'`` this is
            requested points per variable — see :meth:`VariableSpace.sample`).
            Grid size is the product of actual group and ungrouped column counts. Shortened
            linear numeric grids emit UserWarning at this call's caller;
            standard filters apply before evaluation, with no report persistence.
        method : str
            Sampling method passed to :meth:`VariableSpace.sample`.
        seed : int or None
            Sampling seed passed to the space (default 0). None uses fresh
            OS entropy; booleans and non-integers raise TypeError.
        corner : str
            Process corner for every evaluation.
        max_failures : int, optional
            Stop the sweep early if the cumulative failure count across
            all status types reaches this limit.  ``None`` (default)
            means run all *n* points regardless of failures. Points are
            dispatched to the executor in chunks of ``n_workers`` points
            (1 when serial), so this can still stop before every point in
            *n* has been prepared and simulated — but a chunk already
            dispatched to the pool runs to completion before the count is
            checked: with real parallelism, some compute
            beyond the returned records may still have happened.
        n_workers : None, int or 'all', optional
            Requested worker count for ``test``, resolved through
            ``silk.simulation.executor.resolve_n_workers`` (``None``/
            ``1`` -> serial in-process, an int -> that many workers capped
            at the CPU count, ``'all'`` -> every CPU; behavioral is always
            forced to 1). Only ``ModuleTestbench`` test
            objects support ``n_workers > 1`` — a custom test (one whose
            ``run()`` alone silk drives) always runs serially, and
            requesting more than one worker for it raises ``TypeError``.
            Each worker is a full extra OS process with its own parsed PDK
            — size ``n_workers`` for memory, not just CPU count.

            For a ``ModuleTestbench`` test, EVERY point — serial or
            parallel — now gets its own output folder (``run_label=
            'point_<id>'``, BREAKING: previously every point in one sweep
            shared the default, unlabeled output path) via the same
            prepare (parent) / simulate (worker, through the
            executor) / evaluate (parent) sequence
            :meth:`run_corners <silk.circuit.ModuleTestbench.run_corners>` uses, so results, best-point selection
            (NaN handling included) and history are identical
            between ``n_workers=None`` and ``n_workers>1`` — only the
            folder each point lands in and the wall-clock time differ.
            After the call, ``self.test.netlist`` is the SAME object it
            was before (never replaced by a worker's copy —
            ``ModuleTestbench._evaluate_run`` adopts the
            simulate step's state onto it in place) and
            reflects the last point actually evaluated (the last point in
            *n*, or the last one before an early ``max_failures`` stop) —
            exactly the state a serial ``test.run()`` on that one point
            alone would leave.

        Returns
        -------
        list of dict
            The records added during this sweep (subset of
            :attr:`history`).  May be shorter than *n* if *max_failures*
            was reached.
        """
        if isinstance(authored_test(self.test), ModuleTestbench):
            self.test._require_single_run()
        if getattr(self.space.sample, '__func__', None) is VariableSpace.sample:
            points = self.space._sample(n, method=method, seed=seed)
        else:
            # Preserve custom public samplers; the built-in private entry
            # only carries warning depth for the standard implementation.
            points = self.space.sample(n, method=method, seed=seed)

        if (not isinstance(self.test, ModuleTestbench)
                or type(self.test).run is not ModuleTestbench.run):
            if n_workers is not None and resolve_n_workers(n_workers) > 1:
                raise TypeError(
                    "sweep(n_workers=...) > 1 requires a ModuleTestbench "
                    "test (the prepare/simulate/evaluate split, TASK-301.2); "
                    f"got {type(self.test).__name__}, whose run() cannot be "
                    "split across workers."
                )
            records: List[Dict] = []
            failures = 0
            for pt in points:
                rec = self.evaluate(pt, corner=corner)
                records.append(rec)
                if rec['status'] != 'success':
                    failures += 1
                    if max_failures is not None and failures >= max_failures:
                        break
            return records

        resolved = resolve_n_workers(n_workers, backend=self._backend_name())
        chunk = max(resolved, 1)
        records: List[Dict] = []
        failures = 0
        for start in range(0, len(points), chunk):
            batch = self._evaluate_points(points[start:start + chunk], corner, resolved)
            for rec in batch:
                records.append(rec)
                self._history.append(rec)
                if rec['status'] != 'success':
                    failures += 1
                    if max_failures is not None and failures >= max_failures:
                        return records
        return records

    def _backend_name(self):
        """The simulator name :meth:`resolve_n_workers` should check for
        the ``behavioral`` forced-serial rule -- ``self.test.netlist.simulator``
        when :attr:`test` has one, else ``None`` (no forcing)."""
        netlist = getattr(self.test, 'netlist', None)
        return getattr(netlist, 'simulator', None)

    def _evaluate_points(self, points: List[Dict], corner: str, resolved: int) -> List[Dict]:
        """Prepare every point in *points* in the parent, simulate them
        through the shared executor (serially in-process for
        ``resolved<=1``, across a pool otherwise), and evaluate each
        result in the parent, in submission order.

        Assumes ``self.test`` is a ``ModuleTestbench`` —
        callers must check that themselves (:meth:`sweep` does).
        """
        test = self.test
        pending = []
        entries = []
        for i, pt in enumerate(points):
            record_id = len(self._history) + i
            record = {
                'id': record_id, 'params': dict(pt), 'applied_params': None,
                'corner': corner, 'status': None, 'metrics': None,
                'feasible': None, 'error': None, 'error_type': None,
                'timestamp': time.time(), 'constraints': self._effective_constraints(),
            }
            prepared = None
            try:
                self.space.apply(pt)
                record['applied_params'] = self._applied_params()
            except (KeyError, ValueError) as exc:
                record.update(status='parameter_error', error=str(exc),
                              error_type=type(exc).__name__)
            if record['status'] is None:
                try:
                    prepared = test._prepare_single_run(
                        corner=corner, run_label=f'point_{record_id}')
                    if isinstance(prepared, _RunSpec):
                        if resolved > 1:
                            prepared = pickle.loads(pickle.dumps(prepared))
                        pending.append((i, prepared))
                except Exception as exc:
                    self._record_run_error(record, exc)
            # Even a rejected point retains the last successfully applied
            # full sample, not the last point later prepared in this batch.
            entries.append((record, prepared, self._applied_params()))

        tasks = [RunTask(fn=_run_design_point, run_dir=spec.tb_dir, args=(spec,),
                         simulator=spec.netlist.simulator)
                 for _, spec in pending]
        outcomes = dict(zip((i for i, _ in pending),
                            run_tasks(tasks, n_workers=resolved))) if tasks else {}
        records = []
        for i, (record, prepared, applied) in enumerate(entries):
            self.space.apply(applied)
            if record['status'] is None:
                try:
                    report = (test._consume_run_outcome(_planned_outcome(outcomes[i], prepared))
                              if i in outcomes else prepared)
                except Exception as exc:
                    self._record_run_error(record, exc)
                    if i in outcomes:
                        outcome = outcomes[i]
                        if outcome.error is not None:
                            record['error_type'] = 'WorkerError'
                        elif not isinstance(exc, SimulationError) and outcome.value.error is not None:
                            record['error_type'] = outcome.value.error_type
                else:
                    self._record_from_result(record, report)
            records.append(record)
        return records

    @staticmethod
    def _record_run_error(record, exc):
        status = ('simulation_failure' if isinstance(exc, SimulationError) else
                  'netlist_error' if isinstance(exc, CircuitError) else
                  'unexpected_error')
        record.update(status=status, error=str(exc),
                      error_type=('SimulationError' if isinstance(exc, SimulationError)
                                  else type(exc).__name__))

    def _record_from_result(self, record: Dict, result: Any) -> Dict:
        """Fill in *record*'s status/metrics/feasible from a run's
        ``SpecReport`` (or legacy ``{'metrics': {...}}`` dict) -- the same
        classification :meth:`evaluate` applies after ``test.run()``
        returns.
        """
        metrics = (result.metrics if isinstance(result, SpecReport)
                  else result['metrics'])
        failed = {k: v for k, v in metrics.items()
                 if isinstance(v, str) and v.startswith('Error:')}
        if failed:
            record.update(status='metric_error', metrics=metrics,
                         error='; '.join(f"{k}: {v}" for k, v in failed.items()),
                         error_type='MetricExtractionError')
        else:
            record.update(status='success', metrics=metrics,
                         feasible=self._check_feasible(
                             metrics, record['applied_params'], record['constraints']))
        return record

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def feasible(self, metrics: Dict[str, Any],
                 params: Optional[Dict[str, Any]] = None) -> bool:
        """Check current effective limits; pass sample params for variable limits.

        Missing values and NaN fail. Spec operators preserve strictness;
        explicit min/max bounds are inclusive.
        """
        return self._check_feasible(metrics, params)

    def _check_feasible(self, metrics, params=None, constraints=None):
        ops = {'>': operator.gt, '>=': operator.ge, '<': operator.lt,
               '<=': operator.le, '==': operator.eq}
        if constraints is None:
            constraints = self._effective_constraints()
        for limit in constraints:
            val = self._value(limit['name'], limit['value_source'], metrics, params or {})
            if val is None or isinstance(val, str) or val != val:
                return False
            if 'op' in limit:
                if not ops[limit['op']](val, limit['target']):
                    return False
            else:
                if limit['min'] is not None and val < limit['min']:
                    return False
                if limit['max'] is not None and val > limit['max']:
                    return False
        return True

    # ------------------------------------------------------------------
    # History accessors
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[Dict]:
        """All evaluation records in chronological order."""
        return self._history

    @property
    def successes(self) -> List[Dict]:
        """Records with ``status == 'success'``."""
        return [r for r in self._history if r['status'] == 'success']

    @property
    def failures(self) -> List[Dict]:
        """Records with any non-success status."""
        return [r for r in self._history if r['status'] != 'success']

    @property
    def best(self) -> Optional[Dict]:
        """Best feasible record using weighted linear combination of objectives.

        For multi-objective optimization, combines all declared objectives
        using equal weights. Maximization objectives are negated so that
        ``min()`` finds the best design.

        Records with an absent, None, string or NaN value in any declared
        objective are excluded. Returns ``None`` if no eligible feasible
        successes exist yet or no objective has been declared. Infinity
        retains its existing scoring behavior.

        Eligibility uses current specs, margins and explicit constraints over
        recorded metrics and applied parameters (legacy params fallback).
        Selection returns the original record without rerunning simulations
        or changing historical feasible flags and constraint snapshots.
        summary()['best'] follows this selection; its feasible_count remains
        a count of historical feasible verdicts.
        """
        if not self._objectives:
            return None
        constraints = self._effective_constraints()
        candidates = []
        for record in self.successes:
            if not self._check_feasible(
                    record['metrics'], self._record_params(record), constraints):
                continue
            values = (self._value(name, self._objective_sources[name],
                                  record['metrics'], self._record_params(record))
                      for name in self._objectives)
            if any(val is None or isinstance(val, str) or val != val
                   for val in values):
                continue  # Every objective needs a rankable recorded value.
            candidates.append(record)
        if not candidates:
            return None

        def combined_score(record: Dict) -> float:
            """Compute weighted linear combination of all objectives."""
            score = 0.0
            n_objs = len(self._objectives)
            for obj_name, obj_cfg in self._objectives.items():
                val = self._value(obj_name, self._objective_sources[obj_name],
                                  record['metrics'], self._record_params(record))
                # Normalize to minimization: negate if maximizing
                if obj_cfg['maximize']:
                    val = -val
                # Equal weight across all objectives
                score += val / n_objs
            return score

        return min(candidates, key=combined_score)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the optimization run so far.

        ``'objectives'`` lists the declared objectives in declaration order
        (the order :attr:`best` uses): ``name``, ``goal`` (``'minimize'`` or
        ``'maximize'``) and ``value_source`` (``'metric'`` or ``'variable'``).
        """
        total    = len(self._history)
        by_status: Dict[str, int] = {}
        for r in self._history:
            by_status[r['status']] = by_status.get(r['status'], 0) + 1
        return {
            'total_evaluations': total,
            'by_status':         by_status,
            'feasible_count':    sum(1 for r in self.successes if r.get('feasible')),
            'best':              self.best,
            'objectives':        [
                {'name': name,
                 'goal': 'maximize' if cfg['maximize'] else 'minimize',
                 'value_source': self._objective_sources[name]}
                for name, cfg in self._objectives.items()
            ],
            'constraints':       self._effective_constraints(),
        }
