"""Fresh per-run spec evaluation for :class:`silk.faults.FaultCampaign`.

The campaign binds to the one testbench whose netlist *is* the campaign module
(:func:`resolve_binding`), then scores every run's own rawfile through that
bench's ordinary ``evaluate(rawfile=...)`` path (:func:`evaluate_run`) inside a
private, non-publishing selection. Nothing recorded earlier on a ``Spec`` is
ever read, so a verdict always follows the run's own output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from silk.exceptions import ConfigurationError

#: Marker stored on :class:`silk.faults.RunResult` for runs scored here.
EVALUATION_FRESH = 'fresh-v1'
#: Marker for ``reconcile_specs=False`` runs: simulated, nothing scored.
EVALUATION_EXECUTION_ONLY = 'execution-only'


@dataclass(frozen=True)
class _SpecView:
    name: Optional[str]
    key: Optional[str]


@dataclass(frozen=True)
class CampaignSpecOutcome:
    """One spec's verdict for one run, extracted from that run's own output.

    A frozen record of scalars; it holds no live circuit. ``status`` is
    ``'PASS'``, ``'FAIL'`` or ``'UNMEASURED'``. ``value`` is the fresh real
    measurement, or ``None`` when no usable value exists (an extractor error,
    NaN, or an unbound spec). ``attempt_kind`` is ``'measured'``,
    ``'extractor_error'``, ``'unavailable'`` or ``'unbound'``. ``checked``
    means evaluation was attempted for this row, not that extraction worked.
    """
    owner_path: str
    spec_key: Optional[str]
    spec_name: Optional[str]
    test_name: Optional[str]
    run_id: int
    seed: int
    backend: Optional[str]
    requested_corner: Optional[str]
    proved_conditions: Optional[dict]
    status: str
    value: Optional[float]
    attempt_kind: str
    reason: Optional[str] = None
    warnings: Tuple[str, ...] = ()
    checked: bool = True
    report_key: Optional[str] = None

    @property
    def spec(self):
        return _SpecView(self.spec_name, self.spec_key)

    @property
    def measurement(self):
        """A detached :class:`~silk.specs.Measurement` of the fresh value, or
        ``None`` when this run produced no usable value. Built on access from
        this row alone; it is never a recorded nominal Measurement."""
        if self.value is None:
            return None
        from silk.simulation.conditions import ConditionState
        from silk.specs.spec import Measurement
        proved = self.proved_conditions or {}
        corner = (ConditionState.NONE if proved.get('corner_kind') == 'none'
                  else proved.get('corner'))
        return Measurement(
            'behavioral' if self.backend == 'behavioral' else 'spice',
            self.value, passes=self.status == 'PASS', corner=corner)

    @property
    def identity(self):
        return (self.owner_path, self.spec_key)


@dataclass
class CampaignBinding:
    """Which bench and which specs a campaign scores (see module docstring)."""
    bench: Any = None
    dut_path: Optional[str] = None
    checked: List[Tuple[Any, dict]] = field(default_factory=list)
    unbound: List[Tuple[str, Any]] = field(default_factory=list)
    excluded: List[dict] = field(default_factory=list)

    @property
    def test_name(self):
        return None if self.bench is None else self.bench.test_name


def _owner_bench(spec):
    from silk.circuit.test_placement import authored_test
    bench = getattr(spec, '_owning_testbench', None)
    return None if bench is None else authored_test(bench)


def resolve_binding(module) -> CampaignBinding:
    """Resolve the campaign bench and classify every spec in *module*'s tree."""
    from silk.circuit.modules_results import DUT_ELEMENT_NAME
    from silk.circuit.test_placement import authored_test

    instances = list(module.iter_instances(include_self=True))
    benches = {}
    for instance in instances:
        candidates = list(instance.module._tests.values())
        candidates += [b for b in (getattr(s, '_owning_testbench', None)
                                   for s in instance.module._specs)
                       if b is not None]
        for candidate in candidates:
            bench = authored_test(candidate)
            if getattr(bench, 'netlist', None) is module:
                benches[id(bench)] = bench
    if len(benches) > 1:
        raise ConfigurationError(
            'FaultCampaign found more than one testbench whose netlist is '
            'the campaign module: ' + ', '.join(sorted(
                repr(b.test_name) for b in benches.values())))
    bench = next(iter(benches.values()), None)
    binding = CampaignBinding(bench=bench)
    bench_specs = {}
    if bench is not None:
        bench_specs = {id(e['spec']): e for e in bench._specs
                       if e.get('spec') is not None}

    for instance in instances:
        for spec in instance.module._specs:
            entry = bench_specs.get(id(spec))
            if entry is not None:
                if instance.module is bench.module and instance.path == DUT_ELEMENT_NAME:
                    binding.dut_path = instance.owner_path
                    binding.checked.append((instance.owner_path, entry))
                else:
                    binding.excluded.append(_excluded(
                        instance.owner_path, spec,
                        f'placement {instance.owner_path!r} is not the '
                        f"campaign bench's DUT placement"))
                continue
            owner = _owner_bench(spec)
            if owner is not None:
                binding.excluded.append(_excluded(
                    instance.owner_path, spec,
                    f'owned by test {owner.test_name!r}, whose deck this '
                    f'campaign does not simulate'))
            else:
                binding.unbound.append((instance.owner_path, spec))
    return binding


def _excluded(owner_path, spec, reason):
    return {'owner_path': owner_path, 'spec_key': spec.key,
            'spec_name': spec.name, 'checked': False, 'reason': reason}


def preflight(binding: CampaignBinding) -> None:
    """Refuse an inspectable bad metric binding before any run launches."""
    if binding.bench is not None:
        binding.bench._require_unambiguous_analyses()


@dataclass
class CampaignRun:
    """The producing-run facts one evaluation reads; private to the campaign."""
    backend: str
    rawfile_path: str
    label: str
    reader: Any = None


def _run_reader(bench, backend):
    from silk.circuit.modules_sim import _invalidate_current_results
    from silk.circuit.test_placement import EvidenceResults
    from silk.simulation._worker_pickle import copy_simulation_netlist
    from silk.simulation.backend import SimulatorBackend

    snapshot = copy_simulation_netlist(bench.netlist)
    _invalidate_current_results(snapshot)
    snapshot._prepared_deck_record = None
    snapshot._mc_sample = None
    reader = EvidenceResults(snapshot, authored=bench.netlist)
    reader.simulator = backend
    reader.backend = SimulatorBackend.get(backend)
    return reader


def evaluate_run(binding: CampaignBinding, *, run_id: int, seed: int,
                 backend: str, rawfile_path: str, label: str,
                 no_pdk_device: bool):
    """Score one run's rawfile. Returns ``(outcomes, excluded)``.

    Raises whatever the bench's evaluation raises (a configuration error, an
    unreadable rawfile); the caller turns that into an ``errored`` run.
    """
    from silk.circuit.test_placement import campaign_evaluation
    from silk.simulation.conditions import ConditionState

    bench = binding.bench
    outcomes: List[CampaignSpecOutcome] = []
    if bench is not None and binding.checked:
        run = CampaignRun(backend=backend, rawfile_path=rawfile_path,
                          label=label, reader=_run_reader(bench, backend))
        proved = ConditionState.NONE if no_pdk_device else label
        with campaign_evaluation(bench, run):
            report = bench.evaluate(corner=proved, rawfile=rawfile_path)
        conditions = report.operating_conditions
        outcomes = _classify_rows(
            binding, report, run_id=run_id, seed=seed, backend=backend,
            label=label,
            proved=None if conditions is None else conditions.to_dict())
    excluded = [dict(e, run_id=run_id, seed=seed, backend=backend,
                     requested_corner=label, proved_conditions=None)
                for e in binding.excluded]
    for owner_path, spec in binding.unbound:
        excluded.append(dict(_excluded(
            owner_path, spec,
            'no metric binding: the spec is not declared on the campaign '
            'bench, so no fresh value can be extracted'),
            run_id=run_id, seed=seed, backend=backend, requested_corner=label,
            proved_conditions=None))
        outcomes.append(CampaignSpecOutcome(
            owner_path=owner_path, spec_key=spec.key, spec_name=spec.name,
            test_name=None, run_id=run_id, seed=seed, backend=backend,
            requested_corner=label, proved_conditions=None,
            status='UNMEASURED', value=None, attempt_kind='unbound',
            reason='no metric binding: the spec is not declared on the '
                   "campaign bench, so no fresh value can be extracted",
            checked=False))
    return outcomes, excluded


def _classify_rows(binding, report, *, run_id, seed, backend, label, proved):
    from silk.specs.attempt import classify_metric

    results = {r.spec_key: r for r in report.results}
    rows = []
    for owner_path, entry in binding.checked:
        spec = entry['spec']
        metric = entry['metric']
        kind, value, reason = classify_metric(
            report.metrics.get(metric), metric=metric,
            metric_errors=report.metric_errors,
            metric_warnings=report.metric_warnings)
        result = results.get(spec.key)
        if kind == 'measured':
            status = 'PASS' if (result is not None and result.passed) else 'FAIL'
        else:
            status = 'FAIL'
        warnings = tuple(w.message for w in report.metric_warnings.get(metric, ()))
        rows.append(CampaignSpecOutcome(
            owner_path=owner_path, spec_key=spec.key, spec_name=spec.name,
            test_name=binding.test_name, run_id=run_id, seed=seed,
            backend=backend, requested_corner=label, proved_conditions=proved,
            status=status, value=value, attempt_kind=kind, reason=reason,
            warnings=warnings, checked=True, report_key=report.report_key))
    return rows


def run_status(outcomes) -> str:
    """The run status for an evaluated run: passed, failed or unverified."""
    checked = [o for o in outcomes if o.checked]
    if any(o.status == 'FAIL' for o in checked):
        return 'failed'
    unmeasured = any(o.status == 'UNMEASURED' for o in outcomes)
    if checked and not unmeasured and all(o.status == 'PASS' for o in checked):
        return 'passed'
    return 'unverified'


def excluded_to_json(entry: dict) -> dict:
    """``outcome.json`` row for a not-checked spec: no verdict, no value."""
    return {
        'spec_name': entry['spec_name'], 'spec_key': entry['spec_key'],
        'owner_path': entry['owner_path'], 'report_key': None,
        'status': None, 'measurement_value': None, 'checked': False,
        'test_name': None, 'attempt_kind': 'not_checked',
        'reason': entry['reason'], 'warnings': [],
        'run_id': entry.get('run_id'), 'seed': entry.get('seed'),
        'backend': entry.get('backend'),
        'requested_corner': entry.get('requested_corner'),
        'proved_conditions': None,
    }


def outcome_to_json(o) -> dict:
    """JSON projection of a :class:`CampaignSpecOutcome`."""
    return {
        'spec_name': o.spec_name, 'spec_key': o.spec_key,
        'owner_path': o.owner_path, 'report_key': o.report_key,
        'status': o.status, 'measurement_value': o.value,
        'checked': o.checked, 'test_name': o.test_name,
        'attempt_kind': o.attempt_kind, 'reason': o.reason,
        'warnings': list(o.warnings), 'run_id': o.run_id, 'seed': o.seed,
        'backend': o.backend, 'requested_corner': o.requested_corner,
        'proved_conditions': o.proved_conditions,
    }
