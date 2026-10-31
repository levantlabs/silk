"""Latest attempts, rather than the last usable number, govern reconciliation."""
import copy
from types import SimpleNamespace

import pytest

from silk.circuit.module_test import ModuleTestbench, _RunOutcome, _planned_outcome
from silk.exceptions import SimulationError, ConfigurationError
from silk.specs import Measurement, CalibrationBinding
from silk.specs.reconciler import reconcile
from tests.specs.test_testbench_integration import _module


def bench(tmp_path, monkeypatch):
    mod = _module(tmp_path)
    tb = mod.create_test('gain')
    tb.add_metric('gain', lambda c: 70.0)
    spec = tb.add_spec('Gain', 'gain', '>=', 60.)
    tb._create_netlist()
    tb._module_added = True
    def simulate(self, corner='tt'):
        self.sim_results_loaded = True
    monkeypatch.setattr(type(tb.netlist), 'simulate', simulate)
    return mod, tb, spec


def failure(spec, *, timestamp=200., kind='extractor_error', attempt_id='failed'):
    from silk.specs import SpecAttempt
    spec.record_attempt(SpecAttempt(attempt_id=attempt_id, source='spice',
                                    timestamp=timestamp, kind=kind, reason='missing gain'))


@pytest.mark.parametrize('value', ['error', None, float('nan'), complex(1, 2)])
@pytest.mark.parametrize('previous', [True, False])
def test_extraction_failure_is_latest_attempt(tmp_path, monkeypatch, value, previous):
    mod, tb, spec = bench(tmp_path, monkeypatch)
    if previous:
        tb.run()
    def extract(c):
        if value == 'error':
            raise RuntimeError('missing gain')
        return value
    tb._metrics[0]['extractor'] = extract
    report = tb.run()
    row, = mod.reconcile_specs().outcomes
    assert row.status == 'FAIL'
    assert row.measurement is None
    assert row.margin is None and row.calibrated is False
    assert row.latest_attempt.reason
    assert row.latest_attempt.timestamp > 0
    if previous:
        assert row.last_successful_attempt.measurement.value == 70
    else:
        assert row.last_successful_attempt is None
        assert 'no successful attempt' in mod.reconcile_specs().summary()
    assert report.passed is False
    tb._metrics[0]['extractor'] = lambda c: 80.
    tb.run()
    assert mod.reconcile_specs().outcomes[0].measurement.value == 80.


@pytest.mark.parametrize('entry', ['direct', 'module', 'all', 'corners', 'plan'])
@pytest.mark.parametrize('previous', [True, False])
def test_simulation_failure_reaches_each_adapter(tmp_path, monkeypatch, entry, previous):
    mod, tb, spec = bench(tmp_path, monkeypatch)
    if previous:
        tb.run()
    before = dict(spec.measurements)
    def fail(self, corner='tt'):
        raise SimulationError('synthetic simulation failure')
    monkeypatch.setattr(type(tb.netlist), 'simulate', fail)
    if entry == 'plan':
        tb.set_corners(['tt'])
    try:
        if entry == 'module':
            mod.run_test('gain')
        elif entry == 'all':
            mod.run_all_tests(n_workers=1)
        elif entry == 'corners':
            tb.run_corners(['tt'], n_workers=1)
        else:
            tb.run()
    except SimulationError:
        pass
    row, = mod.reconcile_specs().outcomes
    assert row.status == 'FAIL'
    assert 'synthetic simulation failure' in row.latest_attempt.reason
    assert spec.measurements == before
    assert (row.last_successful_attempt is not None) == previous


def test_never_attempted_and_preflight_refusal(tmp_path, monkeypatch):
    mod, tb, spec = bench(tmp_path, monkeypatch)
    assert mod.reconcile_specs().outcomes[0].status == 'UNMEASURED'
    def refuse(*args, **kwargs):
        raise ConfigurationError('bad declaration')
    monkeypatch.setattr(tb, '_prepare_run', refuse)
    with pytest.raises(ConfigurationError):
        tb.run()
    assert mod.reconcile_specs().outcomes[0].status == 'UNMEASURED'
    assert spec.attempt_history() == ()


def test_transport_failure_has_unknown_proved_conditions(tmp_path, monkeypatch):
    mod, tb, spec = bench(tmp_path, monkeypatch)
    prepared = tb._prepare_run('tt')
    outcome = _planned_outcome(SimpleNamespace(error='worker disappeared', value=None), prepared)
    with pytest.raises(RuntimeError):
        tb._consume_run_outcome(outcome)
    row, = mod.reconcile_specs().outcomes
    assert row.status == 'FAIL'
    assert row.latest_attempt.kind == 'transport_error'
    assert row.latest_attempt.proved_conditions is None
    assert row.latest_attempt.requested_conditions['corner'] == 'tt'


def test_snapshot_is_owned_and_legacy_passes_is_updated(tmp_path):
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60)
    m = Measurement('spice', 50., timestamp=100.)
    spec.record_measurement(m)
    first = mod.reconcile_specs().outcomes[0]
    assert m.passes is False
    m.value, m.timestamp, m.passes = 999., 999., True
    failure(spec)
    row = mod.reconcile_specs().outcomes[0]
    assert row.status == 'FAIL'
    assert row.last_successful_attempt.measurement.value == 50.
    assert row.last_successful_attempt.measurement.passes is False
    assert row.last_successful_attempt.timestamp == 100.
    assert first.measurement.value == 50.
    exposed = row.last_successful_attempt.measurement
    exposed.value = -1
    assert row.last_successful_attempt.measurement.value == 50.


def test_dotted_reads_do_not_publish_new_attempts(tmp_path):
    mod = _module(tmp_path)
    mod.add_net('out')
    mod.net_dict['out'].peak_current = 70.
    spec = mod.declare_spec('Gain', 'net.out.peak_current', '>=', 60.)
    first = mod.reconcile_specs().outcomes[0]
    before = spec.attempt_history()
    assert first.measurement.source == 'user'
    assert mod.reconcile_specs().outcomes[0].measurement.timestamp == first.measurement.timestamp
    assert spec.attempt_history() == before
    failure(spec)
    assert mod.reconcile_specs().outcomes[0].status == 'FAIL'
    assert mod.reconcile_specs().outcomes[0].status == 'FAIL'
    spec.record_measurement(Measurement('user', 80, timestamp=300))
    assert mod.reconcile_specs().outcomes[0].status == 'PASS'


def test_owner_scope_with_only_failures_and_later_recovery(tmp_path):
    from silk.circuit.test_placement import evidence_index
    from silk.specs.evidence import owner_scope
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60.)
    index = evidence_index(mod)
    for owner in ('$/a', '$/b'):
        with owner_scope(index, owner):
            spec.record_measurement(Measurement('spice', 70, timestamp=100))
    with owner_scope(index, '$/a'):
        failure(spec)
    rows = reconcile(mod, [('$/a', spec), ('$/b', spec)]).outcomes
    assert [r.status for r in rows] == ['FAIL', 'PASS']
    assert rows[0].last_successful_attempt.measurement.value == 70
    with owner_scope(index, '$/a'):
        spec.record_measurement(Measurement('spice', 80, timestamp=300))
    assert [r.status for r in reconcile(mod, [('$/a', spec), ('$/b', spec)]).outcomes] == ['PASS', 'PASS']


def test_failed_attempt_never_calibrates_and_prefer_source_filters_attempts(tmp_path):
    mod = _module(tmp_path)
    mod._behavioral_params = {'gain': 0.}
    spec = mod.declare_spec('Gain', 'gain', '>=', 60.)
    spec.calibration = CalibrationBinding(mod, 'gain')
    spec.record_measurement(Measurement('spice', 70, timestamp=100))
    failure(spec)
    row = mod.reconcile_specs(prefer_source='spice').outcomes[0]
    assert row.status == 'FAIL' and not row.calibrated
    assert mod._behavioral_params['gain'] == 0.


def test_attempt_id_deduplicates_and_rejects_conflicting_payload(tmp_path):
    from silk.specs import SpecAttempt
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60.)
    failure(spec)
    failure(spec)
    assert len(spec.attempt_history()) == 1
    with pytest.raises(ValueError, match='conflict'):
        spec.record_attempt(SpecAttempt(attempt_id='failed', source='spice', timestamp=200,
                                        kind='unavailable', reason='different'))


def test_retained_annotation_identity_and_changed_computation(tmp_path):
    from silk.circuit.modules_results import CurrentAnnotationReport
    from silk.specs.evidence import retain_annotations
    mod = _module(tmp_path)
    mod.add_net('out')
    mod.net_dict['out'].peak_current = 70.
    spec = mod.declare_spec('Gain', 'net.out.peak_current', '>=', 60.)
    report = CurrentAnnotationReport(instance_modules={'': mod}, window=(0.,))
    # Producer metadata is deliberately explicit; filenames are not identity.
    report._observation_id, report._observation_time = 'annotation-1', 100.
    retain_annotations(mod, report)
    first = mod.reconcile_specs().outcomes[0]
    assert first.measurement.source == 'user'
    assert first.latest_attempt.timestamp == 100.
    retain_annotations(mod, report)
    assert len(spec.attempt_history()) == 1
    from silk.circuit.test_placement import evidence_index
    from silk.specs.evidence import owner_scope
    with owner_scope(evidence_index(mod), '$'):
        failure(spec)
    assert mod.reconcile_specs().outcomes[0].status == 'FAIL'
    report2 = CurrentAnnotationReport(instance_modules={'': mod}, window=(1.,))
    report2._observation_id, report2._observation_time = 'annotation-2', 300.
    retain_annotations(mod, report2)
    assert mod.reconcile_specs().outcomes[0].status == 'PASS'
    assert len(spec.attempt_history()) == 3


def test_same_failed_outcome_is_consumed_for_two_placements(tmp_path, monkeypatch):
    from silk.circuit.test_placement import evidence_index
    from silk.specs.evidence import owner_scope
    mod, tb, spec = bench(tmp_path, monkeypatch)
    prepared = tb._prepare_run('tt')
    from silk.circuit.module_test import _simulate_run_outcome
    def fail(prepared):
        raise SimulationError('shared failed launch')
    outcome = _simulate_run_outcome(prepared, fail)
    assert outcome.replacement_started is False
    for owner in ('$/a', '$/b'):
        with owner_scope(evidence_index(mod), owner):
            with pytest.raises(SimulationError):
                tb._consume_run_outcome(outcome)
    rows = reconcile(mod, [('$/a', spec), ('$/b', spec)]).outcomes
    assert [r.status for r in rows] == ['FAIL', 'FAIL']
    assert all('shared failed launch' in r.latest_attempt.reason for r in rows)


@pytest.mark.parametrize('failed', [True, False])
def test_mc_sample_context_suppresses_attempts_before_consumption(tmp_path, monkeypatch, failed):
    mod, tb, spec = bench(tmp_path, monkeypatch)
    prepared = tb._prepare_run('tt', sample=SimpleNamespace(seed=1))
    outcome = _planned_outcome(SimpleNamespace(error='lost sample', value=None), prepared)
    assert outcome.attempt_context['sample'] == {'index': None, 'seed': 1}
    if failed:
        with pytest.raises(RuntimeError):
            tb._consume_run_outcome(outcome)
    else:
        # Successful sample evaluation already has a scalar-publication guard.
        tb.netlist.sim_results_loaded = True
        from silk.simulation.conditions import OperatingConditions
        tb.netlist.operating_conditions = OperatingConditions('silk', 'ngspice', corner='tt')
        tb._record_measurements = False
        tb.evaluate()
    assert spec.attempt_history() == ()
    assert mod.reconcile_specs().outcomes[0].status == 'UNMEASURED'


# These fixtures exercise the real sample planning and rerun paths with only
# the simulator replaced, as in the existing Monte Carlo contract tests.
from tests.test_monte_carlo import events, fake_sim, sky130_tb


def test_monte_carlo_success_failure_and_rerun_do_not_record_attempts(sky130_tb, fake_sim):
    tb = sky130_tb
    tb.set_monte_carlo(3)
    fake_sim.fail_on(('tt', 1))
    results = tb.run()
    spec = tb._specs[0]['spec']
    assert spec.attempt_history() == ()
    first = next(iter(results.reports))
    results.rerun(first)
    assert spec.attempt_history() == ()


def test_repeated_consume_is_one_attempt_and_success_has_request_context(tmp_path, monkeypatch):
    from silk.circuit.module_test import _simulate_run_outcome
    mod, tb, spec = bench(tmp_path, monkeypatch)
    prepared = tb._prepare_run('tt')
    successful = _simulate_run_outcome(prepared)
    tb._consume_run_outcome(successful)
    row = mod.reconcile_specs().outcomes[0]
    assert row.latest_attempt.requested_conditions['corner'] == 'tt'
    failed = _simulate_run_outcome(tb._prepare_run('tt'),
        lambda spec: (_ for _ in ()).throw(SimulationError('gone')))
    for _ in range(2):
        with pytest.raises(SimulationError):
            tb._consume_run_outcome(failed)
    assert len(spec.attempt_history()) == 2


def test_failure_reconciliation_updates_legacy_verdict_without_mutating_history(tmp_path):
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60.)
    m = Measurement('spice', 50., timestamp=100.)
    spec.record_measurement(m)
    m.value, m.passes = 70., None
    failure(spec)
    row = mod.reconcile_specs().outcomes[0]
    assert m.passes is True
    assert row.last_successful_attempt.measurement.passes is False


def test_last_success_selection_across_sources_and_explicit_preference(tmp_path):
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60.)
    spec.record_measurement(Measurement('spice', 65., timestamp=100.))
    spec.record_measurement(Measurement('behavioral', 70., timestamp=150.))
    failure(spec)
    row = mod.reconcile_specs().outcomes[0]
    assert row.last_successful_attempt.measurement.value == 70.
    row = mod.reconcile_specs(prefer_source='spice').outcomes[0]
    assert row.status == 'FAIL'
    assert row.last_successful_attempt.measurement.value == 65.


@pytest.mark.parametrize('overrides', [
    {'source': 'user'}, {'timestamp': 101.}, {'reason': 'failure with a value'},
    {'attempt_id': ''}, {'test_name': []},
])
def test_attempt_rejects_inconsistent_measurement_payload(overrides):
    from silk.specs import SpecAttempt
    fields = dict(source='spice', timestamp=100.,
                  measurement=Measurement('spice', 70., passes=True, timestamp=100.))
    fields.update(overrides)
    with pytest.raises(ValueError):
        SpecAttempt(**fields)


def test_unknown_corner_preflight_changes_no_attempt(tmp_path, monkeypatch):
    from silk.circuit.module_test import _simulate_run_outcome
    from silk.circuit.modules import Module
    real_simulate = Module.simulate
    mod, tb, spec = bench(tmp_path, monkeypatch)
    prepared = tb._prepare_run('tt')
    prepared.netlist.pdk_details.corners = {'tt': {}}
    outcome = _simulate_run_outcome(prepared,
        lambda prepared: real_simulate(prepared.netlist, corner='invalid'))
    with pytest.raises(ValueError, match='unknown corner'):
        tb._consume_run_outcome(outcome)
    assert spec.attempt_history() == ()
    assert mod.reconcile_specs().outcomes[0].status == 'UNMEASURED'


def test_transport_context_is_captured_before_mutable_netlist_changes(tmp_path, monkeypatch):
    from dataclasses import replace
    mod, tb, spec = bench(tmp_path, monkeypatch)
    prepared = replace(tb._prepare_run('tt'), declared_supply=1.8)
    prepared.netlist.simulator = 'behavioral'
    prepared.netlist._mc_sample = SimpleNamespace(index=3, seed=4)
    outcome = _planned_outcome(SimpleNamespace(error='worker disappeared', value=None), prepared)
    assert outcome.attempt_context['source'] == 'spice'
    assert outcome.attempt_context['suppressed'] is False
    assert outcome.attempt_context['requested_conditions']['voltage'] == 1.8


def test_one_reused_physical_failure_reaches_both_real_placements(tmp_path, monkeypatch):
    from tests.circuit.test_selected_testbench import leaf, root
    from silk.circuit.modules import Module
    child, tb = leaf(tmp_path)
    tb.add_spec('level', 'level', '>=', .5)
    top = root(tmp_path)
    top.add_module(child, 'a', out='out', vss='0')
    top.add_module(child, 'b', out='out', vss='0')
    launches = []
    def fail(module, **kwargs):
        launches.append(module)
        raise SimulationError('one shared launch failed')
    monkeypatch.setattr(Module, 'simulate', fail)
    top.run_all_tests(n_workers=1)
    rows = top.reconcile_specs().outcomes
    assert len(launches) == 1
    assert [row.owner_path for row in rows] == ['$/a', '$/b']
    assert [row.status for row in rows] == ['FAIL', 'FAIL']
    assert all('one shared launch failed' in row.latest_attempt.reason for row in rows)


@pytest.mark.parametrize('value', [None, float('nan'), 'error'])
def test_extraction_failure_stays_with_its_real_placement(tmp_path, monkeypatch, value):
    from tests.circuit.test_selected_testbench import leaf, root
    from silk.circuit.modules import Module
    child, tb = leaf(tmp_path)
    tb.add_spec('level', 'level', '>=', 60.)
    tb._metrics[0]['extractor'] = lambda c: 70.
    top = root(tmp_path)
    for name in ('a', 'b'):
        top.add_module(child, name, out='out', vss='0')
    launches = []
    def simulate(module, **kwargs):
        launches.append(module)
        module.sim_results_loaded = True
    monkeypatch.setattr(Module, 'simulate', simulate)
    top.run_all_tests(n_workers=1)
    first = top.reconcile_specs().outcomes
    assert [row.measurement.value for row in first] == [70., 70.]
    values = iter([value, 80.])
    def extract(c):
        result = next(values)
        if result == 'error':
            raise RuntimeError('one placement extraction failed')
        return result
    tb._metrics[0]['extractor'] = extract
    top.run_all_tests(n_workers=1)
    rows = top.reconcile_specs().outcomes
    assert len(launches) == 2  # One physical run per invocation, two consumers.
    assert [row.status for row in rows] == ['FAIL', 'PASS']
    assert rows[0].measurement is None
    assert rows[0].last_successful_attempt == first[0].latest_attempt
    assert rows[1].measurement.value == 80.
    tb._metrics[0]['extractor'] = lambda c: 90.
    top.run_all_tests(n_workers=1)
    assert [row.measurement.value for row in top.reconcile_specs().outcomes] == [90., 90.]


@pytest.mark.parametrize('value,verdict', [(70., True), (50., False)])
def test_public_measurement_attempt_accepts_owning_spec_verdict(tmp_path, value, verdict):
    from silk.specs import SpecAttempt
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60., units='dB')
    original = Measurement('spice', value, passes=verdict, timestamp=100.)
    attempt = SpecAttempt(source='spice', timestamp=100., measurement=original)
    spec.record_attempt(attempt)
    original.value, original.passes = 999., not verdict
    failure(spec)
    row, = mod.reconcile_specs().outcomes
    assert row.last_successful_attempt.measurement.value == value
    assert row.last_successful_attempt.measurement.passes is verdict
    assert ('PASS' if verdict else 'FAIL') in row.attempt_description


@pytest.mark.parametrize('value,verdict', [(50., True), (70., False)])
def test_public_measurement_attempt_refuses_contradictory_verdict(tmp_path, value, verdict):
    from silk.specs import SpecAttempt
    mod = _module(tmp_path)
    spec = mod.declare_spec('Gain', 'gain', '>=', 60., units='dB')
    failure(spec)
    before = spec.attempt_history()
    attempt = SpecAttempt(source='spice', timestamp=100., measurement=Measurement(
        'spice', value, passes=verdict, timestamp=100.))
    with pytest.raises(ValueError, match='verdict'):
        spec.record_attempt(attempt)
    assert spec.attempt_history() == before
    assert spec.measurements == {}
