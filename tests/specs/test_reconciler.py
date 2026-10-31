"""Reconciler: evaluate specs against measurements, build report."""
import pytest

from silk.specs import (
    CalibrationBinding,
    Measurement,
    Spec,
    SpecReconciliationReport,
)
from silk.specs.reconciler import reconcile


class _FakeModule:
    def __init__(self, name='m'):
        self.module_name = name
        self.net_dict = {}
        self.elem_dict = {}
        self._behavioral_params = None


def _spec(name='lna_NF', target=2.0, op='<=', parent=None, metric='noise_figure',
          module=None):
    return Spec(name=name, module=module or _FakeModule(),
                metric=metric, op=op, target=target, parent=parent)


def test_reconcile_empty_spec_list():
    r = reconcile(module=_FakeModule(), specs=[])
    assert isinstance(r, SpecReconciliationReport)
    assert r.outcomes == []
    assert r.pass_count == 0
    assert r.fail_count == 0


def test_reconcile_unmeasured_spec():
    mod = _FakeModule()
    s = _spec(module=mod)
    r = reconcile(module=mod, specs=[s])
    assert len(r.outcomes) == 1
    assert r.outcomes[0].status == 'UNMEASURED'
    assert r.outcomes[0].measurement is None
    assert r.outcomes[0].source_used is None


def test_reconcile_pass_with_margin():
    mod = _FakeModule()
    s = _spec(module=mod, target=2.0, op='<=')
    s.record_measurement(Measurement(source='spice', value=1.8, timestamp=100.0))
    r = reconcile(module=mod, specs=[s])
    assert r.outcomes[0].status == 'PASS'
    assert r.outcomes[0].source_used == 'spice'
    assert r.outcomes[0].margin == pytest.approx(0.2)
    assert r.outcomes[0].measurement.value == 1.8


def test_reconcile_fail_with_margin():
    mod = _FakeModule()
    s = _spec(module=mod, target=2.0, op='<=')
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=100.0))
    r = reconcile(module=mod, specs=[s])
    assert r.outcomes[0].status == 'FAIL'
    assert r.outcomes[0].margin == pytest.approx(0.3)


def test_reconcile_prefers_latest_across_sources():
    mod = _FakeModule()
    s = _spec(module=mod, target=2.0, op='<=')
    s.record_measurement(Measurement(source='behavioral', value=1.9, timestamp=100.0))
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=200.0))
    r = reconcile(module=mod, specs=[s])
    assert r.outcomes[0].source_used == 'spice'
    assert r.outcomes[0].status == 'FAIL'


def test_reconcile_prefer_source_override():
    mod = _FakeModule()
    s = _spec(module=mod, target=2.0, op='<=')
    s.record_measurement(Measurement(source='behavioral', value=1.9, timestamp=100.0))
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=200.0))
    r = reconcile(module=mod, specs=[s], prefer_source='behavioral')
    assert r.outcomes[0].source_used == 'behavioral'
    assert r.outcomes[0].status == 'PASS'


def test_reconcile_updates_measurement_passes_field():
    mod = _FakeModule()
    s = _spec(module=mod, target=2.0, op='<=')
    m = Measurement(source='spice', value=1.8, timestamp=100.0)
    s.record_measurement(m)
    reconcile(module=mod, specs=[s])
    assert m.passes is True


def test_reconcile_resolves_dotted_metric_from_net_state():
    """When metric is a dotted path, the resolver produces a Measurement
    from module.net_dict and the spec is reconciled as 'user'-sourced."""
    mod = _FakeModule()
    # Fake net with a peak_current field (Phase 2 would populate this).
    class _N:
        def __init__(self, pc):
            self.peak_current = pc
    mod.net_dict['vout'] = _N(pc=3.2e-3)
    s = Spec(name='vout_current', module=mod,
             metric='net.vout.peak_current',
             op='<=', target=5.0e-3)
    r = reconcile(module=mod, specs=[s])
    assert r.outcomes[0].status == 'PASS'
    assert r.outcomes[0].measurement.value == pytest.approx(3.2e-3)
    assert r.outcomes[0].measurement.source == 'user'  # dotted-path resolves as user-source
    assert r.outcomes[0].source_used == 'user'


def test_reconcile_dotted_metric_unresolved_is_unmeasured():
    mod = _FakeModule()   # no net_dict entry
    s = Spec(name='vout_current', module=mod,
             metric='net.vout.peak_current',
             op='<=', target=5.0e-3)
    r = reconcile(module=mod, specs=[s])
    assert r.outcomes[0].status == 'UNMEASURED'


from silk.specs import CalibrationBinding


def test_reconcile_calibration_writes_to_target_module_params():
    target = _FakeModule('lna')
    target._behavioral_params = {'NF': 2.0}  # architect placeholder
    spec_mod = _FakeModule('top')
    s = _spec(module=spec_mod, target=2.0, op='<=')
    s.calibration = CalibrationBinding(target_module=target, param_name='NF')
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=100.0))

    r = reconcile(module=spec_mod, specs=[s])

    assert target._behavioral_params['NF'] == 2.3
    assert r.outcomes[0].calibrated is True
    assert len(r.calibrations_applied) == 1
    assert 'NF' in r.calibrations_applied[0]
    assert 'lna' in r.calibrations_applied[0]


def test_reconcile_calibration_respects_source_filter():
    """A behavioral measurement must NOT trigger calibration by default."""
    target = _FakeModule('lna')
    target._behavioral_params = {'NF': 2.0}
    spec_mod = _FakeModule('top')
    s = _spec(module=spec_mod, target=2.0, op='<=')
    s.calibration = CalibrationBinding(target_module=target, param_name='NF')
    s.record_measurement(Measurement(source='behavioral', value=2.3, timestamp=100.0))

    r = reconcile(module=spec_mod, specs=[s])

    assert target._behavioral_params['NF'] == 2.0  # untouched
    assert r.outcomes[0].calibrated is False
    assert r.calibrations_applied == []


def test_reconcile_calibration_refuses_when_target_has_no_params_dict():
    """A binding to a module with no behavioral parameters is a declaration
    error: refused like Module.calibrate_behavioral (RuntimeError), not skipped."""
    target = _FakeModule('lna')
    target._behavioral_params = None
    spec_mod = _FakeModule('top')
    s = _spec(module=spec_mod, target=2.0, op='<=')
    s.calibration = CalibrationBinding(target_module=target, param_name='NF')
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=100.0))

    with pytest.raises(RuntimeError, match=r"'NF' on module 'lna'.*has no declared parameters"):
        reconcile(module=spec_mod, specs=[s])

    assert target._behavioral_params is None


def test_reconcile_calibration_refuses_when_param_name_missing():
    """A binding naming a parameter the model does not declare is refused
    (KeyError, as Module.calibrate_behavioral does) and nothing is written."""
    target = _FakeModule('lna')
    target._behavioral_params = {'gain': 15.0, 'vth0': 0.4}
    spec_mod = _FakeModule('top')
    s = _spec(module=spec_mod, target=2.0, op='<=')
    s.calibration = CalibrationBinding(target_module=target, param_name='vto')
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=100.0))

    with pytest.raises(KeyError) as excinfo:
        reconcile(module=spec_mod, specs=[s])

    message = str(excinfo.value)
    assert "'vto'" in message and "'lna'" in message
    assert "['gain', 'vth0']" in message  # the parameters the model does declare
    assert target._behavioral_params == {'gain': 15.0, 'vth0': 0.4}  # not auto-created


def test_reconcile_calibration_refusal_writes_no_value_when_another_binding_is_valid():
    """Preflight: a valid binding listed before an invalid one must not be written."""
    target = _FakeModule('lna')
    target._behavioral_params = {'NF': 2.0, 'gain': 15.0}
    spec_mod = _FakeModule('top')
    good = _spec(name='lna_NF', module=spec_mod, target=2.0, op='<=')
    good.calibration = CalibrationBinding(target_module=target, param_name='NF')
    good.record_measurement(Measurement(source='spice', value=2.3, timestamp=100.0))
    bad = _spec(name='lna_gain', module=spec_mod, target=20.0, op='>=')
    bad.calibration = CalibrationBinding(target_module=target, param_name='vto')
    bad.record_measurement(Measurement(source='spice', value=18.0, timestamp=100.0))

    with pytest.raises(KeyError, match="'vto'"):
        reconcile(module=spec_mod, specs=[good, bad])

    assert target._behavioral_params == {'NF': 2.0, 'gain': 15.0}


def test_reconcile_calibration_deliberate_skips_stay_silent():
    """No binding, no measurement yet, and a source outside source_filter are
    not declaration errors: they skip without raising, even for a name the
    model does not declare."""
    target = _FakeModule('lna')
    target._behavioral_params = {'NF': 2.0}
    spec_mod = _FakeModule('top')

    no_binding = _spec(name='no_binding', module=spec_mod, target=2.0, op='<=')
    no_binding.record_measurement(Measurement(source='spice', value=2.3, timestamp=100.0))

    no_measurement = _spec(name='no_measurement', module=spec_mod, target=2.0, op='<=')
    no_measurement.calibration = CalibrationBinding(target_module=target, param_name='vto')

    outside_filter = _spec(name='outside_filter', module=spec_mod, target=2.0, op='<=')
    outside_filter.calibration = CalibrationBinding(target_module=target, param_name='vto')
    outside_filter.record_measurement(Measurement(source='behavioral', value=2.3, timestamp=100.0))

    r = reconcile(module=spec_mod, specs=[no_binding, no_measurement, outside_filter])

    assert [o.status for o in r.outcomes] == ['FAIL', 'UNMEASURED', 'FAIL']
    assert r.calibrations_applied == []
    assert target._behavioral_params == {'NF': 2.0}


def test_reconcile_calibration_declared_param_still_writes():
    """The refusal does not touch a binding to a declared parameter."""
    target = _FakeModule('lna')
    target._behavioral_params = {'NF': 2.0}
    spec_mod = _FakeModule('top')
    s = _spec(module=spec_mod, target=2.0, op='<=')
    s.calibration = CalibrationBinding(target_module=target, param_name='NF')
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=100.0))

    r = reconcile(module=spec_mod, specs=[s])

    assert target._behavioral_params['NF'] == 2.3
    assert r.outcomes[0].calibrated is True


def test_reconcile_calibration_only_fires_for_fresh_latest_measurement():
    """Calibration fires on the measurement used for reconciliation
    (prefer_source or latest), not on every stored measurement."""
    target = _FakeModule('lna')
    target._behavioral_params = {'NF': 2.0}
    spec_mod = _FakeModule('top')
    s = _spec(module=spec_mod, target=2.0, op='<=')
    s.calibration = CalibrationBinding(target_module=target, param_name='NF')
    s.record_measurement(Measurement(source='behavioral', value=1.9, timestamp=100.0))
    s.record_measurement(Measurement(source='spice', value=2.3, timestamp=200.0))

    # prefer behavioral -> should NOT calibrate (source filter excludes behavioral)
    r = reconcile(module=spec_mod, specs=[s], prefer_source='behavioral')
    assert target._behavioral_params['NF'] == 2.0

    # Natural latest -> spice, calibration fires
    r = reconcile(module=spec_mod, specs=[s])
    assert target._behavioral_params['NF'] == 2.3


def test_declaration_calibration_preflights_all_bindings():
    from silk.exceptions import VariationError
    from silk.simulation.variation import Normal
    mod = _FakeModule()
    declaration = Normal(0.0, 0.001)
    mod._behavioral_params = {'gain': 1.0, 'offset': declaration}
    specs = []
    for param in ('gain', 'offset'):
        spec = _spec(name=param, module=mod)
        spec.calibration = CalibrationBinding(target_module=mod, param_name=param)
        spec.record_measurement(Measurement(source='spice', value=0.5, timestamp=1.0))
        specs.append(spec)
    with pytest.raises(VariationError, match='offset.*field'):
        reconcile(module=mod, specs=specs)
    assert mod._behavioral_params == {'gain': 1.0, 'offset': declaration}


def test_declaration_calibration_source_filter_still_skips():
    from silk.simulation.variation import Normal
    mod = _FakeModule()
    declaration = Normal(0.0, 0.001)
    mod._behavioral_params = {'offset': declaration}
    spec = _spec(module=mod)
    spec.calibration = CalibrationBinding(target_module=mod, param_name='offset')
    spec.record_measurement(Measurement(source='behavioral', value=0.5, timestamp=1.0))
    report = reconcile(module=mod, specs=[spec])
    assert not report.calibrations_applied
    assert mod._behavioral_params['offset'] is declaration


@pytest.mark.parametrize('metric', ['gain', 'net.out.peak_current'])
def test_outcomes_retain_owner_for_measured_and_unmeasured_specs(metric):
    from types import SimpleNamespace
    mod = _FakeModule()
    if metric.startswith('net.'):
        mod.net_dict['out'] = SimpleNamespace(peak_current=3.)
    spec = _spec(module=mod, metric=metric)
    if metric == 'gain':
        spec.record_measurement(Measurement('user', 3.))
    unmeasured = _spec('empty', module=mod, metric='missing')
    report = reconcile(mod, [('$/a', spec), ('$/b', spec), ('$/b', unmeasured)])
    assert [o.owner_path for o in report.outcomes] == ['$/a', '$/b', '$/b']
    assert [o.report_key for o in report.outcomes] == [None, None, None]
    assert [o.status for o in report.outcomes] == ['FAIL', 'FAIL', 'UNMEASURED']
    assert '(instance a)' in report.summary() and '(instance b)' in report.summary()
    assert '$' not in report.summary()


def test_originating_report_key_follows_selected_measurement(tmp_path):
    from pathlib import Path
    from tests.circuit.test_selected_testbench import leaf
    child, tb = leaf(tmp_path)
    failing = False
    def extract(_):
        if failing:
            raise ValueError('latest attempt failed')
        return 70.
    tb.add_metric('gain', extract)
    spec = tb.add_spec('Gain', 'gain', '>=', 60, key='gain')
    try:
        tb.evaluate(rawfile=Path(__file__).parents[1] / 'fixtures/rawfiles/e_ng_tran.raw')
        failing = True
        tb.evaluate(rawfile=Path(__file__).parents[1] / 'fixtures/rawfiles/e_ng_tran.raw')
        row, = child.reconcile_specs().outcomes
        assert row.report_key == 'leaf.step'
        assert row.status == 'FAIL'
        assert row.measurement is None
        assert row.last_successful_attempt.measurement.value == 70
        spec.record_measurement(Measurement('user', 65, timestamp=0))
        from silk.specs.evidence import owner_scope
        from silk.circuit.test_placement import evidence_index
        # Manual evidence must be recorded in the same owner scope.
        with owner_scope(evidence_index(child), '$'):
            spec.record_measurement(Measurement('user', 65, timestamp=0))
        manual, = child.reconcile_specs(prefer_source='user').outcomes
        assert manual.measurement.value == 65 and manual.report_key is None
    finally:
        child.close()


@pytest.mark.parametrize('params', [None, {}])
@pytest.mark.parametrize('apply', [False, True])
def test_nan_skips_before_target_validation(params, apply):
    from silk.specs.reconciler import _apply_calibration
    target = _FakeModule()
    target._behavioral_params = params
    spec = _spec(module=target)
    spec.calibration = CalibrationBinding(target, 'missing')
    log = []
    assert _apply_calibration(spec, Measurement('spice', float('nan')),
                              log, apply=apply) is False
    assert log == []
    assert target._behavioral_params is params


@pytest.mark.parametrize('value', [float('inf'), -float('inf')])
def test_signed_infinity_still_calibrates_scalar(value):
    target = _FakeModule()
    target._behavioral_params = {'NF': 1.0}
    spec = _spec(module=target)
    spec.calibration = CalibrationBinding(target, 'NF')
    spec.record_measurement(Measurement('spice', value))
    report = reconcile(target, [spec])
    assert target._behavioral_params['NF'] == value
    assert report.outcomes[0].calibrated is True
    assert len(report.calibrations_applied) == 1
