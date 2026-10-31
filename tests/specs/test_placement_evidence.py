"""Shared declarations retain evidence for each hierarchy owner."""
import gc
import weakref

import pytest

from tests._env import needs_ngspice
from tests.circuit.test_selected_testbench import leaf, root


@needs_ngspice
def test_reconcile_selects_root_before_latest_and_drops_dead_scope(tmp_path):
    child, tb = leaf(tmp_path)
    spec = tb.add_spec('level limit', 'level', '>=', 1.5)
    a, b, unrun = [root(tmp_path) for _ in range(3)]
    for top, voltage in zip((a, b, unrun), (1, 2, 3)):
        top.set_supply_voltage(voltage)
        top.add_module(child, 'x1', out='out', vss='0')
    try:
        a.run_all_tests()
        b.run_all_tests()
        assert a.reconcile_specs().outcomes[0].status == 'FAIL'
        assert b.reconcile_specs().outcomes[0].status == 'PASS'
        assert unrun.reconcile_specs().outcomes[0].status == 'UNMEASURED'
        assert spec.latest().value == pytest.approx(2)
        index_ref = weakref.ref(a.__dict__['_test_evidence'])
        raw_ref = weakref.ref(next(iter(a.__dict__['_test_evidence'].records.values())).results._raw)
        measurement_ref = weakref.ref(a.reconcile_specs().outcomes[0].measurement)
        root_ref = weakref.ref(a)
        a.close()
        del a
        gc.collect()
        assert root_ref() is None
        assert index_ref() is None
        assert raw_ref() is None
        assert measurement_ref() is None
        assert spec.latest().value == pytest.approx(2)
    finally:
        b.close()
        unrun.close()
        child.close()


@needs_ngspice
def test_shared_leaf_reconciliation_keeps_both_placements(tmp_path):
    child, tb = leaf(tmp_path)
    tb.add_spec('level limit', 'level', '>=', 1.5)
    top = root(tmp_path)
    carriers = []
    try:
        for name, voltage in [('low', 1), ('high', 2)]:
            carrier = root(tmp_path)
            carrier.set_supply_voltage(voltage)
            carrier.define_module_name('carrier')
            carrier.define_module_ports({})
            carrier.add_module(child, 'cell', out='out', vss='0')
            top.add_module(carrier, name, nets={})
            carriers.append(carrier)
        reports = list(top.run_all_tests().values())
        assert [r.passed for r in reports] == [False, True]
        outcomes = top.reconcile_specs().outcomes
        assert [o.status for o in outcomes] == ['FAIL', 'PASS']
        assert [o.measurement.value for o in outcomes] == pytest.approx([1, 2])
    finally:
        top.close()
        child.close()
        for carrier in carriers:
            carrier.close()


@pytest.mark.parametrize('case', ['conflict', 'agree', 'filtered', 'distinct_targets'])
def test_calibration_stages_every_candidate_before_writing(tmp_path, case):
    from silk.specs import Spec, Measurement, CalibrationBinding
    from silk.specs.reconciler import reconcile
    from silk.exceptions import ConfigurationError
    top = root(tmp_path)
    first = root(tmp_path)
    shared = root(tmp_path)
    other = root(tmp_path)
    for target in (first, shared, other):
        target.define_module_name('same_name')
        target._behavioral_params = {'gain': -1}
    specs = []
    for name, target, value, source in [
            ('unrelated', first, 3, 'spice'),
            ('a', shared, 1, 'spice'),
            ('b', other if case == 'distinct_targets' else shared,
             1 if case == 'agree' else 2,
             'behavioral' if case == 'filtered' else 'spice')]:
        spec = Spec(name, top, 'gain', '>=', 10)
        spec.calibration = CalibrationBinding(target, 'gain')
        spec.record_measurement(Measurement(source, value, timestamp=1))
        specs.append(('$/'+name, spec))
    try:
        if case == 'conflict':
            with pytest.raises(ConfigurationError, match=r'\$/a.*\$/b'):
                reconcile(top, specs)
            assert first._behavioral_params['gain'] == -1
            assert shared._behavioral_params['gain'] == -1
        else:
            report = reconcile(top, specs)
            assert first._behavioral_params['gain'] == 3
            assert shared._behavioral_params['gain'] == 1
            if case == 'distinct_targets':
                assert other._behavioral_params['gain'] == 2
            assert all(outcome.status == 'FAIL' for outcome in report.outcomes)
    finally:
        for module in (top, first, shared, other):
            module.close()


@pytest.mark.parametrize('kind', ['net', 'device'])
def test_dotted_metrics_use_owner_annotations_without_rewriting_shared_net(tmp_path, kind):
    from silk.circuit.modules_results import CurrentAnnotationReport
    from silk.simulation._worker_pickle import copy_simulation_netlist
    from silk.specs.evidence import retain_annotations
    child, _ = leaf(tmp_path)
    metric = 'net.out.peak_current' if kind == 'net' else 'device.r.peak_current'
    child.declare_spec('current', metric, '<=', 3)
    top = root(tmp_path)
    for name in ('a', 'b', 'unmeasured'):
        top.add_module(child, name, out=name, vss='0')
    def subject(module):
        return module.net_dict['out'] if kind == 'net' else module.elem_dict['r']['device']
    subject(child).peak_current = 99
    report = CurrentAnnotationReport()
    for name, value in [('a', 2), ('b', 5)]:
        snapshot = copy_simulation_netlist(child)
        subject(snapshot).peak_current = value
        report.instance_modules['x' + name] = snapshot
    try:
        retain_annotations(top, report)
        outcomes = top.reconcile_specs().outcomes
        assert [o.status for o in outcomes] == ['PASS', 'FAIL', 'UNMEASURED']
        assert [o.measurement.value for o in outcomes[:2]] == [2, 5]
        assert subject(child).peak_current == 99
        other = root(tmp_path)
        try:
            other.add_module(child, 'a', out='out', vss='0')
            assert other.reconcile_specs().outcomes[0].status == 'UNMEASURED'
        finally:
            other.close()
    finally:
        top.close()
        child.close()


@needs_ngspice
def test_single_placement_public_spec_reads_keep_main_contract(tmp_path):
    from silk.specs import Measurement
    from silk.simulation.conditions import ConditionState
    top = root(tmp_path)
    modules = []
    specs = []
    try:
        for name, value in [('a', 1.0), ('b', 2.0)]:
            child, tb = leaf(tmp_path / name, name=name)
            tb.netlist.sig_sources_dict['rail'].dc = value
            spec = tb.add_spec('level', 'level', '>=', 1.5)
            top.add_module(child, name, out=name, vss='0')
            modules.append(child)
            specs.append(spec)
        top.run_all_tests()
        report = top.reconcile_specs()
        assert report.pass_count == report.fail_count == 1
        assert report.unmeasured == []
        assert [outcome.status for outcome in report.outcomes] == ['FAIL', 'PASS']
        for spec, value, passed in zip(specs, (1.0, 2.0), (False, True)):
            measurement = spec.latest()
            assert isinstance(measurement, Measurement)
            assert measurement is spec.latest(source='spice')
            assert measurement.value == value
            assert measurement.passes is passed
            assert spec.evaluate(measurement) is passed
            assert spec.latest(source='behavioral') is None
            assert list(spec.measurements) == [('spice', ConditionState.NONE)]
            assert spec.measurements[('spice', ConditionState.NONE)] is measurement
            assert list(spec.measurements.values()) == [measurement]
        serialized = report.as_dict()
        assert [o['measurement']['value'] for o in serialized['outcomes']] == [1, 2]
        assert all('owner' not in key for o in serialized['outcomes']
                   for key in o['measurement'])
        assert '[FAIL] level (instance a): 1 >=1.5' in report.summary()
        assert '[PASS] level (instance b): 2 >=1.5' in report.summary()
    finally:
        top.close()
        for child in modules:
            child.close()


def test_serialization_drops_runtime_scopes_but_keeps_manual_measurements(tmp_path):
    import pickle
    from copy import deepcopy
    from silk.specs import Measurement
    from silk.specs.evidence import bind_measurement
    from silk.circuit.test_placement import evidence_index
    top = root(tmp_path)
    try:
        spec = top.declare_spec('level', 'level', '>=', 1)
        spec.record_measurement(Measurement('user', 3))
        spec.record_measurement(bind_measurement(spec, Measurement('spice', 2),
                                                evidence_index(top), '$'))
        for clone in (pickle.loads(pickle.dumps(spec)), deepcopy(spec)):
            assert clone.latest(source='spice') is None
            assert clone.latest(source='user').value == 3
            assert '_scope_references' not in vars(clone)
            assert '_test_evidence' not in vars(clone.module)
        assert spec.latest(source='spice').value == 2
    finally:
        top.close()
