"""ModuleTestbench integration with first-class Spec objects (Task 15)."""
import os
import time

import pytest

from silk.circuit.modules import Module
from silk.circuit.module_test import ModuleTestbench
from silk.exceptions import DuplicateNameError
from silk.specs import CalibrationBinding, Measurement, Spec


_PDK_YAML = os.path.join(
    os.path.dirname(__file__), '..', '..', 'silk', 'pdk', 'data',
    'skywater_130_device_details.yaml'
)


def _pdk():
    try:
        from silk.pdk.basepdk import BasePDK
    except Exception:
        pytest.skip('PDK not available')
    return BasePDK(_PDK_YAML, library_location='/dev/null')


def _module(tmp_path, name='m'):
    return Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
                  pdk_details=_pdk(), module_name=name)


def test_testbench_add_spec_returns_spec_object(tmp_path):
    m = _module(tmp_path)
    tb = ModuleTestbench(m, name='unit')
    tb.add_metric('gain_dB', lambda ckt: 65.0)
    result = tb.add_spec('DC gain > 60 dB', 'gain_dB', '>', 60.0)
    assert isinstance(result, Spec)
    assert result.name == 'DC gain > 60 dB'
    assert result.metric == 'gain_dB'
    assert result.op == '>'
    assert result.target == 60.0
    assert result.module is m
    assert result in m._specs
    # Back-reference to the testbench is set so chain forwarding works.
    assert result._owning_testbench is tb
    # Legacy dict entry is preserved with the new 'spec' key.
    assert len(tb._specs) == 1
    assert tb._specs[0]['spec'] is result


def test_testbench_chain_pattern_preserved(tmp_path):
    """Legacy chaining ``tb.add_spec(...).add_spec(...)`` still works via
    Spec.add_spec forwarder."""
    m = _module(tmp_path)
    tb = ModuleTestbench(m, name='unit')
    tb.add_metric('a', lambda ckt: 0.0)
    tb.add_metric('b', lambda ckt: 0.0)
    result = (tb
        .add_spec('a_spec', 'a', '>', 0.0)
        .add_spec('b_spec', 'b', '>', 0.0)
    )
    assert isinstance(result, Spec)
    assert result.name == 'b_spec'
    assert len(m._specs) == 2
    assert len(tb._specs) == 2


def test_testbench_and_chain_forward_explicit_keys(tmp_path):
    m = _module(tmp_path)
    tb = ModuleTestbench(m, name='unit')
    tb.add_metric('a', lambda ckt: 0.0)
    tb.add_metric('b', lambda ckt: 0.0)

    first = tb.add_spec('Readable A', 'a', '>', 0.0, key='a_key')
    second = first.add_spec('Readable B', 'b', '>', 0.0, key='b_key')

    assert first.key == 'a_key'
    assert second.key == 'b_key'
    assert [spec.key for spec in m._specs] == ['a_key', 'b_key']


def test_spec_add_spec_without_owning_testbench_raises(tmp_path):
    """A Spec created directly via Module.declare_spec has no owning
    testbench, so its add_spec forwarder must raise."""
    m = _module(tmp_path)
    spec = m.declare_spec(name='direct', metric='x', op='>', target=0.0)
    with pytest.raises(RuntimeError, match='not created via a ModuleTestbench'):
        spec.add_spec('child', 'y', '>', 0.0)


def test_testbench_run_populates_spec_measurements(tmp_path, monkeypatch):
    """After run() completes, the attached Spec must hold a Measurement
    tagged 'spice' with the extracted metric value.

    We bypass actual SPICE simulation by stubbing ``netlist.simulate`` and
    relying on the metric extractor to return a constant — this exercises
    the Measurement-recording hook in _simulate without needing a live
    simulator."""
    m = _module(tmp_path)
    tb = ModuleTestbench(m, name='unit')
    tb.add_metric('gain_dB', lambda ckt: 70.0)
    spec = tb.add_spec('DC gain > 60 dB', 'gain_dB', '>', 60.0)

    # Build the netlist (no DUT actually wired — we just need a Module
    # instance for simulate() to be called on).
    tb._create_netlist()
    tb._module_added = True

    # Stub the simulate() call so we don't hit ngspice. Also flips
    # sim_results_loaded (TASK-115: run() calls the public evaluate(),
    # whose "nothing loaded" guard would otherwise reject this
    # never-really-simulated netlist) so the extraction tail still runs.
    def _fake_simulate(corner='tt'):
        tb.netlist.sim_results_loaded = True
    monkeypatch.setattr(tb.netlist, 'simulate', _fake_simulate)

    report = tb.run(corner='tt')

    latest = spec.latest()
    assert latest is not None
    assert latest.source == 'spice'
    assert latest.value == 70.0
    assert latest.timestamp > 0
    assert latest.passes is True
    # The legacy SpecReport path is still populated.
    assert report.metrics['gain_dB'] == 70.0
    assert report.results[0].spec_key == spec.key
    assert report.results[0].owner_path == '$'


@pytest.mark.parametrize('key', [None, 'stable_gain'])
def test_testbench_duplicate_key_preserves_original_declaration(tmp_path, key):
    m = _module(tmp_path)
    tb = m.create_test('unit')
    tb.add_metric('gain', lambda ckt: 65.0)
    first = tb.add_spec('Gain', 'gain', '>=', 60.0, key=key)
    measurement = Measurement(source='spice', value=65.0, passes=True)
    first.record_measurement(measurement)
    original_entry = dict(tb._specs[0])

    with pytest.raises(DuplicateNameError, match=first.key):
        tb.add_spec('Gain', 'gain', '<', 99.0, key=key)

    assert len(m._specs) == 1 and m._specs[0] is first
    assert tb._specs == [original_entry]
    assert (first.target, first.op, first.metric, first.key) == (
        60.0, '>=', 'gain', key or 'Gain',
    )
    assert first.latest() is measurement
    assert first.evaluate(measurement) is True

    second = tb.add_spec('Gain', 'gain', '<', 99.0, key='other_gain')
    assert second is not first
    assert m._specs == [first, second]
    assert [entry['spec'] for entry in tb._specs] == [first, second]
    assert first.target == 60.0 and second.target == 99.0


@pytest.mark.parametrize('composition', ['module', 'testbench'])
def test_explicit_copy_preserves_protected_specs_and_runtime_relationships(
    tmp_path, composition,
):
    m = _module(tmp_path, 'block')
    def model_1(ports, t, params):
        return
    m.set_behavioral_model(model_1, params={'gain': 0.0}, execution='vector')
    tb = m.create_test('unit')
    tb.add_metric('gain', lambda ckt: 65.0)
    parent = tb.add_spec('Gain ceiling', 'gain', '<=', 100.0, key='ceiling')
    child = m.declare_spec('Gain floor', 'gain', '>=', 60.0, parent=parent)
    # These links are intentionally attached after construction.
    child.calibration = CalibrationBinding(m, 'gain')
    late_child = m.declare_spec('Late child', 'gain', '>', 0.0)
    late_child.parent = parent
    parent.children.append(late_child)
    for spec in (parent, child, late_child):
        spec.record_measurement(Measurement(source='spice', value=65.0))
    assert all(o.status == 'PASS' for o in m.reconcile_specs().outcomes)

    import copy
    independent = copy.deepcopy(m)
    if composition == 'module':
        top = _module(tmp_path, 'top')
        top.add_module(independent, 'block1', nets={})
        copied = top.elem_dict['block1']['device']
    else:
        independent.define_module_ports({})
        copied_tb = independent._tests['unit']
        copied_tb.add_module_under_test({})
        copied = copied_tb.netlist.elem_dict['dut']['device']

    assert copied is not m
    copied_parent, copied_child, copied_late = copied._specs
    for original, replica in zip((parent, child, late_child), copied._specs):
        assert replica is not original
        assert replica.module is copied
        assert (replica.target, replica.op, replica.metric, replica.key) == (
            original.target, original.op, original.metric, original.key,
        )
        assert replica.latest() == original.latest()
        assert replica.latest() is not original.latest()
        for field in ('target', 'op', 'metric', 'key'):
            with pytest.raises(AttributeError, match=field):
                setattr(replica, field, getattr(replica, field))
            with pytest.raises(AttributeError, match=field):
                delattr(replica, field)
    assert copied_child.parent is copied_parent
    assert copied_late.parent is copied_parent
    assert copied_parent.children == [copied_child, copied_late]
    assert copied_child.calibration.target_module is copied
    copied_tb = copied_parent._owning_testbench
    assert copied_tb is not tb and copied_tb.module is copied
    assert copied_tb._specs[0]['spec'] is copied_parent
    chained = copied_parent.add_spec('Extra', 'gain', '>', 0.0, key='extra')
    assert chained.module is copied and chained._owning_testbench is copied_tb
    assert chained in copied._specs and chained not in m._specs

    copied_child.record_measurement(Measurement(source='spice', value=55.0))
    outcomes = copied.reconcile_specs().outcomes
    assert [o.status for o in outcomes] == ['PASS', 'FAIL', 'PASS', 'UNMEASURED']
    assert copied_child.latest().passes is False
    assert copied._behavioral_params['gain'] == 55.0
    assert child.latest().passes is True
    assert m._behavioral_params['gain'] == 65.0
