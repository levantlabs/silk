"""TASK-458 replaces TASK-458.1: live edits cannot diverge from a placement."""
import copy
import gc
import pickle
import weakref
from pathlib import Path

import numpy as np
import pytest

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.exceptions import CircuitError, SimulationError
from silk.simulation import Transient
from tests._env import needs_ngspice


def block(tmp_path, name='load', **kwargs):
    mod = Module('sky130', str(tmp_path), name + '.cir', load_pdk('sky130'),
                 module_name=name, **kwargs)
    mod.define_module_ports({'a': 'input', 'b': 'output'})
    mod.add(elements.resistor('r1', p='a', n='b', value=1000))
    return mod


def top(tmp_path):
    mod = Module('sky130', str(tmp_path), 'parent.cir', load_pdk('sky130'))
    mod.circuit_name = 'parent'
    return mod


def change(mod, kind):
    if kind == 'add':
        mod.add(elements.resistor('r2', p='a', n='b', value=2000))
    elif kind == 'remove':
        # Test-only removal of a successful placement, not add rollback.
        graph = mod._element_mgr
        device = graph.elem_dict['r1']['device']
        graph._retire_instance('elem_dict', 'r1', device)
        for port in device._pin_names:
            net_name = graph._pin_to_net.pop(('r1', port), None)
            if net_name is not None:
                net = graph.net_dict[net_name]
                net.pins[:] = [b for b in net.pins
                               if not (b.kind == 'device' and b.instance == 'r1' and b.port == port)]
        graph.elem_dict.pop('r1')
    elif kind == 'value':
        mod.elem_dict['r1']['device'].value = 2000
    elif kind == 'source':
        mod.add_source(sources.VoltageSource('bias', positive='a', negative='b', dc=1))
    elif kind == 'model':
        def model_3(ports, t, params):
            ports.b.v = ports.a.v
        mod.set_behavioral_model(model_3, execution='vector')


@pytest.mark.parametrize('method', ['add_module', 'add_element'])
@pytest.mark.parametrize('side', ['original', 'placed handle'])
@pytest.mark.parametrize('kind', ['add', 'remove', 'value', 'source', 'model'])
def test_content_edits_reach_emitted_definition(tmp_path, method, side, kind):
    child = block(tmp_path)
    parent = top(tmp_path)
    getattr(parent, method)(child, 'one', a='vin', b='0')
    target = child if side == 'original' else parent.elem_dict['one']['device']
    change(target, kind)
    assert target is child
    parent_deck_path = parent.generate_netlist()
    text = Path(parent_deck_path).read_text()
    if kind == 'add':
        assert 'Rr2 a b 2k' in text
    elif kind == 'remove':
        assert 'Rr1 ' not in text
    elif kind == 'value':
        assert 'Rr1 a b 2k' in text
    elif kind == 'source':
        assert 'Vbias a b' in text
    else:
        assert next(parent.iter_instances()).module._behavioral_model is child._behavioral_model



@pytest.mark.parametrize('side', ['original', 'placed handle'])
def test_nested_handles_retain_live_definition(tmp_path, side):
    leaf = block(tmp_path, 'leaf')
    middle = block(tmp_path, 'middle')
    middle.add_module(leaf, 'inner', a='a', b='b')
    parent = top(tmp_path)
    parent.add_module(middle, 'outer', a='vin', b='0')
    target = leaf if side == 'original' else (
        parent.elem_dict['outer']['device'].elem_dict['inner']['device'])
    change(target, 'value')
    assert target is leaf
    parent_deck_path = parent.generate_netlist()
    assert 'Rr1 a b 2k' in Path(parent_deck_path).read_text()


def test_diverging_distinct_definitions_still_clash(tmp_path):
    first, second = block(tmp_path), block(tmp_path)
    parent = top(tmp_path)
    parent.add_module(first, 'one', a='vin', b='0')
    parent.add_module(second, 'two', a='vin', b='0')
    change(second, 'value')
    with pytest.raises(CircuitError, match='xone.*xtwo.*structurally different'):
        parent.generate_netlist()


@pytest.mark.parametrize('side', ['original', 'placed handle'])
def test_calibration_reaches_behavioral_testbench_run(tmp_path, side):
    dut = block(tmp_path)
    def model_1(ports, t, p):
        ports.b.v = ports.a.v * p['gain']
    dut.set_behavioral_model(model_1,
                             params={'gain': 1.0, 'array': np.array([1., 2.])}, execution='vector')
    dut.simulator = 'behavioral'
    tb = dut.create_test('gain')
    tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
    tb.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    tb.add_sim_type(Transient(step=1e-9, stop=2e-9))
    target = dut if side == 'original' else tb.netlist.elem_dict['dut']['device']
    target.calibrate_behavioral('gain', 2.0)
    assert target is dut
    tb.run()
    assert tb.netlist.load_signal('vout').y[-1] == pytest.approx(2)


@pytest.mark.parametrize('side', ['original', 'placed handle'])
def test_injected_parasitics_reach_parent(tmp_path, side):
    from silk.layout import inject_parasitics
    from tests.layout.test_parasitics import _estimate_for
    child = block(tmp_path)
    parent = top(tmp_path)
    parent.add_module(child, 'one', a='vin', b='0')
    target = child if side == 'original' else parent.elem_dict['one']['device']
    inject_parasitics(target, _estimate_for('a'), mode='c_only', return_net='b')
    assert target is child
    parent_deck_path = parent.generate_netlist()
    assert any(line.startswith('C') and ' a b ' in line
               for line in Path(parent_deck_path).read_text().splitlines())


def test_parent_retains_live_definition_until_collected(tmp_path):
    child = block(tmp_path)
    parent = top(tmp_path)
    parent.add_module(child, 'one', a='vin', b='0')
    reference = weakref.ref(child)
    del child
    gc.collect()
    assert reference() is parent.elem_dict['one']['device']
    parent_deck_path = parent.generate_netlist()
    assert 'Rr1 a b 1k' in Path(parent_deck_path).read_text()
    change(parent.elem_dict['one']['device'], 'value')
    parent_deck_path = parent.generate_netlist()
    assert 'Rr1 a b 2k' in Path(parent_deck_path).read_text()
    del parent
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize('clone', [copy.deepcopy, lambda m: pickle.loads(pickle.dumps(m))])
def test_explicit_copies_own_live_definitions(tmp_path, clone):
    child = block(tmp_path, variants=True)
    parent = top(tmp_path)
    parent.add_module(child, 'one', a='vin', b='0')
    cloned = clone(parent)
    cloned.generate_netlist()
    change(cloned.elem_dict['one']['device'], 'value')
    cloned_deck_path = cloned.generate_netlist()
    assert 'Rr1 a b 2k' in Path(cloned_deck_path).read_text()
    assert child.elem_dict['r1']['device'].value == 1000


@needs_ngspice
@pytest.mark.parametrize('setting', ['original_simulator', 'instance_simulator',
                                    'save_device_params', 'other_settings'])
def test_run_settings_do_not_diverge(tmp_path, setting):
    child = block(tmp_path)
    parent = top(tmp_path)
    parent.add_module(child, 'one', a='vin', b='0')
    parent.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    parent.add_sim_type(Transient(step=1e-9, stop=2e-9))
    if setting == 'original_simulator':
        child.simulator = 'xyce'
    elif setting == 'instance_simulator':
        parent.elem_dict['one']['device'].simulator = 'xyce'
    elif setting == 'save_device_params':
        parent.save_device_params = True
    else:
        child.set_output_options(mode='new')
        child.set_save_list(['a'])
        child.save_device_params = True
        child.save_device_currents = True
        parent.set_save_list(['vin'])
    parent.generate_netlist()
    parent.simulate()
    assert parent.results_file is not None


def test_behavioral_run_settings_and_array_parameters_remain_valid(tmp_path):
    dut = block(tmp_path)
    def model_2(ports, t, p):
        ports.b.v = ports.a.v * p['gain'][0]
    dut.set_behavioral_model(model_2, params={'gain': np.array([2.0, np.nan])}, execution='vector')
    tb = dut.create_test('settings')
    tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
    tb.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    tb.add_sim_type(Transient(step=1e-9, stop=2e-9))
    # Each switch is after placement; the run uses the explicitly selected
    # bench backend while canonical body comparisons ignore all selections.
    dut.simulator = 'xyce'
    tb.netlist.elem_dict['dut']['device'].simulator = 'ngspice'
    tb.netlist.simulator = 'behavioral'
    for _ in range(2):
        tb.run()
        assert tb.netlist.load_signal('vout').y[-1] == pytest.approx(2)
    dut._behavioral_params['gain'][0] = 3
    tb.run()
    assert tb.netlist.load_signal('vout').y[-1] == pytest.approx(3)


@pytest.mark.parametrize('side', ['original', 'placed handle'])
def test_renamed_module_reaches_parent(tmp_path, side):
    child = block(tmp_path)
    parent = top(tmp_path)
    parent.add_module(child, 'one', a='vin', b='0')
    target = child if side == 'original' else parent.elem_dict['one']['device']
    target.define_module_name('renamed')
    parent_deck_path = parent.generate_netlist()
    assert '.subckt renamed ' in Path(parent_deck_path).read_text()
    assert 'xone vin 0 renamed' in Path(parent_deck_path).read_text()


def test_original_mutation_after_preparation_refuses_launch(tmp_path):
    child = block(tmp_path)
    parent = top(tmp_path)
    parent.add_module(child, 'one', a='vin', b='0')
    parent.add_sim_type(Transient(step=1e-9, stop=2e-9))
    parent.generate_netlist()
    change(child, 'value')
    with pytest.raises(SimulationError, match='changed'):
        parent.run_simulator()


class _SlottedGain:
    __slots__ = ('gain',)

    def __init__(self, gain):
        self.gain = gain

    def __call__(self, ports, time, params):
        ports.b.v = ports.a.v * self.gain


def test_callable_model_state_remains_live(tmp_path):
    child = block(tmp_path)
    child.set_behavioral_model(_SlottedGain(2), execution='vector')
    parent = top(tmp_path)
    parent.add_module(child, 'one', a='vin', b='out')
    parent.generate_netlist()
    child._behavioral_model.gain = 3
    parent.generate_netlist()
    parent.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    parent.simulate_behavioral(1e-9, 2e-9)
    assert parent.load_signal('out').y[-1] == pytest.approx(3)


class _ShadowedGain(_SlottedGain):
    __slots__ = ('gain',)

    def __init__(self):
        _SlottedGain.gain.__set__(self, 2)
        self.gain = 99

    def __call__(self, ports, time, params):
        ports.b.v = ports.a.v * _SlottedGain.gain.__get__(self)

    def __deepcopy__(self, memo):
        cloned = type(self)()
        memo[id(self)] = cloned
        _SlottedGain.gain.__set__(cloned, _SlottedGain.gain.__get__(self))
        cloned.gain = self.gain
        return cloned


def test_shadowed_inherited_model_slot_remains_live(tmp_path):
    child = block(tmp_path)
    child.set_behavioral_model(_ShadowedGain(), execution='vector')
    parent = top(tmp_path)
    parent.add_module(child, 'one', a='vin', b='out')
    parent.generate_netlist()
    _SlottedGain.gain.__set__(child._behavioral_model, 3)
    parent.generate_netlist()
    parent.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    parent.simulate_behavioral(1e-9, 2e-9)
    assert parent.load_signal('out').y[-1] == pytest.approx(3)


@pytest.mark.parametrize('side', ['original', 'placed handle'])
@pytest.mark.parametrize('kind', ['add', 'value'])
def test_uninstantiated_definition_edits_reach_parent(tmp_path, side, kind):
    child = block(tmp_path)
    parent = top(tmp_path)
    extra = block(tmp_path, 'unused')
    if kind == 'value':
        child.add_submodule('unused', extra)
    parent.add_module(child, 'one', a='vin', b='0')
    target = child if side == 'original' else parent.elem_dict['one']['device']
    if kind == 'add':
        target.add_submodule('unused', extra)
    else:
        change(target.module_dict['unused']['device'], 'value')
    assert '.subckt unused ' in target.generate_netlist()
    assert target is child
    parent_deck_path = parent.generate_netlist()
    text = Path(parent_deck_path).read_text()
    assert '.subckt unused ' in text
    if kind == 'value':
        assert 'Rr1 a b 2k' in text


def _factory_model_block(tmp_path, scale):
    child = block(tmp_path)

    def model(ports, t, params):
        ports.b.v = ports.a.v * scale * params['gain']

    child.set_behavioral_model(model, params={'gain': 1.0}, execution='vector')
    return child


def _factory_pair(tmp_path):
    first = _factory_model_block(tmp_path, 1)
    second = _factory_model_block(tmp_path, 2)
    parent = top(tmp_path)
    parent.add_module(first, 'one', a='vin', b='out1')
    parent.add_module(second, 'two', a='vin', b='out2')
    return parent, first, second


def test_factory_models_share_definition_but_run_own_closures(tmp_path):
    parent, first, second = _factory_pair(tmp_path)
    assert first._behavioral_model is not second._behavioral_model
    assert len(parent.module_dict) == 1
    parent.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    parent.generate_netlist()
    parent.simulate_behavioral(1e-9, 2e-9)
    assert parent.load_signal('out1').y[-1] == pytest.approx(1)
    assert parent.load_signal('out2').y[-1] == pytest.approx(2)


def test_calibrating_distinct_same_named_model_clashes(tmp_path):
    parent, first, second = _factory_pair(tmp_path)
    second.calibrate_behavioral('gain', 3)
    with pytest.raises(CircuitError, match='xone.*xtwo.*behavioral model differs'):
        parent.generate_netlist()


def test_swapping_placed_model_with_same_code_remains_live(tmp_path):
    parent, first, second = _factory_pair(tmp_path)
    replacement = _factory_model_block(tmp_path, 99)
    parent.elem_dict['two']['device'].set_behavioral_model(
        replacement._behavioral_model, params={'gain': 1.0}, execution='vector')
    parent.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    parent.simulate_behavioral(1e-9, 2e-9)
    assert parent.load_signal('out2').y[-1] == pytest.approx(99)


@pytest.mark.parametrize('nested', [False, True])
def test_swapping_model_with_same_code_reaches_nested_run(tmp_path, nested):
    parent, first, second = _factory_pair(tmp_path)
    if nested:
        parent.define_module_name('pair')
        parent.define_module_ports({'vin': 'input', 'out1': 'output', 'out2': 'output'})
        outer = top(tmp_path)
        outer.add_module(parent, 'pair', vin='vin', out1='out1', out2='out2')
    else:
        outer = parent
    replacement = _factory_model_block(tmp_path, 99)
    second.set_behavioral_model(replacement._behavioral_model, params={'gain': 1.0}, execution='vector')
    outer.add_source(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    outer.simulate_behavioral(1e-9, 2e-9)
    assert outer.load_signal('out2').y[-1] == pytest.approx(99)


@pytest.mark.parametrize('side', ['original', 'placed handle'])
def test_distinct_callable_factories_keep_live_state(tmp_path, side):
    first, second = block(tmp_path), block(tmp_path)
    first.set_behavioral_model(_SlottedGain(2), execution='vector')
    second.set_behavioral_model(_SlottedGain(3), execution='vector')
    parent = top(tmp_path)
    parent.add_module(first, 'one', a='vin', b='0')
    parent.add_module(second, 'two', a='vin', b='0')
    parent.generate_netlist()
    target = second if side == 'original' else parent.elem_dict['two']['device']
    target._behavioral_model.gain = 2
    parent.generate_netlist()
    assert target is second
    assert next(item for item in parent.iter_instances() if item.path == 'two').module._behavioral_model.gain == 2
