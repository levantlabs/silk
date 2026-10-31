"""TASK-350: tests are circuits — tb.add() forwards any element to the
testbench's top-level Module, and testbench codegen emits rig/test devices
through the circuit generator's emitters."""
import ast
from pathlib import Path
import pytest

from silk.codegen.testbench_generator import TestbenchSpecError, generate_testbenches

R = {'type': 'resistor', 'instance': 'rload',
     'nets': {'p': 'y', 'n': '0'}, 'params': {'value': 1000.0}}
M = {'type': 'nmos1p8', 'instance': 'Mload',
     'nets': {'d': 'y', 'g': 'a', 's': '0', 'b': '0'},
     'params': {'w': 1.0, 'l': 0.15}}
V = {'type': 'VoltageSource', 'instance': 'vdd', 'positive': 'vdd',
     'negative': '0', 'dc': 1.8}


def _spec(test=None, rig=None, applied=()):
    t = {'name': 't', 'dut_portmap': {'a': 'a'}, 'applied_rigs': list(applied)}
    t.update(test or {})
    return {'circuit_name': 'c', 'tests': [t],
            'rigs': [dict({'name': 'r'}, **rig)] if rig is not None else []}


def test_no_element_keys_is_unchanged():
    out = generate_testbenches(_spec({'sources': [V]}))
    assert 'pdk' not in out and 'elements' not in out


def test_emission_shape_inline_no_variable():
    out = generate_testbenches(_spec({'devices': [M], 'ideal_elements': [R],
                                      'sources': [V]}))
    assert "from silk.elements import elements" in out
    assert ("    pdk = tb.pdk_details\n"
            "    tb.add(pdk.nmos1p8('Mload', d='y', g='a', s='0', b='0', "
            "w=1.0, l=0.15))\n"
            "    tb.add(elements.resistor('rload', p='y', n='0', value=1000.0))"
            ) in out
    assert 'Mload =' not in out and 'rload =' not in out
    ast.parse(out)


def test_rig_emission_and_arrays():
    arr = dict(M, instance='MA', array=2)
    out = generate_testbenches(_spec(rig={'devices': [arr]}, applied=['r']))
    assert ("    tb.add(*[pdk.nmos1p8(f'MA[{i}]', d='y', g='a', s='0', b='0', "
            "w=1.0, l=0.15) for i in range(2)])") in out
    ast.parse(out)


@pytest.mark.parametrize('name', ['tb', 'pdk', 'mod', 'elements', 'sources',
                                  'analysis', 'i', 'range'])
def test_instance_named_like_generator_locals_is_harmless(name):
    """Instance names are never Python variables, so nothing is reserved."""
    out = generate_testbenches(_spec({'devices': [dict(M, instance=name)],
                                      'ideal_elements': [dict(R, instance=name + '_r')]}))
    assert f"tb.add(pdk.nmos1p8({name!r}," in out
    ast.parse(out)


@pytest.mark.parametrize('other', [
    {'devices': [dict(M, instance='RLOAD')]},
    {'sources': [dict(V, instance='Rload')]},
    {'submodules': [{'instance_name': 'rLoad',
                     'module_ref': {'import_path': 'pkg.m', 'module_var': 'm'},
                     'nets': {}}]},
    {'ideal_elements': [R]},
])
def test_duplicate_names_refused_case_insensitive(other):
    test = {'ideal_elements': [R]}
    for k, v in other.items():
        test[k] = test.get(k, []) + v if k == 'ideal_elements' else v
    with pytest.raises(TestbenchSpecError, match='collides'):
        generate_testbenches(_spec(test))


def test_duplicate_across_applied_rig_and_test_and_dut():
    with pytest.raises(TestbenchSpecError, match='collides'):
        generate_testbenches(_spec({'ideal_elements': [R]},
                                   rig={'ideal_elements': [dict(R, instance='RLOAD')]},
                                   applied=['r']))
    with pytest.raises(TestbenchSpecError, match='DUT'):
        generate_testbenches(_spec({'ideal_elements': [dict(R, instance='dut')]}))


def test_arrayed_expanded_names_collide():
    arr = dict(R, instance='rr', array=2, naming='underscore')
    with pytest.raises(TestbenchSpecError, match='collides'):
        generate_testbenches(_spec({'ideal_elements': [arr, dict(R, instance='RR_1')]}))
    generate_testbenches(_spec({'ideal_elements': [arr, dict(R, instance='rx')]}))


# --- Part 1: tb.add forwards any element to the top-level Module ------------

def _dut(tmp_path):
    from silk import load_pdk
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', str(tmp_path), 'd.cir', pdk_details=pdk,
                 module_name='d')
    mod.define_module_ports({'a': PortDir.IN, 'y': PortDir.OUT,
                             'vss': PortDir.GROUND})
    with mod:
        pdk.nmos1p8('mn', d='y', g='a', s='vss', b='vss', w=1.0, l=0.15)
    return pdk, mod


def test_tb_add_pdk_device_matches_module_add(tmp_path):
    from tests._env import HAVE_PDK
    if not HAVE_PDK:
        pytest.skip('needs an installed sky130 PDK')
    from silk import sources
    from silk.elements import elements
    pdk, mod = _dut(tmp_path)
    tb = mod.create_test('t')
    tb.add_module_under_test({'a': 'a', 'y': 'y', 'vss': '0'})
    assert tb.add(pdk.nmos1p8('Mload', d='y', g='a', s='0', b='0', w=1.0, l=0.15),
                  elements.resistor('rload', p='y', n='0', value=1e3),
                  sources.vdc('vdd', positive='a', negative='0', dc=1.0)) is tb
    ref = pdk.nmos1p8('Mref', d='y', g='a', s='0', b='0', w=1.0, l=0.15)
    from silk.circuit.modules import Module
    plain = Module('PDK_skywater130', str(tmp_path), 'p.cir', pdk_details=pdk)
    plain.add(ref)
    tb_dev = tb.netlist._element_mgr.elem_dict['Mload']['device']
    assert type(tb_dev) is type(ref)
    assert tb_dev.name == 'Mload'


def test_tb_add_still_refuses_non_elements(tmp_path):
    from tests._env import HAVE_PDK
    if not HAVE_PDK:
        pytest.skip('needs an installed sky130 PDK')
    _, mod = _dut(tmp_path)
    tb = mod.create_test('t')
    with pytest.raises(TypeError, match='Module.add'):
        tb.add(object())


@pytest.mark.slow
def test_slow_generated_test_with_r_and_pdk_mosfet_simulates_on_ngspice(tmp_path):
    from tests._env import HAVE_NGSPICE, HAVE_PDK
    if not (HAVE_PDK and HAVE_NGSPICE):
        pytest.skip('needs an installed sky130 PDK and the ngspice binary')
    from silk import load_pdk
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module

    spec = {
        'circuit_name': 'inv', 'circuit_builder': '_b',
        'tests': [{
            'name': 'tran',
            'dut_portmap': {'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd', 'vss': '0'},
            'sources': [
                {'type': 'VoltageSource', 'instance': 'vdd', 'positive': 'vdd',
                 'negative': '0', 'dc': 1.8},
                {'type': 'VoltageSource', 'instance': 'vin', 'positive': 'vin',
                 'negative': '0',
                 'waveform': {'type': 'Pulse', 'v1': 0.0, 'v2': 1.8, 'td': 1e-9,
                              'tr': 100e-12, 'tf': 100e-12, 'pw': 4e-9,
                              'per': 10e-9}}],
            # Extra load: a gate-grounded (off) NMOS drain and a 1 Mohm R.
            'devices': [{'type': 'nmos1p8', 'instance': 'Mload',
                         'nets': {'d': 'vout', 'g': '0', 's': '0', 'b': '0'},
                         'params': {'w': 1.0, 'l': 0.15}}],
            'ideal_elements': [{'type': 'resistor', 'instance': 'rload',
                                'nets': {'p': 'vout', 'n': '0'},
                                'params': {'value': 1e6}}],
            'analyses': [{'kind': 'transient', 'step': 10e-12, 'stop': 40e-9}],
            'metrics': [{'name': 'tpd', 'type': 'propagation_delay',
                         'signals': {'sig_in': 'vin', 'sig_out': 'vout'},
                         'params': {'in_edge': 'any'}}],
        }]}
    code = generate_testbenches(spec)

    def _b(pdk):
        mod = Module('PDK_skywater130', str(tmp_path), 'inv.cir',
                     pdk_details=pdk, module_name='inv')
        mod.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT,
                                 'vdd': PortDir.POWER, 'vss': PortDir.GROUND})
        with mod:
            pdk.nmos1p8('mn', d='vout', g='vin', s='vss', b='vss', w=1.0, l=0.15)
            pdk.pmos1p8('mp', d='vout', g='vin', s='vdd', b='vdd', w=2.0, l=0.15)
        return mod

    ns = {'_b': _b}
    exec(code, ns)
    mod = ns['build_inv_with_tests'](load_pdk('sky130'))
    report = mod._tests['tran'].run('tt')
    assert 0 < report.metrics['tpd'] < 200e-12
    elems = mod._tests['tran'].netlist._element_mgr.elem_dict
    assert 'Mload' in elems and 'rload' in elems


def test_arrayed_independent_source_emits_all_n_inline():
    out = generate_testbenches(_spec({'sources': [
        dict(V, instance='vb', array=3, positive='p$i', negative='0')]}))
    assert ("    tb.add(*[sources.vdc(f'vb[{i}]', positive=f'p{i}', "
            "negative='0', value=1.8) for i in range(3)])") in out
    ast.parse(out)
    out = generate_testbenches(_spec({'sources': [
        dict(V, instance='vb', array=[1, 3], naming='underscore')]}))
    assert "f'vb_{i}'" in out and "range(1, 3)" in out


def test_arrayed_source_collides_on_expanded_names():
    with pytest.raises(TestbenchSpecError, match='collides'):
        generate_testbenches(_spec({
            'sources': [dict(V, instance='vb', array=2, naming='underscore')],
            'ideal_elements': [dict(R, instance='VB_1')]}))


def test_arrayed_controlled_source_with_wrong_pin_names_refused():
    e = {'type': 'VCVS', 'instance': 'e1', 'positive': 'a', 'negative': '0',
         'ctrl_positive': 'b', 'ctrl_negative': '0', 'gain': 2.0, 'array': 2}
    with pytest.raises(TestbenchSpecError):
        generate_testbenches(_spec({'sources': [e]}))


@pytest.mark.parametrize('reference', ['vdd', 'vdd[0]'])
def test_scalar_rig_control_can_still_reference_test_source(reference):
    """Array validation must not narrow the pre-existing scalar rig path."""
    consumer = dict(type='CCCS', instance='readout', out_pos='o', out_neg='0',
                    ctrl_vname=reference, gain=2)
    controller = dict(V, array=2) if reference == 'vdd[0]' else V
    code = generate_testbenches(_spec({'sources': [controller]},
                                      rig={'sources': [consumer]}, applied=['r']))
    assert f"ctrl_vname={reference!r}" in code


@pytest.mark.parametrize('mutate_expected', [False, True])
@pytest.mark.parametrize('location', ['test', 'rig'])
@pytest.mark.parametrize('kind', ['devices', 'ideal_elements', 'submodules'])
@pytest.mark.parametrize('array,indices,index,range_text', [
    (3, [0, 1, 2], 'i', 'range(3)'),
    ([4, 7], [4, 5, 6], 'i - (4)', 'range(4, 7)'),
])
def test_per_copy_entries_execute_like_handwritten_loop(tmp_path, monkeypatch, location, kind,
                                                       array, indices, index, range_text,
                                                       mutate_expected):
    import sys
    import types
    from silk import Module, load_pdk, sources
    from silk.elements import elements

    pdk = load_pdk('sky130')
    child = Module('leaf', str(tmp_path), 'leaf.cir', pdk_details=None, module_name='leaf')
    from silk import Port, PortDir
    child.define_module_ports({'d': Port(dir=PortDir.IN, bits=4), 'n': 'output'})
    child.add(elements.resistor('r', p='d[2]', n='n', value=1000))
    imported = types.ModuleType('_per_copy_helper')
    imported.mod = child
    monkeypatch.setitem(sys.modules, '_per_copy_helper', imported)

    def build(pdk):
        mod = Module('parent', str(tmp_path), 'parent.cir', pdk_details=pdk, module_name='parent')
        mod.define_module_ports({'a': 'input'})
        return mod

    nets = [{'p': net, 'n': '0'} for net in ['n3', 'n0', 'clk']]
    entry = dict(type='resistor', instance='amp1', array=array, nets=nets, params={'value': 1000})
    call = f"tb.add(elements.resistor(f'amp1[{{i}}]', value=1000, **__silk_amp1_nets[{index}]))"
    if kind == 'devices':
        entry.update(type='nmos1p8', params={'w': 1, 'l': 0.15},
                     nets=[{'d': n['p'], 'g': 'a', 's': '0', 'b': '0'} for n in nets])
        call = f"tb.add(pdk.nmos1p8(f'amp1[{{i}}]', w=1, l=0.15, **__silk_amp1_nets[{index}]))"
    elif kind == 'submodules':
        nets = [{'d[0]': 'a', 'd[1]': 'b', 'd[2]': net, 'd[3]': 'c', 'n': '0'}
                for net in ['p', 'q', 'r']]
        entry = dict(instance_name='amp1', array=array, nets=nets,
                     module_ref={'import_path': '_per_copy_helper', 'module_var': 'mod'})
        call = f"tb.add_module(mod__per_copy_helper, f'amp1[{{i}}]', nets=__silk_amp1_nets[{index}])"
    group = {kind: [entry]}
    spec = _spec(test=group) if location == 'test' else _spec(rig=group, applied=['r'])
    spec['circuit_builder'] = '_build'
    code = generate_testbenches(spec)
    assert f"__silk_amp1_nets = {entry['nets']!r}" in code
    assert f"for i in {range_text}: {call}" in code
    namespace = {'_build': build}
    exec(code, namespace)
    actual = namespace['build_c_with_tests'](pdk)._tests['t']
    expected = build(pdk).create_test('t')
    expected.add_module_under_test({'a': 'a'})
    for i, connections in zip(indices, entry['nets']):
        if mutate_expected and i == indices[0]:
            connections = dict(connections)
            connections['d[2]' if kind == 'submodules' else 'd' if kind == 'devices' else 'p'] = 'different_expected'
        if kind == 'submodules':
            expected.add_module(child, f'amp1[{i}]', nets=connections)
        else:
            factory = pdk.nmos1p8 if kind == 'devices' else elements.resistor
            expected.add(factory(f'amp1[{i}]', **connections, **entry['params']))
    actual.netlist.set_output_options(mode='overwrite')
    expected.netlist.set_output_options(mode='overwrite')
    actual_netlist_deck_path = actual.netlist.generate_netlist()
    actual_deck = Path(actual_netlist_deck_path).read_text()
    expected_netlist_deck_path = expected.netlist.generate_netlist()
    expected_deck = Path(expected_netlist_deck_path).read_text()
    assert actual_deck and expected_deck
    if mutate_expected:
        assert 'different_expected' in expected_deck
        with pytest.raises(AssertionError):
            assert actual_deck == expected_deck
    else:
        assert actual_deck == expected_deck


@pytest.mark.parametrize('location', ['test', 'rig'])
@pytest.mark.parametrize('kind', ['devices', 'ideal_elements', 'submodules'])
@pytest.mark.parametrize('nets', [
    [{'p': 'n3'}], [{'p': 'n3'}, {'n': 'clk'}],
    [{'p': 'n3'}, 'clk'], [{'p': 'n3'}, {'p': None}],
    [{'p': 'n3'}, {'p': 'n[$i]'}], [{'p': 'n3'}, {'p': 'bad net'}],
    [{'d[]': 'a'}, {'d[]': 'b'}],
    [{'d[-1]': 'a'}, {'d[-1]': 'b'}],
    [{'d[1:0]': 'a'}, {'d[1:0]': 'b'}],
    [{'d [2]': 'a'}, {'d [2]': 'b'}],
    [{'p': 'd[1:0]'}, {'p': 'b'}],
])
def test_per_copy_invalid_entries_keep_group_context(location, kind, nets):
    entry = dict(type='resistor', instance='amp1', array=2, nets=nets, params={'value': 1000})
    if kind == 'devices':
        entry['type'] = 'nmos1p8'
    elif kind == 'submodules':
        entry = dict(instance_name='amp1', array=2, nets=nets,
                     module_ref={'import_path': 'helper', 'module_var': 'mod'})
    group = {kind: [entry]}
    spec = _spec(test=group) if location == 'test' else _spec(rig=group, applied=['r'])
    with pytest.raises(TestbenchSpecError) as exc:
        generate_testbenches(spec)
    assert ("test 't'" if location == 'test' else "rig 'r'") in str(exc.value)
    assert 'amp1' in str(exc.value)


@pytest.mark.parametrize('location', ['test', 'rig'])
@pytest.mark.parametrize('pin', ['__silk_x[2]', '__SILK_x[2]', 2, 'net_like'])
def test_per_copy_submodule_pin_token_refusals_keep_context(location, pin):
    from unittest.mock import Mock

    if pin == 'net_like':
        pin = Mock()
        pin.name = 'd[2]'
    entry = dict(instance_name='amp1', array=2,
                 nets=[{pin: 'a'}, {pin: 'b'}],
                 module_ref={'import_path': 'helper', 'module_var': 'mod'})
    group = {'submodules': [entry]}
    spec = _spec(test=group) if location == 'test' else _spec(rig=group, applied=['r'])
    with pytest.raises(TestbenchSpecError) as exc:
        generate_testbenches(spec)
    assert ("test 't'" if location == 'test' else "rig 'r'") in str(exc.value)
    assert 'amp1' in str(exc.value)


@pytest.mark.parametrize('location', ['test', 'rig'])
@pytest.mark.parametrize('kind', ['devices', 'ideal_elements'])
def test_per_copy_device_bit_pin_keeps_group_context(location, kind):
    entry = dict(type='resistor', instance='amp1', array=2,
                 nets=[{'p[2]': 'a'}, {'p[2]': 'b'}])
    group = {kind: [entry]}
    spec = _spec(test=group) if location == 'test' else _spec(rig=group, applied=['r'])
    with pytest.raises(TestbenchSpecError) as exc:
        generate_testbenches(spec)
    assert ("test 't'" if location == 'test' else "rig 'r'") in str(exc.value)
    assert 'amp1' in str(exc.value)


@pytest.mark.parametrize('location', ['test', 'rig'])
@pytest.mark.parametrize('kind', ['devices', 'ideal_elements', 'submodules'])
def test_per_copy_net_inventory_refuses_derived_name_collision(location, kind):
    nets = [{'p': 'n3'}, {'p': 'collision'}]
    entry = dict(type='resistor', instance='amp1', array=2, nets=nets, params={'value': 1000})
    if kind == 'devices':
        entry['type'] = 'nmos1p8'
    elif kind == 'submodules':
        entry = dict(instance_name='amp1', array=2, nets=nets,
                     module_ref={'import_path': 'helper', 'module_var': 'mod'})
    group = {kind: [entry]}
    spec = _spec(test=group) if location == 'test' else _spec(rig=group, applied=['r'])
    spec['tests'][0]['analyses'] = [{'kind': 'transient', 'step': 1e-9, 'stop': 1e-6}]
    spec['tests'][0]['derived_signals'] = [dict(name='COLLISION', operation='abs', inputs=['a'])]
    with pytest.raises(TestbenchSpecError, match='clashes'):
        generate_testbenches(spec)


@pytest.mark.parametrize('location', ['test', 'rig'])
@pytest.mark.parametrize('collection', ['devices', 'ideal_elements'])
@pytest.mark.parametrize('container', [None, 42])
def test_invalid_element_containers_use_named_public_error(location, collection, container):
    from silk.codegen.generator import Generator
    group = {collection: container}
    spec = _spec(test=group) if location == 'test' else _spec(rig=group, applied=['r'])
    name = 't' if location == 'test' else 'r'
    with pytest.raises(TestbenchSpecError, match=f"{location} '{name}': {collection} must be a list"):
        Generator().generate_testbenches(spec)


@pytest.mark.parametrize('location', ['test', 'rig'])
@pytest.mark.parametrize('kind', ['vsource', 'isource'])
def test_ideal_source_rejection_keeps_group_context(location, kind):
    group = {'ideal_elements': [dict(type=kind, instance='bias',
                                    nets={'p': 'a', 'n': '0'})]}
    spec = _spec(test=group) if location == 'test' else _spec(rig=group, applied=['r'])
    with pytest.raises(TestbenchSpecError) as exc:
        generate_testbenches(spec)
    assert ("tests 't'" if location == 'test' else "rigs 'r'") in str(exc.value)
    assert "ideal_elements entry 'bias'" in str(exc.value)
    assert 'sources[]' in str(exc.value)
