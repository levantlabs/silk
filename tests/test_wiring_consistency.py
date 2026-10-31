"""Emission and lint reject disagreement between authored and indexed wiring."""
import pytest
from silk.circuit.modules import Module
from silk.elements import elements
from silk.exceptions import CircuitError
from silk.lint import RuleRegistry


@pytest.mark.parametrize('damage', ['reverse_missing', 'reverse_stale', 'pins_missing', 'pins_duplicate', 'pins_stale'])
@pytest.mark.parametrize('entry', ['lint', 'deck'])
def test_corruption_refused(pdk, tmp_path, damage, entry):
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    r = elements.resistor('r', p='x', n='0', value=1)
    m.add(r)
    if damage == 'reverse_missing':
        del m._element_mgr._pin_to_net['r', 'p']
    elif damage == 'reverse_stale':
        m._element_mgr._pin_to_net['r', 'p'] = '0'
    elif damage == 'pins_missing':
        m.net_dict['x'].pins.clear()
    elif damage == 'pins_duplicate':
        m.net_dict['x'].pins.append(m.net_dict['x'].pins[0])
    else:
        m.net_dict['x'].pins[0].net = '0'
    with pytest.raises(CircuitError, match='r.*p'):
        if entry == 'lint':
            RuleRegistry.default().run(m)
        else:
            m.generate_netlist()


def test_case_aliases_and_independent_placements_are_consistent(pdk, tmp_path):
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    m.add_net('VDD')
    m.add_passive(elements.resistor('r', p='vdd', n='0', value=1))
    RuleRegistry.default().run(m)
    m.generate_netlist()


def test_unbound_pin_without_either_binding_is_consistent(pdk, tmp_path):
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    r = elements.resistor('r', p='x', n='0', value=1)
    m.add(r)
    r.nets['p'] = None
    m.net_dict['x'].pins.clear()
    m._element_mgr._pin_to_net.pop(('r', 'p'))
    RuleRegistry.default().run(m)


def test_custom_emitter_is_excluded_without_hiding_supported_sibling(pdk, tmp_path):
    from silk.circuit.wiring_consistency import assert_consistent
    class Opaque(type(elements.resistor('unused', p='x', n='0', value=1))):
        def to_spice(self):
            raise AssertionError('must not render')
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    opaque = Opaque('opaque', p='x', n='0', value=1)
    m.add(opaque)
    opaque.nets['p'] = 'other'
    assert_consistent(m)
    r = elements.resistor('r', p='x', n='0', value=1)
    m.add(r)
    m._element_mgr._pin_to_net.pop(('r', 'p'))
    with pytest.raises(CircuitError, match='r.*p'):
        assert_consistent(m)


def test_one_scope_for_recursive_diamond_and_next_call_rechecks(pdk, tmp_path, monkeypatch):
    from silk.circuit import wiring_consistency as wc
    leaf = Module('PDK_skywater130', str(tmp_path / 'leaf'), 'leaf.cir', pdk_details=pdk, module_name='leaf')
    leaf.define_module_ports({})
    leaf.add(elements.resistor('r', p='x', n='0', value=1))
    mid = Module('PDK_skywater130', str(tmp_path / 'mid'), 'mid.cir', pdk_details=pdk, module_name='mid')
    mid.define_module_ports({})
    mid.add_module(leaf, 'inner')
    top = Module('PDK_skywater130', str(tmp_path / 'top'), 'top.cir', pdk_details=pdk)
    top.add_module(mid, 'left'); top.add_module(mid, 'right')
    calls = []
    real = wc._check_local
    def check(m):
        calls.append(m)
        real(m)
    monkeypatch.setattr(wc, '_check_local', check)
    top.generate_netlist()
    assert calls == [top, mid, leaf]
    calls.clear()
    leaf._element_mgr._pin_to_net.pop(('r', 'p'))
    with pytest.raises(CircuitError, match='r.*p'):
        top.generate_netlist()
    assert calls == [top, mid, leaf]


def test_placement_record_is_the_wiring_authority(pdk, tmp_path):
    from silk import PortDir
    from silk.circuit.wiring_consistency import assert_consistent
    child = Module('PDK_skywater130', str(tmp_path / 'child'), 'child.cir', pdk_details=pdk, module_name='child')
    child.define_module_ports({'p': PortDir.INOUT})
    child.add(elements.resistor('r', p='p', n='0', value=1))
    top = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    top.add_module(child, 'left', p='a')
    top.add_module(child, 'right', p='b')
    assert_consistent(top)
    top.elem_dict['right']['pin_list']['p'].net = 'a'
    with pytest.raises(CircuitError, match='right.*p'):
        assert_consistent(top)


@pytest.mark.parametrize('damage', ['reverse_case', 'binding_case', 'unbound_reverse_entry', 'net_name'])
def test_noncanonical_or_spurious_index_is_refused(pdk, tmp_path, damage):
    from silk.circuit.wiring_consistency import assert_consistent
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    m.add_net('VDD')
    r = elements.resistor('r', p='vdd', n='0', value=1)
    m.add(r)
    if damage == 'reverse_case':
        m._element_mgr._pin_to_net['r', 'p'] = 'vdd'
    elif damage == 'binding_case':
        m.net_dict['VDD'].pins[0].net = 'vdd'
    elif damage == 'net_name':
        m.net_dict['VDD'].name = 'other'
    else:
        r.nets['p'] = None
        m.net_dict['VDD'].pins.clear()
        m._element_mgr._pin_to_net['r', 'p'] = None
    with pytest.raises(CircuitError, match='r.*p'):
        assert_consistent(m)
