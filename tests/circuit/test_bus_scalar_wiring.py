"""Direction-aware scalar bus wiring and actionable width diagnostics."""

import pytest

from silk.circuit.modules import Module
from silk.circuit.ports import Port, PortDir
from silk.exceptions import CircuitError


def modules(tmp_path, direction, bits=4, naming='bracketed', top=False):
    child = Module('sky130', str(tmp_path), 'child.cir', None,
                   module_name='child')
    child.define_module_ports({'d': Port(dir=direction, bits=bits, naming=naming)})
    parent = Module('sky130', str(tmp_path), 'parent.cir', None,
                    module_name=None if top else 'parent')
    return parent, child


@pytest.mark.parametrize('as_string', [False, True])
@pytest.mark.parametrize('bits,naming', [(4, 'bracketed'), (range(4, 8), 'underscore')])
def test_input_scalar_broadcast_emits_repeated_net_without_new_nets(
        tmp_path, as_string, bits, naming):
    parent, child = modules(tmp_path, PortDir.IN, bits, naming)
    scalar = parent.add_net('s')
    before = set(parent.net_dict)
    parent.add_module(child, 'u1', d='s' if as_string else scalar)
    assert set(parent.net_dict) == before
    assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1] == 'xu1 s s s s child\n'


@pytest.mark.parametrize('as_string', [False, True])
@pytest.mark.parametrize('bits,naming', [(4, 'bracketed'), (range(4, 8), 'underscore')])
def test_inout_scalar_broadcast_matches_input(tmp_path, as_string, bits, naming):
    results = []
    for direction in (PortDir.IN, PortDir.INOUT):
        parent, child = modules(tmp_path, direction, bits, naming)
        scalar = parent.add_net('s')
        before = set(parent.net_dict)
        parent.add_module(child, 'u1', d='s' if as_string else scalar)
        assert set(parent.net_dict) == before
        results.append(parent.netlister.generate_device_instances(parent.elem_dict))
    assert results[0] == results[1]
    assert results[1].splitlines(keepends=True)[-1] == 'xu1 s s s s child\n'


def test_inout_broadcast_through_one_bit_parent_bus_and_nested_placement(tmp_path):
    # A one-bit bus on the middle level, whose only bit fans out to the
    # four-bit inout bus of a child placed inside it.
    top = Module('sky130', str(tmp_path), 'top.cir', None)
    mid = Module('sky130', str(tmp_path), 'mid.cir', None, module_name='mid')
    child = Module('sky130', str(tmp_path), 'leaf.cir', None, module_name='leaf')
    child.define_module_ports({'d': Port(dir=PortDir.INOUT, bits=4)})
    mid.define_module_ports({'s': Port(dir=PortDir.INOUT, bits=1)})
    mid.add_module(child, 'u1', d=mid.nets.s)
    mid_text = mid.netlister.generate_device_instances(mid.elem_dict)
    assert mid_text.splitlines(keepends=True)[-1] == 'xu1 s[0] s[0] s[0] s[0] leaf\n'
    top.add_module(mid, 'xm', s=top.add_net('vss'))
    assert top.netlister.generate_device_instances(top.elem_dict).splitlines(keepends=True)[-1] == 'xxm vss mid\n'


@pytest.mark.parametrize('as_string', [False, True])
def test_inout_broadcast_through_bus_slice(tmp_path, as_string):
    parent, child = modules(tmp_path, PortDir.INOUT, bits=4)
    handle = parent.add_net('b', bits=4)[2:3]
    parent.add_module(child, 'u1', d='b[2]' if as_string else handle)
    assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1] == (
        'xu1 b[2] b[2] b[2] b[2] child\n')


@pytest.mark.parametrize('direction', [PortDir.OUT])
@pytest.mark.parametrize('as_string', [False, True])
@pytest.mark.parametrize('top', [False, True])
def test_scalar_output_refusal_names_connection(tmp_path, direction, as_string, top):
    parent, child = modules(tmp_path, direction, top=top)
    scalar = parent.add_net('s')
    with pytest.raises(CircuitError) as caught:
        parent.add_module(child, 'u1', d='s' if as_string else scalar)
    message = str(caught.value)
    for detail in ['BUS_WIDTH_MISMATCH', "module 'parent.cir'" if top else "module 'parent'",
                   "element 'u1'", "port 'd'", direction.value,
                   'width 4', "net 's'", 'width 1']:
        assert detail in message
    assert 'non-Net' not in message
    assert '(str)' not in message


@pytest.mark.parametrize('direction', [PortDir.IN, PortDir.OUT, PortDir.INOUT])
def test_wider_mismatch_names_connection(tmp_path, direction):
    parent, child = modules(tmp_path, direction)
    bus = parent.add_net('wide', bits=8)
    with pytest.raises(CircuitError) as caught:
        parent.add_module(child, 'u1', d=bus)
    for detail in ['BUS_WIDTH_MISMATCH', "module 'parent'", "element 'u1'",
                   "port 'd'", direction.value, 'width 4', "net 'wide'", 'width 8']:
        assert detail in str(caught.value)


@pytest.mark.parametrize('direction', [PortDir.IN, PortDir.OUT, PortDir.INOUT])
def test_matching_bus_emission_unchanged(tmp_path, direction):
    parent, child = modules(tmp_path, direction)
    bus = parent.add_net('b', bits=4)
    parent.add_module(child, 'u1', d=bus)
    assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1] == 'xu1 b[0] b[1] b[2] b[3] child\n'


@pytest.mark.parametrize('as_string', [False, True])
@pytest.mark.parametrize('direction', [PortDir.IN, PortDir.OUT, PortDir.INOUT])
@pytest.mark.parametrize('bits,naming,bit', [
    (1, 'bracketed', 'd[0]'), (range(3, 4), 'bracketed', 'd[3]'),
    (range(3, 4), 'underscore', 'd_3')])
def test_width_one_bus_accepts_scalar_by_bus_name(
        tmp_path, as_string, direction, bits, naming, bit):
    parent, child = modules(tmp_path, direction, bits=bits, naming=naming)
    scalar = parent.add_net('s')
    before = set(parent.net_dict)
    parent.add_module(child, 'u1', d='s' if as_string else scalar)
    assert set(parent.net_dict) == before
    last = parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1]
    assert last == 'xu1 s child\n'
    parent.add_module(child, 'u2', **{bit: 's' if as_string else scalar})
    last = parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1]
    assert last == 'xu2 s child\n'


def test_width_one_bus_undeclared_name_refuses_before_placement(tmp_path):
    parent, child = modules(tmp_path, PortDir.IN, bits=1)
    before = set(parent.net_dict)
    with pytest.raises(CircuitError, match='not declared'):
        parent.add_module(child, 'u1', d='nope')
    assert set(parent.net_dict) == before
    assert 'u1' not in parent.elem_dict


def test_width_one_bus_wider_parent_refuses(tmp_path):
    parent, child = modules(tmp_path, PortDir.IN, bits=1)
    with pytest.raises(CircuitError, match='BUS_WIDTH_MISMATCH'):
        parent.add_module(child, 'u1', d=parent.add_net('w', bits=2))


@pytest.mark.parametrize('sliced', [False, True])
@pytest.mark.parametrize('direction', [PortDir.IN, PortDir.OUT, PortDir.INOUT])
def test_single_bit_handle_uses_existing_bit(tmp_path, sliced, direction):
    parent, child = modules(tmp_path, direction)
    handle = parent.add_net('b', bits=4 if sliced else 1)
    bit_name = 'b[2]' if sliced else 'b[0]'
    if sliced:
        handle = handle[2:3]
    before = set(parent.net_dict)
    if direction != PortDir.OUT:
        parent.add_module(child, 'u1', d=handle)
        expected = ('xu1 b[2] b[2] b[2] b[2] child\n' if sliced
                    else 'xu1 b[0] b[0] b[0] b[0] child\n')
        assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1] == expected
    else:
        with pytest.raises(CircuitError) as caught:
            parent.add_module(child, 'u1', d=handle)
        assert f"parent net '{bit_name}' has width 1" in str(caught.value)
    assert set(parent.net_dict) == before


@pytest.mark.parametrize('naming,expected', [
    ('bracketed', 'xu1 A[4] A[5] parent_child\n'),
    ('underscore', 'xu1 A_4 A_5 parent_child\n'),
])
@pytest.mark.parametrize('connection', ['handle', 'string', 'add_net'])
def test_parent_bus_port_passes_to_child(tmp_path, naming, expected, connection):
    parent, child = modules(tmp_path, PortDir.IN, bits=2)
    child.define_module_name('parent_child')
    parent.define_module_ports({'A': Port(dir=PortDir.IN, bits=range(4, 6), naming=naming)})
    if connection == 'string':
        value = 'A'
    elif connection == 'add_net':
        value = parent.add_net('A', bits=range(4, 6))
        assert parent.add_net('A', bits=range(4, 6)) is value
    else:
        value = parent.nets.A
    parent.add_module(child, 'u1', d=value)
    assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1] == expected
    assert 'A' not in parent.net_dict


@pytest.mark.parametrize('direction', [PortDir.IN, PortDir.INOUT])
@pytest.mark.parametrize('port_bus', [False, True])
@pytest.mark.parametrize('width', [1, 2, 8])
def test_string_bus_width_mismatch_is_not_scalar_broadcast(tmp_path, port_bus, width, direction):
    parent, child = modules(tmp_path, direction)
    if port_bus:
        parent.define_module_ports({'A': Port(dir=PortDir.IN, bits=width)})
    else:
        parent.add_net('A', bits=width)
    with pytest.raises(CircuitError, match='BUS_WIDTH_MISMATCH'):
        parent.add_module(child, 'bad', d='A')
    assert 'bad' not in parent.elem_dict


def test_internal_bus_string_expands(tmp_path):
    parent, child = modules(tmp_path, PortDir.OUT, bits=2)
    parent.add_net('A', bits=range(4, 6), naming='underscore')
    parent.add_module(child, 'u1', d='A')
    assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1] == 'xu1 A_4 A_5 child\n'


@pytest.mark.parametrize('direction', [PortDir.IN, PortDir.INOUT])
def test_unknown_string_cannot_implicitly_tie_off_bus(tmp_path, direction):
    parent, child = modules(tmp_path, direction)
    with pytest.raises(CircuitError, match='missing.*declare|declare.*missing'):
        parent.add_module(child, 'bad', d='missing')
    assert 'missing' not in parent.net_dict
    assert 'bad' not in parent.elem_dict


@pytest.mark.parametrize('direction', [PortDir.POWER, PortDir.GROUND])
@pytest.mark.parametrize('as_string', [False, True])
def test_supply_direction_bus_is_not_scalar_broadcast(tmp_path, direction, as_string):
    parent, child = modules(tmp_path, direction)
    scalar = parent.add_net('vss')
    with pytest.raises(CircuitError, match='BUS_WIDTH_MISMATCH'):
        parent.add_module(child, 'u1', d='vss' if as_string else scalar)
    assert 'u1' not in parent.elem_dict


@pytest.mark.parametrize('as_string', [False, True])
def test_inout_supply_role_bus_broadcasts_scalar(tmp_path, as_string):
    from silk.circuit.ports import PortRole
    child = Module('sky130', str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({'d': Port(dir=PortDir.INOUT, bits=4, role=PortRole.POWER)})
    parent = Module('sky130', str(tmp_path), 'parent.cir', None, module_name='parent')
    scalar = parent.add_net('vdd')
    parent.add_module(child, 'u1', d='vdd' if as_string else scalar)
    assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1] == 'xu1 vdd vdd vdd vdd child\n'


@pytest.mark.parametrize('naming,bit_names', [
    ('bracketed', ['P[4]', 'P[5]', 'P[6]']),
    ('underscore', ['P_4', 'P_5', 'P_6']),
])
@pytest.mark.parametrize('path', ['element', 'passive', 'submodule'])
def test_bare_bus_port_cannot_be_wired_as_scalar(tmp_path, naming, bit_names, path):
    from silk.elements import ElementLibrary
    from silk.elements.passives import Resistor

    parent, child = modules(tmp_path, PortDir.IN)
    parent.define_module_ports({'P': Port(dir=PortDir.IN, bits=range(4, 7), naming=naming)})
    with pytest.raises(CircuitError) as caught:
        if path == 'element':
            parent.add(ElementLibrary().resistor('r1', p='P', n='gnd', value=1000))
        elif path == 'passive':
            parent.add(Resistor('r1', 'P', 'gnd', 1000))
        else:
            child = Module('sky130', str(tmp_path), 'scalar.cir', None, module_name='scalar')
            child.define_module_ports({'pin': PortDir.IN})
            parent.add_module(child, 'u1', pin='P')
    message = str(caught.value)
    assert 'P' in message
    assert all(name in message for name in bit_names)
    assert 'P' not in parent.net_dict


@pytest.mark.parametrize('as_string', [False, True])
@pytest.mark.parametrize('scalar_port', [False, True])
def test_declared_scalar_net_or_port_ties_off_input_bus(tmp_path, as_string, scalar_port):
    parent, child = modules(tmp_path, PortDir.IN)
    if scalar_port:
        parent.define_module_ports({'vss': PortDir.GROUND})
    else:
        parent.add_net('vss')
    value = 'vss' if as_string else parent.add_net('vss')
    parent.add_module(child, 'x1', nets={'d': value})
    assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines(keepends=True)[-1] == 'xx1 vss vss vss vss child\n'


@pytest.mark.parametrize('naming,bit_names', [
    ('bracketed', ['P[4]', 'P[5]']), ('underscore', ['P_4', 'P_5']),
])
@pytest.mark.parametrize('strict', [False, True])
def test_port_handle_materializes_only_connected_bits(tmp_path, naming, bit_names, strict):
    from silk.elements import ElementLibrary

    parent = Module('sky130', str(tmp_path), 'parent.cir', None,
                    module_name='parent', strict_nets=strict)
    parent.define_module_ports({'P': Port(dir=PortDir.IN, bits=range(4, 6), naming=naming)})
    handle = parent.nets.P
    assert not parent.net_dict
    parent.add_net('vss')
    parent.add(ElementLibrary().resistor('r1', p=handle[4], n='vss', value=1000))
    assert set(parent.net_dict) == {bit_names[0], 'vss'}
    assert parent.net_dict[bit_names[0]] is handle[4]
    assert handle[4].pins[0].instance == 'r1'
    assert parent.add_net(bit_names[1]) is handle[5]
    assert parent.add_net('P', bits=range(4, 6)) is handle


def test_lazy_port_bit_case_variant_keeps_binding_index_consistent(tmp_path):
    from silk.elements import ElementLibrary

    parent, _ = modules(tmp_path, PortDir.IN)
    parent.define_module_ports({'P': Port(dir=PortDir.IN, bits=2)})
    parent.add(ElementLibrary().resistor('r1', p='p[0]', n='gnd', value=1000))
    stored = parent.elem_dict['r1']['device']
    stored.p = 'P[0]'
    assert len(parent.nets.P[0].pins) == 1
    assert parent.nets.P[0].pins[0].net == 'P[0]'
    stored.p = 'other'
    assert not parent.nets.P[0].pins


def test_port_handle_reuses_preexisting_case_alias(tmp_path):
    from silk.elements import ElementLibrary

    parent, _ = modules(tmp_path, PortDir.IN)
    existing = parent.add_net('p[0]')
    parent.define_module_ports({'P': Port(dir=PortDir.IN, bits=2)})
    handle = parent.nets.P
    assert handle[0] is existing
    parent.add(ElementLibrary().resistor('r1', p='P[1]', n='gnd', value=1000))
    assert parent.net_dict['P[1]'] is handle[1]
    assert handle[1].pins[0].instance == 'r1'
    assert parent.add_net('P', bits=2) is handle


@pytest.mark.parametrize('naming,bit', [('bracketed', 'd[0]'), ('underscore', 'd_0')])
@pytest.mark.parametrize('width', [1, 4])
def test_bit_only_maps_and_bus_alone(tmp_path, naming, bit, width):
    parent, child = modules(tmp_path, PortDir.IN, bits=width, naming=naming)
    assert list(child.pin_list) == list(child._buses['d'].expanded_names())
    assert child.generate_subckt(0) == '.subckt child ' + ' '.join(child.pin_list) + ' \n'
    wiring = {name: f'x{i}' for i, name in enumerate(child.pin_list)}
    parent.add_module(child, 'u', wiring)
    line = parent.netlister.generate_device_instances(parent.elem_dict).splitlines()[-1]
    assert line == 'xu ' + ' '.join(wiring.values()) + ' child'


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('entry', ['module', 'dut', 'helper'])
def test_overlapping_keys_refuse_before_placement(tmp_path, reverse, mixed, entry):
    from silk.exceptions import DuplicateNameError
    parent, child = modules(tmp_path, PortDir.IN)
    bus = parent.add_net('b', bits=4)
    target = parent
    if entry != 'module':
        tb = child.create_test('overlap')
        tb._create_netlist()
        target = tb.netlist
        bus = target.add_net('b', bits=4)
    items = [('d', bus), ('d[0]', bus[0])]
    if reverse:
        items.reverse()
    before = (dict(target.net_dict), dict(target.elem_dict),
              {key: list(net.pins) for key, net in target.net_dict.items()},
              dict(target.module_dict))
    with pytest.raises(DuplicateNameError, match=r'd\[0\]'):
        if entry == 'dut':
            tb.add_module_under_test(dict(items))
        elif entry == 'helper':
            tb.add_module(child, 'bad', dict(items[:1]) if mixed else dict(items),
                          **dict(items[1:]) if mixed else {})
        else:
            parent.add_module(child, 'bad', dict(items[:1]) if mixed else dict(items),
                              **dict(items[1:]) if mixed else {})
    assert dict(target.net_dict) == before[0]
    assert dict(target.elem_dict) == before[1]
    assert {key: list(net.pins) for key, net in target.net_dict.items()} == before[2]
    assert dict(target.module_dict) == before[3]
    assert all(pin.net is None for pin in child.pin_list.values())


def test_kwargs_override_same_key_before_expansion(tmp_path):
    parent, child = modules(tmp_path, PortDir.IN)
    original = parent.add_net('a', bits=4)
    replacement = parent.add_net('b', bits=4)
    parent.add_module(child, 'u', {'d': original}, d=replacement)
    assert parent.netlister.generate_device_instances(parent.elem_dict).splitlines()[-1] == (
        'xu b[0] b[1] b[2] b[3] child')


@pytest.mark.parametrize('marker', [False, True])
def test_internal_only_bus_is_not_formal_target(tmp_path, marker):
    from silk.exceptions import CircuitError
    parent, child = modules(tmp_path, PortDir.IN)
    child.add_net('internal', bits=2)
    if marker:
        parent.add_module(child, 'u', d=parent.add_net('b', bits=4))
        with pytest.raises(CircuitError, match='not declared'):
            parent.add_no_connect_port('u', 'internal', 'unused')
        assert not parent._no_connect_ports
    else:
        before = dict(parent.net_dict)
        with pytest.raises(CircuitError, match='not in defined pins'):
            parent.add_module(child, 'bad', {'internal': 'new'})
        assert dict(parent.net_dict) == before
        assert not parent.elem_dict


def test_no_connect_bus_and_bit_share_idempotence(tmp_path):
    from silk.exceptions import DuplicateNameError
    parent, child = modules(tmp_path, PortDir.IN)
    parent.add_module(child, 'u')
    markers = parent.add_no_connect_port('u', 'd', 'unused')
    assert parent.add_no_connect_port('u', 'd', 'unused') == markers
    assert parent.add_no_connect_port('u', 'd[0]', 'unused') is markers[0]
    with pytest.raises(DuplicateNameError, match='different'):
        parent.add_no_connect_port('u', 'd[0]', 'different')
    assert parent._no_connect_ports == markers


@pytest.mark.parametrize('naming', ['bracketed', 'underscore'])
@pytest.mark.parametrize('start', [0, 4])
@pytest.mark.parametrize('direction', [None, PortDir.IN, PortDir.OUT, PortDir.INOUT])
def test_internal_bus_adopts_formal_ports_without_replacing_electrical_bits(
        tmp_path, naming, start, direction):
    from silk.circuit.ports import PortRole
    from silk.circuit.voltage_domain import VoltageDomain
    from silk.elements import ElementLibrary
    m = Module(None, str(tmp_path), 'adopt.cir', None, module_name='adopt')
    m.define_module_ports({'before': PortDir.IN})
    indices = range(start, start + 4)
    bus = m.add_net('d', bits=indices, naming=naming)
    bit_names = m._buses['d'].expanded_names()
    m.add(ElementLibrary().resistor('r', p=bit_names[0], n='0', value=1000))
    nets = dict(m.net_dict)
    bindings = {name: list(net.pins) for name, net in nets.items()}
    domain = VoltageDomain('vdd', nominal=1.8)
    port = Port(dir=direction, bits=indices, naming=naming,
                role=PortRole.DIGITAL, domain=domain)
    m.define_module_ports({'d': port, 'after': PortDir.OUT})
    assert list(m.pin_list) == ['before', *bit_names, 'after']
    assert m.nets.d is bus
    assert dict(m.net_dict) == nets
    assert all(m.net_dict[name] is net for name, net in nets.items())
    assert {name: list(net.pins) for name, net in m.net_dict.items()} == bindings
    assert all(current is previous for name in bindings
               for current, previous in zip(m.net_dict[name].pins, bindings[name]))
    for name in bit_names:
        assert m.pin_list[name].direction == (direction or PortDir.INOUT)
        assert m.pin_list[name].role == PortRole.DIGITAL
        assert m.pin_list[name].domain is domain
    assert m._buses['d'].domain == 'vdd'
    pins = dict(m.pin_list)
    m.define_module_ports({'d': port})
    assert list(m.pin_list) == list(pins)
    assert all(m.pin_list[name] is pin for name, pin in pins.items())
    assert m.nets.d is bus


@pytest.mark.parametrize('change', ['indices', 'naming', 'case', 'formal_bit', 'formal_alias'])
def test_internal_bus_adoption_preflights_all_collisions(tmp_path, change):
    from silk.exceptions import DuplicateNameError
    m = Module(None, str(tmp_path), 'refuse.cir', None, module_name='refuse')
    bus = m.add_net('d', bits=range(4, 8))
    if change.startswith('formal_'):
        # A preexisting formal claim must still refuse, even in retained state.
        from silk.circuit._module_base import ModulePin
        key = 'd[4]' if change == 'formal_bit' else 'd'
        m.pin_list = {key: ModulePin(name=key, direction=PortDir.IN)}
    before = (dict(m.pin_list or {}), dict(m.net_dict), dict(m._buses))
    args = {'bits': range(5, 9) if change == 'indices' else range(4, 8),
            'naming': 'underscore' if change == 'naming' else 'bracketed'}
    with pytest.raises(DuplicateNameError):
        m.define_module_ports({'D' if change == 'case' else 'd': Port(dir=None, **args)})
    assert (dict(m.pin_list or {}), dict(m.net_dict), dict(m._buses)) == before
    assert m.nets.d is bus


@pytest.mark.parametrize('port_bus', [False, True])
def test_bus_alias_connection_diagnostic_restores_handle_hint(tmp_path, port_bus):
    from silk.elements import ElementLibrary
    from silk.exceptions import DuplicateNameError
    m = Module(None, str(tmp_path), 'hint.cir', None, module_name='hint')
    if port_bus:
        m.define_module_ports({'d': Port(dir=None, bits=2)})
    else:
        m.add_net('d', bits=2)
    with pytest.raises(DuplicateNameError) as caught:
        m.add(ElementLibrary().resistor('r', p='D', n='0', value=1000))
    text = str(caught.value)
    for expected in ["module 'hint'", "bus 'd'", "'D'", 'scalar net',
                     'd[0]', 'd[1]', "module.nets['d']", 'matching child bus port',
                     'differs only by case']:
        assert expected in text
    assert 'conflicting declaration' not in text


@pytest.mark.parametrize('direction', [PortDir.IN, PortDir.OUT, PortDir.INOUT])
@pytest.mark.parametrize('entry', ['dut', 'helper'])
def test_width_one_bus_scalar_by_name_through_testbench(tmp_path, direction, entry):
    _, child = modules(tmp_path, direction, bits=range(3, 4))
    tb = child.create_test('w1')
    tb._create_netlist()
    tb.netlist.add_net('s')
    if entry == 'dut':
        tb.add_module_under_test({'d': 's'})
        assert 'xdut s child' in tb.netlist.netlister.generate_device_instances(tb.netlist.elem_dict)
    else:
        tb.add_module(child, 'u1', {'d': 's'})
        assert 'xu1 s child' in tb.netlist.netlister.generate_device_instances(tb.netlist.elem_dict)
