"""Atomic rewiring of existing module placements and read-only handles."""
import copy
import pickle

import pytest

from silk import Module
from silk.circuit.ports import Port, PortDir
from silk.elements import elements
from silk.exceptions import CircuitError, DuplicateNameError, UndeclaredNetError


def circuit(tmp_path, name=None, strict=False):
    return Module(None, str(tmp_path), (name or 'top') + '.cir', None,
                  module_name=name, strict_nets=strict)


def setup(tmp_path):
    child = circuit(tmp_path, 'block')
    child.define_module_ports({'p': 'in', 'q': 'out',
                               'd': Port.digital_input(bits=2),
                               'y': Port.digital_output(bits=2),
                               'io': Port(dir=PortDir.INOUT, bits=2)})
    child.add_net('inside')
    child.add(elements.resistor('r', p='p', n='inside', value=1000))
    top = circuit(tmp_path)
    a = top.add_net('a', bits=2)
    top.add_net('b', bits=2)
    top.add_net('tie')
    stage = top.add_module(child, 'stage', p='old', q='out', d=a, y=a, io=a)
    return top, child, stage


def state(top):
    """Include object identity, ordering and metadata, without triggering sync."""
    mgr = top._element_mgr
    return (tuple((k, id(e), id(e.get('pin_list')),
                   tuple((p, id(pin), tuple(vars(pin).items()))
                         for p, pin in e.get('pin_list', {}).items()))
                  for k, e in mgr.elem_dict.items()),
            tuple((k, id(net), id(net.pins),
                   tuple((id(b), tuple(vars(b).items())) for b in net.pins))
                  for k, net in mgr.net_dict.items()),
            dict(mgr._pin_to_net), dict(mgr._net_casefold), set(mgr._case_merge_warned))


def test_partial_mapping_and_placement_identity(tmp_path):
    top, child, stage = setup(tmp_path)
    entry = top._element_mgr.elem_dict['stage']
    top.add_module(child, 'other', p='old', q='out')
    elsewhere = circuit(tmp_path, 'elsewhere')
    elsewhere.add_module(child, 'stage', p='old', q='out')
    foreign = elsewhere.add_net('foreign')
    top.connect_module('stage', {'p': 'ignored'}, p=foreign)
    assert top._element_mgr.elem_dict['stage'] is entry
    assert entry['pin_list']['p'].net == 'foreign'
    assert entry['pin_list']['q'].net == 'out'
    assert top._element_mgr.elem_dict['other']['pin_list']['p'].net == 'old'
    assert elsewhere._element_mgr.elem_dict['stage']['pin_list']['p'].net == 'old'
    assert not any(b.instance == 'stage' and b.port == 'p' for b in top.net_dict['old'].pins)
    assert top._element_mgr._pin_to_net['stage', 'p'] == 'foreign'


@pytest.mark.parametrize('value', ['b', 'handle'])
def test_whole_bus_and_individual_bit(tmp_path, value):
    top, _, _ = setup(tmp_path)
    top.connect_module('stage', d=top.nets.b if value == 'handle' else value)
    pins = top._element_mgr.elem_dict['stage']['pin_list']
    assert [pins[f'd[{i}]'].net for i in range(2)] == ['b[0]', 'b[1]']
    top.connect_module('stage', {'d[1]': 'tie'})
    assert [pins[f'd[{i}]'].net for i in range(2)] == ['b[0]', 'tie']
    top.connect_module('stage', d='tie', io=top.nets.tie)
    assert all(pins[f'{port}[{i}]'].net == 'tie' for port in ('d', 'io') for i in range(2))


@pytest.mark.parametrize('wiring,error', [
    ({'p': 'new', 'missing': 'x'}, CircuitError),
    ({'p': 'new', 'd': 'narrow'}, CircuitError),
    ({'p': 'new', 'y': 'tie'}, CircuitError),
    ({'p': 'new', 'd': 'undeclared_bus'}, UndeclaredNetError),
    ({'p': 'new', 'd': 'b', 'd[0]': 'tie'}, DuplicateNameError),
    ({'p': 'new', 'q': 'bad token'}, ValueError),
])
def test_refusal_preserves_all_maps_and_declarations(tmp_path, wiring, error):
    top, child, _ = setup(tmp_path)
    top.add_net('narrow', bits=3)
    top.add_module(child, 'second', p='old', q='out')
    child.pin_list['p'].direction = PortDir.OUT
    child.pin_list['p'].domain = 'changed'
    child.define_module_ports({'extra': 'in'})
    before = state(top)
    with pytest.raises(error):
        top.connect_module('stage', wiring)
    assert state(top) == before


def test_strict_preflight_uses_shared_acceptance(tmp_path):
    top, _, _ = setup(tmp_path)
    top.define_module_ports({'external': Port.digital_input(bits=2)})
    top.add_net('UPPER')
    top.strict_nets = top._element_mgr.strict_nets = True
    top.connect_module('stage', p='upper', q='external[1]')
    assert top._element_mgr._pin_to_net['stage', 'p'] == 'UPPER'
    assert top._element_mgr._pin_to_net['stage', 'q'] == 'external[1]'
    before = state(top)
    with pytest.raises(UndeclaredNetError):
        top.connect_module('stage', p='tie', q='missing')
    assert state(top) == before


@pytest.mark.parametrize('first_net', ['new', 'upper', 'external[0]'])
def test_later_binding_failure_restores_sync_and_pin_values(tmp_path, monkeypatch, first_net):
    top, child, _ = setup(tmp_path)
    top.add_module(child, 'second', p='old', q='out')
    child.define_module_ports({'unused': 'in'})
    top.elem_dict  # synchronize an unwired declaration before removing it
    del child.pin_list['unused']
    top.add_net('UPPER')
    top.define_module_ports({'external': Port.digital_input(bits=2)})
    deferred = top.nets.external[0]
    deferred_pins = list(deferred.pins)
    child.pin_list['p'].direction = PortDir.OUT
    child.pin_list['p'].domain = 'changed'
    child.define_module_ports({'extra': 'in'})
    before = state(top)
    mgr = top._element_mgr
    bind = mgr._bind_pin_to_net
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError('injected binding failure')
        return bind(*args, **kwargs)

    monkeypatch.setattr(mgr, '_bind_pin_to_net', fail_second)
    with pytest.raises(RuntimeError, match='injected'):
        top.connect_module('stage', p=first_net, q='new_out')
    assert len(calls) == 2
    assert state(top) == before
    assert deferred.pins == deferred_pins


def test_success_synchronizes_metadata_without_rewiring_others(tmp_path):
    top, child, _ = setup(tmp_path)
    child.pin_list['p'].direction = PortDir.OUT
    child.pin_list['p'].domain = 'changed'
    top.connect_module('stage', p='tie')
    pin = top._element_mgr.elem_dict['stage']['pin_list']['p']
    assert (pin.net, pin.direction, pin.domain) == ('tie', PortDir.OUT, 'changed')
    binding = next(b for b in top.net_dict['tie'].pins if b.port == 'p')
    assert (binding.direction, binding.domain) == (PortDir.OUT.value, 'changed')


def test_missing_and_device_target_and_removed_api(tmp_path):
    top, _, _ = setup(tmp_path)
    with pytest.raises(CircuitError, match='stage'):
        top.connect_module('absent', p='tie')
    top.add(elements.resistor('r', p='tie', n='old', value=1000))
    with pytest.raises(CircuitError, match='assignment'):
        top.connect_module('r', p='out')
    assert not hasattr(Module, 'connect_element')
    with pytest.raises(AttributeError):
        top.connect_element('stage', ['p'], ['tie'])


def test_testbench_forwarding_and_no_implicit_netlist(tmp_path):
    _, child, _ = setup(tmp_path)
    tb = child.create_test('rewire')
    assert tb.netlist is None
    with pytest.raises(CircuitError, match='add_module_under_test.*add_module'):
        tb.connect_module('dut', p='new')
    assert tb.netlist is None
    tb.add_module(child, 'helper', p='a', q='b')
    tb.add_module_under_test({'p': 'a', 'q': 'b'})
    for name in ('helper', 'dut'):
        tb.connect_module(name, {'p': 'new'}, q='new_out')
        pins = tb.netlist._element_mgr.elem_dict[name]['pin_list']
        assert (pins['p'].net, pins['q'].net) == ('new', 'new_out')


def test_held_references_follow_rewire_bound_probes_do_not(tmp_path):
    _, child, _ = setup(tmp_path)
    tb = child.create_test('refs')
    stage = tb.add_module(child, 'helper', p='old', q='out')
    tb.netlist.connect_module('helper', d=tb.netlist.add_net('a', bits=2))
    refs = [stage.p, stage.net('p'), stage.bus('d')]
    bound = [tb.probe(r) for r in refs]
    old_names = [s.name for s in bound[:2]]
    old_bus = [s.name for s in bound[2]]
    tb.connect_module('helper', p='new', d=tb.netlist.add_net('b', bits=2))
    assert [tb.probe(r).name for r in refs[:2]] == ['new', 'new']
    assert [s.name for s in tb.probe(refs[2])] == ['b[0]', 'b[1]']
    assert [s.name for s in bound[:2]] == old_names
    assert [s.name for s in bound[2]] == old_bus


def handles(tmp_path):
    top, child, stage = setup(tmp_path)
    leaf = circuit(tmp_path, 'leaf')
    leaf.define_module_ports({'p': 'in'})
    child.add_module(leaf, 'inner', p='p')
    child.add_module(leaf, 'x[0]', p='p')
    child.add_module(leaf, 'x[1]', p='p')
    return top, stage, stage.x


@pytest.mark.parametrize('name', ['p', 'd', 'inside', 'inner', 'r', 'unknown', 'net', 'bus', 'name', 'instance_name', 'subcircuit', 'path', 'child', 'device'])
@pytest.mark.parametrize('target', ['stage', 'array'])
def test_public_handle_assignment_refused(tmp_path, name, target):
    _, stage, array = handles(tmp_path)
    handle = stage if target == 'stage' else array
    with pytest.raises(AttributeError, match='connect_module'):
        setattr(handle, name, 'shadow')
    handle._private = 1
    assert handle._private == 1
    from silk.circuit.device_view import PlacementDeviceView
    view = stage.r
    registered = stage._def.elem_dict['r']['device']
    assert isinstance(view, PlacementDeviceView) and view is not registered
    assert (view.name, view.nets, view.params) == (registered.name, registered.nets, registered.params)
    assert len(array[:]) == 2


@pytest.mark.parametrize('clone', [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_handle_array_and_module_copy(tmp_path, clone):
    top, stage, array = handles(tmp_path)
    top._held_stage = stage
    top._held_array = array
    for original in (stage, array, top):
        result = clone(original)
        if original is top:
            assert result._held_stage.net('p').name == 'p'
            assert len(result._held_array[:]) == 2
        elif original is array:
            assert len(result[:]) == 2
            assert result.name == 'x'
        else:
            assert result.net('p').name == 'p'


@pytest.mark.parametrize('nets', [['pq'], ['p', 'q'], 42])
def test_bare_net_lists_refuse_like_add_module(tmp_path, nets):
    top, _, _ = setup(tmp_path)
    before = state(top)
    with pytest.raises(CircuitError, match='nets|mapping'):
        top.connect_module('stage', nets=nets)
    assert state(top) == before


def test_port_pair_list_matches_add_module(tmp_path):
    top, _, _ = setup(tmp_path)
    top.connect_module('stage', [('p', 'tie')])
    assert top._element_mgr.elem_dict['stage']['pin_list']['p'].net == 'tie'
