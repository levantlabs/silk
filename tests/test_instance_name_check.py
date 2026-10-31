"""Placement identity, immutable names and atomic rollback (TASK-528)."""
import copy
import pickle

import pytest

from silk import Module, sources
from silk.circuit import element_names
from silk.circuit.emitted_names import emitted_component
from silk.elements import elements
from silk.elements.passives import PassiveBase, Resistor
from silk.exceptions import CircuitError, DuplicateNameError, NotSupportedError, UndeclaredNetError


def bench(tmp_path, pdk, *, strict=False, child=False):
    return Module('sky130', str(tmp_path), 'names.cir', pdk_details=pdk,
                  strict_nets=strict, module_name='child' if child else None)


def resistor(name):
    return elements.resistor(name, p='out', n='0', value=1000)


class CustomResistor(PassiveBase):
    def physical_instance_names(self):
        return ('R' + emitted_component(self.name),)

    def to_spice(self):
        return f'{self.physical_instance_names()[0]} {self.pos} {self.neg} {self.value}\n'


@pytest.mark.parametrize('path', ['add', 'add_element_instance', 'factory', 'legacy',
                                 'passive', 'source', 'child', 'context'])
def test_each_placement_path_checks_case_fold(tmp_path, pdk, path):
    mod = bench(tmp_path, pdk)
    child = bench(tmp_path / 'child', pdk, child=True)
    def place(name):
        if path == 'add':
            mod.add(resistor(name))
        elif path == 'add_element_instance':
            mod.add_element(resistor(name), name)
        elif path in ('factory', 'legacy'):
            device = pdk.nmos1p8 if path == 'factory' else pdk._devices['nmos1p8']
            mod.add_element(device, name, nets={'d': 'out', 'g': 'out', 's': '0', 'b': '0'})
        elif path == 'passive':
            mod.add_passive(Resistor(name, 'out', '0', 1000))
        elif path == 'source':
            mod.add_source(sources.VoltageSource(name, 'out', '0', dc=1))
        elif path == 'child':
            return mod.add_module(child, name)
        else:
            with mod:
                resistor(name)
    place('Part')
    before = {n: tuple(net.pins) for n, net in mod.net_dict.items()}
    with pytest.raises(DuplicateNameError) as exc:
        place('part')
    assert 'Part' in str(exc.value) and 'part' in str(exc.value)
    assert 'SPICE folds instance-name case' in str(exc.value)
    assert before == {n: tuple(net.pins) for n, net in mod.net_dict.items()}


@pytest.mark.parametrize('kind', ['element', 'source'])
def test_exact_slot_clash(tmp_path, pdk, kind):
    mod = bench(tmp_path, pdk)
    if kind == 'element':
        mod.add(resistor('a'))
        candidate = elements.capacitor('a', p='out', n='0', value=1e-12)
    else:
        mod.add(sources.VoltageSource('a', 'out', '0', dc=1))
        candidate = sources.CurrentSource('a', 'out', '0', dc=1)
    with pytest.raises(DuplicateNameError, match='exact destination-slot clash'):
        mod.add(candidate)
    candidate.name = 'free'


@pytest.mark.parametrize('same_object', [False, True])
def test_cross_registry_emitted_clash(tmp_path, pdk, same_object):
    mod = bench(tmp_path, pdk)
    first = resistor('a')
    mod.add(first)
    candidate = first if same_object else Resistor('a', 'out', '0', 1000)
    before = {n: tuple(net.pins) for n, net in mod.net_dict.items()}
    with pytest.raises(DuplicateNameError, match='both paths emit the same SPICE instance') as exc:
        mod.add_passive(candidate)
    assert 'case' not in str(exc.value)
    assert not mod.passives_dict
    assert before == {n: tuple(net.pins) for n, net in mod.net_dict.items()}


@pytest.mark.parametrize('path', ['add', 'add_element', 'passive', 'source'])
def test_same_slot_same_object_is_idempotent(tmp_path, pdk, path):
    mod = bench(tmp_path, pdk)
    obj = sources.VoltageSource('a', 'out', '0', dc=1) if path == 'source' else resistor('a')
    def place():
        if path == 'add_element':
            mod.add_element(obj, obj.name)
        else:
            getattr(mod, 'add_' + path if path in ('source', 'passive') else 'add')(obj)
    place()
    before = {n: tuple(net.pins) for n, net in mod.net_dict.items()}
    place()
    assert before == {n: tuple(net.pins) for n, net in mod.net_dict.items()}


def test_source_and_resistor_same_designer_name_allowed(tmp_path, pdk):
    mod = bench(tmp_path, pdk)
    mod.add(resistor('a'), sources.VoltageSource('a', 'out', '0', dc=1))
    mod.generate_netlist()
    assert len(mod.elem_dict) == len(mod.sig_sources_dict) == 1


def test_child_two_placement_names_allowed(tmp_path, pdk):
    mod = bench(tmp_path, pdk)
    child = bench(tmp_path / 'child', pdk, child=True)
    assert mod.add_module(child, 'one')._def is child
    assert mod.add_module(child, 'two')._def is child


@pytest.mark.parametrize('path', ['add', 'passive', 'source', 'custom'])
def test_placed_name_read_only(tmp_path, pdk, path):
    mod = bench(tmp_path, pdk)
    obj = (sources.VoltageSource('a', 'out', '0', dc=1) if path == 'source' else
           CustomResistor('a', 'out', '0', 1000) if path == 'custom' else resistor('a'))
    getattr(mod, 'add_source' if path == 'source' else 'add_passive' if path in ('passive', 'custom') else 'add')(obj)
    with pytest.raises(CircuitError, match='name is read-only'):
        obj.name = 'renamed'
    assert obj.name == 'a'


@pytest.mark.parametrize('path', ['add', 'passive', 'source'])
def test_failed_add_remains_renamable(tmp_path, pdk, path):
    mod = bench(tmp_path, pdk, strict=True)
    mod.add_net('out')
    obj = sources.VoltageSource('a', 'out', 'missing', dc=1) if path == 'source' else elements.resistor('a', p='out', n='missing', value=1000)
    with pytest.raises(UndeclaredNetError):
        getattr(mod, 'add_' + path if path in ('source', 'passive') else 'add')(obj)
    assert not mod.net_dict['out'].pins
    assert not mod.elem_dict and not mod.passives_dict and not mod.sig_sources_dict
    obj.name = 'free'


@pytest.mark.parametrize('path', ['add', 'source'])
def test_shared_object_failed_add_preserves_lock_and_owner(tmp_path, pdk, path):
    owner = bench(tmp_path / 'owner', pdk)
    target = bench(tmp_path / 'target', pdk, strict=True)
    target.add_net('out')
    obj = sources.VoltageSource('a', 'out', '0', dc=1) if path == 'source' else resistor('a')
    method = 'add_source' if path == 'source' else 'add'
    getattr(owner, method)(obj)
    with pytest.raises(UndeclaredNetError):
        getattr(target, method)(obj)
    assert obj._module is owner
    with pytest.raises(CircuitError, match='read-only'):
        obj.name = 'free'
    assert not target.net_dict['out'].pins


@pytest.mark.parametrize('clone', [copy.deepcopy, lambda m: pickle.loads(pickle.dumps(m))], ids=['deepcopy', 'pickle'])
def test_module_copy_preserves_index_and_locks(tmp_path, pdk, clone):
    mod = bench(tmp_path, pdk)
    mod.add(resistor('a'))
    mod.add_passive(CustomResistor('b', 'out', '0', 1000))
    mod.add_source(sources.VoltageSource('c', 'out', '0', dc=1))
    other = clone(mod)
    for registry, name, obj in other._element_mgr._instance_names.values():
        entry = getattr(other, registry)[name]
        assert obj is (entry['device'] if registry == 'elem_dict' else entry)
        assert obj is not (getattr(mod, registry)[name]['device'] if registry == 'elem_dict' else getattr(mod, registry)[name])
        with pytest.raises(CircuitError, match='read-only'):
            obj.name = 'free'
    with pytest.raises(DuplicateNameError):
        other.add(resistor('A'))


@pytest.mark.parametrize('port', ['inp', 'positive'])
def test_failed_child_placement_preserves_same_named_source(tmp_path, pdk, port):
    mod = bench(tmp_path, pdk, strict=True)
    mod.add_net('out')
    mod.add_net('0')
    mod.add_net('other')
    mod.add(resistor('before'))
    src = sources.VoltageSource('a', 'out', '0', dc=1)
    mod.add_source(src)
    mod.add(resistor('after'))
    child = bench(tmp_path / 'child', pdk, child=True)
    child.define_module_ports({port: 'input', 'out': 'output'})
    before = {n: tuple(net.pins) for n, net in mod.net_dict.items()}
    reverse = dict(mod._element_mgr._pin_to_net)
    with pytest.raises(UndeclaredNetError):
        mod.add_module(child, 'a', nets={port: 'other' if port == 'positive' else 'out', 'out': 'missing'})
    assert mod.sig_sources_dict['a'] is src and src._module is mod
    assert reverse == mod._element_mgr._pin_to_net
    for n, pins in before.items():
        assert len(mod.net_dict[n].pins) == len(pins)
        assert all(a is b for a, b in zip(mod.net_dict[n].pins, pins))
    assert set(mod.elem_dict) == {'before', 'after'} and not mod.module_dict
    assert mod._element_mgr._instance_names['va'] == ('sig_sources_dict', 'a', src)
    with pytest.raises(CircuitError, match='read-only'):
        src.name = 'free'


def test_custom_passive_resistor_card_collides_with_ideal_resistor(tmp_path, pdk):
    mod = bench(tmp_path, pdk)
    custom = CustomResistor('a', 'out', '0', 1000)
    assert element_names(custom).card_names == ('Ra',)
    assert element_names(custom).current_probe is None
    mod.add_passive(custom)
    with pytest.raises(DuplicateNameError):
        mod.add(resistor('a'))


def test_custom_passive_resistor_card_does_not_collide_with_child(tmp_path, pdk):
    mod = bench(tmp_path, pdk)
    mod.add_passive(CustomResistor('a', 'out', '0', 1000))
    child = bench(tmp_path / 'child', pdk, child=True)
    assert mod.add_module(child, 'a')._def is child


def test_custom_passive_without_card_identity_refused_before_mutation(tmp_path, pdk):
    mod = bench(tmp_path, pdk)
    with pytest.raises(NotSupportedError, match='physical_instance_names'):
        mod.add_passive(PassiveBase('a', 'out', '0', 1000))
    assert not mod.passives_dict and not mod.net_dict


@pytest.mark.parametrize('cards', [(), 'Ra', ('',), ('R a',), ('Ra', 'rA')])
def test_malformed_custom_identity_refused(tmp_path, pdk, cards):
    class Malformed(CustomResistor):
        def physical_instance_names(self):
            return cards
    mod = bench(tmp_path, pdk)
    with pytest.raises(CircuitError, match='card'):
        mod.add_passive(Malformed('a', 'out', '0', 1000))
    assert not mod.passives_dict and not mod.net_dict


def test_source_physical_cards_registered_and_topology_stable(tmp_path, pdk):
    mod = bench(tmp_path, pdk)
    src = sources.VoltageSource('a', 'out', '0', dc=1, pwl=[(0, 0), (1, 1)])
    mod.add_source(src)
    assert set(mod._element_mgr._instance_names) == {'va__silk_pwl', 'va__silk_dc'}
    with pytest.raises(CircuitError):
        src.pwl = None
    assert set(mod._element_mgr._instance_names) == {'va__silk_pwl', 'va__silk_dc'}


@pytest.mark.parametrize('size', [20, 100, 500])
def test_name_inventory_calls_are_linear(tmp_path, pdk, monkeypatch, size):
    import silk.circuit.emitted_names as names
    original = names.element_names
    calls = []
    def counted(obj):
        calls.append(obj)
        return original(obj)
    monkeypatch.setattr(names, 'element_names', counted)
    mod = bench(tmp_path, pdk)
    for i in range(size):
        mod.add(resistor(f'r{i}'))
    assert len(mod.elem_dict) == size
    assert len(calls) == size


def test_child_same_slot_readd_keeps_wiring_and_handle(tmp_path, pdk):
    mod = bench(tmp_path, pdk)
    child = bench(tmp_path / 'child', pdk, child=True)
    child.define_module_ports({'inp': 'input'})
    first = mod.add_module(child, 'a', inp='out')
    pins = tuple(mod.net_dict['out'].pins)
    again = mod.add_module(child, 'a', inp='different')
    assert again._def is first._def is child
    assert 'different' not in mod.net_dict
    assert tuple(mod.net_dict['out'].pins) == pins


def test_retirement_uses_stored_tokens_and_matches_owner(tmp_path, pdk, monkeypatch):
    mod = bench(tmp_path, pdk)
    obj = CustomResistor('a', 'out', '0', 1000)
    mod.add_passive(obj)
    graph = mod._element_mgr
    def refuse_inventory(self):
        raise AssertionError('retirement must use the placed tokens')
    monkeypatch.setattr(CustomResistor, 'physical_instance_names', refuse_inventory)
    graph._retire_instance('passives_dict', 'a', CustomResistor('a', 'out', '0', 1000))
    assert graph._instance_names['ra'][2] is obj
    graph._retire_instance('passives_dict', 'a', obj)
    assert 'ra' not in graph._instance_names
    with pytest.raises(CircuitError, match='read-only'):
        obj.name = 'free'


def test_add_and_retirement_never_enumerate_name_index(tmp_path, pdk):
    class LookupOnly(dict):
        def __iter__(self):
            raise AssertionError('index enumeration makes adds grow with inventory')
        def items(self):
            raise AssertionError('index enumeration makes adds grow with inventory')
        def values(self):
            raise AssertionError('index enumeration makes adds grow with inventory')
        def keys(self):
            raise AssertionError('index enumeration makes adds grow with inventory')
    mod = bench(tmp_path, pdk)
    for i in range(100):
        mod.add(resistor(f'r{i}'))
    graph = mod._element_mgr
    graph._instance_names = LookupOnly(graph._instance_names)
    mod.add(resistor('new'))
    graph._retire_instance('elem_dict', 'new', mod.elem_dict['new']['device'])
    assert 'rnew' not in graph._instance_names


@pytest.mark.parametrize('path', ['factory', 'legacy'])
def test_unknown_pin_failure_removes_only_attempt_created_nets(tmp_path, pdk, path):
    from silk.exceptions import CircuitError
    mod = bench(tmp_path, pdk)
    source = sources.VoltageSource('a', 'out', '0', dc=1)
    mod.add_source(source)
    before = {n: tuple(net.pins) for n, net in mod.net_dict.items()}
    reverse = dict(mod._element_mgr._pin_to_net)
    device = pdk.nmos1p8 if path == 'factory' else pdk._devices['nmos1p8']
    with pytest.raises(CircuitError, match='not in defined pins'):
        mod.add_element(device, 'a', nets={'d': 'temporary', 'unknown': 'also_temporary'})
    assert set(mod.net_dict) == set(before)
    assert mod._element_mgr._pin_to_net == reverse
    assert {n: tuple(net.pins) for n, net in mod.net_dict.items()} == before
    assert mod._element_mgr._instance_names == {'va': ('sig_sources_dict', 'a', source)}
    mod.add_element(device, 'a', nets={'d': 'out', 'g': 'out', 's': '0', 'b': '0'})
    assert 'a' in mod.elem_dict


@pytest.mark.parametrize('path', ['add_module', 'add_element'])
def test_child_name_refusal_precedes_bus_net_case_mutation(tmp_path, pdk, capsys, path):
    from silk.circuit.ports import Port, PortDir
    mod = bench(tmp_path, pdk)
    child = bench(tmp_path / 'child', pdk, child=True)
    child.define_module_ports({'d': Port(dir=PortDir.IN, bits=2)})
    mod.add_net('Supply')
    mod.add_module(child, 'A', d='Supply')
    graph = mod._element_mgr
    before = (dict(graph._net_casefold), set(graph._case_merge_warned),
              dict(graph._pin_to_net), tuple(mod.net_dict['Supply'].pins),
              dict(graph.elem_dict), dict(graph.module_dict), dict(graph._instance_names))
    capsys.readouterr()
    with pytest.raises(DuplicateNameError):
        getattr(mod, path)(child, 'a', d='supply')
    after = (dict(graph._net_casefold), set(graph._case_merge_warned),
             dict(graph._pin_to_net), tuple(mod.net_dict['Supply'].pins),
             dict(graph.elem_dict), dict(graph.module_dict), dict(graph._instance_names))
    assert after == before
    assert 'WARNING' not in capsys.readouterr().err


def test_child_bus_same_slot_readd_does_not_validate_new_wiring(tmp_path, pdk, capsys):
    from silk.circuit.ports import Port, PortDir
    mod = bench(tmp_path, pdk)
    child = bench(tmp_path / 'child', pdk, child=True)
    child.define_module_ports({'d': Port(dir=PortDir.IN, bits=2)})
    mod.add_net('Supply')
    first = mod.add_module(child, 'A', d='Supply')
    graph = mod._element_mgr
    before = (dict(graph._net_casefold), set(graph._case_merge_warned),
              dict(graph._pin_to_net), tuple(mod.net_dict['Supply'].pins))
    capsys.readouterr()
    again = mod.add_module(child, 'A', d='supply')
    assert again._def is first._def is child
    assert (dict(graph._net_casefold), set(graph._case_merge_warned),
            dict(graph._pin_to_net), tuple(mod.net_dict['Supply'].pins)) == before
    assert 'WARNING' not in capsys.readouterr().err

# TASK-553: authored components, unlike selectors, cannot contain separators.
@pytest.mark.parametrize('name', ['a:b', 'a.b', 'v-bias', '1net', 'bus[en]',
                                  'net[3]bias', 'd[1][2]', 'a[-1]', 'gnd!', 'é', 'a$'])
@pytest.mark.parametrize('path', ['net', 'object', 'port', 'pin', 'pdk', 'passive',
                                 'source_pin', 'map', 'instance', 'source', 'child'])
def test_authored_name_rule_every_path(tmp_path, pdk, name, path):
    from silk.circuit.net import Net
    from silk.circuit.ports import PortDir
    mod = bench(tmp_path, pdk)
    child = bench(tmp_path / 'child', pdk, child=True)
    child.define_module_ports({'p': PortDir.INOUT})
    with pytest.raises((ValueError, CircuitError)) as exc:
        if path == 'net':
            mod.add_net(name)
        elif path == 'object':
            Net(name, mod)
        elif path == 'port':
            child.define_module_ports({name: PortDir.INOUT})
        elif path == 'pin':
            elements.resistor('r', p=name, n='0', value=1)
        elif path == 'pdk':
            pdk.nmos1p8('m', d=name, g='g', s='0', b='0')
        elif path == 'passive':
            Resistor('r', name, '0', 1)
        elif path == 'source_pin':
            sources.VoltageSource('v', name, '0', dc=1)
        elif path == 'map':
            mod.add_module(child, 'x', nets={'p': name})
        elif path == 'instance':
            resistor(name)
        elif path == 'source':
            sources.VoltageSource(name, 'out', '0', dc=1)
        else:
            mod.add_module(child, name, nets={'p': 'out'})
    assert repr(name) in str(exc.value)
    assert 'ASCII' in str(exc.value)
    if ':' in name:
        assert "':' separates hierarchy levels" in str(exc.value)


@pytest.mark.parametrize('mutation', ['net_key', 'net_name', 'pin', 'instance',
                                     'source_pin', 'source_name', 'port'])
def test_emission_rechecks_mutated_names(tmp_path, pdk, mutation):
    from silk.circuit.ports import PortDir
    mod = bench(tmp_path, pdk, child=True)
    mod.define_module_ports({'out': PortDir.INOUT})
    part = resistor('r')
    mod.add(part)
    src = sources.VoltageSource('v', 'out', '0', dc=1)
    mod.add(src)
    bad = 'bad:name'
    if mutation == 'net_key':
        mod._element_mgr.net_dict[bad] = mod._element_mgr.net_dict.pop('out')
    elif mutation == 'net_name':
        mod.net_dict['out'].__dict__['name'] = bad
    elif mutation == 'pin':
        part.nets['p'] = bad
    elif mutation == 'instance':
        object.__setattr__(part, 'name', bad)
    elif mutation == 'source_pin':
        src.__dict__['positive'] = bad
    elif mutation == 'source_name':
        object.__setattr__(src, 'name', bad)
    else:
        mod.pin_list[bad] = mod.pin_list.pop('out')
    with pytest.raises((ValueError, CircuitError), match='bad:name'):
        mod.generate_netlist()


def test_direct_child_entry_rechecks_mutated_pin(tmp_path, pdk):
    from silk.circuit.ports import PortDir
    child = bench(tmp_path, pdk, child=True)
    child.define_module_ports({'p': PortDir.INOUT})
    child.pin_list['p'].net = 'bad:name'
    with pytest.raises(ValueError, match='bad:name'):
        child.generate_netlist_entry('x')


@pytest.mark.parametrize('name', ['part', '_part', 'part_2', 'part[0]', 'part[123]'])
def test_valid_names_shared_across_nets_ports_and_instances(tmp_path, pdk, name):
    from silk.circuit.ports import PortDir
    child = bench(tmp_path, pdk, child=True)
    child.define_module_ports({name: PortDir.INOUT})
    child.add(elements.resistor(name, p=name, n='0', value=1))
    text = child.generate_netlist()
    assert name in text
    assert 'R' + emitted_component(name) in text


def test_gnd_bang_message_explains_ground(tmp_path, pdk):
    with pytest.raises(ValueError) as exc:
        bench(tmp_path, pdk).add_net('gnd!')
    for expected in ("'gnd!'", "ordinary node in SPICE", "silk's ground is '0'",
                     'Port.ground()', 'vss'):
        assert expected in str(exc.value)


@pytest.mark.parametrize('path', ['legacy', 'passive', 'custom_passive', 'pdk', 'legacy_child'])
def test_other_instance_paths_refuse_hierarchy_separator(tmp_path, pdk, path):
    mod = bench(tmp_path, pdk)
    with pytest.raises(CircuitError, match='bad:name'):
        if path == 'legacy':
            mod.add_element(pdk._devices['nmos1p8'], 'bad:name',
                            nets={'d': 'out', 'g': 'out', 's': '0', 'b': '0'})
        elif path == 'passive':
            Resistor('bad:name', 'out', '0', 1)
        elif path == 'custom_passive':
            CustomResistor('bad:name', 'out', '0', 1)
        elif path == 'pdk':
            pdk.nmos1p8('bad:name', d='out', g='out', s='0', b='0')
        else:
            mod.add_element(bench(tmp_path / 'child', pdk, child=True), 'bad:name')


@pytest.mark.parametrize('mutation', ['legacy_pin', 'child_pin', 'nested_net', 'custom_passive',
                                       'ideal_passive_p', 'ideal_passive_n'])
def test_emission_rechecks_legacy_and_nested_names(tmp_path, pdk, mutation):
    from silk.circuit.ports import PortDir
    mod = bench(tmp_path, pdk)
    child = bench(tmp_path / 'child', pdk, child=True)
    child.define_module_ports({'p': PortDir.INOUT})
    child.add(elements.resistor('r', p='p', n='0', value=1))
    mod.add_module(child, 'x', nets={'p': 'out'})
    if mutation == 'legacy_pin':
        mod.add_element(pdk._devices['nmos1p8'], 'm',
                        nets={'d': 'out', 'g': 'out', 's': '0', 'b': '0'})
        mod.elem_dict['m']['device'].pin_list['d'].net = 'bad:name'
    elif mutation == 'child_pin':
        mod.elem_dict['x']['pin_list']['p'].net = 'bad:name'
    elif mutation == 'nested_net':
        child.net_dict['p'].__dict__['name'] = 'bad:name'
    elif mutation.startswith('ideal_passive_'):
        passive = elements.resistor('r', p='out', n='0', value=1)
        mod.add_passive(passive)
        passive.nets[mutation[-1]] = 'bad:name'
    else:
        passive = CustomResistor('r', 'out', '0', 1)
        mod.add_passive(passive)
        passive.pos = 'bad:name'
    with pytest.raises(ValueError, match='bad:name'):
        mod.generate_netlist()


def test_zero_is_only_a_net_name():
    from silk.circuit import validate_instance_name, validate_net_name
    assert validate_net_name('0') is None
    with pytest.raises(CircuitError, match="'0'"):
        validate_instance_name('0')
