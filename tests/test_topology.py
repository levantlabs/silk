"""Ground connectivity is permissive wiring reachability, not a DC solve."""
from dataclasses import FrozenInstanceError

import pytest

from silk import sources, load_pdk
from silk.circuit.modules import Module
from silk.circuit.ports import Port, PortDir
from silk.elements import elements
from silk.exceptions import SimulationError


def top(tmp_path, backend='ngspice', name=None):
    mod = Module(None, str(tmp_path), 'top.cir', None, module_name=name)
    mod.simulator = backend
    return mod


def inspect(mod):
    from silk.simulation.topology import inspect_topology
    return inspect_topology(mod)


@pytest.mark.parametrize('backend', ['ngspice', 'xyce', 'behavioral'])
@pytest.mark.parametrize('spelling', ['0', 'gnd', 'GND'])
@pytest.mark.parametrize('modeled', [False, True])
def test_interface_guard_precedes_alias_resolution_and_modeled_leaf(tmp_path, backend, spelling, modeled):
    from silk.exceptions import CircuitError
    from tests.test_ground_port_guard import _legacy_interface, _assert_ground_diagnosis
    child = _legacy_interface(tmp_path / 'child', spelling)
    if modeled:
        def model(ports, t, params):
            pytest.fail('inspection must not call the model')
        child.set_behavioral_model(model, execution='vector')
    middle = top(tmp_path / 'middle', backend, 'middle')
    middle.add_module(child, 'leaf', nets={spelling: '0'})
    mod = top(tmp_path, backend)
    mod.add_module(middle, 'branch')
    with pytest.raises(CircuitError) as caught:
        inspect(mod)
    _assert_ground_diagnosis(str(caught.value), spelling)


@pytest.mark.parametrize('backend', ['ngspice', 'xyce', 'behavioral'])
@pytest.mark.parametrize('kind', ['direct', 'resistor', 'capacitor', 'inductor'])
def test_ground_path_passes_regardless_of_dc_conduction(tmp_path, backend, kind):
    mod = top(tmp_path, backend)
    negative = '0' if kind == 'direct' else 'return'
    mod.add_source(sources.VoltageSource('drive', positive='out', negative=negative, dc=1))
    if kind != 'direct':
        mod.add(getattr(elements, kind)('return', p='return', n='0', value=1))
    assert inspect(mod).disconnected == ()


def test_ground_elsewhere_does_not_anchor_a_source_group(tmp_path):
    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('good', positive='supply', negative='0', dc=1))
    mod.add_source(sources.CurrentSource('floating', positive='island', negative='return', dc=1))
    result = inspect(mod)
    assert [(d.source, d.nets) for d in result.disconnected] == [
        ('floating', ('island', 'return'))]
    from silk.simulation.topology import require_ground_connections
    with pytest.raises(SimulationError, match='floating.*no connection to ground'):
        require_ground_connections(mod)
    with pytest.raises(FrozenInstanceError):
        result.disconnected = ()


@pytest.mark.parametrize('value', [0, 1, -1])
def test_grounded_voltage_loops_and_current_groups_are_not_solvability_checked(tmp_path, value):
    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('a', positive='out', negative='0', dc=1))
    mod.add_source(sources.VoltageSource('b', positive='out', negative='0', dc=value))
    mod.add_source(sources.VoltageSource('same', positive='0', negative='0', dc=value))
    mod.add_source(sources.CurrentSource('i', positive='out', negative='0', dc=value))
    assert inspect(mod).disconnected == ()


@pytest.mark.parametrize('kind', ['VoltageControlledVoltageSource', 'VoltageControlledCurrentSource'])
def test_sense_pins_join_the_wiring_group(tmp_path, kind):
    mod = top(tmp_path)
    param = {'gain': 2} if kind.endswith('VoltageSource') else {'gm': 2}
    mod.add_source(getattr(sources, kind)('controlled', out_pos='out', out_neg='ref',
                                        ctrl_pos='sense', ctrl_neg='0', **param))
    assert inspect(mod).disconnected == ()


@pytest.mark.parametrize('kind,params', [
    ('CurrentControlledVoltageSource', {'transresistance': 1}),
    ('CurrentControlledCurrentSource', {'gain': 1}),
])
def test_current_control_reference_is_not_a_node_pin(tmp_path, kind, params):
    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('meter', positive='input', negative='0', dc=0))
    mod.add_source(getattr(sources, kind)('controlled', out_pos='out', out_neg='ref',
                                        ctrl_vname='meter', **params))
    assert [d.source for d in inspect(mod).disconnected] == ['controlled']


def test_pdk_transistor_joins_pins_without_exempting_separate_island(tmp_path):
    mod = top(tmp_path)
    pdk = load_pdk('sky130')
    mod.add(pdk.nmos1p8('m', d='out', g='gate', s='0', b='0', w=1, l=.15))
    mod.add_source(sources.VoltageSource('gate', positive='gate', negative='ref', dc=1))
    mod.add_source(sources.VoltageSource('bad', positive='island', negative='return', dc=1))
    result = inspect(mod)
    assert [d.source for d in result.disconnected] == ['bad']
    assert result.unchecked == ()


@pytest.mark.parametrize('backend,ground,connected', [
    ('ngspice', '0', True), ('ngspice', 'GND', True), ('ngspice', 'vss', False),
    ('xyce', '0', True), ('xyce', 'gnd', False), ('xyce', 'vss', False),
    ('behavioral', '0', True), ('behavioral', 'gnd', True), ('behavioral', 'vss', False),
])
def test_internal_ground_is_global_but_vss_role_is_not_ground(tmp_path, backend, ground, connected):
    child = top(tmp_path, backend, 'child')
    child.add_net(ground)
    child.net_dict[ground].classification = 'ground'
    child.add_source(sources.VoltageSource('drive', positive='out', negative=ground, dc=1))
    mod = top(tmp_path, backend)
    mod.add_module(child, 'u')
    result = inspect(mod)
    assert bool(result.disconnected) is not connected
    if not connected:
        assert result.disconnected[0].source == 'xu.drive'


def test_independent_waveform_layout_does_not_add_floating_sources(tmp_path):
    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('bias[3]', positive='a', negative='0', dc=1,
                                        pwl=[(0, 0), (1, 1)]))
    mod.add_source(sources.CurrentSource('current[3]', positive='b', negative='0', dc=1,
                                        pwl=[(0, 0), (1, 1)]))
    assert inspect(mod).disconnected == ()


def test_repeated_placements_keep_unrelated_internal_source_groups_separate(tmp_path):
    child = top(tmp_path, name='child')
    child.define_module_ports({'p': PortDir.IN, 'n': PortDir.IN})
    child.add_source(sources.VoltageSource('drive', positive='p', negative='n', dc=1))
    child.add_source(sources.VoltageSource('private', positive='inside', negative='other', dc=1))
    mod = top(tmp_path)
    mod.add_module(child, 'u[2]', p='out', n='0')
    mod.add_module(child, 'u[3]', p='out2', n='return2')
    result = inspect(mod)
    assert [d.source for d in result.disconnected] == [
        'xu<2>.private', 'xu<3>.drive', 'xu<3>.private']
    assert result.disconnected[0].nets == ('xu<2>.inside', 'xu<2>.other')
    assert result.disconnected[-1].nets == ('xu<3>.inside', 'xu<3>.other')
    # Live placement-local wiring, not the reusable definition's pin values.
    mod.connect_module('u[3]', n='0')
    assert [d.source for d in inspect(mod).disconnected] == ['xu<2>.private', 'xu<3>.private']


def test_nested_bus_aliases_and_input_broadcast_are_resolved(tmp_path):
    leaf = top(tmp_path, name='leaf')
    leaf.define_module_ports({'d': Port(dir=PortDir.IN, bits=range(4, 6))})
    leaf.add_source(sources.VoltageSource('drive', positive='d[4]', negative='d[5]', dc=1))
    middle = top(tmp_path, name='middle')
    middle.define_module_ports({'bus': Port(dir=PortDir.IN, bits=2)})
    middle.add_module(leaf, 'u', nets={'d[4]': 'bus[0]', 'd[5]': 'bus[1]'})
    mod = top(tmp_path)
    mod.add_module(middle, 'a', bus=mod.add_net('0'))
    assert inspect(mod).disconnected == ()
    mod.add_module(middle, 'b', bus=mod.add_net('floating'))
    assert [(d.source, d.nets) for d in inspect(mod).disconnected] == [
        ('xb.xu.drive', ('floating',))]


@pytest.mark.parametrize('backend', ['ngspice', 'behavioral'])
def test_modelled_leaf_joins_ports_but_behavioral_ignores_replaced_sources(tmp_path, backend):
    leaf = top(tmp_path, backend, 'leaf')
    leaf.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    leaf.add_source(sources.VoltageSource('replaced', positive='hidden', negative='other', dc=1))
    def model_1(ports, t, params):
        ports.z.v = ports.a.v
    leaf.set_behavioral_model(model_1, execution='vector')
    mod = top(tmp_path, backend)
    mod.add_module(leaf, 'leaf', a='return', z='0')
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    assert [d.source for d in inspect(mod).disconnected] == (
        [] if backend == 'behavioral' else ['xleaf.replaced'])


def test_inspection_is_read_only_and_does_not_emit_or_evaluate_values(tmp_path):
    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='0', dc=1))
    before = (dict(mod.__dict__), dict(mod._element_mgr.__dict__), set(tmp_path.iterdir()))
    assert inspect(mod).disconnected == ()
    assert mod.__dict__ == before[0]
    assert mod._element_mgr.__dict__ == before[1]
    assert set(tmp_path.iterdir()) == before[2]


def test_custom_emitter_is_not_executed_or_assumed_limited_to_its_pins(tmp_path):
    class Opaque(sources.VoltageSource):
        def to_spice(self):
            raise AssertionError('inspection must not execute arbitrary code')

    mod = top(tmp_path)
    mod.add_source(Opaque('opaque', positive='out', negative='0', dc=1))
    mod.add_source(sources.VoltageSource('separate', positive='island', negative='ref', dc=1))
    result = inspect(mod)
    assert result.disconnected == ()  # Unbounded emitter can mention island.
    assert any('opaque' in reason and 'emitter' in reason for reason in result.unchecked)


def test_custom_waveform_and_selection_are_not_executed(tmp_path):
    class Wave(sources.Pulse):
        def to_spice(self):
            raise AssertionError('no emission')

    mod = top(tmp_path)
    src = sources.VoltageSource('drive', positive='out', negative='ref', dc=1,
                                pulse=Wave(v1=0, v2=1, td=0, tr=1, tf=1, pw=1, per=4))
    mod.add_source(src)
    src.selected_waveform = lambda: pytest.fail('no waveform selection callback')
    result = inspect(mod)
    assert result.disconnected == ()
    assert any('waveform' in reason for reason in result.unchecked)


def test_unbounded_external_include_exempts_without_parsing_it(tmp_path):
    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    mod.add_custom_line('.include "file-that-does-not-exist.spice"')
    result = inspect(mod)
    assert result.disconnected == ()
    assert result.unchecked


def test_replacement_netlister_does_not_certify_live_wiring(tmp_path):
    from silk.circuit.spice_netlister import PySpiceNetlister

    class OpaqueNetlister(PySpiceNetlister):
        pass

    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    mod.netlister = OpaqueNetlister(mod.pdk_details)
    assert inspect(mod).disconnected == ()
    assert inspect(mod).unchecked


@pytest.mark.parametrize('kind', ['BehavioralVoltage', 'BehavioralCurrent'])
def test_behavioral_spice_sources_join_only_declared_terminals(tmp_path, kind):
    mod = top(tmp_path)
    mod.add_source(getattr(sources, kind)('drive', positive='out', negative='return',
                                        expression='v(0)'))
    assert [d.source for d in inspect(mod).disconnected] == ['drive']


def test_legacy_pdk_device_pins_join_without_model_loading(tmp_path):
    mod = Module('sky130', str(tmp_path), 'top.cir', load_pdk('sky130'))
    with pytest.warns(DeprecationWarning):
        mod.add_element(mod.pdk_details.nmos1p8, 'm', nets={'d': 'out', 'g': 'gate', 's': '0', 'b': '0'})
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    assert inspect(mod).disconnected == ()


def test_added_external_model_include_is_opaque_but_shipped_model_is_not(tmp_path):
    mod = Module('sky130', str(tmp_path), 'top.cir', load_pdk('sky130'))
    mod.add(load_pdk('sky130').nmos1p8('m', d='d', g='g', s='0', b='0', w=1, l=.15))
    mod.add_source(sources.VoltageSource('island', positive='a', negative='b', dc=1))
    assert [d.source for d in inspect(mod).disconnected] == ['island']
    mod.pdk_details.extra_includes.append('/unknown/user-wiring.spice')
    assert inspect(mod).disconnected == ()
    assert any('include' in reason for reason in inspect(mod).unchecked)


def test_custom_analysis_and_multiline_option_do_not_certify_connectivity(tmp_path):
    from silk.simulation import Transient

    class Custom(Transient):
        def to_spice(self):
            raise AssertionError('must not execute analysis emitter')

    mod = top(tmp_path)
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    mod.add_sim_type(Custom(1, 2))
    assert inspect(mod).disconnected == ()
    mod.sim_analysis_dict.clear()
    mod.simulator_options['method'] = 'gear\n.global return'
    assert inspect(mod).disconnected == ()


@pytest.mark.parametrize('pdk_name', ['sky130', 'gf180mcu', 'sg13g2'])
@pytest.mark.parametrize('backend', ['ngspice', 'xyce'])
def test_packaged_model_includes_do_not_exempt_source_groups(tmp_path, pdk_name, backend):
    mod = Module(pdk_name, str(tmp_path), 'top.cir', load_pdk(pdk_name))
    mod.simulator = backend
    mod.add_source(sources.VoltageSource('island', positive='a', negative='b', dc=1))
    result = inspect(mod)
    assert [d.source for d in result.disconnected] == ['island']
    assert result.unchecked == ()


def test_unregistered_external_include_is_exempt_not_a_registry_failure(tmp_path):
    mod = Module('sky130', str(tmp_path), 'top.cir', load_pdk('sky130'))
    mod.pdk_details.pdk_name = 'custom'
    mod.pdk_details.extra_includes = ['/unknown/global.spice']
    mod.add_source(sources.VoltageSource('island', positive='a', negative='b', dc=1))
    assert inspect(mod).disconnected == ()


def test_diode_and_switch_join_all_pins(tmp_path):
    from silk.elements.factory import make_device_class
    switch = make_device_class('switch', {
        'spice_prefix': 'S', 'pinList': {'p': 'p', 'n': 'n', 'cp': 'cp', 'cn': 'cn'},
        'paramList': {}}, is_pdk=False)
    mod = top(tmp_path)
    mod.add(load_pdk('sky130').diode_std('d', a='a', k='return'))
    mod.add(switch('s', p='return', n='b', cp='sense', cn='0'))
    mod.add_source(sources.VoltageSource('drive', positive='out', negative='a', dc=1))
    assert inspect(mod).disconnected == ()


def test_array_diagnostic_names_the_authored_path(tmp_path):
    child = top(tmp_path, name='child')
    child.add_source(sources.VoltageSource('drive[3]', positive='a', negative='b', dc=1))
    mod = top(tmp_path)
    mod.add_module(child, 'u[2]')
    from silk.simulation.topology import require_ground_connections
    with pytest.raises(SimulationError) as error:
        require_ground_connections(mod)
    assert 'xu[2].drive[3]' in str(error.value)


def test_opaque_property_renderer_is_not_evaluated(tmp_path):
    class Opaque(sources.VoltageSource):
        @property
        def to_spice(self):
            raise AssertionError('must not execute a renderer descriptor')

    mod = top(tmp_path)
    mod.add_source(Opaque('drive', positive='a', negative='b', dc=1))
    assert inspect(mod).disconnected == ()
    assert inspect(mod).unchecked


def test_multiline_literal_in_builtin_source_is_refused(tmp_path):
    """TASK-593: a multi-line value in a built-in source is refused when it
    is set, so it can never reach a deck (or this check) as opaque text."""
    from silk.exceptions import DeckTextError
    with pytest.raises(DeckTextError, match='line break'):
        sources.VoltageSource('drive', positive='a', negative='b',
                              dc='1\n.global b')


@pytest.mark.parametrize('level', ['class', 'instance'])
@pytest.mark.parametrize('raises', [False, True])
def test_selector_override_is_opaque_without_selection_or_emission(tmp_path, level, raises):
    calls = []

    class HiddenWave(sources.Sine):
        def to_spice(self):
            calls.append('renderer')
            return 'SIN(0 1 1k)\nRanchor return 0 1k'

    hidden = HiddenWave(voffset=0, vampl=1, freq=1000)

    def selector(self=None):
        calls.append('selector')
        if raises:
            raise AssertionError('must not evaluate selector')
        return hidden

    class Source(sources.VoltageSource):
        pass

    if level == 'class':
        Source.selected_waveform = selector
    mod = top(tmp_path)
    src = Source('drive', positive='out', negative='return', dc=1,
                 sine=sources.Sine(voffset=0, vampl=1, freq=1000))
    mod.add_source(src)
    if level == 'instance':
        src.selected_waveform = selector
    result = inspect(mod)
    assert result.disconnected == ()
    assert any('selected_waveform' in reason for reason in result.unchecked)
    assert calls == []


def test_plain_subclass_with_inherited_selector_still_refuses(tmp_path):
    class Source(sources.VoltageSource):
        pass

    mod = top(tmp_path)
    mod.add_source(Source('drive', positive='out', negative='return', dc=1,
                          sine=sources.Sine(voffset=0, vampl=1, freq=1000)))
    assert [d.source for d in inspect(mod).disconnected] == ['drive']
    assert inspect(mod).unchecked == ()


@pytest.mark.parametrize('hook', ['physical_layout', '_node', '_validate_terminal',
                                  '_resolve_ctrl_vname', 'instance_name'])
def test_delegated_source_hook_override_is_opaque_without_calling_it(tmp_path, hook):
    mod = top(tmp_path)
    if hook == '_resolve_ctrl_vname':
        src = sources.CurrentControlledVoltageSource('drive', out_pos='out', out_neg='return',
                                                       ctrl_vname='meter', transresistance=1)
    else:
        src = sources.VoltageSource('drive', positive='out', negative='return', dc=1)
    mod.add_source(src)
    calls = []

    def custom(*args):
        calls.append(hook)
        raise AssertionError('must not execute an emission hook')

    if hook == 'instance_name':
        # A class descriptor override must be examined without invoking it.
        class Custom(sources.VoltageSource):
            instance_name = property(custom)
        src.__class__ = Custom
    else:
        setattr(src, hook, custom)
    result = inspect(mod)
    assert result.disconnected == ()
    assert any(hook in reason for reason in result.unchecked)
    assert calls == []


@pytest.mark.parametrize('depth', [1, 2])
def test_nested_replacement_netlister_is_opaque_without_emission(tmp_path, depth):
    from silk.circuit.spice_netlister import PySpiceNetlister
    calls = []

    class Custom(PySpiceNetlister):
        def generate_passives(self, passives):
            calls.append('renderer')
            return super().generate_passives(passives) + 'Ranchor return 0 1k\n'

    leaf = top(tmp_path, name='leaf')
    leaf.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    branch = leaf
    if depth == 2:
        branch = top(tmp_path, name='middle')
        branch.add_module(leaf, 'leaf')
    mod = top(tmp_path)
    mod.add_module(branch, 'child')
    placed = mod.elem_dict['child']['device']
    if depth == 2:
        placed = placed.elem_dict['leaf']['device']
    placed.netlister = Custom(placed.pdk_details)
    result = inspect(mod)
    assert result.disconnected == ()
    assert any('replacement netlister' in reason for reason in result.unchecked)
    assert calls == []
    assert '_netlister' not in vars(mod)  # No lazy netlister created.


def test_ordinary_nested_modules_do_not_exempt_floating_sibling(tmp_path):
    leaf = top(tmp_path, name='leaf')
    leaf.add_source(sources.VoltageSource('good', positive='out', negative='0', dc=1))
    middle = top(tmp_path, name='middle')
    middle.add_module(leaf, 'leaf')
    mod = top(tmp_path)
    mod.add_module(middle, 'ordinary')
    mod.add_source(sources.VoltageSource('sibling', positive='a', negative='b', dc=1))
    assert [d.source for d in inspect(mod).disconnected] == ['sibling']
    assert inspect(mod).unchecked == ()


def test_behavioral_modelled_leaf_does_not_inherit_nested_spice_opacity(tmp_path):
    from silk.circuit.spice_netlister import PySpiceNetlister

    class Custom(PySpiceNetlister):
        def generate_passives(self, passives):
            raise AssertionError('must not execute nested emitter')

    leaf = top(tmp_path, name='leaf')
    leaf.define_module_ports({'a': PortDir.IN, 'z': PortDir.OUT})
    def model_2(ports, t, params):
        ports.z.v = ports.a.v
    leaf.set_behavioral_model(model_2, execution='vector')
    mod = top(tmp_path, 'behavioral')
    mod.add_module(leaf, 'leaf', a='input', z='0')
    placed = mod.elem_dict['leaf']['device']
    placed.netlister = Custom(placed.pdk_details)
    mod.add_source(sources.VoltageSource('sibling', positive='a', negative='b', dc=1))
    assert [d.source for d in inspect(mod).disconnected] == ['sibling']
    assert inspect(mod).unchecked == ()


def test_selector_bound_to_another_source_is_not_transparent(tmp_path):
    class HiddenWave(sources.Sine):
        def to_spice(self):
            raise AssertionError('must not execute waveform emitter')

    other = sources.VoltageSource('other', positive='out', negative='0',
                                  sine=HiddenWave(voffset=0, vampl=1, freq=1000))
    mod = top(tmp_path)
    src = sources.VoltageSource('drive', positive='out', negative='return', dc=1)
    mod.add_source(src)
    src.selected_waveform = other.selected_waveform
    assert inspect(mod).disconnected == ()
    assert any('selected_waveform' in reason for reason in inspect(mod).unchecked)


@pytest.mark.parametrize('stale', [False, True])
def test_nested_renderer_cache_follows_child_selection_without_lazy_creation(tmp_path, stale):
    from silk.circuit.spice_netlister import PySpiceNetlister

    class Custom(PySpiceNetlister):
        def generate_passives(self, passives):
            raise AssertionError('must not execute renderer')

    leaf = top(tmp_path, name='leaf')
    leaf.add_source(sources.VoltageSource('drive', positive='out', negative='return', dc=1))
    mod = top(tmp_path)
    mod.add_module(leaf, 'child')
    placed = mod.elem_dict['child']['device']
    placed.simulator = 'xyce'
    custom = Custom(placed.pdk_details)
    placed.netlister = custom
    if stale:
        placed.simulator = 'ngspice'
    before = dict(vars(placed))
    result = inspect(mod)
    assert vars(placed) == before
    assert placed._netlister is custom
    if stale:
        assert [d.source for d in result.disconnected] == ['xchild.drive']
        assert result.unchecked == ()
    else:
        assert result.disconnected == ()
        assert any('xchild' in reason and 'replacement netlister' in reason
                   for reason in result.unchecked)
