"""Tests for silk.layout.parasitics — inject_parasitics."""
import pytest

from silk.exceptions import CircuitError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_amp(pdk, tmp_path):
    from silk.circuit.modules import Module
    amp = Module('PDK_skywater130', str(tmp_path), 'amp.cir',
                 pdk_details=pdk, module_name='amp')
    amp.define_module_ports({'vin': 'input', 'vout': 'output', 'vss': 'inout'})
    amp.add(pdk.nmos1p8('mn1', d='vout', g='vin', s='vss', b='vss', w=4, l=0.18, m=1))
    return amp


def _routing_model():
    from silk.layout.routing_model import RoutingModel
    return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                        j_max=1e-3, min_width=0.14, max_width=10.0)


def _run_estimate(module):
    from silk.layout.estimator import estimate_layout
    return estimate_layout(module, _routing_model())


# ---------------------------------------------------------------------------
# c_only mode
# ---------------------------------------------------------------------------

class TestInjectParasiticsCapOnly:

    def test_adds_shunt_capacitors(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='c_only')
        par_caps = [k for k in amp.passives_dict if k.startswith('__par_C_')]
        assert len(par_caps) > 0

    def test_capacitor_for_each_net(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='c_only')
        for net_name in est.nets:
            assert f'__par_C_{net_name}' in amp.passives_dict

    def test_capacitor_value_matches_estimate(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='c_only')
        for net_name, nl in est.nets.items():
            cap = amp.passives_dict[f'__par_C_{net_name}']
            # capacitance stored in F = c (fF) * 1e-15
            assert cap.value == pytest.approx(nl.c * 1e-15)

    def test_no_midpoint_nets_created(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='c_only')
        par_nets = [k for k in amp.net_dict if k.startswith('__par_')]
        assert len(par_nets) == 0

    def test_no_resistors_created(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='c_only')
        par_rs = [k for k in amp.passives_dict if k.startswith('__par_R_')]
        assert len(par_rs) == 0


# ---------------------------------------------------------------------------
# rc mode (lumped-π)
# ---------------------------------------------------------------------------

class TestInjectParasiticsRC:

    def test_adds_resistors(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        par_rs = [k for k in amp.passives_dict if k.startswith('__par_R_')]
        assert len(par_rs) > 0

    def test_adds_two_capacitors_per_net(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        for net_name in est.nets:
            assert f'__par_C1_{net_name}' in amp.passives_dict
            assert f'__par_C2_{net_name}' in amp.passives_dict

    def test_midpoint_nets_created(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        for net_name in est.nets:
            mid = f'__par_{net_name}_mid'
            assert mid in amp.net_dict

    def test_resistor_connects_net_to_midpoint(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        for net_name in est.nets:
            r = amp.passives_dict[f'__par_R_{net_name}']
            mid = f'__par_{net_name}_mid'
            assert r.nets['p'] == net_name
            assert r.nets['n'] == mid

    def test_c1_connects_net_to_zero(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        for net_name in est.nets:
            c1 = amp.passives_dict[f'__par_C1_{net_name}']
            assert c1.nets['p'] == net_name
            assert c1.nets['n'] == '0'

    def test_c2_connects_midpoint_to_zero(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        for net_name in est.nets:
            c2 = amp.passives_dict[f'__par_C2_{net_name}']
            mid = f'__par_{net_name}_mid'
            assert c2.nets['p'] == mid
            assert c2.nets['n'] == '0'

    def test_half_cap_values(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        for net_name, nl in est.nets.items():
            c_half_f = nl.c * 1e-15 / 2.0
            c1 = amp.passives_dict[f'__par_C1_{net_name}']
            c2 = amp.passives_dict[f'__par_C2_{net_name}']
            assert c1.value == pytest.approx(c_half_f)
            assert c2.value == pytest.approx(c_half_f)

    def test_resistor_value(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        for net_name, nl in est.nets.items():
            r = amp.passives_dict[f'__par_R_{net_name}']
            assert r.value == pytest.approx(nl.r)


# ---------------------------------------------------------------------------
# Idempotency / re-injection
# ---------------------------------------------------------------------------

class TestInjectParasiticsIdempotent:

    def test_reinject_replaces_not_duplicates(self, pdk, tmp_path):
        """Calling inject_parasitics twice should not duplicate passives."""
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='c_only')
        count_after_first = len(amp.passives_dict)
        inject_parasitics(amp, est, mode='c_only')
        assert len(amp.passives_dict) == count_after_first

    def test_rc_then_c_only_clears_rc_elements(self, pdk, tmp_path):
        """Switching from rc to c_only should remove rc-specific elements."""
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        inject_parasitics(amp, est, mode='rc')
        inject_parasitics(amp, est, mode='c_only')
        # No midpoint nets should remain
        par_nets = [k for k in amp.net_dict if k.startswith('__par_')]
        assert len(par_nets) == 0
        # No rc resistors remain
        par_rs = [k for k in amp.passives_dict if k.startswith('__par_R_')]
        assert len(par_rs) == 0

    def test_invalid_mode_raises(self, pdk, tmp_path):
        from silk.layout.parasitics import inject_parasitics
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        with pytest.raises(ValueError, match='mode'):
            inject_parasitics(amp, est, mode='invalid')


@pytest.mark.parametrize('load_count', [1, 2], ids=['two-pins', 'three-pins'])
@pytest.mark.parametrize('mode', [None, 'rc', 'c_only'], ids=['default', 'rc', 'c-only'])
def test_injection_preserves_original_terminals(pdk, tmp_path, load_count, mode):
    """Pin the loading contract through public bindings and emitted SPICE.

    The PDK fixture loads only the packaged catalog and a stub library path;
    ideal resistors and netlist emission need no simulator or real models.
    """
    from silk.circuit.modules import Module
    from silk.elements import elements
    from silk.layout import LayoutEstimate, NetLayout, inject_parasitics

    ckt = Module('PDK_skywater130', str(tmp_path), 'loading.cir',
                 pdk_details=pdk, module_name='loading')
    ckt.define_module_ports({'vin': 'input', 'vss': 'inout'})
    connections = {'driver': ('vin', 'out')}
    connections.update({f'load_{i}': ('out', 'vss') for i in range(load_count)})
    for name, (pos, neg) in connections.items():
        ckt.add(elements.resistor(name, p=pos, n=neg, value=1e3))

    original_nets = set(ckt.net_dict)
    original_bindings = {
        (net.name, pin.instance, pin.port, pin.net)
        for net in ckt.net_dict.values() for pin in net.pins
    }
    assert len(ckt.net_dict['out'].pins) == load_count + 1
    original_lines = {
        line.split()[0]: line for line in ckt.generate_netlist().splitlines()
        if line.startswith('R')
    }
    assert set(original_lines) == {f'R{name}' for name in connections}

    net_estimate = NetLayout(length=12.0, width=0.5, r=37.0, c=24.0,
                             equivalence_class=None)
    estimate = LayoutEstimate(
        computation_mode='statistical', routing_model_id='supplied',
        timestamp=0.0, nets={'out': net_estimate}, devices={},
        bounding_box=(10.0, 10.0), equivalence_classes={}, warnings=(),
    )
    if mode is None:
        inject_parasitics(ckt, estimate)
    else:
        inject_parasitics(ckt, estimate, mode=mode)

    assert {
        name: (ckt.elem_dict[name]['device'].nets['p'], ckt.elem_dict[name]['device'].nets['n'])
        for name in connections
    } == connections
    assert {
        (net.name, pin.instance, pin.port, pin.net)
        for net in ckt.net_dict.values() for pin in net.pins
        if pin.instance in connections
    } == original_bindings

    lines = ckt.generate_netlist().splitlines()
    emitted = {line.split()[0]: line for line in lines if line.strip()}
    assert {name: emitted[name] for name in original_lines} == original_lines
    injected = {
        name: passive for name, passive in ckt.passives_dict.items()
        if name not in connections
    }
    total_c = sum(p.value for p in injected.values()
                  if isinstance(p, elements.capacitor))
    # Zero absolute tolerance matters at fF scale: zero must never pass.
    assert total_c == pytest.approx(24e-15, rel=1e-12, abs=0)

    mid = '__par_out_mid'
    if mode == 'c_only':
        assert set(injected) == {'__par_C_out'}
        assert set(ckt.net_dict) == original_nets | {'0'}
        assert not any(isinstance(p, elements.resistor) for p in injected.values())
        assert not any(mid in line.split() for line in lines)
        assert emitted['C__par_C_out'].split()[1:3] == ['out', '0']
        cap = injected['__par_C_out']
        assert (cap.nets['p'], cap.nets['n']) == ('out', '0')
    else:
        assert set(injected) == {'__par_R_out', '__par_C1_out', '__par_C2_out'}
        assert set(ckt.net_dict) == original_nets | {'0', mid}
        assert {
            (pin.instance, pin.port, pin.net) for pin in ckt.net_dict[mid].pins
        } == {('__par_R_out', 'neg', mid), ('__par_C2_out', 'pos', mid)}
        assert {
            line.split()[0] for line in lines if mid in line.split()
        } == {'R__par_R_out', 'C__par_C2_out'}
        resistor = injected['__par_R_out']
        assert isinstance(resistor, elements.resistor)
        assert (resistor.nets['p'], resistor.nets['n']) == ('out', mid)
        assert resistor.value == pytest.approx(37.0)
        assert emitted['R__par_R_out'].split()[1:3] == ['out', mid]
        for name, pos in [('__par_C1_out', 'out'), ('__par_C2_out', mid)]:
            cap = injected[name]
            assert (cap.nets['p'], cap.nets['n']) == (pos, '0')
            assert cap.value == pytest.approx(12e-15, rel=1e-12, abs=0)
            assert emitted[f'C{name}'].split()[1:3] == [pos, '0']



def _estimate_for(*nets):
    from silk.layout import LayoutEstimate, NetLayout
    return LayoutEstimate(
        computation_mode='statistical', routing_model_id='test', timestamp=0,
        nets={n: NetLayout(length=1, width=1, r=1, c=1e6,
                           equivalence_class=None) for n in nets},
        devices={}, bounding_box=(1, 1), equivalence_classes={}, warnings=(),
    )


def _snapshot(module):
    return (dict(module.passives_dict), dict(module.net_dict),
            {n: list(net.pins) for n, net in module.net_dict.items()},
            dict(module._element_mgr._pin_to_net))


@pytest.mark.parametrize('mode', ['rc', 'c_only'])
@pytest.mark.parametrize('return_net', ['0', 'vss', 'vin'])
def test_return_net_and_self_branch(pdk, tmp_path, mode, return_net):
    from silk.layout import inject_parasitics
    from silk.elements import elements
    amp = _make_amp(pdk, tmp_path)
    amp.add_net('0')
    estimate = _estimate_for('vout', return_net)
    inject_parasitics(amp, estimate, mode, return_net=return_net)
    caps = [p for p in amp.passives_dict.values() if isinstance(p, elements.capacitor)]
    assert len(caps) == (2 if mode == 'rc' else 1)
    assert all(p.nets['n'] == return_net and p.nets['p'] != return_net for p in caps)
    assert f'__par_{return_net}_mid' not in amp.net_dict
    assert f'__par_R_{return_net}' not in amp.passives_dict
    assert 'gnd' not in amp.net_dict


@pytest.mark.parametrize('bad', ['mode', 'return', 'estimate'])
def test_invalid_injection_preserves_everything(pdk, tmp_path, bad):
    from silk.layout import inject_parasitics
    from silk.exceptions import UndeclaredNetError
    amp = _make_amp(pdk, tmp_path)
    estimate = _estimate_for('vout')
    inject_parasitics(amp, estimate)
    before = _snapshot(amp)
    kwargs = {'mode': 'invalid'} if bad == 'mode' else {}
    if bad == 'return':
        kwargs['return_net'] = 'missing'
    if bad == 'estimate':
        estimate = _estimate_for('vout', 'missing')
    with pytest.raises(ValueError if bad == 'mode' else UndeclaredNetError):
        inject_parasitics(amp, estimate, **kwargs)
    assert _snapshot(amp) == before


def test_return_is_keyword_only(pdk, tmp_path):
    from silk.layout import inject_parasitics
    amp = _make_amp(pdk, tmp_path)
    with pytest.raises(TypeError):
        inject_parasitics(amp, _estimate_for('vout'), 'rc', 'vss')


@pytest.mark.parametrize('first_mode', ['rc', 'c_only'])
@pytest.mark.parametrize('second_mode', ['rc', 'c_only'])
def test_reinjection_matches_fresh_including_pins(pdk, tmp_path, first_mode, second_mode):
    from silk.layout import inject_parasitics
    amp = _make_amp(pdk, tmp_path / 'repeat')
    fresh = _make_amp(pdk, tmp_path / 'fresh')
    estimate = _estimate_for('vout', 'vss')
    inject_parasitics(amp, estimate, first_mode)
    assert '0' in amp.net_dict
    inject_parasitics(amp, estimate, second_mode, return_net='vss')
    inject_parasitics(fresh, estimate, second_mode, return_net='vss')
    assert '0' not in amp.net_dict
    assert amp.generate_netlist() == fresh.generate_netlist()
    assert set(amp.net_dict) == set(fresh.net_dict)
    assert {n: net.pins for n, net in amp.net_dict.items()} == {
        n: net.pins for n, net in fresh.net_dict.items()}
    assert amp._element_mgr._pin_to_net == fresh._element_mgr._pin_to_net


@pytest.mark.parametrize('mode', ['rc', 'c_only'])
def test_unrecorded_objects_survive(pdk, tmp_path, mode):
    from silk.layout import inject_parasitics
    from silk.elements.passives import Capacitor
    amp = _make_amp(pdk, tmp_path)
    for net in ('gnd', '0', '__par_user_mid'):
        amp.add_net(net)
    user_cap = Capacitor('__par_user', pos='vin', neg='gnd', capacitance=1e-12)
    amp.add_passive(user_cap)
    estimate = _estimate_for('vout')
    inject_parasitics(amp, estimate, mode)
    inject_parasitics(amp, estimate, mode, return_net='vss')
    assert {'gnd', '0', '__par_user_mid'} <= set(amp.net_dict)
    assert amp.passives_dict['__par_user'] is user_cap


@pytest.mark.parametrize('mode', ['rc', 'c_only'])
def test_default_return_works_with_strict_nets(pdk, tmp_path, mode):
    from silk.layout import inject_parasitics
    amp = _make_amp(pdk, tmp_path)
    amp.strict_nets = True
    inject_parasitics(amp, _estimate_for('vout'), mode)
    assert '0' in amp.net_dict


@pytest.mark.parametrize('mode', ['rc', 'c_only'])
@pytest.mark.parametrize('when', [
    'before',
    'instance',
])
def test_return_port_follows_parent_binding(pdk, tmp_path, mode, when):
    from silk.circuit.modules import Module
    from silk.elements import elements
    from silk.layout import inject_parasitics
    child = Module('sky130', str(tmp_path), 'child.cir',
                   pdk_details=pdk, module_name='blk')
    child.define_module_ports({'inp': 'input', 'out': 'output', 'vss': 'inout'})
    child.add(elements.resistor('r', p='inp', n='out', value=1000))
    parent = Module('sky130', str(tmp_path), 'parent.cir',
                    pdk_details=pdk, module_name='parent')
    parent.define_module_ports({})
    estimate = _estimate_for('out', 'vss')
    if when == 'before':
        inject_parasitics(child, estimate, mode, return_net='vss')
    parent.add_element(child, 'dut', inp='vin', out='vout', vss='vneg')
    if when == 'instance':
        placed = next(parent.iter_instances()).module
        assert placed is child
        inject_parasitics(placed, estimate, mode, return_net='vss')
        assert child.passives_dict
    deck = parent.generate_netlist()
    assert 'xdut vin vout vneg blk' in deck
    caps = [line.split() for line in deck.splitlines() if line.startswith('C__par_')]
    assert len(caps) == (2 if mode == 'rc' else 1)
    assert all(tokens[2] == 'vss' for tokens in caps)


@pytest.mark.parametrize('mode', ['rc', 'c_only'])
@pytest.mark.parametrize('strict', [False, True])
def test_injection_inside_module_context(pdk, tmp_path, mode, strict):
    from silk.layout import inject_parasitics
    amp = _make_amp(pdk, tmp_path)
    amp.strict_nets = strict
    estimate = _estimate_for('vout')
    with amp:
        inject_parasitics(amp, estimate, mode)
        inject_parasitics(amp, estimate, mode)
    inject_parasitics(amp, estimate, 'c_only', return_net='vss')
    assert '0' not in amp.net_dict
    assert '__par_vout_mid' not in amp.net_dict
    assert set(amp.passives_dict) == {'__par_C_vout'}


@pytest.mark.parametrize('mode', ['rc', 'c_only'])
def test_injection_does_not_touch_active_outer_module(pdk, tmp_path, mode):
    from silk.layout import inject_parasitics
    from silk.circuit._module_base import _active_module
    outer = _make_amp(pdk, tmp_path / 'outer')
    target = _make_amp(pdk, tmp_path / 'target')
    before = _snapshot(outer)
    with outer:
        with outer:
            inject_parasitics(target, _estimate_for('vout'), mode)
            assert _active_module() is outer
    assert _snapshot(outer) == before
    assert target.passives_dict


@pytest.mark.parametrize('collision', ['case', 'cross_registry'])
def test_name_preflight_preserves_previous_injection(pdk, tmp_path, collision):
    from silk.elements import elements
    from silk.elements.passives import Capacitor
    from silk.exceptions import DuplicateNameError
    from silk.layout.parasitics import inject_parasitics
    amp = _make_amp(pdk, tmp_path)
    estimate = _run_estimate(amp)
    inject_parasitics(amp, estimate, mode='c_only')
    # RC's C1 is new relative to the old C-only injection.
    name = '__par_C1_vout'
    if collision == 'case':
        amp.add_passive(Capacitor(name.upper(), 'vout', '0', 1e-12))
    else:
        amp.add(elements.capacitor(name, p='vout', n='0', value=1e-12))
    before = dict(amp.passives_dict)
    ownership = amp._parasitic_ownership
    pins = {n: tuple(net.pins) for n, net in amp.net_dict.items()}
    reverse = dict(amp._element_mgr._pin_to_net)
    with pytest.raises(DuplicateNameError):
        inject_parasitics(amp, estimate, mode='rc')
    assert amp.passives_dict == before
    assert amp._parasitic_ownership == ownership
    assert amp._element_mgr._pin_to_net == reverse
    assert {n: tuple(net.pins) for n, net in amp.net_dict.items()} == pins
    assert all(amp.passives_dict[n] is p for n, p in before.items())


def test_reinjection_replaces_index_entries(pdk, tmp_path):
    from silk.layout.parasitics import inject_parasitics
    amp = _make_amp(pdk, tmp_path)
    estimate = _run_estimate(amp)
    inject_parasitics(amp, estimate, mode='rc')
    old = dict(amp.passives_dict)
    inject_parasitics(amp, estimate, mode='c_only')
    assert not any(key.startswith('r__par_') for key in amp._element_mgr._instance_names)
    for registry, slot, obj in amp._element_mgr._instance_names.values():
        if registry == 'passives_dict':
            assert amp.passives_dict[slot] is obj
            assert obj is not old.get(slot)


def test_batch_preflight_is_atomic(pdk, tmp_path):
    from types import SimpleNamespace
    from silk.exceptions import DuplicateNameError
    from silk.layout.parasitics import inject_parasitics
    amp = _make_amp(pdk, tmp_path)
    estimate = _run_estimate(amp)
    inject_parasitics(amp, estimate, mode='rc')
    before = dict(amp.passives_dict)
    index = dict(amp._element_mgr._instance_names)
    pins = {n: tuple(net.pins) for n, net in amp.net_dict.items()}
    ownership = amp._parasitic_ownership
    nl = estimate.nets['vout']
    duplicate = SimpleNamespace(nets={'vout': nl, 'VOUT': nl})
    with pytest.raises(DuplicateNameError):
        inject_parasitics(amp, duplicate, mode='c_only')
    assert amp.passives_dict == before
    assert amp._element_mgr._instance_names == index
    assert amp._parasitic_ownership == ownership
    assert {n: tuple(net.pins) for n, net in amp.net_dict.items()} == pins


def _indexed_module(tmp_path):
    from silk import Module
    from silk.elements import elements
    module = Module(None, str(tmp_path), 'indexed.cir', None, module_name='indexed')
    module.define_module_ports(dict.fromkeys(
        ('data', 'data[3]', 'data[03]', 'data_3', 'data_mid[3]')))
    for i, name in enumerate(module.pin_list):
        module.add(elements.resistor(f'load_{i}', p=name, n='0', value=1000))
    return module


def test_indexed_rc_injection_reinjection_and_cleanup(tmp_path):
    from silk.layout import inject_parasitics
    module = _indexed_module(tmp_path / 'repeat')
    fresh = _indexed_module(tmp_path / 'fresh')
    original = set(module.net_dict)
    estimate = _estimate_for(*module.pin_list)
    midpoints = {'__par_data_mid', '__par_data_mid[3]', '__par_data_mid[03]',
                 '__par_data_3_mid', '__par_data_mid_mid[3]'}
    inject_parasitics(module, estimate, mode='rc')
    assert set(module.net_dict) == original | midpoints
    assert module.passives_dict['__par_R_data[3]'].nets['n'] == '__par_data_mid[3]'
    assert module.passives_dict['__par_C2_data[3]'].nets['p'] == '__par_data_mid[3]'
    assert 'R__par_R_data<3> data[3] __par_data_mid[3] 1' in module.generate_netlist()
    old = dict(module.passives_dict)
    inject_parasitics(module, estimate, mode='rc')
    inject_parasitics(fresh, estimate, mode='rc')
    assert module.generate_netlist() == fresh.generate_netlist()
    assert {n: net.pins for n, net in module.net_dict.items()} == {
        n: net.pins for n, net in fresh.net_dict.items()}
    assert all(module.passives_dict[n] is not obj for n, obj in old.items())
    inject_parasitics(module, estimate, mode='c_only')
    assert set(module.net_dict) == original
    assert not any(net in midpoints for net in module._element_mgr._pin_to_net.values())
    inject_parasitics(module, estimate, mode='rc')
    assert module.generate_netlist() == fresh.generate_netlist()


@pytest.mark.parametrize('collision', ['midpoint', 'midpoint_case', 'passive'])
def test_indexed_rc_collision_preserves_previous_injection(tmp_path, collision):
    from silk.layout import inject_parasitics
    from silk.elements import elements
    from silk.exceptions import DuplicateNameError
    module = _indexed_module(tmp_path)
    estimate = _estimate_for('data', 'data[3]')
    inject_parasitics(module, estimate, mode='c_only')
    if collision == 'passive':
        module.add(elements.capacitor('__par_C2_data[3]', p='data[3]', n='0', value=1e-12))
    else:
        name = '__par_data_mid[3]'
        module.add_net(name.upper() if collision == 'midpoint_case' else name)
    before = _snapshot(module)
    index = dict(module._element_mgr._instance_names)
    ownership = module._parasitic_ownership
    with pytest.raises(DuplicateNameError):
        inject_parasitics(module, estimate, mode='rc')
    assert _snapshot(module) == before
    assert module._element_mgr._instance_names == index
    assert module._parasitic_ownership == ownership
    assert all(module.passives_dict[n] is obj for n, obj in before[0].items())


def test_indexed_rc_invalid_later_branch_is_atomic(tmp_path):
    from types import SimpleNamespace
    from silk.layout import inject_parasitics
    module = _indexed_module(tmp_path)
    estimate = _estimate_for('data[3]')
    inject_parasitics(module, estimate, mode='rc')
    before = _snapshot(module)
    index = dict(module._element_mgr._instance_names)
    ownership = module._parasitic_ownership
    bad = SimpleNamespace(nets={'data': estimate.nets['data[3]'],
                               'data[3]': SimpleNamespace(r=-1, c_farads=1e-12)})
    with pytest.raises(ValueError, match='resistance'):
        inject_parasitics(module, bad, mode='rc')
    assert _snapshot(module) == before
    assert module._element_mgr._instance_names == index
    assert module._parasitic_ownership == ownership


@pytest.mark.parametrize('mode', ['rc', 'c_only'])
def test_injection_and_reinjection_emit_no_deprecation_warning(pdk, tmp_path, mode):
    import warnings
    from silk.layout import inject_parasitics

    amp = _make_amp(pdk, tmp_path)
    estimate = _run_estimate(amp)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        inject_parasitics(amp, estimate, mode)
        deck = amp.generate_netlist()
        inject_parasitics(amp, estimate, mode)
        assert amp.generate_netlist() == deck

    # Compare the complete deck with the former constructors, including order
    # and formatting, while suppressing only this explicit compatibility oracle.
    from silk.elements import elements
    from silk.elements.passives import Capacitor, Resistor
    legacy = _make_amp(pdk, tmp_path / 'legacy')
    for name in amp.net_dict:
        if name not in legacy.net_dict:
            legacy.add_net(name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        for passive in amp.passives_dict.values():
            constructor = Capacitor if isinstance(passive, elements.capacitor) else Resistor
            legacy.add_passive(constructor(passive.name, passive.nets['p'],
                                           passive.nets['n'], passive.value))
    assert legacy.generate_netlist() == deck
