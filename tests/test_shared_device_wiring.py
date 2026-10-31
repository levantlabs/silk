"""Shared object edits keep each direct holder's connection graph current."""
import copy
import gc
import pickle
import warnings
import weakref

import pytest
from silk import sources
from silk.circuit.modules import Module
from silk.elements import elements
from silk.exceptions import UndeclaredNetError


def module(pdk, tmp_path, name):
    return Module('PDK_skywater130', str(tmp_path / name), name + '.cir', pdk_details=pdk)


def state(m):
    return ({n: [(p.instance, p.port, p.net, p.kind) for p in net.pins]
             for n, net in m.net_dict.items()}, dict(m._element_mgr._pin_to_net))


@pytest.mark.parametrize('passive', [False, True])
def test_each_holder_follows_pin(pdk, tmp_path, passive):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1000)
    a.add(r)
    (b.add_passive if passive else b.add)(r)
    with pytest.warns(UserWarning, match='r.*p.*x.*y.*2'):
        r.p = 'y'
    for m, port, kind in ((a, 'p', 'device'), (b, 'pos' if passive else 'p', 'passive' if passive else 'device')):
        assert m._element_mgr._pin_to_net['r', port] == 'y'
        assert [(p.port, p.kind) for p in m.net_dict['y'].pins] == [(port, kind)]
        assert not m.net_dict['x'].pins


def test_refusal_before_warning_or_implicit_net(pdk, tmp_path):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1000)
    a.add(r); b.add(r)
    b._element_mgr.strict_nets = True
    before = state(a), state(b), dict(r.nets)
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(UndeclaredNetError):
            r.p = 'new'
    assert not caught
    assert (state(a), state(b), r.nets) == before


def test_source_batch_checks_all_holders_before_any_write(pdk, tmp_path):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    s = sources.VoltageSource('v', positive='x', negative='0', dc=1)
    a.add_source(s); b.add_source(s)
    b.add_net('y'); b._element_mgr.strict_nets = True
    before = state(a), state(b), dict(vars(s))
    with pytest.raises(UndeclaredNetError):
        a.update_source('v', ['positive', 'dc', 'negative'], ['y', 2, 'unknown'])
    assert (state(a), state(b), vars(s)) == before


@pytest.mark.parametrize('clone', [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_copy_one_holder_has_complete_isolated_device(pdk, tmp_path, clone):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1000)
    a.add(r); b.add(r)
    result = clone(a)
    copied = result.elem_dict['r']['device']
    assert copied.name == 'r'
    assert copied.nets == {'p': 'x', 'n': '0'}
    with warnings.catch_warnings(record=True) as caught:
        copied.p = 'y'
    assert not caught
    assert result._element_mgr._pin_to_net['r', 'p'] == 'y'
    assert a._element_mgr._pin_to_net['r', 'p'] == 'x'
    assert b._element_mgr._pin_to_net['r', 'p'] == 'x'
    with pytest.raises(Exception, match='read-only'):
        copied.name = 'other'


def test_warning_error_changes_nothing(pdk, tmp_path):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r); b.add(r)
    before = state(a), state(b), dict(r.nets)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(UserWarning):
            r.p = 'y'
    assert (state(a), state(b), r.nets) == before

@pytest.mark.parametrize('clone', [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_source_copy_isolation_and_selected_context(pdk, tmp_path, clone):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    a.add_source(sources.VoltageSource('sense', positive='x', negative='0', dc=1))
    b.add_source(sources.VoltageSource('sense', positive='x', negative='0', dc=1,
                                     sine=sources.Sine(voffset=0, vampl=1, freq=1)))
    s = sources.CurrentControlledCurrentSource('f', out_pos='x', out_neg='0', ctrl_vname='sense', gain=2)
    a.add_source(s); b.add_source(s)
    expected = s.to_spice()
    # Reverse original registration order: context must remain b, not a.
    bc, ac = clone([b, a])
    sc = ac.sig_sources_dict['f']
    assert sc is bc.sig_sources_dict['f']
    assert sc._module is bc
    assert sc.to_spice() == expected
    with pytest.warns(UserWarning):
        sc.out_pos = 'z'
    assert ac._element_mgr._pin_to_net['f', 'out_pos'] == 'z'
    assert bc._element_mgr._pin_to_net['f', 'out_pos'] == 'z'
    assert a._element_mgr._pin_to_net['f', 'out_pos'] == 'x'
    single = clone(a)
    assert single.sig_sources_dict['f']._module is single


def test_device_does_not_keep_discarded_holder_alive(pdk, tmp_path):
    a = module(pdk, tmp_path, 'a')
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r)
    ref = weakref.ref(a)
    del a
    gc.collect()
    assert ref() is None
    r.p = 'z'


def test_all_source_fields_checked_before_writes(pdk, tmp_path):
    a = module(pdk, tmp_path, 'a')
    s = sources.VoltageSource('v', positive='x', negative='0', dc=1)
    a.add_source(s)
    before = state(a), s.positive, s.dc
    with pytest.raises(ValueError):
        a.update_source('v', ['positive', 'dc'], ['y', '1\ninvalid'])
    assert (state(a), s.positive, s.dc) == before


def test_shallow_source_edit_does_not_touch_graph(pdk, tmp_path):
    a = module(pdk, tmp_path, 'a')
    s = sources.VoltageSource('v', positive='x', negative='0', dc=1)
    a.add_source(s)
    detached = copy.copy(s)
    detached.positive = 'y'
    assert a._element_mgr._pin_to_net['v', 'positive'] == 'x'


def test_emission_view_never_registers_original_devices(pdk, tmp_path):
    from silk.db._deck_check import _emission_view
    a = module(pdk, tmp_path, 'a')
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r)
    s = sources.VoltageSource('v', positive='x', negative='0', dc=1)
    a.add_source(s)
    before = state(a)
    view = _emission_view(a)
    view.sig_sources_dict['v'].positive = 'y'
    assert state(a) == before
    assert s._module is a
    with warnings.catch_warnings(record=True) as caught:
        r.p = 'z'
    assert not caught
    assert view.sig_sources_dict['v']._module is view


def test_fault_derivative_has_no_holders(pdk, tmp_path):
    from types import SimpleNamespace
    from silk.circuit.spice_netlister import _derive_faulted_instance
    a = module(pdk, tmp_path, 'a')
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r)
    before = state(a)
    derived = _derive_faulted_instance(r, SimpleNamespace(name_suffix='__bad', nets={}, params={}))
    derived.p = 'z'
    assert state(a) == before
    assert not vars(derived).get('_wiring_holders')


_SOURCE_CASES = [
    (sources.VoltageSource, dict(positive='x', negative='0', dc=1, sine=sources.Sine(vampl=1, freq=1)), 'positive'),
    (sources.CurrentSource, dict(positive='x', negative='0', dc=1, sine=sources.Sine(vampl=1, freq=1)), 'positive'),
    (sources.VoltageControlledVoltageSource, dict(out_pos='x', out_neg='0', ctrl_pos='c', ctrl_neg='0', gain=2), 'ctrl_pos'),
    (sources.VoltageControlledCurrentSource, dict(out_pos='x', out_neg='0', ctrl_pos='c', ctrl_neg='0', gm=2), 'ctrl_neg'),
    (sources.CurrentControlledCurrentSource, dict(out_pos='x', out_neg='0', ctrl_vname='sense', gain=2), 'out_pos'),
    (sources.CurrentControlledVoltageSource, dict(out_pos='x', out_neg='0', ctrl_vname='sense', transresistance=2), 'out_neg'),
    (sources.BehavioralVoltage, dict(positive='x', negative='0', expression='1'), 'positive'),
    (sources.BehavioralCurrent, dict(positive='x', negative='0', expression='1'), 'negative'),
]


@pytest.mark.parametrize('factory,kwargs,pin', _SOURCE_CASES)
@pytest.mark.parametrize('route', ['assignment', 'update', 'connect'])
def test_source_routes_follow_every_holder(pdk, tmp_path, factory, kwargs, pin, route):
    from silk.lint import RuleRegistry
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    s = factory('s', **kwargs)
    a.add_source(s); b.add_source(s)
    a.add_net('y'); b.add_net('y')
    with pytest.warns(UserWarning):
        if route == 'assignment':
            setattr(s, pin, 'y')
        elif route == 'update':
            a.update_source('s', [pin], ['y'])
        else:
            a.connect_source('s', [pin], ['y'])
    for m in (a, b):
        fresh = module(pdk, tmp_path, 'fresh')
        fresh.add_source(factory('s', **dict(kwargs, **{pin: 'y'})))
        for net in m.net_dict:
            fresh.add_net(net)
        # Net inventory can retain now-empty old nets. Bindings agree exactly.
        assert m._element_mgr._pin_to_net == fresh._element_mgr._pin_to_net
        assert [(p.port, p.kind, p.direction) for p in m.net_dict['y'].pins] == [
            (p.port, p.kind, p.direction) for p in fresh.net_dict['y'].pins]
        assert s.to_spice() == fresh.sig_sources_dict['s'].to_spice()
        assert [v.rule_id for v in RuleRegistry.default().run(m).violations] == [
            v.rule_id for v in RuleRegistry.default().run(fresh).violations]


@pytest.mark.parametrize('target', ['new', 'vdd'])
@pytest.mark.parametrize('failure_type', [RuntimeError, KeyboardInterrupt])
def test_warning_hook_and_commit_failure_restore_all(pdk, tmp_path, monkeypatch, target, failure_type):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r); b.add_passive(r)
    for m in (a, b):
        m.add_net('VDD')
    caches = [(dict(m._element_mgr._net_casefold), set(m._element_mgr._case_merge_warned))
              for m in (a, b)]
    before = state(a), state(b), dict(r.nets)
    observed = []
    failure = failure_type('commit failure')
    def show(*args, **kwargs):
        observed.append((state(a), state(b), dict(r.nets)))
    monkeypatch.setattr(warnings, 'showwarning', show)
    real = b._element_mgr._bind_pin_to_net
    def fail(*args, **kwargs):
        real(*args, **kwargs)
        raise failure
    monkeypatch.setattr(b._element_mgr, '_bind_pin_to_net', fail)
    with warnings.catch_warnings():
        warnings.simplefilter('always')
        with pytest.raises(failure_type, match='commit failure') as caught:
            r.p = target
    assert caught.value is failure
    assert observed == [before]
    assert (state(a), state(b), r.nets) == before
    assert [(m._element_mgr._net_casefold, m._element_mgr._case_merge_warned)
            for m in (a, b)] == caches


def test_source_multi_terminal_post_bind_failure_restores_all(pdk, tmp_path, monkeypatch):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    source = sources.VoltageSource('v', positive='x', negative='0', dc=1)
    a.add_source(source); b.add_source(source)
    before = state(a), state(b), dict(vars(source))
    caches = [(dict(m._element_mgr._net_casefold), set(m._element_mgr._case_merge_warned))
              for m in (a, b)]
    real = b._element_mgr._bind_pin_to_net
    failure = RuntimeError('post-bind failure')
    def fail(*args, **kwargs):
        real(*args, **kwargs)
        raise failure
    monkeypatch.setattr(b._element_mgr, '_bind_pin_to_net', fail)
    with pytest.warns(UserWarning, match='all 2 modules') as warned:
        with pytest.raises(RuntimeError, match='post-bind failure') as caught:
            a.update_source('v', ['positive', 'negative', 'dc'], ['new_p', 'new_n', 2])
    assert len(warned) == 2
    assert caught.value is failure
    assert (state(a), state(b), vars(source)) == before
    assert [(m._element_mgr._net_casefold, m._element_mgr._case_merge_warned)
            for m in (a, b)] == caches


def test_retired_passive_follows_only_remaining_holder(pdk, tmp_path):
    from types import SimpleNamespace
    from silk.layout.parasitics import inject_parasitics
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    a.add_net('x')
    estimate = SimpleNamespace(nets={'x': SimpleNamespace(c=2, c_farads=2e-15, r=1)})
    inject_parasitics(a, estimate, mode='c_only')
    old = a.passives_dict['__par_C_x']
    b.add_passive(old)
    inject_parasitics(a, estimate, mode='c_only')
    before = state(a)
    with warnings.catch_warnings(record=True) as caught:
        old.p = 'y'
    assert not caught
    assert state(a) == before
    assert b._element_mgr._pin_to_net[old.name, 'pos'] == 'y'


@pytest.mark.parametrize('factory', [sources.CurrentControlledCurrentSource, sources.CurrentControlledVoltageSource])
def test_live_control_context_retains_last_module(pdk, tmp_path, factory):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    a.add_source(sources.VoltageSource('sense', 'x', '0', dc=1))
    b.add_source(sources.VoltageSource('sense', 'x', '0', dc=1, sine=sources.Sine(vampl=1, freq=1)))
    value = {'gain': 2} if factory is sources.CurrentControlledCurrentSource else {'transresistance': 2}
    s = factory('s', out_pos='x', out_neg='0', ctrl_vname='sense', **value)
    a.add_source(s); b.add_source(s)
    before = s.to_spice()
    ref = weakref.ref(b)
    del b
    gc.collect()
    assert ref() is s._module
    assert s.to_spice() == before


@pytest.mark.parametrize('clone', [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_standalone_copies_are_detached_but_locked(pdk, tmp_path, clone):
    a = module(pdk, tmp_path, 'a')
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r)
    result = clone(r)
    before = state(a)
    result.p = 'z'
    assert state(a) == before
    assert not vars(result).get('_wiring_holders')
    with pytest.raises(Exception, match='read-only'):
        result.name = 'other'


def test_equal_named_objects_and_repeated_definition_do_not_warn(pdk, tmp_path):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1)
    other = elements.resistor('r', p='x', n='0', value=1)
    a.add(r); b.add(other)
    a.module_name = 'child'
    a.define_module_ports({})
    top = module(pdk, tmp_path, 'top')
    top.add_module(a, 'left'); top.add_module(a, 'right')
    with warnings.catch_warnings(record=True) as caught:
        r.p = 'y'
        r.p = 'y'
        r.value = 2
    assert not caught
    assert other.nets['p'] == 'x'
    assert top.elem_dict['left']['device'] is top.elem_dict['right']['device'] is a
    a.generate_netlist(subckt=True)


def test_shared_noop_and_values_are_silent(pdk, tmp_path):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r); b.add(r)
    before = state(a), state(b)
    with warnings.catch_warnings(record=True) as caught:
        r.p = 'x'
        r.value = 2
    assert not caught
    assert (state(a), state(b)) == before


def test_every_catalog_factory_pin_updates_both_holders(pdk, tmp_path):
    for type_name in pdk._devices:
        factory = getattr(pdk, type_name)
        a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
        pins = {p: 'x' for p in factory._pin_names}
        d = factory('dut', **pins)
        a.add(d); b.add(d)
        for pin in factory._pin_names:
            with pytest.warns(UserWarning):
                setattr(d, pin, 'y')
            assert a._element_mgr._pin_to_net['dut', pin] == 'y'
            assert b._element_mgr._pin_to_net['dut', pin] == 'y'


def test_view_preserves_selected_control_context_with_two_holders(pdk, tmp_path):
    from silk.db._deck_check import _emission_view
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    for m in (a, b):
        m.module_name = 'child_' + ('a' if m is a else 'b')
        m.define_module_ports({})
    a.add_source(sources.VoltageSource('sense', 'x', '0', dc=1))
    b.add_source(sources.VoltageSource('sense', 'x', '0', dc=1, sine=sources.Sine(vampl=1, freq=1)))
    s = sources.CurrentControlledCurrentSource('f', 'x', '0', 2, 'sense')
    a.add_source(s); b.add_source(s)
    top = module(pdk, tmp_path, 'top')
    top.add_module(b, 'b'); top.add_module(a, 'a')
    before = s.to_spice(), state(a), state(b), set(vars(s)['_wiring_holders'])
    from silk.circuit.test_execution import normalized_deck_text
    authored_deck, reason = normalized_deck_text(top, 'tt')
    assert reason is None
    view = _emission_view(top)
    viewed_deck, reason = normalized_deck_text(view, 'tt')
    assert reason is None
    assert viewed_deck == authored_deck
    assert normalized_deck_text(top, 'tt') == (authored_deck, None)
    ac = view.elem_dict['a']['device']; bc = view.elem_dict['b']['device']
    sc = ac.sig_sources_dict['f']
    assert sc is bc.sig_sources_dict['f']
    assert sc._module is bc
    assert sc.to_spice() == before[0]
    assert (s.to_spice(), state(a), state(b), set(vars(s)['_wiring_holders'])) == before


@pytest.mark.parametrize('shim,value_name', [('Resistor', 'resistance'), ('Capacitor', 'capacitance'), ('Inductor', 'inductance')])
def test_passive_alias_uses_each_holders_binding_vocabulary(pdk, tmp_path, shim, value_name):
    from silk.elements import passives
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = getattr(passives, shim)('r', pos='x', neg='0', **{value_name: 1})
    a.add(r); b.add_passive(r)
    with pytest.warns(UserWarning):
        r.pos = 'y'
    assert a._element_mgr._pin_to_net['r', 'p'] == 'y'
    assert b._element_mgr._pin_to_net['r', 'pos'] == 'y'
    assert a.net_dict['y'].pins[0].kind == 'device'
    assert b.net_dict['y'].pins[0].kind == 'passive'
    b._element_mgr.strict_nets = True
    before = state(a), state(b), dict(r.nets)
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises(UndeclaredNetError):
            r.neg = 'refused'
    assert not caught
    assert (state(a), state(b), r.nets) == before


def test_two_holders_decks_and_lint_equal_fresh_builds(pdk, tmp_path):
    from silk.lint import RuleRegistry
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r); b.add(r)
    with pytest.warns(UserWarning):
        r.p = 'y'
    for m in (a, b):
        fresh = module(pdk, tmp_path, 'fresh')
        fresh.add(elements.resistor('r', p='y', n='0', value=1))
        fresh.add_net('x')
        for bench in (m, fresh):
            bench.module_name = 'dut'
            bench.define_module_ports({})
        assert m.generate_netlist(subckt=True) == fresh.generate_netlist(subckt=True)
        assert m._element_mgr._pin_to_net == fresh._element_mgr._pin_to_net
        assert [v.rule_id for v in RuleRegistry.default().run(m).violations] == [
            v.rule_id for v in RuleRegistry.default().run(fresh).violations]


@pytest.mark.parametrize('factory', [sources.CurrentControlledCurrentSource, sources.CurrentControlledVoltageSource])
def test_replay_detached_copy_retains_control_tokens(pdk, tmp_path, factory):
    a = module(pdk, tmp_path, 'a')
    a.add_source(sources.VoltageSource('sense', 'x', '0', dc=1, sine=sources.Sine(vampl=1, freq=1)))
    value = {'gain': 2} if factory is sources.CurrentControlledCurrentSource else {'transresistance': 2}
    s = factory('s', out_pos='x', out_neg='0', ctrl_vname='sense', **value)
    a.add_source(s)
    expected = s.to_spice()
    effective = copy.copy(s)
    assert effective.to_spice() == expected
    assert '__silk_wave' in expected
    assert not vars(effective).get('_wiring_holders')


def test_source_batch_warns_once_per_changed_terminal_before_values(pdk, tmp_path):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    s = sources.VoltageSource('v', 'x', '0', dc=1)
    a.add_source(s); b.add_source(s)
    before = state(a), state(b), s.positive, s.negative, s.dc
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        with pytest.raises(UserWarning):
            a.update_source('v', ['positive', 'negative', 'dc'], ['y', 'z', 2])
    assert (state(a), state(b), s.positive, s.negative, s.dc) == before
    with pytest.warns(UserWarning) as caught:
        a.update_source('v', ['positive', 'negative', 'dc'], ['y', 'z', 2])
    assert len(caught) == 2
    assert s.dc == 2
    for m in (a, b):
        assert m._element_mgr._pin_to_net['v', 'positive'] == 'y'
        assert m._element_mgr._pin_to_net['v', 'negative'] == 'z'
    with warnings.catch_warnings(record=True) as caught:
        a.update_source('v', ['dc'], [3])
    assert not caught
    assert s.dc == 3


@pytest.mark.parametrize('clone', [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))])
def test_copied_root_keeps_shared_factory_identity_and_exact_holders(pdk, tmp_path, clone):
    a, b = [module(pdk, tmp_path, n) for n in ('a', 'b')]
    r = elements.resistor('r', p='x', n='0', value=1)
    a.add(r); b.add_passive(r)
    ac, bc = clone([a, b])
    rc = ac.elem_dict['r']['device']
    assert rc is bc.passives_dict['r']
    assert set(vars(rc)['_wiring_holders']) == {ac, bc}
    with pytest.warns(UserWarning) as caught:
        rc.p = 'y'
    assert len(caught) == 1
    assert ac._element_mgr._pin_to_net['r', 'p'] == 'y'
    assert bc._element_mgr._pin_to_net['r', 'pos'] == 'y'
    assert a._element_mgr._pin_to_net['r', 'p'] == 'x'
    assert b._element_mgr._pin_to_net['r', 'pos'] == 'x'


def test_connect_source_net_argument_uses_each_holders_local_name(pdk, tmp_path):
    a, b, foreign = [module(pdk, tmp_path, n) for n in ('a', 'b', 'foreign')]
    s = sources.VoltageSource('s', 'x', '0', dc=1)
    a.add_source(s); b.add_source(s)
    for m in (a, b, foreign):
        m.add_net('y')
    with pytest.warns(UserWarning):
        a.connect_source('s', ['positive'], [foreign.net_dict['y']])
    assert s.positive == 'y'
    assert a._element_mgr._pin_to_net['s', 'positive'] == 'y'
    assert b._element_mgr._pin_to_net['s', 'positive'] == 'y'
    assert a.net_dict['y'] is not b.net_dict['y']
    assert not foreign.net_dict['y'].pins
