"""Device reads retain their placement without granting shared-definition edits."""
import copy
import pickle
from collections.abc import Mapping

import pytest

from silk import Module, load_pdk
from silk.circuit import element_names
from silk.circuit.emitted_names import emitted_instance_name
from silk.circuit.variable import Variable
from silk.elements import elements
from silk.exceptions import CircuitError
from silk.faults import (stuck_open, short_ds, short_gs, short_gd,
                         w_variation, l_variation, vth_shift)
from silk.layout.area_model import compute_device_area
from silk.pdk import device_geometry_um


def module(tmp_path, name, pdk=None):
    result = Module(None, str(tmp_path / name), name + '.cir', pdk,
                    module_name=name)
    result.define_module_ports({})
    return result


def design(tmp_path, *, si=False, legacy=False):
    pdk = load_pdk('sky130', user_units='si' if si else 'u')
    leaf = module(tmp_path, 'leaf', pdk)
    leaf.define_module_ports({'out': 'output', 'inp': 'input'})
    for name in ('m1', 'bank[5]', 'bank[8]'):
        kw = dict(w=4e-6 if si else 4., l=.5e-6 if si else .5, m=2)
        pins = dict(d='out', g='inp', s='0', b='0')
        if legacy:
            with pytest.warns(DeprecationWarning):
                leaf.add_element(pdk.nmos1p8, name, nets=pins, parameters=kw)
        else:
            leaf.add(pdk.nmos1p8(name, **pins, **kw))
    top = module(tmp_path, 'top', pdk)
    h = top.add_module(leaf, 'stage[8]', out='o', inp='i')
    return top, leaf, h, leaf.elem_dict['m1']['device']


def access(h, route):
    return {'attribute': lambda: h.m1, 'named': lambda: h.device('m1'),
            'indexed': lambda: h.bank[5],
            'sliced': lambda: h.bank[5:9][5],
            'iterated': lambda: next(iter(h.bank))}[route]()


@pytest.mark.parametrize('route', ['attribute', 'named', 'indexed', 'sliced', 'iterated'])
@pytest.mark.parametrize('name,value', [('d', 'new'), ('w', 2.), ('name', 'other'),
                                       ('anything', 7), ('layout', {})])
@pytest.mark.parametrize('operation', ['assign', 'delete'])
def test_public_writes_refuse_and_name_holding_module(tmp_path, route, name, value, operation):
    top, leaf, h, dev = design(tmp_path)
    before = top.generate_netlist(subckt=True)
    view = access(h, route)
    with pytest.raises(AttributeError, match='leaf.*edit|edit.*leaf'):
        if operation == 'assign':
            setattr(view, name, value)
        else:
            delattr(view, name)
    assert top.generate_netlist(subckt=True) == before


@pytest.mark.parametrize('route', ['attribute', 'named', 'indexed'])
@pytest.mark.parametrize('field,key,value', [('params', 'w', 2.), ('nets', 'd', 'new'),
                                           ('layout', 'x', 1)])
@pytest.mark.parametrize('operation', ['assign', 'delete', 'update', 'clear'])
def test_public_mappings_are_read_only(tmp_path, route, field, key, value, operation):
    top, leaf, h, dev = design(tmp_path)
    dev.layout['x'] = 0
    leaf.elem_dict['bank[5]']['device'].layout['x'] = 0
    view = access(h, route)
    mapping = getattr(view, field)
    assert isinstance(mapping, Mapping)
    before = dict(mapping)
    with pytest.raises((TypeError, AttributeError)):
        if operation == 'assign':
            mapping[key] = value
        elif operation == 'delete':
            del mapping[key]
        elif operation == 'update':
            mapping.update({key: value})
        else:
            mapping.clear()
    assert dict(mapping) == before


def test_reads_are_live_and_original_edits_still_work(tmp_path):
    top, leaf, h, dev = design(tmp_path)
    view = h.m1
    assert view is not dev
    second = top.add_module(leaf, 'other', out='o2', inp='i2')
    assert second.m1 is not view
    for name in (n for n in dir(dev) if not n.startswith('_')):
        original = getattr(dev, name)
        if not callable(original):
            assert getattr(view, name) == original
        assert name in dir(view)
    assert view.to_spice() == dev.to_spice()
    assert 'm1' in repr(view)
    held_params = view.params
    dev.w = 2.
    dev.d = 'new'
    assert view.w == 2.
    assert held_params['w'] == 2.
    assert view.nets['d'] == dev.nets['d'] == 'new'
    assert leaf._element_mgr._pin_to_net['m1', 'd'] == 'new'
    width = Variable('width', value=3.)
    leaf.update_element('m1', ['w'], [width])
    assert view.w is width
    assert view.params['w'] is width
    width.value = 5.
    assert view.w.value == 5.
    assert 'w=5' in view.to_spice()


HELPERS = [(stuck_open, {}), (short_ds, {}), (short_gs, {}), (short_gd, {}),
           (w_variation, {'sigma': .1}), (l_variation, {'sigma': .1}),
           (vth_shift, {'delta': .01})]


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('helper,options', HELPERS)
def test_fault_targets_follow_authored_keys_and_resolve(tmp_path, legacy, reverse, helper, options):
    top, leaf, h, dev = design(tmp_path, legacy=legacy)
    other = module(tmp_path, 'other', top.pdk_details)
    second = other.add_module(leaf, 'scalar', out='o', inp='i')
    sibling = top.add_module(leaf, 'sibling', out='s', inp='i')
    outer = module(tmp_path, 'outer', top.pdk_details)
    wrapper = outer.add_module(top, 'wrap', nets={})
    lookups = [(top, lambda: h.m1, 'xstage[8].m1'),
               (other, lambda: second.m1, 'xscalar.m1'),
               (top, lambda: sibling.m1, 'xsibling.m1'),
               (outer, lambda: wrapper.child('stage')[8].bank[5],
                'xwrap.xstage[8].bank[5]')]
    if reverse:
        lookups.reverse()
    held = [(root, get(), target) for root, get, target in lookups]
    for root, view, target in held:
        for fault in (helper(view, **options), helper(**options)(view)):
            assert fault.target == target
            assert root._silk_resolve_fault_target(fault.target) is leaf.elem_dict[target.split('.')[-1]]['device']
        assert target in root._silk_fault_all_device_paths()
    assert not hasattr(dev, '_silk_fault_target_path')


@pytest.mark.parametrize('si', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_inspection_geometry_and_area(tmp_path, si, legacy):
    top, leaf, h, dev = design(tmp_path, si=si, legacy=legacy)
    view = h.m1
    assert emitted_instance_name(view, 'm1') == 'XMm1'
    assert emitted_instance_name(view, 'bank[5]') == 'XMbank<5>'
    assert device_geometry_um(view, leaf.pdk_details) == pytest.approx((4., .5, 2.))
    assert leaf.pdk_details.device_geometry_um(view) == pytest.approx((4., .5, 2.))
    assert compute_device_area(view, pdk=leaf.pdk_details) == pytest.approx(10.)
    if legacy:
        # element_names never supported PDKDevice; preserve that refusal.
        with pytest.raises(TypeError):
            element_names(view)
        with pytest.raises(TypeError):
            device_geometry_um(view)
    else:
        assert element_names(view) == element_names(dev)
        assert element_names(view).card_names == ('XMm1',)
        assert device_geometry_um(view) == pytest.approx((4., .5, 2.))


@pytest.mark.parametrize('kind', ['ancestor_entry', 'ancestor_device', 'ancestor_remove',
                                 'leaf_entry', 'leaf_device', 'leaf_remove'])
def test_held_views_and_mappings_refuse_replacement(tmp_path, kind):
    top, leaf, h, dev = design(tmp_path)
    outer = module(tmp_path, 'outer', top.pdk_details)
    view = outer.add_module(top, 'wrap', nets={}).child('stage')[8].m1
    params, nets, layout = view.params, view.nets, view.layout
    if kind.startswith('ancestor'):
        registry, key = outer._element_mgr.elem_dict, 'wrap'
    else:
        registry, key = leaf._element_mgr.elem_dict, 'm1'
    if kind.endswith('entry'):
        registry[key] = dict(registry[key])
    elif kind.endswith('device'):
        registry[key]['device'] = copy.deepcopy(registry[key]['device'])
    else:
        del registry[key]
    for read in (lambda: view.w, lambda: view.name, lambda: repr(view), lambda: dir(view),
                 lambda: params['w'], lambda: dict(params), lambda: dict(nets),
                 lambda: dict(layout), lambda: stuck_open(view),
                 lambda: emitted_instance_name(view, 'm1'), lambda: device_geometry_um(view)):
        with pytest.raises(CircuitError, match='removed or replaced'):
            read()


def pickle_roundtrip(value):
    return pickle.loads(pickle.dumps(value))


@pytest.mark.parametrize('clone', [copy.copy, copy.deepcopy, pickle_roundtrip])
@pytest.mark.parametrize('view_first', [False, True])
def test_copy_keeps_view_and_graph_identity(tmp_path, clone, view_first):
    top, leaf, h, dev = design(tmp_path, si=True)
    view = h.m1
    top._held_view = view  # Cycle: deepcopy must memoize the view before its root.
    pair = [view, top] if view_first else [top, view]
    restored = clone(pair)
    root, result = (restored[1], restored[0]) if view_first else restored
    # Inspect the restored graph itself; successful copying alone is insufficient.
    assert root._held_view is result
    copied_leaf = root.elem_dict['stage[8]']['device']
    actual = copied_leaf.elem_dict['m1']['device']
    assert result is not actual
    if clone is not copy.copy:
        assert actual is not dev
    assert result.params['w'] == 4e-6
    assert root._silk_resolve_fault_target(stuck_open(result).target) is actual
    actual.w = 2e-6
    assert result.w == 2e-6
    assert device_geometry_um(result) == pytest.approx((2., .5, 2.))
    with pytest.raises(AttributeError):
        result.w = 7e-6
    with pytest.raises(TypeError):
        result.params['w'] = 7e-6
    if clone is not copy.copy:
        assert dev.w == 4e-6


@pytest.mark.parametrize('clone', [copy.copy, copy.deepcopy, pickle_roundtrip])
def test_copy_view_alone_stays_read_only(tmp_path, clone):
    top, leaf, h, dev = design(tmp_path)
    result = clone(h.m1)
    assert result.w == 4.
    with pytest.raises(AttributeError):
        result.w = 8.


@pytest.mark.parametrize('clone', [copy.deepcopy, pickle_roundtrip])
def test_stale_view_does_not_rebind_during_copy(tmp_path, clone):
    top, leaf, h, dev = design(tmp_path)
    view = h.m1
    leaf._element_mgr.elem_dict['m1'] = dict(leaf.elem_dict['m1'])
    root, result = clone([top, view])
    with pytest.raises(CircuitError, match='removed or replaced'):
        stuck_open(result)
    with pytest.raises(CircuitError, match='removed or replaced'):
        _ = result.w


@pytest.mark.parametrize('api', ['add', 'add_element', 'add_module', 'add_submodule',
                               'add_passive', 'add_source', 'manager_add', 'manager_factory'])
def test_view_cannot_enter_any_registry(tmp_path, api):
    top, leaf, h, dev = design(tmp_path)
    # An ideal passive is also a DEVICE placement and exercises passive duck typing.
    leaf.add(elements.resistor('r1', p='out', n='0', value=1000))
    view = h.r1
    target = module(tmp_path, 'target', top.pdk_details)
    before = (dict(target.elem_dict), dict(target.module_dict),
              dict(target.passives_dict), dict(target.sig_sources_dict), dict(target.net_dict))
    calls = {'add': lambda: target.add(view),
             'add_element': lambda: target.add_element(view, 'r1'),
             'add_module': lambda: target.add_module(view, 'child'),
             'add_submodule': lambda: target.add_submodule('child', view),
             'add_passive': lambda: target.add_passive(view),
             'add_source': lambda: target.add_source(view),
             'manager_add': lambda: target._element_mgr.add_element(view, 'r1'),
             'manager_factory': lambda: target._element_mgr._add_analog_element(view)}
    with pytest.raises(TypeError, match='view|View'):
        calls[api]()
    assert (dict(target.elem_dict), dict(target.module_dict), dict(target.passives_dict),
            dict(target.sig_sources_dict), dict(target.net_dict)) == before


@pytest.mark.parametrize('initialized', [False, True])
@pytest.mark.parametrize('api', ['add', 'add_source', 'add_passive'])
def test_bench_forwarding_refuses_view_registration(tmp_path, initialized, api):
    top, leaf, h, dev = design(tmp_path)
    leaf.add(elements.resistor('r1', p='out', n='0', value=1))
    tb = leaf.create_test('forward')
    if initialized:
        tb.add(elements.resistor('existing', p='x', n='0', value=2))
    assert (tb.netlist is not None) == initialized
    with pytest.raises(TypeError, match='view|View'):
        getattr(tb, api)(h.r1)
    # Lazy netlist creation is preserved; only registration must be refused.
    assert not tb.netlist.module_dict
    assert set(tb.netlist.elem_dict) == ({'existing'} if initialized else set())
    assert not tb.netlist.passives_dict
    assert not tb.netlist.sig_sources_dict


@pytest.mark.parametrize('api', ['update_element', 'connect_module', 'update_source', 'connect_source'])
def test_named_edit_apis_do_not_treat_view_as_a_key(tmp_path, api):
    top, leaf, h, dev = design(tmp_path)
    before = top.generate_netlist(subckt=True)
    args = (h.m1, {}) if api == 'connect_module' else (h.m1, ['w'], [2])
    with pytest.raises((CircuitError, TypeError, KeyError)):
        getattr(top, api)(*args)
    assert top.generate_netlist(subckt=True) == before


def test_device_view_does_not_add_whole_device_probe(tmp_path):
    top, leaf, h, dev = design(tmp_path)
    tb = leaf.create_test('probe')
    placed = tb.add_module(leaf, 'dut', out='o', inp='i')
    with pytest.raises(TypeError):
        tb.probe(placed.m1)


def test_device_reads_use_owner_validator(tmp_path, monkeypatch):
    top, leaf, owner, dev = design(tmp_path)
    view = owner.m1
    params = view.params
    def refuse():
        raise CircuitError('owner validation invoked')
    monkeypatch.setattr(owner, '_placement_chain', refuse)
    for read in (lambda: view.w, lambda: params['w'], lambda: stuck_open(view)):
        with pytest.raises(CircuitError, match='owner validation invoked'):
            read()


def test_old_seven_field_state_refuses(tmp_path):
    top, leaf, owner, dev = design(tmp_path)
    view = owner.m1
    with pytest.raises(ValueError):
        view.__setstate__((top, (), (), leaf, 'm1', leaf.elem_dict['m1'], dev))


def test_retained_params_follow_dictionary_replacement(tmp_path):
    top, leaf, owner, dev = design(tmp_path)
    params = owner.m1.params
    dev.params = dict(dev.params, w=7.)
    assert params['w'] == 7.
