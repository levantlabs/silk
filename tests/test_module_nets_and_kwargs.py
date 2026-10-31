"""Tests for the typed wiring API:

  * ``Module.nets`` attribute accessor returning Net objects
  * ``Module.add_element(...)`` kwarg form for port wiring
  * Mixing Net objects and net-name strings as wiring values
  * Reserved-word port names via ``**{...}`` dict unpacking
  * ``add_net()`` returning the created Net
"""

import pytest

from silk.circuit.modules import Module
from silk.circuit.net import Net


# ---------------------------------------------------------------------------
# Module.nets accessor
# ---------------------------------------------------------------------------

class TestNetsAccessor:

    @pytest.mark.parametrize('name', ['_module', '_x'])
    @pytest.mark.parametrize('bus', [False, True])
    def test_declared_private_names_take_precedence(self, pdk, tmp_dir, name, bus):
        mod = Module('PDK_skywater130', tmp_dir, 'names.cir', pdk_details=pdk)
        mod.add_net(name, **({'bits': range(2, 4)} if bus else {}))
        mod.add_net('__class__')
        accessor = mod.nets
        item = getattr(accessor, name)
        assert item is accessor[name]
        if bus:
            assert item[2].name == name + '[2]'
        else:
            assert item.name == name
        assert name in accessor
        assert name in list(accessor)
        assert name in dir(accessor)
        assert '<nets on' in repr(accessor)
        assert accessor.__class__.__name__ == '_NetAccessor'
        assert accessor['__class__'].name == '__class__'
        with pytest.raises(AttributeError):
            accessor._missing
        with pytest.raises(KeyError):
            accessor['_missing']

        import copy
        import pickle

        for control_name in ('plain', '_module'):
            control = Module('PDK_skywater130', tmp_dir, 'copy.cir', pdk_details=pdk)
            control.add_net(control_name)
            original = control.nets
            for clone in (copy.copy, copy.deepcopy,
                          lambda value: pickle.loads(pickle.dumps(value))):
                restored = clone(original)
                assert isinstance(restored[control_name], Net)
                assert restored[control_name].name == original[control_name].name

    def test_returns_net_object_for_existing_net(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        mod.add_net('vout')
        got = mod.nets.vout
        assert isinstance(got, Net)
        assert got.name == 'vout'

    def test_missing_net_raises_attribute_error(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with pytest.raises(AttributeError, match='No net'):
            mod.nets.does_not_exist

    def test_dir_lists_nets_for_autocomplete(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        mod.add_net('vinp')
        mod.add_net('vout')
        listed = dir(mod.nets)
        assert 'vinp' in listed
        assert 'vout' in listed

    def test_contains_and_iter(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        mod.add_net('avdd')
        assert 'avdd' in mod.nets
        assert 'nope' not in mod.nets
        assert list(mod.nets) == ['avdd']

    def test_accessor_is_live_view_not_snapshot(self, pdk, tmp_dir):
        """Nets added after the accessor is obtained must still be visible."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        accessor = mod.nets
        mod.add_net('vinp')
        assert accessor.vinp.name == 'vinp'

    def test_bracket_access_for_non_identifier_names(self, pdk, tmp_dir):
        """Net names like '0' (ngspice ground) can't be attribute-accessed.
        __getitem__ is the documented escape hatch."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        mod.add_net('0')
        assert mod.nets['0'].name == '0'

    def test_bracket_access_missing_raises_key_error(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        with pytest.raises(KeyError):
            mod.nets['nope']


# ---------------------------------------------------------------------------
# add_net now returns the Net object
# ---------------------------------------------------------------------------

class TestAddNetReturn:

    def test_returns_created_net(self, pdk, tmp_dir):
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        net = mod.add_net('avdd')
        assert isinstance(net, Net)
        assert net.name == 'avdd'
        assert net is mod.net_dict['avdd']

    def test_returns_same_net_on_repeat_call(self, pdk, tmp_dir):
        """add_net is idempotent; the second call returns the same Net
        object so callers can safely ignore the 'exists already' case."""
        mod = Module('PDK_skywater130', tmp_dir, 'm.cir', pdk_details=pdk)
        first = mod.add_net('avdd')
        second = mod.add_net('avdd')
        assert first is second


# ---------------------------------------------------------------------------
# add_element port-kwarg form
# ---------------------------------------------------------------------------

def _build_subckt(pdk, tmp_dir):
    """Minimal subcircuit with two ports so we can exercise wiring."""
    from silk.circuit.circuit_elements import Port
    from silk import VoltageDomain

    dom = VoltageDomain('vdom', nominal=1.8, tolerance=0.1)
    sub = Module('PDK_skywater130', tmp_dir, 'sub.cir', pdk_details=pdk,
                 module_name='sub')
    with sub:
        pdk.nmos1p8('mn', d='d', g='g', s='0', b='0', w=1.0, l=0.18, m=1)
    sub.define_module_ports({
        'g': Port.analog_input(dom),
        'd': Port.analog_output(dom),
    })
    return sub


class TestAddElementKwargs:

    def test_kwarg_form_with_net_objects(self, pdk, tmp_dir):
        sub = _build_subckt(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        in_net = tb.add_net('vin')
        out_net = tb.add_net('vout')
        tb.add_module(sub, 'u1', g=in_net, d=out_net)

        stored = tb.elem_dict['u1']['pin_list']
        assert stored['g'].net == 'vin'
        assert stored['d'].net == 'vout'

    def test_kwarg_form_with_strings_still_works(self, pdk, tmp_dir):
        """Net-name strings as values must keep working for the escape-hatch
        and legacy paths."""
        sub = _build_subckt(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        tb.add_net('vin')
        tb.add_net('vout')
        tb.add_module(sub, 'u1', g='vin', d='vout')

        stored = tb.elem_dict['u1']['pin_list']
        assert stored['g'].net == 'vin'
        assert stored['d'].net == 'vout'

    def test_mixed_net_objects_and_strings(self, pdk, tmp_dir):
        sub = _build_subckt(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        vout = tb.add_net('vout')
        tb.add_net('vin')
        tb.add_module(sub, 'u1', g='vin', d=vout)

        stored = tb.elem_dict['u1']['pin_list']
        assert stored['g'].net == 'vin'
        assert stored['d'].net == 'vout'

    def test_legacy_nets_dict_still_works(self, pdk, tmp_dir):
        """Backward compat: nets={...} dict form unchanged."""
        sub = _build_subckt(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        tb.add_net('vin')
        tb.add_net('vout')
        tb.add_module(sub, 'u1', nets={'g': 'vin', 'd': 'vout'})

        stored = tb.elem_dict['u1']['pin_list']
        assert stored['g'].net == 'vin'

    def test_kwargs_override_dict_entries(self, pdk, tmp_dir):
        """When both forms present, kwargs win — enables the common
        'mostly identity, override one port' pattern."""
        sub = _build_subckt(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        tb.add_net('vin')
        new_out = tb.add_net('vnew')
        tb.add_module(sub, 'u1',
                       nets={'g': 'vin', 'd': 'old_net'},
                       d=new_out)

        stored = tb.elem_dict['u1']['pin_list']
        assert stored['d'].net == 'vnew'

    def test_reserved_word_via_dict_unpacking(self, pdk, tmp_dir):
        """Port names that shadow Python keywords can't be passed as raw
        kwargs (SyntaxError). Dict unpacking is the documented fallback."""
        from silk.circuit.circuit_elements import Port
        from silk import VoltageDomain

        dom = VoltageDomain('vdom', nominal=1.8, tolerance=0.1)
        sub = Module('PDK_skywater130', tmp_dir, 'weird.cir',
                     pdk_details=pdk, module_name='weird')
        with sub:
            pdk.nmos1p8('mn', d='out', g='in', s='0', b='0',
                        w=1.0, l=0.18, m=1)
        sub.define_module_ports({
            'in':  Port.analog_input(dom),
            'out': Port.analog_output(dom),
        })

        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        a = tb.add_net('a')
        b = tb.add_net('b')
        tb.add_module(sub, 'u1', **{'in': a, 'out': b})

        stored = tb.elem_dict['u1']['pin_list']
        assert stored['in'].net == 'a'
        assert stored['out'].net == 'b'

    def test_returns_subcircuit_instance_handle(self, pdk, tmp_dir):
        """The kwarg form must still return the SubcircuitInstance handle —
        no regression vs. the legacy dict form."""
        from silk.circuit.modules import SubcircuitInstance

        sub = _build_subckt(pdk, tmp_dir)
        tb = Module('PDK_skywater130', tmp_dir, 'tb.cir', pdk_details=pdk)
        vin = tb.add_net('vin')
        vout = tb.add_net('vout')
        handle = tb.add_module(sub, 'u1', g=vin, d=vout)
        assert isinstance(handle, SubcircuitInstance)


class TestNetCaseInsensitiveIdentity:
    """G16 (2026-08-24): SPICE folds node-name case (ngspice to lower, Xyce
    to upper — proven empirically), so names differing only by case are ONE
    electrical node. silk merges them onto the first-seen spelling with a
    warning; the model (net_dict, Net.pins) is canonical while owner-side
    strings and the emitted deck keep the author's verbatim spelling."""

    def _amp(self, pdk, tmp_dir):
        m = Module('PDK_skywater130', tmp_dir, 'case.cir', pdk_details=pdk)
        m.add(pdk.nmos1p8('m1', d='VDD', g='in', s='0', b='0', w=1.0, l=0.5))
        return m

    def test_case_variant_merges_to_one_net(self, pdk, tmp_dir):
        m = self._amp(pdk, tmp_dir)
        m.add(pdk.nmos1p8('m2', d='vdd', g='in', s='0', b='0', w=1.0, l=0.5))
        assert 'VDD' in m.net_dict and 'vdd' not in m.net_dict
        assert len(m.net_dict['VDD'].pins) == 2

    def test_merge_prints_a_warning_once(self, pdk, tmp_dir, capsys):
        m = self._amp(pdk, tmp_dir)
        m.add(pdk.nmos1p8('m2', d='vdd', g='in', s='0', b='0', w=1.0, l=0.5))
        m.add(pdk.nmos1p8('m3', d='vdd', g='in', s='0', b='0', w=1.0, l=0.5))
        err = capsys.readouterr().err
        assert err.count("differs only by case") == 1
        assert "'VDD'" in err and "'vdd'" in err

    def test_bindings_carry_the_canonical_name(self, pdk, tmp_dir):
        m = self._amp(pdk, tmp_dir)
        m.add(pdk.nmos1p8('m2', d='vdd', g='in', s='0', b='0', w=1.0, l=0.5))
        assert all(b.net == 'VDD' for b in m.net_dict['VDD'].pins)

    def test_owner_side_spelling_stays_verbatim(self, pdk, tmp_dir):
        m = self._amp(pdk, tmp_dir)
        dev = pdk.nmos1p8('m2', d='vdd', g='in', s='0', b='0', w=1.0, l=0.5)
        m.add(dev)
        assert dev.nets['d'] == 'vdd'   # deck emits what the author wrote

    def test_add_net_case_variant_is_a_noop_merge(self, pdk, tmp_dir):
        m = self._amp(pdk, tmp_dir)
        m.add_net('vdd')
        assert 'vdd' not in m.net_dict
        assert 'VDD' in m.net_dict

    def test_distinct_names_unaffected(self, pdk, tmp_dir, capsys):
        m = self._amp(pdk, tmp_dir)
        m.add_net('vout')
        assert 'vout' in m.net_dict
        assert "differs only by case" not in capsys.readouterr().err

    def test_save_list_selector_resolves_to_one_net(self, pdk, tmp_dir):
        from silk.circuit.modules_netlist import _resolve_save_list
        m = self._amp(pdk, tmp_dir)
        m.add(pdk.nmos1p8('m2', d='vdd', g='in', s='0', b='0', w=1.0, l=0.5))
        m.set_save_list(signals=['vdd'])
        assert _resolve_save_list(m) == ['VDD']
