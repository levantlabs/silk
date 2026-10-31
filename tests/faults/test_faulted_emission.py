"""Faulted decks go through the devices' own writer (TASK-415).

A fault realizes to device changes (overrides, removal, added silk
elements); the netlister emits every one of them through the element's own
writer. Checked here on emitted netlist text only — no simulator, no model
library — on every registered PDK:

* an unfaulted parameter in a faulted line is byte-identical to the clean
  deck, and a changed value is the clean value with the realized change
  applied in the PDK's own unit (GF180MCU / SG13G2 decks are metre-native,
  so a hand-written ``w=4.0`` there meant four metres);
* in ammeter mode a faulted device keeps its terminal ammeters;
* added elements emit like any element of their kind.
"""

import re
import warnings

import pytest

from silk import load_pdk
from silk.circuit.modules import Module
from silk.faults import (
    FaultSet, l_variation, short_ds, short_gd, short_gs, stuck_open,
    vth_shift, w_variation,
)
from silk.utils._utils import _fmt

#: One NMOS per registered PDK, and its catalog model name.
_NMOS = {
    'sky130': ('nmos1p8', 'sky130_fd_pr__nfet_01v8'),
    'gf180mcu': ('nmos3p3', 'nfet_03v3'),
    'sg13g2': ('nmos_lv', 'sg13_lv_nmos'),
}
_PDKS = sorted(_NMOS)


def _deck(pdk_name, tmp_path, fault=None, *, m=2, ammeter=False,
          user_units=None, tag='deck', api='factory', manifest=None, seed=7):
    """Build one NMOS on *pdk_name* through *api* — ``'factory'``
    (``pdk.<dev>(...)``) or ``'add_element'`` (the stored-``PDKDevice``
    route, ``add_element(pdk.cls, ...)``) — and return the emitted deck."""
    kwargs = {'user_units': user_units} if user_units else {}
    pdk = load_pdk(pdk_name, **kwargs)
    mod = Module(pdk_name, str(tmp_path), f'{tag}_{api}.cir', pdk_details=pdk)
    for net in ('d', 'g', 's'):
        mod.add_net(net)
    device_type, _ = _NMOS[pdk_name]
    w, l = (4e-6, 1e-6) if user_units == 'si' else (4.0, 1.0)
    if api == 'factory':
        with mod:
            getattr(pdk, device_type)('mn', d='d', g='g', s='s', b='s',
                                      w=w, l=l, m=m)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            mod.add_element(getattr(pdk, device_type), 'mn',
                            nets=dict(d='d', g='g', s='s', b='s'),
                            parameters={'w': w, 'l': l, 'm': m})
    if ammeter:
        from silk.simulation import Transient
        mod.add_sim_type(Transient(step=1e-9, stop=10e-9))
        mod.save_device_currents = True
        mod.terminal_current_method = 'ammeter_wrapper'
    fault_set = FaultSet([fault]) if fault is not None else None
    if manifest is None:
        manifest = []
    with warnings.catch_warnings():
        # No fault_params bounds on GF180MCU / SG13G2: the unclamped-
        # perturbation warning is expected and asserted elsewhere.
        warnings.simplefilter('ignore', UserWarning)
        deck_path = mod.generate_netlist(fault_set=fault_set, manifest=manifest, seed=seed)
    with open(deck_path) as f:
        text = f.read()
    assert len(manifest) == (1 if fault is not None else 0)
    if fault is not None:
        assert manifest[0].target == fault.target
        assert manifest[0].netlist_comment in text
    return text


_APIS = ['factory', 'add_element']


@pytest.fixture(params=_APIS)
def api(request):
    """Both device APIs: a factory-built element and a stored PDKDevice
    (``add_element(pdk.cls, ...)``, TASK-416) must fault identically."""
    return request.param


def _instance_lines(text, name='mn'):
    """Every instance line of device *name* (including generated bad-finger
    instances), keyed by instance token."""
    out = {}
    for line in text.splitlines():
        tok = line.split()
        if tok and re.fullmatch(rf'XM(?:{name}|__silk_fault_bad_{name.encode("utf-8").hex()})', tok[0]):
            out[tok[0]] = line
    return out


def _params(line):
    return dict(t.split('=', 1) for t in line.split() if '=' in t)


def _clean(pdk_name, tmp_path, **kw):
    return _deck(pdk_name, tmp_path, tag='clean', **kw)


# --- every fault type, every PDK: unchanged params byte-identical ---------

@pytest.mark.parametrize('pdk_name', _PDKS)
def test_stuck_open_per_finger_changes_only_m(pdk_name, api, tmp_path):
    clean = _params(_instance_lines(_clean(pdk_name, tmp_path, api=api))['XMmn'])
    faulted = _instance_lines(
        _deck(pdk_name, tmp_path, stuck_open('mn'), api=api))
    assert list(faulted) == ['XMmn']
    got = _params(faulted['XMmn'])
    assert got['m'] == '1'
    assert {k: v for k, v in got.items() if k != 'm'} == \
        {k: v for k, v in clean.items() if k != 'm'}


@pytest.mark.parametrize('pdk_name', _PDKS)
def test_gf180_and_sg13g2_faulted_geometry_is_not_metres(pdk_name, api, tmp_path):
    """The defect TASK-415 fixed: the faulted line re-emitted ``w=4.0
    l=1.0`` raw, which a metre-native PDK reads as 4 m by 1 m."""
    faulted = _params(_instance_lines(
        _deck(pdk_name, tmp_path, stuck_open('mn'), api=api))['XMmn'])
    expected = ('4', '1') if pdk_name == 'sky130' else ('4u', '1u')
    assert (faulted['w'], faulted['l']) == expected


@pytest.mark.parametrize('pdk_name', _PDKS)
def test_stuck_open_whole_device_removes_it(pdk_name, api, tmp_path):
    text = _deck(pdk_name, tmp_path, stuck_open('mn', per_finger=False), api=api)
    assert _instance_lines(text) == {}
    assert '*** FAULT: stuck_open on mn (was m=2, deleted)' in text


@pytest.mark.parametrize('pdk_name', _PDKS)
@pytest.mark.parametrize('sugar, a, b', [
    (short_ds, 'd', 's'), (short_gs, 'g', 's'), (short_gd, 'g', 'd')])
def test_short_per_finger_decomposes_with_clean_params(
        pdk_name, sugar, a, b, api, tmp_path):
    clean = _params(_instance_lines(_clean(pdk_name, tmp_path, api=api))['XMmn'])
    text = _deck(pdk_name, tmp_path, sugar('mn'), api=api)
    lines = _instance_lines(text)
    assert set(lines) == {'XMmn', 'XM__silk_fault_bad_6d6e'}
    good, bad = _params(lines['XMmn']), _params(lines['XM__silk_fault_bad_6d6e'])
    assert (good['m'], bad['m']) == ('1', '1')
    for got in (good, bad):
        assert {k: v for k, v in got.items() if k != 'm'} == \
            {k: v for k, v in clean.items() if k != 'm'}
    # The shunt is an ideal resistor, emitted by the resistor's own writer.
    assert f'R__silk_fault_short_6d6e {a} {b} 1u' in text.splitlines()


@pytest.mark.parametrize('pdk_name', _PDKS)
def test_short_whole_device_keeps_the_clean_line(pdk_name, api, tmp_path):
    clean = _instance_lines(_clean(pdk_name, tmp_path, api=api))
    faulted = _instance_lines(
        _deck(pdk_name, tmp_path, short_ds('mn', per_finger=False), api=api))
    assert faulted == clean


@pytest.mark.parametrize('pdk_name', _PDKS)
@pytest.mark.parametrize('which, sugar', [('w', w_variation),
                                          ('l', l_variation)])
def test_w_l_variation_applies_the_change_in_pdk_units(
        pdk_name, which, sugar, api, tmp_path):
    clean = _params(_instance_lines(_clean(pdk_name, tmp_path, api=api))['XMmn'])
    manifest = []
    text = _deck(pdk_name, tmp_path, sugar('mn', sigma=0.05), api=api,
                 manifest=manifest, seed=3)
    got = _params(_instance_lines(text)['XMmn'])
    assert len(manifest) == 1
    rp = manifest[0].realized_params
    nominal = 4.0 if which == 'w' else 1.0
    # Manifest in the user's units (microns here, the circuit's units).
    assert rp[f'{which}_nominal'] == nominal
    assert rp[f'{which}_new'] == pytest.approx(
        nominal * rp[f'{which}_multiplier'])
    pdk = load_pdk(pdk_name)
    scale = getattr(pdk, _NMOS[pdk_name][0])._param_scales[which]
    assert got[which] == _fmt(rp[f'{which}_new'] * scale)
    assert {k: v for k, v in got.items() if k != which} == \
        {k: v for k, v in clean.items() if k != which}


@pytest.mark.parametrize('pdk_name', _PDKS)
def test_vth_shift_rewires_only_the_gate(pdk_name, api, tmp_path):
    clean = _instance_lines(_clean(pdk_name, tmp_path, api=api))['XMmn'].split()
    text = _deck(pdk_name, tmp_path, vth_shift('mn', delta=0.02), api=api)
    got = _instance_lines(text)['XMmn'].split()
    # XMmn d g s b <model> params: only the gate token (index 2) changes.
    assert got[2] == '__silk_fault_shifted_6d6e'
    assert got[:2] + got[3:] == clean[:2] + clean[3:]
    lines = text.splitlines()
    # Unity-gain VCVS from the gate, sitting on a DC source of delta: the
    # silk VCVS and VoltageSource writers, readable by both simulators.
    assert 'E__silk_fault_vt_6d6e __silk_fault_shifted_6d6e __silk_fault_offset_6d6e g 0 1' in lines
    assert any(l.startswith('V__silk_fault_vt_6d6e __silk_fault_offset_6d6e 0 DC 0.02') for l in lines)


# --- user_units='si' on sky130: clamp in the bounds' own unit -------------

def test_w_variation_clamps_in_pdk_units_when_user_writes_metres(tmp_path):
    """sky130 bounds (min 0.42) are in microns. A circuit written in metres
    (w=4e-6) must be perturbed and clamped as 4 um, not as 4e-6 against
    0.42 — which clamped every perturbation to the minimum."""
    pdk = load_pdk('sky130', user_units='si')
    mod = Module('sky130', str(tmp_path), 'si.cir', pdk_details=pdk)
    for net in ('d', 'g', 's'):
        mod.add_net(net)
    with mod:
        pdk.nmos1p8('mn', d='d', g='g', s='s', b='s', w=4e-6, l=1e-6, m=1)
    manifest = []
    mod_deck_path = mod.generate_netlist(fault_set=FaultSet([w_variation('mn', sigma=0.02)]),
                         manifest=manifest, seed=3)
    assert len(manifest) == 1
    rp = manifest[0].realized_params
    assert rp['clamped'] is False
    assert rp['w_nominal'] == 4e-6
    assert rp['w_new'] == pytest.approx(4e-6 * rp['w_multiplier'])
    with open(mod_deck_path) as f:
        got = _params(_instance_lines(f.read())['XMmn'])
    assert got['w'] == _fmt(rp['w_new'] * 1e6)


# --- ammeter mode ----------------------------------------------------------

# SG13G2 is absent: ammeter mode rides on save_device_currents, which
# refuses an OSDI PDK (no .control block to load the models from).
@pytest.mark.parametrize('pdk_name', ['gf180mcu', 'sky130'])
@pytest.mark.parametrize('fault', [
    stuck_open('mn'), short_ds('mn'), w_variation('mn', sigma=0.05),
    vth_shift('mn', delta=0.02)], ids=lambda f: f.fault_type)
def test_faulted_device_keeps_its_terminal_ammeters(pdk_name, fault, api, tmp_path):
    model = _NMOS[pdk_name][1]
    clean = _instance_lines(_clean(pdk_name, tmp_path, ammeter=True, api=api))['XMmn']
    assert clean.split()[5] == f'amm_{model}'
    text = _deck(pdk_name, tmp_path, fault, ammeter=True, api=api)
    lines = _instance_lines(text)
    assert lines
    for line in lines.values():
        assert line.split()[5] == f'amm_{model}'
    assert f'.subckt amm_{model} d g s b' in text
    # The added shunt is an ideal passive: never ammeter-wrapped, exactly
    # like a resistor the user placed.
    if fault.fault_type == 'short_ds':
        assert 'R__silk_fault_short_6d6e d s 1u' in text.splitlines()


def test_faulting_never_mutates_the_module_device(tmp_path):
    pdk = load_pdk('gf180mcu')
    mod = Module('gf180mcu', str(tmp_path), 'mut.cir', pdk_details=pdk)
    for net in ('d', 'g', 's'):
        mod.add_net(net)
    with mod:
        mn = pdk.nmos3p3('mn', d='d', g='g', s='s', b='s', w=4.0, l=1.0, m=2)
    before = (dict(mn.params), dict(mn.nets), set(mn._caller_supplied))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for fault in (short_gd('mn'), w_variation('mn', sigma=0.1),
                      vth_shift('mn', delta=0.1)):
            mod.generate_netlist(fault_set=FaultSet([fault]))
    assert (dict(mn.params), dict(mn.nets), set(mn._caller_supplied)) == before
    assert set(mod.elem_dict) == {'mn'}
    assert not mod.sig_sources_dict


def test_faulting_never_mutates_a_stored_pdk_device(tmp_path):
    """The add_element(pdk.cls, ...) route (TASK-416): instances are copies
    of the stored PDKDevice; its pins and constraints stay as built."""
    pdk = load_pdk('gf180mcu')
    mod = Module('gf180mcu', str(tmp_path), 'mut_ae.cir', pdk_details=pdk)
    for net in ('d', 'g', 's'):
        mod.add_net(net)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        mod.add_element(pdk.nmos3p3, 'mn', nets=dict(d='d', g='g', s='s', b='s'),
                        parameters={'w': 4.0, 'l': 1.0, 'm': 2})
    dev = mod.elem_dict['mn']['device']

    def snapshot():
        return ({p: pin.net for p, pin in dev.pin_list.items()},
                {p: (c.value, c.default) for p, c in dev.param_list.items()})

    before = snapshot()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for fault in (short_gd('mn'), w_variation('mn', sigma=0.1),
                      vth_shift('mn', delta=0.1)):
            mod.generate_netlist(fault_set=FaultSet([fault]))
    assert snapshot() == before
    assert set(mod.elem_dict) == {'mn'}


@pytest.mark.parametrize('pdk_name', _PDKS)
@pytest.mark.parametrize('user_units', ['u', 'si'])
def test_shared_faulted_definition_preserves_device_writer_units(
        pdk_name, api, user_units, tmp_path):
    from pathlib import Path
    pdk = load_pdk(pdk_name, user_units=user_units)
    sub = Module(pdk_name, str(tmp_path), 'shared.cir', pdk_details=pdk,
                 module_name='shared')
    sub.define_module_ports({n: 'inout' for n in ('d', 'g', 's', 'b')})
    w, l = (4e-6, 1e-6) if user_units == 'si' else (4.0, 1.0)
    factory = getattr(pdk, _NMOS[pdk_name][0])
    if api == 'factory':
        sub.add(factory('mn', d='d', g='g', s='s', b='b', w=w, l=l, m=2))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            sub.add_element(factory, 'mn', nets=dict(d='d', g='g', s='s', b='b'),
                            parameters={'w': w, 'l': l, 'm': 2})
    top = Module(pdk_name, str(tmp_path), 'top.cir', pdk_details=pdk)
    for n in ('d', 'g', 's', 'b'):
        top.add_net(n)
    for name in ('left', 'right'):
        top.add_module(sub, name, nets={n: n for n in ('d', 'g', 's', 'b')})
    manifest = []
    text = Path(top.generate_netlist(
        fault_set=FaultSet([stuck_open('xleft.mn')]), manifest=manifest)).read_text()
    bodies = dict(re.findall(r'(?ms)^\.subckt (\S+)[^\n]*\n(.*?)^\.ends[^\n]*', text))
    good = _instance_lines(bodies['shared'])['XMmn']
    bad = _instance_lines(bodies['__silk_fault_shared_786c656674'])['XMmn']
    assert good.split()[:6] == bad.split()[:6]
    assert good.split()[1:6] == ['d', 'g', 's', 'b', _NMOS[pdk_name][1]]
    gp, bp = _params(good), _params(bad)
    assert gp.pop('m') == '2'
    assert bp.pop('m') == '1'
    assert gp == bp
    scale = factory._param_scales['w']
    assert gp['w'] == _fmt(w * scale)
    assert [r.target for r in manifest] == ['xleft.mn']
    record = next(r for r in manifest if r.target == 'xleft.mn')
    assert record.fault_type == 'stuck_open'
    assert record.realized_params['m_before'] == 2
    assert record.netlist_comment in bodies['__silk_fault_shared_786c656674']


@pytest.mark.parametrize('defaults', [False, True])
@pytest.mark.parametrize('which,sugar', [('w', w_variation), ('l', l_variation)])
def test_si_fault_preparation_preserves_catalog_defaults_and_caller_units(
        tmp_path, api, defaults, which, sugar):
    pdk = load_pdk('sky130', user_units='si')
    mod = Module('sky130', str(tmp_path), 'defaults.cir', pdk_details=pdk)
    for n in ('d', 'g', 's'):
        mod.add_net(n)
    params = {} if defaults else {'w': 4e-6, 'l': 1e-6}
    if api == 'factory':
        mod.add(pdk.nmos1p8('mn', d='d', g='g', s='s', b='s', **params))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            mod.add_element(pdk.nmos1p8, 'mn', nets=dict(d='d', g='g', s='s', b='s'),
                            parameters=params)
    clean_path = mod.generate_netlist()
    with open(clean_path) as f:
        clean = _params(_instance_lines(f.read())['XMmn'])
    manifest = []
    path = mod.generate_netlist(fault_set=FaultSet([sugar('mn', sigma=.01)]),
                               seed=3, manifest=manifest)
    assert len(manifest) == 1
    rp = manifest[0].realized_params
    # The clean writer is the unit oracle, including unscaled catalog defaults.
    assert _fmt(rp[which + '_nominal'] * 1e6) == clean[which]
    with open(path) as f:
        got = _params(_instance_lines(f.read())['XMmn'])
    assert got[which] == _fmt(rp[which + '_new'] * 1e6)
    assert {k: v for k, v in got.items() if k != which} == {
        k: v for k, v in clean.items() if k != which}


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('user_units', ['u', 'si'])
@pytest.mark.parametrize('fault_kind', ['short', 'shift'])
def test_helpers_leave_collision_candidates_unchanged(tmp_path, api, simulator,
                                                       user_units, fault_kind):
    from pathlib import Path
    from silk.elements import elements
    pdk = load_pdk('sky130', user_units=user_units)
    mod = Module('sky130', str(tmp_path), 'collision.cir', pdk_details=pdk)
    mod.set_output_options(mode='overwrite')
    mod.simulator = simulator
    for n in ('d', 'g', 's', 'vp_shifted__mn'):
        mod.add_net(n)
    w, l = (1e-6, .15e-6) if user_units == 'si' else (1., .15)
    for name, width in [('mn', w), ('mn__bad', 2*w)]:
        if api == 'factory':
            mod.add(pdk.nmos1p8(name, d='d', g='g', s='s', b='s',
                               w=width, l=l, m=2))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                mod.add_element(pdk.nmos1p8, name,
                                nets=dict(d='d', g='g', s='s', b='s'),
                                parameters=dict(w=width, l=l, m=2))
    mod.add(elements.resistor('short_mn', p='d', n='s', value=1e3))
    mod.add(elements.resistor('witness', p='vp_shifted__mn', n='s', value=2e3))
    clean = Path(mod.generate_netlist()).read_text().splitlines()
    fault = short_ds('mn') if fault_kind == 'short' else vth_shift('mn', delta=.05)
    manifest = []
    lines = Path(mod.generate_netlist(fault_set=FaultSet([fault]),
                                     manifest=manifest)).read_text().splitlines()
    for prefix in ('XMmn__bad ', 'Rshort_mn ', 'Rwitness '):
        assert [l for l in lines if l.startswith(prefix)] == [
            l for l in clean if l.startswith(prefix)]
    assert [l for l in lines if ' vp_shifted__mn ' in l] == [
        'Rwitness vp_shifted__mn s 2k']
    retained = next(l for l in lines if l.startswith('XMmn '))
    if fault_kind == 'short':
        assert 'R__silk_fault_short_6d6e d s 1u' in lines
        bad = next(l for l in lines if l.startswith('XM__silk_fault_bad_6d6e '))
        assert _params(retained)['m'] == _params(bad)['m'] == '1'
        assert retained.split()[1:] == bad.split()[1:]
        assert _params(bad)['w'] == '1'
    else:
        assert retained.split()[2] == '__silk_fault_shifted_6d6e'
        assert 'E__silk_fault_vt_6d6e __silk_fault_shifted_6d6e __silk_fault_offset_6d6e g 0 1' in lines
    assert len(manifest) == 1 and manifest[0].target == 'mn'
    assert Path(mod.generate_netlist()).read_text().splitlines() == clean


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('fault_kind', ['short', 'shift'])
def test_complete_indexed_targets_have_distinct_helpers(tmp_path, simulator, fault_kind):
    from pathlib import Path
    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'targets.cir', pdk_details=pdk)
    top.simulator = simulator
    for n in ('d', 'g', 's'):
        top.add_net(n)
    sub = Module('sky130', str(tmp_path), 'sub.cir', pdk_details=pdk,
                 module_name='sub')
    sub.define_module_ports({n: 'inout' for n in ('d', 'g', 's')})
    sub.add(pdk.nmos1p8('m', d='d', g='g', s='s', b='s', w=1, l=.15, m=2))
    for name in ('u', 'u[8]'):
        top.add_module(sub, name, nets={n: n for n in ('d', 'g', 's')})
    for name in ('xu_m', 'mi[3]'):
        top.add(pdk.nmos1p8(name, d='d', g='g', s='s', b='s', w=1, l=.15, m=2))
    targets = ['xu.m', 'xu_m', 'xu[8].m', 'mi[3]']
    assert set(targets) == set(top._silk_fault_all_device_paths())
    faults = [short_ds(t) if fault_kind == 'short' else vth_shift(t, delta=.05)
              for t in targets]
    manifest = []
    lines = Path(top.generate_netlist(fault_set=FaultSet(faults),
                                     manifest=manifest)).read_text().splitlines()
    for code in ('78752e6d', '78755f6d', '78755b385d2e6d', '6d695b335d'):
        if fault_kind == 'short':
            assert lines.count(f'R__silk_fault_short_{code} d s 1u') == 1
            assert sum(l.startswith(f'XM__silk_fault_bad_{code} ') for l in lines) == 1
        else:
            assert lines.count(f'E__silk_fault_vt_{code} __silk_fault_shifted_{code} __silk_fault_offset_{code} g 0 1') == 1
            assert lines.count(f'V__silk_fault_vt_{code} __silk_fault_offset_{code} 0 DC 0.05') == 1
    assert [r.target for r in manifest] == targets
    assert any(l.startswith('XMmi<3> ') for l in lines)


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
def test_helpers_coexist_with_faulted_short_and_bad_definitions(tmp_path, simulator):
    from pathlib import Path
    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'roles.cir', pdk_details=pdk)
    top.simulator = simulator
    for n in ('d', 'g', 's'):
        top.add_net(n)
    for role in ('short', 'bad'):
        sub = Module('sky130', str(tmp_path), role+'.cir', pdk_details=pdk,
                     module_name=role)
        sub.define_module_ports({n: 'inout' for n in ('d', 'g', 's')})
        sub.add(pdk.nmos1p8('m', d='d', g='g', s='s', b='s', w=1, l=.15, m=2))
        top.add_module(sub, role, nets={n: n for n in ('d', 'g', 's')})
    text = Path(top.generate_netlist(fault_set=FaultSet([
        short_ds('xshort.m'), short_ds('xbad.m')]))).read_text()
    bodies = dict(re.findall(r'(?ms)^\.subckt (\S+)[^\n]*\n(.*?)^\.ends[^\n]*', text))
    for role, ancestry, target in [('short', '7873686f7274', '7873686f72742e6d'),
                                    ('bad', '78626164', '786261642e6d')]:
        assert _params(_instance_lines(bodies[role], 'm')['XMm'])['m'] == '2'
        body = bodies[f'__silk_fault_{role}_{ancestry}']
        assert f'R__silk_fault_short_{target} d s 1u' in body.splitlines()
        assert f'XM__silk_fault_bad_{target} d g s s ' in body
        assert _params(next(l for l in body.splitlines() if l.startswith('XMm ')))['m'] == '1'


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('kind,params', [
    ('stuck_open', {}), ('short_ds', {}), ('short_gs', {}), ('short_gd', {}),
    ('w_variation', {'sigma': .02}), ('l_variation', {'sigma': .02}),
    ('vth_shift', {'delta': .05}),
])
def test_every_fault_preserves_targets_and_reserves_all_helpers(
        tmp_path, simulator, nested, kind, params):
    from pathlib import Path
    from silk.faults.types import Fault
    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'all.cir', pdk_details=pdk)
    top.simulator = simulator
    for n in ('d', 'g', 's'):
        top.add_net(n)
    if nested:
        owner = Module('sky130', str(tmp_path), 'sub.cir', pdk_details=pdk,
                       module_name='sub')
        owner.define_module_ports({n: 'inout' for n in ('d', 'g', 's')})
    else:
        owner = top
    owner.add(pdk.nmos1p8('mn', d='d', g='g', s='s', b='s', w=1, l=.15, m=2))
    if nested:
        top.add_module(owner, 'u', nets={n: n for n in ('d', 'g', 's')})
    target, code = ('xu.mn', '78752e6d6e') if nested else ('mn', '6d6e')
    records = []
    text = Path(top.generate_netlist(
        fault_set=FaultSet([Fault(target=target, fault_type=kind, params=params)]),
        manifest=records, seed=7)).read_text()
    if nested:
        bodies = dict(re.findall(r'(?ms)^\.subckt (\S+)[^\n]*\n(.*?)^\.ends[^\n]*', text))
        text = bodies['__silk_fault_sub_7875']
    lines = text.splitlines()
    cards = [l.split() for l in lines if l.startswith(('XM', 'R', 'E', 'V'))]
    assert len(records) == 1 and records[0].target == target
    assert records[0].declared_params == params
    assert records[0].netlist_comment in lines
    assert sum(c[0] == 'XMmn' for c in cards) == 1
    helpers = [c for c in cards if c[0] != 'XMmn']
    assert len(helpers) == (2 if kind.startswith('short_') or kind == 'vth_shift' else 0)
    for card in helpers:
        prefix = 'XM' if card[0].startswith('XM') else card[0][0]
        assert card[0][len(prefix):].startswith('__silk_')
        assert card[0].endswith('_'+code)
    target_card = next(c for c in cards if c[0] == 'XMmn')
    expected_gate = '__silk_fault_shifted_'+code if kind == 'vth_shift' else 'g'
    assert target_card[1:5] == ['d', expected_gate, 's', 's']
