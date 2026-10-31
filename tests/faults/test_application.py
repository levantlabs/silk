"""``_realize_fault``: what one fault changes about one device.

A fault realizes to device changes — instances to emit (none = removed),
PDK-unit parameter overrides, pin rewires and added-element specs — never to
SPICE text (TASK-415). Emitted text is covered by test_faulted_emission.py.
"""

import warnings

import numpy as np
import pytest

from silk.faults.application import (
    _AddedElement,
    _DeviceInstance,
    _FaultRealization,
    _realize_fault,
)
from silk.faults.types import Fault


@pytest.fixture
def nmos(pdk):
    def make(name='mn_tail', d='vtail', g='clkb', s='gnd', b='gnd',
             m=1, w=4.0, l=0.18, device_type='nmos1p8'):
        return getattr(pdk, device_type)(name, d=d, g=g, s=s, b=b,
                                         m=m, w=w, l=l)
    return make


def _fault(ft, **kw):
    return Fault(target='xcomp1.mn_tail', fault_type=ft, params=kw)


def _realize(device, fault, seed=0, pdk=None):
    return _realize_fault(device, fault, np.random.default_rng(seed), pdk=pdk)


# --- stuck_open ---

def test_stuck_open_m1_deletes_instance(nmos):
    r = _realize(nmos(m=1), _fault('stuck_open', per_finger=True))
    assert isinstance(r, _FaultRealization)
    assert r.instances == []
    assert r.added == []
    assert any('FAULT: stuck_open' in c for c in r.comments)
    assert r.applied[0].realized_params['m_before'] == 1
    assert r.applied[0].realized_params['deleted'] is True


def test_stuck_open_m4_decrements_m_to_3(nmos):
    r = _realize(nmos(m=4), _fault('stuck_open', per_finger=True))
    assert r.instances == [_DeviceInstance(params={'m': 3})]
    assert r.applied[0].realized_params['m_after'] == 3
    assert r.applied[0].realized_params['deleted'] is False


def test_stuck_open_per_finger_false_whole_device(nmos):
    r = _realize(nmos(m=4), _fault('stuck_open', per_finger=False))
    assert r.instances == []
    assert r.applied[0].realized_params['deleted'] is True


# --- shorts ---

def test_short_ds_m1_keeps_device_adds_shunt(nmos):
    r = _realize(nmos(m=1, d='vtail', s='gnd'),
                 _fault('short_ds', per_finger=True))
    assert r.instances == [_DeviceInstance()]
    assert r.added == [_AddedElement(
        kind='resistor', name='__silk_fault_short_78636f6d70312e6d6e5f7461696c',
        nets={'p': 'vtail', 'n': 'gnd'}, value=1e-6)]
    assert r.applied[0].realized_params['shunt_ohms'] == 1e-6


def test_short_ds_m4_decomposes(nmos):
    r = _realize(nmos(m=4), _fault('short_ds', per_finger=True))
    assert r.instances == [
        _DeviceInstance(params={'m': 3}),
        _DeviceInstance(name='__silk_fault_bad_78636f6d70312e6d6e5f7461696c', params={'m': 1}),
    ]
    assert len(r.added) == 1


def test_short_gs_puts_shunt_between_g_and_s(nmos):
    r = _realize(nmos(g='clkb', s='gnd'), _fault('short_gs', per_finger=True))
    assert r.added[0].nets == {'p': 'clkb', 'n': 'gnd'}


def test_short_gd_puts_shunt_between_g_and_d(nmos):
    r = _realize(nmos(g='clkb', d='vtail'), _fault('short_gd', per_finger=True))
    assert r.added[0].nets == {'p': 'clkb', 'n': 'vtail'}


# --- w_variation / l_variation ---

def test_w_variation_overrides_w(nmos, pdk):
    r = _realize(nmos(w=4.0), _fault('w_variation', sigma=0.02), seed=42,
                 pdk=pdk)
    rp = r.applied[0].realized_params
    assert rp['w_nominal'] == 4.0
    assert abs(rp['w_multiplier'] - 1.0) > 0
    # sky130 user units are its PDK units: the override equals w_new.
    assert r.instances == [_DeviceInstance(params={'w': rp['w_new']})]


def test_l_variation_overrides_l(nmos, pdk):
    r = _realize(nmos(l=0.18), _fault('l_variation', sigma=0.05), seed=42,
                 pdk=pdk)
    rp = r.applied[0].realized_params
    assert rp['l_nominal'] == 0.18
    assert set(r.instances[0].params) == {'l'}


def test_w_variation_override_is_in_pdk_units_on_a_metre_native_pdk():
    """GF180MCU W/L are emitted in metres; the circuit is written in
    microns. The override is in the PDK's unit, the manifest in the user's."""
    from silk import load_pdk
    pdk = load_pdk('gf180mcu')
    mn = pdk.nmos3p3('mn', d='d', g='g', s='s', b='s', w=4.0, l=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        r = _realize(mn, _fault('w_variation', sigma=0.02), seed=42, pdk=pdk)
    rp = r.applied[0].realized_params
    assert rp['w_nominal'] == 4.0
    assert rp['w_new'] == pytest.approx(4.0 * rp['w_multiplier'])
    assert r.instances[0].params['w'] == pytest.approx(rp['w_new'] * 1e-6)


# --- vth_shift ---

def test_vth_shift_rewires_gate_and_adds_vcvs_on_dc_offset(nmos):
    r = _realize(nmos(g='vp_orig'), _fault('vth_shift', delta=0.03, sigma=0.0),
                 seed=42)
    assert r.instances == [_DeviceInstance(
        nets={'g': '__silk_fault_shifted_78636f6d70312e6d6e5f7461696c'})]
    vcvs, vdc = r.added
    assert vcvs == _AddedElement(
        kind='vcvs', name='__silk_fault_vt_78636f6d70312e6d6e5f7461696c',
        nets={'out_pos': '__silk_fault_shifted_78636f6d70312e6d6e5f7461696c',
              'out_neg': '__silk_fault_offset_78636f6d70312e6d6e5f7461696c',
              'ctrl_pos': 'vp_orig', 'ctrl_neg': '0'},
        value=1)
    assert vdc == _AddedElement(
        kind='vdc', name='__silk_fault_vt_78636f6d70312e6d6e5f7461696c',
        nets={'positive': '__silk_fault_offset_78636f6d70312e6d6e5f7461696c', 'negative': '0'},
        value=0.03)


def test_vth_shift_delta_total_captured(nmos):
    r = _realize(nmos(g='vp'), _fault('vth_shift', delta=0.03, sigma=0.005),
                 seed=42)
    rp = r.applied[0].realized_params
    assert rp['delta_declared'] == 0.03
    assert abs(rp['delta_jitter']) < 0.02
    assert abs(rp['delta_total'] - (rp['delta_declared'] + rp['delta_jitter'])) < 1e-9
    assert r.added[1].value == rp['delta_total']


# --- the device is read, never changed ---

def test_realizing_never_mutates_the_device(nmos, pdk):
    mn = nmos(m=4)
    before = (dict(mn.params), dict(mn.nets), set(mn._caller_supplied))
    for ft, kw in [('stuck_open', {}), ('short_gd', {}),
                   ('w_variation', {'sigma': 0.1}),
                   ('vth_shift', {'delta': 0.1})]:
        _realize(mn, _fault(ft, **kw), pdk=pdk)
    assert (dict(mn.params), dict(mn.nets), set(mn._caller_supplied)) == before


def test_unknown_fault_type_raises(nmos):
    with pytest.raises(ValueError, match='Unknown fault_type'):
        _realize(nmos(), _fault('melted'))


# --- determinism ---

def test_w_variation_deterministic_same_seed(nmos, pdk):
    a = _realize(nmos(w=4.0), _fault('w_variation', sigma=0.02), seed=42, pdk=pdk)
    b = _realize(nmos(w=4.0), _fault('w_variation', sigma=0.02), seed=42, pdk=pdk)
    assert a.instances == b.instances
    assert a.applied[0].realized_params == b.applied[0].realized_params


# --- w_variation / l_variation resolve fault_params through the pdk (TASK-24) ---

def test_w_variation_clamped_via_pdk(nmos, pdk):
    """A real PDK clamps the perturbation to that PDK's own fault_params
    bounds — no hardcoded catalog path."""
    r = _realize(nmos(w=0.42), _fault('w_variation', sigma=10000.0), seed=42,
                 pdk=pdk)
    rp = r.applied[0].realized_params
    assert rp['clamped'] is True
    assert 0.42 <= rp['w_new'] <= 100.0


def test_w_variation_no_pdk_warns_unclamped(nmos):
    """No PDK given still warns and applies an unclamped perturbation;
    never a silent no-op."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = _realize(nmos(w=4.0), _fault('w_variation', sigma=0.02), seed=42)
    assert r.applied[0].realized_params['clamped'] is False
    assert any('fault_params' in str(x.message).lower() for x in w)


def test_w_variation_pdk_device_with_no_fault_params_warns_unclamped(pdk):
    """A device the given PDK carries but that declares no fault_params:
    never receives another device's bounds — it warns and applies an
    unclamped perturbation, exactly like no PDK at all."""
    from silk import load_pdk
    gf = load_pdk('gf180mcu')
    mn = gf.nmos3p3('mn', d='d', g='g', s='s', b='s', w=4.0, l=1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = _realize(mn, _fault('w_variation', sigma=0.02), seed=42, pdk=gf)
    assert r.applied[0].realized_params['clamped'] is False
    assert any('fault_params' in str(x.message).lower() for x in w)


def test_w_variation_device_not_on_the_given_pdk_warns_unclamped(pdk):
    """A device type that is not one of the given PDK's own devices (built
    on a different PDK) never receives sky130 (or any other) bounds by
    accident."""
    from silk import load_pdk
    mn = load_pdk('sg13g2').nmos_lv('mn', d='d', g='g', s='s', b='s',
                                    w=4.0, l=1.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r = _realize(mn, _fault('w_variation', sigma=0.02), seed=42, pdk=pdk)
    assert r.applied[0].realized_params['clamped'] is False
    assert any('fault_params' in str(x.message).lower() for x in w)
