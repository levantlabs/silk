"""STR-001 honours the SIGN of a junction pair against a signed rating range
(TASK-91).

Before TASK-91 a catalog rating was one magnitude per pair and the rule
compared ``|V| > m``. sky130's published NMOS body range is ``VBS = +0.3 to
-1.95 V`` (device-details/nfet_01v8/index.rst line 14) — asymmetric — so
TASK-48 could only store the reverse-bias side (1.95) and a body
forward-biased by +1.0 V (0.7 V past the foundry's forward limit) was
silently in range. The catalog now carries ``Vbs_max: {min: -1.95, max:
0.3}`` for nfet_01v8 / nfet_01v8_lvt, the loader normalises every rating to
a ``(lo, hi)`` pair, and both STR-001 data paths score the signed voltage.

The sign convention is stated once (``JUNCTION_PAIR_SIGN_CONVENTION``):
``Vxy = V(x) - V(y)``, the order the pair tables list the pins in.
"""
import types

import numpy as np
import pytest

from silk.circuit.modules import Module
from silk.lint.rules.stress import (
    JunctionOverrangeRule, _MOSFET_PAIRS, _DIODE_PAIRS, _BJT_PAIRS,
    _compute_junction_pairs, _score_pair,
)
from silk.pdk.basepdk import (
    JUNCTION_PAIR_SIGN_CONVENTION, PDKDevice, parse_junction_rating,
)
from silk import sources

# The two sides of the sky130 NMOS body range, straight from the SkyWater
# page the catalog cites ("VBS = +0.3 to -1.95V"). +1.0 V is inside the old
# sign-blind 1.95 V magnitude but 0.7 V past the forward limit; -1.5 V is a
# perfectly ordinary reverse bias inside both.
_VBS_FWD_LIMIT_V = 0.3
_VBS_REV_LIMIT_V = -1.95
_VBS_FORWARD_BAD_V = 1.0
_VBS_REVERSE_OK_V = -1.5

# The PMOS mirror image (TASK-107): pfet_01v8/index.rst line 14 "VBS = -0.1 to
# +1.95V". In Vbs = V(b) - V(s) the n-well body normally sits ABOVE the
# source, so +1.95 is the reverse-bias end and -0.1 the forward one.
_PMOS_VBS_FWD_LIMIT_V = -0.1
_PMOS_VBS_REV_LIMIT_V = 1.95


# ---------------------------------------------------------------------------
# The sign convention, pinned
# ---------------------------------------------------------------------------

def test_sign_convention_is_first_pin_minus_second_pin():
    """``Vxy = V(x) - V(y)``: the pair tables list ``(x, y, 'Vxy')`` and
    ``_compute_junction_pairs`` subtracts in that order, so ``Vbs`` with the
    body 1 V ABOVE the source is +1.0 (forward bias), never -1.0. A signed
    catalog range is read in this same convention; if either side ever
    flipped, ``{min: -1.95, max: 0.3}`` would silently rate the wrong
    junction direction."""
    assert JUNCTION_PAIR_SIGN_CONVENTION == 'Vxy = V(x) - V(y)'
    for pin_a, pin_b, name in _MOSFET_PAIRS + _DIODE_PAIRS + _BJT_PAIRS:
        assert name == f'V{pin_a}{pin_b}', (pin_a, pin_b, name)
    volts = {'d': 1.8, 'g': 0.9, 's': 0.0, 'b': 1.0}
    pairs = _compute_junction_pairs({p: p for p in volts},
                                    lambda p: np.array([volts[p]]))
    assert pairs['Vbs'][0] == pytest.approx(+1.0)
    assert pairs['Vgs'][0] == pytest.approx(+0.9)
    assert pairs['Vgd'][0] == pytest.approx(-0.9)
    assert pairs['Vbd'][0] == pytest.approx(-0.8)


# ---------------------------------------------------------------------------
# The two catalog forms, normalised by ONE parser
# ---------------------------------------------------------------------------

def test_magnitude_shorthand_expands_to_symmetric_range():
    assert parse_junction_rating('Vgs_max', 1.95) == (-1.95, 1.95)
    assert parse_junction_rating('Vgs_max', 5) == (-5.0, 5.0)


def test_signed_range_form_is_kept_as_written():
    assert parse_junction_rating('Vbs_max', {'min': -1.95, 'max': 0.3}) == (-1.95, 0.3)


def test_signed_range_form_must_have_exactly_min_and_max():
    with pytest.raises(ValueError, match="missing \\['max'\\]"):
        parse_junction_rating('Vbs_max', {'min': -1.95})
    with pytest.raises(ValueError, match="unexpected \\['typ'\\]"):
        parse_junction_rating('Vbs_max', {'min': -1.95, 'max': 0.3, 'typ': 0})
    with pytest.raises(ValueError):
        parse_junction_rating('Vbs_max', {'min': 'low', 'max': 0.3})


def test_parser_does_not_judge_min_greater_than_max():
    """An inverted range is a catalog-convention error for
    ``silk.pdk.validate`` to report with the device name; the parser keeps
    it so the whole PDK does not fail to load over one entry."""
    assert parse_junction_rating('Vbs_max', {'min': 0.3, 'max': -1.95}) == (0.3, -1.95)


def test_pdkdevice_carries_both_the_range_and_the_sign_blind_magnitude():
    """``junction_ranges`` is the normalised pair STR-001 scores against;
    ``junction_ratings`` stays the magnitude view (``max(|lo|, |hi|)``, so
    1.95 for the NMOS body — unchanged from TASK-48) for the sign-blind
    consumers that still read it."""
    dev = PDKDevice.from_dict('m', {
        'PDKDevice': 'x', 'paramList': {}, 'pinList': {'d': 'd', 'g': 'g', 's': 's', 'b': 'b'},
        'junction_ratings': {'Vgs_max': 1.95, 'Vbs_max': {'min': -1.95, 'max': 0.3}},
    })
    assert dev.junction_ranges == {'Vgs_max': (-1.95, 1.95), 'Vbs_max': (-1.95, 0.3)}
    assert dev.junction_ratings == {'Vgs_max': 1.95, 'Vbs_max': 1.95}


def test_shipped_sky130_nmos_body_range_is_signed_and_nothing_else_moved(pdk):
    """TASK-91 converted the two 1.8 V NMOS entries TASK-48 documented as
    asymmetric (+0.3 to -1.95 V); TASK-107 finished the citation pass with
    the native 5 V NMOS (+0.3 to -5.5 V) and the three 1.8 V PMOS entries
    (-0.1 to +1.95 V). Every other sky130 rating is still the symmetric
    magnitude form."""
    signed = {n: k for n, d in pdk._devices.items()
              for k, (lo, hi) in d.junction_ranges.items() if lo != -hi}
    assert signed == {n: 'Vbs_max' for n in
                      ('nmos1p8', 'nmos1p8_lvt', 'nmos5v_native',
                       'pmos1p8', 'pmos1p8_lvt', 'pmos1p8_hvt')}
    for n in ('nmos1p8', 'nmos1p8_lvt'):
        assert pdk._devices[n].junction_ranges['Vbs_max'] == (_VBS_REV_LIMIT_V, _VBS_FWD_LIMIT_V)
    assert pdk._devices['nmos5v_native'].junction_ranges['Vbs_max'] == (-5.5, _VBS_FWD_LIMIT_V)
    for n in ('pmos1p8', 'pmos1p8_lvt', 'pmos1p8_hvt'):
        assert pdk._devices[n].junction_ranges['Vbs_max'] == (_PMOS_VBS_FWD_LIMIT_V, _PMOS_VBS_REV_LIMIT_V)


# ---------------------------------------------------------------------------
# _score_pair: the one comparison both data paths use
# ---------------------------------------------------------------------------

def test_symmetric_range_scores_exactly_as_the_old_magnitude_comparison():
    """Magnitude form ``m`` == range ``(-m, m)``: the worst sample is
    ``argmax(|v|)`` (first of ties), ``measured_v`` is ``|v|`` and the bound
    is ``m`` — byte-for-byte what STR-001 produced before TASK-91, including
    for a NEGATIVE excursion (a PMOS ``Vgs`` of -2.5 V reports 2.500 V
    against ``Vgs_max``)."""
    v = np.array([0.5, -2.5, 2.5, -1.0])
    assert _score_pair(v, -1.95, 1.95) == (1, 2.5, 1.95, 'exceeds')
    assert _score_pair(np.array([1.0, -1.9]), -1.95, 1.95) is None
    assert _score_pair(np.array([1.95]), -1.95, 1.95) is None       # no margin, > not >=


def test_asymmetric_range_reports_the_signed_voltage_and_the_bound_crossed():
    """Pre-fix: ``|+1.0| < 1.95`` -> silent. Now +1.0 V is 0.7 V past the
    0.3 V forward limit: the finding carries the signed +1.0 and the 0.3
    bound (not 1.95), so the reader sees WHICH limit was broken."""
    v = np.array([0.0, _VBS_FORWARD_BAD_V, -1.5])
    assert _score_pair(v, _VBS_REV_LIMIT_V, _VBS_FWD_LIMIT_V) == (1, 1.0, 0.3, 'exceeds')


def test_asymmetric_range_lower_bound_reports_below_with_the_negative_limit():
    v = np.array([0.0, 0.2, -2.5])
    assert _score_pair(v, _VBS_REV_LIMIT_V, _VBS_FWD_LIMIT_V) == (2, -2.5, -1.95, 'below')


def test_asymmetric_range_worst_sample_is_the_largest_excursion_not_largest_magnitude():
    """-1.9 V is the larger |V| but is in range; +0.5 V is the violation.
    A sign-blind argmax(|v|) would have looked at the wrong sample."""
    v = np.array([-1.9, 0.5, -1.0])
    assert _score_pair(v, _VBS_REV_LIMIT_V, _VBS_FWD_LIMIT_V) == (1, 0.5, 0.3, 'exceeds')


def test_asymmetric_range_in_range_is_silent():
    v = np.array([_VBS_REVERSE_OK_V, 0.0, 0.3, -1.95])
    assert _score_pair(v, _VBS_REV_LIMIT_V, _VBS_FWD_LIMIT_V) is None


# ---------------------------------------------------------------------------
# sky130 nfet_01v8 end-to-end on both STR-001 data paths (AC #2)
# ---------------------------------------------------------------------------

def _nmos_with_own_body(pdk, tmp_path):
    """A real sky130 nmos1p8 whose body is its own net ``vb`` (not tied to
    the source), so ``Vbs`` is the quantity under test."""
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    with m:
        pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='vb', w=1.0, l=0.15)
    return m


def _behavioral(module, t, voltages):
    module._behavioral_mode = True
    module._behavioral_t = t
    for net_name, wave in voltages.items():
        if net_name not in module.net_dict:
            module.add_net(net_name)
        module.net_dict[net_name].voltage = wave
    return module


def _vbs_findings(module):
    return [v for v in JunctionOverrangeRule().check(module)
            if v.location.instance_path == 'mn' and v.location.terminal == 'Vbs']


def _waves(n, vbs):
    return {'in': np.full(n, 0.9), 'out': np.zeros(n),
            'gnd': np.zeros(n), 'vb': np.full(n, vbs)}


def test_behavioral_sky130_nfet_forward_body_bias_of_1v_is_over_range(pdk, tmp_path):
    """AC #2. Pre-TASK-91 this module produced NO STR-001 finding: the
    catalog stored ``Vbs_max: 1.95`` and ``|+1.0| < 1.95``. SkyWater rates
    the NMOS body at +0.3 V forward, so a +1.0 V forward bias is 0.7 V
    over-range and is now reported with the signed voltage and the forward
    limit it crossed."""
    n = 20
    t = np.linspace(0, 1e-9, n)
    m = _behavioral(_nmos_with_own_body(pdk, tmp_path), t, _waves(n, _VBS_FORWARD_BAD_V))
    hits = _vbs_findings(m)
    assert len(hits) == 1, [v.message for v in JunctionOverrangeRule().check(m)]
    v = hits[0]
    assert v.device_type == 'nmos1p8'
    assert v.measured_v == pytest.approx(+1.0)
    assert v.rated_max == pytest.approx(_VBS_FWD_LIMIT_V)
    assert v.location.terminal_pair == ('b', 's')
    assert v.message.startswith('mn (nmos1p8): Vbs=1.000V exceeds Vbs_max=0.3V at t=')
    # nothing else on the device is stressed
    assert [x.location.terminal for x in JunctionOverrangeRule().check(m)] == ['Vbs']


def test_behavioral_sky130_nfet_reverse_body_bias_of_1p5v_is_in_range(pdk, tmp_path):
    """AC #2. -1.5 V is inside the -1.95 V reverse limit (and was inside the
    old 1.95 magnitude too): silent before and after."""
    n = 20
    m = _behavioral(_nmos_with_own_body(pdk, tmp_path), np.linspace(0, 1e-9, n),
                    _waves(n, _VBS_REVERSE_OK_V))
    assert list(JunctionOverrangeRule().check(m)) == []


def test_behavioral_sky130_nfet_reverse_body_bias_past_1p95v_reports_below_min(pdk, tmp_path):
    """The reverse side still fires (it did before, as ``|V|`` against
    1.95); the finding now says ``below Vbs_min=-1.95V`` with the signed
    value instead of ``exceeds Vbs_max=1.95V`` with its magnitude."""
    n = 20
    m = _behavioral(_nmos_with_own_body(pdk, tmp_path), np.linspace(0, 1e-9, n),
                    _waves(n, -2.5))
    hits = _vbs_findings(m)
    assert len(hits) == 1
    assert hits[0].measured_v == pytest.approx(-2.5)
    assert hits[0].rated_max == pytest.approx(_VBS_REV_LIMIT_V)
    assert 'Vbs=-2.500V below Vbs_min=-1.95V' in hits[0].message


def _spice_module(pdk, tmp_path, waveforms, n=20):
    """The same device stood up as a transient SPICE run: waveforms served
    through ``load_signal`` and a rawfile stub shaped like the part STR-001
    classifies off (``current_plot.vectors[0].name``, ``.real``)."""
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    with m:
        pdk.nmos1p8('mn', d='out', g='in', s='0', b='vb', w=1.0, l=0.15)
        sources.VoltageSource('vb', positive='vb', negative='0', dc=0.0)
    t = np.linspace(0, 1e-9, n)

    def load_signal(name, kind='V'):
        if name not in waveforms:
            raise KeyError(name)
        return types.SimpleNamespace(x=t, y=np.asarray(waveforms[name], dtype=float),
                                     xlabel='time')

    plot = types.SimpleNamespace(vectors=[types.SimpleNamespace(name='time')], real=True)
    m.sim_results_loaded = True
    m.simResultsAll = types.SimpleNamespace(current_plot=plot)
    m.load_signal = load_signal
    return m


def test_spice_path_sky130_nfet_forward_body_bias_of_1v_is_over_range(pdk, tmp_path):
    """AC #2 on the SPICE data path (source on the literal ground node ``0``,
    read as 0 V). Pre-TASK-91: silent, ``|+1.0| < 1.95``."""
    n = 20
    m = _spice_module(pdk, tmp_path, {'in': np.full(n, 0.9), 'out': np.zeros(n),
                                      'vb': np.full(n, _VBS_FORWARD_BAD_V)}, n)
    hits = _vbs_findings(m)
    assert len(hits) == 1, [v.message for v in JunctionOverrangeRule().check(m)]
    assert hits[0].measured_v == pytest.approx(+1.0)
    assert hits[0].rated_max == pytest.approx(_VBS_FWD_LIMIT_V)
    assert hits[0].message.startswith('mn (nmos1p8): Vbs=1.000V exceeds Vbs_max=0.3V at t=')


def test_spice_path_sky130_nfet_reverse_body_bias_of_1p5v_is_in_range(pdk, tmp_path):
    n = 20
    m = _spice_module(pdk, tmp_path, {'in': np.full(n, 0.9), 'out': np.zeros(n),
                                      'vb': np.full(n, _VBS_REVERSE_OK_V)}, n)
    assert list(JunctionOverrangeRule().check(m)) == []


def test_magnitude_form_negative_excursion_message_is_unchanged(pdk, tmp_path):
    """AC #3 (byte-identical magnitude form): a PMOS-style negative gate
    excursion on a magnitude-rated pair still reports ``|V|`` against
    ``<pair>_max`` exactly as before — ``Vgs=2.500V exceeds Vgs_max=1.95V``
    for a gate 2.5 V BELOW the source — because a symmetric range carries
    no sign information to report."""
    n = 20
    m = _spice_module(pdk, tmp_path, {'in': np.full(n, -2.5), 'out': np.full(n, 0.0),
                                      'vb': np.zeros(n)}, n)
    vgs = [v for v in JunctionOverrangeRule().check(m) if v.location.terminal == 'Vgs']
    assert len(vgs) == 1
    assert vgs[0].measured_v == pytest.approx(2.5)
    assert vgs[0].rated_max == pytest.approx(1.95)
    assert vgs[0].message.startswith('mn (nmos1p8): Vgs=2.500V exceeds Vgs_max=1.95V at t=')


def test_device_carrying_only_magnitude_ratings_is_scored_through_the_same_parser(tmp_path):
    """A device object with ``junction_ratings`` alone (the pre-TASK-91
    shape: hand-built fixtures, mocks, third-party PDK objects) is expanded
    by ``parse_junction_rating`` at check time, so it scores exactly like a
    catalog device whose loader already built ``junction_ranges``."""
    from unittest.mock import MagicMock
    pdk = MagicMock()
    pdk.model_names = ['m']
    dev = MagicMock()
    dev.junction_ratings = {'Vgs_max': 1.95}
    dev.pin_list = {'d': MagicMock(net='out'), 'g': MagicMock(net='in'),
                    's': MagicMock(net='gnd'), 'b': MagicMock(net='gnd')}
    pdk._devices = {'m': dev}
    n = 10
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    _behavioral(m, np.linspace(0, 1e-9, n),
                {'in': np.full(n, -2.5), 'out': np.zeros(n), 'gnd': np.zeros(n)})
    m._element_mgr.elem_dict['mn'] = {'type': 'device', 'typeName': 'm', 'device': dev}
    hits = list(JunctionOverrangeRule().check(m))
    assert [(v.location.terminal, v.measured_v, v.rated_max) for v in hits] == [('Vgs', 2.5, 1.95)]


# ---------------------------------------------------------------------------
# TASK-107: the remaining asymmetric sky130 entries, one STR-001 test each
# ---------------------------------------------------------------------------

def _mos_with_own_body(pdk, tmp_path, device, **geom):
    """A real sky130 device of type ``device`` whose body is its own net
    ``vb`` and whose source is its own net ``vs``, so ``Vbs`` is the
    quantity under test for either polarity."""
    m = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    with m:
        getattr(pdk, device)('m1', d='out', g='in', s='vs', b='vb', **geom)
    return m


def _body_waves(n, vs, vb):
    return {'in': np.full(n, vs), 'out': np.full(n, vs),
            'vs': np.full(n, vs), 'vb': np.full(n, vb)}


def _body_findings(module):
    return [v for v in JunctionOverrangeRule().check(module)
            if v.location.instance_path == 'm1' and v.location.terminal == 'Vbs']


def test_behavioral_sky130_native_5v_nfet_forward_body_bias_of_1v_is_over_range(pdk, tmp_path):
    """nmos5v_native (sky130_fd_pr__nfet_05v0_nvt). Pre-TASK-107 this
    module produced NO STR-001 finding: the catalog stored ``Vbs_max: 5.5``
    (its source text even said the forward limit was "not representable")
    and ``|+1.0| < 5.5``. nfet_05v0_nvt/index.rst line 14 rates the body
    at +0.3 V forward, so a +1.0 V forward bias is 0.7 V over-range and is
    now reported with the signed value and the forward limit it crossed;
    the other four pairs sit at 0 V and stay silent."""
    n = 20
    m = _behavioral(_mos_with_own_body(pdk, tmp_path, 'nmos5v_native', w=1.0, l=0.9),
                    np.linspace(0, 1e-9, n), _body_waves(n, vs=0.0, vb=+1.0))
    hits = _body_findings(m)
    assert len(hits) == 1, [v.message for v in JunctionOverrangeRule().check(m)]
    assert hits[0].device_type == 'nmos5v_native'
    assert hits[0].measured_v == pytest.approx(+1.0)
    assert hits[0].rated_max == pytest.approx(_VBS_FWD_LIMIT_V)
    assert hits[0].message.startswith('m1 (nmos5v_native): Vbs=1.000V exceeds Vbs_max=0.3V at t=')
    assert [x.location.terminal for x in JunctionOverrangeRule().check(m)] == ['Vbs']
    # a -5.0 V reverse bias is inside the -5.5 V end: silent before and after
    m = _behavioral(_mos_with_own_body(pdk, tmp_path, 'nmos5v_native', w=1.0, l=0.9),
                    np.linspace(0, 1e-9, n), _body_waves(n, vs=5.0, vb=0.0))
    assert _body_findings(m) == []


@pytest.mark.parametrize('device', ['pmos1p8', 'pmos1p8_lvt', 'pmos1p8_hvt'])
def test_behavioral_sky130_1v8_pfet_body_1v_below_source_is_over_range(pdk, tmp_path, device):
    """pfet_01v8 / _lvt / _hvt. Pre-TASK-107 this module produced NO STR-001
    finding: the catalog stored ``Vbs_max: 1.95`` and ``|-1.0| < 1.95``.
    pfet_01v8*/index.rst line 14 rates the body at ``VBS = -0.1 to +1.95V``
    — a PMOS n-well may sit 1.95 V ABOVE its source (reverse) but only
    0.1 V BELOW it (forward) — so a body 1.0 V below the source (``Vbs =
    V(b) - V(s) = -1.0``, a forward-conducting source-well diode) is 0.9 V
    over-range and is now reported as ``below Vbs_min=-0.1V`` with the
    signed value. The mirror case, body 1.5 V above the source, is the
    normal reverse bias and stays silent (as it was before)."""
    n = 20
    t = np.linspace(0, 1e-9, n)
    m = _behavioral(_mos_with_own_body(pdk, tmp_path, device, w=1.0, l=0.35),
                    t, _body_waves(n, vs=1.8, vb=0.8))
    hits = _body_findings(m)
    assert len(hits) == 1, [v.message for v in JunctionOverrangeRule().check(m)]
    assert hits[0].device_type == device
    assert hits[0].measured_v == pytest.approx(-1.0)
    assert hits[0].rated_max == pytest.approx(_PMOS_VBS_FWD_LIMIT_V)
    assert hits[0].message.startswith(f'm1 ({device}): Vbs=-1.000V below Vbs_min=-0.1V at t=')
    assert [x.location.terminal for x in JunctionOverrangeRule().check(m)] == ['Vbs']
    # body 1.5 V above the source: ordinary PMOS reverse bias, in range
    m = _behavioral(_mos_with_own_body(pdk, tmp_path, device, w=1.0, l=0.35),
                    t, _body_waves(n, vs=0.3, vb=1.8))
    assert list(JunctionOverrangeRule().check(m)) == []
