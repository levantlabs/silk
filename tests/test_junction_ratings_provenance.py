"""Every junction_ratings value in the three shipped catalogs is sourced, and
the values the TASK-48 audit corrected read what the foundry publishes.

Runs on the YAML alone (no PDK tree, no ngspice) so a bare checkout enforces
the provenance contract; the with-PDK static guard
(test_pdk_validate.test_real_pdk_has_no_validation_issues) covers the same
check through load_pdk.
"""
import glob
import os

import pytest
import yaml

from silk.pdk.basepdk import PDKDevice
from silk.pdk.validate import check_junction_provenance

_DATA = os.path.join(os.path.dirname(__file__), '..', 'silk', 'pdk', 'data')
_CATALOGS = sorted(glob.glob(os.path.join(_DATA, '*_device_details.yaml')))


class _Catalog:
    def __init__(self, path):
        with open(path) as fh:
            self._devices = {n: PDKDevice.from_dict(n, d)
                             for n, d in yaml.safe_load(fh).items()}


def _load(name):
    return _Catalog(os.path.join(_DATA, name))._devices


@pytest.mark.parametrize('path', _CATALOGS, ids=os.path.basename)
def test_every_shipped_rating_is_sourced_or_tagged_heuristic(path):
    """AC #1. Before TASK-48 none of the 36 rated devices carried a source
    (23 sky130, 9 gf180mcu, 4 sg13g2), so this returned 36 unsourced_rating
    issues. Each now cites the PDK document / model parameter, or states the
    heuristic rule its value was derived by, and every heuristic rule
    re-checks against the numbers."""
    cat = _Catalog(path)
    rated = [n for n, d in cat._devices.items() if d.junction_ratings]
    assert rated, f'{path} declares no rated device — the guard is vacuous'
    issues = check_junction_provenance(cat)
    assert issues == [], [f'{i.device}:{i.kind}:{i.detail}' for i in issues]
    for n in rated:
        prov = cat._devices[n].junction_ratings_provenance
        assert prov.source, f'{n}: every shipped device cites a document'


def test_sky130_1v8_devices_read_the_documented_1p95_v():
    """Pre-fix 1.98 V = 1.1 x 1.8 (unsourced). SkyWater's device-details pages
    state VDS/VGS 0 to 1.95 V for nfet/pfet_01v8 (+lvt/hvt). The magnitude
    view is unchanged by TASK-91 (the NMOS body's signed range still admits
    |V| up to 1.95)."""
    d = _load('skywater_130_device_details.yaml')
    for n in ('nmos1p8', 'nmos1p8_lvt', 'pmos1p8', 'pmos1p8_lvt', 'pmos1p8_hvt'):
        assert d[n].junction_ratings == {k: 1.95 for k in
                                         ('Vgs_max', 'Vds_max', 'Vgd_max', 'Vbs_max', 'Vbd_max')}, n


# Every sky130 entry whose SkyWater page states an asymmetric body range
# (a non-zero limit of the OPPOSITE sign to the reverse-bias one), and the
# range the page states, read in ``Vbs = V(b) - V(s)``. The pages' one-sided
# "VBS = 0 to -X V" / "0 to +X V" forms (nfet_03v3_nvt, the g5v0d10v5 family,
# every 20 V device) are SkyWater's magnitude form and stay bare numbers.
_SKY130_SIGNED = {
    'nmos1p8': (-1.95, 0.3),         # nfet_01v8/index.rst:14  VBS = +0.3 to -1.95V
    'nmos1p8_lvt': (-1.95, 0.3),     # nfet_01v8_lvt/index.rst:14  same
    'nmos5v_native': (-5.5, 0.3),    # nfet_05v0_nvt/index.rst:14  VBS = +0.3 to -5.5V
    'pmos1p8': (-0.1, 1.95),         # pfet_01v8/index.rst:14  VBS = -0.1 to +1.95V
    'pmos1p8_lvt': (-0.1, 1.95),     # pfet_01v8_lvt/index.rst:14  same
    'pmos1p8_hvt': (-0.1, 1.95),     # pfet_01v8_hvt/index.rst:14  same
}


def test_sky130_nmos_body_is_the_signed_range_the_page_publishes():
    """TASK-91. nfet_01v8/index.rst line 14: ``VBS = +0.3 to -1.95V``. Before
    this the catalog stored the magnitude 1.95 (TASK-48 noted the forward
    limit in ``source`` as unrepresentable), so ``junction_ranges`` read
    (-1.95, 1.95) and a +1.0 V forward body bias was in range. TASK-107
    finished the pass: exactly the six sky130 entries whose page states an
    asymmetric range are signed (``_SKY130_SIGNED``); every other shipped
    rating (all three catalogs) is still the symmetric magnitude form."""
    for path in _CATALOGS:
        signed = {(n, k) for n, dev in _Catalog(path)._devices.items()
                  for k, (lo, hi) in dev.junction_ranges.items() if lo != -hi}
        if os.path.basename(path) == 'skywater_130_device_details.yaml':
            assert signed == {(n, 'Vbs_max') for n in _SKY130_SIGNED}
        else:
            assert signed == set(), path
    d = _load('skywater_130_device_details.yaml')
    for n, (lo, hi) in _SKY130_SIGNED.items():
        assert d[n].junction_ranges['Vbs_max'] == (lo, hi), n
        assert f'{{min: {lo}, max: {hi}}}' in d[n].junction_ratings_provenance.source, n


def test_sky130_native_5v_nfet_body_is_the_signed_range_the_page_publishes():
    """TASK-107. nfet_05v0_nvt/index.rst line 14: ``VBS = +0.3 to -5.5V``
    (restated for this model on esd_nfet/index.rst line 14). Between TASK-48
    and this fix the catalog stored the magnitude 5.5 and its ``source``
    said the forward limit was "not representable", so a native-5V body
    forward-biased by up to 5.5 V read as in range. The magnitude view is
    unchanged (5.5)."""
    d = _load('skywater_130_device_details.yaml')
    dev = d['nmos5v_native']
    assert dev.junction_ranges['Vbs_max'] == (-5.5, 0.3)
    assert dev.junction_ratings['Vbs_max'] == 5.5
    assert 'not representable' not in dev.junction_ratings_provenance.source
    assert 'nfet_05v0_nvt/index.rst' in dev.junction_ratings_provenance.source


def test_sky130_1v8_pfet_body_is_the_signed_range_the_page_publishes():
    """TASK-107. pfet_01v8{,_lvt,_hvt}/index.rst line 14: ``VBS = -0.1 to
    +1.95V``. In ``Vbs = V(b) - V(s)`` a PMOS n-well body normally sits
    ABOVE its source, so +1.95 is the reverse-bias end and -0.1 the forward
    one — the mirror image of the NMOS entry, and the sign a reader must
    get right. Between TASK-48 and this fix the catalog stored the
    magnitude 1.95, so a PMOS body pulled up to 1.95 V BELOW its source
    (a forward-conducting source-well diode) read as in range. The
    magnitude view is unchanged (1.95)."""
    d = _load('skywater_130_device_details.yaml')
    for n in ('pmos1p8', 'pmos1p8_lvt', 'pmos1p8_hvt'):
        dev = d[n]
        assert dev.junction_ranges['Vbs_max'] == (-0.1, 1.95), n
        assert dev.junction_ratings['Vbs_max'] == 1.95, n
        assert 'not representable' not in dev.junction_ratings_provenance.source, n
        assert f'{n.replace("pmos1p8", "pfet_01v8")}/index.rst' in dev.junction_ratings_provenance.source, n


@pytest.mark.parametrize('path', _CATALOGS, ids=os.path.basename)
def test_no_heuristic_rule_claims_str001_is_sign_blind(path):
    """TASK-107. After TASK-91 sixteen sky130 ``Vgd_max`` rules still
    justified ``= Vds_max`` with "STR-001 compares |V| so one magnitude must
    cover both signs" — false since STR-001 scores the signed pair. The
    rules still OPEN with ``= Vds_max`` (the form ``silk.pdk.validate``
    re-checks) and no rule in any shipped catalog repeats the claim."""
    cat = _Catalog(path)
    for n, dev in cat._devices.items():
        heur = (dev.junction_ratings_provenance.heuristic or {}) if dev.junction_ratings else {}
        for key, text in heur.items():
            assert 'one magnitude must cover both signs' not in text, (n, key)
            assert 'compares |V|' not in text, (n, key)
            if key == 'Vgd_max' and os.path.basename(path).startswith('skywater'):
                assert text.startswith('= Vds_max'), (n, key)


def test_sky130_20v_devices_gate_is_a_5p5_v_terminal_not_22_v():
    """Pre-fix Vgs_max = Vbs_max = 22.0 on every 20 V device: a 22 V gate
    would be 4x the VHV gate-oxide limit (rules/hv.rst: 5.5 V). SkyWater:
    VGS 0 to 5.5 V, VBS 0 to -2.0 V, VDS 0 to 22 V."""
    d = _load('skywater_130_device_details.yaml')
    for n in ('nmos20v', 'nmos20v_native', 'nmos20v_zvt', 'nmos20v_iso', 'pmos20v'):
        r = d[n].junction_ratings
        assert (r['Vgs_max'], r['Vds_max'], r['Vbs_max']) == (5.5, 22.0, 2.0), n


def test_sky130_5v_devices_read_the_documented_5p5_and_11_v():
    """Pre-fix Vgs_max 6.0 (1.2 x 5, unsourced) and Vds_max 10.5 (the cell
    name). Documented: VGS 0 to 5.5 V, VDS 0 to 11.0 V, VBS 0 to -5.5 V; the
    ESD nfet variant is documented at VGS 5.0 V; nfet_05v0_nvt is a 5.5 V
    drain device (10.5 V was a copy of the g5v0d10v5 row)."""
    d = _load('skywater_130_device_details.yaml')
    for n in ('nmos5v', 'pmos5v', 'pmos_esd'):
        r = d[n].junction_ratings
        assert (r['Vgs_max'], r['Vds_max'], r['Vbs_max']) == (5.5, 11.0, 5.5), n
    r = d['nmos_esd'].junction_ratings
    assert (r['Vgs_max'], r['Vds_max'], r['Vbs_max']) == (5.0, 11.0, 5.5)
    r = d['nmos5v_native'].junction_ratings
    assert (r['Vgs_max'], r['Vds_max'], r['Vbs_max']) == (5.5, 5.5, 5.5)


def test_sky130_11v0_diodes_and_npn_11v_are_documented_at_5_v():
    """AC #2. The audit suspected diode_pwnd/diode_nw (11v0 parts) and npn_11v
    carried a copy-pasted 5.5 V. SkyWater documents ONE operating regime for
    the whole diode family (|Vd0-Vd1| 0 to 5.0 V, the _11v0 models listed)
    and one for npn_05v5 AND npn_11v0 (|VCE| 0 to 5.0 V) — so the 11v0 parts
    are correctly rated with the 5 V ones; the number moves 5.5 -> 5.0."""
    d = _load('skywater_130_device_details.yaml')
    for n in ('diode_std', 'diode_pwnd', 'diode_nw'):
        assert d[n].junction_ratings == {'Vak_max': 5.0}, n
    for n in ('npn_std', 'npn_11v', 'pnp_small', 'pnp_large'):
        assert d[n].junction_ratings['Vce_max'] == 5.0, n
        assert '11v0' in d['npn_11v'].junction_ratings_provenance.source


def test_sky130_bjt_vbe_is_the_documented_5_v_not_a_forward_drop():
    """AC #2. Pre-fix Vbe_max = Vbc_max = 0.7 V — a forward diode drop, so
    STR-001 fired on every forward-active BJT (bandgap_example runs pnp_small
    at ~0.65-0.75 V Vbe). SkyWater: |VBE| 0 to 5.0 V; Vbc is derived
    (= Vce_max) and says so."""
    d = _load('skywater_130_device_details.yaml')
    for n in ('npn_std', 'npn_11v', 'pnp_small', 'pnp_large'):
        r = d[n].junction_ratings
        assert r['Vbe_max'] == 5.0 and r['Vbc_max'] == 5.0, n
        assert d[n].junction_ratings_provenance.heuristic['Vbc_max'].startswith('= Vce_max')


def test_gf180mcu_ratings_match_the_design_manual_10_percent_rule():
    """Unchanged numbers, now sourced: GF180MCU DRM 14.1.4.2 gate-node max
    3.63 V (thin oxide); 14.1.2 nominal 3.3 V / 6 V + 10% supply overshoot
    tolerance -> 3.63 / 6.6 V."""
    d = _load('gf180mcu_device_details.yaml')
    for n in ('nmos3p3', 'pmos3p3', 'nmos3p3_dss', 'pmos3p3_dss'):
        assert set(d[n].junction_ratings.values()) == {3.63}, n
        assert '3.63V (for thin oxide)' in d[n].junction_ratings_provenance.source
    for n in ('nmos6p0', 'pmos6p0', 'nmos6p0_nvt', 'nmos6p0_dss', 'pmos6p0_dss'):
        assert set(d[n].junction_ratings.values()) == {6.6}, n


def test_sg13g2_hv_reads_the_spec_3p3_v_maximum_and_lv_the_1p65_v():
    """Pre-fix HV 3.63 V = 1.1 x 3.3 (unsourced). IHP's process spec headers
    say 'VGS <= 3,3V (Maximum)' for HV-NMOS/PMOS and 'VGS <= 1,65 V @ 125°C'
    for the LV devices; the other pairs are derived (= Vgs_max) and say so."""
    d = _load('sg13g2_device_details.yaml')
    for n in ('nmos_hv', 'pmos_hv'):
        assert set(d[n].junction_ratings.values()) == {3.3}, n
    for n in ('nmos_lv', 'pmos_lv'):
        assert set(d[n].junction_ratings.values()) == {1.65}, n
    for n in ('nmos_hv', 'pmos_hv', 'nmos_lv', 'pmos_lv'):
        h = d[n].junction_ratings_provenance.heuristic
        assert set(h) == {'Vds_max', 'Vgd_max', 'Vbs_max', 'Vbd_max'}, n
        assert all(v.startswith('= Vgs_max') for v in h.values()), n
