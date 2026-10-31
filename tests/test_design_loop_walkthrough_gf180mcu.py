"""Guards for ``silk/examples/design_loop_walkthrough_gf180mcu.py`` — the
flagship design loop run unchanged on a metre-native PDK (TASK-16, the
cross-PDK integration proof that no step still assumes sky130 microns).

Fast tests pin what needs no simulator: the intent is a pure restatement, the
``Technology`` record is the only thing that differs from the flagship, and —
with GF180MCU installed — the geometry story: the emitted netlist carries
metres, the single accessor reads canonical microns, the estimator's areas
equal ``W·L·M × packing`` computed by hand, the factory and ``add_element``
styles agree, and a sky130 device with the same W/L in µm has the same
footprint. The slow test executes the documented command in process and
checks every stage's numbers, provenance and artifacts.
"""
import os
import subprocess
import sys

import pytest

from tests._env import needs_ngspice, needs_pdk, needs_pdk_named
from tests._example_conditions import assert_walkthrough_temperatures

# Importing the example must not touch a PDK or a simulator.
from silk.examples import design_loop_walkthrough as dl
from silk.examples import design_loop_walkthrough_gf180mcu as gf

needs_gf180 = needs_pdk_named('gf180mcu')
DEVICE_W = {'M1': 'w_in_um', 'M2': 'w_in_um', 'M3': 'w_load_um',
            'M4': 'w_load_um', 'M5': 'w_tail_um', 'M6': 'w_ref_um'}
DEVICE_L = {'M1': dl.L_INPUT_UM, 'M2': dl.L_INPUT_UM, 'M3': dl.L_LOAD_UM,
            'M4': dl.L_LOAD_UM, 'M5': dl.L_TAIL_UM, 'M6': dl.L_TAIL_UM}


# --------------------------------------------------------------------------- #
# fast, PDK-free
# --------------------------------------------------------------------------- #
def test_example_has_no_import_time_pdk_or_simulator_dependency():
    probe = subprocess.run(
        [sys.executable, '-c',
         'from silk.examples import design_loop_walkthrough as dl; '
         'from silk.examples import design_loop_walkthrough_gf180mcu; '
         'assert not dl._PDK_CACHE, "import must not load a PDK"'],
        capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr


def test_only_the_technology_record_and_the_intent_differ():
    """The proof is only a proof if the loop is the *same* code: the sibling
    contributes a Technology, a restated intent and the unit check — nothing
    that re-implements a step."""
    assert gf.GF180MCU.pdk == 'gf180mcu'
    assert gf.GF180MCU.corner == 'typical'            # GF's spelling, not 'tt'
    assert gf.GF180MCU.nmos == 'nmos3p3' and gf.GF180MCU.pmos == 'pmos3p3'
    assert dl.SKY130.corner == 'tt'
    # The intent restates the supply and the floors; everything else is shared.
    changed = {k for k in gf.INTENT if gf.INTENT[k] != dl.INTENT[k]}
    assert changed == {'vdd_v', 'hot_vdd_v', 'gain_db_min', 'bw_hz_min'}
    assert gf.INTENT['hot_vdd_v'] == pytest.approx(0.9 * gf.INTENT['vdd_v'])
    # No step is re-implemented here.
    for step in ('build_sampling_buffer', 'characterize_and_store',
                 'reopen_and_query', 'rebuild_and_restore', 'layout_feedback',
                 'demonstrate_mismatch'):
        assert not hasattr(gf, step), step


def test_sizing_is_the_same_design_rule_on_both_technologies():
    """Sizing depends on intent + grid only; the sky130 knobs are unchanged
    (the flagship's own guard pins them) and GF180MCU's come from the same
    rule, so identical settling/load numbers give identical widths."""
    sky = dl.size_from_intent()
    gfk = dl.size_from_intent(gf.INTENT, gf.GF180MCU)
    assert gfk == sky
    assert dl.size_from_intent(dl.INTENT, dl.SKY130) == sky
    for key in ('w_in_um', 'w_load_um', 'w_tail_um', 'w_ref_um'):
        steps = gfk[key] / gf.GF180MCU.width_grid_um
        assert abs(steps - round(steps)) < 1e-6, (key, gfk[key])


# --------------------------------------------------------------------------- #
# fast, needs the GF180MCU install — geometry through the accessor (AC1/AC2)
# --------------------------------------------------------------------------- #
@needs_gf180
class TestGeometryOnMetreNativePdk:

    def test_netlist_is_in_metres_and_the_accessor_in_microns(self, tmp_path):
        knobs = dl.size_from_intent(gf.INTENT, gf.GF180MCU)
        mod, _tests = dl.build_sampling_buffer(knobs, tech=gf.GF180MCU,
                                               intent=gf.INTENT,
                                               work_dir=str(tmp_path))
        pdk = mod.pdk_details
        assert pdk.pdk_name == 'gf180mcu'
        text = mod.generate_netlist()
        assert text.startswith('.subckt sampling_buffer ')
        assert 'nfet_03v3' in text and 'pfet_03v3' in text
        assert 'w=15.35u l=500n' in text                    # metres, suffixed
        assert 'w=15.35 ' not in text                       # not bare microns
        for name, key in DEVICE_W.items():
            W, L, M = pdk.device_geometry_um(mod.elem_dict[name]['device'])
            assert (W, L, M) == pytest.approx((knobs[key], DEVICE_L[name], 1.0))

    def test_estimated_areas_equal_hand_computed_w_l_packing(self, tmp_path):
        """AC1: every device area is W·L·M × packing — not 1e12 or 1e-12 off."""
        from silk.layout import estimate_layout
        knobs = dl.size_from_intent(gf.INTENT, gf.GF180MCU)
        mod, _tests = dl.build_sampling_buffer(knobs, tech=gf.GF180MCU,
                                               intent=gf.INTENT,
                                               work_dir=str(tmp_path))
        est = estimate_layout(mod, dl.MET1)
        assert est.packing_factor == 2.5
        for name, key in DEVICE_W.items():
            hand = knobs[key] * DEVICE_L[name] * 1 * 2.5
            assert est.devices[name].area == pytest.approx(hand), name
            assert 1.0 < est.devices[name].area < 100.0, name
        assert 1.0 < est.nets['VOUT'].length < 100.0
        assert est.nets['VOUT'].c < 10.0                      # fF
        assert 5.0 < est.bounding_box[0] < 50.0

    def test_factory_and_add_element_styles_agree(self, tmp_path):
        """AC2: both construction styles, same footprint on the metre-native PDK."""
        knobs = dl.size_from_intent(gf.INTENT, gf.GF180MCU)
        geometry = gf.prove_geometry_units(knobs, str(tmp_path))
        assert set(geometry) == set(DEVICE_W)
        for name, g in geometry.items():
            assert g['accessor_um'] == pytest.approx(g['twin_um'])
            assert g['area_um2'] == pytest.approx(g['twin_area_um2'])
            assert g['area_um2'] == pytest.approx(g['hand_um2'])
            assert g['netlist_w_m'] == pytest.approx(g['accessor_um'][0] * 1e-6)

    @needs_pdk
    def test_same_microns_give_the_same_footprint_as_sky130(self, tmp_path):
        """AC2: the estimator reads canonical microns, so the sky130 build of
        the same knobs — a µm-native PDK — has exactly the same device areas
        and bounding box as the metre-native one."""
        from silk.layout import estimate_layout
        knobs = dl.size_from_intent()
        assert knobs == dl.size_from_intent(gf.INTENT, gf.GF180MCU)
        sky, _ = dl.build_sampling_buffer(knobs, work_dir=str(tmp_path / 'sky'))
        g180, _ = dl.build_sampling_buffer(knobs, tech=gf.GF180MCU,
                                           intent=gf.INTENT,
                                           work_dir=str(tmp_path / 'gf'))
        assert sky.pdk_details.pdk_name != g180.pdk_details.pdk_name
        est_s, est_g = estimate_layout(sky, dl.MET1), estimate_layout(g180, dl.MET1)
        for name in DEVICE_W:
            assert est_s.devices[name].area == pytest.approx(est_g.devices[name].area)
            assert sky.pdk_details.device_geometry_um(sky.elem_dict[name]['device']) \
                == pytest.approx(g180.pdk_details.device_geometry_um(
                    g180.elem_dict[name]['device']))
        assert est_s.bounding_box == pytest.approx(est_g.bounding_box)
        # ...while the netlists say different things, because the PDKs do.
        assert 'w=15.35 ' in sky.generate_netlist()          # bare µm (sky130)
        assert 'w=15.35u ' in g180.generate_netlist()        # metres (GF180MCU)


# --------------------------------------------------------------------------- #
# slow — the documented command, in process (AC1, AC3, AC4)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.requires_gf180
@needs_gf180
@needs_ngspice
def test_gf180mcu_walkthrough_end_to_end(tmp_path):
    summary = gf.main(str(tmp_path))
    assert_walkthrough_temperatures(summary)
    assert summary['tech'] is gf.GF180MCU
    knobs = summary['knobs']

    # Step 0 — every device: netlist metres == accessor microns; area by hand.
    for name, g in summary['geometry'].items():
        assert g['netlist_w_m'] == pytest.approx(knobs[DEVICE_W[name]] * 1e-6)
        assert g['area_um2'] == pytest.approx(
            knobs[DEVICE_W[name]] * DEVICE_L[name] * 2.5)
        assert g['area_um2'] == pytest.approx(g['twin_area_um2'])
    est = summary['layout']
    for name in DEVICE_W:
        assert est.devices[name].area == pytest.approx(
            summary['geometry'][name]['area_um2'])

    # Two distinct profiles at GF's own corner, both with selected current sets
    # whose magnitudes are the sized tail current, not a unit artifact.
    prov = summary['provenance']
    assert prov[dl.T_STEP_LIGHT]['profile_id'] != prov[dl.T_STEP_HEAVY]['profile_id']
    for name in (dl.T_STEP_LIGHT, dl.T_STEP_HEAVY):
        assert prov[name]['origin_version_id'] == summary['version_id']
        assert prov[name]['rows'] > 0
        assert 0.5 * knobs['i_tail_a'] < prov[name]['tail_peak_a'] < 2 * knobs['i_tail_a']

    # AC3: the fresh rebuild restored the highest-current context — chosen from
    # the stored evidence — with profile / current-set / origin provenance.
    restored = summary['restored']
    picked = summary['restored_context']
    assert picked == max((dl.T_STEP_LIGHT, dl.T_STEP_HEAVY),
                         key=lambda n: prov[n]['tail_peak_a'])
    assert restored.profile_id == prov[picked]['profile_id']
    assert restored.origin_version_id == summary['version_id']
    assert restored.target_version_id == summary['version_id']
    assert restored.current_set_id
    assert restored.current_count > 0 and restored.measurement_count == 1

    # Restored currents drove EM sizing above min width on the internal nets;
    # port nets stayed at min width by design.
    em = summary['em']
    for net in ('tail', 'vint'):
        assert em[net]['peak_a'] is not None
        assert em[net]['width_um'] == pytest.approx(
            min(dl.MET1.max_width, em[net]['peak_a'] / dl.MET1.j_max))
        assert em[net]['width_um'] > dl.MET1.min_width
    assert em['tail']['peak_a'] == pytest.approx(prov[picked]['tail_peak_a'])
    for net in ('VOUT', 'VDD', 'VSS', 'VINP', 'VINM', 'IBIAS'):
        assert em[net]['peak_a'] is None
        assert em[net]['width_um'] == pytest.approx(dl.MET1.min_width)

    # Injected parasitics moved the bandwidth the physical way, by more than
    # the sweep's 1.2 % grid (measured ~5.6 %) — and the 3.3 V intent holds.
    bw_pre, bw_post = summary['bw_pre_hz'], summary['bw_post_hz']
    assert bw_pre > gf.INTENT['bw_hz_min']
    assert summary['metrics']['gain_dB'] > gf.INTENT['gain_db_min']
    drop = (bw_pre - bw_post) / bw_pre
    assert 0.02 < drop < 0.30, f'bandwidth drop {drop:.1%} outside the expected band'
    assert summary['post_version_id'] != summary['version_id']

    assert summary['mismatch'] == {'edited': 'EvidenceMismatchError',
                                   'post_layout': 'EvidenceMismatchError'}

    # Artifacts exist; every spec passed at GF's corner; the netlist is the
    # post-layout design in the PDK's own unit.
    out = summary['out_dir']
    for name in ('report_pre_layout.json', 'report_pre_layout.html',
                 'report_post_layout.json', 'report_post_layout.html',
                 'pre_vs_post_layout.md', 'design_loop.db'):
        assert os.path.getsize(os.path.join(out, name)) > 0, name
    recon = summary['report_post'].spec_reconciliation
    statuses = {o.spec.key: o.status for o in recon.outcomes}
    assert statuses == {'settle_light': 'PASS', 'settle_heavy': 'PASS',
                        'gain': 'PASS', 'bw_pre_layout': 'PASS',
                        'bw_post_layout': 'PASS'}
    netlist = open(summary['netlist_path']).read()
    assert netlist.startswith('.subckt sampling_buffer ')
    assert 'XMM1 vint VINP tail VSS nfet_03v3 m=1 w=15.35u l=500n' in netlist
    assert 'R__par_R_VOUT' in netlist
