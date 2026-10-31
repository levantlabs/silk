"""Guards for ``silk/examples/design_loop_walkthrough_sg13g2.py`` — the
flagship design loop, adapted for a third, OSDI-model PDK (TASK-17, following
TASK-16's GF180MCU proof and TASK-18's OSDI bring-up).

Fast tests pin what needs no OSDI-capable simulator: the intent is a pure
restatement, the ``Technology`` record is the only thing that differs from
the flagship for the steps that *are* shared, and — with the SG13G2 PDK
installed — the geometry story (metre-native netlist, micron accessor,
estimator agreement, both construction styles) plus the current-annotation
boundary itself, which fails during netlist generation and so needs no
simulator at all. The slow test executes the documented end-to-end command
in process and checks every stage's numbers, provenance and artifacts,
including that the boundary is hit and worked around exactly as documented.
"""
import os
import subprocess
import sys

import pytest

from tests._env import needs_ngspice, needs_osdi_runnable, needs_pdk_named
from tests._example_conditions import assert_walkthrough_temperatures

# Importing the example must not touch a PDK or a simulator.
from silk.examples import design_loop_walkthrough as dl
from silk.examples import design_loop_walkthrough_sg13g2 as sg

needs_sg13g2 = needs_pdk_named('sg13g2')
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
         'from silk.examples import design_loop_walkthrough_sg13g2; '
         'assert not dl._PDK_CACHE, "import must not load a PDK"'],
        capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr


def test_only_three_steps_are_reimplemented_and_two_stay_shared():
    """The proof is still a proof of the *shared* loop: the sibling
    contributes a Technology, a restated intent, the unit check — and,
    unlike TASK-16's GF180MCU sibling, three reimplemented steps whose
    flagship originals assume current evidence exists. ``layout_feedback``
    and ``demonstrate_mismatch`` are NOT reimplemented — they are called
    from ``dl`` unchanged, proving the boundary is scoped to exactly the
    steps that need current evidence."""
    assert sg.SG13G2.pdk == 'sg13g2'
    assert sg.SG13G2.corner == 'typical'
    assert sg.SG13G2.nmos == 'nmos_lv' and sg.SG13G2.pmos == 'pmos_lv'
    assert dl.SKY130.corner == 'tt'
    changed = {k for k in sg.INTENT if sg.INTENT[k] != dl.INTENT[k]}
    assert changed == {'vdd_v', 'hot_vdd_v', 'gain_db_min', 'bw_hz_min'}
    assert sg.INTENT['hot_vdd_v'] == pytest.approx(0.9 * sg.INTENT['vdd_v'])
    # Not reimplemented: the two steps that need no current evidence at all.
    for step in ('build_sampling_buffer', 'wire_pre_layout_tests', 'wire_ac_test',
                 'run_test', 'size_from_intent', 'layout_feedback',
                 'demonstrate_mismatch'):
        assert not hasattr(sg, step), f'{step} should be imported from dl, not reimplemented'
    # Reimplemented, deliberately: the three steps whose flagship originals
    # assume stored current evidence, which no OSDI-PDK design ever has.
    for step in ('characterize_and_store', 'reopen_and_query', 'rebuild_and_restore'):
        assert callable(getattr(sg, step))


def test_sizing_is_the_same_design_rule_on_both_technologies():
    """Sizing depends on intent + grid only; sky130's knobs are unchanged
    (the flagship's own guard pins them) and SG13G2's come from the same
    rule with the same grid, so identical settling/load numbers give
    identical widths."""
    sky = dl.size_from_intent()
    sgk = dl.size_from_intent(sg.INTENT, sg.SG13G2)
    assert sgk == sky
    for key in ('w_in_um', 'w_load_um', 'w_tail_um', 'w_ref_um'):
        steps = sgk[key] / sg.SG13G2.width_grid_um
        assert abs(steps - round(steps)) < 1e-6, (key, sgk[key])


# --------------------------------------------------------------------------- #
# fast, needs the SG13G2 install — geometry + the boundary (no simulator)
# --------------------------------------------------------------------------- #
@needs_sg13g2
class TestGeometryOnMetreNativePdk:

    def test_netlist_is_in_metres_and_the_accessor_in_microns(self, tmp_path):
        knobs = dl.size_from_intent(sg.INTENT, sg.SG13G2)
        mod, _tests = dl.build_sampling_buffer(knobs, tech=sg.SG13G2,
                                               intent=sg.INTENT,
                                               work_dir=str(tmp_path))
        pdk = mod.pdk_details
        assert pdk.pdk_name == 'sg13g2'
        text = mod.generate_netlist()
        assert text.startswith('.subckt sampling_buffer ')
        assert 'sg13_lv_nmos' in text and 'sg13_lv_pmos' in text
        assert 'w=15.35u l=500n' in text                    # metres, suffixed
        assert 'w=15.35 ' not in text                       # not bare microns
        for name, key in DEVICE_W.items():
            W, L, M = pdk.device_geometry_um(mod.elem_dict[name]['device'])
            assert (W, L, M) == pytest.approx((knobs[key], DEVICE_L[name], 1.0))

    def test_estimated_areas_equal_hand_computed_w_l_packing(self, tmp_path):
        """Every device area is W·L·M × packing — not 1e12 or 1e-12 off."""
        from silk.layout import estimate_layout
        knobs = dl.size_from_intent(sg.INTENT, sg.SG13G2)
        mod, _tests = dl.build_sampling_buffer(knobs, tech=sg.SG13G2,
                                               intent=sg.INTENT,
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
        knobs = dl.size_from_intent(sg.INTENT, sg.SG13G2)
        geometry = sg.prove_geometry_units(knobs, str(tmp_path))
        assert set(geometry) == set(DEVICE_W)
        for name, g in geometry.items():
            assert g['accessor_um'] == pytest.approx(g['twin_um'])
            assert g['area_um2'] == pytest.approx(g['twin_area_um2'])
            assert g['area_um2'] == pytest.approx(g['hand_um2'])
            assert g['netlist_w_m'] == pytest.approx(g['accessor_um'][0] * 1e-6)

    def test_current_annotation_boundary_is_real_and_documented(self, tmp_path):
        """The refusal happens during netlist generation, before any
        simulator is invoked — this test needs the SG13G2 PDK (to build a
        real design against it) but NOT an OSDI-capable ngspice, and must
        still pass on a machine stuck with ngspice-42."""
        knobs = dl.size_from_intent(sg.INTENT, sg.SG13G2)
        message = sg.prove_current_annotation_boundary(knobs, str(tmp_path))
        assert message == sg.CURRENT_BOUNDARY_MESSAGE
        assert 'OSDI' in message and 'save_device_currents' in message


# --------------------------------------------------------------------------- #
# slow — the documented command, in process (needs OSDI-capable ngspice)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.requires_sg13g2
@needs_ngspice
@needs_osdi_runnable('sg13g2')
def test_sg13g2_walkthrough_end_to_end(tmp_path):
    summary = sg.main(str(tmp_path))
    assert_walkthrough_temperatures(summary)
    assert summary['tech'] is sg.SG13G2
    knobs = summary['knobs']

    # Step 0a — every device: netlist metres == accessor microns; area by hand.
    for name, g in summary['geometry'].items():
        assert g['netlist_w_m'] == pytest.approx(knobs[DEVICE_W[name]] * 1e-6)
        assert g['area_um2'] == pytest.approx(
            knobs[DEVICE_W[name]] * DEVICE_L[name] * 2.5)
        assert g['area_um2'] == pytest.approx(g['twin_area_um2'])
    est = summary['layout']
    for name in DEVICE_W:
        assert est.devices[name].area == pytest.approx(
            summary['geometry'][name]['area_um2'])

    # Step 0b — the boundary is hit for real and the message is exact.
    assert summary['boundary_message'] == sg.CURRENT_BOUNDARY_MESSAGE

    # Steps 3+4 — physically sensible measured bands at SG13G2's 'typical'
    # corner, VDD 1.2 V, light load. The AC sweep is 200 pts/decade, so
    # bandwidth carries a stated +-1.2 % grid resolution (design_loop_
    # walkthrough.wire_ac_test); gain has no comparable sweep-grid quantization.
    metrics = summary['metrics']
    assert 15.0 < metrics['gain_dB'] < 35.0
    assert metrics['gain_dB'] > sg.INTENT['gain_db_min']
    assert 150e6 < metrics['bw_hz'] < 450e6
    assert metrics['bw_hz'] > sg.INTENT['bw_hz_min']
    assert 0.0 < metrics['t_settle_light_s'] < dl.INTENT['settle_light_s_max']
    assert 0.0 < metrics['t_settle_heavy_s'] < dl.INTENT['settle_heavy_s_max']

    # Steps 5+6 — every net's stored spec measurement is restored (all four
    # pre-layout specs); no current evidence exists to restore (the boundary),
    # demonstrated via the real EvidenceNotFoundError rather than skipped.
    assert summary['restored_count'] == 4
    assert summary['post_version_id'] != summary['version_id']

    # Step 7 — reused unchanged from the flagship: every net (ports AND the
    # internal tail/vint nets) falls back to MET1.min_width, because no net
    # carries a restored current on this OSDI-PDK design.
    em = summary['em']
    for name, row in em.items():
        assert row['peak_a'] is None, name
        assert row['width_um'] == pytest.approx(dl.MET1.min_width), name

    # Injected parasitics moved the bandwidth the physical way, by more than
    # the sweep's 1.2 % grid — and the 1.2 V intent holds both pre- and
    # post-layout.
    bw_pre, bw_post = summary['bw_pre_hz'], summary['bw_post_hz']
    assert bw_pre > sg.INTENT['bw_hz_min']
    drop = (bw_pre - bw_post) / bw_pre
    assert 0.01 < drop < 0.15, f'bandwidth drop {drop:.1%} outside the expected band'
    assert summary['post_version_id'] != summary['version_id']

    assert summary['mismatch'] == {'edited': 'EvidenceMismatchError',
                                   'post_layout': 'EvidenceMismatchError'}

    # Artifacts exist; every measured spec passed at SG13G2's corner; the
    # netlist is the post-layout design in the PDK's own unit.
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
    assert 'XMM1 vint VINP tail VSS sg13_lv_nmos m=1 w=15.35u l=500n ng=1' in netlist
    assert 'R__par_R_VOUT' in netlist
