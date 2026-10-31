"""Layout intelligence integration tests.

1. Diff-pair symmetry — matched NFET diff pair must get identical R/C/length/width
   on both output nets after estimate_layout with a declared HalfCircuit.

2. Capture-then-diff — two sequential estimates where the second has larger
   devices should show an increased bounding box in the layout diff.

3. Full pipeline smoke test — estimate + inject_parasitics + generate_report
   round-trip without errors.
"""
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _routing_model():
    from silk.layout.routing_model import RoutingModel
    return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                        j_max=1e-3, min_width=0.14, max_width=10.0)


def _make_diff_pair(pdk, tmp_path, w=4):
    """Build a symmetric diff-pair with mn1/mn2 on a shared tail."""
    from silk.circuit.modules import Module
    amp = Module('PDK_skywater130', str(tmp_path), 'amp.cir',
                 pdk_details=pdk, module_name='amp')
    amp.add(pdk.nmos1p8('mn1', d='vout_p', g='vin_p', s='tail', b='vss', w=w, l=0.18, m=1))
    amp.add(pdk.nmos1p8('mn2', d='vout_n', g='vin_n', s='tail', b='vss', w=w, l=0.18, m=1))
    return amp


# ---------------------------------------------------------------------------
# 1. Diff-pair symmetry
# ---------------------------------------------------------------------------

class TestDiffPairSymmetry:

    def test_matched_pair_identical_r(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = _make_diff_pair(pdk, tmp_path)
        amp.declare_match_pair('diff', 'mn1', 'mn2')
        amp.net_dict['vout_p'].peak_current = 2.0e-3
        amp.net_dict['vout_n'].peak_current = 1.5e-3   # asymmetric input
        est = estimate_layout(amp, _routing_model())
        assert est.nets['vout_p'].r == pytest.approx(est.nets['vout_n'].r)

    def test_matched_pair_identical_c(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = _make_diff_pair(pdk, tmp_path)
        amp.declare_match_pair('diff', 'mn1', 'mn2')
        amp.net_dict['vout_p'].peak_current = 2.0e-3
        amp.net_dict['vout_n'].peak_current = 1.5e-3
        est = estimate_layout(amp, _routing_model())
        assert est.nets['vout_p'].c == pytest.approx(est.nets['vout_n'].c)

    def test_matched_pair_identical_width(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = _make_diff_pair(pdk, tmp_path)
        amp.declare_match_pair('diff', 'mn1', 'mn2')
        amp.net_dict['vout_p'].peak_current = 2.0e-3
        amp.net_dict['vout_n'].peak_current = 1.5e-3
        est = estimate_layout(amp, _routing_model())
        assert est.nets['vout_p'].width == pytest.approx(est.nets['vout_n'].width)

    def test_matched_pair_equivalence_class_populated(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp = _make_diff_pair(pdk, tmp_path)
        amp.declare_match_pair('diff', 'mn1', 'mn2')
        est = estimate_layout(amp, _routing_model())
        # vout_p and vout_n should appear in the same equivalence class
        eq_sets = [set(v) for v in est.equivalence_classes.values()]
        assert any({'vout_p', 'vout_n'} <= s for s in eq_sets)

    def test_unmatched_pair_may_differ(self, pdk, tmp_path):
        """Without declaration, asymmetric inputs can produce different scalars."""
        from silk.layout.estimator import estimate_layout
        amp = _make_diff_pair(pdk, tmp_path)
        amp.net_dict['vout_p'].peak_current = 5.0e-3   # much larger
        amp.net_dict['vout_n'].peak_current = 0.1e-3
        est = estimate_layout(amp, _routing_model())
        # Without a match declaration the estimator does not symmetrize
        # so widths *can* differ (they will unless both happen to be clamped).
        # We just check the call succeeds — not that they differ.
        assert 'vout_p' in est.nets
        assert 'vout_n' in est.nets


# ---------------------------------------------------------------------------
# 2. Bounding-box growth after device size increase
# ---------------------------------------------------------------------------

class TestBoundingBoxGrowth:

    def test_larger_devices_give_larger_bbox(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        amp_small = _make_diff_pair(pdk, tmp_path / 'small', w=2)
        amp_large = _make_diff_pair(pdk, tmp_path / 'large', w=20)
        est_small = estimate_layout(amp_small, _routing_model())
        est_large = estimate_layout(amp_large, _routing_model())
        area_small = est_small.bounding_box[0] * est_small.bounding_box[1]
        area_large = est_large.bounding_box[0] * est_large.bounding_box[1]
        assert area_large > area_small

    def test_layout_diff_detects_bbox_change(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        from silk.circuit.report import generate_report
        from silk.circuit.report_diff import diff_reports
        amp_small = _make_diff_pair(pdk, tmp_path / 'small', w=2)
        amp_large = _make_diff_pair(pdk, tmp_path / 'large', w=20)
        est_small = estimate_layout(amp_small, _routing_model())
        est_large = estimate_layout(amp_large, _routing_model())
        r_small = generate_report(amp_small, layout_estimate=est_small,
                                  auto_discover_config=False)
        r_large = generate_report(amp_large, layout_estimate=est_large,
                                  auto_discover_config=False)
        diff = diff_reports(r_small, r_large)
        assert diff.layout_diff is not None
        assert diff.layout_diff.bbox_area_delta > 0


# ---------------------------------------------------------------------------
# 3. Full pipeline smoke test
# ---------------------------------------------------------------------------

class TestFullPipelineSmoke:

    def test_estimate_inject_report_no_errors(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout
        from silk.layout.parasitics import inject_parasitics
        from silk.circuit.report import generate_report
        amp = _make_diff_pair(pdk, tmp_path)
        amp.declare_match_pair('diff', 'mn1', 'mn2')
        amp.net_dict['vout_p'].peak_current = 2.0e-3
        amp.net_dict['vout_n'].peak_current = 2.0e-3
        est = estimate_layout(amp, _routing_model())
        inject_parasitics(amp, est, mode='rc')
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        assert report.layout_estimate is est
        # Markdown should render without errors
        md = report._to_markdown()
        assert '## Layout' in md

    def test_capture_after_estimate_matches_net_state(self, pdk, tmp_path):
        from silk.layout.estimator import estimate_layout, capture_layout
        amp = _make_diff_pair(pdk, tmp_path)
        est = estimate_layout(amp, _routing_model())
        snap = capture_layout(amp)
        # captured snapshot should have same net scalars as the estimate
        for net_name in est.nets:
            if net_name in snap.nets:
                assert snap.nets[net_name].r == pytest.approx(est.nets[net_name].r)
                assert snap.nets[net_name].c == pytest.approx(est.nets[net_name].c)

    def test_parasitics_nets_excluded_from_capture(self, pdk, tmp_path):
        """__par_* midpoint nets should not appear in capture_layout output."""
        from silk.layout.estimator import estimate_layout, capture_layout
        from silk.layout.parasitics import inject_parasitics
        amp = _make_diff_pair(pdk, tmp_path)
        est = estimate_layout(amp, _routing_model())
        inject_parasitics(amp, est, mode='rc')
        snap = capture_layout(amp)
        for net_name in snap.nets:
            assert not net_name.startswith('__par_')
