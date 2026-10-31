"""Tests for LayoutNetDelta, LayoutDiff, and layout diff in diff_reports."""
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_amp(pdk, tmp_path, name='amp'):
    from silk.circuit.modules import Module
    amp = Module('PDK_skywater130', str(tmp_path), f'{name}.cir',
                 pdk_details=pdk, module_name=name)
    amp.add(pdk.nmos1p8('mn1', d='vout', g='vin', s='vss', b='vss', w=4, l=0.18, m=1))
    return amp


def _routing_model():
    from silk.layout.routing_model import RoutingModel
    return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                        j_max=1e-3, min_width=0.14, max_width=10.0)


def _run_estimate(module):
    from silk.layout.estimator import estimate_layout
    return estimate_layout(module, _routing_model())


def _make_report(pdk, tmp_path, name='amp', layout=True):
    from silk.circuit.report import generate_report
    amp = _make_amp(pdk, tmp_path, name)
    est = _run_estimate(amp) if layout else None
    return generate_report(amp, layout_estimate=est,
                           auto_discover_config=False)


# ---------------------------------------------------------------------------
# LayoutNetDelta
# ---------------------------------------------------------------------------

class TestLayoutNetDelta:

    def test_construction(self):
        from silk.circuit.report_diff import LayoutNetDelta
        d = LayoutNetDelta(
            name='vout',
            length_a=50.0, length_b=60.0,
            width_a=0.5,   width_b=0.6,
            r_a=1.0,       r_b=1.2,
            c_a=10.0,      c_b=12.0,
        )
        assert d.name == 'vout'
        assert d.length_b - d.length_a == pytest.approx(10.0)

    def test_delta_r_property(self):
        from silk.circuit.report_diff import LayoutNetDelta
        d = LayoutNetDelta(
            name='vout',
            length_a=50.0, length_b=50.0,
            width_a=0.5,   width_b=0.5,
            r_a=1.0,       r_b=2.0,
            c_a=10.0,      c_b=10.0,
        )
        assert d.delta_r == pytest.approx(1.0)

    def test_delta_c_property(self):
        from silk.circuit.report_diff import LayoutNetDelta
        d = LayoutNetDelta(
            name='vout',
            length_a=50.0, length_b=50.0,
            width_a=0.5,   width_b=0.5,
            r_a=1.0,       r_b=1.0,
            c_a=10.0,      c_b=15.0,
        )
        assert d.delta_c == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# LayoutDiff
# ---------------------------------------------------------------------------

class TestLayoutDiff:

    def test_construction(self):
        from silk.circuit.report_diff import LayoutDiff
        d = LayoutDiff(
            changed=[],
            unchanged=[],
            nets_added=[],
            nets_dropped=[],
            bbox_a=(10.0, 10.0),
            bbox_b=(12.0, 12.0),
        )
        assert d.bbox_a == (10.0, 10.0)
        assert d.bbox_b == (12.0, 12.0)

    def test_bbox_delta_property(self):
        from silk.circuit.report_diff import LayoutDiff
        d = LayoutDiff(
            changed=[],
            unchanged=[],
            nets_added=[],
            nets_dropped=[],
            bbox_a=(10.0, 10.0),
            bbox_b=(12.0, 12.0),
        )
        # area = w * h
        assert d.bbox_area_delta == pytest.approx(12.0 * 12.0 - 10.0 * 10.0)


# ---------------------------------------------------------------------------
# diff_reports layout section
# ---------------------------------------------------------------------------

class TestDiffReportsLayout:

    def test_layout_diff_none_when_both_missing(self, pdk, tmp_path):
        from silk.circuit.report_diff import diff_reports
        r_a = _make_report(pdk, tmp_path / 'a', 'a', layout=False)
        r_b = _make_report(pdk, tmp_path / 'b', 'b', layout=False)
        diff = diff_reports(r_a, r_b)
        assert diff.layout_diff is None

    def test_layout_diff_none_when_a_missing(self, pdk, tmp_path):
        from silk.circuit.report_diff import diff_reports
        r_a = _make_report(pdk, tmp_path / 'a', 'a', layout=False)
        r_b = _make_report(pdk, tmp_path / 'b', 'b', layout=True)
        diff = diff_reports(r_a, r_b)
        assert diff.layout_diff is None

    def test_layout_diff_populated_when_both_present(self, pdk, tmp_path):
        from silk.circuit.report_diff import diff_reports, LayoutDiff
        r_a = _make_report(pdk, tmp_path / 'a', 'a', layout=True)
        r_b = _make_report(pdk, tmp_path / 'b', 'b', layout=True)
        diff = diff_reports(r_a, r_b)
        assert isinstance(diff.layout_diff, LayoutDiff)

    def test_unchanged_nets_when_same_estimate(self, pdk, tmp_path):
        from silk.circuit.report_diff import diff_reports
        r_a = _make_report(pdk, tmp_path / 'a', 'a', layout=True)
        r_b = _make_report(pdk, tmp_path / 'b', 'b', layout=True)
        diff = diff_reports(r_a, r_b)
        # Both modules are identical — all nets should be unchanged
        assert len(diff.layout_diff.changed) == 0
        assert len(diff.layout_diff.unchanged) > 0

    def test_changed_nets_when_r_differs(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        from silk.circuit.report_diff import diff_reports

        r_a = _make_report(pdk, tmp_path / 'a', 'a', layout=True)

        # Build report b with a manually tweaked R
        amp_b = _make_amp(pdk, tmp_path / 'b', 'b')
        est_b = _run_estimate(amp_b)
        # Patch vout R upward
        amp_b.net_dict['vout'].estimated_r = 999.0
        from silk.layout.estimator import capture_layout
        est_b_tweaked = capture_layout(amp_b)
        r_b = generate_report(amp_b, layout_estimate=est_b_tweaked,
                               auto_discover_config=False)
        diff = diff_reports(r_a, r_b)
        changed_names = [d.name for d in diff.layout_diff.changed]
        assert 'vout' in changed_names

    def test_layout_diff_in_markdown(self, pdk, tmp_path):
        from silk.circuit.report_diff import diff_reports
        r_a = _make_report(pdk, tmp_path / 'a', 'a', layout=True)
        r_b = _make_report(pdk, tmp_path / 'b', 'b', layout=True)
        diff = diff_reports(r_a, r_b)
        md = diff._to_markdown()
        assert 'Layout' in md
