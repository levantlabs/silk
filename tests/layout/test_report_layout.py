"""Tests for layout section in CircuitReport / generate_report."""
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_amp(pdk, tmp_path):
    from silk.circuit.modules import Module
    amp = Module('PDK_skywater130', str(tmp_path), 'amp.cir',
                 pdk_details=pdk, module_name='amp')
    amp.add(pdk.nmos1p8('mn1', d='vout', g='vin', s='vss', b='vss', w=4, l=0.18, m=1))
    return amp


def _routing_model():
    from silk.layout.routing_model import RoutingModel
    return RoutingModel(sheet_resistance=0.1, cap_per_um2=0.04,
                        j_max=1e-3, min_width=0.14, max_width=10.0)


def _run_estimate(module):
    from silk.layout.estimator import estimate_layout
    return estimate_layout(module, _routing_model())


# ---------------------------------------------------------------------------
# CircuitReport.layout_estimate field
# ---------------------------------------------------------------------------

class TestCircuitReportLayoutField:

    def test_layout_estimate_field_defaults_none(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        report = generate_report(amp, auto_discover_config=False)
        assert report.layout_estimate is None

    def test_layout_estimate_field_set_when_passed(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        from silk.layout.estimator import LayoutEstimate
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        assert isinstance(report.layout_estimate, LayoutEstimate)

    def test_layout_estimate_matches_passed_value(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        assert report.layout_estimate is est


# ---------------------------------------------------------------------------
# _to_markdown layout section
# ---------------------------------------------------------------------------

class TestLayoutMarkdownSection:

    def test_no_layout_section_when_none(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        report = generate_report(amp, auto_discover_config=False)
        md = report._to_markdown()
        assert 'Layout' not in md

    def test_layout_section_present_when_estimate_set(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        md = report._to_markdown()
        assert '## Layout' in md

    def test_bounding_box_in_markdown(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        md = report._to_markdown()
        assert 'Bounding box' in md or 'bounding' in md.lower()

    def test_net_table_in_markdown(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        md = report._to_markdown()
        # Net names should appear
        for net_name in est.nets:
            assert net_name in md

    def test_warnings_in_markdown_when_present(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        if est.warnings:
            md = report._to_markdown()
            assert 'warning' in md.lower() or 'Warning' in md


# ---------------------------------------------------------------------------
# save_json layout section
# ---------------------------------------------------------------------------

class TestLayoutJsonSection:

    def test_save_json_without_layout(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        report = generate_report(amp, auto_discover_config=False)
        out = tmp_path / 'report.json'
        report.save_json(str(out))
        import json
        d = json.loads(out.read_text())
        # layout key should be absent or null
        assert d.get('layout') is None

    def test_save_json_with_layout(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        out = tmp_path / 'report_layout.json'
        report.save_json(str(out))
        import json
        d = json.loads(out.read_text())
        assert d['layout'] is not None
        assert 'bounding_box' in d['layout']
        assert 'nets' in d['layout']

    def test_layout_json_nets_have_scalars(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_estimate(amp)
        report = generate_report(amp, layout_estimate=est,
                                 auto_discover_config=False)
        out = tmp_path / 'report_layout2.json'
        report.save_json(str(out))
        import json
        d = json.loads(out.read_text())
        nets = d['layout']['nets']
        for net_name, nl in est.nets.items():
            assert net_name in nets
            assert 'length' in nets[net_name]
            assert 'width' in nets[net_name]
            assert 'r' in nets[net_name]
            assert 'c' in nets[net_name]


# ---------------------------------------------------------------------------
# TASK-28 AC7: packing_factor/routing_overhead provenance is recorded and
# rendered INDEPENDENTLY — one may be present while the other is absent
# (a hand-built or legacy LayoutEstimate), and Markdown must not raise
# TypeError formatting a None heuristic with ':g'.
# ---------------------------------------------------------------------------

def _make_estimate(**provenance):
    from silk.layout.estimator import LayoutEstimate
    return LayoutEstimate(
        computation_mode='statistical',
        routing_model_id='rm',
        timestamp=0.0,
        nets={},
        devices={},
        bounding_box=(1.0, 1.0),
        equivalence_classes={},
        warnings=(),
        **provenance,
    )


class TestLayoutProvenanceMarkdown:

    def test_both_present_renders_both(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _make_estimate(packing_factor=2.5, routing_overhead=2.0)
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        md = report._to_markdown()
        assert 'packing factor 2.5' in md
        assert 'routing overhead 2' in md

    def test_packing_factor_only_does_not_raise(self, pdk, tmp_path):
        """routing_overhead is None while packing_factor is set — the bug
        the reviewer found: f'{None:g}' used to raise TypeError here."""
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _make_estimate(packing_factor=1.5, routing_overhead=None)
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        md = report._to_markdown()   # must not raise
        assert 'packing factor 1.5' in md
        assert 'routing overhead' not in md

    def test_routing_overhead_only_does_not_raise(self, pdk, tmp_path):
        """The mirror case — packing_factor absent, routing_overhead set."""
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _make_estimate(packing_factor=None, routing_overhead=3.0)
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        md = report._to_markdown()   # must not raise
        assert 'routing overhead 3' in md
        assert 'packing factor' not in md

    def test_both_absent_omits_heuristics_line(self, pdk, tmp_path):
        """A captured estimate (no provenance at all) prints no heuristics
        line rather than 'packing factor None'."""
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _make_estimate(packing_factor=None, routing_overhead=None)
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        md = report._to_markdown()
        assert 'Area heuristics' not in md

    def test_save_html_does_not_raise_on_partial_provenance(self, pdk, tmp_path):
        """save_html does not raise on partial provenance (content: see the
        PDK-free reproduction test at the end of this file)."""
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _make_estimate(packing_factor=1.5, routing_overhead=None)
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        out = tmp_path / 'partial.html'
        report.save_html(str(out))   # must not raise
        assert out.exists()


# ---------------------------------------------------------------------------
# TASK-28 AC7: JSON compatibility across provenance shapes — old files
# (written before packing_factor/routing_overhead existed), captured
# estimates (both null), complete (both present), and partial (one present).
# ---------------------------------------------------------------------------

def _layout_dict(**provenance):
    """A minimal on-disk 'layout' dict, as ``save_json`` would write it,
    with the given provenance keys/values (or omitted entirely)."""
    return {
        'computation_mode': 'statistical',
        'routing_model_id': 'rm',
        'timestamp': 0.0,
        'bounding_box': [1.0, 1.0],
        'nets': {},
        'devices': {},
        'warnings': [],
        **provenance,
    }


class TestLayoutProvenanceJsonCompatibility:

    def test_old_format_missing_keys(self):
        """A file written before packing_factor/routing_overhead existed:
        the keys are absent entirely, not null."""
        from silk.circuit.report import _layout_from_dict
        le = _layout_from_dict(_layout_dict())
        assert le.packing_factor is None
        assert le.routing_overhead is None

    def test_captured_both_null(self):
        from silk.circuit.report import _layout_from_dict
        le = _layout_from_dict(_layout_dict(packing_factor=None, routing_overhead=None))
        assert le.packing_factor is None
        assert le.routing_overhead is None

    def test_complete_both_present(self):
        from silk.circuit.report import _layout_from_dict
        le = _layout_from_dict(_layout_dict(packing_factor=2.5, routing_overhead=2.0))
        assert le.packing_factor == 2.5
        assert le.routing_overhead == 2.0


# ---------------------------------------------------------------------------
# required_width_exceeds_max warning through the report paths (TASK-204 AC#9)
# ---------------------------------------------------------------------------

def _run_capped_estimate(module):
    """Same shape as the TASK-204 reproducer: 20 mA on a 1e-3 A/um,
    10 um max_width layer requires 20 um -> capped, with a warning."""
    from silk.layout.estimator import estimate_layout
    for net in module.net_dict.values():
        net.peak_current = 0.020
    return estimate_layout(module, _routing_model())


class TestRequiredWidthExceedsMaxReportPaths:

    def test_reason_severity_net_detail_in_markdown(self, pdk, tmp_path):
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_capped_estimate(amp)
        w = next(w for w in est.warnings if w.reason == 'required_width_exceeds_max')
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        md = report._to_markdown()
        assert w.reason in md
        assert w.severity in md
        assert w.net_name in md
        assert w.detail in md

    def test_no_global_pass_fail_change(self, pdk, tmp_path):
        """AC#9: the warning must not alter report pass/fail status."""
        from silk.circuit.report import generate_report
        amp = _make_amp(pdk, tmp_path)
        est = _run_capped_estimate(amp)
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        assert report.interface_check_passed is True

    def test_survives_json_save_and_load(self, pdk, tmp_path):
        from silk.circuit.report import generate_report, _layout_from_dict
        amp = _make_amp(pdk, tmp_path)
        est = _run_capped_estimate(amp)
        w = next(w for w in est.warnings if w.reason == 'required_width_exceeds_max')
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        out = tmp_path / 'capped.json'
        report.save_json(str(out))
        import json
        d = json.loads(out.read_text())
        json_warnings = d['layout']['warnings']
        match = next(jw for jw in json_warnings if jw['reason'] == 'required_width_exceeds_max')
        assert match['severity'] == w.severity
        assert match['net_name'] == w.net_name
        assert match['detail'] == w.detail

        reloaded = _layout_from_dict(d['layout'])
        reloaded_match = next(rw for rw in reloaded.warnings
                              if rw.reason == 'required_width_exceeds_max')
        assert reloaded_match.severity == w.severity
        assert reloaded_match.net_name == w.net_name
        assert reloaded_match.detail == w.detail

    def test_partial_packing_factor_only(self):
        from silk.circuit.report import _layout_from_dict
        le = _layout_from_dict(_layout_dict(packing_factor=1.5, routing_overhead=None))
        assert le.packing_factor == 1.5
        assert le.routing_overhead is None

    def test_partial_routing_overhead_only(self):
        from silk.circuit.report import _layout_from_dict
        le = _layout_from_dict(_layout_dict(packing_factor=None, routing_overhead=3.0))
        assert le.packing_factor is None
        assert le.routing_overhead == 3.0

    def test_partial_provenance_survives_a_real_json_file_round_trip(self, pdk, tmp_path):
        """End-to-end: save_json -> load_json -> render, with only one
        heuristic recorded."""
        from silk.circuit.report import generate_report, CircuitReport
        amp = _make_amp(pdk, tmp_path)
        est = _make_estimate(packing_factor=1.5, routing_overhead=None)
        report = generate_report(amp, layout_estimate=est, auto_discover_config=False)
        out = str(tmp_path / 'partial_roundtrip.json')
        report.save_json(out)
        loaded = CircuitReport.load_json(out)
        assert loaded.layout_estimate.packing_factor == 1.5
        assert loaded.layout_estimate.routing_overhead is None
        md = loaded._to_markdown()   # must not raise
        assert 'packing factor 1.5' in md


def test_child_footprint_reaches_attached_markdown_and_json(pdk, tmp_path):
    """Report consumers receive the corrected hierarchy values unchanged."""
    import json
    import math

    from silk.circuit.modules import Module
    from silk.circuit.report import generate_report
    from silk.layout import estimate_layout

    child = _make_amp(pdk, tmp_path)
    child_est = _run_estimate(child)
    child_area = math.prod(child_est.bounding_box)
    assert child_area == pytest.approx(3.6)
    parent = Module('PDK_skywater130', str(tmp_path), 'parent.cir',
                    pdk_details=pdk)
    parent.add_module(child, 'xamp')
    instance = parent.elem_dict['xamp']['device']
    assert instance is child
    est = estimate_layout(parent, _routing_model(), packing_factor=7)
    side = math.sqrt(7.2)
    assert est.bounding_box == pytest.approx((side, side))
    assert est.devices['xamp'].area == instance.estimated_area == pytest.approx(3.6)
    assert instance.bounding_box == child_est.bounding_box

    report = generate_report(parent, layout_estimate=est,
                             auto_discover_config=False)
    assert f'**Bounding box:** {side:.2f} × {side:.2f} μm' in report._to_markdown()
    out = tmp_path / 'hierarchy.json'
    report.save_json(str(out))
    layout = json.loads(out.read_text())['layout']
    assert layout['bounding_box'] == pytest.approx([side, side])
    assert layout['devices']['xamp']['area'] == pytest.approx(child_area)
    assert layout['packing_factor'] == 7
    assert layout['routing_overhead'] == 2


def test_attached_estimate_and_warning_reach_markdown_json_and_html(tmp_path):
    """PDK-free repro: overlapping match pairs -> matching_conflict_large_class
    must appear in every format; no estimate -> no layout section."""
    import json
    from silk import Module, generate_report
    from silk.elements import ElementLibrary
    from silk.layout import RoutingModel, estimate_layout

    ckt = Module(None, str(tmp_path), 'review.cir',
                 pdk_details=None, module_name='layout_review')
    elements = ElementLibrary()
    with ckt:
        elements.resistor('r1', p='n1', n='vss', value=1000.0)
        elements.resistor('r2', p='n2', n='vss', value=1000.0)
        elements.resistor('r3', p='n3', n='vss', value=1000.0)
    ckt.declare_match_pair('pair12', 'r1', 'r2')
    ckt.declare_match_pair('pair23', 'r2', 'r3')
    estimate = estimate_layout(ckt, RoutingModel(
        sheet_resistance=0.1, cap_per_um2=0.04,
        j_max=1e-3, min_width=0.14, max_width=10.0,
    ))
    report = generate_report(ckt, layout_estimate=estimate,
                             auto_discover_config=False)
    plain = generate_report(ckt, auto_discover_config=False)
    md_path, html_path, json_path = (str(tmp_path / n) for n in
                                     ('r.md', 'r.html', 'r.json'))
    report.save(md_path)
    report.save_html(html_path)
    report.save_json(json_path)
    md = open(md_path, encoding='utf-8').read()
    html_text = open(html_path, encoding='utf-8').read()
    js = open(json_path, encoding='utf-8').read()
    assert estimate.warnings
    assert 'matching_conflict_large_class' in md
    assert 'matching_conflict_large_class' in js
    json.loads(js)
    for needle in ('Layout Estimate', 'Layout Warnings',
                   'matching_conflict_large_class', 'Bounding box',
                   'Length (μm)', 'n1', 'n3', 'Equivalence class has 3 members'):
        assert needle in html_text, needle
    # Detail text contains quotes/brackets: must be escaped, not raw.
    assert "[&#x27;n1&#x27;" in html_text or "['n1'" not in html_text
    assert 'Layout Estimate' not in plain._to_html()
    assert 'Layout Estimate' not in plain._to_markdown()
    ckt.close()
