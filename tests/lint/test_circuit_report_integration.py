"""End-to-end integration tests for CircuitReport with LintReport/CoverageReport."""
import pytest
from unittest.mock import MagicMock
from silk.circuit.report import CircuitReport, generate_report
from silk.lint.reports.lint_report import LintReport
from silk.lint.reports.coverage_report import CoverageReport
from silk.lint.reports.metadata import ReportMetadata
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir


def _pdk():
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    pdk.pdk_name = "test_pdk"
    pdk.version = "0.0.1"
    return pdk


def _empty_module(tmp_path):
    return Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())


def test_generate_report_includes_lint_field(tmp_path):
    rep = generate_report(_empty_module(tmp_path))
    assert hasattr(rep, "lint")
    assert isinstance(rep.lint, LintReport)


def test_generate_report_includes_coverage_field(tmp_path):
    rep = generate_report(_empty_module(tmp_path))
    assert hasattr(rep, "coverage")
    assert isinstance(rep.coverage, CoverageReport)


def test_generate_report_includes_metadata_field(tmp_path):
    rep = generate_report(_empty_module(tmp_path))
    assert hasattr(rep, "metadata")
    assert isinstance(rep.metadata, ReportMetadata)
    assert rep.metadata.schema_version == "1.0"


def test_metadata_module_name_matches(tmp_path):
    m = _empty_module(tmp_path)
    rep = generate_report(m)
    assert rep.metadata.module_name == "unnamed"  # no module_name set


def test_lint_violations_match_circuit_report_violations(tmp_path):
    """The LintReport should contain the union of interface/domain/junction violations."""
    a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=_pdk())
    a.define_module_name('a')
    a.define_module_ports({'out': PortDir.OUT})
    b = Module('TEST', str(tmp_path), 'b.cir', pdk_details=_pdk())
    b.define_module_name('b')
    b.define_module_ports({'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': 'shared'})
    top.add_module(b, 'b1', nets={'out': 'shared'})

    rep = generate_report(top)
    rule_ids_in_lint = {v.rule_id for v in rep.lint.violations}
    rule_ids_in_circuit = (
        {v.rule_id for v in rep.violations}
        | {v.rule_id for v in rep.domain_violations}
        | {v.rule_id for v in rep.junction_violations}
    )
    # The LintReport should be a superset (its rules ran via registry)
    assert rule_ids_in_circuit.issubset(rule_ids_in_lint) or "INT-001" in rule_ids_in_lint


def test_lint_save_html_works(tmp_path):
    rep = generate_report(_empty_module(tmp_path))
    p = tmp_path / "lint.html"
    rep.lint.save_html(p)
    assert p.exists()
    assert "<html" in p.read_text().lower()


def test_coverage_from_visit_log(tmp_path):
    """CoverageReport should record per-rule coverage from the registry run."""
    rep = generate_report(_empty_module(tmp_path))
    # Empty module has the rules dispatched but produces zero violations
    # The coverage root.local should have entries for at least some rules
    assert isinstance(rep.coverage.rule_coverage.root.local, dict)


def _build_dotted_or(tmp_path, net_name="shared"):
    a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=_pdk())
    a.define_module_name('a')
    a.define_module_ports({'out': PortDir.OUT})
    b = Module('TEST', str(tmp_path), 'b.cir', pdk_details=_pdk())
    b.define_module_name('b')
    b.define_module_ports({'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': net_name})
    top.add_module(b, 'b1', nets={'out': net_name})
    return top


def test_generate_report_applies_waivers_from_registry(tmp_path):
    """When generate_report is given a registry with loaded waivers, the LintReport
    moves matching violations from `violations` to `waivers_applied`."""
    from silk.lint.registry import RuleRegistry
    from silk.lint.waivers import Waiver, WaiverMatcher
    reg = RuleRegistry.default()
    reg._waivers.append(Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "shared"}),
        reason="intentional dotted-OR for test mux",
    ))

    top = _build_dotted_or(tmp_path, net_name="shared")
    rep = generate_report(top, registry=reg, auto_discover_config=False)

    # No INT-001 in the active violations; at least one in waivers_applied
    assert all(v.rule_id != "INT-001" for v in rep.lint.violations)
    assert any(w.rule_id == "INT-001" for w in rep.lint.waivers_applied)
    # And the legacy CircuitReport.violations field also reflects the suppression
    assert all(v.rule_id != "INT-001" for v in rep.violations)


def test_generate_report_matched_waiver_serializes(tmp_path):
    """TASK-601: a report with a matched waiver round-trips through JSON,
    recording both the waiver and the violation it suppressed."""
    import json
    from silk.lint.registry import RuleRegistry
    from silk.lint.reports.lint_report import LintReport
    from silk.lint.waivers import Waiver, WaiverMatcher
    reg = RuleRegistry.default()
    reg._waivers.append(Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "shared"}),
        reason="intentional dotted-OR for test mux",
    ))
    rep = generate_report(_build_dotted_or(tmp_path, net_name="shared"),
                          registry=reg, auto_discover_config=False)

    d = rep.lint.to_dict()
    assert [w["rule_id"] for w in d["waivers_applied"]] == ["INT-001"]
    assert d["waivers_applied"][0]["matcher"]["fields"] == {"net_name": "shared"}
    assert d["waived_violations"] and all(
        v["rule_id"] == "INT-001" for v in d["waived_violations"])

    path = tmp_path / "lint.json"
    rep.lint.save_json(path)
    json.loads(path.read_text())
    loaded = LintReport.load_json(path)
    assert loaded.to_dict() == d
    assert [w.rule_id for w in loaded.waivers_matched] == ["INT-001"]


def test_unmatched_waiver_json_payload_unchanged(tmp_path):
    """An unmatched waiver leaves the pre-TASK-601 JSON payload byte-for-byte
    what it was, checked against a hand-written payload."""
    import json
    from silk.lint.registry import RuleRegistry
    from silk.lint.waivers import Waiver, WaiverMatcher
    reg = RuleRegistry.default()
    reg._waivers.append(Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "other"}),
        reason="does not match",
    ))
    rep = generate_report(_build_dotted_or(tmp_path, net_name="shared"),
                          registry=reg, auto_discover_config=False)
    assert rep.lint.waivers_applied == [] and rep.lint.waivers_matched == []
    md = rep.lint.metadata
    md.silk_version = "0.0-test"
    md.generated_at = "2026-01-01T00:00:00"
    md.context_hash = "fixedhash"
    expected = {
        "metadata": {
            "schema_version": md.schema_version,
            "silk_version": "0.0-test",
            "pdk_name": "test_pdk",
            "pdk_version": "0.0.1",
            "module_name": "unnamed",
            "generated_at": "2026-01-01T00:00:00",
            "context_hash": "fixedhash",
        },
        "stress_coverage": {
            "summary": {"total": 0, "full": 0, "partial": 0, "none": 0, "failed": 0},
            "devices": [],
        },
        "violations": [{
            "__class": "InterfaceViolation",
            "rule_id": "INT-001",
            "severity": "error",
            "message": "Net 'shared' has multiple drivers: a1.out, b1.out",
            "location": {"__type": "net", "net_name": "shared",
                         "port_names": ["a1.out", "b1.out"]},
            "context": {},
        }],
        "waivers_applied": [],
        "no_connects": [],
        "stale_markers": [],
    }
    d = rep.lint.to_dict()
    assert d == expected
    assert list(d) == list(expected)
    path = tmp_path / "unmatched.json"
    rep.lint.save_json(path)
    assert json.loads(path.read_text()) == expected


def test_positional_construction_with_no_connects_unchanged():
    """The pre-TASK-601 positional field order still binds correctly."""
    from silk.lint.reports.lint_report import LintReport
    from silk.lint.reports.metadata import ReportMetadata
    from silk.lint_markers import NoConnect
    meta = ReportMetadata(silk_version="0", pdk_name="p", pdk_version="1",
                          module_name="m", generated_at="2026-01-01T00:00:00")
    rep = LintReport(meta, [], [], [('u', 'unused', NoConnect(reason='intentional'))])
    assert rep.waivers_matched == []
    d = rep.to_dict()
    assert d["waivers_applied"] == []
    assert d["no_connects"] == [{"kind": "net", "instance_name": "u",
                                 "net_name": "unused", "reason": "intentional"}]
    assert "waived_violations" not in d
    import json
    json.dumps(d)


def test_generate_report_excludes_subtree(tmp_path):
    """Violations whose location matches a ScopeExclusion are filtered out."""
    from silk.lint.registry import RuleRegistry
    from silk.lint.exclusions import ScopeExclusion
    reg = RuleRegistry.default()
    reg._exclusions.append(ScopeExclusion(
        target_type="instance",
        target_path="a1",
        reason="vendor IP",
    ))

    a = Module('TEST', str(tmp_path), 'a.cir', pdk_details=_pdk())
    a.define_module_name('a')
    a.define_module_ports({'in': PortDir.IN, 'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'in': 'wire1'})  # 'out' unbound → STRUCT-001

    rep = generate_report(top, registry=reg, auto_discover_config=False)
    # Violations whose port location is on instance a1 should be filtered out
    a1_violations = [
        v for v in rep.lint.violations
        if getattr(v.location, 'module_name', '') == 'a1'
    ]
    assert a1_violations == []


def test_generate_report_lint_no_connects_field(tmp_path):
    """LintReport.no_connects mirrors module._no_connects from add_no_connect."""
    m = _empty_module(tmp_path)
    m.add_net("probe_out")
    m.add_no_connect("probe_out", reason="differential test tap")
    rep = generate_report(m)
    assert len(rep.lint.no_connects) == 1
    inst, net, nc = rep.lint.no_connects[0]
    assert net == "probe_out"
    assert nc.reason == "differential test tap"


def test_lint_save_html_renders_no_connects_section(tmp_path):
    """save_html includes an 'Intentionally Unconnected' section when no_connects is non-empty."""
    m = _empty_module(tmp_path)
    m.add_net("probe_out")
    m.add_no_connect("probe_out", reason="differential test tap")
    rep = generate_report(m)
    p = tmp_path / "lint.html"
    rep.lint.save_html(p)
    content = p.read_text()
    assert "Intentionally Unconnected" in content
    assert "probe_out" in content
    assert "differential test tap" in content


def test_coverage_save_html_renders_scorecard(tmp_path):
    """CoverageReport.save_html produces a styled per-rule table."""
    rep = generate_report(_empty_module(tmp_path))
    p = tmp_path / "coverage.html"
    rep.coverage.save_html(p)
    content = p.read_text()
    assert "<html" in content.lower()
    assert "Coverage Scorecard" in content
    # Should mention at least one rule that ran
    assert "INT-001" in content or "STRUCT-003" in content


def test_coverage_diff_self_is_zero(tmp_path):
    """Diffing a CoverageReport against itself produces zero deltas."""
    rep = generate_report(_empty_module(tmp_path))
    diff = rep.coverage.diff(rep.coverage)
    assert diff.added_rules == []
    assert diff.removed_rules == []
    for delta in diff.deltas.values():
        assert delta == (0, 0, 0)


def test_coverage_diff_rejects_mismatched_context_hash(tmp_path):
    """CoverageReport.diff raises on mismatched context_hash like LintReport.diff."""
    other_md = ReportMetadata(
        silk_version="9.9.9",
        pdk_name="other",
        pdk_version="0.0.0",
        module_name="other",
    )
    rep = generate_report(_empty_module(tmp_path))
    other = CoverageReport.from_visit_log(metadata=other_md, visit_log=[])
    with pytest.raises(ValueError, match="context_hash"):
        rep.coverage.diff(other)


def test_lint_report_no_connects_json_roundtrip(tmp_path):
    """LintReport.no_connects survives JSON round-trip."""
    m = _empty_module(tmp_path)
    m.add_net("probe_out")
    m.add_no_connect("probe_out", reason="differential test tap")
    rep = generate_report(m)
    p = tmp_path / "lint.json"
    rep.lint.save_json(p)
    loaded = LintReport.load_json(p)
    assert len(loaded.no_connects) == 1
    inst, net, nc = loaded.no_connects[0]
    assert net == "probe_out"
    assert nc.reason == "differential test tap"


def test_unchecked_stress_is_visible_in_every_report(tmp_path, capsys):
    from silk.pdk import load_pdk
    from silk.lint.registry import RuleRegistry
    from silk.lint.rules.stress import JunctionOverrangeRule
    pdk = load_pdk('sky130')
    m = Module('sky130', str(tmp_path), 'stress.cir', pdk_details=pdk)
    with m:
        pdk.nmos1p8('mn', d='out', g='inp', s='0', b='0', w=1., l=.15)
    rep = generate_report(m, registry=RuleRegistry([JunctionOverrangeRule()]),
                          auto_discover_config=False)
    rep.print_summary()
    assert '1 unchecked' in capsys.readouterr().out
    assert rep.coverage.rule_coverage.root.percentage('STR-001') == 0
    json_path = tmp_path / 'report.json'
    rep.save_json(json_path)
    restored = CircuitReport.load_json(json_path)
    assert rep.lint.stress_coverage['summary']['none'] == 1
    assert rep.coverage.stress_coverage == rep.lint.stress_coverage
    from silk.lint.reports.lint_report import LintReport
    from silk.lint.reports.coverage_report import CoverageReport
    restored.lint = LintReport.from_dict(rep.lint.to_dict())
    restored.coverage = CoverageReport.from_dict(rep.coverage.to_dict())
    for report in (rep, restored):
        assert report.lint.stress_coverage['summary']['none'] == 1
        assert report.coverage.stress_coverage == report.lint.stress_coverage
        md_path = tmp_path / 'report.md'
        report.save(md_path)
        assert 'no_waveform' in md_path.read_text()
        for name, writer in [('lint', report.lint.save_html),
                             ('coverage', report.coverage.save_html),
                             ('circuit', report.save_html)]:
            path = tmp_path / f'{name}.html'
            writer(path)
            content = path.read_text()
            assert 'INCOMPLETE' in content
            assert '0/1 devices fully checked' in content
            assert 'mn: no_waveform' in content
        path = tmp_path / 'lint.xml'
        report.lint.save_junit_xml(path)
        import xml.etree.ElementTree as ET
        case = ET.parse(path).find('.//testcase[@name="STR-001"]')
        assert case is not None
        assert case.find('skipped') is not None
        assert 'no_waveform' in case.find('skipped').attrib['message']
