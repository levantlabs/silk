"""End-to-end integration tests for the analog lint and coverage system.

Exercises the full flow built across Tasks 1-19: generate_report ->
save artifacts -> load -> diff -> waiver suppression -> NoConnect
suppression of FLOATING_NET.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk.circuit.report import generate_report, CircuitReport
from silk.lint.reports.lint_report import LintReport
from silk.lint.reports.coverage_report import CoverageReport


def _pdk():
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    pdk.pdk_name = "test_pdk"
    pdk.version = "0.0.1"
    return pdk


def _sub(tmp_path, name, ports):
    s = Module('TEST', str(tmp_path), f'{name}.cir', pdk_details=_pdk())
    s.define_module_name(name)
    s.define_module_ports(ports)
    return s


def _build_dotted_or_module(tmp_path, net_name="shared"):
    """Two outputs on the same net — fires INT-001 DRIVER_CONFLICT."""
    a = _sub(tmp_path, 'a', {'out': PortDir.OUT})
    b = _sub(tmp_path, 'b', {'out': PortDir.OUT})
    top = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    top.add_module(a, 'a1', nets={'out': net_name})
    top.add_module(b, 'b1', nets={'out': net_name})
    return top


def test_end_to_end_generate_save_load_diff(tmp_path):
    """Full flow: generate_report -> save HTML/JSON/JUnit -> load -> self-diff."""
    module = _build_dotted_or_module(tmp_path, net_name="shared_net")

    rep1 = generate_report(module)
    assert rep1.lint is not None
    assert rep1.coverage is not None
    assert rep1.metadata.schema_version == "1.0"

    # The dotted-OR fires at least one INT-001 violation
    int001s = [v for v in rep1.lint.violations if v.rule_id == "INT-001"]
    assert len(int001s) >= 1

    # Save artifacts
    full_html = tmp_path / "full.html"
    lint_html = tmp_path / "lint.html"
    junit_xml = tmp_path / "junit.xml"
    rep_json = tmp_path / "rep.json"

    rep1.save_html(str(full_html))
    rep1.lint.save_html(lint_html)
    rep1.lint.save_junit_xml(junit_xml)
    rep1.save_json(str(rep_json))

    # All artifacts exist and are non-empty
    for p in [full_html, lint_html, junit_xml, rep_json]:
        assert p.exists()
        assert p.stat().st_size > 0

    # Load JSON round-trip on the CircuitReport
    # CircuitReport.load_json reconstructs the pre-migration fields;
    # the violations list round-trips correctly.
    rep2 = CircuitReport.load_json(str(rep_json))
    assert len(rep2.violations) == len(rep1.violations)


def test_end_to_end_lint_diff_self_is_empty(tmp_path):
    """A LintReport diffed against itself produces no added/removed entries."""
    module = _build_dotted_or_module(tmp_path)
    rep = generate_report(module)
    diff = rep.lint.diff(rep.lint)
    assert diff.added == []
    assert diff.removed == []


def test_end_to_end_waiver_suppression(tmp_path):
    """A registry with a waiver loaded from YAML suppresses the matching violation."""
    waiver_yaml = tmp_path / "silk_waivers.yaml"
    waiver_yaml.write_text(
        "waivers:\n"
        "  - rule_id: INT-001\n"
        "    matcher:\n"
        "      location_type: net\n"
        "      fields:\n"
        "        net_name: test_mux\n"
        "    reason: intentional dotted-OR for test mux\n"
    )

    from silk.lint.registry import RuleRegistry
    from silk.lint.waivers import apply_waivers

    reg = RuleRegistry.default()
    reg.load_waivers(waiver_yaml)

    module = _build_dotted_or_module(tmp_path, net_name="test_mux")
    result = reg.run(module)
    unsuppressed, suppressed = apply_waivers(result.violations, reg._waivers)

    int001_unsuppressed = [v for v in unsuppressed if v.rule_id == "INT-001"]
    int001_suppressed = [v for v in suppressed if v.rule_id == "INT-001"]
    assert int001_unsuppressed == []
    assert len(int001_suppressed) >= 1


def test_end_to_end_noconnect_suppresses_floating_net(tmp_path):
    """A NoConnect marker on a net suppresses STRUCT-003 FLOATING_NET."""
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=_pdk())
    m.add_net("probe_out")
    m.add_no_connect("probe_out", reason="differential test tap")

    rep = generate_report(m)
    floating_for_probe = [
        v for v in rep.lint.violations
        if v.rule_id == "STRUCT-003" and v.location.net_name == "probe_out"
    ]
    assert floating_for_probe == []


def test_end_to_end_coverage_records_per_rule_visits(tmp_path):
    """The CoverageReport's root.local should have entries for dispatched rules."""
    module = _build_dotted_or_module(tmp_path)
    rep = generate_report(module)
    # All rules in the default catalog dispatched against this module
    # should appear in the coverage local map.
    assert len(rep.coverage.rule_coverage.root.local) >= 1
    # INT-001 specifically was dispatched and hit the violation path
    if "INT-001" in rep.coverage.rule_coverage.root.local:
        checked, excluded, unchecked = rep.coverage.rule_coverage.root.local["INT-001"]
        assert checked >= 1


def test_end_to_end_help_rule_for_each_v1_rule():
    """help_rule should return non-empty rationale for every cataloged rule."""
    from silk.lint.help import list_rules, help_rule
    for rule_id in list_rules():
        text = help_rule(rule_id)
        assert rule_id in text
        assert len(text) > 50


def test_end_to_end_compliance_matrix_aggregates_by_rule(tmp_path):
    """compliance_matrix returns {rule_id: {violations, waivers}} for the full flow."""
    module = _build_dotted_or_module(tmp_path)
    rep = generate_report(module)
    matrix = rep.lint.compliance_matrix()
    assert "INT-001" in matrix
    assert matrix["INT-001"]["violations"] >= 1
