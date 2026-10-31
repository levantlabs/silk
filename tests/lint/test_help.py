import pytest
from silk.lint.help import help_rule, list_rules


def test_help_rule_returns_rationale_for_known_rule():
    text = help_rule("INT-001")
    assert "driver" in text.lower() or "conflict" in text.lower()
    assert len(text) > 50


def test_help_rule_includes_metadata():
    text = help_rule("INT-001")
    assert "INT-001" in text
    assert "interface" in text.lower()
    assert "error" in text.lower()


def test_help_rule_unknown_raises():
    with pytest.raises(KeyError, match="INT-999"):
        help_rule("INT-999")


def test_list_rules_returns_all_v1_rule_ids():
    ids = list_rules()
    expected = {
        "INT-001", "INT-002", "INT-003", "INT-004", "INT-005",
        "DOM-001", "DOM-002", "DOM-003",
        "STR-001",
        "STRUCT-001", "STRUCT-002", "STRUCT-003", "STRUCT-004",
        "STRUCT-005", "STRUCT-006", "STRUCT-007", "STRUCT-008",
        "STRUCT-011", "STRUCT-012", "STRUCT-013", "STRUCT-010",
        "BUS-002",
    }
    assert set(ids) == expected


def test_silk_top_level_has_lint_attribute():
    import silk
    assert hasattr(silk, "lint") or hasattr(silk, "RuleRegistry")


def test_silk_top_level_imports_no_connect():
    from silk.lint_markers import NoConnect
    assert NoConnect is not None


def test_silk_top_level_imports_lint_report():
    from silk.lint.reports import LintReport, CoverageReport
    assert LintReport is not None
    assert CoverageReport is not None
