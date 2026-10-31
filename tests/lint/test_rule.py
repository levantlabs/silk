import pytest
from silk.lint.rule import Rule, RuleCategory
from silk.lint.severity import Severity
from silk.lint.violations import Violation
from silk.lint.locations import ModuleLocation


def test_rule_requires_rule_id():
    class MyRule(Rule):
        category = "structural"
        default_severity = Severity.WARNING
        applicable_to = (object,)
        rationale = "test"

        def check(self, scope):
            return iter([])

    with pytest.raises(TypeError, match="rule_id"):
        MyRule()


def test_rule_with_all_required_fields():
    class MyRule(Rule):
        rule_id = "TEST-001"
        category = "structural"
        default_severity = Severity.WARNING
        applicable_to = (object,)
        rationale = "test rationale"

        def check(self, scope):
            return iter([])

    r = MyRule()
    assert r.rule_id == "TEST-001"
    assert r.category == "structural"
    assert r.default_severity == Severity.WARNING
    assert r.rationale == "test rationale"
    assert r.requires_simulation is False  # default
    assert r.config == {}  # default


def test_rule_config_override():
    class MyRule(Rule):
        rule_id = "TEST-002"
        category = "structural"
        default_severity = Severity.WARNING
        applicable_to = (object,)
        rationale = "x"
        config_schema = {"margin_pct": float}

        def check(self, scope):
            return iter([])

    r = MyRule(config={"margin_pct": 5.0})
    assert r.config["margin_pct"] == 5.0


def test_rule_severity_override():
    class MyRule(Rule):
        rule_id = "TEST-003"
        category = "structural"
        default_severity = Severity.WARNING
        applicable_to = (object,)
        rationale = "x"

        def check(self, scope):
            return iter([])

    r = MyRule(severity_override=Severity.ERROR)
    assert r.severity == Severity.ERROR


def test_rule_severity_defaults_to_class_default():
    class MyRule(Rule):
        rule_id = "TEST-004"
        category = "structural"
        default_severity = Severity.INFO
        applicable_to = (object,)
        rationale = "x"

        def check(self, scope):
            return iter([])

    assert MyRule().severity == Severity.INFO


def test_rule_check_yields_violations():
    class YieldingRule(Rule):
        rule_id = "TEST-005"
        category = "structural"
        default_severity = Severity.WARNING
        applicable_to = (object,)
        rationale = "x"

        def check(self, scope):
            yield Violation(
                rule_id=self.rule_id,
                severity=self.severity,
                message="found it",
                location=ModuleLocation(module_name="m"),
            )

    r = YieldingRule()
    violations = list(r.check(scope=None))
    assert len(violations) == 1
    assert violations[0].rule_id == "TEST-005"


def test_rule_requires_rationale():
    with pytest.raises(TypeError, match="rationale"):
        class NoRationale(Rule):
            rule_id = "TEST-006"
            category = "structural"
            default_severity = Severity.WARNING
            applicable_to = (object,)

            def check(self, scope):
                return iter([])

        NoRationale()


def test_rule_rejects_invalid_category():
    class BadCat(Rule):
        rule_id = "TEST-007"
        category = "bogus"
        default_severity = Severity.WARNING
        applicable_to = (object,)
        rationale = "x"

        def check(self, scope):
            return iter([])

    with pytest.raises(TypeError, match="category"):
        BadCat()
