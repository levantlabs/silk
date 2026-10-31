import pytest
import time
from silk.lint.registry import RuleRegistry, VisitEvent, VisitOutcome
from silk.lint.rule import Rule
from silk.lint.severity import Severity
from silk.lint.violations import Violation
from silk.lint.locations import ModuleLocation


class _DummyScope:
    """Stand-in for a Module during registry testing."""
    def __init__(self, name, elements=None):
        self.name = name
        self.elements = elements or []


class _AlwaysPassRule(Rule):
    rule_id = "TEST-PASS"
    category = "structural"
    default_severity = Severity.WARNING
    applicable_to = (_DummyScope,)
    rationale = "never fires"

    def check(self, scope):
        return iter([])


class _AlwaysFailRule(Rule):
    rule_id = "TEST-FAIL"
    category = "structural"
    default_severity = Severity.ERROR
    applicable_to = (_DummyScope,)
    rationale = "always fires"

    def check(self, scope):
        yield Violation(
            rule_id=self.rule_id,
            severity=self.severity,
            message="boom",
            location=ModuleLocation(module_name=scope.name),
        )


class _SimRequiredRule(Rule):
    rule_id = "TEST-SIM"
    category = "stress"
    default_severity = Severity.ERROR
    applicable_to = (_DummyScope,)
    requires_simulation = True
    rationale = "needs sim"

    def check(self, scope):
        yield Violation(
            rule_id=self.rule_id,
            severity=self.severity,
            message="sim-based",
            location=ModuleLocation(module_name=scope.name),
        )


def test_registry_empty_run_returns_no_violations():
    reg = RuleRegistry(rules=[])
    scope = _DummyScope(name="m")
    result = reg.run(scope)
    assert result.violations == []
    assert result.visit_log == []


def test_registry_dispatches_matching_rules():
    reg = RuleRegistry(rules=[_AlwaysPassRule(), _AlwaysFailRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope)
    assert len(result.violations) == 1
    assert result.violations[0].rule_id == "TEST-FAIL"


def test_registry_records_visit_events_for_both_rules():
    reg = RuleRegistry(rules=[_AlwaysPassRule(), _AlwaysFailRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope)
    rule_ids = {e.rule_id for e in result.visit_log}
    assert rule_ids == {"TEST-PASS", "TEST-FAIL"}


def test_registry_passes_outcome_recorded():
    reg = RuleRegistry(rules=[_AlwaysPassRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope)
    assert result.visit_log[0].outcome == VisitOutcome.PASS


def test_registry_violation_outcome_recorded():
    reg = RuleRegistry(rules=[_AlwaysFailRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope)
    assert result.visit_log[0].outcome == VisitOutcome.VIOLATION


def test_registry_skips_sim_required_when_no_sim_data():
    reg = RuleRegistry(rules=[_SimRequiredRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope, has_simulation_data=False)
    assert result.violations == []
    assert result.visit_log[0].outcome == VisitOutcome.SKIPPED_NO_SIM


def test_registry_runs_sim_required_when_sim_data_available():
    reg = RuleRegistry(rules=[_SimRequiredRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope, has_simulation_data=True)
    assert len(result.violations) == 1
    assert result.visit_log[0].outcome == VisitOutcome.VIOLATION


def test_registry_category_filter():
    reg = RuleRegistry(rules=[_AlwaysFailRule(), _SimRequiredRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope, categories={"structural"}, has_simulation_data=True)
    assert len(result.violations) == 1
    assert result.violations[0].rule_id == "TEST-FAIL"


def test_registry_run_category_convenience():
    reg = RuleRegistry(rules=[_AlwaysFailRule(), _SimRequiredRule()])
    scope = _DummyScope(name="m")
    result = reg.run_category("structural", scope)
    assert len(result.violations) == 1
    assert result.violations[0].rule_id == "TEST-FAIL"


def test_visit_event_has_timestamp():
    reg = RuleRegistry(rules=[_AlwaysFailRule()])
    scope = _DummyScope(name="m")
    before = time.time()
    result = reg.run(scope)
    after = time.time()
    assert before <= result.visit_log[0].timestamp <= after


def test_registry_profile_parameter_accepts_default():
    reg = RuleRegistry(rules=[_AlwaysFailRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope, profile="default")
    assert len(result.violations) == 1


def test_registry_profile_unknown_raises():
    reg = RuleRegistry(rules=[_AlwaysFailRule()])
    scope = _DummyScope(name="m")
    with pytest.raises(ValueError, match="profile"):
        reg.run(scope, profile="strict")


def test_registry_category_filter_excludes_other_rules_from_visit_log():
    reg = RuleRegistry(rules=[_AlwaysFailRule(), _SimRequiredRule()])
    scope = _DummyScope(name="m")
    result = reg.run(scope, categories={"structural"}, has_simulation_data=True)
    rule_ids = {e.rule_id for e in result.visit_log}
    assert "TEST-SIM" not in rule_ids
    assert "TEST-FAIL" in rule_ids


def test_visit_event_location_reflects_scope_name():
    reg = RuleRegistry(rules=[_AlwaysFailRule()])
    scope = _DummyScope(name="my_module")
    result = reg.run(scope)
    from silk.lint.locations import ModuleLocation
    assert isinstance(result.visit_log[0].location, ModuleLocation)
    assert result.visit_log[0].location.module_name == "my_module"


def test_registry_add_rule_appends():
    reg = RuleRegistry(rules=[])
    reg.add_rule(_AlwaysPassRule())
    assert len(reg.rules) == 1
    assert reg.rules[0].rule_id == "TEST-PASS"


def test_registry_default_constructs_with_catalog():
    reg = RuleRegistry.default()
    assert isinstance(reg, RuleRegistry)
    assert len(reg.rules) == 22  # 5 interface + 3 domain + 1 stress + 12 structural + 1 bus rules


from silk.lint.exclusions import ScopeExclusion


def test_registry_is_excluded_returns_match():
    reg = RuleRegistry(
        rules=[_AlwaysFailRule()],
        exclusions=[ScopeExclusion(
            target_type="instance",
            target_path="top.sram",
            reason="vendor IP",
        )],
    )
    assert reg.is_excluded("top.sram") is not None
    assert reg.is_excluded("top.sram.cell") is not None
    assert reg.is_excluded("top.opamp") is None


def test_registry_load_exclusions_from_yaml(tmp_path):
    p = tmp_path / "exclusions.yaml"
    p.write_text(
        "exclusions:\n"
        "  - target:\n"
        "      type: instance\n"
        "      path: top.macro\n"
        "    reason: vendor IP\n"
    )
    reg = RuleRegistry(rules=[])
    reg.load_exclusions(p)
    assert reg.is_excluded("top.macro") is not None


def test_registry_load_waivers_from_yaml(tmp_path):
    from silk.lint.waivers import Waiver
    p = tmp_path / "waivers.yaml"
    p.write_text(
        "waivers:\n"
        "  - rule_id: INT-001\n"
        "    matcher:\n"
        "      location_type: net\n"
        "      fields:\n"
        "        net_name: vout\n"
        "    reason: intentional\n"
    )
    reg = RuleRegistry(rules=[])
    reg.load_waivers(p)
    assert len(reg._waivers) == 1
    assert reg._waivers[0].rule_id == "INT-001"
