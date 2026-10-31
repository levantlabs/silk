import pytest
from unittest.mock import MagicMock
from silk.lint.rules.domain import (
    RailMismatchRule,
    DomainCrossingRule,
    SupplyDisabledStressRule,
)
from silk.lint.severity import Severity
from silk.lint.violations import InterfaceViolation
from silk.lint.locations import NetLocation
from silk.circuit.modules import Module


def test_domain_rule_ids():
    assert RailMismatchRule().rule_id == "DOM-001"
    assert DomainCrossingRule().rule_id == "DOM-002"
    assert SupplyDisabledStressRule().rule_id == "DOM-003"


def test_domain_rule_category():
    for cls in (RailMismatchRule, DomainCrossingRule, SupplyDisabledStressRule):
        assert cls().category == "domain"


def test_domain_rule_severity_default_error():
    for cls in (RailMismatchRule, DomainCrossingRule, SupplyDisabledStressRule):
        assert cls().default_severity == Severity.ERROR


def test_domain_rule_rationale_non_empty():
    for cls in (RailMismatchRule, DomainCrossingRule, SupplyDisabledStressRule):
        assert len(cls().rationale) > 20


def test_domain_rule_applicable_to_module():
    for cls in (RailMismatchRule, DomainCrossingRule, SupplyDisabledStressRule):
        assert Module in cls().applicable_to


def test_domain_rules_yield_interface_violation_subclass(tmp_path):
    """Smoke test: rules return InterfaceViolation instances when they fire.

    Uses a minimal voltage-domain setup. The detection-semantics tests
    proper live in tests/test_voltage_domain.py and continue to verify
    check_voltage_domains() unchanged.
    """
    # Empty module — no bindings, all rules should yield zero violations
    pdk = MagicMock()
    pdk.model_names = []
    pdk._devices = {}
    m = Module('TEST', str(tmp_path), 'top.cir', pdk_details=pdk)
    for cls in (RailMismatchRule, DomainCrossingRule, SupplyDisabledStressRule):
        violations = list(cls().check(m))
        assert violations == []
