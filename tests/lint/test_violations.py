import pytest
from silk.lint.severity import Severity
from silk.lint.locations import (
    NetLocation, TerminalLocation, ModuleLocation,
)
from silk.lint.violations import (
    Violation, InterfaceViolation, JunctionViolation,
)


def test_violation_base_fields():
    v = Violation(
        rule_id="INT-001",
        severity=Severity.ERROR,
        message="driver conflict",
        location=NetLocation(net_name="vout", port_names=["o1", "o2"]),
    )
    assert v.rule_id == "INT-001"
    assert v.severity == Severity.ERROR
    assert v.message == "driver conflict"
    assert v.context == {}


def test_violation_context_default_factory():
    v1 = Violation(rule_id="X", severity=Severity.INFO, message="",
                   location=ModuleLocation(module_name="a"))
    v2 = Violation(rule_id="X", severity=Severity.INFO, message="",
                   location=ModuleLocation(module_name="b"))
    v1.context["corner"] = "tt"
    assert "corner" not in v2.context


def test_interface_violation_is_violation_subclass():
    v = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="x",
        location=NetLocation(net_name="vout", port_names=["o1"]),
    )
    assert isinstance(v, Violation)


def test_interface_violation_backward_compat_net_property():
    v = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="x",
        location=NetLocation(net_name="vout", port_names=["o1", "o2"]),
    )
    assert v.net == "vout"
    assert v.ports == ["o1", "o2"]
    assert v.kind == "INT-001"  # alias for rule_id


def test_junction_violation_subclass_with_specific_fields():
    v = JunctionViolation(
        rule_id="STR-001",
        severity=Severity.ERROR,
        message="Vgs > rating",
        location=TerminalLocation(
            instance_path="m_pass", terminal="gs", terminal_pair=("g", "s"),
        ),
        device_type="nmos",
        measured_v=2.1,
        rated_max=1.98,
        worst_time=1.2e-6,
    )
    assert isinstance(v, Violation)
    assert v.device_type == "nmos"
    assert v.measured_v == 2.1
    assert v.rated_max == 1.98
    assert v.worst_time == 1.2e-6
    # Backward-compat properties
    assert v.instance == "m_pass"
    assert v.terminal_pair == ("g", "s")
    assert v.kind == "STR-001"


def test_violations_distinguishable_by_isinstance():
    iv = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="x",
        location=NetLocation(net_name="a", port_names=[]),
    )
    jv = JunctionViolation(
        rule_id="STR-001", severity=Severity.ERROR, message="x",
        location=TerminalLocation(instance_path="m", terminal="d"),
    )
    assert isinstance(iv, InterfaceViolation)
    assert not isinstance(iv, JunctionViolation)
    assert isinstance(jv, JunctionViolation)
    assert not isinstance(jv, InterfaceViolation)
