import pytest
from datetime import date
from silk.lint.waivers import Waiver, WaiverFile, WaiverMatcher, apply_waivers
from silk.lint.violations import InterfaceViolation
from silk.lint.severity import Severity
from silk.lint.locations import NetLocation, PortLocation


def test_waiver_requires_reason():
    with pytest.raises(TypeError):
        Waiver(rule_id="INT-001", matcher=WaiverMatcher(location_type="net"))


def test_waiver_empty_reason_rejected():
    with pytest.raises(ValueError, match="reason"):
        Waiver(
            rule_id="INT-001",
            matcher=WaiverMatcher(location_type="net"),
            reason="",
        )


def test_waiver_minimal_valid():
    w = Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "vout"}),
        reason="intentional dotted-OR",
    )
    assert w.rule_id == "INT-001"
    assert w.expires is None
    assert w.author == ""


def test_waiver_matches_interface_violation_by_rule_and_net():
    w = Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "vout"}),
        reason="x",
    )
    v = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="vout", port_names=["o1"]),
    )
    assert w.matches(v)


def test_waiver_non_match_different_rule():
    w = Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "vout"}),
        reason="x",
    )
    v = InterfaceViolation(
        rule_id="INT-002", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="vout", port_names=[]),
    )
    assert not w.matches(v)


def test_waiver_non_match_different_net():
    w = Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "vout"}),
        reason="x",
    )
    v = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="other", port_names=[]),
    )
    assert not w.matches(v)


def test_waiver_matches_port_location():
    w = Waiver(
        rule_id="STRUCT-001",
        matcher=WaiverMatcher(location_type="port", fields={"port_name": "debug_out"}),
        reason="debug tap",
    )
    v = InterfaceViolation(
        rule_id="STRUCT-001", severity=Severity.WARNING, message="",
        location=PortLocation(port_name="debug_out", module_name="top"),
    )
    assert w.matches(v)


def test_waiver_expired_does_not_match():
    w = Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "vout"}),
        reason="x",
        expires=date(2020, 1, 1),
    )
    v = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="vout", port_names=[]),
    )
    assert not w.matches(v)


def test_waiver_file_loads_yaml(tmp_path):
    p = tmp_path / "waivers.yaml"
    p.write_text(
        "waivers:\n"
        "  - rule_id: INT-001\n"
        "    matcher:\n"
        "      location_type: net\n"
        "      fields:\n"
        "        net_name: vout_ref\n"
        "    reason: intentional dotted-OR for test mux\n"
        "    author: jdoe\n"
        "    expires: 2026-06-01\n"
    )
    wf = WaiverFile.load(p)
    assert len(wf.waivers) == 1
    w = wf.waivers[0]
    assert w.rule_id == "INT-001"
    assert w.reason == "intentional dotted-OR for test mux"
    assert w.author == "jdoe"
    assert w.expires == date(2026, 6, 1)


def test_waiver_file_missing_reason_rejected(tmp_path):
    p = tmp_path / "waivers.yaml"
    p.write_text(
        "waivers:\n"
        "  - rule_id: INT-001\n"
        "    matcher:\n"
        "      location_type: net\n"
        "      fields:\n"
        "        net_name: vout\n"
    )
    with pytest.raises(ValueError, match="reason"):
        WaiverFile.load(p)


def test_apply_waivers_separates_suppressed():
    v_match = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="vout", port_names=[]),
    )
    v_unmatched = InterfaceViolation(
        rule_id="INT-002", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="vout", port_names=[]),
    )
    waivers = [
        Waiver(
            rule_id="INT-001",
            matcher=WaiverMatcher(location_type="net", fields={"net_name": "vout"}),
            reason="x",
        )
    ]
    remaining, suppressed = apply_waivers([v_match, v_unmatched], waivers)
    assert len(remaining) == 1
    assert remaining[0].rule_id == "INT-002"
    assert len(suppressed) == 1
    assert suppressed[0].rule_id == "INT-001"
