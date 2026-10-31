import json
import pytest
from silk.lint.reports.metadata import ReportMetadata
from silk.lint.reports.lint_report import LintReport
from silk.lint.violations import InterfaceViolation, JunctionViolation
from silk.lint.severity import Severity
from silk.lint.locations import NetLocation, TerminalLocation
from silk.lint.waivers import Waiver, WaiverMatcher


@pytest.fixture
def sample_metadata():
    return ReportMetadata(
        silk_version="0.5.0",
        pdk_name="skywater130",
        pdk_version="1.0.0",
        module_name="ldo",
    )


@pytest.fixture
def sample_violation():
    return InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR,
        message="driver conflict on vout",
        location=NetLocation(net_name="vout", port_names=["o1", "o2"]),
    )


def test_lint_report_basic_construction(sample_metadata, sample_violation):
    rep = LintReport(
        metadata=sample_metadata,
        violations=[sample_violation],
        waivers_applied=[],
    )
    assert len(rep.violations) == 1
    assert rep.violations[0].rule_id == "INT-001"


def test_lint_report_summary_counts_by_severity(sample_metadata):
    violations = [
        InterfaceViolation(
            rule_id="INT-001", severity=Severity.ERROR, message="",
            location=NetLocation(net_name=f"n{i}", port_names=[]),
        )
        for i in range(3)
    ]
    violations.append(InterfaceViolation(
        rule_id="STRUCT-001", severity=Severity.WARNING, message="",
        location=NetLocation(net_name="nw", port_names=[]),
    ))
    rep = LintReport(metadata=sample_metadata, violations=violations, waivers_applied=[])
    s = rep.summary()
    assert s["errors"] == 3
    assert s["warnings"] == 1
    assert s["infos"] == 0
    assert s["waived"] == 0


def test_lint_report_save_and_load_json_roundtrip(tmp_path, sample_metadata, sample_violation):
    rep = LintReport(
        metadata=sample_metadata,
        violations=[sample_violation],
        waivers_applied=[],
    )
    p = tmp_path / "lint.json"
    rep.save_json(p)

    loaded = LintReport.load_json(p)
    assert loaded.metadata.context_hash == rep.metadata.context_hash
    assert len(loaded.violations) == 1
    v = loaded.violations[0]
    assert v.rule_id == "INT-001"
    assert isinstance(v, InterfaceViolation)
    assert isinstance(v.location, NetLocation)
    assert v.location.net_name == "vout"
    assert v.location.port_names == ["o1", "o2"]


def test_lint_report_json_has_schema_version(tmp_path, sample_metadata):
    rep = LintReport(metadata=sample_metadata, violations=[], waivers_applied=[])
    p = tmp_path / "lint.json"
    rep.save_json(p)
    data = json.loads(p.read_text())
    assert data["metadata"]["schema_version"] == "1.0"


def test_lint_report_junction_violation_roundtrip(tmp_path, sample_metadata):
    """JunctionViolation-specific fields survive JSON roundtrip."""
    jv = JunctionViolation(
        rule_id="STR-001", severity=Severity.ERROR, message="vds out",
        location=TerminalLocation(
            instance_path="m1", terminal="Vds", terminal_pair=("d", "s"),
        ),
        device_type="nmos1p8",
        measured_v=2.5,
        rated_max=1.98,
        worst_time=1.5e-6,
    )
    rep = LintReport(metadata=sample_metadata, violations=[jv], waivers_applied=[])
    p = tmp_path / "lint.json"
    rep.save_json(p)
    loaded = LintReport.load_json(p)
    v = loaded.violations[0]
    assert isinstance(v, JunctionViolation)
    assert v.device_type == "nmos1p8"
    assert v.measured_v == 2.5
    assert v.rated_max == 1.98
    assert v.worst_time == 1.5e-6
    assert isinstance(v.location, TerminalLocation)
    assert v.location.terminal_pair == ("d", "s")  # tuple, not list


def test_lint_report_waiver_roundtrip(tmp_path, sample_metadata):
    from datetime import date
    w = Waiver(
        rule_id="INT-001",
        matcher=WaiverMatcher(location_type="net", fields={"net_name": "vout"}),
        reason="intentional",
        author="jdoe",
        expires=date(2026, 6, 1),
    )
    rep = LintReport(metadata=sample_metadata, violations=[], waivers_applied=[w])
    p = tmp_path / "lint.json"
    rep.save_json(p)
    loaded = LintReport.load_json(p)
    assert len(loaded.waivers_applied) == 1
    lw = loaded.waivers_applied[0]
    assert lw.rule_id == "INT-001"
    assert lw.reason == "intentional"
    assert lw.author == "jdoe"
    assert lw.expires == date(2026, 6, 1)
    assert lw.matcher.location_type == "net"
    assert lw.matcher.fields == {"net_name": "vout"}


# === Task 15: HTML, JUnit XML, diff, compliance matrix ===
import xml.etree.ElementTree as ET


def test_lint_report_save_html(tmp_path, sample_metadata, sample_violation):
    rep = LintReport(
        metadata=sample_metadata, violations=[sample_violation], waivers_applied=[],
    )
    p = tmp_path / "lint.html"
    rep.save_html(p)
    content = p.read_text()
    assert "<html" in content.lower()
    assert "INT-001" in content
    assert "vout" in content
    assert "driver conflict" in content


def test_lint_report_save_junit_xml(tmp_path, sample_metadata):
    violations = [
        InterfaceViolation(
            rule_id="INT-001", severity=Severity.ERROR, message="bad driver",
            location=NetLocation(net_name="vout", port_names=[]),
        ),
        InterfaceViolation(
            rule_id="STRUCT-001", severity=Severity.WARNING, message="unused port",
            location=NetLocation(net_name="unused", port_names=[]),
        ),
    ]
    rep = LintReport(
        metadata=sample_metadata, violations=violations, waivers_applied=[],
    )
    p = tmp_path / "junit.xml"
    rep.save_junit_xml(p)
    tree = ET.parse(p)
    root = tree.getroot()
    assert root.tag == "testsuites"
    testcases = root.findall(".//testcase")
    assert len(testcases) >= 2
    failures = root.findall(".//failure")
    assert len(failures) == 1


def test_lint_report_diff_new_violation(sample_metadata):
    v1_old = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="vout", port_names=[]),
    )
    v2_new = InterfaceViolation(
        rule_id="INT-002", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="vdd", port_names=[]),
    )
    old_rep = LintReport(
        metadata=sample_metadata, violations=[v1_old], waivers_applied=[],
    )
    new_rep = LintReport(
        metadata=sample_metadata, violations=[v1_old, v2_new], waivers_applied=[],
    )
    diff = new_rep.diff(old_rep)
    assert len(diff.added) == 1
    assert diff.added[0].rule_id == "INT-002"
    assert len(diff.removed) == 0


def test_lint_report_diff_fixed_violation(sample_metadata):
    v1 = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="vout", port_names=[]),
    )
    old_rep = LintReport(metadata=sample_metadata, violations=[v1], waivers_applied=[])
    new_rep = LintReport(metadata=sample_metadata, violations=[], waivers_applied=[])
    diff = new_rep.diff(old_rep)
    assert len(diff.removed) == 1
    assert diff.removed[0].rule_id == "INT-001"


def test_lint_report_diff_rejects_mismatched_context_hash(sample_metadata):
    other_metadata = ReportMetadata(
        silk_version="0.6.0",
        pdk_name="skywater130",
        pdk_version="1.0.0",
        module_name="ldo",
    )
    v = InterfaceViolation(
        rule_id="INT-001", severity=Severity.ERROR, message="",
        location=NetLocation(net_name="a", port_names=[]),
    )
    a = LintReport(metadata=sample_metadata, violations=[v], waivers_applied=[])
    b = LintReport(metadata=other_metadata, violations=[], waivers_applied=[])
    with pytest.raises(ValueError, match="context_hash"):
        a.diff(b)


def test_lint_report_compliance_matrix_counts_by_rule(sample_metadata):
    violations = [
        InterfaceViolation(
            rule_id="INT-001", severity=Severity.ERROR, message="",
            location=NetLocation(net_name=f"n{i}", port_names=[]),
        )
        for i in range(3)
    ]
    waivers = [
        Waiver(
            rule_id="STRUCT-001",
            matcher=WaiverMatcher(location_type="port", fields={"port_name": "debug"}),
            reason="dev",
        )
    ]
    rep = LintReport(
        metadata=sample_metadata, violations=violations, waivers_applied=waivers,
    )
    matrix = rep.compliance_matrix()
    assert matrix["INT-001"]["violations"] == 3
    assert matrix["INT-001"]["waivers"] == 0
    assert matrix["STRUCT-001"]["violations"] == 0
    assert matrix["STRUCT-001"]["waivers"] == 1


@pytest.mark.parametrize('existing', [False, True])
@pytest.mark.parametrize('subclass', [False, True])
def test_unsupported_location_save_preserves_destination(tmp_path, sample_metadata, existing, subclass):
    class UnsupportedNet(NetLocation):
        pass

    location = UnsupportedNet('vdd') if subclass else object()
    report = LintReport(metadata=sample_metadata, waivers_applied=[], violations=[
        InterfaceViolation(rule_id='INT-901', severity=Severity.ERROR,
                           message='unsupported', location=location)])
    path = tmp_path / 'unsupported.json'
    if existing:
        path.write_bytes(b'keep existing bytes')
    with pytest.raises(TypeError, match=type(location).__name__):
        report.save_json(path)
    assert path.read_bytes() == b'keep existing bytes' if existing else not path.exists()
