import pytest
from silk.lint.reports.coverage_report import (
    CoverageReport, RuleCoverageNode, RuleCoverageReport,
)
from silk.lint.reports.metadata import ReportMetadata
from silk.lint.registry import VisitEvent, VisitOutcome
from silk.lint.locations import InstanceLocation


@pytest.fixture
def sample_metadata():
    return ReportMetadata(
        silk_version="0.5.0", pdk_name="skywater130",
        pdk_version="1.0.0", module_name="ldo",
    )


def test_rule_coverage_node_local_and_aggregate():
    root = RuleCoverageNode(module_name="top")
    child = RuleCoverageNode(module_name="sub")
    root.children.append(child)

    root.local["INT-001"] = (2, 0, 1)
    child.local["INT-001"] = (3, 0, 0)

    agg = root.aggregate
    assert agg["INT-001"] == (5, 0, 1)


def test_rule_coverage_node_percentage():
    node = RuleCoverageNode(module_name="m")
    node.local["X"] = (8, 0, 2)
    assert node.percentage("X") == pytest.approx(80.0)


def test_rule_coverage_node_excluded_not_in_denominator():
    node = RuleCoverageNode(module_name="m")
    node.local["X"] = (3, 7, 0)
    assert node.percentage("X") == pytest.approx(100.0)


def test_rule_coverage_node_zero_applicable_returns_100():
    node = RuleCoverageNode(module_name="m")
    node.local["Y"] = (0, 0, 0)
    assert node.percentage("Y") == pytest.approx(100.0)


def test_rule_coverage_node_missing_rule_returns_zero():
    node = RuleCoverageNode(module_name="m")
    assert node.percentage("Z") == pytest.approx(0.0)


def test_coverage_report_from_visit_log(sample_metadata):
    events = [
        VisitEvent(
            rule_id="INT-001",
            location=InstanceLocation(instance_path="top.m1"),
            outcome=VisitOutcome.PASS, timestamp=1.0,
        ),
        VisitEvent(
            rule_id="INT-001",
            location=InstanceLocation(instance_path="top.m2"),
            outcome=VisitOutcome.VIOLATION, timestamp=1.0,
        ),
        VisitEvent(
            rule_id="INT-001",
            location=InstanceLocation(instance_path="top.m3"),
            outcome=VisitOutcome.EXCLUDED, timestamp=1.0,
        ),
    ]
    rep = CoverageReport.from_visit_log(
        metadata=sample_metadata, visit_log=events,
    )
    assert "INT-001" in rep.rule_coverage.root.local
    checked, excluded, unchecked = rep.rule_coverage.root.local["INT-001"]
    assert checked == 2  # PASS and VIOLATION both count as checked
    assert excluded == 1
    assert unchecked == 0


def test_coverage_report_skipped_no_sim_counts_as_unchecked(sample_metadata):
    events = [
        VisitEvent(
            rule_id="STR-001",
            location=InstanceLocation(instance_path="top.m1"),
            outcome=VisitOutcome.SKIPPED_NO_SIM, timestamp=1.0,
        ),
    ]
    rep = CoverageReport.from_visit_log(
        metadata=sample_metadata, visit_log=events,
    )
    checked, excluded, unchecked = rep.rule_coverage.root.local["STR-001"]
    assert checked == 0
    assert unchecked == 1


def test_coverage_report_save_and_load_json(tmp_path, sample_metadata):
    rep = CoverageReport(
        metadata=sample_metadata,
        rule_coverage=RuleCoverageReport(root=RuleCoverageNode(module_name="ldo")),
    )
    rep.rule_coverage.root.local["INT-001"] = (5, 1, 2)
    p = tmp_path / "cov.json"
    rep.save_json(p)
    loaded = CoverageReport.load_json(p)
    assert loaded.metadata.context_hash == rep.metadata.context_hash
    assert loaded.rule_coverage.root.local["INT-001"] == (5, 1, 2)


def test_coverage_report_functional_coverage_reserved_null(tmp_path, sample_metadata):
    rep = CoverageReport(
        metadata=sample_metadata,
        rule_coverage=RuleCoverageReport(root=RuleCoverageNode(module_name="ldo")),
    )
    p = tmp_path / "cov.json"
    rep.save_json(p)
    import json
    data = json.loads(p.read_text())
    assert data["functional_coverage"] is None


def test_stress_counts_devices_and_keeps_failure_independent(sample_metadata, tmp_path):
    stress = {'summary': {'total': 3, 'full': 1, 'partial': 1, 'none': 1, 'failed': 1},
              'devices': [
                  {'instance_path': 'mn', 'coverage': 'partial', 'failed': True,
                   'reasons': ['no_ratings'], 'pairs': {'Vgs': 'fail', 'Vbs': 'unrated'}},
                  {'instance_path': 'mp', 'coverage': 'none', 'failed': False,
                   'reasons': ['no_waveform'], 'pairs': {'Vgs': 'unavailable'}},
                  {'instance_path': 'mq', 'coverage': 'full', 'failed': False,
                   'reasons': [], 'pairs': {'Vgs': 'pass'}},
              ]}
    event = VisitEvent(rule_id='STR-001', location=InstanceLocation(instance_path='top'),
                       outcome=VisitOutcome.VIOLATION, timestamp=1.,
                       context={'stress_coverage': stress})
    report = CoverageReport.from_visit_log(sample_metadata, [event, event])
    assert report.rule_coverage.root.local['STR-001'] == (2, 0, 4)
    assert report.rule_coverage.root.percentage('STR-001') == pytest.approx(100 / 3)
    assert report.stress_coverage['summary']['failed'] == 2
    restored = CoverageReport.from_dict(report.to_dict())
    assert restored.stress_coverage == report.stress_coverage
    restored.save_html(tmp_path / 'coverage.html')
    text = (tmp_path / 'coverage.html').read_text()
    assert '2/6 devices fully checked' in text
    assert 'mn: no_ratings' in text
    assert 'mp: no_waveform' in text
