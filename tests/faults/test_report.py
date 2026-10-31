"""CampaignReport markdown / HTML / JSON rendering (Task 9)."""

import json
from unittest.mock import MagicMock

from silk.faults import FaultSet
from silk.faults.campaign import CampaignResult, RunResult
from silk.faults.report import CampaignReport
from silk.faults.types import FaultManifest


def _outcome(spec_name, status='PASS', value=0.0):
    o = MagicMock()
    o.spec.name = spec_name
    o.spec.target = 0.0
    o.spec.op = '>='
    o.spec.units = ''
    o.status = status
    o.measurement.value = value
    return o


def _run(i, status='passed', outcomes=None, applied=None):
    return RunResult(
        run_id=i, seed=i,
        status=status,
        manifest=FaultManifest(seed=i, input_spec={}, applied=applied or []),
        spec_outcomes=outcomes or [],
    )


def _result(runs):
    return CampaignResult(
        campaign_id='c', module_name='m', n_runs=len(runs),
        fault_set=FaultSet(), seed=0, runs=runs, out_dir='/tmp')


def test_campaignreport_summary_contains_pass_count():
    runs = [_run(0, 'passed', [_outcome('adc_sndr', 'PASS', 25.0)]),
            _run(1, 'failed', [_outcome('adc_sndr', 'FAIL', 15.0)])]
    r = CampaignReport(_result(runs))
    s = r.summary()
    lower = s.lower()
    assert '1 passed' in s or 'passed: 1' in lower
    assert '1 failed' in s or 'failed: 1' in lower


def test_campaignreport_save_json_roundtrip(tmp_path):
    runs = [_run(0, 'passed', [_outcome('adc_sndr', 'PASS', 25.0)])]
    r = CampaignReport(_result(runs))
    path = tmp_path / 'c.json'
    r.save_json(str(path))
    data = json.loads(path.read_text())
    assert 'runs' in data
    assert len(data['runs']) == 1


def test_campaignreport_save_md_writes_file(tmp_path):
    runs = [_run(0, 'passed', [_outcome('adc_sndr', 'PASS', 25.0)])]
    r = CampaignReport(_result(runs))
    path = tmp_path / 'c.md'
    r.save(str(path))
    text = path.read_text()
    assert '# Campaign' in text
    assert 'adc_sndr' in text
    assert 'Pass' in text or 'pass' in text


def test_campaignreport_save_html_writes_file(tmp_path):
    runs = [_run(0, 'passed', [_outcome('adc_sndr', 'PASS', 25.0)])]
    r = CampaignReport(_result(runs))
    path = tmp_path / 'c.html'
    r.save_html(str(path))
    assert path.exists()
    assert path.read_text().lstrip().startswith('<')
