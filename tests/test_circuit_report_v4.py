"""Every new report is v5, with complete evidence and aggregate provenance."""
import json
from pathlib import Path

import pytest

from silk.circuit.report import CircuitReport, generate_report
from silk.specs import Measurement
from tests.test_aggregate_measurement import aggregate
from tests.test_circuit_report_v3 import _top_with_instances

FIXTURES = Path(__file__).parent / 'fixtures/reports'


def test_ordinary_report_is_v5(tmp_path):
    report = generate_report(_top_with_instances(tmp_path), auto_discover_config=False)
    path = tmp_path / 'ordinary.json'
    report.save_json(path)
    assert json.loads(path.read_text())['schema_version'] == 5


def test_v3_complete_evidence_survives_resave(tmp_path):
    source = FIXTURES / 'circuit_report_v3_wire.json'
    before = source.read_bytes()
    report = CircuitReport.load_json(source)
    path = tmp_path / 'upgraded.json'
    report.save_json(path)
    upgraded = json.loads(path.read_text())
    assert upgraded['schema_version'] == 5
    assert upgraded['test_results'] == json.loads(before)['test_results']
    assert source.read_bytes() == before


def test_aggregate_and_complete_evidence_roundtrip(tmp_path):
    from silk.circuit.spec_report import SpecReport
    module = _top_with_instances(tmp_path)
    spec = module.declare_spec('gain', 'gain_dB', '<=', 10)
    m = Measurement(source='spice', value=2., sim_run_id='campaign-1', aggregate=aggregate())
    spec.record_measurement(m)
    evidence = SpecReport('ac', 'top', {'complex': 1+2j, 'missing': float('nan')})
    report = generate_report(module, spec_reports=[evidence], auto_discover_config=False)
    path = tmp_path / 'aggregate.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    assert restored.spec_reconciliation.outcomes[0].measurement.aggregate == m.aggregate
    assert restored.test_results[0].metrics['complex'] == 1+2j
    assert restored.test_results[0].metrics['missing'] != restored.test_results[0].metrics['missing']
    restored.save_json(tmp_path / 'again.json')
    assert json.loads(path.read_text()) == json.loads((tmp_path / 'again.json').read_text())


def test_old_reader_refuses_v5(tmp_path, monkeypatch):
    import silk.circuit.report as reports
    report = generate_report(_top_with_instances(tmp_path), auto_discover_config=False)
    path = tmp_path / 'v5.json'
    report.save_json(path)
    monkeypatch.setattr(reports, 'REPORT_JSON_SCHEMA_VERSION', 4)
    with pytest.raises(ValueError, match='version 5'):
        CircuitReport.load_json(path)


def test_frozen_v4_fixture(tmp_path):
    source = FIXTURES / 'circuit_report_v4_wire.json'
    report = CircuitReport.load_json(source)
    assert report.spec_reconciliation.outcomes[0].measurement.aggregate == aggregate()
    path = tmp_path / 'roundtrip.json'
    report.save_json(path)
    upgraded = json.loads(path.read_text())
    row = upgraded['spec_reconciliation']['outcomes'][0]
    assert row.pop('latest_attempt') is None
    assert row.pop('last_successful_attempt') is None
    expected = json.loads(source.read_text())
    expected['schema_version'] = 5
    assert upgraded == expected


@pytest.mark.parametrize('bad', [{'scheme': 'unknown'}, {'scheme': 'silk.measurement.aggregate/v1'}])
def test_malformed_aggregate_names_report_file(tmp_path, bad):
    wire = json.loads((FIXTURES / 'circuit_report_v4_wire.json').read_text())
    wire['spec_reconciliation']['outcomes'][0]['measurement']['aggregate'] = bad
    path = tmp_path / 'bad-aggregate.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError, match='bad-aggregate.json.*aggregate'):
        CircuitReport.load_json(path)


def test_legacy_measurement_has_no_aggregate(tmp_path):
    wire = json.loads((FIXTURES / 'circuit_report_v4_wire.json').read_text())
    wire['schema_version'] = 3
    del wire['spec_reconciliation']['outcomes'][0]['measurement']['aggregate']
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(wire))
    assert CircuitReport.load_json(path).spec_reconciliation.outcomes[0].measurement.aggregate is None


@pytest.mark.parametrize('field,bad', [('metric', 'unrelated_metric'),
                                       ('campaign_id', 'another_campaign')])
@pytest.mark.parametrize('existing', [False, True])
def test_save_refuses_contradictory_aggregate_before_opening(tmp_path, field, bad, existing):
    from dataclasses import replace
    report = CircuitReport.load_json(FIXTURES / 'circuit_report_v4_wire.json')
    measurement = report.spec_reconciliation.outcomes[0].measurement
    measurement.aggregate = replace(measurement.aggregate, **{field: bad})
    path = tmp_path / 'must-not-write.json'
    if existing:
        path.write_bytes(b'keep existing evidence')
    with pytest.raises(ValueError, match='aggregate.*(metric|campaign_id)'):
        report.save_json(path)
    if existing:
        assert path.read_bytes() == b'keep existing evidence'
    else:
        assert not path.exists()


@pytest.mark.parametrize('field,bad', [('metric', 'unrelated_metric'),
                                       ('campaign_id', 'another_campaign')])
def test_load_refuses_contradictory_aggregate_with_outcome_context(tmp_path, field, bad):
    wire = json.loads((FIXTURES / 'circuit_report_v4_wire.json').read_text())
    outcome = wire['spec_reconciliation']['outcomes'][0]
    outcome['measurement']['aggregate'][field] = bad
    path = tmp_path / 'contradictory-aggregate.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError) as error:
        CircuitReport.load_json(path)
    message = str(error.value)
    assert str(path) in message
    assert 'outcome' in message and outcome['spec']['name'] in message
    assert 'aggregate' in message and field in message


def test_ordinary_measurement_roundtrip_still_has_no_aggregate(tmp_path):
    report = CircuitReport.load_json(FIXTURES / 'circuit_report_v4_wire.json')
    measurement = report.spec_reconciliation.outcomes[0].measurement
    measurement.aggregate = None
    measurement.sim_run_id = 'ordinary-run'
    path = tmp_path / 'ordinary-measurement.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path).spec_reconciliation.outcomes[0].measurement
    assert restored.aggregate is None and restored.sim_run_id == 'ordinary-run'
    assert restored.value == measurement.value


@pytest.mark.parametrize('slot', ['latest_attempt', 'last_successful_attempt'])
@pytest.mark.parametrize('field,bad', [('metric', 'unrelated_metric'),
                                       ('campaign_id', 'another_campaign')])
def test_attempt_aggregate_load_refuses_contradictory_attribution(tmp_path, slot, field, bad):
    from silk.specs import SpecAttempt
    from silk.specs.aggregate import encode_aggregate
    wire = json.loads((FIXTURES / 'circuit_report_v4_wire.json').read_text())
    row = wire['spec_reconciliation']['outcomes'][0]
    m = Measurement('spice', 2., passes=True, sim_run_id='campaign-1', aggregate=aggregate())
    attempt = SpecAttempt(source='spice', measurement=m).as_dict()
    attempt['measurement']['aggregate'] = encode_aggregate(aggregate())
    attempt['measurement']['aggregate'][field] = bad
    row[slot] = attempt
    path = tmp_path / 'bad-attempt.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError, match='bad-attempt.json.*sigma'):
        CircuitReport.load_json(path)


@pytest.mark.parametrize('slot', ['latest_attempt', 'last_successful_attempt'])
def test_attempt_aggregate_save_refuses_wrong_metric_without_truncation(tmp_path, slot):
    from silk.specs import SpecAttempt
    report = CircuitReport.load_json(FIXTURES / 'circuit_report_v4_wire.json')
    m = Measurement('spice', 2., passes=True, sim_run_id='campaign-1',
                    aggregate=aggregate(metric='unrelated_metric'))
    setattr(report.spec_reconciliation.outcomes[0], slot,
            SpecAttempt(source='spice', measurement=m))
    path = tmp_path / 'existing.json'
    path.write_bytes(b'keep existing evidence')
    with pytest.raises(ValueError, match='sigma'):
        report.save_json(path)
    assert path.read_bytes() == b'keep existing evidence'


def test_all_failed_campaign_retains_unavailable_aggregate_in_report(tmp_path):
    from silk.circuit.monte_carlo import MonteCarloResults, MonteCarloSettings, SampleFailure
    module = _top_with_instances(tmp_path)
    spec = module.declare_spec('gain', 'gain_dB', '<=', 10)
    campaign = MonteCarloResults({}, settings=MonteCarloSettings(n=2, seed=2**100),
        seeds={('ss', 0): 7, ('ss', 1): 11},
        failed={('ss', 0): SampleFailure('failed', 7), ('ss', 1): SampleFailure('failed', 11)},
        metric_names=['gain_dB'], campaign_source='spice')
    with pytest.warns(Warning, match='every sample failed'):
        measurement = campaign.measurement('gain_dB')
    spec.record_measurement(measurement)
    report = generate_report(module, auto_discover_config=False)
    path = tmp_path / 'failed-campaign.json'
    report.save_json(path)
    for candidate in (report, CircuitReport.load_json(path)):
        row, = candidate.spec_reconciliation.outcomes
        assert row.status == 'FAIL' and row.measurement is None
        assert row.last_successful_attempt is None
        attempt = row.latest_attempt
        assert attempt.kind == 'unavailable' and attempt.reason
        assert attempt.measurement is None
        assert attempt.aggregate == measurement.aggregate
        assert attempt.aggregate.campaign_id == measurement.sim_run_id
        assert attempt.aggregate.samples_total == attempt.aggregate.samples_failed == 2
        assert attempt.aggregate.samples_used == 0
        assert attempt.aggregate.sample_seeds == ((0, 7), (1, 11))
        assert attempt.aggregate.base_seed == 2**100
        assert attempt.aggregate.selected_corner == 'ss'
        assert attempt.aggregate.variation == 'mismatch'
        assert attempt.proved_conditions is None
    wire = json.loads(path.read_text())['spec_reconciliation']['outcomes'][0]
    assert wire['measurement'] is None
    assert wire['latest_attempt']['measurement'] is None
    assert 'value' not in wire['latest_attempt']['aggregate']


@pytest.mark.parametrize('bad', [{'scheme': 'unknown'}, {'scheme': 'silk.measurement.aggregate/v1'}])
def test_malformed_unavailable_aggregate_report_names_file_and_attempt(tmp_path, bad):
    from silk.specs import SpecAttempt
    wire = json.loads((FIXTURES / 'circuit_report_v4_wire.json').read_text())
    attempt = SpecAttempt(source='spice', kind='unavailable', reason='no samples',
                          attempt_id='unavailable-campaign').as_dict()
    attempt['aggregate'] = bad
    wire['spec_reconciliation']['outcomes'][0]['latest_attempt'] = attempt
    path = tmp_path / 'bad-unavailable.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError, match='bad-unavailable.json.*unavailable-campaign.*aggregate'):
        CircuitReport.load_json(path)


@pytest.mark.parametrize('boundary', ['save', 'load'])
def test_unavailable_aggregate_metric_must_match_outcome(tmp_path, boundary):
    from silk.specs import SpecAttempt
    report = CircuitReport.load_json(FIXTURES / 'circuit_report_v4_wire.json')
    row, = report.spec_reconciliation.outcomes
    row.measurement = None
    row.status = 'FAIL'
    row.latest_attempt = SpecAttempt(source='spice', kind='unavailable', reason='no samples',
        attempt_id='wrong-metric-attempt', aggregate=aggregate(metric='unrelated_metric'))
    path = tmp_path / 'wrong-unavailable.json'
    path.write_bytes(b'keep evidence')
    if boundary == 'save':
        with pytest.raises(ValueError, match='sigma.*aggregate'):
            report.save_json(path)
        assert path.read_bytes() == b'keep evidence'
    else:
        wire = json.loads((FIXTURES / 'circuit_report_v4_wire.json').read_text())
        outcome = wire['spec_reconciliation']['outcomes'][0]
        outcome['latest_attempt'] = row.latest_attempt.as_dict()
        outcome['measurement'] = None
        outcome['status'] = 'FAIL'
        path.write_text(json.dumps(wire))
        with pytest.raises(ValueError, match='wrong-unavailable.json.*sigma.*aggregate'):
            CircuitReport.load_json(path)


@pytest.mark.parametrize('mc', [False, True])
def test_grid_report_detached_roundtrip_and_latest_reconciliation(tmp_path, capsys, mc):
    import warnings
    from tests.test_results_json import grid_collection
    module = _top_with_instances(tmp_path)
    spec = module.declare_spec('gain', 'gain', '>', 1.0)
    spec.record_measurement(Measurement(source='user', value=3.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        source = grid_collection(mc)
        report = generate_report(module, test_results={'ac': source}, auto_discover_config=False)
        html = report._to_html()
        md = report._to_markdown()
        report.print_summary()
        terminal = capsys.readouterr().out
        for output in (html, md, terminal):
            if mc:
                assert 'PVT Grid: ac — FAIL' in output
                assert 'every sample must pass' in output
                assert 'All points (informational)' in output
                assert 'observed.' not in output and 'run.corner' not in output
            else:
                assert 'PVT Grid: ac — FAIL' in output
                assert 'Summary: 3 runs (ss, tt, ff × -40, 25, 125 °C × 1.6, 1.8, 2 V): 1 PASS, 0 FAIL, 2 failed and produced no report' in output
                assert 'observed.' not in output and 'run.corner' not in output
                assert 'check.gain floor' not in output
                assert output.index('PVT Grid: ac') < output.index('Specs measured in a single run')
        assert next(iter(spec.measurements.values())).value == 3.0
        path = tmp_path / 'grid.json'
        report.save_json(path)
        restored = CircuitReport.load_json(path)
        assert restored._to_html() == html
        assert restored._to_markdown() == md
        restored.print_summary()
        assert capsys.readouterr().out == terminal


def test_grid_report_uses_stable_check_keys_and_escapes_evidence(tmp_path):
    from silk.circuit.spec_report import CornerResults, SpecReport
    source = SpecReport('ac', 'amp', {'gain': 2.0}).check('same label', 'gain', '>', 1.0)
    source.results[0].spec_key = 'left.gain'
    source.results[0].owner_path = 'left'
    declarations = [dict(key=key, name='same label', owner_path=owner,
                         metric='gain', op='>', target=1.0)
                    for key, owner in [('left.gain', 'left'), ('right.gain', 'right')]]
    point = ('tt', 27.0, 1.8)
    grid = CornerResults({point: source}, failed={('ss', 27.0, 1.8): '<script>x</script>|bad'},
                         check_declarations=declarations)
    report = generate_report(_top_with_instances(tmp_path), test_results=grid,
                             auto_discover_config=False)
    html = report._to_html()
    assert html.count('<td>same label</td>') == 2
    assert 'check.left.gain' not in html and 'check.right.gain' not in html
    assert '<script>x</script>' not in html
    assert '&lt;script&gt;x&lt;/script&gt;' in html
    assert report.test_results.passed is False
    assert report.test_results.table()[0]['check.right.gain'] == 'UNAVAILABLE'


def test_empty_failure_message_grid_renders_after_json_reload(tmp_path):
    from silk.circuit.spec_report import CornerResults
    grid = CornerResults({}, failed={('tt', 27.0, 1.8): ''})
    report = generate_report(_top_with_instances(tmp_path), test_results=grid,
                             auto_discover_config=False)
    path = tmp_path / 'empty-error.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    assert 'PVT Grid:' in restored._to_html()
    assert 'PVT Grid:' in restored._to_markdown()
    assert restored.test_results.failed == grid.failed
    restored.print_summary()


def test_monte_carlo_grid_report_keeps_per_point_yield_verdict(tmp_path):
    from silk.circuit.monte_carlo import MonteCarloResults, MonteCarloSettings, SampleFailure
    from silk.circuit.spec_report import SpecReport
    good = ('tt', 27.0, 1.8)
    bad = ('ss', 125.0, 1.6)
    sample = SpecReport('ac', 'amp', {'gain': 2.0}).check('gain floor', 'gain', '>', 1.0)
    keys = [(p, i) for p in (good, bad) for i in range(2)]
    mc = MonteCarloResults({k: sample for k in keys[:3]},
        failed={keys[3]: SampleFailure('no data', 8)}, seeds={k: i+5 for i, k in enumerate(keys)},
        settings=MonteCarloSettings(n=2, min_yield=.7))
    from silk.metrics.diagnostics import MetricWarning
    with pytest.warns(MetricWarning):
        report = generate_report(_top_with_instances(tmp_path), test_results=mc,
                                 auto_discover_config=False)
        md = report._to_markdown()
    assert 'PVT Grid: ac — FAIL' in md
    assert 'required yield 0.7 per point' in md
    assert '| ss | 125 | 1.6 | gain floor | 2 | 0 | 1 | 0.500 |' in md
    assert '| All points (informational) |  |  | gain floor | 4 | 0 | 1 | 0.750 |' in md
    assert '| FAIL | gain floor | &gt; | 1 | 2 | tt, 27 °C, 1.8 V | 1 | 2 | 2 |' in md


def presentation_grid(*, failed=False):
    from silk.circuit.spec_report import CornerResults, SpecReport
    points = [('ff', -40.0, 1.62), ('tt', -40.0, 1.62)]
    reports = {point: SpecReport('ac', 'amp', {'gain_dB': value, 'offset': offset})
               .check('DC gain > 20 dB', 'gain_dB', '>', 20.0)
               for point, value, offset in zip(points, [30.349, 37.533], [1.23456789, 2.0])}
    failures = {points[-1]: 'SimulationAborted: no data\nsecond line'} if failed else {}
    if failed:
        del reports[points[-1]]
    return CornerResults(reports, failed=failures, skip_unavailable=True,
        run_inventory=[dict(key=p, corner=p[0], temperature=p[1], supply=p[2], supply_source='vdd')
                       for p in points], metric_names=['gain_dB', 'offset'])


def test_deterministic_grid_presentation_uses_spec_extrema_and_compact_runs(tmp_path):
    import warnings
    grid = presentation_grid()
    report = generate_report(_top_with_instances(tmp_path), test_results={'ac': grid},
                             auto_discover_config=False)
    md = report._render_grid('markdown')
    assert '## PVT Grid: ac — PASS' in md
    assert 'Summary: 2 runs (ff, tt × -40 °C × 1.62 V): 2 PASS, 0 FAIL, 0 failed and produced no report' in md
    assert '| Status | Spec | Op | Target | Worst | Worst at | Margin | Min | Max |' in md
    assert '| PASS | DC gain &gt; 20 dB | &gt; | 20 | 30.349 | ff, -40 °C, 1.62 V | 10.349 | 30.349 | 37.533 |' in md
    assert '| — | offset | — | — | — | — | — | 1.23457 | 2 |' in md
    assert '| Corner | Temp (°C) | Supply (V) | Status | gain_dB | offset |' in md
    assert '| ff | -40 | 1.62 | PASS | 30.349 | 1.23457 |' in md
    for forbidden in ('summary()', 'CornerResults —', 'observed.', 'run.', 'metric.', 'check.'):
        assert forbidden not in md
    html = report._render_grid('html')
    assert '<table class="spec-reconciliation">' in html
    assert '<tr class="spec-pass"><td class="status">PASS</td>' in html
    assert '<details><summary>All 2 runs</summary>' in html
    assert '<details open' not in html
    for mode in ('html', 'markdown', 'text'):
        assert '\u2212' not in report._render_grid(mode)
    assert grid.table()[0]['metric.offset'] == 1.23456789
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        failed = generate_report(_top_with_instances(tmp_path / 'failed'),
            test_results={'ac': presentation_grid(failed=True)}, auto_discover_config=False)
        failed_md = failed._render_grid('markdown')
    assert '| tt | -40 | 1.62 | NO REPORT |  |  |' in failed_md
    assert 'tt, -40 °C, 1.62 V: SimulationAborted: no data' in failed_md
    assert 'second line' not in failed_md


def test_grid_reconciliation_heading_stays_plain_with_recorded_conditions(tmp_path):
    from silk.specs import SpecAttempt
    module = _top_with_instances(tmp_path)
    spec = module.declare_spec('gain', 'gain_dB', '>', 20.)
    measurement = Measurement(source='spice', value=34.5, passes=True, timestamp=100.)
    spec.record_attempt(SpecAttempt(source='spice', timestamp=100., measurement=measurement,
        proved_conditions=dict(corner='ff', corner_kind='fixed', temperature_c=125.,
            temperature_kind='fixed', supply_voltage_v=1.98, supply_kind='fixed')))
    report = generate_report(module, test_results={'ac': presentation_grid()}, auto_discover_config=False)
    assert report._reconciliation_title() == 'Spec Reconciliation'
    path = tmp_path / 'conditions.json'
    report.save_json(path)
    assert CircuitReport.load_json(path)._reconciliation_title() == report._reconciliation_title()


def test_circuit_provenance_display_groups_shared_versions_and_libraries(tmp_path):
    from dataclasses import replace
    from silk.simulation.provenance import RunProvenance
    report = generate_report(_top_with_instances(tmp_path), auto_discover_config=False)
    original = RunProvenance(silk_version='0.2.0', simulator_version='ngspice-42',
        pdk_name='test-pdk', pdk_libraries=[dict(file='models.lib', section='tt', file_sha256='a'*64)],
        deck_sha256='b'*64)
    report.provenance = [original, replace(original, pdk_libraries=[dict(file='models.lib',
        section='ff', file_sha256='a'*64)], deck_sha256='c'*64),
        replace(original, simulator_version='ngspice-43')]
    before = [p.to_dict() for p in report.provenance]
    html = report._to_html().split('<section class="provenance">', 1)[-1]
    md = report._to_markdown()
    for output in (html, md):
        assert 'Run details: silk 0.2.0, ngspice-42, test-pdk (models.lib aaaaaaaaaa)' in output
        assert output.count('ngspice-42') == 1
        assert output.count('ngspice-43') == 1
        assert 'models.lib' in output
        assert '[tt]' not in output and '[ff]' not in output
        assert 'deck_sha256' not in output and 'Field' not in output
    path = tmp_path / 'provenance.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    assert [p.to_dict() for p in restored.provenance] == before


def test_grid_preserves_structured_metric_cells_without_numeric_reduction(tmp_path):
    from silk.circuit.spec_report import CornerResults
    source = presentation_grid()
    for record in source.reports.values():
        record.metrics['structured'] = {'peak': 1.25, 'bins': [1, 2]}
    grid = CornerResults(source.reports, metric_names=['gain_dB', 'offset', 'structured'])
    report = generate_report(_top_with_instances(tmp_path), test_results={'ac': grid},
                             auto_discover_config=False)
    for mode in ('html', 'markdown', 'terminal'):
        rendered = report._render_grid(mode)
        assert 'structured' in rendered and 'peak' in rendered and 'bins' in rendered
    assert '| — | structured | — | — | — | — | — | — | — |' in report._render_grid('markdown')


def test_grid_reconciliation_does_not_display_recorded_corner_markup(tmp_path):
    from silk.specs import SpecAttempt
    module = _top_with_instances(tmp_path)
    spec = module.declare_spec('gain', 'gain_dB', '>', 20.)
    spec.record_attempt(SpecAttempt(source='spice', timestamp=100.,
        measurement=Measurement(source='spice', value=34.5, passes=True, timestamp=100.),
        proved_conditions=dict(corner='<script>x</script>|y\nz', corner_kind='fixed',
            temperature_c=125., temperature_kind='fixed', supply_voltage_v=1.98, supply_kind='fixed')))
    report = generate_report(module, test_results={'ac': presentation_grid()}, auto_discover_config=False)
    for rendered in (report._render_spec_reconciliation_html(), report._render_spec_reconciliation_md()):
        assert '<script>' not in rendered
        assert '&lt;script&gt;x&lt;/script&gt;' not in rendered
        assert 'Specs measured in a single run' in rendered
    from silk.circuit.spec_report import CornerResults, SpecReport
    hostile = CornerResults({('<script>corner</script>', 27., 1.8):
        SpecReport('ac', 'amp', {'gain': 2.0})})
    report = generate_report(module, test_results={'<script>test</script>': hostile},
                             auto_discover_config=False)
    assert '<script>' not in report._render_grid('markdown')


def test_multiple_monte_carlo_grids_keep_declared_test_order(tmp_path):
    from tests.test_results_json import grid_collection
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        report = generate_report(_top_with_instances(tmp_path),
            test_results={'ac': grid_collection(True), 'other': grid_collection(True)},
            auto_discover_config=False)
        md = report._render_grid('markdown')
        assert md.count('## PVT Grid:') == 2
        assert md.index('## PVT Grid: ac') < md.index('## PVT Grid: other')


@pytest.mark.parametrize('coverage', ['some', 'all', 'no-grid'])
def test_reconciliation_rendering_omits_only_specs_covered_by_grid_identity(tmp_path, coverage):
    from dataclasses import replace
    from silk.circuit.spec_report import CornerResults, SpecReport
    module = _top_with_instances(tmp_path)
    first = module.declare_spec('same label', 'gain', '>', 1., key='stable-gain')
    second = module.declare_spec('same label', 'offset', '<', 3., key='stable-offset')
    first.record_measurement(Measurement(source='user', value=2.))
    record = SpecReport('ac', 'amp', {'gain': 2., 'offset': 1.})
    record.check('different grid label', 'gain', '>', 1., spec_key=first.key, owner_path='$')
    if coverage == 'all':
        record.check('offset grid label', 'offset', '<', 3., spec_key=second.key, owner_path='$')
    key = 'tt' if coverage == 'no-grid' else ('tt', 27., 1.8)
    report = generate_report(module, test_results={'ac': CornerResults({key: record})},
                             auto_discover_config=False)
    if coverage == 'some':
        # A second placement of the same stable spec key is not covered.
        report.spec_reconciliation.outcomes.append(replace(
            report.spec_reconciliation.outcomes[0], owner_path='$/other'))
    before = report.spec_reconciliation.as_dict()
    note = 'Specs the PVT grid measures are in its table above.'
    for rendered in (report._render_spec_reconciliation_html(),
                     report._render_spec_reconciliation_md(), report._render_metrics_terminal()):
        if coverage == 'all':
            assert rendered == ''
        elif coverage == 'some':
            assert note in rendered
            measured, unmeasured = rendered.split('Specs not measured', 1)
            measured = measured.split('Specs measured in a single run', 1)[1]
            assert 'same label' in unmeasured  # uncovered root spec
            assert '(instance other)' not in unmeasured
            assert 'same label (instance other)' in measured  # same key, different owner
            assert '1 PASS, 0 FAIL' in rendered
            assert 'Specs not measured' in rendered
            assert rendered.index('Specs measured in a single run') < rendered.index('Specs not measured')
            assert 'latest single run' not in rendered
            assert rendered.index(note) < rendered.index('1 PASS')
        else:
            assert note not in rendered
            assert 'same label' in rendered
            assert '1 PASS, 0 FAIL, 1 UNMEASURED' in rendered
            assert 'Results from the last run.' in rendered
    assert report.spec_reconciliation.as_dict() == before
    path = tmp_path / 'coverage.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    assert restored.spec_reconciliation.as_dict() == before
    assert restored._render_spec_reconciliation_html() == report._render_spec_reconciliation_html()
    assert restored._render_spec_reconciliation_md() == report._render_spec_reconciliation_md()
    assert restored._render_metrics_terminal() == report._render_metrics_terminal()


@pytest.mark.parametrize('measured', [False, True])
def test_grid_leftover_spec_sections_and_detached_rendering(tmp_path, measured):
    module = _top_with_instances(tmp_path)
    module.declare_spec('never measured', 'offset', '<', 3., key='unmeasured')
    if measured:
        single = module.declare_spec('single run', 'settling', '<', 5., key='single')
        single.record_measurement(Measurement(source='user', value=2.))
    report = generate_report(module, test_results={'ac': presentation_grid(failed=True)},
                             auto_discover_config=False)
    before = report.spec_reconciliation.as_dict()
    path = tmp_path / 'leftovers.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    for render in ('_render_spec_reconciliation_html', '_render_spec_reconciliation_md',
                   '_render_metrics_terminal'):
        output = getattr(report, render)()
        assert output == getattr(restored, render)()
        assert 'Spec Reconciliation' not in output
        assert 'Specs not measured' in output
        assert f'These specs are declared on {report.module_name}, but no test checked them.' in output
        unmeasured = output.split('Specs not measured', 1)[1]
        assert 'UNMEASURED' not in unmeasured
        assert 'Summary:' not in unmeasured
        assert 'Status' not in unmeasured and 'Measured' not in unmeasured
        assert 'never measured' in unmeasured
        assert ('Specs measured in a single run' in output) == measured
        if measured:
            assert output.index('Specs measured in a single run') < output.index('Specs not measured')
            assert '1 PASS, 0 FAIL' in output
        else:
            assert '0 PASS' not in output
    assert report.spec_reconciliation.as_dict() == before
    for mode in ('html', 'markdown', 'terminal'):
        assert 'NO REPORT' in report._render_grid(mode)
        assert restored._render_grid(mode) == report._render_grid(mode)
    assert report.test_results['ac'].table()[-1]['run.status'] == 'failed'
    assert restored.test_results['ac'].table()[-1]['run.status'] == 'failed'


def test_non_grid_reconciliation_keeps_frozen_rendered_bytes(tmp_path):
    import hashlib
    module = _top_with_instances(tmp_path)
    module.declare_spec('unmeasured', 'offset', '<', 3., key='unmeasured')
    single = module.declare_spec('single run', 'settling', '<', 5., key='single')
    single.record_measurement(Measurement(source='user', value=2., timestamp=100.))
    report = generate_report(module, auto_discover_config=False)
    # Frozen reconciliation output with root placement suffixes omitted.
    expected = {
        '_render_spec_reconciliation_html': '1b1a185a862339e15eb3246c570f57334f0e2dd4ca082fbd43f7c000df63b5e7',
        '_render_spec_reconciliation_md': '644fdf8743c5f842cc7b026d1649d6f3a956efa317abfb3a4fa0869293b5fddf',
        '_render_metrics_terminal': '143750ed2b747a90fda2226e4311685a0d79199ef9451506eb02a07fae496195',
    }
    for method, digest in expected.items():
        assert hashlib.sha256(getattr(report, method)().encode()).hexdigest() == digest

    from dataclasses import replace
    report.spec_reconciliation.outcomes.extend([
        replace(report.spec_reconciliation.outcomes[1], owner_path='$/u0'),
        replace(report.spec_reconciliation.outcomes[0], owner_path='$/u0/x1'),
    ])
    # Frozen reconciliation output with root, placed and nested labels.
    expected = {
        '_render_spec_reconciliation_html': '300ac5177f3ac4adddb0c16eac5a55a1d6d08700b630e106db5c80c2773cd83b',
        '_render_spec_reconciliation_md': '725c5945f38362a5f186ef489202e7bb8a8259019f432c4e80894ba3e9152859',
        '_render_metrics_terminal': 'e12556597dcf6f6a9a6b98c5d3eba2510bddc6fa5f56871e1a674c08d74f90d8',
    }
    for method, digest in expected.items():
        assert hashlib.sha256(getattr(report, method)().encode()).hexdigest() == digest


def test_grouped_variable_grid_keeps_named_columns_and_worst_label(tmp_path):
    from silk.circuit.spec_report import CornerResults, SpecReport
    points = [('tt', -40.0, None, (('trim', value),)) for value in (1, 3)]
    grid = CornerResults({point: SpecReport('ac', 'amp', {'gain_dB': value}).check('gain floor', 'gain_dB', '>', 20)
                          for point, value in zip(points, (30.0, 40.0))},
        run_inventory=[dict(key=point, corner='tt', temperature=-40.0, supply=None,
                            supply_source=None, variables=list(point[3]), groups=[['trim']])
                       for point in points])
    report = generate_report(_top_with_instances(tmp_path), test_results={'ac': grid},
                             auto_discover_config=False)
    md = report._render_grid('markdown')
    assert 'run.trim' in md
    assert 'trim=1' in md
    assert '2 planned points' in md
    path = tmp_path / 'variable-report.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    assert restored._render_grid('markdown') == md
