# tests/test_monte_carlo_report.py — CircuitReport.add_monte_carlo (TASK-431)
import hashlib
import json

import numpy as np
import pytest

from silk.circuit.interface_check import AnnotationCoverage
from silk.circuit.monte_carlo import (MonteCarloResults, MonteCarloSettings,
                                      SampleFailure, derive_sample_seed,
                                      wilson_interval)
from silk.circuit.monte_carlo_draws import DrawTable
from silk.circuit.report import CircuitReport, TopologySummary
from silk.circuit.spec_report import SpecReport

SPEC = 'gain > 1'
IMG = '<img src="data:image/png;base64,'


def _report(value):
    return SpecReport('ac', 'amp', {'gain': value}).check(SPEC, 'gain', '>', 1.0)


def _mc(values, failed=(), corners=('tt',)):
    """``values`` per corner; sample indices in ``failed`` fail in every corner."""
    reports, fails, seeds = {}, {}, {}
    n = len(values) + len(failed)
    for c in corners:
        it = iter(values)
        for i in range(n):
            key = (c, i)
            seeds[key] = derive_sample_seed(0, i)
            if i in failed:
                fails[key] = SampleFailure('SimulationAborted: no data',
                                           seeds[key])
            else:
                reports[key] = _report(next(it))
    return MonteCarloResults(reports, failed=fails, seeds=seeds,
                             settings=MonteCarloSettings(n=n))


def _blank_report():
    return CircuitReport(
        module_name='amp', generated_at='2026-10-03T00:00:00',
        topology=TopologySummary('amp', [], 0),
        annotation=AnnotationCoverage(0, 0, []),
        violations=[], simulation=None)


def _draws(mc, columns, seed=0):
    rng = np.random.default_rng(seed)
    gain = {k: r.metrics['gain'] for k, r in mc.reports.items()}
    keys = sorted(gain)
    # first column drives the metric; second is noise
    x = np.column_stack([[gain[k] for k in keys],
                         rng.normal(size=len(keys))])
    return DrawTable(columns, {k: x[i] for i, k in enumerate(keys)}, raw=True)


def _html(mc, **kw):
    rep = _blank_report()
    rep.add_monte_carlo(mc, **kw)
    return rep, rep._to_html()


def test_yield_table_intervals_counts_histogram_and_failed_seed():
    mc = _mc([2.0, 0.5, 3.0, 4.0, 2.5, 1.5, 0.8, 5.0], failed={3})
    _, h = _html(mc)
    lo, hi = wilson_interval(6, 9)
    assert 'gain &gt; 1' in h
    assert f'[{lo:.3f}, {hi:.3f}]' in h
    assert '<td>9</td><td>1</td><td>8</td>' in h        # run, failed, used
    assert h.count(IMG) == 1                              # one metric, no draws
    assert str(derive_sample_seed(0, 3)) in h
    assert 'SimulationAborted: no data' in h
    assert 'Sensitivity' not in h


def test_yield_rows_per_corner():
    mc = _mc([2.0, 0.5, 3.0, 4.0], corners=('ss', 'tt'))
    d = _blank_report()
    d.add_monte_carlo(mc)
    ys = d.monte_carlo[0].data['yields']
    assert [(y['corner'], y['passed'], y['run']) for y in ys] == \
        [('ss', 3, 4), ('tt', 3, 4)]


def test_sensitivity_table_in_share_order_and_scatter_plots():
    mc = _mc(list(np.linspace(0.5, 4, 30)))
    # list the weak (noise) column first so share order differs from input
    cols = (('m2', 'noise'), ('m1', 'vth0'))
    draws = _draws(mc, (('m1', 'vth0'), ('m2', 'noise')))
    draws = DrawTable(cols, {k: v[::-1] for k, v in draws.values.items()},
                      raw=True)
    mc.draws = draws
    rep, h = _html(mc, top=1)
    assert 'Sensitivity' in h
    assert h.index('m1:vth0') < h.index('m2:noise')
    assert h.count(IMG) == 2                              # histogram + 1 scatter
    rows = rep.monte_carlo[0].data['sensitivity'][0]['rows']
    assert [r['name'] for r in rows] == ['m1:vth0', 'm2:noise']


def test_json_carries_numbers_without_images(tmp_path):
    mc = _mc([2.0, 0.5, 3.0], failed={1})
    rep, _ = _html(mc)
    p = tmp_path / 'r.json'
    rep.save_json(str(p))
    text = p.read_text()
    assert 'base64' not in text and 'png_b64' not in text
    d = json.loads(text)['monte_carlo'][0]
    y = d['yields'][0]
    lo, hi = wilson_interval(y['passed'], 4)
    assert (y['run'], y['failed'], y['used']) == (4, 1, 3)
    assert (y['lo'], y['hi']) == pytest.approx((lo, hi))
    assert d['failed_samples'][0]['seed'] == derive_sample_seed(0, 1)
    again = CircuitReport.load_json(str(p))
    assert again.monte_carlo[0].data['yields'] == d['yields']


def test_report_without_monte_carlo_is_unchanged(tmp_path):
    rep = _blank_report()
    p = tmp_path / 'r.json'
    rep.save_json(str(p))
    assert 'monte_carlo' not in json.loads(p.read_text())
    assert 'Monte Carlo' not in rep._to_html()


def test_metrics_selection_and_bad_arguments():
    mc = _mc([2.0, 3.0])
    rep = _blank_report()
    with pytest.raises(ValueError, match='top'):
        rep.add_monte_carlo(mc, top=0)
    with pytest.raises(Exception, match='nope'):
        rep.add_monte_carlo(mc, metrics=['nope'])


@pytest.mark.parametrize('value', [1+2j, np.complex128(1+2j)])
def test_complex_mc_report_refuses_without_losing_value(value):
    from silk.exceptions import MonteCarloError
    mc = _mc([value])
    with pytest.raises(MonteCarloError, match=r'gain.*complex.*\.real.*abs'):
        _blank_report().add_monte_carlo(mc)
    with pytest.raises(ValueError, match=r'gain.*complex.*\.real.*abs'):
        mc.stat('gain', 'mean')
    assert mc.reports[('tt', 0)].metrics['gain'] is value


def test_current_json_preserves_complex_legacy_payload(tmp_path):
    from silk.circuit.monte_carlo_report import MonteCarloSection
    report = _blank_report()
    report.monte_carlo = [MonteCarloSection({'legacy_value': 1+2j})]
    path = tmp_path / 'report.json'
    report.save_json(path)
    payload = json.loads(path.read_text())
    assert payload['monte_carlo'][0]['legacy_value'] == {
        'status': 'complex', 're': 1.0, 'im': 2.0}
    restored = CircuitReport.load_json(path)
    assert restored.monte_carlo[0].data['legacy_value'] == 1+2j


def test_real_json_v4_bytes_preserved_except_schema_version(tmp_path):
    # Frozen main 0c210c6f v4 bytes from this identical report.
    report = _blank_report()
    path = tmp_path / 'report.json'
    report.save_json(path)
    assert json.loads(path.read_bytes())['schema_version'] == 5
    normalized = path.read_bytes().replace(b'"schema_version": 5', b'"schema_version": 4', 1)
    assert hashlib.sha256(normalized).hexdigest() == 'ebf68c775f7694c37089ca9a2e4a171cd14920f7eb9ccd381dfd120c622c4cc0'


def test_complex_mc_table_and_csv_preserve_components(tmp_path):
    import csv
    value = 1+2j
    mc = _mc([value])
    assert mc.table()[0]['metric.gain'] is value
    path = tmp_path / 'complex.csv'
    mc.to_csv(path)
    with path.open() as stream:
        row, = csv.DictReader(stream)
    assert row['metric.gain'] == '(1+2j)'


@pytest.mark.parametrize('corner_only', [False, True])
def test_failed_corner_has_zero_yield_row_in_html_and_json(corner_only, tmp_path):
    mc = MonteCarloResults(
        {('tt', 0): _report(2.0)}, settings=MonteCarloSettings(n=1),
        seeds={('tt', 0): 1, ('ss', 0): 1},
        failed={('ss', 0): SampleFailure('no data', 1)})
    if corner_only:
        mc = mc.corner('ss')
    rep, html = _html(mc)
    assert '<td>gain &gt; 1</td><td>ss</td><td>0.000</td>' in html
    path = tmp_path / 'report.json'
    rep.save_json(path)
    data = json.loads(path.read_text())['monte_carlo'][0]
    ss = next(row for row in data['yields'] if row['corner'] == 'ss')
    assert (ss['passed'], ss['run'], ss['failed'], ss['used'], ss['fraction']) == (0, 1, 1, 0, 0.0)
    assert data['failed_samples'] == [dict(corner='ss', sample=0, seed=1, error='no data')]
    assert dict(corner='ss', run=1, failed=1, used=0) in data['corners']


@pytest.mark.parametrize('metrics', [None, ['gain']])
def test_failed_corner_report_keeps_default_metrics_and_limits(metrics):
    from silk.circuit.monte_carlo_report import build_section
    mc = MonteCarloResults(
        {}, settings=MonteCarloSettings(n=1), seeds={('ss', 0): 1},
        failed={('ss', 0): SampleFailure('no data', 1)},
        spec_declarations=[dict(name=SPEC, metric='gain', op='>', target=1.0)])
    section = build_section(mc, metrics=metrics)
    assert section.data['metrics'] == [dict(metric='gain', used=0,
                                           limits=[dict(op='>', target=1.0)])]
    assert not section.images


def grid_fixture(*, min_yield=.9):
    """Three ordered points, twenty trials each, one missing report."""
    points = [('ss', -40., 1.6), ('tt', 25., 1.8), ('ff', 125., 2.)]
    keys = [(point, i) for point in points for i in range(20)]
    failed = (points[0], 19)
    reports = {}
    for point, i in keys:
        if (point, i) == failed:
            continue
        value = (.8 + .05*i if i < (1 if point == points[2] else 2)
                 else 1.5 + .03*i + .1*points.index(point))
        reports[point, i] = _report(value)
    return MonteCarloResults(reports, settings=MonteCarloSettings(n=20, min_yield=min_yield),
        seeds={key: 100+i for i, key in enumerate(keys)},
        failed={failed: SampleFailure('SimulationAborted: no data\nsecond line', 119)},
        run_inventory=[dict(key=k, corner=k[0][0], temperature=k[0][1], supply=k[0][2],
                            supply_source='vdd') for k in keys], metric_names=['gain'])


@pytest.mark.parametrize('minimum,requirement', [(.9, 'required yield 0.9 per point'),
                                               (None, 'every sample must pass'),
                                               (.8, 'required yield 0.8 per point')])
def test_grid_report_states_yield_requirement_and_tabular_point_evidence(tmp_path, minimum, requirement):
    import warnings
    mc = grid_fixture(min_yield=minimum)
    report = _blank_report()
    report.test_results = {'ac': mc}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        md = report._to_markdown()
        html = report._to_html()
        terminal = report._render_grid('terminal')
    for output in (md, html, terminal):
        assert 'PVT Grid: ac — ' + ('PASS' if minimum == .8 else 'FAIL') in output
        assert f'Summary: 60 samples at 3 operating points, 20 samples per point; {requirement}; 1 failed and produced no report.' in output
        assert 'All points (informational)' in output
        assert '0.850' in output and '[0.640, 0.948]' in output
        assert 'observed.' not in output and 'run.' not in output
        assert 'second line' not in output
        assert 'NO REPORT' in output
    assert '| Corner | Temp (°C) | Supply (V) | Spec | Samples | FAIL | No report | Yield | 95% interval | Status |' in md
    ss_status = 'PASS' if minimum == .8 else 'FAIL'
    assert f'| ss | -40 | 1.6 | gain &gt; 1 | 20 | 2 | 1 | 0.850 | [0.640, 0.948] | {ss_status} |' in md
    tt_status = 'FAIL' if minimum is None else 'PASS'
    assert f'| tt | 25 | 1.8 | gain &gt; 1 | 20 | 2 | 0 | 0.900 | [0.699, 0.972] | {tt_status} |' in md
    assert '| All points (informational) |  |  | gain &gt; 1 | 60 | 5 | 1 | 0.900 | [0.799, 0.953] | — |' in md
    assert '| ss | -40 | 1.6 | 19 | NO REPORT |  |' in md
    assert '<details><summary>All 60 samples</summary>' in html
    assert '<details open' not in html
    assert next(row for row in mc.table() if row['run.status'] == 'failed')['run.status'] == 'failed'
    assert '<th>FAIL</th><th>No report</th>' in html
    assert '<td>20</td><td>2</td><td>1</td><td>0.850</td>' in html
    assert '| All points (informational) |  |  | gain > 1 | 60 | 5 | 1 | 0.900 | [0.799, 0.953] | — |' in terminal
    assert '<table class="spec-reconciliation">' in html
    assert '```text' not in md and '<pre>MonteCarloResults' not in html
    assert md.index('| ss |') < md.index('| tt |') < md.index('| ff |')
    path = tmp_path / 'grid.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert restored._to_html() == html
        assert restored._to_markdown() == md
        assert restored._render_grid('terminal') == terminal


def test_grid_spec_rows_keep_each_checks_direction_and_yield_status():
    from silk.circuit.monte_carlo_report import build_grid_section
    point = ('tt', 27., 1.8)
    records = {}
    for i, gain in enumerate((2., 4.)):
        records[point, i] = SpecReport('ac', 'amp', {'gain': gain, 'offset': i+.25})
        records[point, i].check('floor', 'gain', '>', 1., spec_key='low', owner_path='$')
        records[point, i].check('ceiling', 'gain', '<', 3., spec_key='high', owner_path='$')
    mc = MonteCarloResults(records, seeds={k: i for i, k in enumerate(records)},
                          settings=MonteCarloSettings(n=2, min_yield=.9))
    d = build_grid_section(mc).data
    assert d['spec_rows'] == [
        ['PASS', 'floor', '>', '1', '2', 'tt, 27 °C, 1.8 V', '1', '2', '4'],
        ['FAIL', 'ceiling', '<', '3', '4', 'tt, 27 °C, 1.8 V', '1', '2', '4'],
        ['—', 'offset', '—', '—', '—', '—', '—', '0.25', '1.25']]
    assert [row[3] for row in d['yield_rows']] == ['floor', 'ceiling', 'floor', 'ceiling']
    assert [row[4:8] for row in d['yield_rows']] == [
        [2, 0, 0, '1.00'], [2, 1, 0, '0.500'],
        [2, 0, 0, '1.00'], [2, 1, 0, '0.500']]
    assert [row[-1] for row in d['yield_rows'][-2:]] == ['—', '—']
    assert [len(r.results) for r in records.values()] == [2, 2]


def test_grid_yield_counts_include_unavailable_checks_only_when_run_produced_report():
    from silk.circuit.monte_carlo_report import build_grid_section
    point = ('tt', 27., 1.8)
    mc = MonteCarloResults({(point, 0): _report(2.), (point, 1): _report(float('nan'))},
        seeds={(point, i): i for i in range(3)},
        failed={(point, 2): SampleFailure('no data', 2)},
        settings=MonteCarloSettings(n=3, min_yield=.9))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows = build_grid_section(mc).data['yield_rows']
    # The unavailable extraction ran but did not meet the spec; the third run has no report.
    assert rows[0][4:8] == [3, 1, 1, '0.333']
    assert rows[1][4:8] == [3, 1, 1, '0.333']
    assert rows[1][-1] == '—'


def test_grid_yield_precision_keeps_small_nonzero_probabilities():
    from silk.circuit.monte_carlo_report import build_grid_section
    point = ('tt', 27., 1.8)
    good, bad = _report(2.), _report(.5)
    keys = [(point, i) for i in range(2000)]
    mc = MonteCarloResults({key: good if i == 0 else bad for i, key in enumerate(keys)},
        seeds={key: i for i, key in enumerate(keys)}, settings=MonteCarloSettings(n=2000))
    rows = build_grid_section(mc).data['yield_rows']
    assert rows[0][7] == '0.000500'
    assert rows[1][7] == '0.000500'
    lower = float(rows[0][8].lstrip('[').split(',')[0])
    assert lower > 0  # Formatting must not turn a small confidence bound into zero.


def test_variable_grid_labels_yields_and_samples_keep_coordinates():
    from silk.circuit.monte_carlo_report import build_grid_section
    points = [('tt', 27.0, None, (('trim', value),)) for value in (1, 3)]
    keys = [(point, 0) for point in points]
    mc = MonteCarloResults({keys[0]: _report(2.0)},
        settings=MonteCarloSettings(n=1), seeds={key: 7 for key in keys},
        failed={keys[1]: SampleFailure('no data', 7)},
        run_inventory=[dict(key=key, corner='tt', temperature=27.0, supply=None,
                            supply_source=None, variables=list(key[0][3]), groups=[['trim']])
                       for key in keys], metric_names=['gain'])
    data = build_grid_section(mc).data
    assert 'run.trim' in data['yield_columns']
    assert 'run.trim' in data['run_columns']
    index = data['yield_columns'].index('run.trim')
    assert [row[index] for row in data['yield_rows'][:2]] == ['1', '3']
    assert 'trim=3' in data['errors'][0]
    assert any('trim=1' in str(row) for row in data['spec_rows'])
