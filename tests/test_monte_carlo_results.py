# tests/test_monte_carlo_results.py
import math
import numpy as np
import pytest
from silk.circuit.monte_carlo import (MonteCarloResults, MonteCarloSettings,
                                      SampleFailure, SampleWorstCase,
                                      derive_sample_seed, wilson_interval)
from silk.circuit.spec_report import SpecReport
from silk.exceptions import MonteCarloError

SPEC = 'gain > 1'


def _report(value):
    return SpecReport('ac', 'amp', {'gain': value}).check(SPEC, 'gain', '>', 1.0)


def _mc(values, failed=(), n=None, **settings):
    keys = [('tt', i) for i in range(n or len(values) + len(failed))]
    reports, fails, seeds, it = {}, {}, {}, iter(values)
    for key in keys:
        seeds[key] = derive_sample_seed(0, key[1])
        if key[1] in failed:
            fails[key] = SampleFailure('SimulationAborted: no data', seeds[key])
        else:
            reports[key] = _report(next(it))
    return MonteCarloResults(reports, failed=fails, seeds=seeds,
                             settings=MonteCarloSettings(n=len(keys), **settings))


def test_wilson_interval_matches_the_spec_example():
    lo, hi = wilson_interval(197, 200)
    assert (round(lo, 3), round(hi, 3)) == (0.957, 0.995)


def test_sample_seeds_are_31_bit_positive_and_reproducible():
    seeds = [derive_sample_seed(0, k) for k in range(500)]
    assert all(1 <= s <= 2**31 - 1 for s in seeds)
    assert seeds == [derive_sample_seed(0, k) for k in range(500)]
    assert len(set(seeds)) == 500


@pytest.mark.parametrize('kwargs', [dict(n=0), dict(n=-1), dict(n=2.5),
                                    dict(n=10, min_yield=0),
                                    dict(n=10, min_yield=1.5),
                                    dict(n=10, variation='global')])
def test_bad_settings_raise(kwargs):
    with pytest.raises(MonteCarloError):
        MonteCarloSettings(**kwargs)


def test_failed_and_nan_samples_are_skipped_counted_and_fail_yield():
    mc = _mc([2.0, float('nan'), 4.0], failed={1})
    with pytest.warns(Warning, match='1 of 4 samples failed.*1 gave nan'):
        assert mc.stat('gain', 'mean') == pytest.approx(3.0)
    frac, (lo, hi) = mc.yield_fraction()[SPEC]
    assert frac == pytest.approx(2 / 4) and lo < frac < hi
    assert ('tt', 1) in mc.failed and ('tt', 1) not in mc.reports
    assert '1 failed' in mc.summary()


def test_n_equal_one_std_is_nan_with_the_warning():
    mc = _mc([2.0])
    with pytest.warns(Warning):
        assert math.isnan(mc.stat('gain', 'std'))


def test_worst_returns_corner_and_sample():
    mc = _mc([5.0, 1.5, 3.0])
    assert mc.worst('gain') == SampleWorstCase(1.5, 'tt', 1)


def test_percentile_table_and_csv(tmp_path):
    mc = _mc([1.5, 2.0, 3.0], failed={3})
    with pytest.warns(Warning):
        assert mc.percentile('gain', 50) == pytest.approx(2.0)
    rows = mc.table()
    assert len(rows) == 4 and rows[3]['run.error'].startswith('SimulationAborted')
    assert {'run.corner', 'run.sample', 'run.seed', 'metric.gain'} <= set(rows[0])
    mc.to_csv(tmp_path / 'mc.csv')
    assert (tmp_path / 'mc.csv').read_text().count('\n') == 5


def test_passed_with_and_without_min_yield():
    assert _mc([2.0, 0.5, 3.0]).passed is False
    assert _mc([2.0, 0.5, 3.0], min_yield=0.6).passed is True
    assert _mc([2.0, 0.5, 3.0], min_yield=0.7).passed is False


def test_corner_restricts_and_rerun_without_hook_raises():
    mc = _mc([2.0, 3.0])
    assert list(mc.corner('tt').reports) == [('tt', 0), ('tt', 1)]
    with pytest.raises(MonteCarloError):
        mc.rerun(('tt', 0))


def test_histogram_draws_the_spec_limit():
    import matplotlib
    matplotlib.use('Agg')
    ax = _mc([1.5, 2.0, 3.0]).plot_histogram('gain')
    assert any(abs(line.get_xdata()[0] - 1.0) < 1e-12 for line in ax.get_lines())


def test_wilson_interval_edges():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson_interval(10, 10)
    assert hi == pytest.approx(1.0) and 0 < lo < 1
    assert all(math.isnan(v) for v in wilson_interval(0, 0))
    with pytest.raises(MonteCarloError):
        wilson_interval(11, 10)


def test_settings_defaults_and_valid_values():
    s = MonteCarloSettings(n=1)
    assert (s.variation, s.seed, s.record_draws, s.correlate, s.min_yield) == \
        ('mismatch', 0, False, (), None)
    assert MonteCarloSettings(n=5, min_yield=1.0, correlate=['a']).correlate == ('a',)
    for bad in (dict(n=True), dict(seed=1.5), dict(record_draws=1)):
        with pytest.raises(MonteCarloError):
            MonteCarloSettings(**{'n': 3, **bad})


def test_summary_shows_run_failed_used_per_corner():
    text = _mc([2.0, 3.0], failed={2}).summary()
    assert 'tt' in text and '3 run' in text and '1 failed' in text and '2 used' in text


def test_percentile_and_worst_warn_when_skipping():
    mc = _mc([2.0, 3.0], failed={0})
    with pytest.warns(Warning, match='1 of 3 samples failed'):
        assert mc.percentile('gain', 100) == pytest.approx(3.0)
    with pytest.warns(Warning, match='1 of 3 samples failed'):
        assert mc.worst('gain') == SampleWorstCase(2.0, 'tt', 1)


def test_no_warning_when_nothing_skipped(recwarn):
    _mc([2.0, 3.0]).stat('gain', 'max')
    assert not [w for w in recwarn if 'samples failed' in str(w.message)]


def test_all_samples_failed_never_passes():
    mc = _mc([], failed={0, 1}, min_yield=0.5)
    assert mc.passed is False
    assert 'FAIL' in mc.summary()
    assert _mc([], failed={0, 1}).passed is False


def test_worst_without_spec_raises_before_any_skip_warning(recwarn):
    mc = _mc([2.0], failed={1})
    mc.reports[('tt', 0)].metrics['other'] = 1.0
    with pytest.raises(ValueError, match='no spec checks'):
        mc.worst('other')
    assert not [w for w in recwarn if 'samples failed' in str(w.message)]


def test_metric_handle_accepted_everywhere():
    from silk.circuit.metric_handle import MetricHandle
    h = MetricHandle(None, ('metric', 'gain'))
    mc = _mc([2.0, 3.0])
    assert mc.stat(h, 'mean') == pytest.approx(2.5)
    assert mc.percentile(h, 100) == pytest.approx(3.0)
    assert mc.worst(h) == mc.worst('gain')
    import matplotlib
    matplotlib.use('Agg')
    assert mc.plot_histogram(h) is not None


# TASK-515: selecting observations must not discard the campaign declaration.
def _failed_corner_campaign(**kwargs):
    seeds = {('tt', 0): derive_sample_seed(0, 0),
             ('ss', 0): derive_sample_seed(0, 0)}
    return MonteCarloResults(
        {('tt', 0): _report(2.0)}, settings=MonteCarloSettings(n=1),
        seeds=seeds,
        failed={('ss', 0): SampleFailure('SimulationAborted: no data', seeds[('ss', 0)])},
        **kwargs)


def _declared_empty(**kwargs):
    return MonteCarloResults(
        {}, settings=MonteCarloSettings(n=1), seeds={},
        metric_names=['gain', 'unspecced', 'eye.width'],
        spec_declarations=[dict(name=SPEC, metric='gain', op='>', target=1.0)],
        **kwargs)


def test_failed_corner_retains_campaign_yields():
    mc = _failed_corner_campaign()
    assert mc.yield_fraction()[SPEC][0] == 0.5
    assert not mc.passed
    ss = mc.corner('ss')
    fraction, (lo, hi) = ss.yield_fraction()[SPEC]
    assert fraction == lo == 0.0
    assert hi == pytest.approx(0.7934506856)
    assert not ss.passed
    assert len(ss.failed) == 1 and not ss.reports


@pytest.mark.parametrize('how', ['mean', 'rms', 'std', 'min', 'max', 'percentile', 'worst'])
def test_failed_corner_declared_statistics_warn_nan(how):
    from silk.metrics.diagnostics import MetricWarning
    ss = _failed_corner_campaign().corner('ss')
    with pytest.warns(MetricWarning, match='every sample failed') as caught:
        result = (ss.percentile('gain', 50) if how == 'percentile' else
                  ss.worst('gain') if how == 'worst' else ss.stat('gain', how))
    assert len(caught) == 1
    if how == 'worst':
        assert result.corner is result.sample is None
        result = result.value
    assert math.isnan(result)


@pytest.mark.parametrize('how', ['mean', 'rms', 'std', 'min', 'max', 'percentile'])
def test_zero_observations_warn_once_for_declared_metric(how):
    from silk.metrics.diagnostics import MetricWarning
    mc = _declared_empty()
    with pytest.warns(MetricWarning, match='0.*used|no samples|no observations') as caught:
        result = mc.percentile('gain', 50) if how == 'percentile' else mc.stat('gain', how)
    assert len(caught) == 1
    assert math.isnan(result)


def test_declared_inventory_survives_repeated_corner_selection():
    from silk.metrics.diagnostics import MetricWarning
    mc = _failed_corner_campaign(metric_names=['gain', 'unspecced', 'eye.width'])
    ss = mc.corner('ss').corner('ss')
    for name in ('unspecced', 'eye.width'):
        with pytest.warns(MetricWarning, match='every sample failed'):
            assert math.isnan(ss.stat(name, 'mean'))


def test_explicit_inventory_does_not_depend_on_successful_reports():
    declaration = dict(name='missing check', metric='unmeasured', op='<', target=4.0)
    mc = _failed_corner_campaign(spec_declarations=[
        dict(name=SPEC, metric='gain', op='>', target=1.0), declaration])
    declaration['name'] = 'changed'
    declaration['target'] = 99.0
    assert mc.yield_fraction()['missing check'][0] == 0.0
    assert set(mc.corner('ss').yield_fraction()) == {SPEC, 'missing check'}
    dead = _declared_empty(failed={('ss', 0): SampleFailure('no data', 1)})
    assert dead.yield_fraction()[SPEC][0] == 0.0
    fraction, interval = _declared_empty().yield_fraction()[SPEC]
    assert math.isnan(fraction) and all(math.isnan(v) for v in interval)


def test_failed_corner_table_csv_and_summary_keep_inventory(tmp_path):
    ss = _failed_corner_campaign().corner('ss')
    assert ss.table()[0]['metric.gain'] is None
    path = tmp_path / 'failed.csv'
    ss.to_csv(path)
    import csv
    with path.open() as stream:
        row, = list(csv.DictReader(stream))
    assert row['metric.gain'] == '' and row['run.error'] == 'SimulationAborted: no data'
    assert 'yield gain > 1: 0.000' in ss.summary()


def test_failed_corner_histogram_keeps_declared_limits():
    from matplotlib.figure import Figure
    from silk.metrics.diagnostics import MetricWarning
    ax = Figure().subplots()
    with pytest.warns(MetricWarning, match='every sample failed'):
        _failed_corner_campaign().corner('ss').plot_histogram('gain', ax=ax)
    assert [line.get_xdata()[0] for line in ax.lines] == [1.0]


def test_failed_corner_rejects_invalid_requests(recwarn):
    ss = _failed_corner_campaign(metric_names=['gain', 'other']).corner('ss')
    for call in (lambda: ss.stat('typo', 'mean'),
                 lambda: ss.stat('gain', 'typo'),
                 lambda: ss.percentile('gain', 101),
                 lambda: ss.worst('other')):
        with pytest.raises(ValueError):
            call()
    assert not recwarn


@pytest.mark.parametrize('field,value', [('metric', 'other'), ('op', '<'), ('target', 3.0)])
@pytest.mark.parametrize('source', ['explicit', 'legacy', 'report_vs_declaration'])
def test_conflicting_spec_definitions_raise_with_both_definitions(field, value, source):
    first = dict(name=SPEC, metric='gain', op='>', target=1.0)
    second = {**first, field: value}
    kwargs = {}
    reports = {}
    if source == 'explicit':
        kwargs['spec_declarations'] = [first, second]
    else:
        other = SpecReport('ac', 'amp', {'gain': 2.0, 'other': 2.0}).check(
            SPEC, second['metric'], second['op'], second['target'])
        reports = {('ss', 0): other}
        if source == 'legacy':
            reports[('tt', 0)] = _report(2.0)
        else:
            kwargs['spec_declarations'] = [first]
    with pytest.raises(ValueError) as caught:
        MonteCarloResults(reports, settings=MonteCarloSettings(n=1), seeds={}, **kwargs)
    message = str(caught.value)
    assert SPEC in message and 'gain' in message and '1.0' in message
    assert str(value) in message


def test_identical_spec_definitions_are_not_conflicts():
    declaration = dict(name=SPEC, metric='gain', op='>', target=1.0)
    mc = _failed_corner_campaign(spec_declarations=[declaration, dict(declaration)])
    assert list(mc.yield_fraction()) == [SPEC]
    assert mc.yield_fraction()[SPEC][0] == 0.5


def test_zero_observation_csv_keeps_declared_columns(tmp_path):
    import csv
    path = tmp_path / 'empty.csv'
    _declared_empty().to_csv(path)
    with path.open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames[:6] == ['run.corner', 'run.temperature', 'run.supply',
                                        'run.supply_source', 'run.sample', 'run.seed']
        assert [c for c in reader.fieldnames if c.startswith('metric.')] == [
            'metric.eye.width', 'metric.gain', 'metric.unspecced']
        assert {'check.' + SPEC, 'observed.temperature_state', 'run.error'} <= set(reader.fieldnames)
        assert list(reader) == []


@pytest.mark.parametrize('source', ['explicit', 'legacy'])
def test_conflicting_targets_are_compared_without_float_rounding(source):
    first = dict(name=SPEC, metric='gain', op='>', target=2**53)
    second = {**first, 'target': 2**53 + 1}
    reports, kwargs = {}, {}
    if source == 'explicit':
        kwargs['spec_declarations'] = [first, second]
    else:
        reports = {('tt', i): SpecReport('ac', 'amp', {'gain': 1.0}).check(
            SPEC, d['metric'], d['op'], d['target'])
            for i, d in enumerate((first, second))}
    with pytest.raises(ValueError) as caught:
        MonteCarloResults(reports, settings=MonteCarloSettings(n=2), seeds={}, **kwargs)
    assert str(2**53) in str(caught.value)
    assert str(2**53 + 1) in str(caught.value)


def _numeric_target_inventory(first_target, second_target, source):
    first = dict(name=SPEC, metric='gain', op='>', target=first_target)
    second = {**first, 'target': second_target}
    reports, kwargs = {}, {}
    if source == 'explicit':
        kwargs['spec_declarations'] = [first, second]
    else:
        reports[('ss', 0)] = SpecReport('ac', 'amp', {'gain': 0.0}).check(
            SPEC, 'gain', '>', second_target)
        if source == 'inferred':
            reports[('tt', 0)] = SpecReport('ac', 'amp', {'gain': 0.0}).check(
                SPEC, 'gain', '>', first_target)
        else:
            kwargs['spec_declarations'] = [first]
    return MonteCarloResults(reports, settings=MonteCarloSettings(n=2),
                             seeds={}, **kwargs)


@pytest.mark.parametrize('source', ['explicit', 'inferred', 'mixed'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('integer_type', [int, np.int64])
def test_numpy_promotion_cannot_hide_conflicting_targets(source, reverse, integer_type):
    targets = (integer_type(2**53 + 1), np.float64(2**53))
    if reverse:
        targets = targets[::-1]
    with pytest.raises(ValueError, match='conflicting definitions') as caught:
        _numeric_target_inventory(*targets, source)
    message = str(caught.value)
    assert SPEC in message and 'gain' in message and "op='>'" in message
    assert '9007199254740993' in message and '9007199254740992' in message


@pytest.mark.parametrize('source', ['explicit', 'inferred', 'mixed'])
@pytest.mark.parametrize('targets', [(1, 1.0), (np.float64(0.5), 0.5),
                                     (np.int64(2**53 + 1), 2**53 + 1)])
def test_equal_numeric_targets_keep_one_spec(source, targets):
    mc = _numeric_target_inventory(*targets, source)
    assert list(mc.yield_fraction()) == [SPEC]


def test_grid_yield_is_per_point_and_views_keep_failed_inventory():
    p = ('tt', -40.0, 1.62)
    q = ('tt', 125.0, 1.62)
    keys = [(point, i) for point in [p, q] for i in range(2)]
    reports = {keys[0]: _report(2), keys[1]: _report(3), keys[2]: _report(0.5)}
    mc = MonteCarloResults(reports, failed={keys[3]: SampleFailure('no data', 2)},
        seeds={k: i+1 for i,k in enumerate(keys)}, settings=MonteCarloSettings(n=2, min_yield=0.5),
        run_inventory=[dict(key=k, corner=k[0][0], temperature=k[0][1], supply=k[0][2]) for k in keys],
        spec_declarations=[dict(name=SPEC, metric='gain', op='>', target=1)])
    assert mc.yield_fraction()[SPEC][0] == 0.5
    assert not mc.passed  # pooled 0.5 hides q's zero yield
    assert mc.yield_by_point()[p][SPEC][0] == 1
    assert mc.yield_by_point()[q][SPEC][0] == 0
    assert mc.corners == ['tt']
    assert list(mc.corner('tt').reports) == list(reports)
    assert mc.point(q)._keys() == keys[2:]
    assert mc.table()[3]['run.status'] == 'failed'
    assert mc.table()[3]['run.corner'] == 'tt'
    with pytest.raises(MonteCarloError, match='point'):
        mc.measurement('gain')


def test_corner_only_yield_does_not_pool_bad_corner():
    reports = {('tt', i): _report(2) for i in range(3)}
    reports[('ss', 0)] = _report(0)
    mc = MonteCarloResults(reports, seeds={k:i for i,k in enumerate(reports)},
                          settings=MonteCarloSettings(n=3, min_yield=0.7))
    assert mc.yield_fraction()[SPEC][0] == 0.75
    assert not mc.passed


def test_retained_plan_drives_corner_order_even_if_seed_map_is_reordered():
    p=('ss',125.0,1.62)
    q=('tt',125.0,1.62)
    keys=[(p,0),(q,0)]
    mc=MonteCarloResults({keys[1]:_report(2),keys[0]:_report(3)},
        seeds={keys[1]:1,keys[0]:1},settings=MonteCarloSettings(n=1),
        run_inventory=[dict(key=k,corner=k[0][0],temperature=k[0][1],supply=k[0][2]) for k in keys])
    assert mc.corners==['ss','tt']
    assert list(mc.yield_by_point())==[p,q]


def test_missing_declared_check_is_unavailable_and_cannot_pass_without_floor():
    report=_report(2.0)
    mc=MonteCarloResults({('tt',0):report},seeds={('tt',0):1},settings=MonteCarloSettings(n=1),
        spec_declarations=[dict(name=SPEC,metric='gain',op='>',target=1),
                           dict(name='missing',metric='gain',op='<',target=3)])
    assert mc.table()[0]['check.missing']=='UNAVAILABLE'
    assert not mc.passed


@pytest.mark.parametrize('floor',[None,0.5])
def test_unavailable_check_evidence_does_not_count_as_yield(floor):
    report=SpecReport('ac','amp',{'flag':True}).check('flag','flag','>=',0)
    assert report.passed  # SpecReport's legacy boolean is unchanged.
    mc=MonteCarloResults({('tt',0):report},seeds={('tt',0):1},
                        settings=MonteCarloSettings(n=1,min_yield=floor))
    assert mc.table()[0]['check.flag']=='UNAVAILABLE'
    assert mc.yield_fraction()['flag'][0]==0
    assert not mc.passed


def test_corner_summary_names_floor_verdicts_and_informational_pooled_yield():
    values={'ss':[0.5,2,2,2],'tt':[2,2,2,2]}
    reports={(corner,i):_report(value) for corner,items in values.items()
             for i,value in enumerate(items)}
    mc=MonteCarloResults(reports,seeds={key:derive_sample_seed(0,key[1]) for key in reports},
                        settings=MonteCarloSettings(n=4,min_yield=0.8))
    assert mc.yield_fraction()[SPEC][0]==0.875
    assert not mc.passed
    text=mc.summary()
    assert 'corner=ss: 4 run, 0 failed, 4 used — FAIL' in text
    assert 'corner=tt: 4 run, 0 failed, 4 used — PASS' in text
    assert 'aggregate yield gain > 1: 0.875' in text
    assert '(informational)' in text
