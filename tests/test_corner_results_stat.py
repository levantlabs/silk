"""CornerResults.stat: aggregation of one metric across runs (TASK-370)."""
import math
import warnings

import pytest

from silk.circuit.spec_report import CornerResults, SpecReport
from silk.metrics.diagnostics import MetricWarning


def _rep(v, spec=None):
    r = SpecReport('t', 'm', {'gain': v})
    if spec:
        r.check('g', 'gain', *spec)
    return r


def _cr(vals, spec=None):
    return CornerResults({c: _rep(v, spec) for c, v in vals.items()})


def test_basic_stats():
    cr = _cr({'tt': 2.0, 'ss': 4.0, 'ff': 6.0})
    assert cr.stat('gain', 'mean') == 4.0
    assert cr.stat('gain', 'rms') == pytest.approx(math.sqrt(56 / 3))
    assert cr.stat('gain', 'std') == pytest.approx(2.0)
    assert cr.stat('gain', 'min') == 2.0
    assert cr.stat('gain', 'max') == 6.0


def test_worst_lower_bound_and_upper_bound():
    cr = _cr({'tt': 60.0, 'ss': 52.0, 'ff': 70.0}, ('>', 50.0))
    assert cr.worst('gain') == (52.0, 'ss')
    cr = _cr({'tt': 60.0, 'ss': 52.0, 'ff': 70.0}, ('<', 65.0))
    assert cr.worst('gain') == (70.0, 'ff')


def test_worst_furthest_past_when_failing():
    cr = _cr({'tt': 40.0, 'ss': 30.0}, ('>=', 50.0))
    w = cr.worst('gain')
    assert (w.value, w.corner) == (30.0, 'ss')


@pytest.mark.parametrize('bad', ['failed: boom', float('nan'), None])
def test_failed_or_nan_corner_gives_nan_with_warning(bad):
    cr = _cr({'tt': 1.0, 'ss': bad, 'ff': 3.0}, ('>', 0.0))
    with pytest.warns(MetricWarning) as w:
        assert math.isnan(cr.stat('gain', 'mean'))
    assert w[0].message.details['corners'] == ['ss']
    with pytest.warns(MetricWarning):
        wc = cr.worst('gain')
    assert math.isnan(wc.value) and wc.corner is None


def test_clean_run_emits_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        _cr({'a': 1.0, 'b': 2.0}).stat('gain', 'max')


def test_unknown_metric_and_how_name_valid_ones():
    cr = _cr({'tt': 1.0})
    with pytest.raises(ValueError, match="gain"):
        cr.stat('nope', 'mean')
    with pytest.raises(ValueError, match="rms"):
        cr.stat('gain', 'median')


def test_worst_without_spec_raises():
    with pytest.raises(ValueError, match="no spec"):
        _cr({'tt': 1.0}).worst('gain')


def test_stat_rejects_worst_and_always_returns_float():
    cr = _cr({'tt': 1.0, 'ss': 2.0}, ('>', 0.0))
    with pytest.raises(ValueError, match="rms"):
        cr.stat('gain', 'worst')
    assert all(isinstance(cr.stat('gain', h), float) for h in cr.STAT_KINDS)


def test_std_needs_two_runs():
    with pytest.warns(MetricWarning):
        assert math.isnan(_cr({'tt': 1.0}).stat('gain', 'std'))


def test_handle_accepted_wherever_a_name_is():
    from unittest.mock import Mock
    from silk.circuit.metric_handle import MetricHandle
    h = MetricHandle(Mock(), ('metric', 'gain'))
    cr = _cr({'tt': 60.0, 'ss': 52.0}, ('>', 50.0))
    assert cr.stat(h, 'mean') == cr.stat('gain', 'mean')
    assert cr.worst(h) == cr.worst('gain') == (52.0, 'ss')
    assert cr['tt'].metrics[h] == 60.0


@pytest.mark.parametrize('value', [1+2j, complex(float('nan'), 1)])
def test_complex_corner_reductions_refuse_projection(value):
    cr = _cr({'tt': value}, ('>', 0.0))
    for action in (lambda: cr.stat('gain', 'mean'), lambda: cr.worst('gain')):
        with pytest.raises(ValueError, match=r'gain.*complex.*\.real.*abs'):
            action()
    assert cr.reports['tt'].metrics['gain'] is value


def test_grid_table_preserves_order_conditions_and_spec_keys(tmp_path):
    from silk.circuit.spec_report import CornerResults, SpecReport
    from silk.simulation.conditions import OperatingConditions, ConditionSweep
    keys = [('ss', 125.0, 1.62), ('ss', -40.0, 1.62), ('tt', 125.0, 1.62)]
    report = SpecReport('ac', 'amp', {'gain': 61.0, 'corner': 7.0})
    report.check('same label', 'gain', '>=', 60, spec_key='floor')
    report.check('same label', 'gain', '<=', 70, spec_key='ceiling')
    report.operating_conditions = OperatingConditions(
        'silk', 'ngspice', 'ss', corner_basis='configured-and-emitted',
        temperature_kind='sweep', temperature_basis='configured',
        temperature_sweep=ConditionSweep(-40, 125, 5))
    inventory = [dict(key=k, corner=k[0], temperature=k[1], supply=k[2], supply_source='vdd') for k in keys]
    results = CornerResults({keys[0]: report, keys[2]: report}, failed={keys[1]: 'no data'},
                            run_inventory=inventory, metric_names=['gain', 'corner'],
                            check_declarations=[dict(key='floor', name='same label'), dict(key='ceiling', name='same label')])
    rows = results.table()
    assert [r['run.temperature'] for r in rows] == [125, -40, 125]
    assert rows[0]['observed.temperature_state'] == 'sweep'
    assert rows[0]['observed.temperature_sweep.stop'] == 125
    assert rows[0]['observed.supply_state'] == 'unknown'
    assert rows[1]['observed.supply_state'] is None
    assert rows[1]['run.status'] == 'failed'
    assert rows[1]['check.floor'] == 'UNAVAILABLE'
    assert rows[0]['check.floor'] == rows[0]['check.ceiling'] == 'PASS'
    assert rows[0]['metric.corner'] == 7
    assert not results.passed
    assert 'planned=3: 2 reports (2 PASS, 0 FAIL), 1 failed and produced no report' in results.summary()
    results.to_csv(tmp_path / 'grid.csv')
    import csv
    loaded = list(csv.DictReader((tmp_path / 'grid.csv').open()))
    assert loaded[1]['check.ceiling'] == 'UNAVAILABLE'
    assert loaded[1]['observed.temperature_state'] == ''


@pytest.mark.parametrize('value,expected', [(59.0,'FAIL'), (61.0,'PASS'),
    (float('nan'),'UNAVAILABLE'), (None,'UNAVAILABLE'), (1+2j,'UNAVAILABLE'),
    ('extractor failed','UNAVAILABLE'), (float('inf'),'PASS')])
def test_check_cells_use_captured_measurement_evidence(value, expected):
    key = ('tt', 27.0, 1.8)
    report = SpecReport('ac','amp',{'gain':value}).check('gain floor','gain','>=',60,spec_key='floor')
    cr = CornerResults({key:report})
    assert cr.table()[0]['check.floor'] == expected


def test_grid_summary_keeps_structured_and_complex_outputs_in_table():
    key = ('tt',27.0,1.8)
    values = {'eye': {'width': 2}, 'transfer': 1+2j, 'eye.width':2.0}
    cr = CornerResults({key:SpecReport('ac','amp',values)})
    assert cr.table()[0]['metric.eye'] == values['eye']
    assert cr.table()[0]['metric.transfer'] == 1+2j
    text=cr.summary()
    assert 'eye: not summarized' in text and 'transfer: not summarized' in text
    assert 'eye.width: used=1' in text


def test_grid_worst_summary_keeps_adjacent_coordinates_distinct():
    first = ('tt',125.0,1.62)
    second = ('tt',125.0,1.6200000000000003)
    cr = CornerResults({first:_rep(61, ('>=',60)), second:_rep(59, ('>=',60))})
    assert 'temperature=125 supply=1.6200000000000003' in cr.summary()
