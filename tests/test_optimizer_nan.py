"""NaN measurement policy at the optimizer consumer, without a simulator."""

from unittest.mock import Mock, call

import numpy as np
import pytest

from silk.circuit.variable import Variable
from silk.circuit.spec_report import SpecReport
from silk.metrics.ac import q_from_peak
from silk.optimize.optimizer import Optimizer
from silk.optimize.parameter_space import VariableSpace
from silk.signals.base import Signal


@pytest.fixture(params=['spec_report', 'legacy_dict'])
def result_factory(request):
    def make(metrics):
        if request.param == 'spec_report':
            return SpecReport('ac', 'amp', metrics)
        return {'metrics': metrics}
    return make


@pytest.fixture
def make_optimizer(result_factory):
    def make(*measurements):
        results = [result_factory(metrics) for metrics in measurements]
        test = Mock(run=Mock(side_effect=results))
        width = Variable('w', value=1.0, bounds=(1.0, 3.0))
        return Optimizer(test, VariableSpace([width]))
    return make


@pytest.fixture(params=[float, np.float32, np.float64],
                ids=['python_float', 'numpy_float32', 'numpy_float64'])
def nan(request):
    return request.param('nan')


@pytest.mark.parametrize('bounds', [
    {'min': 1.0}, {'max': 10.0}, {'min': 1.0, 'max': 10.0}, {},
])
def test_nan_constraint_is_infeasible(make_optimizer, nan, bounds):
    opt = make_optimizer({'q': nan, 'power': 2.0})
    opt.add_constraint('power', max=3.0)
    opt.add_constraint('q', **bounds)

    assert opt.feasible({'q': nan, 'power': 2.0}) is False
    record = opt.evaluate({'w': 1.0}, corner='ss')

    assert record['feasible'] is False
    assert record['status'] == 'success'
    assert record['metrics']['q'] is nan
    assert record['metrics']['power'] == 2.0
    assert record['error'] is record['error_type'] is None
    assert set(record) == {
        'id', 'params', 'corner', 'status', 'metrics', 'feasible',
        'error', 'error_type', 'timestamp', 'constraints', 'applied_params',
    }
    assert record['id'] == 0
    assert record['params'] == {'w': 1.0}
    assert record['corner'] == 'ss'
    assert isinstance(record['timestamp'], float)
    assert opt.history == opt.successes == [record]
    assert opt.failures == []
    opt.test.run.assert_called_once_with(corner='ss')


@pytest.mark.parametrize('nan_first', [True, False])
@pytest.mark.parametrize('constrained', [True, False])
@pytest.mark.parametrize('maximize', [True, False])
def test_nan_objective_cannot_be_best(
    make_optimizer, nan, nan_first, constrained, maximize,
):
    measurements = [{'q': nan, 'power': 2.0}, {'q': 2.0, 'power': 2.0}]
    if not nan_first:
        measurements.reverse()
    opt = make_optimizer(*measurements)
    opt.add_objective('q', maximize=maximize, minimize=not maximize)
    if constrained:
        opt.add_constraint('power', max=3.0)

    records = opt.sweep(n=2, method='grid', corner='ff', max_failures=1)

    assert len(records) == len(opt.history) == 2
    assert all(r['status'] == 'success' and r['feasible'] is True
               for r in records)
    valid = records[1 if nan_first else 0]
    undefined = records[0 if nan_first else 1]
    assert undefined['metrics']['q'] is nan
    assert opt.best is valid
    assert opt.summary()['best'] is valid
    assert opt.summary()['feasible_count'] == 2
    assert opt.failures == []
    assert opt.test.run.call_args_list == [call(corner='ff')] * 2


@pytest.mark.parametrize('constrained', [True, False])
@pytest.mark.parametrize('multi_objective', [True, False])
def test_all_nan_candidates_have_no_best(
    make_optimizer, constrained, multi_objective,
):
    opt = make_optimizer({'q': float('nan'), 'power': 2.0},
                         {'q': np.float64('nan'), 'power': 1.0})
    opt.add_objective('q', maximize=True)
    if multi_objective:
        opt.add_objective('power', minimize=True)
    if constrained:
        opt.add_constraint('q', min=1.0, max=10.0)

    records = opt.sweep(n=2, method='grid', max_failures=1)

    assert opt.history == opt.successes == records
    assert len(records) == 2
    assert all(np.isnan(r['metrics']['q']) for r in records)
    assert all(r['feasible'] is (not constrained) for r in records)
    assert opt.best is None
    summary = opt.summary()
    assert summary == {
        'total_evaluations': 2, 'by_status': {'success': 2},
        'feasible_count': 0 if constrained else 2, 'best': None,
        'objectives': ([{'name': 'q', 'goal': 'maximize', 'value_source': 'metric'}]
                       + ([{'name': 'power', 'goal': 'minimize',
                            'value_source': 'metric'}] if multi_objective else [])),
        'constraints': ([{'name': 'q', 'value_source': 'metric',
                          'min': 1.0, 'max': 10.0,
                          'source': 'explicit constraint', 'overrides': []}]
                        if constrained else []),
    }


@pytest.mark.parametrize('reverse', [True, False])
def test_any_nan_objective_excludes_multi_objective_candidate(
    make_optimizer, reverse,
):
    measurements = [
        {'q': float('nan'), 'power': -100.0},
        {'q': 100.0, 'power': np.float64('nan')},
        {'q': 2.0, 'power': 1.0},
        {'q': 4.0, 'power': 2.0},
    ]
    if reverse:
        measurements.reverse()
    opt = make_optimizer(*measurements)
    opt.add_objective('q', maximize=True)
    opt.add_objective('power', minimize=True)

    records = opt.sweep(n=4, method='grid')

    assert len(records) == 4
    assert all(r['feasible'] is True for r in records)
    assert opt.best['metrics'] == {'q': 4.0, 'power': 2.0}


def test_unrelated_nan_and_failed_report_spec_do_not_gate_best(
    result_factory, nan,
):
    result = result_factory({'q': 2.0, 'diagnostic': nan})
    if isinstance(result, SpecReport):
        result.check('diagnostic measurable', 'diagnostic', '>', 0.0)
        assert result.passed is False
    space = VariableSpace([Variable('w', value=1.0, bounds=(1.0, 3.0))])
    opt = Optimizer(Mock(run=Mock(return_value=result)), space)
    opt.add_objective('q', maximize=True)
    opt.add_constraint('q', min=1.0, max=10.0)

    record = opt.evaluate({'w': 1.0})

    assert record['status'] == 'success'
    assert record['feasible'] is True
    assert record['metrics']['diagnostic'] is nan
    assert opt.best is record


@pytest.mark.parametrize('reverse', [True, False])
def test_q_from_peak_sweep_preserves_undefined_response(make_optimizer, reverse):
    frequencies = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    responses = [np.zeros(5), np.array([0.0, 0.8, 1.0, 0.8, 0.0])]
    measurements = [
        {'q': q_from_peak(Signal('h', 'voltage', 'frequency', 'V',
                                frequencies, response))}
        for response in responses
    ]
    assert np.isnan(measurements[0]['q'])
    assert measurements[1]['q'] == 2.0
    if reverse:
        measurements.reverse()
    opt = make_optimizer(*measurements)
    opt.add_constraint('q', min=1.0, max=10.0)
    opt.add_objective('q', maximize=True)

    records = opt.sweep(n=2, method='grid', max_failures=1)

    assert opt.history == opt.successes == records
    assert len(records) == 2
    assert [r['id'] for r in records] == [0, 1]
    assert [r['params'] for r in records] == [{'w': 1.0}, {'w': 3.0}]
    for record, measurement in zip(records, measurements):
        assert record['metrics']['q'] is measurement['q']
        assert record['feasible'] is opt.feasible(measurement)
        assert record['feasible'] is (not np.isnan(measurement['q']))
    valid = records[0 if reverse else 1]
    assert opt.best is valid
    assert opt.summary()['feasible_count'] == 1


@pytest.mark.parametrize(('value', 'bounds', 'feasible'), [
    (1.0, {'min': 1.0}, True),
    (10.0, {'max': 10.0}, True),
    (1.0, {'min': 1.0, 'max': 10.0}, True),
    (10.0, {'min': 1.0, 'max': 10.0}, True),
    (0.5, {'min': 1.0, 'max': 10.0}, False),
    (10.5, {'min': 1.0, 'max': 10.0}, False),
    (float('inf'), {'min': 1.0}, True),
    (float('-inf'), {'max': 10.0}, True),
    (float('inf'), {'max': 10.0}, False),
    (float('-inf'), {'min': 1.0}, False),
    (float('inf'), {'max': float('inf')}, True),
    (float('-inf'), {'min': float('-inf')}, True),
])
def test_constraint_boundaries_and_infinity_are_unchanged(
    make_optimizer, value, bounds, feasible,
):
    opt = make_optimizer({'q': value})
    opt.add_constraint('q', **bounds)

    assert opt.feasible({'q': value}) is feasible
    assert opt.evaluate({'w': 1.0})['feasible'] is feasible


@pytest.mark.parametrize('maximize', [True, False])
@pytest.mark.parametrize('values', [
    [2.0, 1.0, 3.0, 3.0, 1.0],
    [2.0, float('inf'), float('-inf')],
    [float('inf'), float('inf')],
    [float('-inf'), float('-inf')],
])
def test_finite_and_infinite_ranking_preserves_first_tie(
    make_optimizer, maximize, values,
):
    opt = make_optimizer(*({'q': value} for value in values))
    opt.add_objective('q', maximize=maximize, minimize=not maximize)

    records = opt.sweep(n=len(values), method='grid')

    best_value = max(values) if maximize else min(values)
    assert opt.best is records[values.index(best_value)]


@pytest.mark.parametrize('value', [1+2j, np.complex128(1+2j)])
def test_complex_optimizer_consumers_require_projection(make_optimizer, value):
    opt = make_optimizer({'q': value})
    opt.add_constraint('q', min=0)
    with pytest.raises(ValueError, match=r'q.*complex.*\.real.*abs'):
        opt.feasible({'q': value})
    unconstrained = make_optimizer({'q': value})
    unconstrained.add_objective('q', minimize=True)
    unconstrained.evaluate({'w': 1.0})
    with pytest.raises(ValueError, match=r'q.*complex.*\.real.*abs'):
        _ = unconstrained.best


@pytest.mark.parametrize('missing', [{}, {'q': None}, {'q': 'unmeasured'}])
@pytest.mark.parametrize('multi_objective', [False, True])
def test_incomplete_objectives_have_no_best(make_optimizer, missing, multi_objective):
    measurements = [dict(missing, power=1.0), dict(missing, power=2.0)]
    opt = make_optimizer(*measurements).add_objective('q', maximize=True)
    if multi_objective:
        opt.add_objective('power', minimize=True)
    records = [opt.evaluate({'w': width}) for width in (1.0, 2.0)]

    assert opt.best is None
    assert opt.summary()['best'] is None
    assert opt.history == opt.successes == records
    assert opt.failures == []
    assert [r['metrics'] for r in records] == measurements
    assert all(r['feasible'] is True and r['status'] == 'success'
               and r['error'] is r['error_type'] is None for r in records)
    assert opt.summary()['feasible_count'] == 2


@pytest.mark.parametrize('missing', [{}, {'q': None}, {'q': 'unmeasured'}])
@pytest.mark.parametrize('incomplete_first', [True, False])
@pytest.mark.parametrize('maximize,measured', [
    (True, float('-inf')), (False, float('inf')),
    (True, 39.8), (False, 39.8),
])
def test_measured_objective_beats_incomplete_in_either_order(
    make_optimizer, missing, incomplete_first, maximize, measured,
):
    measurements = [missing, {'q': measured}]
    if not incomplete_first:
        measurements.reverse()
    opt = make_optimizer(*measurements)
    opt.add_objective('q', maximize=maximize, minimize=not maximize)
    records = [opt.evaluate({'w': width}) for width in (1.0, 2.0)]
    valid = records[1 if incomplete_first else 0]

    assert opt.best is valid
    assert opt.summary()['best'] is valid
    assert all(r['feasible'] is True for r in records)
    assert [r['metrics'] for r in records] == measurements


def test_objective_added_after_evaluation_excludes_earlier_incomplete_records(
    make_optimizer,
):
    opt = make_optimizer({'q': 100.0}, {'q': None, 'power': -100.0},
                         {'q': 4.0, 'power': 2.0}, {'q': 2.0, 'power': 1.0})
    opt.add_objective('q', maximize=True)
    early = opt.evaluate({'w': 1.0})
    assert opt.best is early
    opt.add_objective('power', minimize=True)
    assert opt.best is None
    records = [opt.evaluate({'w': width}) for width in (1.5, 2.0, 3.0)]

    assert opt.best is records[1]
    assert opt.history == [early, *records]
    assert all(r['feasible'] is True for r in opt.history)


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('legacy', [False, True])
def test_variable_objectives_use_recorded_values_with_missing_samples(
    make_optimizer, mixed, legacy,
):
    opt = make_optimizer(*([{'q': 0.0}] * 3 if mixed else [{}] * 3))
    width = opt.space._vars['w']
    opt.add_objective(width, maximize=True)
    if mixed:
        opt.add_objective('q', minimize=True)
    records = [opt.evaluate({'w': value}) for value in (1.0, 2.0, 3.0)]
    # Exercise persisted histories lacking a declared sample, including the
    # legacy params-only protocol. The live variable must not fill the gap.
    for record in records:
        if legacy:
            del record['applied_params']
        else:
            record['params']['w'] = 100.0
    records[2]['params' if legacy else 'applied_params'].clear()
    width.value = 1.0

    assert opt.best is records[1]
    assert all(r['feasible'] is True for r in records)


@pytest.mark.parametrize('constrained', [False, True])
def test_below_target_measurement_and_error_classification_stay_unchanged(
    make_optimizer, constrained,
):
    opt = make_optimizer({}, {'q': 39.8}, {'q': 'Error: extraction failed'})
    opt.add_objective('q', maximize=True)
    if constrained:
        opt.add_constraint('q', min=40.0)
    missing, measured, failed = [
        opt.evaluate({'w': value}) for value in (1.0, 2.0, 3.0)
    ]

    assert opt.best is (None if constrained else measured)
    assert missing['status'] == measured['status'] == 'success'
    assert missing['feasible'] is measured['feasible'] is (not constrained)
    assert measured['metrics'] == {'q': 39.8}
    assert failed['status'] == 'metric_error'
    assert failed['error_type'] == 'MetricExtractionError'
    assert opt.failures == [failed]
    assert opt.history == [missing, measured, failed]


@pytest.mark.parametrize('sample', [{}, {'w': None}, {'w': 'unmeasured'}])
@pytest.mark.parametrize('legacy', [False, True])
def test_incomplete_recorded_variable_objective_has_no_best(
    make_optimizer, sample, legacy,
):
    opt = make_optimizer({})
    width = opt.space._vars['w']
    opt.add_objective(width)
    record = opt.evaluate({'w': 2.0})
    if legacy:
        del record['applied_params']
        record['params'] = sample
    else:
        record['applied_params'] = sample
    width.value = 3.0

    assert opt.best is None
    assert record['metrics'] == {}
    assert record['status'] == 'success'
    assert record['feasible'] is True
