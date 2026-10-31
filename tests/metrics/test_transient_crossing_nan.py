"""Crossing measurements must not skip unknown earlier events (TASK-257)."""
import math
import warnings

import numpy as np
import pytest

from silk.circuit.spec_report import SpecReport
import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def signal(y, *, x=None, name='waveform'):
    return Signal(name, 'voltage', 'time', 'V',
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x),
                  np.asarray(y, dtype=float))


CASES = [
    ('propagation_delay', [0, np.nan, 1, 0, 1], 2,
     {'in_threshold': .5, 'out_threshold': .5}, 1),
    ('average_period', [0, np.nan, 1, 0, 1, 0, 1], 1, {'threshold': .5}, 10),
    ('pulse_width', [0, np.nan, 1, 0, 1, 0], 1, {'threshold': .5}, 2),
    ('hysteresis', [0, np.nan, 1, 0, 1, 0], 4, {'out_threshold': .5}, 1),
]


@pytest.mark.parametrize('name,y,count,kwargs,target', CASES, ids=[c[0] for c in CASES])
def test_recorded_false_pass_fails_spec(name, y, count, kwargs, target):
    # Keep the spec verdict first: each base failure demonstrates the false pass,
    # independently of the new diagnostic side channel.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = getattr(M, name)(*[signal(y) for _ in range(count)], **kwargs)
    report = SpecReport('tran', 'dut', {name: value})
    assert not report.check('upper', name, '<', target).passed
    assert type(value) is float and math.isnan(value)
    for op in ('<=', '>', '>=', '=='):
        report = SpecReport('tran', 'dut', {name: value})
        assert not report.check('spec', name, op, target).passed
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert isinstance(diagnostic, MetricWarning)
    assert diagnostic.metric == name
    assert diagnostic.code == 'nan_waveform'
    assert 'consumed data contains NaNs' in diagnostic.message
    assert 'source result' in diagnostic.suggestion



@pytest.mark.parametrize('name,y,count,kwargs,target', CASES, ids=[c[0] for c in CASES])
def test_default_threshold_diagnoses_the_same_unavailable_data(name, y, count, kwargs, target):
    with pytest.warns(MetricWarning) as caught:
        value = getattr(M, name)(*[signal(y) for _ in range(count)])
    assert type(value) is float and math.isnan(value)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == name
    assert diagnostic.details['purpose'] == {
        'average_period': 'period_search', 'pulse_width': 'crossing_search',
    }.get(name, 'threshold')


@pytest.mark.parametrize('name,count,role,kwargs', [
    ('propagation_delay', 2, 'sig_in', {'in_threshold': .5, 'out_threshold': .5}),
    ('propagation_delay', 2, 'sig_out', {'in_threshold': .5, 'out_threshold': .5}),
    ('hysteresis', 4, 'sig_out_rising', {'out_threshold': .5}),
    ('hysteresis', 4, 'sig_out_falling', {'out_threshold': .5}),
])
@pytest.mark.parametrize('explicit', [False, True])
def test_each_crossing_signal_is_checked(name, count, role, kwargs, explicit):
    roles = (['sig_in', 'sig_out'] if count == 2 else
             ['sig_in_rising', 'sig_out_rising', 'sig_in_falling', 'sig_out_falling'])
    args = [signal([0, 0, 1, 0, 1, 0], name=r) for r in roles]
    args[roles.index(role)] = signal([0, np.nan, 1, 0, 1, 0], name=role)
    with pytest.warns(MetricWarning) as caught:
        value = getattr(M, name)(*args, **(kwargs if explicit else {}))
    assert math.isnan(value)
    # TASK-438: hysteresis also reports multi-crossing outputs (its own code)
    caught = [w for w in caught if w.message.code != 'multiple_crossings']
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == name
    assert diagnostic.details['signal_role'] == role
    assert diagnostic.details['signal_name'] == role
    assert diagnostic.details['nan_sample_count'] == 1
    assert diagnostic.details['nan_time_start_s'] == 1
    assert diagnostic.details['nan_time_stop_s'] == 1
    assert all(type(v) in (str, float, int, type(None))
               for v in diagnostic.details.values())


@pytest.mark.parametrize('edge', ['rising', 'falling', 'any'])
def test_unknown_earlier_interval_cannot_select_a_later_direction(edge):
    s = signal([0, np.nan, 1, 0, 1, 0])
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.propagation_delay(s, s, in_threshold=.5, out_threshold=.5, in_edge=edge))
    assert len(caught) == 1
    assert caught[0].message.details['purpose'] == 'crossing_search'
    assert caught[0].message.details['sample_count'] == 2
    assert caught[0].message.details['time_start_s'] == 0
    assert caught[0].message.details['time_stop_s'] == 1


@pytest.mark.parametrize('y', [[0, 1, np.nan, 0, 1, 0], [0, .5, np.nan, 0, 1, 0]])
def test_pulse_falling_search_cannot_skip_unknown_interval(y):
    # TASK-367: pulse_width follows the crossings contract, so the NaN
    # check covers every consumed sample (the whole trace without a window).
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.pulse_width(signal(y), threshold=.5))
    assert len(caught) == 1
    assert caught[0].message.details['purpose'] == 'crossing_search'
    assert caught[0].message.details['time_start_s'] == 0
    assert caught[0].message.details['nan_time_start_s'] == 2


def test_pulse_window_before_nan_tail_is_measured():
    # TASK-367: a NaN tail outside [start, end] is not consumed.
    s = signal([0, 1, 0, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert M.pulse_width(s, threshold=.5, end=2) == pytest.approx(1)
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.pulse_width(s, threshold=.5))
    assert caught[0].message.details['purpose'] == 'crossing_search'


@pytest.mark.parametrize('name,count,kwargs,expected', [
    ('propagation_delay', 2, {'in_threshold': .5, 'out_threshold': .5}, 0),
    ('hysteresis', 4, {'out_threshold': .5}, 0),
])
def test_explicit_threshold_keeps_completed_measurement_before_nan_tail(name, count, kwargs, expected):
    args = [signal([0, 1, 0, np.nan]) for _ in range(count)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = getattr(M, name)(*args, **kwargs)
    assert type(value) is float
    assert value == pytest.approx(expected)
    assert caught == []
    # The same tail IS consumed to determine the automatic threshold.
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(M, name)(*args))
    assert len(caught) == 1
    assert caught[0].message.details['purpose'] == 'threshold'


@pytest.mark.parametrize('threshold', [None, .5])
def test_period_search_consumes_unknown_tail_after_completed_cycles(threshold):
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.average_period(signal([0, 1, 0, 1, 0, np.nan]), threshold))
    assert len(caught) == 1
    assert caught[0].message.details['purpose'] == 'period_search'
    assert caught[0].message.details['sample_count'] == 6


@pytest.mark.parametrize('out_threshold', [None, .5])
def test_hysteresis_uses_only_input_interpolation_samples(out_threshold):
    args = [signal([np.nan, 0, 2, np.nan]), signal([0, 0, 1, 1]),
            signal([np.nan, 4, 0, np.nan]), signal([1, 1, 0, 0])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = M.hysteresis(*args, out_threshold=out_threshold)
    assert value == -1
    assert caught == []


def test_hysteresis_default_does_not_consume_falling_output_tail_for_threshold():
    # The shared automatic threshold comes from the rising output only.
    args = [signal([0, 1, 2, 3]), signal([0, 1, 1, 1]),
            signal([3, 2, 1, 0]), signal([1, 0, np.nan, np.nan])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert M.hysteresis(*args) == -2
    assert caught == []


@pytest.mark.parametrize('in_threshold,out_threshold,expected_role', [
    (None, .5, 'sig_in'), (.5, None, 'sig_out'),
])
def test_independent_auto_threshold_consumption(in_threshold, out_threshold, expected_role):
    s = signal([0, 1, np.nan])
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.propagation_delay(
            s, s, in_threshold=in_threshold, out_threshold=out_threshold,
        ))
    assert len(caught) == 1
    assert caught[0].message.details['signal_role'] == expected_role
    assert caught[0].message.details['purpose'] == 'threshold'


@pytest.mark.parametrize('threshold', [None, .5])
@pytest.mark.parametrize('name,y,count,kwargs,target', CASES, ids=[c[0] for c in CASES])
def test_single_nan_sample_is_unavailable_without_numpy_warnings(name, y, count, kwargs, target, threshold):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = getattr(M, name)(*[signal([np.nan]) for _ in range(count)],
                                 **(kwargs if threshold is not None else {}))
    assert type(value) is float and math.isnan(value)
    assert len(caught) == 1
    assert isinstance(caught[0].message, MetricWarning)


@pytest.mark.parametrize('name,y,count,kwargs,target', CASES, ids=[c[0] for c in CASES])
def test_warning_filter_preserves_float_and_error_policy_is_explicit(name, y, count, kwargs, target):
    args = [signal(y) for _ in range(count)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore', MetricWarning)
        value = getattr(M, name)(*args, **kwargs)
    assert type(value) is float and math.isnan(value)
    assert caught == []
    with warnings.catch_warnings():
        warnings.simplefilter('error', MetricWarning)
        with pytest.raises(MetricWarning) as caught:
            getattr(M, name)(*args, **kwargs)
    assert caught.value.metric == name


@pytest.mark.parametrize('name,y,count,kwargs,target', CASES, ids=[c[0] for c in CASES])
def test_empty_or_wrong_axis_remains_caller_error(name, y, count, kwargs, target):
    for malformed in (signal([]), Signal('ac', 'voltage', 'frequency', 'V',
                                       np.array([1., 2.]), np.array([0., np.nan]))):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(ValueError):
                getattr(M, name)(*[malformed for _ in range(count)], **kwargs)
        assert caught == []


def test_invalid_edge_precedes_nan_diagnostic():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='propagation_delay: in_edge'):
            M.propagation_delay(signal([0, np.nan]), signal([0, np.nan]), in_edge='typo')
    assert caught == []


def test_hysteresis_axis_mismatch_precedes_nan_diagnostic():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='rising-sweep input and output must share x-axis'):
            M.hysteresis(signal([0, 1], x=[0., 2.]), signal([0, np.nan]),
                         signal([1, 0]), signal([1, 0]))
    assert caught == []


@pytest.mark.parametrize('name,y,count,kwargs,target', CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize('explicit', [False, True])
def test_finite_measurements_keep_existing_values_without_warning(name, y, count, kwargs, target, explicit):
    values = [0, 0, 1, 0, 1, 0, 1]
    expected = {'propagation_delay': 0, 'average_period': 2,
                'pulse_width': 1, 'hysteresis': 0}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = getattr(M, name)(*[signal(values) for _ in range(count)],
                                 **(kwargs if explicit else {}))
    assert type(value) is float
    assert value == pytest.approx(expected[name])
    # TASK-438: this square output crosses repeatedly, which hysteresis reports
    allowed = {'multiple_crossings'} if name == 'hysteresis' else set()
    assert [w for w in caught if w.message.code not in allowed] == []


@pytest.mark.parametrize('edge,y,expected', [
    ('rising', [0, .5, np.nan], 0),
    ('any', [0, .5, np.nan], 0),
    ('falling', [1, .5, 0, np.nan], 0),
])
def test_completed_boundary_crossing_is_preserved(edge, y, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert M.propagation_delay(signal(y), signal(y), in_threshold=.5, out_threshold=.5, in_edge=edge) == expected
    assert caught == []


def test_starting_on_threshold_does_not_hide_unknown_first_rise():
    with pytest.warns(MetricWarning):
        assert math.isnan(M.propagation_delay(
            signal([.5, np.nan, 0, 1]), signal([.5, np.nan, 0, 1]),
            in_threshold=.5, out_threshold=.5, in_edge='rising',
        ))


@pytest.mark.parametrize('y', [[0, np.nan, 1], [0, 0, 0]])
def test_invalid_output_threshold_remains_caller_error(y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='could not convert string to float'):
            M.propagation_delay(signal(y), signal(y),
                                in_threshold=.5, out_threshold='invalid')
    assert caught == []


def test_period_invalid_threshold_precedes_nan_diagnostic():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='could not convert string to float'):
            M.average_period(signal([0, np.nan, 1]), threshold='invalid')
    assert caught == []
