"""TASK-268 ledger coverage without changing waveform math or consumed support."""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def signal(y, *, x=None, axis='time', current=False):
    return Signal('current' if current else 'waveform',
                  'current' if current else 'voltage', axis,
                  'A' if current else 'V',
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x),
                  np.asarray(y))


WINDOWS = ('average', 'rms', 'stddev', 'peak', 'peak_to_peak',
           'minimum', 'maximum', 'argmin_x', 'argmax_x', 'integral')
NAN = signal([1, np.nan, 1])
FINITE = signal([1, 1, 1])
CURRENT = signal([1, 1, 1], current=True)
LEDGER = [
    *[(name, (NAN,), {}, 'nan_waveform') for name in WINDOWS],
    ('overshoot', (NAN, 0, 1), {}, 'nan_waveform'),
    ('undershoot', (NAN, 0, 1), {}, 'nan_waveform'),
    ('average_power', (NAN, CURRENT), {}, 'nan_power'),
    ('dynamic_energy_per_cycle', (NAN, CURRENT, 1), {}, 'nan_power'),
    ('value_at', (FINITE, -1), {}, 'point_outside_trace'),
    ('value_at', (NAN, 1), {}, 'nan_waveform'),
    ('error_norm', (NAN, FINITE), {'kind': 'max'}, 'nan_waveform'),
    ('error_norm', (NAN, FINITE), {'kind': 'rms'}, 'nan_waveform'),
    ('running_integral', (NAN,), {}, 'nan_waveform'),
    ('derivative', (NAN,), {}, 'nan_waveform'),
    ('convolve', (NAN, FINITE), {}, 'nan_waveform'),
    ('cross_correlation', (NAN, FINITE), {}, 'nan_waveform'),
]


@pytest.mark.parametrize('name,args,kwargs,code', LEDGER)
@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_ledger_warnings_follow_policy(name, args, kwargs, code, policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning) as error:
                getattr(M, name)(*args, **kwargs)
            diagnostic = error.value
        else:
            result = getattr(M, name)(*args, **kwargs)
            if isinstance(result, Signal):
                assert np.any(np.isnan(result.y))
                assert np.all(np.isfinite(result.x))
            else:
                assert type(result) is float and math.isnan(result)
            if policy == 'ignore':
                assert caught == []
                return
            assert len(caught) == 1
            diagnostic = caught[0].message
    assert type(diagnostic) is MetricWarning
    assert diagnostic.metric == name
    assert diagnostic.code == code
    assert diagnostic.message and diagnostic.suggestion
    details = diagnostic.details
    assert any('sample_count' in key for key in details)
    assert any(key.endswith(('_min', '_start_s')) for key in details)
    assert any(key.endswith(('_max', '_stop_s')) for key in details)
    assert any(key.endswith('_unit') for key in details)
    assert all(type(value) in (str, float, int, type(None))
               for value in details.values())


@pytest.mark.parametrize('name', WINDOWS)
@pytest.mark.parametrize('axis,unit', [('time', 's'), ('frequency', 'Hz')])
def test_windows_warn_only_for_selected_samples(name, axis, unit):
    sig = signal([np.nan, 1, 1, np.nan], axis=axis)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = getattr(M, name)(sig, start=1, end=2)
    expected = {'stddev': 0, 'peak_to_peak': 0, 'argmin_x': 1, 'argmax_x': 1}
    assert result == expected.get(name, 1)
    assert caught == []
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(M, name)(sig, start=.5, end=3,
                                           interpolate=False))
    assert len(caught) == 1
    details = caught[0].message.details
    assert details['sample_count'] == 3
    assert details['nan_sample_count'] == 1
    assert (details['x_min'], details['x_max']) == (1, 3)
    assert details['x_unit'] == unit and details['y_unit'] == 'V'
    # TASK-378: the interpolated edge at .5 lies on the NaN segment
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(M, name)(sig, start=.5, end=3))
    details = caught[0].message.details
    assert details['sample_count'] == 4
    assert details['nan_sample_count'] == 2
    assert (details['x_min'], details['x_max']) == (.5, 3)


@pytest.mark.parametrize('name', WINDOWS)
def test_complex_window_contract_and_explicit_magnitude(name):
    sig = signal([3 + 4j, 3 + 4j, complex(1, np.nan)], axis='frequency')
    if name not in ('average', 'rms', 'stddev', 'integral'):
        with pytest.raises(ValueError, match=name + '.*complex'):
            getattr(M, name)(sig, end=1)
    sig = sig.mag()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        finite = getattr(M, name)(sig, end=1)
    assert type(finite) is float and math.isfinite(finite)
    assert caught == []
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(M, name)(sig))
    assert len(caught) == 1
    assert caught[0].message.details['nan_sample_count'] == 1


@pytest.mark.parametrize('name', ['value_at'])
@pytest.mark.parametrize('point,count,bounds', [(1, 1, (1, 1)), (.5, 2, (0, 1)), (1.5, 2, (1, 2))])
def test_point_nan_diagnostics_describe_exact_support(name, point, count, bounds):
    sig = signal([1, np.nan, 1])
    args = {'value_at': (sig, point)}[name]
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(M, name)(*args))
    assert len(caught) == 1
    details = caught[0].message.details
    assert details['sample_count'] == count
    assert details['nan_sample_count'] == 1
    assert (details['x_min'], details['x_max']) == bounds
    assert details['x_unit'] == 's'


@pytest.mark.parametrize('name,args', [
    ('value_at', (NAN, 0)), ('value_at', (NAN, 2)),
    ('value_at', (signal([np.nan]), 2)),
])
def test_unconsumed_point_nans_and_outside_coverage(name, args):
    outside = len(args[0].y) == 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = getattr(M, name)(*args)
    if outside:
        assert math.isnan(value)
        assert len(caught) == 1
        d = caught[0].message
        assert d.code == 'point_outside_trace'
        assert d.details['sample_count'] == 0
        assert d.details['available_sample_count'] == 1
    else:
        assert math.isfinite(value)
        assert caught == []


@pytest.mark.parametrize('point', [0, 2])
def test_nan_at_exact_endpoint(point):
    sig = signal([np.nan, 1, np.nan])
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.value_at(sig, point))
    assert caught[0].message.details['sample_count'] == 1
    assert caught[0].message.details['nan_sample_count'] == 1


@pytest.mark.parametrize('name', ['average_power', 'dynamic_energy_per_cycle'])
@pytest.mark.parametrize('role', ['voltage', 'current'])
def test_power_warns_on_either_operand(name, role):
    v = NAN if role == 'voltage' else FINITE
    i = signal([1, np.nan, 1] if role == 'current' else [1, 1, 1], current=True)
    args = (v, i, 1) if name == 'dynamic_energy_per_cycle' else (v, i)
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(M, name)(*args))
    assert len(caught) == 1
    d = caught[0].message.details
    assert d['sample_count'] == 3 and d['nan_sample_count'] == 1
    assert (d['time_start_s'], d['time_stop_s']) == (0, 2)
    assert (d['voltage_unit'], d['current_unit'], d['power_unit']) == ('V', 'A', 'W')


def test_complete_cycle_tail_is_not_consumed():
    sig = signal([1, 1, 1, 1, np.nan], x=[0, .5, 1, 1.5, 1.75])
    current = signal([2] * 5, x=sig.x, current=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert M.dynamic_energy_per_cycle(sig, current, 1) == 2
    assert caught == []
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.dynamic_energy_per_cycle(sig, current, 1.75))
    assert caught[0].message.details['complete_cycle_count'] == 1
    assert caught[0].message.details['time_stop_s'] == 1.75


@pytest.mark.parametrize('complex_values', [False, True])
@pytest.mark.parametrize('name,expected_y,expected_x', [
    ('running_integral', [0, np.nan, np.nan], [0, 1, 2]),
    ('derivative', [np.nan, 0, np.nan], [0, 1, 2]),
    ('convolve', [1, np.nan, np.nan, np.nan, 1], [0, 1, 2, 3, 4]),
    ('cross_correlation', [1, np.nan, np.nan, np.nan, 1], [-2, -1, 0, 1, 2]),
])
def test_partial_waveform_outputs_remain_intact(name, expected_y, expected_x, complex_values):
    sig = signal([1, complex(1, np.nan), 1] if complex_values else [1, np.nan, 1])
    args = (sig, FINITE) if name in ('convolve', 'cross_correlation') else (sig,)
    before = sig.y.copy()
    with pytest.warns(MetricWarning) as caught:
        out = getattr(M, name)(*args)
    assert len(caught) == 1
    assert np.iscomplexobj(out.y) == complex_values
    np.testing.assert_allclose(out.y, expected_y, equal_nan=True)
    np.testing.assert_array_equal(out.x, expected_x)
    np.testing.assert_array_equal(sig.y, before)


@pytest.mark.parametrize('name', ['convolve', 'cross_correlation'])
def test_binary_waveform_warning_combines_both_operand_facts(name):
    a = signal([1, np.nan, 1], x=[2, 3, 4])
    b = signal([np.nan, 1], x=[8, 9])
    with pytest.warns(MetricWarning) as caught:
        out = getattr(M, name)(a, b)
    assert len(out.y) == 4 and len(caught) == 1
    d = caught[0].message.details
    assert d['a_sample_count'] == 3 and d['b_sample_count'] == 2
    assert d['a_nan_sample_count'] == d['b_nan_sample_count'] == 1
    assert (d['a_x_min'], d['a_x_max'], d['b_x_min'], d['b_x_max']) == (2, 4, 8, 9)


@pytest.mark.parametrize('name,args,expected', [
    ('overshoot', (FINITE, 0, 1), 0), ('undershoot', (FINITE, 0, 1), 0),
    ('average_power', (FINITE, CURRENT), 1),
    ('dynamic_energy_per_cycle', (FINITE, CURRENT, 1), 1),
    ('value_at', (FINITE, 1), 1),
    ('error_norm', (FINITE, FINITE), 0),
    ('running_integral', (FINITE,), [0, 1, 2]),
    ('derivative', (FINITE,), [0, 0, 0]),
    ('convolve', (FINITE, FINITE), [1, 2, 3, 2, 1]),
    ('cross_correlation', (FINITE, FINITE), [1, 2, 3, 2, 1]),
    ('running_integral', (signal([np.nan]),), [0]),
    ('running_integral', (signal([complex(1, np.nan)]),), [0]),
    ('value_at', (signal([np.inf, np.inf]), .5), np.inf),
    ('average', (signal([np.inf, np.inf]),), np.inf),
    ('peak', (signal([0, 0]),), 0),
])
def test_success_valid_zero_infinity_and_no_interval_stay_quiet(name, args, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = getattr(M, name)(*args)
    np.testing.assert_allclose(result.y if isinstance(result, Signal) else result, expected)
    assert caught == []


@pytest.mark.parametrize('name,args', [
    ('overshoot', (FINITE, 0, np.nan)), ('undershoot', (FINITE, 0, np.nan)),
    ('value_at', (NAN, np.nan)),
])
def test_deferred_nan_scalar_provenance_is_not_called_waveform_contamination(name, args):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert math.isnan(getattr(M, name)(*args))
    assert caught == []


@pytest.mark.parametrize('kind', ['max', 'rms'])
def test_error_norm_keeps_existing_warning_and_resampled_goal_locality(kind):
    goal = signal([np.nan, 1, 1, np.nan], x=[-1, 0, 1, 2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert M.error_norm(signal([1, 1], x=[0, 1]), goal, kind=kind) == 0
    assert caught == []
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.error_norm(signal([1, 1], x=[-.5, .5]), goal, kind=kind))
    assert len(caught) == 1
    d = caught[0].message.details
    assert d['goal_nan_sample_count'] == 1
    assert (d['x_min'], d['x_max'], d['x_unit']) == (-.5, .5, 's')


@pytest.mark.parametrize('name,args,kwargs', [
    *[(name, (NAN,), {'start': 10, 'end': 20}) for name in WINDOWS],
    ('average_power', (NAN, FINITE), {}),
    ('dynamic_energy_per_cycle', (NAN, CURRENT, 3), {}),
    ('derivative', (signal([np.nan]),), {}),
    ('error_norm', (NAN, FINITE), {'kind': 'invalid'}),
    ('convolve', (NAN, signal([np.nan, 1, 1], x=[0, 2, 3])), {}),
    ('cross_correlation', (NAN, signal([np.nan, 1, 1], x=[0, 1, 3])), {}),
])
def test_existing_request_errors_precede_new_diagnostics(name, args, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError):
            getattr(M, name)(*args, **kwargs)
    assert caught == []


def test_composed_calls_keep_origin_without_repeating_upstream_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert M.value_at(M.running_integral(NAN), 0) == 0
    assert len(caught) == 1
    assert caught[0].message.metric == 'running_integral'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert math.isnan(M.average(M.window(NAN, 0, 2)))
    assert len(caught) == 1
    assert caught[0].message.metric == 'average'


def test_transport_only_nan_transforms_stay_quiet():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for out in (M.lshift(NAN, 1), M.flip(NAN), M.window(NAN, 0, 2), M.sample(NAN, n=3)):
            assert np.any(np.isnan(out.y))
    assert caught == []


@pytest.mark.parametrize('name,params,expected', [
    ('average', {'end': 1}, 3+4j),
    ('integral', {'end': 1}, 3+4j),
    ('stddev', {'end': 1}, 0.0),
    ('variance', {'end': 1}, 0.0),
    ('value_at', {'x': 1}, 3+4j),
])
def test_complex_defined_metrics_only_warn_for_consumed_nan(name, params, expected):
    sig = signal([3+4j, 3+4j, complex(1, np.nan)], axis='frequency')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert getattr(M, name)(sig, **params) == expected
    assert not caught
    full_params = {'x': 2} if name == 'value_at' else {}
    with pytest.warns(MetricWarning, match='NaN') as caught:
        assert np.isnan(getattr(M, name)(sig, **full_params))
    assert len(caught) == 1
    assert caught[0].message.metric == name
    assert caught[0].message.details['nan_sample_count'] == 1


def test_complex_final_value_preserves_nan_warning():
    sig = signal([3+4j, complex(1, np.nan)], axis='frequency')
    with pytest.warns(MetricWarning, match='NaN') as caught:
        assert np.isnan(M.final_value(sig))
    assert caught[0].message.metric == 'value_at'
