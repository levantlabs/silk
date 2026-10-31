"""Public caller context without changes to transient measurement policy."""
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def signal(y=(0, 1, 0), *, x=None, axis='time', quantity='voltage'):
    return Signal('trace', quantity, axis, 'I' if quantity == 'current' else 'V',
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float),
                  np.asarray(y))


def error(fn, *args, **kwargs):
    with pytest.raises(ValueError) as caught:
        fn(*args, **kwargs)
    assert type(caught.value) is ValueError
    return str(caught.value)


AXIS_CASES = [
    ('rise_time', 1, {}), ('fall_time', 1, {}),
    ('slew_rate', 1, {}), ('settling_time', 1, {'initial': 0, 'target': 1}),
    ('propagation_delay', 2, {}), ('propagation_delays', 2, {}),
    ('overshoot', 1, {'initial': 0, 'target': 1}),
    ('undershoot', 1, {'initial': 0, 'target': 1}), ('average_duty', 1, {}),
    ('average_period', 1, {}), ('pulse_width', 1, {}),
    ('period_jitter', 1, {}), ('cycle_to_cycle_jitter', 1, {}),
    ('average_power', 2, {}), ('dynamic_energy_per_cycle', 2, {'period': 1}),
    ('hysteresis', 4, {}),
    ('on_time', 1, {'threshold': .5}), ('off_time', 1, {'threshold': .5}),
    ('cycle_frequency', 1, {}), ('eye_diagram', 1, {'period': 1}),
    ('tie', 1, {}), ('long_term_jitter', 1, {}), ('allan_deviation', 1, {}),
    ('sndr', 1, {'tclk': 1}), ('sinad', 1, {'tclk': 1}),
    ('enob', 1, {'tclk': 1}), ('fft', 1, {'tclk': 1}),
    ('thd', 1, {'tclk': 1}), ('thd_percent', 1, {'tclk': 1}),
]


@pytest.mark.parametrize('name,count,kwargs', AXIS_CASES, ids=[c[0] for c in AXIS_CASES])
def test_wrong_axis_names_public_metric_and_available_coordinates(name, count, kwargs):
    bad = signal(axis='wrong')
    message = error(getattr(M, name), *([bad] * count), **kwargs)
    assert message.startswith(name + ':')
    for part in ("xlabel='time'", "xlabel='wrong'", 'sample_count=3',
                 'trace spans [0.0, 2.0]'):
        assert part in message
    if name in ('slew_rate', 'period_jitter', 'cycle_to_cycle_jitter', 'sinad', 'enob'):
        assert not any(inner + ':' in message for inner in ('rise_time', 'crossings', 'sndr'))


@pytest.mark.parametrize('name', ['average', 'rms', 'stddev', 'peak', 'peak_to_peak',
                                  'minimum', 'maximum', 'argmin_x',
                                  'argmax_x', 'integral'])
@pytest.mark.parametrize('axis', ['time', 'frequency', 'voltage'])
@pytest.mark.parametrize('start,stop,count', [(100, 110, 0), (.5, 1.5, 1)])
def test_window_error_reports_requested_and_available_support(name, axis, start, stop, count):
    message = error(getattr(M, name), signal(axis=axis), start=start, end=stop,
                    interpolate=False)
    for part in (name + ':', f'[{start:g}, {stop:g}]', f'on {axis} axis',
                 f'contains {count} samples', 'trace spans [0.0, 2.0]',
                 'sample_count=3', 'need at least 2 samples'):
        assert part in message
    assert '] s' not in message


@pytest.mark.parametrize('name', ['rise_time', 'fall_time', 'slew_rate'])
@pytest.mark.parametrize('y', [(0, 1, 0), (0, 0, 0), (0, np.nan, 1)])
def test_swing_fraction_error_precedes_data_and_retains_caller(name, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        message = error(getattr(M, name), signal(y), low=.9, high=.1)
    assert message.startswith(name + ':')
    assert '0 <= low < high <= 1' in message
    assert 'low=0.9, high=0.1' in message


@pytest.mark.parametrize('name', ['crossings', 'period_jitter', 'cycle_to_cycle_jitter', 'crossing'])
@pytest.mark.parametrize('y', [(0, 0, 0), (0, np.nan, 1)])
def test_edge_choice_error_precedes_data_and_retains_caller(name, y):
    kwargs = {'threshold': .5, 'n': 0} if name == 'crossing' else {}
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        message = error(getattr(M, name), signal(y), edge='bad', **kwargs)
    assert message.startswith(name + ': edge')
    assert "'rising'|'falling'|'any', got 'bad'" in message


@pytest.mark.parametrize('name', ['convolve', 'cross_correlation'])
@pytest.mark.parametrize('counts', [(1, 1), (1, 3), (3, 1)])
def test_pair_sample_counts_name_both_inputs(name, counts):
    a, b = (signal(np.ones(n), axis='frequency') for n in counts)
    message = error(getattr(M, name), a, b)
    assert message.startswith(name + ':')
    assert 'at least 2 samples each' in message
    for role, n in zip(('a', 'b'), counts):
        assert f"{role}: xlabel='frequency', sample_count={n}" in message
        assert f'trace spans [0.0, {float(n - 1)}]' in message


@pytest.mark.parametrize('kwargs,parts', [
    ({}, ("exactly one of 'x_new' or 'n'", 'n=None', 'x_new=not provided')),
    ({'n': 3, 'x_new': [0, 1]}, ('exactly one', 'n=3', 'x_new=provided')),
    ({'n': 1}, ('n must be >= 2', 'got 1')),
    ({'x_new': []}, ('x_new must contain at least 1 point', 'got 0')),
])
def test_sample_request_errors_include_available_axis(kwargs, parts):
    message = error(M.sample, signal(axis='frequency'), **kwargs)
    for part in ('sample:', "xlabel='frequency'", 'trace spans [0.0, 2.0]',
                 'sample_count=3', *parts):
        assert part in message


@pytest.mark.parametrize('name,kwargs,parts', [
    ('crossing', {'threshold': .5, 'n': 0}, ('n must be a nonzero integer', 'got 0')),
    ('window', {'start': 2, 'end': 1}, ('end > start', 'start=2, end=1', 'trace spans')),
    ('average', {'start': 2, 'end': 1}, ('end > start', 'start=2.0, end=1.0', 'trace spans')),
    ('eye_diagram', {'period': 0}, ('period must be > 0', 'got 0')),
    ('eye_diagram', {'period': 2}, ('fewer than 2 periods', '1 complete periods available', 'trace spans')),
    ('eye_diagram', {'period': .5, 'n_traces': 0}, ('n_traces must be >= 1', 'got 0', '4 traces available')),
    ('eye_diagram', {'period': .5, 'n_traces': 5}, ('requested 5 traces', 'only 4 fit', 'trace spans')),
])
def test_local_parameter_errors_retain_request(name, kwargs, parts):
    message = error(getattr(M, name), signal(), **kwargs)
    assert message.startswith(name + ':')
    for part in parts:
        assert part in message


def test_error_norm_kind_names_metric_before_contamination():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        message = error(M.error_norm, signal([np.nan] * 3), signal(), kind='bad')
    assert message == "error_norm: kind must be 'max'|'rms'|'mean_abs', got 'bad'"


@pytest.mark.parametrize('name,kwargs', [('average_power', {}), ('dynamic_energy_per_cycle', {'period': 1})])
def test_power_axis_and_count_errors_retain_both_inputs(name, kwargs):
    fn = getattr(M, name)
    message = error(fn, signal(), signal(x=[1, 2, 3], quantity='current'), **kwargs)
    for part in (name + ':', 'same time axis', 'sig_v:', 'sig_i:',
                 'trace spans [0.0, 2.0]', 'trace spans [1.0, 3.0]'):
        assert part in message
    message = error(fn, signal([1]), signal([1], quantity='current'), **kwargs)
    for part in (name + ':', 'need at least 2 samples, got 1', 'trace spans [0.0, 0.0]'):
        assert part in message


def test_power_parameter_precedence_is_unchanged():
    bad = signal([], axis='wrong')
    assert error(M.dynamic_energy_per_cycle, bad, bad, period=0).startswith(
        'dynamic_energy_per_cycle: period must be > 0')
    assert 'same time axis' in error(M.average_power, signal(), signal(x=[1, 2, 3]))
    assert 'sig_i must be a current Signal' in error(M.average_power, signal(), signal())
    assert 'axis goes down' in error(M.average_power, signal(x=[2, 1, 0]),
                                signal(x=[2, 1, 0], quantity='current'))


@pytest.mark.parametrize('direction', ['rising', 'falling'])
def test_hysteresis_pair_error_names_direction_and_available_axes(direction):
    args = [signal()] * 4
    args[1 if direction == 'rising' else 3] = signal(x=[1, 2, 3])
    message = error(M.hysteresis, *args)
    assert message.startswith(f'hysteresis: {direction}-sweep input and output')
    assert 'trace spans [0.0, 2.0]' in message
    assert 'trace spans [1.0, 3.0]' in message


@pytest.mark.parametrize('name,kwargs', [
    ('slew_rate', {'low': .9, 'high': .1}),
    ('period_jitter', {'edge': 'bad'}), ('cycle_to_cycle_jitter', {'edge': 'bad'}),
    ('sinad', {'tclk': 0}), ('enob', {'tclk': 0}),
])
def test_delegated_empty_then_axis_then_parameter_precedence(name, kwargs):
    fn = getattr(M, name)
    assert 'has no samples' in error(fn, signal([], axis='wrong'), **kwargs)
    message = error(fn, signal(axis='wrong'), **kwargs)
    assert message.startswith(name + ':') and "xlabel='time'" in message


def test_direct_axis_precedes_empty_and_window_empty_precedes_bounds():
    assert "xlabel='time'" in error(M.rise_time, signal([], axis='wrong'), low=2)
    assert 'has no samples' in error(M.average, signal([]), start=2, end=1)
    assert 'end > start' in error(M.average, signal([np.nan]), start=2, end=1)


@pytest.mark.parametrize('name,slot,kwargs', [
    ('slew_rate', 'sig', {'low': .9, 'high': .1}),
    ('period_jitter', 'clock', {'edge': 'bad'}),
    ('cycle_to_cycle_jitter', 'clock', {'edge': 'bad'}),
    ('average', 'sig', {'start': 100, 'end': 110}),
    ('sinad', 'sig', {'tclk': 1}), ('enob', 'sig', {'tclk': 1}),
])
def test_function_form_binding_keeps_body_value_error(basic_module, name, slot, kwargs):
    tb = basic_module.create_test('validator_context')
    sig = signal(axis='wrong') if name in ('sinad', 'enob') else signal()
    tb.add_metric('measured_value', getattr(M, name), **{slot: sig}, **kwargs)
    message = error(tb._metrics[-1]['extractor'], tb.netlist)
    assert message.startswith(name + ':')


def test_finite_results_clipping_and_complex_shapes_are_unchanged():
    ramp = signal([0, 1, 2], axis='frequency')
    assert M.average(ramp, start=-10, end=1.5, interpolate=False) == .5
    assert M.integral(ramp, start=-10, end=1.5, interpolate=False) == .5
    # TASK-378: the edge at 1.5 is interpolated (y=1.5), never extrapolated at -10
    assert M.average(ramp, start=-10, end=1.5) == .75
    assert M.integral(ramp, start=-10, end=1.5) == 1.125
    empty = M.window(ramp, 100, 110)
    assert isinstance(empty, Signal) and empty.x.size == empty.y.size == 0
    original = signal([1 + 2j, 3 + 4j], x=[0, 2], axis='frequency')
    resampled = M.sample(original, n=3)
    np.testing.assert_array_equal(resampled.x, [0, 1, 2])
    np.testing.assert_array_equal(resampled.y, [1 + 2j, 2 + 3j, 3 + 4j])
    assert resampled.xlabel == 'frequency'
    assert M.rise_time(signal([0, 1, 2])) == pytest.approx(1.6)
    assert M.slew_rate(signal([0, 1, 2])) == 1
    clock = signal([0, 1] * 5)
    assert M.period_jitter(clock) == M.cycle_to_cycle_jitter(clock) == 0
    sndr = M.sndr(clock, tclk=1, n_samples=8)
    assert M.sinad(clock, tclk=1, n_samples=8) == sndr
    assert M.enob(clock, tclk=1, n_samples=8) == (sndr - 1.76) / 6.02


def test_existing_unavailable_results_and_warning_policy_are_unchanged():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        assert np.isnan(M.slew_rate(signal([0, 0, 0])))
        assert np.isnan(M.period_jitter(signal([0, 0, 0])))
        assert np.isnan(M.cycle_to_cycle_jitter(signal([0, 0, 0])))
        assert np.isnan(M.enob(signal([np.nan] * 8), tclk=1, n_samples=8))
    with pytest.warns(MetricWarning, match='slew_rate'):
        assert np.isnan(M.slew_rate(signal([0, 0, 0])))
