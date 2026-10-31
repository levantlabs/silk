"""Timing requests precede observations; unavailable measurements stay numeric."""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def signal(y, *, x=None):
    return Signal('clock', 'voltage', 'time', 'V',
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float),
                  np.asarray(y, dtype=float))


def clock():
    t = np.arange(65, dtype=float)
    return signal((t % 8 >= 4).astype(float), x=t)


@pytest.mark.parametrize('name', ['tie', 'long_term_jitter'])
@pytest.mark.parametrize('period', [0, -1])
@pytest.mark.parametrize('y', [[0], [0, 0, 0], [0, 1, 0], [0, np.nan, 1],
                             [0, 1, 0, 1, 0, 1]])
def test_invalid_period_precedes_unavailable_data(name, period, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError) as caught:
            getattr(M, name)(signal(y), ideal_period=period)
    message = str(caught.value)
    for fragment in (name + ':', 'ideal_period', repr(period), 'available',
                     'sample_count', str(len(y)), 'time_start_s', 'time_stop_s'):
        assert fragment in message


@pytest.mark.parametrize('taus', [[0], [-1], [8, 0], [np.nan], [np.inf], 1, [[1]]])
@pytest.mark.parametrize('y', [[0], [0, 0, 0], [0, np.nan, 1],
                             [0, 1, 0, 1, 0, 1, 0, 1]])
def test_invalid_taus_precede_unavailable_data(taus, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError) as caught:
            M.allan_deviation(signal(y), taus=taus)
    message = str(caught.value)
    for fragment in ('allan_deviation:', 'requested taus=', '> 0 s', 'available',
                     'sample_count', str(len(y)), 'time_start_s', 'time_stop_s'):
        assert fragment in message


@pytest.mark.parametrize('name,kwargs', [
    ('settling_time', {'initial': 0, 'target': 1}), ('tie', {}),
    ('long_term_jitter', {}), ('allan_deviation', {}),
])
@pytest.mark.parametrize('x', [[2, 1, 0], [0, 0, 1], [0, 2, 1], [0, np.nan, 2],
                             [0, 1, np.inf]])
@pytest.mark.parametrize('y', [[0, 0, 0], [0, np.nan, 1]])
def test_time_axis_is_a_request_error_before_event_search(name, kwargs, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError) as caught:
            getattr(M, name)(signal(y, x=x), **kwargs)
    message = str(caught.value)
    for fragment in (name + ':', 'x-axis', 'needs', 'increasing', 'index'):
        assert fragment in message


@pytest.mark.parametrize('name,kwargs', [
    ('settling_time', {'initial': 0, 'target': 1}), ('tie', {}),
    ('long_term_jitter', {}), ('allan_deviation', {}),
])
def test_wrong_axis_retains_invoking_function(name, kwargs):
    sig = Signal('response', 'voltage', 'frequency', 'V',
                 np.arange(3), np.array([0, np.nan, 1]))
    with pytest.raises(ValueError) as caught:
        getattr(M, name)(sig, **kwargs)
    assert str(caught.value).startswith(name + ':')
    assert "available xlabel='frequency', sample_count=3" in str(caught.value)


@pytest.mark.parametrize('y', [[1, 1, 1], [np.nan, 1, 1]])
def test_empty_settling_window_precedes_nan_data(y):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError) as caught:
            M.settling_time(signal(y), initial=0, target=1, start=100)
    message = str(caught.value)
    for fragment in ('settling_time:', 'start=100', '0 of 3', '[0.0, 2.0] s'):
        assert fragment in message


@pytest.mark.parametrize('start,expected', [(1, 1), (1.5, .5), (2, 0), (3, 0)])
def test_settling_ignores_nan_before_consumed_start(start, expected):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = M.settling_time(signal([np.nan, 0, 1, 1]), initial=0, target=1, start=start)
    assert result == expected


def test_settling_nan_diagnostic_describes_only_consumed_samples():
    with pytest.warns(MetricWarning) as caught:
        result = M.settling_time(signal([np.nan, 1, np.nan, 1]), initial=0, target=1, start=1)
    assert math.isnan(result)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == 'settling_time'
    assert diagnostic.code == 'nan_waveform'
    assert diagnostic.details['sample_count'] == 3
    assert diagnostic.details['nan_sample_count'] == 1
    assert diagnostic.details['time_start_s'] == 1
    assert diagnostic.details['nan_time_start_s'] == 2


def test_zero_jitter_and_settled_zero_are_valid_quiet_measurements():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        np.testing.assert_array_equal(M.tie(clock(), ideal_period=8).y, np.zeros(8))
        assert M.long_term_jitter(clock(), ideal_period=8) == 0
        for taus in (None, [8, 24]):
            result = M.allan_deviation(clock(), taus=taus)
            assert np.all(result.y == 0)
        assert M.settling_time(signal([0, 0, 0]), initial=1, target=0) == 0
        assert M.slew_rate(signal([0, 1, 2])) == pytest.approx(1)


def test_allan_long_tau_preserves_each_requested_point_and_explains_support():
    taus = np.array([8, 24, 32, 1e6])
    with pytest.warns(MetricWarning) as caught:
        result = M.allan_deviation(clock(), taus=taus)
    np.testing.assert_array_equal(result.x, taus)
    np.testing.assert_array_equal(result.y[:2], [0, 0])
    assert np.isnan(result.y[2:]).all()
    assert (result.name, result.type, result.xlabel, result.ylabel) == (
        'allan(clock)', 'allan_dev', 'tau', '')
    assert len(caught) == 2
    for i, item in enumerate(caught, start=2):
        d = item.message
        assert d.metric == 'allan_deviation'
        assert d.code == 'insufficient_tau_support'
        assert d.details['tau_index'] == i
        assert d.details['tau_s'] == taus[i]
        assert d.details['period_count'] == 7
        assert d.details['required_group_count'] == 2
        assert d.details['group_count'] < 2
        assert d.details['observed_duration_s'] == 56


def test_allan_rounding_policy_unchanged_but_x_labels_the_tau_used():
    edges = np.array([1, 9, 18, 26, 36, 44, 53, 61, 72], dtype=float)
    times = np.array([[e - .25, e + .25, e + .5] for e in edges]).ravel()
    sig = signal(np.tile([0, 1, 0], len(edges)), x=times)
    periods = np.diff(edges)
    mean = periods.mean()
    taus = mean * np.array([.01, .5, 1, 1.5, 2.5, 3])
    fractional = (periods - mean) / mean
    expected = []
    for m in (1, 1, 1, 2, 2, 3):
        n = len(periods) // m
        groups = fractional[:n * m].reshape(n, m).mean(axis=1)
        expected.append(np.sqrt(.5 * np.mean(np.diff(groups) ** 2)))
    with pytest.warns(MetricWarning) as caught:
        result = M.allan_deviation(sig, taus=taus)
    np.testing.assert_allclose(result.x, mean * np.array([1, 1, 1, 2, 2, 3]),
                               rtol=1e-14)
    np.testing.assert_allclose(result.y, expected, rtol=1e-14)
    rounded = [c.message for c in caught if c.message.code == 'tau_rounded']
    assert len(rounded) == 1
    d = rounded[0].details
    np.testing.assert_allclose(d['requested_tau_s'], (taus * [1, 1, 1, 1, 1, 1])[[0, 1, 3, 4]])
    np.testing.assert_allclose(d['used_tau_s'], mean * np.array([1, 1, 2, 2]))


@pytest.mark.parametrize('name', ['tie', 'long_term_jitter', 'allan_deviation'])
@pytest.mark.parametrize('y,code,count', [
    ([0, 0, 0], 'insufficient_edges', 0),
    ([0, np.nan, 1], 'nan_waveform', 1),
])
def test_unavailable_clock_results_preserve_shapes_and_have_one_warning(name, y, code, count):
    with pytest.warns(MetricWarning) as caught:
        result = getattr(M, name)(signal(y))
    assert len(caught) == 1
    assert caught[0].message.metric == name
    assert caught[0].message.code == code
    if name == 'long_term_jitter':
        assert type(result) is float and math.isnan(result)
    else:
        assert isinstance(result, Signal)
        assert result.x.shape == result.y.shape == (count,)
        if count:
            assert np.isnan(result.x).all() and np.isnan(result.y).all()


@pytest.mark.parametrize('x,y', [([0, 0], [0, 1]), ([0, 2, 1], [0, 0, 1])])
def test_slew_invalid_time_order_is_a_caller_error(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='slew_rate:.*axis.*needs.*increasing'):
            M.slew_rate(signal(y, x=x))


WARNING_CASES = [
    ('settling_time', signal([0, 0, 0]), {'initial': 0, 'target': 1}),
    ('settling_time', signal([0, np.nan, 1]), {'initial': 0, 'target': 1}),
    ('slew_rate', signal([0, 0, 0]), {}),
    ('tie', signal([0, 0, 0]), {}),
    ('long_term_jitter', signal([0, np.nan, 1]), {}),
    ('allan_deviation', signal([0, 0, 0]), {}),
    ('allan_deviation', clock(), {'taus': [1e6]}),
]


@pytest.mark.parametrize('name,sig,kwargs', WARNING_CASES)
@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_warning_policy(name, sig, kwargs, policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning) as exc:
                getattr(M, name)(sig, **kwargs)
            assert exc.value.metric == name
            return
        result = getattr(M, name)(sig, **kwargs)
    assert len(caught) == (1 if policy == 'always' else 0)
    if isinstance(result, Signal):
        assert result.y.size == 0 or np.isnan(result.y).all()
    else:
        assert math.isnan(result)
    if caught:
        d = caught[0].message
        assert d.metric == name
        assert d.suggestion
        assert all(type(v) in (str, float, int, type(None)) for v in d.details.values())


def test_scalar_nan_provenance_policy_remains_deferred():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert np.isnan(M.tie(clock(), ideal_period=np.nan).y).all()
        assert math.isnan(M.long_term_jitter(clock(), ideal_period=np.nan))


def _jittered_clock(offset=0.0, lo=0.0, hi=1.0):
    edges = np.array([1, 9, 18, 26, 36, 44, 53, 61, 72, 80, 89, 97], dtype=float)
    times = np.array([[e - .25, e + .25, e + .5, e + .75] for e in edges]).ravel()
    y = np.tile([lo, hi, hi, lo], len(edges)) + offset
    return signal(y, x=times)


def test_allan_whole_period_taus_are_labelled_exactly_and_do_not_warn():
    sig = _jittered_clock()
    mean = np.diff(M.crossings(sig)).mean()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = M.allan_deviation(sig, taus=[mean, 2 * mean])
    np.testing.assert_allclose(result.x, [mean, 2 * mean], rtol=1e-12)


def test_allan_window_threshold_and_edge_follow_crossings():
    sig = _jittered_clock()
    full = M.allan_deviation(sig, taus=[8.0])
    windowed = M.allan_deviation(sig, taus=[8.0], start=40.0)
    assert len(M.crossings(sig, start=40.0)) < len(M.crossings(sig))
    assert windowed.x[0] != full.x[0] or windowed.y[0] != full.y[0]
    # threshold: shifted waveform, shifted threshold, identical answer
    shifted = _jittered_clock(offset=3.0)
    a = M.allan_deviation(sig, taus=[8.0], threshold=0.5)
    b = M.allan_deviation(shifted, taus=[8.0], threshold=3.5)
    np.testing.assert_allclose(a.y, b.y)
    # falling edges of a 50% clock are a shifted copy: same periods, same answer
    f = M.allan_deviation(sig, taus=[8.0], edge='falling')
    assert np.isfinite(f.y).all()
    # 'any' keeps consecutive periods (first-direction chain)
    anyr = M.allan_deviation(sig, taus=[8.0], edge='any')
    np.testing.assert_allclose(anyr.y, full.y)


def test_allan_default_ladder_does_not_warn_on_jittery_clock():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        M.allan_deviation(_jittered_clock())


@pytest.mark.parametrize('kwargs', [
    {'edge': 'sideways'}, {'start': 5.0, 'end': 1.0}, {'start': 1e9},
])
def test_allan_bad_edge_or_window_raises(kwargs):
    with pytest.raises(ValueError, match='allan_deviation'):
        M.allan_deviation(_jittered_clock(), **kwargs)


def test_allan_window_with_too_few_edges_returns_empty_with_warning():
    with pytest.warns(MetricWarning) as caught:
        result = M.allan_deviation(_jittered_clock(), start=40.0, end=60.0)
    assert len(result.x) == 0 and len(result.y) == 0
    assert caught[0].message.code == 'insufficient_edges'


@pytest.mark.parametrize('varying', [False, True])
@pytest.mark.parametrize('scale,tau', [(1.0, 1e20), (1e-12, 1e308)])
def test_allan_extreme_taus_are_unavailable_in_request_order(varying, scale, tau):
    sig = _jittered_clock() if varying else clock()
    sig.x[:] *= scale
    before_x, before_y = sig.x.copy(), sig.y.copy()
    # Fixture rising edges: 1,9,...,97 (11 periods, mean 96/11),
    # or 3.5,11.5,...,59.5 (7 periods, mean 8).
    mean = (96 / 11 if varying else 8.0) * scale
    count = 11 if varying else 7
    taus = np.array([tau, mean, tau, 2 * mean])
    before_taus = taus.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.allan_deviation(sig, taus=taus)
    assert np.isnan(result.y[[0, 2]]).all()
    assert np.isfinite(result.y[[1, 3]]).all()
    assert result.x.dtype.kind == result.y.dtype.kind == 'f'
    assert (result.name, result.xlabel, result.type, result.ylabel) == (
        'allan(clock)', 'tau', 'allan_dev', '')
    np.testing.assert_array_equal(sig.x, before_x)
    np.testing.assert_array_equal(sig.y, before_y)
    np.testing.assert_array_equal(taus, before_taus)
    assert len(caught) == 2
    for index, warning in zip([0, 2], caught):
        d = warning.message
        assert isinstance(d, MetricWarning)
        assert (d.metric, d.code) == ('allan_deviation', 'insufficient_tau_support')
        assert d.details['tau_index'] == index
        assert d.details['tau_s'] == tau
        assert d.details['period_count'] == count
        assert d.details['mean_period_s'] == pytest.approx(mean)
        assert d.details['observed_duration_s'] == pytest.approx(count * mean)
        assert d.details['group_count'] == 0
        assert d.details['required_group_count'] == 2
        assert d.suggestion
        m = d.details['periods_per_group']
        if scale == 1.0:
            assert type(m) is int and m > 2**63
            assert m == int(round(tau / d.details['mean_period_s']))
            assert result.x[index] == m * d.details['mean_period_s']
        else:
            assert m == float('inf')
            assert result.x[index] == tau


def test_allan_support_boundary_and_rounding_numerics():
    # Seven equal periods: m=3 has two groups, m=4 only one.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.allan_deviation(clock(), taus=[24, 32, 4, 12, 20])
    np.testing.assert_array_equal(result.x, [24, 32, 8, 16, 16])
    np.testing.assert_array_equal(result.y, [0, np.nan, 0, 0, 0])
    assert [w.message.code for w in caught] == ['insufficient_tau_support', 'tau_rounded']
    assert caught[0].message.details['group_count'] == 1
    assert caught[1].message.details['tau_indices'] == [2, 3, 4]
    # Hand-calculated non-overlapping fractional period differences.
    # Periods are [8,9,8,10,8,9,8,11,8,9,8], mean 96/11.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = M.allan_deviation(_jittered_clock())
    np.testing.assert_allclose(result.x, [96 / 11, 192 / 11])
    np.testing.assert_allclose(result.y, [np.sqrt(8 / 5) / (96 / 11),
                                         np.sqrt(5 / 16) / (96 / 11)])


@pytest.mark.parametrize('policy', ['ignore', 'error'])
def test_allan_extreme_support_warning_filter(policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('error', RuntimeWarning)
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning, match='insufficient_tau_support'):
                M.allan_deviation(clock(), taus=[1e20])
        else:
            assert np.isnan(M.allan_deviation(clock(), taus=[1e20]).y).all()
    assert not caught


@pytest.mark.parametrize('runtime_policy', ['always', 'error'])
def test_allan_finite_group_size_with_overflowing_coordinate(runtime_policy):
    sig = clock()
    sig.x[:] *= 3 / 16  # Seven periods with mean exactly 1.5 s.
    tau = np.finfo(float).max
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.simplefilter(runtime_policy, RuntimeWarning)
        result = M.allan_deviation(sig, taus=[tau])
    np.testing.assert_array_equal(result.x, [tau])
    assert np.isnan(result.y).all()
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert isinstance(diagnostic, MetricWarning)
    assert (diagnostic.metric, diagnostic.code) == (
        'allan_deviation', 'insufficient_tau_support')
    assert diagnostic.details['tau_index'] == 0
    assert diagnostic.details['tau_s'] == tau
    assert diagnostic.details['mean_period_s'] == 1.5
    assert diagnostic.details['period_count'] == 7
    assert diagnostic.details['observed_duration_s'] == 10.5
    assert diagnostic.details['group_count'] == 0
    assert diagnostic.details['required_group_count'] == 2
    m = diagnostic.details['periods_per_group']
    assert type(m) is int
    assert m == int(round(tau / 1.5))
    assert diagnostic.suggestion
