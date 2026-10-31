"""TASK-365: rise_time, fall_time and slew_rate share one edge core.

Owner rulings 2026-10-01: (1) the caller may state the settled
``initial``/``final`` levels the 10/90 points are based on (blank = the
observed min/max, the pre-TASK-365 result); (2) any edge, not only the
first: ``n=k`` the k-th, ``n=-k`` the k-th from the end, inside a
``start``/``end`` window on the TASK-364 crossings rules; (3) slew_rate
takes the same parameters (plus ``edge``, so it measures falling edges
too) and is a different output of the same edge.
"""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal
from silk.signals.schema import export_metric_schema

NAN = float('nan')


def tran(t, y):
    return Signal('v', 'voltage', 'time', 'V', np.asarray(t, dtype=float),
                  np.asarray(y, dtype=float))


# ---------------------------------------------------------------------------
# Default calls preserve the pre-change numbers within platform rounding
# ---------------------------------------------------------------------------

def _cases():
    """The traces the pre-change outputs below were captured from."""
    t = np.linspace(0, 10e-9, 201)
    out = {}
    out['exp_step'] = (t, 1.8 * (1 - np.exp(-t / 1e-9)))
    out['exp_decay'] = (t, 1.8 * np.exp(-t / 1e-9))
    out['ringing_step'] = (t, 1.0 - np.exp(-t / 0.8e-9) * np.cos(2 * np.pi * t / 2e-9))
    out['ringing_fall'] = (t, np.exp(-t / 0.8e-9) * np.cos(2 * np.pi * t / 2e-9))
    out['clock'] = (t, 0.9 + 0.9 * np.tanh(np.sin(2 * np.pi * t / 2.5e-9) * 4))
    out['ramp'] = (np.array([0., 1., 2., 3.]), np.array([0., 0., 1., 1.]))
    out['one_interval'] = (np.array([0, 1e-9, 2e-9, 3e-9]), np.array([0, 0, 1, 1.]))
    out['glitchy'] = (np.arange(10.), np.array([0, .2, .05, .3, .95, .5, 1, .02, .9, 1.]))
    rng = np.random.default_rng(365)
    out['noisy_pulse'] = (t, np.where((t > 2e-9) & (t < 6e-9), 1.0, 0.0)
                          + 0.03 * rng.standard_normal(len(t)))
    return out


#: Captured from the pre-TASK-365 rise_time/fall_time/slew_rate (branch
#: base 12fa56f6) as ``repr`` floats: ``name|metric|low|high``.
PINNED = {
    'exp_step|rise_time|0.1|0.9': 2.1967532203820033e-09,
    'exp_step|fall_time|0.1|0.9': NAN,
    'exp_step|slew_rate|0.1|0.9': 655483106.0408075,
    'exp_step|rise_time|0.2|0.8': 1.3860027467888472e-09,
    'exp_step|fall_time|0.2|0.8': NAN,
    'exp_step|slew_rate|0.2|0.8': 779183858.4576656,
    'exp_decay|rise_time|0.1|0.9': NAN,
    'exp_decay|fall_time|0.1|0.9': 2.196753220382003e-09,
    'exp_decay|slew_rate|0.1|0.9': NAN,
    'exp_decay|rise_time|0.2|0.8': NAN,
    'exp_decay|fall_time|0.2|0.8': 1.3860027467888468e-09,
    'exp_decay|slew_rate|0.2|0.8': NAN,
    'ringing_step|rise_time|0.1|0.9': 5.455995242709829e-10,
    'ringing_step|fall_time|0.1|0.9': NAN,
    'ringing_step|slew_rate|0.1|0.9': 1919008776.7927527,
    'ringing_step|rise_time|0.2|0.8': 3.7968717072732034e-10,
    'ringing_step|fall_time|0.2|0.8': NAN,
    'ringing_step|slew_rate|0.2|0.8': 2068170239.3664045,
    'ringing_fall|rise_time|0.1|0.9': NAN,
    'ringing_fall|fall_time|0.1|0.9': 5.455995242709828e-10,
    'ringing_fall|slew_rate|0.1|0.9': NAN,
    'ringing_fall|rise_time|0.2|0.8': NAN,
    'ringing_fall|fall_time|0.2|0.8': 3.7968717072732044e-10,
    'ringing_fall|slew_rate|0.2|0.8': NAN,
    'clock|rise_time|0.1|0.9': 2.284857091358068e-10,
    'clock|fall_time|0.1|0.9': 2.28485709135807e-10,
    'clock|slew_rate|0.1|0.9': 6298069296.796291,
    'clock|rise_time|0.2|0.8': 1.4605580957410823e-10,
    'clock|fall_time|0.2|0.8': 1.4605580957410782e-10,
    'clock|slew_rate|0.2|0.8': 7389395363.633923,
    'ramp|rise_time|0.1|0.9': 0.7999999999999998,
    'ramp|fall_time|0.1|0.9': NAN,
    'ramp|slew_rate|0.1|0.9': 1.0000000000000002,
    'ramp|rise_time|0.2|0.8': 0.6000000000000001,
    'ramp|fall_time|0.2|0.8': NAN,
    'ramp|slew_rate|0.2|0.8': 1.0,
    'one_interval|rise_time|0.1|0.9': 7.999999999999999e-10,
    'one_interval|fall_time|0.1|0.9': NAN,
    'one_interval|slew_rate|0.1|0.9': 1000000000.0000001,
    'one_interval|rise_time|0.2|0.8': 6.000000000000002e-10,
    'one_interval|fall_time|0.2|0.8': NAN,
    'one_interval|slew_rate|0.2|0.8': 999999999.9999998,
    'glitchy|rise_time|0.1|0.9': 3.4230769230769234,
    'glitchy|fall_time|0.1|0.9': 2.8072562358276647,
    'glitchy|slew_rate|0.1|0.9': 0.2337078651685393,
    'glitchy|rise_time|0.2|0.8': 2.769230769230769,
    'glitchy|fall_time|0.2|0.8': 2.482993197278912,
    'glitchy|slew_rate|0.2|0.8': 0.2166666666666667,
    'noisy_pulse|rise_time|0.1|0.9': 1.9095573843279406e-09,
    'noisy_pulse|fall_time|0.1|0.9': 3.9429477123298376e-09,
    'noisy_pulse|slew_rate|0.1|0.9': 499538918.1075272,
    'noisy_pulse|rise_time|0.2|0.8': 3.811973388756039e-11,
    'noisy_pulse|fall_time|0.2|0.8': 3.6114110811988716e-11,
    'noisy_pulse|slew_rate|0.2|0.8': 18767803429.15847,
}


@pytest.mark.parametrize('key', sorted(PINNED))
def test_default_call_preserves_the_pre_change_number(key):
    name, fn, low, high = key.split('|')
    x, y = _cases()[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        got = getattr(M, fn)(tran(x, y), float(low), float(high))
    want = PINNED[key]
    # rel=1e-12 allows last-bit platform rounding (CI run 37536524664).
    # abs=0 is essential: nanosecond results must not use an absolute floor.
    assert (math.isnan(got) and math.isnan(want)) or got == pytest.approx(
        want, rel=1e-12, abs=0)


# ---------------------------------------------------------------------------
# Stated levels: the T3 ringing edge (TASK-295)
# ---------------------------------------------------------------------------

TAU, PERIOD = 0.8e-9, 2e-9


def _ring(t):
    return 1.0 - np.exp(-t / TAU) * np.cos(2 * np.pi * t / PERIOD)


def _root(f, level, a, b):
    """Bisection on a function that rises through ``level`` in [a, b]."""
    for _ in range(200):
        m = 0.5 * (a + b)
        a, b = (m, b) if f(m) < level else (a, m)
    return 0.5 * (a + b)


@pytest.fixture
def ringing():
    t = np.linspace(0, 10e-9, 10001)        # dt = 1 ps
    return tran(t, _ring(t))


def test_stated_levels_put_the_10_90_points_on_the_settled_values(ringing):
    """0 -> 1 step that overshoots to ~1.29: stated 0/1 measures the true
    0.1 -> 0.9 interval; the observed extremes measure 0.129 -> 1.158."""
    # Both crossings lie before the first peak (omega*t < 3*pi/4).
    t10 = _root(_ring, 0.1, 0.0, 0.75e-9)
    t90 = _root(_ring, 0.9, 0.0, 0.75e-9)
    stated = M.rise_time(ringing, initial=0.0, final=1.0)
    assert stated == pytest.approx(t90 - t10, abs=1e-15)
    # Only the final level needs stating here: the observed min IS 0.
    assert M.rise_time(ringing, final=1.0) == stated
    peak = float(np.max(ringing.y))
    assert peak > 1.25
    o10 = _root(_ring, 0.1 * peak, 0.0, 0.75e-9)
    o90 = _root(_ring, 0.9 * peak, 0.0, 0.75e-9)
    observed = M.rise_time(ringing)
    assert observed == pytest.approx(o90 - o10, abs=1e-15)
    assert abs(observed - stated) > 20e-12   # the T3 bias, not resolution


def test_slew_rate_uses_the_stated_swing(ringing):
    stated = M.rise_time(ringing, initial=0.0, final=1.0)
    assert M.slew_rate(ringing, initial=0.0, final=1.0) == pytest.approx(
        0.8 * 1.0 / stated, rel=1e-12)


def test_falling_ringing_edge_with_stated_levels():
    t = np.linspace(0, 10e-9, 10001)
    sig = tran(t, 1.0 - _ring(t))          # 1 -> 0, undershoots to ~-0.29
    t90 = _root(_ring, 0.1, 0.0, 0.75e-9)  # y = 0.9 where _ring = 0.1
    t10 = _root(_ring, 0.9, 0.0, 0.75e-9)
    assert M.fall_time(sig, initial=1.0, final=0.0) == pytest.approx(
        t10 - t90, abs=1e-15)
    assert M.slew_rate(sig, edge='falling', initial=1.0, final=0.0) == \
        pytest.approx(0.8 / M.fall_time(sig, initial=1.0, final=0.0),
                      rel=1e-15)


def test_ringing_about_a_threshold_does_not_split_the_edge():
    """Lightly damped: the ring dips back through 0.9 and climbs again,
    but never back to 0.1 — one edge, not several."""
    t = np.linspace(0, 10e-9, 10001)
    sig = tran(t, 1.0 - np.exp(-t / 2e-9) * np.cos(2 * np.pi * t / PERIOD))
    assert len(M.crossings(sig, threshold=0.9, edge='rising')) > 1
    assert M.rise_time(sig, initial=0.0, final=1.0) == M.rise_time(
        sig, initial=0.0, final=1.0, n=-1)
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.rise_time(sig, initial=0.0, final=1.0, n=2))
    d = caught[0].message
    assert d.code == 'insufficient_edges'
    assert d.details['edge_count'] == 1 and d.details['requested_edge'] == 2


@pytest.mark.parametrize('name,kwargs,role', [
    # 0 -> 0.85: the high point 0.9 * 2.0 = 1.8 is never reached.
    ('rise_time', {'final': 2.0}, 'high'),
    # The trace starts at 0, above the low point -1 + 0.1 * 2 = -0.8.
    ('rise_time', {'initial': -1.0, 'final': 1.0}, 'low'),
    # 1 -> 0: the low point -2 + 0.1 * 3 = -1.7 is never reached.
    ('fall_time', {'initial': 1.0, 'final': -2.0}, 'low'),
])
def test_stated_level_never_crossed_is_nan_with_a_warning(name, kwargs, role):
    sig = tran([0., 1., 2., 3.], [0., 0., 0.85, 0.85]) if name == 'rise_time' \
        else tran([0., 1., 2., 3.], [1., 1., 0., 0.])
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(M, name)(sig, **kwargs))
    assert len(caught) == 1
    d = caught[0].message
    assert d.metric == name and d.code == 'missing_crossing'
    assert d.details['threshold_role'] == role
    for level in ('initial', 'final'):
        source = 'stated' if level in kwargs else 'observed'
        assert d.details[f'{level}_source'] == source
        if level in kwargs:
            assert d.details[level] == kwargs[level]


def test_one_stated_level_that_leaves_no_swing_is_nan_with_a_warning():
    sig = tran([0., 1., 2., 3.], [0., 0., 1., 1.])
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.rise_time(sig, initial=1.5))   # final = max = 1
    d = caught[0].message
    assert d.code == 'zero_swing'
    assert d.details['initial_source'] == 'stated'
    assert d.details['final_source'] == 'observed'
    assert d.details['final'] == 1.0


# ---------------------------------------------------------------------------
# Edge occurrence and window
# ---------------------------------------------------------------------------

def pulses():
    """Piecewise-linear, swing 0..1, so every crossing is exact:
    rise 1 over [1, 2] (0.8), fall 1 over [3, 4] (0.8),
    rise 2 over [5, 7] (1.6), fall 2 over [8, 10] (1.6)."""
    return tran(np.arange(11.), [0, 0, 1, 1, 0, 0, .5, 1, 1, .5, 0])


@pytest.mark.parametrize('name,n,expected', [
    ('rise_time', 1, 0.8), ('rise_time', 2, 1.6),
    ('rise_time', -1, 1.6), ('rise_time', -2, 0.8),
    ('fall_time', 1, 0.8), ('fall_time', 2, 1.6),
    ('fall_time', -1, 1.6), ('fall_time', -2, 0.8),
])
def test_nth_and_from_the_end_edge(name, n, expected):
    assert getattr(M, name)(pulses(), n=n) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize('edge,n,expected', [
    ('rising', 1, 0.8 / 0.8), ('rising', -1, 0.8 / 1.6),
    ('falling', 1, 0.8 / 0.8), ('falling', 2, 0.8 / 1.6),
    # 'any' counts both directions in time order: r1, f1, r2, f2.
    ('any', 1, 1.0), ('any', 2, 1.0), ('any', 3, 0.5), ('any', -1, 0.5),
])
def test_slew_rate_edges_including_falling_and_any(edge, n, expected):
    got = M.slew_rate(pulses(), edge=edge, n=n)
    assert got > 0                                   # a magnitude
    assert got == pytest.approx(expected, rel=1e-12)


def test_slew_rate_is_the_same_edge_as_rise_and_fall_time():
    sig = pulses()
    for n in (1, 2, -1):
        assert M.slew_rate(sig, n=n) == pytest.approx(
            0.8 / M.rise_time(sig, n=n), rel=1e-15)
        assert M.slew_rate(sig, edge='falling', n=n) == pytest.approx(
            0.8 / M.fall_time(sig, n=n), rel=1e-15)


@pytest.mark.parametrize('name,kwargs,expected', [
    ('rise_time', {'start': 4.5}, 1.6),
    ('rise_time', {'end': 4.5, 'n': -1}, 0.8),
    ('fall_time', {'start': 4.5}, 1.6),
    ('fall_time', {'start': 2.5, 'end': 4.5}, 0.8),
    # Window bounds between samples; the edge's crossings must lie inside.
    ('rise_time', {'start': 5.1, 'end': 6.9, 'initial': 0, 'final': 1}, 1.6),
])
def test_window_selects_the_edge(name, kwargs, expected):
    assert getattr(M, name)(pulses(), **kwargs) == pytest.approx(expected,
                                                                  rel=1e-12)


def test_edge_cut_by_the_window_is_not_counted():
    # rise 2 crosses 0.1 at 5.2 and 0.9 at 6.8: end=6.5 cuts it. (With
    # observed levels the window's own 0..0.75 swing would be measured.)
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.rise_time(pulses(), start=4.5, end=6.5,
                                      initial=0.0, final=1.0))
    assert caught[0].message.code == 'missing_crossing'
    assert caught[0].message.details['window_end'] == 6.5


def test_observed_levels_come_from_inside_the_window():
    """0 -> 1 then 1 -> 3. The whole trace swings 0..3 (10 % = 0.3 at
    t = 1.3, 90 % = 2.7 at t = 6.85); inside [4.5, 10] it swings 1..3
    (1.2 at t = 6.1, 2.8 at t = 6.9)."""
    sig = tran(np.arange(11.), [0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3])
    assert M.rise_time(sig) == pytest.approx(5.55, rel=1e-12)
    assert M.rise_time(sig, start=4.5) == pytest.approx(0.8, rel=1e-12)
    assert M.rise_time(sig, initial=1.0, final=3.0) == pytest.approx(
        0.8, rel=1e-12)


def test_fewer_edges_than_n_is_nan_with_a_warning():
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.slew_rate(pulses(), edge='any', n=-5))
    d = caught[0].message
    assert d.metric == 'slew_rate' and d.code == 'insufficient_edges'
    assert d.details['edge_count'] == 4 and d.details['requested_edge'] == -5


# ---------------------------------------------------------------------------
# Caller errors (decision-1: raise)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['rise_time', 'fall_time', 'slew_rate'])
@pytest.mark.parametrize('kwargs,match', [
    ({'n': 0}, 'n must be a nonzero integer'),
    ({'n': None}, 'n must be a nonzero integer'),
    ({'n': True}, 'n must be a nonzero integer'),
    ({'n': 1.0}, 'n must be a nonzero integer'),
    ({'initial': float('nan')}, 'initial must be a finite number'),
    ({'final': 'high'}, 'final must be a finite number'),
    ({'initial': 0.5, 'final': 0.5}, 'zero-swing'),
    ({'start': 5.0, 'end': 4.0}, 'end >= start'),
    ({'start': 20.0}, 'does not overlap'),
])
def test_malformed_request_raises(name, kwargs, match):
    with pytest.raises(ValueError, match=match) as exc:
        getattr(M, name)(pulses(), **kwargs)
    assert str(exc.value).startswith(f'{name}:')


@pytest.mark.parametrize('name,kwargs', [
    ('rise_time', {'initial': 1.0, 'final': 0.0}),
    ('fall_time', {'initial': 0.0, 'final': 1.0}),
    ('slew_rate', {'initial': 1.0, 'final': 0.0}),
    ('slew_rate', {'edge': 'falling', 'initial': 0.0, 'final': 1.0}),
])
def test_stated_levels_in_the_wrong_order_raise(name, kwargs):
    with pytest.raises(ValueError, match=f'{name}: a (rising|falling) edge needs'):
        getattr(M, name)(pulses(), **kwargs)


def test_slew_rate_any_takes_the_stated_levels_in_either_order():
    sig = pulses()
    a = M.slew_rate(sig, edge='any', n=2, initial=0.0, final=1.0)
    b = M.slew_rate(sig, edge='any', n=2, initial=1.0, final=0.0)
    assert a == b == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize('kwargs', [{'initial': 0.0}, {'final': 1.0}])
def test_slew_rate_any_needs_both_levels_or_neither(kwargs):
    with pytest.raises(ValueError, match="slew_rate: edge='any' needs both"):
        M.slew_rate(pulses(), edge='any', **kwargs)


def test_slew_rate_rejects_a_bad_edge():
    with pytest.raises(ValueError, match="slew_rate: edge must be"):
        M.slew_rate(pulses(), edge='up')


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_the_three_metrics_export_the_same_edge_parameters():
    schema = export_metric_schema()
    names = {m: [p['name'] for p in schema[m]['parameters']]
             for m in ('rise_time', 'fall_time', 'slew_rate')}
    common = ['low', 'high', 'initial', 'final', 'n', 'start', 'end']
    assert names['rise_time'] == names['fall_time'] == common
    assert names['slew_rate'] == ['low', 'high', 'edge'] + common[2:]
    units = {p['name']: p['unit'] for p in schema['slew_rate']['parameters']}
    assert units['initial'] == units['final'] == '[y]'
    assert units['start'] == units['end'] == '[x]'
