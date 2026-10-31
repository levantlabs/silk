"""TASK-367: average_duty / cycle_duty on the TASK-366 per-cycle core;
pulse_width / pulse_widths with polarity, occurrence and window;
on_time / off_time with threshold and window.

The PINNED values below were captured from ``duty_cycle``, ``pulse_width``,
``on_time`` and ``off_time`` on the branch base 97f50a52, before they were
removed or changed: the defaults of the new forms must reproduce them
exactly.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals import Signal
from silk.signals.schema import export_metric_schema


def _sig(t, y, name='clk'):
    return Signal(name, 'voltage', 'time', 'V', np.asarray(t, float),
                  np.asarray(y, float))


def _pulses(rises, highs, stop, step=0.01, start_high=False):
    """Piecewise-linear 0/1 train: a pulse rises at each ``rises[k]`` and
    falls ``highs[k]`` later. Every edge crosses 0.5 exactly at its
    nominal time, so each crossing is analytic.
    """
    level = 1.0 if start_high else 0.0
    t, y = [0.0], [level]
    if start_high:
        # A leading high segment that falls at 0.1.
        t += [0.1 - step, 0.1 + step]
        y += [1.0, 0.0]
    for r, h in zip(rises, highs):
        for x, v in ((r - step, 0.0), (r + step, 1.0),
                     (r + h - step, 1.0), (r + h + step, 0.0)):
            t.append(x)
            y.append(v)
    t.append(stop)
    y.append(0.0)
    return _sig(t, y)


def _square(n=6, high=0.3, period=1.0):
    """Clock rising at k*period + 0.25, falling ``high`` later."""
    rises = [k * period + 0.25 for k in range(n)]
    return _pulses(rises, [high] * n, n * period + 0.5)


# --- pinned pre-change results (branch base 97f50a52) -----------------------

def _pin_signals():
    rng = np.random.default_rng(367)
    t = np.linspace(0, 10e-9, 2001)
    sq = (0.9 + 0.9 * np.sign(np.sin(2 * np.pi * t / 1.3e-9 + 0.4))
          + 0.05 * rng.standard_normal(t.size))
    sine = np.clip(0.9 + 1.4 * np.sin(2 * np.pi * t / 1.7e-9), 0, 1.8)
    return {'sq': _sig(t, sq, 'sq'), 'sin': _sig(t, sine, 'sin')}


# (duty_cycle, pulse_width, on_time, off_time); on/off use level 0.9 when
# the threshold is None (they had no default).
PINNED = {
    ('sq', None): (0.5000161499348147, 6.500153575159185e-10,
                   5.1175e-09, 4.8825e-09),
    ('sq', 0.6): (0.5012903475524852, 6.516823085240361e-10,
                  5.1175e-09, 4.8825e-09),
    ('sq', 1.2): (0.49873342075732624, 6.4833722644483e-10,
                  5.1175e-09, 4.8825e-09),
    ('sin', None): (0.5000000000000002, 8.499999999999998e-10,
                    5.0949999999999995e-09, 4.905000000000001e-09),
    ('sin', 0.6): (0.5687449006033742, 9.66866331025736e-10,
                   5.7325e-09, 4.2675e-09),
    ('sin', 1.2): (0.4312550993966258, 7.33133668974264e-10,
                   4.410000000000001e-09, 5.589999999999999e-09),
}


class TestPinnedAgainstOldFunctions:
    @pytest.mark.parametrize('key', list(PINNED), ids=str)
    def test_defaults_reproduce_old_results(self, key):
        name, threshold = key
        s = _pin_signals()[name]
        duty, width, on, off = PINNED[key]
        level = 0.9 if threshold is None else threshold
        assert M.average_duty(s, threshold) == duty
        assert M.pulse_width(s, threshold) == width
        assert M.on_time(s, level) == on
        assert M.off_time(s, level) == off

    def test_duty_cycle_removed(self):
        import silk.metrics.timing as T
        assert not hasattr(M, 'duty_cycle')
        assert not hasattr(T, 'duty_cycle')
        assert 'duty_cycle' not in M.__all__


# --- average_duty / cycle_duty ----------------------------------------------

class TestDuty:
    def test_square_duty_exact(self):
        assert M.average_duty(_square(high=0.3)) == pytest.approx(0.3, abs=1e-12)
        c = M.cycle_duty(_square(high=0.3))
        np.testing.assert_allclose(c.y, 0.3, atol=1e-12)
        assert c.xlabel == 'time' and c.ylabel == '' and c.name == 'cycle_duty(clk)'

    def test_cycle_duty_shares_cycle_period_times(self):
        s = _pulses([0.25, 1.25, 2.25, 3.25], [0.2, 0.4, 0.6, 0.1], 4.5)
        c = M.cycle_duty(s)
        p = M.cycle_period(s)
        np.testing.assert_array_equal(c.x, p.x)
        np.testing.assert_allclose(c.y, [0.2, 0.4, 0.6], atol=1e-12)

    def test_average_is_total_high_over_total_time(self):
        # Unequal periods: rises at 0.25, 1.25, 3.25 (periods 1 and 2).
        s = _pulses([0.25, 1.25, 3.25], [0.5, 0.5, 0.5], 4.0)
        c = M.cycle_duty(s)
        np.testing.assert_allclose(c.y, [0.5, 0.25], atol=1e-12)
        # Period-weighted: 1.0 high over 3.0 s, not mean(0.5, 0.25).
        assert M.average_duty(s) == pytest.approx(1.0 / 3.0, abs=1e-12)

    def test_falling_edge_cycles(self):
        s = _pulses([0.25, 1.25, 2.25, 3.25], [0.2, 0.4, 0.6, 0.1], 4.5)
        # Falling at 0.45, 1.65, 2.85, 3.35: cycles hold the NEXT pulse.
        c = M.cycle_duty(s, edge='falling')
        np.testing.assert_allclose(np.diff([0.45, 1.65, 2.85, 3.35]),
                                   M.cycle_period(s, edge='falling').y)
        np.testing.assert_allclose(c.y, [0.4 / 1.2, 0.6 / 1.2, 0.1 / 0.5],
                                   atol=1e-12)
        assert M.average_duty(s, edge='falling') == pytest.approx(
            1.1 / 2.9, abs=1e-12)

    def test_any_edge_one_cycle_per_crossing(self):
        s = _square(n=4, high=0.3)
        c = M.cycle_duty(s, edge='any')
        # 8 crossings -> 6 full cycles, each with duty 0.3.
        assert len(c.y) == 6
        np.testing.assert_allclose(c.y, 0.3, atol=1e-12)
        np.testing.assert_array_equal(c.x, M.cycle_period(s, edge='any').x)
        assert M.average_duty(s, edge='any') == pytest.approx(0.3, abs=1e-12)

    def test_windowed(self):
        s = _pulses([0.25, 1.25, 2.25, 3.25, 4.25],
                    [0.2, 0.4, 0.6, 0.8, 0.1], 5.5)
        # Window keeps rises 1.25, 2.25, 3.25 -> cycles 2 and 3 only.
        assert M.average_duty(s, start=1.0, end=3.5) == pytest.approx(
            0.5, abs=1e-12)
        np.testing.assert_allclose(M.cycle_duty(s, start=1.0, end=3.5).y,
                                   [0.4, 0.6], atol=1e-12)
        assert M.average_duty(s, start=2.0) == pytest.approx(
            (0.6 + 0.8) / 2, abs=1e-12)
        assert M.average_duty(s, end=2.3) == pytest.approx(0.3, abs=1e-12)

    def test_threshold_moves_duty(self):
        # Triangle wave 0..1, period 1: time above th is (1 - th).
        t = np.linspace(0, 5, 5001)
        y = 1 - np.abs(((t + 0.5) % 1.0) - 0.5) * 2
        s = _sig(t, y)
        for th in (0.2, 0.5, 0.7):
            assert M.average_duty(s, th) == pytest.approx(1 - th, abs=1e-9)

    def test_window_threshold_is_swing_inside_window(self):
        # Bigger pulses early, 0..0.4 pulses late: midpoint inside the late
        # window is 0.2, which the early midpoint (0.5) would never cross.
        t = [0, 0.24, 0.26, 0.54, 0.56, 1.24, 1.26, 1.54, 1.56,
             2.24, 2.26, 2.74, 2.76, 3.24, 3.26, 3.74, 3.76, 4.24, 4.26, 4.5]
        y = [0, 0, 1, 1, 0, 0, 1, 1, 0,
             0, .4, .4, 0, 0, .4, .4, 0, 0, .4, .4]
        s = _sig(t, y)
        assert M.average_duty(s, start=2.0) == pytest.approx(0.5, abs=1e-12)

    @pytest.mark.parametrize('kwargs,count', [({}, 1), ({'edge': 'any'}, 2)])
    def test_too_few_edges_nan_and_warning(self, kwargs, count):
        s = _pulses([0.25], [0.3], 1.0)
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(M.average_duty(s, **kwargs))
        d = caught[0].message
        assert d.code == 'insufficient_crossings'
        assert d.metric == 'average_duty'
        assert d.details['crossing_count'] == count
        with pytest.warns(MetricWarning) as caught:
            c = M.cycle_duty(s, **kwargs)
        assert len(c.y) == 0
        assert caught[0].message.metric == 'cycle_duty'

    def test_partial_record_no_fallback(self):
        # duty_cycle used to fall back to the whole-trace time-above
        # fraction here; the full-cycle duty is unknown, on_time answers.
        s = _sig([0, 1, 2, 3], [0, 0, 1, 1])
        with pytest.warns(MetricWarning):
            assert math.isnan(M.average_duty(s))
        assert M.on_time(s) == pytest.approx(1.5)

    def test_flat_trace(self):
        s = _sig([0, 1, 2], [1, 1, 1])
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(M.average_duty(s))
        assert caught[0].message.code == 'insufficient_crossings'

    @pytest.mark.parametrize('fn', ['average_duty', 'cycle_duty'])
    def test_nan_in_consumed_data(self, fn):
        s = _square(n=4)
        y = s.y.copy()
        y[5] = np.nan
        s = _sig(s.x, y)
        with pytest.warns(MetricWarning) as caught:
            out = getattr(M, fn)(s)
        assert caught[0].message.code == 'nan_waveform'
        if fn == 'average_duty':
            assert math.isnan(out)
        else:
            assert len(out.y) == 1 and math.isnan(out.y[0])

    def test_nan_outside_window_ignored(self):
        s = _square(n=5)
        y = s.y.copy()
        y[1] = np.nan
        s = _sig(s.x, y)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert M.average_duty(s, start=1.0) == pytest.approx(0.3, abs=1e-12)

    @pytest.mark.parametrize('fn', ['average_duty', 'cycle_duty'])
    def test_caller_errors_raise(self, fn):
        s = _square()
        with pytest.raises(ValueError, match=fn):
            getattr(M, fn)(s, edge='up')
        with pytest.raises(ValueError, match=fn):
            getattr(M, fn)(s, start=3, end=2)
        with pytest.raises(ValueError, match=fn):
            getattr(M, fn)(s, start=100)
        f = Signal('f', 'voltage', 'frequency', 'V', np.arange(4.), np.arange(4.))
        with pytest.raises(ValueError, match=fn):
            getattr(M, fn)(f)


# --- pulse_width / pulse_widths ---------------------------------------------

def _train():
    """High pulses 0.2, 0.4, 0.6, 0.1 rising at 0.25, 1.25, 2.25, 3.25.

    Low pulses between them: 0.8, 0.6, 0.4.
    """
    return _pulses([0.25, 1.25, 2.25, 3.25], [0.2, 0.4, 0.6, 0.1], 4.5)


class TestPulseWidth:
    def test_positive_pulses(self):
        s = _train()
        np.testing.assert_allclose(M.pulse_widths(s), [0.2, 0.4, 0.6, 0.1],
                                   atol=1e-12)
        assert M.pulse_width(s) == pytest.approx(0.2, abs=1e-12)

    def test_negative_pulses(self):
        s = _train()
        np.testing.assert_allclose(M.pulse_widths(s, polarity='negative'),
                                   [0.8, 0.6, 0.4], atol=1e-12)
        assert M.pulse_width(s, polarity='negative') == pytest.approx(
            0.8, abs=1e-12)

    def test_negative_pulse_at_record_start_is_cut(self):
        # The record starts high: the first falling crossing at 0.1 opens
        # a low pulse closed by the rise at 0.25.
        s = _pulses([0.25, 1.25], [0.3, 0.3], 2.0, start_high=True)
        np.testing.assert_allclose(M.pulse_widths(s, polarity='negative'),
                                   [0.15, 0.7], atol=1e-12)
        # The leading high segment never rose inside the record: not a pulse.
        np.testing.assert_allclose(M.pulse_widths(s), [0.3, 0.3], atol=1e-12)

    @pytest.mark.parametrize('n,expected', [(1, 0.2), (2, 0.4), (4, 0.1),
                                            (-1, 0.1), (-2, 0.6), (-4, 0.2)])
    def test_occurrence(self, n, expected):
        assert M.pulse_width(_train(), n=n) == pytest.approx(expected,
                                                             abs=1e-12)

    @pytest.mark.parametrize('n,expected', [(1, 0.8), (-1, 0.4), (2, 0.6)])
    def test_negative_occurrence(self, n, expected):
        assert M.pulse_width(_train(), polarity='negative',
                             n=n) == pytest.approx(expected, abs=1e-12)

    def test_threshold(self):
        # Trapezoid pulse: 0 -> 1 over [0, 1], flat to 2, down to 0 at 3.
        s = _sig([-1, 0, 1, 2, 3, 4], [0, 0, 1, 1, 0, 0])
        for th in (0.25, 0.5, 0.75):
            # Rises at th, falls at 3 - th.
            assert M.pulse_width(s, th) == pytest.approx(3 - 2 * th,
                                                         abs=1e-12)

    def test_window(self):
        s = _train()
        np.testing.assert_allclose(M.pulse_widths(s, start=1.0, end=3.0),
                                   [0.4, 0.6], atol=1e-12)
        # A pulse cut by the window is not measured.
        np.testing.assert_allclose(M.pulse_widths(s, start=1.3, end=2.5),
                                   [], atol=0)
        assert M.pulse_width(s, start=2.0) == pytest.approx(0.6, abs=1e-12)
        assert M.pulse_width(s, end=2.0, n=-1) == pytest.approx(0.4,
                                                                abs=1e-12)
        np.testing.assert_allclose(
            M.pulse_widths(s, polarity='negative', start=1.0),
            [0.6, 0.4], atol=1e-12)

    def test_too_few_pulses_nan_and_warning(self):
        s = _train()
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(M.pulse_width(s, n=5))
        d = caught[0].message
        assert d.code == 'insufficient_crossings'
        assert d.metric == 'pulse_width'
        assert d.details['pulse_count'] == 4
        assert d.details['requested_pulse'] == 5
        assert d.details['polarity'] == 'positive'
        with pytest.warns(MetricWarning):
            assert math.isnan(M.pulse_width(s, polarity='negative', n=-4))

    def test_zero_pulses_plural_is_quiet_and_empty(self):
        s = _sig([0, 1, 2], [0, 0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = M.pulse_widths(s)
        assert isinstance(out, np.ndarray) and len(out) == 0

    def test_nan(self):
        s = _train()
        y = s.y.copy()
        y[3] = np.nan
        s = _sig(s.x, y)
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(M.pulse_width(s))
        assert caught[0].message.code == 'nan_waveform'
        with pytest.warns(MetricWarning):
            out = M.pulse_widths(s)
        assert len(out) == 1 and math.isnan(out[0])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert M.pulse_width(s, start=1.0) == pytest.approx(0.4, abs=1e-12)

    def test_return_types_never_depend_on_arguments(self):
        s = _train()
        assert type(M.pulse_width(s)) is float
        assert type(M.pulse_width(s, n=-1)) is float
        assert isinstance(M.pulse_widths(s), np.ndarray)

    @pytest.mark.parametrize('bad', [0, None, 1.0, True])
    def test_bad_occurrence_raises(self, bad):
        with pytest.raises(ValueError, match='pulse_width.*pulse_widths'):
            M.pulse_width(_train(), n=bad)

    @pytest.mark.parametrize('fn', ['pulse_width', 'pulse_widths'])
    def test_caller_errors_raise(self, fn):
        s = _train()
        with pytest.raises(ValueError, match=f'{fn}.*polarity'):
            getattr(M, fn)(s, polarity='high')
        with pytest.raises(ValueError, match=fn):
            getattr(M, fn)(s, start=3, end=2)
        with pytest.raises(ValueError, match=fn):
            getattr(M, fn)(s, end=-1)


# --- on_time / off_time -----------------------------------------------------

class TestOnOffTime:
    def test_window_exact_on_piecewise_linear_steps(self):
        # Sample-aligned edges: samples at 0.01 spacing, high in [1, 2).
        t = np.round(np.arange(0, 3.0001, 0.01), 10)
        y = ((t >= 1.0) & (t < 2.0)).astype(float)
        s = _sig(t, y)
        whole = M.on_time(s, 0.5)
        assert whole == pytest.approx(1.0, abs=0.011)
        assert M.on_time(s, 0.5, start=0.0, end=3.0) == pytest.approx(whole)
        assert M.on_time(s, 0.5, start=1.5) == pytest.approx(0.5, abs=0.011)
        assert M.on_time(s, 0.5, end=1.5) == pytest.approx(0.5, abs=0.011)
        assert M.on_time(s, 0.5, start=2.5) == 0.0
        assert M.off_time(s, 0.5, start=2.5) == pytest.approx(0.5)
        for kw in ({}, {'start': 0.7}, {'end': 2.2},
                   {'start': 1.234, 'end': 1.789}):
            a = kw.get('start', 0.0)
            b = kw.get('end', 3.0)
            assert (M.on_time(s, 0.5, **kw) + M.off_time(s, 0.5, **kw)
                    == pytest.approx(b - a, rel=1e-12))

    def test_window_bound_between_samples(self):
        s = _sig([0, 10], [1, 1])
        assert M.on_time(s, 0.5, start=2.5, end=7.25) == pytest.approx(4.75)
        assert M.off_time(s, 0.5, start=2.5, end=7.25) == 0.0

    def test_window_clipped_to_trace(self):
        s = _sig([0, 10], [1, 1])
        assert M.on_time(s, 0.5, start=-5, end=4) == pytest.approx(4.0)
        assert M.off_time(s, 0.5, start=-5, end=4) == 0.0

    def test_threshold_default_is_window_midpoint(self):
        s = _square(n=6, high=0.3)
        assert M.on_time(s) == M.on_time(s, 0.5)
        # Window inside one low stretch: flat, so nothing lies strictly
        # above its own midpoint.
        assert M.on_time(s, start=0.6, end=1.2) == 0.0

    def test_threshold_keyword_and_level_removed(self):
        s = _square()
        assert M.on_time(s, threshold=0.5) == M.on_time(s, 0.5)
        with pytest.raises(TypeError):
            M.on_time(s, level=0.5)
        with pytest.raises(TypeError):
            M.off_time(s, level=0.5)

    def test_nan_inside_window_only(self):
        t = np.linspace(0, 4, 401)
        y = (np.sin(2 * np.pi * t) > 0).astype(float)
        y[10] = np.nan
        s = _sig(t, y)
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(M.off_time(s, 0.5))
        assert len(caught) == 1 and caught[0].message.metric == 'on_time'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert not math.isnan(M.on_time(s, 0.5, start=1.0))

    @pytest.mark.parametrize('fn', ['on_time', 'off_time'])
    def test_window_errors_name_the_metric(self, fn):
        s = _square()
        with pytest.raises(ValueError, match=fn):
            getattr(M, fn)(s, 0.5, start=3, end=2)
        with pytest.raises(ValueError, match=fn):
            getattr(M, fn)(s, 0.5, start=100)


# --- schema -----------------------------------------------------------------

class TestSchema:
    def test_registered_entries(self):
        schema = export_metric_schema()
        for gone in ('duty_cycle', 'cycle_duty', 'pulse_widths'):
            assert gone not in schema
        duty = schema['average_duty']
        assert duty['unit'] == '' and duty['domain'] == 'tran'
        params = {p['name']: p for p in duty['parameters']}
        assert list(params) == ['threshold', 'edge', 'start', 'end']
        assert params['edge']['choices'] == ['rising', 'falling', 'any']
        assert params['threshold']['unit'] == '[y]'

        pw = schema['pulse_width']
        assert pw['unit'] == 's'
        params = {p['name']: p for p in pw['parameters']}
        assert list(params) == ['threshold', 'polarity', 'n', 'start', 'end']
        assert params['polarity']['default'] == 'positive'
        assert params['polarity']['choices'] == ['positive', 'negative']
        assert params['n']['default'] == 1

        for name in ('on_time', 'off_time'):
            params = {p['name']: p for p in schema[name]['parameters']}
            assert list(params) == ['threshold', 'start', 'end']
            assert params['threshold']['unit'] == '[y]'
            assert params['start']['unit'] == params['end']['unit'] == 's'
