"""TASK-366: average_period / average_frequency / cycle_period /
cycle_frequency on one per-cycle core.

The PINNED values below were captured from ``period_zero_cross`` and
``inst_frequency`` (and from ``tie``/``allan_deviation``) on the branch base
32378604, before they were removed or moved onto the core: the defaults of
the new functions must reproduce them exactly.
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


def _wobbly_clock():
    """Phase-modulated 1 GHz clock with a harmonic: periods all differ."""
    t = np.linspace(0, 10e-9, 2001)
    y = (0.9 + 0.9 * np.sin(2 * np.pi * 1e9 * t
                            + 0.3 * np.sin(2 * np.pi * 0.13e9 * t))
         + 0.05 * np.sin(2 * np.pi * 3.7e9 * t))
    return _sig(t, y)


def _square(period=1.0, n=6, high=0.3, step=0.01):
    """Piecewise-linear clock: rises at k*period, falls at k*period+high.

    Edges are exact at mid-swing (0.5), so every crossing is analytic.
    """
    t, y = [0.0], [0.0]
    for k in range(n):
        r = k * period + 0.25
        for x, v in ((r - step, 0.0), (r + step, 1.0),
                     (r + high - step, 1.0), (r + high + step, 0.0)):
            t.append(x)
            y.append(v)
    t.append(n * period + 0.5)
    y.append(0.0)
    return _sig(t, y)


# --- pinned pre-change results ----------------------------------------------

PZC = {None: 9.961096812899226e-10, 0.5: 9.999103665765928e-10,
       1.2: 9.95471690472612e-10}
IF_X = [4.85473877869062e-10, 1.4574726246680457e-09, 2.456751139457325e-09,
        3.491578024516593e-09, 4.530685065458397e-09, 5.543515836105391e-09,
        6.5322816565597795e-09, 7.508759859920911e-09, 8.47260032350712e-09,
        9.457014124090356e-09]
IF_Y = [1030039835.0595539, 1027578929.7864128, 975233196.6995806,
        957617758.380298, 967159062.4975463, 1008363920.8733724,
        1014377598.5389835, 1033986922.814233, 1041069453.5890859,
        991791050.741564]
TIE_Y = [0.0, -2.5273441794211283e-11, -4.822186898187748e-11,
         -1.893577479549721e-11, 2.921253855681291e-11,
         6.705894450826566e-11, 6.265471727095572e-11,
         5.237122283719802e-11, 2.3391761413373157e-11,
         -1.2167212570229403e-11, -1.6543612251060553e-24]
ALLAN_Y = [0.02074476725064238, 0.02706751290550906]
ALLAN_X = [9.961096812899226e-10, 1.9922193625798452e-09]


class TestPinnedAgainstRemovedFunctions:
    @pytest.mark.parametrize('threshold', [None, 0.5, 1.2])
    def test_average_period_equals_period_zero_cross(self, threshold):
        assert M.average_period(_wobbly_clock(), threshold) == PZC[threshold]

    def test_cycle_frequency_equals_inst_frequency(self):
        f = M.cycle_frequency(_wobbly_clock())
        assert list(f.x) == IF_X
        assert list(f.y) == IF_Y
        assert (f.xlabel, f.ylabel, f.type) == ('time', 'Hz', 'frequency')

    def test_tie_and_allan_unchanged_on_the_core(self):
        s = _wobbly_clock()
        # TASK-418: the default reference is now least squares; the pinned
        # pre-418 result stays one argument away.
        assert list(M.tie(s, reference='endpoints').y) == TIE_Y
        a = M.allan_deviation(s)
        assert list(a.y) == ALLAN_Y
        assert list(a.x) == ALLAN_X

    def test_old_names_removed(self):
        import silk.metrics.timing as T
        for name in ('period_zero_cross', 'inst_frequency'):
            assert not hasattr(M, name)
            assert not hasattr(T, name)
            assert name not in M.__all__


class TestOneCore:
    def test_frequency_is_reciprocal_of_period(self):
        s = _wobbly_clock()
        assert M.average_frequency(s) == 1.0 / M.average_period(s)
        p, f = M.cycle_period(s), M.cycle_frequency(s)
        np.testing.assert_array_equal(p.x, f.x)
        np.testing.assert_array_equal(f.y, 1.0 / p.y)
        assert (p.xlabel, p.ylabel) == ('time', 's')
        assert p.name == 'cycle_period(clk)'
        assert f.name == 'cycle_frequency(clk)'

    def test_average_is_mean_of_cycle_waveform(self):
        s = _wobbly_clock()
        for edge in ('rising', 'falling', 'any'):
            assert M.average_period(s, edge=edge) == pytest.approx(
                float(np.mean(M.cycle_period(s, edge=edge).y)), rel=1e-15)

    def test_average_frequency_is_not_mean_of_cycle_frequencies(self):
        s = _wobbly_clock()
        mean_f = float(np.mean(M.cycle_frequency(s).y))
        assert M.average_frequency(s) != mean_f


class TestAnalytic:
    def test_rising_periods_exact(self):
        s = _square(period=1.0, n=6)
        p = M.cycle_period(s)
        np.testing.assert_allclose(p.y, np.ones(5), rtol=0, atol=1e-12)
        np.testing.assert_allclose(p.x, np.arange(5) + 0.75, atol=1e-12)
        assert M.average_period(s) == pytest.approx(1.0, abs=1e-12)
        assert M.average_frequency(s) == pytest.approx(1.0, abs=1e-12)

    def test_falling_edge_periods(self):
        # Rising every 1.0; falling edges at k + 0.25 + high with high
        # varying per cycle, so falling periods differ from rising ones.
        t, y = [0.0], [0.0]
        highs = [0.2, 0.4, 0.3, 0.5]
        for k, h in enumerate(highs):
            r = k + 0.25
            t += [r - .01, r + .01, r + h - .01, r + h + .01]
            y += [0.0, 1.0, 1.0, 0.0]
        t.append(5.0)
        y.append(0.0)
        s = _sig(t, y)
        falls = np.array([k + 0.25 + h for k, h in enumerate(highs)])
        p = M.cycle_period(s, 0.5, edge='falling')
        np.testing.assert_allclose(p.y, np.diff(falls), atol=1e-12)
        np.testing.assert_allclose(p.x, 0.5 * (falls[:-1] + falls[1:]),
                                   atol=1e-12)
        assert M.average_period(s, 0.5, edge='falling') == pytest.approx(
            (falls[-1] - falls[0]) / 3, abs=1e-12)
        assert M.average_period(s, 0.5) == pytest.approx(1.0, abs=1e-12)

    def test_any_edge_gives_full_periods_not_half(self):
        s = _square(period=1.0, n=6, high=0.3)
        p = M.cycle_period(s, edge='any')
        # 12 crossings -> 10 same-direction full periods, all 1.0
        assert len(p.y) == 10
        np.testing.assert_allclose(p.y, 1.0, atol=1e-12)
        assert M.average_frequency(s, edge='any') == pytest.approx(1.0,
                                                                   abs=1e-12)

    def test_windowed_average(self):
        # Period 1.0 for the first three cycles, then 2.0.
        t, y = [0.0], [0.0]
        rises = [0.25, 1.25, 2.25, 3.25, 5.25, 7.25]
        for r in rises:
            t += [r - .01, r + .01, r + .29, r + .31]
            y += [0.0, 1.0, 1.0, 0.0]
        t.append(8.0)
        y.append(0.0)
        s = _sig(t, y)
        assert M.average_period(s) == pytest.approx(7.0 / 5, abs=1e-12)
        assert M.average_period(s, end=3.3) == pytest.approx(1.0, abs=1e-12)
        assert M.average_period(s, start=3.0) == pytest.approx(2.0, abs=1e-12)
        assert M.average_period(s, start=1.0, end=5.5) == pytest.approx(
            4.0 / 3, abs=1e-12)
        p = M.cycle_period(s, start=1.0, end=5.5)
        np.testing.assert_allclose(p.y, [1.0, 1.0, 2.0], atol=1e-12)

    def test_window_threshold_is_swing_inside_window(self):
        # Swing 0..1 before t=4, 0..3 after; the auto threshold inside a late
        # window is 1.5, which the early small pulses never cross.
        t, y = [0.0], [0.0]
        for r, a in ((0.25, 1), (1.25, 1), (2.25, 1),
                     (4.25, 3), (5.25, 3), (6.25, 3)):
            t += [r - .01, r + .01, r + .29, r + .31]
            y += [0.0, a, a, 0.0]
        t.append(7.0)
        y.append(0.0)
        s = _sig(t, y)
        assert M.average_period(s, start=4.0) == pytest.approx(1.0, abs=1e-12)
        # whole trace: threshold 1.5 -> only the three tall pulses count
        assert M.average_period(s) == pytest.approx(1.0, abs=1e-12)
        assert len(M.cycle_period(s).y) == 2


class TestUnavailable:
    @pytest.mark.parametrize('fn', ['average_period', 'average_frequency'])
    def test_too_few_edges_nan_and_warning(self, fn):
        s = _square(n=1)
        with pytest.warns(MetricWarning) as rec:
            assert math.isnan(getattr(M, fn)(s))
        w = rec[0].message
        assert w.code == 'insufficient_crossings'
        assert w.metric == fn
        assert w.details['crossing_count'] == 1
        assert w.details['required_crossing_count'] == 2

    def test_any_edge_needs_three_crossings(self):
        s = _square(n=1)  # one rise, one fall
        with pytest.warns(MetricWarning) as rec:
            assert math.isnan(M.average_period(s, edge='any'))
        assert rec[0].message.details['required_crossing_count'] == 3

    def test_window_with_one_edge(self):
        s = _square(n=6)
        with pytest.warns(MetricWarning, match='full cycle|period'):
            assert math.isnan(M.average_period(s, start=1.0, end=1.5))

    @pytest.mark.parametrize('fn', ['cycle_period', 'cycle_frequency'])
    def test_waveform_too_few_edges_empty_and_warning(self, fn):
        with pytest.warns(MetricWarning) as rec:
            out = getattr(M, fn)(_square(n=1))
        assert len(out.x) == 0 and len(out.y) == 0
        assert rec[0].message.code == 'insufficient_crossings'

    @pytest.mark.parametrize('fn', ['average_period', 'average_frequency',
                                    'cycle_period', 'cycle_frequency'])
    def test_nan_in_consumed_data(self, fn):
        s = _square(n=6)
        y = s.y.copy()
        y[5] = np.nan
        s = _sig(s.x, y)
        with pytest.warns(MetricWarning) as rec:
            out = getattr(M, fn)(s)
        assert [w.message.code for w in rec] == ['nan_waveform']
        if fn.startswith('average'):
            assert math.isnan(out)
        else:
            assert len(out.y) == 1 and math.isnan(out.y[0])

    def test_nan_outside_window_ignored(self):
        s = _square(n=6)
        y = s.y.copy()
        y[-1] = np.nan
        s = _sig(s.x, y)
        with warnings.catch_warnings():
            warnings.simplefilter('error', MetricWarning)
            assert M.average_period(s, 0.5, end=4.5) == pytest.approx(
                1.0, abs=1e-12)

    def test_flat_trace(self):
        with pytest.warns(MetricWarning):
            assert math.isnan(M.average_period(_sig([0, 1, 2], [1, 1, 1])))


class TestCallerErrors:
    @pytest.mark.parametrize('fn', ['average_period', 'average_frequency',
                                    'cycle_period', 'cycle_frequency'])
    def test_bad_edge_and_window_raise(self, fn):
        s = _square()
        f = getattr(M, fn)
        with pytest.raises(ValueError, match=fn):
            f(s, edge='both')
        with pytest.raises(ValueError, match='end >= start'):
            f(s, start=3.0, end=2.0)
        with pytest.raises(ValueError, match='does not overlap'):
            f(s, start=100.0)

    def test_non_transient_raises(self):
        s = Signal('h', 'voltage', 'frequency', 'V', np.arange(3.0),
                   np.arange(3.0))
        with pytest.raises(ValueError, match='average_period'):
            M.average_period(s)


class TestSchema:
    def test_registered_entries(self):
        schema = export_metric_schema()
        assert 'period_zero_cross' not in schema
        assert 'cycle_period' not in schema
        assert 'cycle_frequency' not in schema
        for name, unit in (('average_period', 's'),
                           ('average_frequency', 'Hz')):
            entry = schema[name]
            assert entry['unit'] == unit
            assert entry['domain'] == 'tran'
            params = {p['name']: p for p in entry['parameters']}
            assert list(params) == ['threshold', 'edge', 'start', 'end']
            assert params['edge']['default'] == 'rising'
            assert params['edge']['choices'] == ['rising', 'falling', 'any']
            assert params['threshold']['unit'] == '[y]'
            assert params['start']['unit'] == params['end']['unit'] == 's'
