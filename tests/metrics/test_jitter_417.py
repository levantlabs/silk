"""TASK-417: period_jitter / cycle_to_cycle_jitter on one per-cycle core.

Threshold, edge, start/end and a reference (the mean, or a centred moving
average over ``cycles`` periods); RMS and peak-to-peak scalars in the
catalog; ``period_deviation`` / ``cycle_to_cycle_difference`` as the
unregistered waveform forms.

Every clock below is piecewise linear with a steep ramp centred on each
edge, so each mid-swing crossing is exact and the expected values are
computed from the edge list alone, never from the code under test.
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

SCALARS = ('period_jitter', 'period_jitter_pp',
           'cycle_to_cycle_jitter', 'cycle_to_cycle_jitter_pp')
WAVEFORMS = ('period_deviation', 'cycle_to_cycle_difference')


def _sig(t, y, name='clk'):
    return Signal(name, 'voltage', 'time', 'V', np.asarray(t, float),
                  np.asarray(y, float))


def _clock(rises, high=0.3, low_v=0.0, high_v=1.0, ramp=1e-3, tail=0.5):
    """A clock rising at each time in ``rises`` and falling ``high`` later
    (``high`` a fraction of the following period), crossing mid-swing at
    exactly those times."""
    rises = np.asarray(rises, float)
    periods = np.diff(rises)
    falls = rises[:-1] + high * periods
    falls = np.append(falls, rises[-1] + high * periods[-1])
    t, y = [rises[0] - tail], [low_v]
    for r, f in zip(rises, falls):
        t += [r - ramp, r + ramp, f - ramp, f + ramp]
        y += [low_v, high_v, high_v, low_v]
    t.append(falls[-1] + tail)
    y.append(low_v)
    return _sig(t, y), falls


def _rms(x):
    return math.sqrt(np.mean(np.square(x)))


P = np.array([1.0, 1.2, 0.9, 1.1, 1.0, 1.3, 0.8])
RISES = np.concatenate([[0.0], np.cumsum(P)]) + 1.0


def _quiet():
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter('error')
    return ctx


# --- AC#1: analytic edge sequences ------------------------------------------

class TestAnalyticMeanReference:
    def setup_method(self):
        self.clk, _ = _clock(RISES)

    def test_period_jitter_is_rms_of_deviation_from_mean(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert M.period_jitter(self.clk) == pytest.approx(
                _rms(P - P.mean()), rel=1e-9)
            assert M.period_jitter_pp(self.clk) == pytest.approx(
                P.max() - P.min(), rel=1e-9)

    def test_cycle_to_cycle_is_rms_of_period_differences(self):
        d = np.diff(P)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert M.cycle_to_cycle_jitter(self.clk) == pytest.approx(
                _rms(d), rel=1e-9)
            assert M.cycle_to_cycle_jitter_pp(self.clk) == pytest.approx(
                d.max() - d.min(), rel=1e-9)

    def test_window_counts_only_periods_with_both_edges_inside(self):
        # [RISES[2] - eps, RISES[6] + eps] holds edges 2..6 -> periods 2..5.
        sub = P[2:6]
        kw = dict(start=RISES[2] - 0.01, end=RISES[6] + 0.01, threshold=0.5)
        assert M.period_jitter(self.clk, **kw) == pytest.approx(
            _rms(sub - sub.mean()), rel=1e-9)
        assert M.cycle_to_cycle_jitter_pp(self.clk, **kw) == pytest.approx(
            np.ptp(np.diff(sub)), rel=1e-9)

    def test_falling_edge_reads_the_falling_periods(self):
        clk, falls = _clock(RISES, high=0.4)
        fp = np.diff(falls)
        assert M.period_jitter(clk, edge='falling') == pytest.approx(
            _rms(fp - fp.mean()), rel=1e-9)
        assert M.cycle_to_cycle_jitter(clk, edge='falling') == pytest.approx(
            _rms(np.diff(fp)), rel=1e-9)

    def test_edge_any_measures_full_cycles_not_half_periods(self):
        # Catalog review T6: a jitter-free 30 % duty clock used to read
        # 0.20*P period jitter and 0.41*P c2c with edge='any'.
        clk, _ = _clock(np.arange(12, dtype=float), high=0.3)
        for name in SCALARS:
            assert getattr(M, name)(clk, edge='any') == pytest.approx(
                0.0, abs=1e-12), name

    def test_edge_any_pairs_each_cycle_with_the_one_after_it(self):
        clk, falls = _clock(RISES, high=0.3)
        edges = np.sort(np.concatenate([RISES, falls]))
        periods = edges[2:] - edges[:-2]
        dev = periods - periods.mean()
        c2c = dev[2:] - dev[:-2]
        assert M.period_jitter(clk, edge='any') == pytest.approx(
            _rms(dev), rel=1e-9)
        assert M.cycle_to_cycle_jitter_pp(clk, edge='any') == pytest.approx(
            np.ptp(c2c), rel=1e-9)
        w = M.cycle_to_cycle_difference(clk, edge='any')
        np.testing.assert_allclose(w.x, edges[2:-2], rtol=0, atol=1e-12)


class TestMovingAverageReference:
    def setup_method(self):
        self.clk, _ = _clock(RISES)

    def test_odd_window_is_the_centred_mean_of_n_periods(self):
        ref = np.convolve(P, np.ones(3) / 3, mode='valid')
        dev = P[1:-1] - ref
        kw = dict(reference='moving_average', cycles=3)
        assert M.period_jitter(self.clk, **kw) == pytest.approx(
            _rms(dev), rel=1e-9)
        assert M.period_jitter_pp(self.clk, **kw) == pytest.approx(
            np.ptp(dev), rel=1e-9)
        assert M.cycle_to_cycle_jitter(self.clk, **kw) == pytest.approx(
            _rms(np.diff(dev)), rel=1e-9)
        w = M.period_deviation(self.clk, **kw)
        mids = 0.5 * (RISES[:-1] + RISES[1:])
        np.testing.assert_allclose(w.x, mids[1:-1], atol=1e-12)
        np.testing.assert_allclose(w.y, dev, atol=1e-12)

    def test_even_window_half_weights_the_two_end_periods(self):
        # cycles=4: weights [1/8, 1/4, 1/4, 1/4, 1/8] centred on the period.
        k = np.array([0.5, 1, 1, 1, 0.5]) / 4
        dev = P[2:-2] - np.convolve(P, k, mode='valid')
        got = M.period_deviation(self.clk, reference='moving_average',
                                 cycles=4)
        np.testing.assert_allclose(got.y, dev, atol=1e-12)
        assert M.period_jitter(self.clk, reference='moving_average',
                               cycles=4) == pytest.approx(_rms(dev), rel=1e-9)


# --- AC#2: the waveform forms -----------------------------------------------

class TestWaveforms:
    def setup_method(self):
        self.clk, _ = _clock(RISES)

    def test_period_deviation_against_period_midpoints(self):
        w = M.period_deviation(self.clk)
        assert (w.name, w.type, w.xlabel, w.ylabel) == (
            'period_deviation(clk)', 'period_deviation', 'time', 's')
        np.testing.assert_allclose(w.x, M.cycle_period(self.clk).x, atol=0)
        np.testing.assert_allclose(w.y, P - P.mean(), atol=1e-12)
        assert M.period_jitter(self.clk) == pytest.approx(_rms(w.y), rel=1e-12)
        assert M.period_jitter_pp(self.clk) == pytest.approx(np.ptp(w.y),
                                                             rel=1e-12)

    def test_cycle_to_cycle_difference_against_the_shared_edge(self):
        w = M.cycle_to_cycle_difference(self.clk)
        assert (w.name, w.type, w.xlabel, w.ylabel) == (
            'cycle_to_cycle_difference(clk)', 'period_difference', 'time', 's')
        np.testing.assert_allclose(w.x, RISES[1:-1], atol=1e-12)
        np.testing.assert_allclose(w.y, np.diff(P), atol=1e-12)
        assert M.cycle_to_cycle_jitter(self.clk) == pytest.approx(_rms(w.y),
                                                                  rel=1e-12)

    def test_waveforms_honour_window_threshold_and_edge(self):
        kw = dict(threshold=0.5, edge='falling', start=RISES[1],
                  end=RISES[-1], reference='moving_average', cycles=3)
        clk, falls = _clock(RISES, high=0.4)
        f = falls[(falls >= RISES[1]) & (falls <= RISES[-1])]
        fp = np.diff(f)
        dev = fp[1:-1] - np.convolve(fp, np.ones(3) / 3, mode='valid')
        w = M.period_deviation(clk, **kw)
        np.testing.assert_allclose(w.y, dev, atol=1e-12)
        np.testing.assert_allclose(w.x, 0.5 * (f[1:-2] + f[2:-1]), atol=1e-12)
        c = M.cycle_to_cycle_difference(clk, **kw)
        np.testing.assert_allclose(c.y, np.diff(dev), atol=1e-12)
        np.testing.assert_allclose(c.x, f[2:-2], atol=1e-12)

    def test_waveforms_are_not_registered(self):
        schema = export_metric_schema()
        for name in WAVEFORMS:
            assert name not in schema


# --- AC#3: drift, and threshold=None ----------------------------------------

def _chirp_rises(n=40, t0=1.0, slope=0.01):
    periods = t0 + slope * np.arange(n)
    return np.concatenate([[0.0], np.cumsum(periods)]) + 1.0, periods


class TestDrift:
    def test_mean_reports_linear_drift_moving_average_removes_it(self):
        rises, periods = _chirp_rises()
        clk, _ = _clock(rises)
        n = len(periods)
        drift_rms = 0.01 * math.sqrt((n * n - 1) / 12)
        assert M.period_jitter(clk) == pytest.approx(drift_rms, rel=1e-6)
        assert M.period_jitter_pp(clk) == pytest.approx(0.01 * (n - 1),
                                                        rel=1e-6)
        # c2c is a true RMS: the constant step of the drift shows up.
        assert M.cycle_to_cycle_jitter(clk) == pytest.approx(0.01, rel=1e-6)
        for cycles in (2, 3, 5, 8):
            kw = dict(reference='moving_average', cycles=cycles)
            for name in SCALARS:
                assert getattr(M, name)(clk, **kw) == pytest.approx(
                    0.0, abs=1e-9), (name, cycles)

    def test_sampled_sine_chirp(self):
        # A sampled sine whose frequency ramps 1 -> 1.2 GHz over 200 ns.
        t = np.linspace(0, 200e-9, 400001)
        f0, k = 1e9, 0.2e9 / 200e-9
        y = 0.5 + 0.5 * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
        clk = _sig(t, y)
        mean_pj = M.period_jitter(clk)
        ma_pj = M.period_jitter(clk, reference='moving_average', cycles=5)
        assert mean_pj > 40e-12
        assert ma_pj < 1e-3 * mean_pj
        assert M.cycle_to_cycle_jitter(
            clk, reference='moving_average', cycles=5) < 1e-3 * \
            M.cycle_to_cycle_jitter(clk)

    def test_threshold_none_is_the_mid_swing(self):
        clk, _ = _clock(RISES + 0.0, low_v=0.2, high_v=1.0)
        for name in SCALARS:
            fn = getattr(M, name)
            assert fn(clk) == fn(clk, threshold=0.6), name
        np.testing.assert_array_equal(M.period_deviation(clk).y,
                                      M.period_deviation(clk, threshold=0.6).y)


# --- AC#4: decision-1 -------------------------------------------------------

@pytest.mark.parametrize('name', SCALARS + WAVEFORMS)
class TestDecisionOne:
    def setup_method(self):
        self.clk, _ = _clock(RISES)

    @pytest.mark.parametrize('kwargs', [
        dict(reference='median'),
        dict(reference='moving_average'),
        dict(reference='moving_average', cycles=1),
        dict(reference='moving_average', cycles=0),
        dict(reference='moving_average', cycles=2.5),
        dict(reference='moving_average', cycles=True),
        dict(cycles=3),
        dict(edge='rsing'),
        dict(start=3.0, end=3.0),
        dict(start=4.0, end=3.0),
    ])
    def test_malformed_request_raises(self, name, kwargs):
        with pytest.raises(ValueError, match=name):
            getattr(M, name)(self.clk, **kwargs)

    def test_too_few_edges_for_the_moving_average(self, name):
        # 7 periods: a 7-period average leaves one deviation, 9 leaves none.
        scalar = name in SCALARS
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = getattr(M, name)(self.clk, reference='moving_average',
                                      cycles=9)
        if scalar:
            assert type(result) is float and math.isnan(result)
        else:
            assert len(result.x) == len(result.y) == 0
        [w] = [c.message for c in caught]
        assert isinstance(w, MetricWarning)
        assert (w.metric, w.code) == (name, 'insufficient_edges')
        assert w.details['edge_count'] == 8
        assert w.details['reference'] == 'moving_average'
        assert w.details['cycles'] == 9
        c2c = name.startswith('cycle')
        points = 2 if scalar else 1
        needed = points + (1 if c2c else 0) + 9 - 1   # periods
        assert w.details['required_edge_count'] == needed + 1
        assert 'average fewer cycles' in w.suggestion

    def test_too_few_edges_for_the_mean(self, name):
        clk, _ = _clock([1.0, 2.0, 3.1])        # two periods
        fn = getattr(M, name)
        if name in ('period_jitter', 'period_jitter_pp', 'period_deviation'):
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                fn(clk)
            return
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = fn(clk)
        if name in SCALARS:
            assert math.isnan(result)
            assert caught[0].message.details['required_edge_count'] == 4
        else:
            assert len(result.y) == 1     # one difference is a waveform
            assert caught == []

    def test_nan_sample_is_unknown_not_short(self, name):
        y = self.clk.y.copy()
        y[5] = np.nan
        clk = _sig(self.clk.x, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = getattr(M, name)(clk)
        if name in SCALARS:
            assert math.isnan(result)
        else:
            assert len(result.y) == 1 and np.isnan(result.y).all()
        [w] = [c.message for c in caught]
        assert (w.metric, w.code) == (name, 'nan_waveform')


# --- catalog ----------------------------------------------------------------

def test_catalog_entries():
    schema = export_metric_schema()
    for name in SCALARS:
        spec = schema[name]
        assert spec['unit'] == 's'
        params = {p['name']: p for p in spec['parameters']}
        assert list(params) == ['threshold', 'edge', 'start', 'end',
                                'reference', 'cycles']
        assert params['reference']['choices'] == ['mean', 'moving_average']
        assert params['reference']['default'] == 'mean'
        assert params['edge']['choices'] == ['rising', 'falling', 'any']
