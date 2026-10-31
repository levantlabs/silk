"""TASK-418: tie / long_term_jitter units, least-squares reference,
ideal_period and phase overrides, and the crossings selection keywords.

The ``MAIN`` digests below were captured from ``tie`` and ``long_term_jitter``
on the branch base 13742d33, before this change. ``reference='endpoints'``
must reproduce them bit for bit (reviewer ruling 2026-10-02).
"""
from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals import Signal

T = 1e-9


def _sig(t, y, name='clk'):
    return Signal(name, 'voltage', 'time', 'V', np.asarray(t, float),
                  np.asarray(y, float))


def _clock(edges, high=0.4 * T, tr=0.02 * T, t_end=None):
    """Piecewise-linear clock rising through 0.5 exactly at each of *edges*.

    The samples sit on the corners of each ramp, so linear interpolation
    places every rising crossing exactly at its edge time and every
    falling one exactly at ``edge + high``.
    """
    edges = np.asarray(edges, float)
    t, y = [edges[0] - 0.5 * T], [0.0]
    for e in edges:
        for x, v in ((e - tr / 2, 0.0), (e + tr / 2, 1.0),
                     (e + high - tr / 2, 1.0), (e + high + tr / 2, 0.0)):
            t.append(x)
            y.append(v)
    t.append(edges[-1] + T if t_end is None else t_end)
    y.append(0.0)
    return _sig(t, y)


def _wobbly():
    t = np.linspace(0, 10e-9, 2001)
    y = (0.9 + 0.9 * np.sin(2 * np.pi * 1e9 * t
                            + 0.3 * np.sin(2 * np.pi * 0.13e9 * t))
         + 0.05 * np.sin(2 * np.pi * 3.7e9 * t))
    return _sig(t, y)


def _noisy():
    rng = np.random.default_rng(418)
    t = np.linspace(0, 40e-9, 40 * 64 + 1)
    jit = np.cumsum(rng.normal(0, 2e-12, len(t)))
    y = 0.5 + 0.5 * np.tanh(4 * np.sin(2 * np.pi * (t + jit * 0.01) / T))
    return _sig(t, y)


ERR = np.array([3, -1, 4, -1, -5, 9, -2, 6, -5, 3, -5, 8]) * 1e-12
K = np.arange(len(ERR), dtype=float)


def _lsq_residual(edges):
    k = np.arange(len(edges), dtype=float)
    slope, icept = np.polyfit(k, edges, 1)
    return edges - (icept + slope * k)


# --- AC#1: units -------------------------------------------------------------

class TestUnits:
    PHASE = 2e-9

    @pytest.mark.parametrize('units,scale', [
        ('s', 1.0), ('ui', 1 / T), ('rad', 2 * math.pi / T),
        ('deg', 360 / T)])
    def test_tie_conversion_with_given_reference(self, units, scale):
        clk = _clock(self.PHASE + K * T + ERR)
        tie = M.tie(clk, units=units, ideal_period=T, phase=self.PHASE)
        np.testing.assert_allclose(tie.y, ERR * scale, rtol=1e-9, atol=1e-9 * scale * 1e-12)
        assert tie.ylabel == units
        np.testing.assert_allclose(tie.x, self.PHASE + K * T + ERR, rtol=0, atol=1e-21)

    @pytest.mark.parametrize('units,scale', [
        ('ui', 1 / T), ('rad', 2 * math.pi / T), ('deg', 360 / T)])
    def test_long_term_jitter_scales_with_the_fitted_period(self, units, scale):
        # Without ideal_period the conversion uses the FITTED period.
        edges = self.PHASE + K * 1.01 * T + ERR
        clk = _clock(edges)
        t_fit = np.polyfit(K, edges, 1)[0]
        seconds = M.long_term_jitter(clk)
        assert M.long_term_jitter(clk, units=units) == pytest.approx(
            seconds * scale * T / t_fit, rel=1e-9)

    def test_rad_and_deg_agree(self):
        clk = _clock(self.PHASE + K * T + ERR)
        rad = M.long_term_jitter(clk, units='rad')
        deg = M.long_term_jitter(clk, units='deg')
        assert deg == pytest.approx(math.degrees(rad), rel=1e-12)


# --- AC#2: least-squares default, overrides, outlier prediction --------------

class TestReference:
    def test_default_is_the_least_squares_line(self):
        edges = 0.7e-9 + K * 1.003 * T + ERR
        tie = M.tie(_clock(edges))
        np.testing.assert_allclose(tie.y, _lsq_residual(edges), rtol=0,
                                   atol=1e-22)
        assert abs(np.sum(tie.y)) < 1e-22           # residuals sum to 0
        assert abs(np.sum(K * tie.y)) < 1e-20       # and are orthogonal to k
        assert M.long_term_jitter(_clock(edges)) == pytest.approx(
            float(np.std(_lsq_residual(edges), ddof=1)), rel=1e-9)

    def test_ideal_period_fixes_the_slope_only(self):
        edges = 0.7e-9 + K * T + ERR
        tie = M.tie(_clock(edges), ideal_period=T)
        np.testing.assert_allclose(tie.y, ERR - ERR.mean(), rtol=0,
                                   atol=1e-22)

    def test_phase_fixes_the_intercept_only(self):
        phase = 0.7e-9
        edges = phase + K * T + ERR
        tie = M.tie(_clock(edges), phase=phase)
        slope = np.sum(K * (edges - phase)) / np.sum(K * K)
        np.testing.assert_allclose(tie.y, edges - (phase + K * slope),
                                   rtol=0, atol=1e-22)
        assert tie.y[0] == pytest.approx(ERR[0], abs=1e-22)

    @pytest.mark.parametrize('reference', ['fit', 'endpoints'])
    def test_both_overrides_fully_give_the_line(self, reference):
        phase = 0.7e-9
        tie = M.tie(_clock(phase + K * T + ERR), ideal_period=T,
                    phase=phase, reference=reference)
        np.testing.assert_allclose(tie.y, ERR, rtol=0, atol=1e-22)

    @pytest.mark.parametrize('where', [0, -1])
    def test_one_outlier_end_edge_matches_prediction(self, where):
        """A single edge off by ``d`` at either end of an otherwise ideal clock.

        Endpoints: the line passes through the outlier, so its own TIE is 0
        and the error is smeared over every other edge as
        ``-d·(1 - j/(N-1))``, ``j`` = distance from the outlier (almost ``-d``
        beside it, ``0`` at the far end).
        Least squares: the TIE is ``d·(δ_k,o - h_k,o)`` with the hat
        matrix ``h_k,o = 1/N + (k - k̄)(o - k̄)/Σ(k - k̄)²``.
        """
        n, d = 16, 20e-12
        k = np.arange(n, dtype=float)
        edges = 1e-9 + k * T
        o = int(k[where])
        edges[o] += d
        clk = _clock(edges)
        j = np.abs(k - o)
        endpoint_pred = np.where(k == o, 0.0, -d * (1 - j / (n - 1)))
        kc = k - k.mean()
        lsq_pred = d * ((k == o) - (1 / n + kc * kc[o] / np.sum(kc * kc)))
        ep = M.tie(clk, reference='endpoints').y
        fit = M.tie(clk).y
        np.testing.assert_allclose(ep, endpoint_pred, rtol=0, atol=1e-22)
        np.testing.assert_allclose(fit, lsq_pred, rtol=0, atol=1e-22)
        # The outlier is invisible to the endpoint line but is the largest
        # TIE of the fit.
        assert ep[o] == pytest.approx(0, abs=1e-22)
        assert int(np.argmax(np.abs(fit))) == o
        assert M.long_term_jitter(clk) == pytest.approx(
            float(np.std(lsq_pred, ddof=1)), rel=1e-9)
        assert M.long_term_jitter(clk, reference='endpoints') == pytest.approx(
            float(np.std(endpoint_pred, ddof=1)), rel=1e-9)


# --- reviewer ruling: reference='endpoints' == main, bit for bit -------------

MAIN = {
    'wobbly': (_wobbly, {},
               '6535474cd74cb5e340b5f8cccebcd740787349c5a9353dc88e87c51fb377a283',
               '0x1.4eec0dcfe0f1fp-35'),
    'wobbly_period': (_wobbly, {'ideal_period': 1e-9},
                      '91ba87290315e016db6e36490d92043ed61d6b246fe5c068d5ce39d8dde087b5',
                      '0x1.39a21298e078bp-35'),
    'noisy': (_noisy, {},
              'c42069978d38ec0c2064fa5cb468d1317d99cc1b06f9ca50d708e7f67cb2ea0a',
              '0x1.91f365943b010p-43'),
    'noisy_period': (_noisy, {'ideal_period': 1.001e-9},
                     '3db595a10399712dcdeabfc5e9249a8aa35c4549a0164f081f06ff948393c291',
                     '0x1.ab4f5d297b5b1p-37'),
}


@pytest.mark.parametrize('case', sorted(MAIN))
def test_endpoints_reproduces_main_bit_for_bit(case):
    make, kwargs, tie_digest, ltj_hex = MAIN[case]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        tie = M.tie(make(), reference='endpoints', **kwargs)
        ltj = M.long_term_jitter(make(), reference='endpoints', **kwargs)
    assert hashlib.sha256(tie.x.tobytes() + tie.y.tobytes()).hexdigest() == tie_digest
    assert ltj.hex() == ltj_hex


@pytest.mark.parametrize('case', sorted(MAIN))
def test_default_differs_from_main(case):
    make, kwargs, _, ltj_hex = MAIN[case]
    assert M.long_term_jitter(make(), **kwargs).hex() != ltj_hex


# --- AC#1: threshold, edge, start, end ---------------------------------------

class TestSelection:
    def test_falling_edges(self):
        high = 0.4 * T
        edges = 1e-9 + K * T + ERR
        tie = M.tie(_clock(edges, high=high), edge='falling', ideal_period=T,
                    phase=1e-9 + high)
        np.testing.assert_allclose(tie.y, ERR, rtol=0, atol=1e-21)

    def test_any_edge_uses_half_period_spacing(self):
        high = 0.5 * T
        edges = 1e-9 + K * T
        tie = M.tie(_clock(edges, high=high), edge='any', ideal_period=T,
                    units='ui')
        assert len(tie.y) == 2 * len(K)
        np.testing.assert_allclose(tie.y, 0, atol=1e-9)
        # Duty-cycle distortion of 0.1 T reads as ±0.05 UI alternating.
        tie = M.tie(_clock(edges, high=0.6 * T), edge='any', ideal_period=T,
                    units='ui')
        np.testing.assert_allclose(np.abs(tie.y), 0.05, atol=1e-9)

    def test_start_end_window(self):
        edges = 1e-9 + K * T + ERR
        sel = slice(2, 9)
        tie = M.tie(_clock(edges), start=edges[2] - 0.1 * T,
                    end=edges[8] + 0.1 * T)
        np.testing.assert_allclose(tie.x, edges[sel], rtol=0, atol=1e-21)
        np.testing.assert_allclose(tie.y, _lsq_residual(edges[sel]),
                                   rtol=0, atol=1e-22)

    def test_threshold(self):
        # Ramps are symmetric, so a threshold 0.25 shifts every rising edge
        # by the same amount: the TIE is unchanged, the edge times are not.
        edges = 1e-9 + K * T + ERR
        clk = _clock(edges)
        low = M.tie(clk, threshold=0.25)
        np.testing.assert_allclose(low.x, edges - 0.25 * 0.02 * T, atol=1e-21)
        np.testing.assert_allclose(low.y, M.tie(clk).y, atol=1e-22)


# --- AC#4: decision-1 --------------------------------------------------------

GOOD = _clock(1e-9 + K * T + ERR)


class TestFailures:
    @pytest.mark.parametrize('kwargs,text', [
        ({'units': 'ms'}, 'units'),
        ({'units': 'UI'}, 'units'),
        ({'reference': 'lsq'}, 'reference'),
        ({'ideal_period': 0.0}, 'ideal_period'),
        ({'ideal_period': -1e-9}, 'ideal_period'),
        ({'ideal_period': math.inf}, 'ideal_period'),
        ({'ideal_period': True}, 'ideal_period'),
        ({'ideal_period': '1n'}, 'ideal_period'),
        ({'phase': -math.inf}, 'phase'),
        ({'edge': 'rsing'}, 'edge'),
        ({'start': 5e-9, 'end': 2e-9}, 'end >= start'),
        ({'start': 1.0}, 'does not overlap'),
    ])
    @pytest.mark.parametrize('name', ['tie', 'long_term_jitter'])
    def test_caller_error_raises(self, name, kwargs, text):
        with pytest.raises(ValueError, match=text) as exc:
            getattr(M, name)(GOOD, **kwargs)
        assert name in str(exc.value)

    @pytest.mark.parametrize('n_edges', [0, 1, 2])
    def test_fewer_than_three_edges(self, n_edges):
        edges = 1e-9 + np.arange(n_edges) * T
        clk = (_sig([0, 1e-9], [0, 0]) if n_edges == 0 else _clock(edges))
        for name in ('tie', 'long_term_jitter'):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = getattr(M, name)(clk)
            assert len(caught) == 1
            d = caught[0].message
            assert isinstance(d, MetricWarning)
            assert (d.metric, d.code) == (name, 'insufficient_edges')
            assert d.details['edge_count'] == n_edges
            assert d.details['required_edge_count'] == 3
            if name == 'tie':
                assert len(result.x) == len(result.y) == 0
            else:
                assert math.isnan(result)

    def test_window_with_two_edges_is_insufficient(self):
        edges = 1e-9 + K * T
        with pytest.warns(MetricWarning, match='Too few edges'):
            r = M.long_term_jitter(_clock(edges), start=edges[3] - 0.1 * T,
                                   end=edges[4] + 0.1 * T)
        assert math.isnan(r)

    def test_nan_sample_is_unknown(self):
        y = GOOD.y.copy()
        y[5] = np.nan
        clk = _sig(GOOD.x, y)
        with pytest.warns(MetricWarning) as caught:
            tie = M.tie(clk, units='deg')
        assert caught[0].message.code == 'nan_waveform'
        assert len(tie.y) == 1 and np.isnan(tie.y).all()
        assert tie.ylabel == 'deg'

    def test_nan_period_passes_through(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert math.isnan(M.long_term_jitter(GOOD, ideal_period=math.nan))
            assert np.isnan(M.tie(GOOD, phase=math.nan).y).all()
