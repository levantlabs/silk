"""REVIEW-METRICS (TASK-34) — one analytic fixture per public metric.

Every public function in ``silk.metrics`` is
run against a SYNTHETIC signal with a closed-form answer (no simulator
output anywhere in this file). The tolerance of each assertion is the
metric's *stated resolution* — grid step, interpolation error bound,
sample-count statistics — so a fixture that passes here also proves the
docstring's resolution claim. Where the closed form is exact (piecewise-
linear signals with breakpoints on the grid, coherent FFT bins) the
tolerance is floating-point.

``test_every_public_metric_has_a_fixture`` at the bottom enumerates the
public surface mechanically and fails if a metric is added without a
fixture here.

"""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from silk.signals.base import Signal
import silk.metrics as M


def _load_fixture_module(name):
    path = Path(__file__).resolve().parent.parent / 'fixtures' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ZETA02 = _load_fixture_module('task326_zeta02_lowpass')


# ---------------------------------------------------------------------------
# Coverage registry (drives the completeness guard)
# ---------------------------------------------------------------------------

COVERED: set = set()


def covers(*names):
    """Mark a test as the analytic fixture for the named public metrics."""
    def deco(fn):
        COVERED.update(names)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------

def _tran(t, y, name='v', type_='voltage', unit='V'):
    return Signal(name, type_, 'time', unit,
                  np.asarray(t, dtype=float), np.asarray(y, dtype=float))


def _freq(f, y, name='h'):
    return Signal(name, 'voltage', 'frequency', 'V',
                  np.asarray(f, dtype=float), np.asarray(y))


def _code(codes, v):
    return Signal('tr', 'voltage', 'code', 'V',
                  np.asarray(codes, dtype=float), np.asarray(v, dtype=float))


P = 1e-9          # clock period used by the PWL clock fixtures
SPP = 40          # samples per period (P/4 breakpoints land on the grid)


def _pwl_clock(n_periods=16, duty=0.5, edge_jitter=None, t0=0.0):
    """Square clock, edges exactly on the sample grid. Rising edges at
    ``t0 + k·P`` (+ ``edge_jitter[k]``, itself a grid multiple)."""
    dt = P / SPP
    n = n_periods * SPP + 1
    t = np.arange(n) * dt
    y = np.zeros(n)
    for k in range(n_periods):
        j = 0.0 if edge_jitter is None else edge_jitter[k]
        r = int(round((t0 + k * P + j) / dt))
        f = int(round((t0 + k * P + j + duty * P) / dt))
        y[r:f] = 1.0
    return _tran(t, y)


def _triangle(n_periods=16):
    """Triangle wave, rising through 0 at exactly k·P, peaks ±1 at P/4."""
    dt = P / SPP
    t = np.arange(n_periods * SPP + 1) * dt
    phase = (t / P) % 1.0
    y = np.where(phase < 0.25, 4 * phase,
                 np.where(phase < 0.75, 2 - 4 * phase, -4 + 4 * phase))
    return _tran(t, y)


TCLK = 1e-9
N = 1024
FS = 1.0 / TCLK
DF = FS / N


def _tones(*tones, n=N):
    """Coherent multi-tone: ``tones = (bin, amplitude), ...``."""
    k = np.arange(n)
    y = np.zeros(n)
    for b, a in tones:
        y = y + a * np.sin(2 * np.pi * b * k / n)
    return _tran(k * TCLK, y)


# ===========================================================================
# transient
# ===========================================================================

class TestTransientEdges:

    @covers('crossings')
    def test_crossings_of_triangle_are_exact_multiples_of_period(self):
        cs = M.crossings(_triangle(), threshold=0.0, edge='rising')
        # convention: ``a < level <= b`` — the trace STARTING on the level at
        # t = 0 is not a crossing, the sample landing on it at 16·P is.
        assert len(cs) == 16
        assert np.allclose(cs, np.arange(1, 17) * P, rtol=1e-12, atol=0)

    @covers('rise_time')
    def test_rise_time_first_order_step(self):
        tau, dt = 1e-9, 1e-11
        t = np.arange(0, 40 * tau, dt)              # settled: swing = observed max − min
        y = 1.0 - np.exp(-t / tau)
        # closed form: τ·ln(9). Linear interpolation of a convex curve
        # errs by at most Δt²·|y''|/(8·|y'|) = Δt²/(8τ) per crossing.
        expected = tau * math.log(9.0)
        assert abs(M.rise_time(_tran(t, y)) - expected) <= 2 * dt ** 2 / (8 * tau)

    @covers('fall_time')
    def test_fall_time_first_order_decay(self):
        tau, dt = 1e-9, 1e-11
        t = np.arange(0, 40 * tau, dt)
        y = np.exp(-t / tau)
        expected = tau * math.log(9.0)
        assert abs(M.fall_time(_tran(t, y)) - expected) <= 2 * dt ** 2 / (8 * tau)

    @covers('slew_rate')
    def test_slew_rate_of_linear_ramp(self):
        T = 1e-9
        t = np.linspace(0, T, 101)
        y = np.clip(t / T, 0, 1)
        assert M.slew_rate(_tran(t, y)) == pytest.approx(1.0 / T, rel=1e-12)

    @covers('settling_time')
    def test_settling_time_first_order_is_grid_quantized_upwards(self):
        tau, dt = 1e-9, 1e-11
        t = np.arange(0, 12 * tau, dt)
        y = 1.0 - np.exp(-t / tau)
        ts = tau * math.log(100.0)               # 1 % band
        got = M.settling_time(_tran(t, y), initial=0.0, target=1.0, tol=0.01)
        # resolution: the first GRID point at/after the true instant
        assert ts <= got < ts + dt

    @covers('propagation_delay')
    def test_propagation_delay_of_shifted_ramps(self):
        dt = 1e-11
        t = np.arange(0, 5e-9, dt)
        ramp = lambda t0: np.clip((t - t0) / 1e-9, 0, 1)
        d = 37 * dt
        got = M.propagation_delay(_tran(t, ramp(1e-9)), _tran(t, ramp(1e-9 + d)))
        assert got == pytest.approx(d, rel=1e-9, abs=0)

    @covers('propagation_delays')
    def test_propagation_delays_of_a_shifted_clock(self):
        # Every rising input edge paired with the next rising output edge,
        # one period-aligned 37-sample delay each (TASK-363).
        dt = 1e-11
        k = np.arange(800)                       # 100 samples per half period
        t = k * dt
        clk = lambda k0: (((k - k0) // 100 % 2 == 1) & (k >= k0)).astype(float)
        d = 37 * dt
        got = M.propagation_delays(_tran(t, clk(0)), _tran(t, clk(37)))
        assert len(got) == 4
        assert got == pytest.approx([d] * 4, rel=1e-9, abs=0)

    @covers('rise_times', 'fall_times', 'slew_rates')
    def test_edge_plurals_of_known_ramps(self):
        # 1 V ramps of 0.5 s and 0.2 s: 10-90 % takes 80 % of each (TASK-372).
        t = np.linspace(0, 10, 4001)
        y = np.interp(t, [1, 1.5, 3, 3.2, 5, 5.5, 7, 7.2],
                      [0, 1, 1, 0, 0, 1, 1, 0])
        sig = _tran(t, y)
        assert M.rise_times(sig) == pytest.approx([0.4, 0.4], rel=1e-9)
        assert M.fall_times(sig) == pytest.approx([0.16, 0.16], rel=1e-9)
        assert M.slew_rates(sig, edge='any') == pytest.approx(
            [2.0, 5.0, 2.0, 5.0], rel=1e-9)

    @covers('overshoot', 'undershoot')
    def test_overshoot_second_order_step(self):
        zeta, wn = 0.5, 2 * np.pi * 1e6
        t = np.linspace(0, 10 / wn, 200001)
        wd = wn * math.sqrt(1 - zeta ** 2)
        y = 1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / math.sqrt(1 - zeta ** 2) * np.sin(wd * t))
        mp = math.exp(-math.pi * zeta / math.sqrt(1 - zeta ** 2))     # 0.1630
        # the sampled peak sits within ½·|y''|·(Δt/2)² of the true peak
        assert M.overshoot(_tran(t, y), initial=0.0, target=1.0) == pytest.approx(mp, rel=1e-5)
        # post-transition undershoot (IEEE 181-2003) is the NEXT reversal
        # after the trace first reaches target, one half-cycle later: each
        # successive extremum decays by another factor of mp, so the first
        # post-crossing trough is mp**2, not mp (rlc_step.py derives the
        # same closed form for its own undershoot reading).
        assert M.undershoot(_tran(t, -y), initial=0.0, target=-1.0) == pytest.approx(mp ** 2, rel=1e-5)

    @covers('average_duty')
    def test_duty_cycle_pwl_30pct(self):
        assert M.average_duty(_pwl_clock(duty=0.3)) == pytest.approx(0.3, rel=1e-12)

    @covers('average_period')
    def test_period_of_triangle(self):
        assert M.average_period(_triangle(), threshold=0.0) == pytest.approx(P, rel=1e-12, abs=0)

    @covers('average_frequency')
    def test_frequency_of_triangle(self):
        assert M.average_frequency(_triangle(), threshold=0.0) == pytest.approx(1.0 / P, rel=1e-12, abs=0)

    @covers('pulse_width')
    def test_pulse_width_pwl(self):
        assert M.pulse_width(_pwl_clock(duty=0.3)) == pytest.approx(0.3 * P, rel=1e-12, abs=0)

    @covers('pulse_widths')
    def test_pulse_widths_pwl(self):
        clk = _pwl_clock(duty=0.3)
        assert np.allclose(M.pulse_widths(clk), 0.3 * P, rtol=1e-12, atol=0)
        assert np.allclose(M.pulse_widths(clk, polarity='negative'), 0.7 * P,
                           rtol=1e-12, atol=0)

    @covers('cycle_duty')
    def test_cycle_duty_pwl(self):
        s = M.cycle_duty(_pwl_clock(duty=0.3))
        assert len(s.y) == 14  # the edge at t = 0 is the trace's start, not a crossing
        assert np.allclose(s.y, 0.3, rtol=1e-12)

    def _jittered_edges(self):
        dt = P / SPP
        delta = 2 * dt
        jit = np.array([delta * (-1) ** k for k in range(16)])
        return _pwl_clock(edge_jitter=jit), np.arange(16) * P + jit

    @covers('period_jitter', 'period_jitter_pp')
    def test_period_jitter_alternating_edges(self):
        # TASK-417: RMS (not std) of each period's deviation from the mean.
        clk, edges = self._jittered_edges()
        periods = np.diff(edges)
        dev = periods - periods.mean()
        expected = float(np.sqrt(np.mean(dev ** 2)))
        assert M.period_jitter(clk) == pytest.approx(expected, rel=1e-9, abs=0)
        assert M.period_jitter_pp(clk) == pytest.approx(np.ptp(periods),
                                                        rel=1e-9, abs=0)

    @covers('cycle_to_cycle_jitter', 'cycle_to_cycle_jitter_pp')
    def test_c2c_jitter_alternating_edges(self):
        # TASK-417: a true RMS of the adjacent-period differences.
        clk, edges = self._jittered_edges()
        d = np.diff(np.diff(edges))
        expected = float(np.sqrt(np.mean(d ** 2)))
        assert M.cycle_to_cycle_jitter(clk) == pytest.approx(expected, rel=1e-9, abs=0)
        assert M.cycle_to_cycle_jitter_pp(clk) == pytest.approx(
            np.ptp(d), rel=1e-9, abs=0)

    @covers('period_deviation', 'cycle_to_cycle_difference')
    def test_jitter_waveforms_alternating_edges(self):
        clk, edges = self._jittered_edges()
        periods = np.diff(edges)
        dev = M.period_deviation(clk)
        np.testing.assert_allclose(dev.y, periods - periods.mean(),
                                   rtol=0, atol=1e-9 * P)
        c2c = M.cycle_to_cycle_difference(clk)
        # Stamped at the shared edge: every crossing but the first and last
        # (each sits half a sample before its grid step, the interpolation
        # offset the std-based tests cancel).
        np.testing.assert_allclose(c2c.x, M.crossings(clk)[1:-1], rtol=0,
                                   atol=1e-9 * P)
        np.testing.assert_allclose(c2c.x, edges[1:-1] - P / SPP / 2, rtol=0,
                                   atol=1e-9 * P)
        np.testing.assert_allclose(c2c.y, np.diff(periods), rtol=0,
                                   atol=1e-9 * P)

    @covers('crossing_jitter_rms')
    def test_crossing_jitter_alternating_edges(self):
        # Rising edges at k·P ± delta: each is delta off the grid, so the
        # result is std(jit, ddof=1); the constant half-sample interpolation
        # offset of every grid-aligned step cancels in the std.
        clk, edges = self._jittered_edges()
        expected = float(np.std(edges - np.arange(16) * P, ddof=1))
        assert M.crossing_jitter_rms(clk, period=P) == pytest.approx(expected, rel=1e-9, abs=0)


class TestTransientPowerAndComparator:

    @covers('average_power')
    def test_average_power_sinusoidal_with_phase(self):
        V0, I0, phi, f = 1.0, 2e-3, math.pi / 3, 1e6
        t = np.linspace(0, 10 / f, 10 * 64 + 1)   # 10 whole cycles, 64 pts each
        v = _tran(t, V0 * np.sin(2 * np.pi * f * t))
        i = _tran(t, I0 * np.sin(2 * np.pi * f * t + phi), 'i', 'current', 'A')
        # trapezoid is spectrally exact for a band-limited periodic integrand
        assert M.average_power(v, i) == pytest.approx(0.5 * V0 * I0 * math.cos(phi), rel=1e-9)

    @covers('dynamic_energy_per_cycle')
    def test_energy_per_cycle_constant_power_whole_cycles(self):
        period = 1e-9
        t = np.linspace(0, 8 * period, 8001)
        v = _tran(t, 1.8 * np.ones_like(t))
        i = _tran(t, 1e-3 * np.ones_like(t), 'i', 'current', 'A')
        assert M.dynamic_energy_per_cycle(v, i, period) == pytest.approx(1.8e-3 * period, rel=1e-9, abs=0)

    def _decision(self, t, t_trip, dt):
        """Digital output crossing mid-rail EXACTLY at t_trip (PWL over one Δt each side)."""
        return np.clip((t - (t_trip - dt)) / (2 * dt), 0, 1)

    @covers('hysteresis')
    def test_hysteresis_from_rising_and_falling_sweeps(self):
        dt = 1e-3
        t = np.arange(0, 1 + dt / 2, dt)
        up, down = _tran(t, t), _tran(t, 1 - t)
        out_up = _tran(t, self._decision(t, 0.6, dt))            # trips at v = 0.6
        out_down = _tran(t, 1 - self._decision(t, 0.6, dt))      # releases at v = 0.4
        assert M.hysteresis(up, out_up, down, out_down) == pytest.approx(0.2, rel=1e-9)

    @covers('propagation_delay')
    def test_propagation_delay_analog_in_digital_out(self):
        # The comparator case comparator_prop_delay used to cover (TASK-363):
        # analog input ramp, digital output, each side its own mid-swing.
        dt = 1e-11
        t = np.arange(0, 5e-9, dt)
        v_in = _tran(t, np.clip((t - 1e-9) / 1e-9, 0, 1))        # mid-swing at 1.5 ns
        out = _tran(t, self._decision(t, 1.5e-9 + 123 * dt, dt))
        assert M.propagation_delay(v_in, out) == pytest.approx(123 * dt, rel=1e-9, abs=0)


class TestWaveformPrimitives:

    @covers('average')
    def test_average_of_linear_ramp(self):
        t = np.linspace(0, 2.0, 11)
        assert M.average(_tran(t, 1.0 + 3.0 * t)) == pytest.approx(1.0 + 3.0, rel=1e-12)

    @covers('rms', 'stddev', 'variance')
    def test_rms_and_stddev_of_offset_sine(self):
        A, f = 0.7, 1e6
        t = np.linspace(0, 4 / f, 4 * 128 + 1)
        y = 2.0 + A * np.sin(2 * np.pi * f * t)
        assert M.rms(_tran(t, y)) == pytest.approx(math.sqrt(4.0 + A ** 2 / 2), rel=1e-9)
        assert M.stddev(_tran(t, y)) == pytest.approx(A / math.sqrt(2), rel=1e-9)
        assert M.variance(_tran(t, y)) == pytest.approx(A * A / 2, rel=1e-9)

    @covers('peak', 'peak_to_peak', 'minimum', 'maximum')
    def test_extrema(self):
        t = np.linspace(0, 1, 5)
        s = _tran(t, [0.0, 2.0, -3.0, 1.0, 0.0])
        assert M.peak(s) == 3.0
        assert M.peak_to_peak(s) == 5.0
        assert M.minimum(s) == -3.0
        assert M.maximum(s) == 2.0
        assert M.maximum(s, start=0.5, end=1.0) == 1.0

    @covers('argmin_x', 'argmax_x')
    def test_arg_extrema_of_parabola(self):
        t = np.linspace(-1, 1, 201)                 # vertex at t = 0.3 is on the grid
        assert M.argmax_x(_tran(t, -(t - 0.3) ** 2)) == pytest.approx(0.3, abs=1e-15)
        assert M.argmin_x(_tran(t, (t - 0.3) ** 2)) == pytest.approx(0.3, abs=1e-15)

    @covers('integral')
    def test_integral_of_t_squared_within_trapezoid_bound(self):
        dt = 1e-2
        t = np.arange(0, 1 + dt / 2, dt)
        got = M.integral(_tran(t, t ** 2))
        # trapezoid error bound: Δt²·(b−a)·max|y''|/12 = Δt²/6
        assert abs(got - 1 / 3) <= dt ** 2 / 6 * (1 + 1e-9)   # bound is exact for a quadratic
        assert got > 1 / 3                          # and convex ⇒ overestimates

    @covers('running_integral')
    def test_running_integral_endpoint_and_start(self):
        dt = 1e-2
        t = np.arange(0, 1 + dt / 2, dt)
        r = M.running_integral(_tran(t, t ** 2))
        assert r.y[0] == 0.0
        assert abs(r.y[-1] - 1 / 3) <= dt ** 2 / 6 * (1 + 1e-9)
        assert r.xlabel == 'time'

    @covers('moving_average')
    def test_moving_average_of_ramp_lags_by_half_window(self):
        t = np.linspace(0, 10, 1001)
        r = M.moving_average(_tran(t, 2 * t), window=1.0)
        ok = t >= 1.0
        assert np.allclose(r.y[ok], 2 * (t[ok] - 0.5))
        assert np.isnan(r.y[t < 1 - 1e-9]).all()

    @covers('derivative')
    def test_derivative_of_quadratic_is_exact_in_the_interior(self):
        t = np.linspace(0, 1, 101)
        d = M.derivative(_tran(t, t ** 2))
        assert np.allclose(d.y[1:-1], 2 * t[1:-1], rtol=1e-12, atol=1e-14)
        # endpoints are one-sided differences: first-order, error = Δt
        assert abs(d.y[0] - 0.0) == pytest.approx(0.01, rel=1e-9)

    @covers('final_value')
    def test_final_value_is_the_last_sample(self):
        s = _tran(np.array([0.0, 1.0, 2.0, 3.0]), [0.0, 2.0, 0.0, 1.5])
        assert M.final_value(s) == 1.5

    @covers('value_at', 'crossing')
    def test_value_at_and_crossing_on_pwl(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        s = _tran(t, [0.0, 2.0, 0.0, 2.0])
        assert M.value_at(s, 0.25) == pytest.approx(0.5, rel=1e-15)
        assert M.crossing(s, 1.0, edge='rising', n=2) == pytest.approx(2.5, rel=1e-15)
        assert M.crossing(s, 1.0, edge='falling') == pytest.approx(1.5, rel=1e-15)


class TestTier2Transient:

    @covers('cycle_frequency')
    def test_cycle_frequency_of_triangle(self):
        s = M.cycle_frequency(_triangle())
        assert len(s.y) == 15
        assert np.allclose(s.y, 1.0 / P, rtol=1e-12)

    @covers('cycle_period')
    def test_cycle_period_of_triangle(self):
        s = M.cycle_period(_triangle())
        assert len(s.y) == 15
        assert np.allclose(s.y, P, rtol=1e-12)

    @covers('on_time', 'off_time')
    def test_on_off_time_of_30pct_clock(self):
        clk = _pwl_clock(n_periods=16, duty=0.3)
        dt = P / SPP
        total = float(clk.x[-1] - clk.x[0])
        # PWL edges ON the grid: the indicator's trapezoid smears each edge by
        # ±Δt/2, so the answer is (samples-high)·Δt — within Δt per edge.
        assert abs(M.on_time(clk, 0.5) - 0.3 * 16 * P) <= 16 * dt
        assert M.on_time(clk, 0.5) + M.off_time(clk, 0.5) == pytest.approx(total, rel=1e-12)

    @covers('histogram', 'HistogramResult')
    def test_histogram_of_triangle_is_uniform(self):
        # A symmetric triangle sweeps every value at the same rate.
        h = M.histogram(_triangle(), bins=5)
        assert isinstance(h, M.HistogramResult)
        assert np.allclose(h.fractions, 0.2, rtol=1e-12)

    # --- BEGIN commented-out (TASK-226.11): eye_diagram analytic fixture,
    # superseded by TASK-226.12+. Kept for reference until the replacement
    # is accepted; do not delete. ---
    # @covers('eye_diagram', 'EyeResult')
    # def test_eye_of_clean_alternating_pattern(self):
    #     # 1010… pattern: UI = P/2; edges land P/8 into each UI window so the
    #     # window centre (P/4) reads a clean 1 or 0 and the crossings are inside.
    #     clk = _pwl_clock(n_periods=32, duty=0.5, t0=P / 8)
    #     eye = M.eye_diagram(clk, period=P / 2, samples_per_period=SPP)
    #     assert isinstance(eye, M.EyeResult)
    #     assert eye.height == pytest.approx(1.0, rel=1e-12)
    #     assert eye.width == pytest.approx(P / 2, rel=1e-9, abs=0)      # all crossings coincide
    #     assert eye.jitter_rms == pytest.approx(0.0, abs=1e-20)
    #     assert eye.traces.shape == (64, SPP)
    #     assert eye.time_resolution_s == pytest.approx(P / 2 / SPP, rel=1e-12, abs=0)
    # --- END commented-out (TASK-226.11) ---

    @covers('EyeResult')
    def test_eye_result_fields_and_their_defaults(self):
        """``EyeResult`` is public in its own right, so its shape is pinned
        by constructing one directly — the producer fills only ``traces``
        (TASK-226.12), and this catches a change to the fields it does not
        fill yet."""
        import dataclasses
        # TASK-226.21: the dead placeholders are gone; the field list is the
        # contract.
        assert [f.name for f in dataclasses.fields(M.EyeResult)] == [
            'height', 'width', 'traces', 'sampling_instant_s',
            'levels', 'heights', 'widths']
        r = M.EyeResult(height=1.0, width=2.0, traces=[])
        assert r.height == 1.0 and r.width == 2.0
        assert r.traces == []
        assert math.isnan(r.sampling_instant_s)
        assert r.levels == r.heights == r.widths == ()

    @covers('eye_diagram')
    def test_eye_fold_of_a_clean_clock_is_the_trace_re_indexed(self):
        """The fold in closed form. A 50 % clock folded at its OWN period
        gives 16 windows that together hold every one of the 640 samples in
        the covered span, each still carrying its own time and value — not
        a resampling of them. Nothing is measured: the scalar fields are
        placeholders (TASK-226.12).

        The windows are the same PICTURE but not the same NUMBERS, and the
        counts say so: 40, 40, 41, 39, … . ``_pwl_clock`` lays its samples
        down as one ``arange``, so a sample nominally on a period boundary
        can be an ulp below it and belongs to the earlier window. The old
        implementation could not show this — it resampled every window onto
        one 128-point grid, which made all 16 identical by construction.
        Here the ulp survives, and the fold stays exactly reversible."""
        clk = _pwl_clock(n_periods=16, duty=0.5)
        dt = P / SPP
        r = M.eye_diagram(clk, period=P)

        assert isinstance(r, M.EyeResult)
        assert len(r.traces) == 16                     # whole periods only
        assert [len(w.x) for w in r.traces] == [40, 40, 41, 39, 40, 41, 39,
                                                40, 41, 39, 40, 41, 39, 40,
                                                40, 40]
        pos = 0
        for k, w in enumerate(r.traces):
            n = len(w.x)
            # window k IS the original samples, undone by adding its own
            # origin back — exactly, not to a tolerance
            assert np.array_equal(np.asarray(w.x) + k * P,
                                  np.asarray(clk.x)[pos:pos + n])
            assert np.array_equal(np.asarray(w.y),
                                  np.asarray(clk.y)[pos:pos + n])
            assert 0.0 <= float(np.min(w.x)) and float(np.max(w.x)) < P
            # every phase is a grid multiple (to within an ulp) — the
            # window that gave its opening sample away starts at dt, not 0
            steps = np.asarray(w.x) / dt
            assert np.allclose(steps, np.round(steps), rtol=0.0, atol=1e-9)
            pos += n
        assert pos == 16 * SPP                         # every covered sample
        assert math.isnan(r.height) and math.isnan(r.width)
        assert math.isnan(r.sampling_instant_s)

    @covers('error_norm')
    def test_error_norms_against_offset_goal(self):
        t = np.linspace(0, 1, 11)
        goal = _tran(t, 2.0 * np.ones_like(t))
        sig = _tran(t, 2.1 * np.ones_like(t))
        assert M.error_norm(sig, goal, kind='max') == pytest.approx(0.1, rel=1e-12)
        assert M.error_norm(sig, goal, kind='rms') == pytest.approx(0.1, rel=1e-12)

    @covers('relative_error_norm')
    def test_relative_error_norm_against_offset_goal(self):
        t = np.linspace(0, 1, 11)
        goal = _tran(t, 2.0 * np.ones_like(t))
        sig = _tran(t, 2.1 * np.ones_like(t))
        for kind in ('max', 'rms', 'mean_abs'):
            assert M.relative_error_norm(sig, goal, kind=kind, floor=1.0) \
                == pytest.approx(0.05, rel=1e-12)

    @covers('convolve')
    def test_rect_convolved_with_rect_is_triangle(self):
        dx, n = 1e-3, 100
        x = np.arange(n) * dx
        box = _tran(x, np.ones(n))
        c = M.convolve(box, box)
        assert len(c.y) == 2 * n - 1
        assert c.y.max() == pytest.approx(n * dx, rel=1e-12)          # width W = n·dx
        assert float(np.sum(c.y) * dx) == pytest.approx((n * dx) ** 2, rel=1e-12)


class TestLeftoversAndBeyondVendor:

    @covers('lshift')
    def test_lshift(self):
        s = M.lshift(_tran([0.0, 1.0], [1.0, 2.0]), 0.25)
        assert np.allclose(s.x, [-0.25, 0.75])

    @covers('rshift')
    def test_rshift(self):
        s = M.rshift(_tran([0.0, 1.0], [1.0, 2.0]), 0.25)
        assert np.allclose(s.x, [0.25, 1.25])

    @covers('window')
    def test_window(self):
        s = M.window(_tran(np.arange(10.0), np.arange(10.0)), 2.0, 5.0)
        assert list(s.x) == [2.0, 3.0, 4.0, 5.0]

    @covers('clip')
    def test_clip(self):
        s = M.clip(_tran(np.arange(10.0), np.arange(10.0)), low=2.0, high=5.0)
        assert list(s.y) == [2, 2, 2, 3, 4, 5, 5, 5, 5, 5] and list(s.x) == list(range(10))

    @covers('flip')
    def test_flip(self):
        s = M.flip(_tran([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]))
        assert list(s.y) == [3.0, 2.0, 1.0] and list(s.x) == [0.0, 1.0, 2.0]

    @covers('sample')
    def test_sample_linear_is_exact(self):
        s = M.sample(_tran([0.0, 1.0], [0.0, 2.0]), n=5)
        assert np.allclose(s.y, [0.0, 0.5, 1.0, 1.5, 2.0])

    @covers('tie')
    def test_tie_of_alternating_edges(self):
        dt = P / SPP
        delta = 2 * dt
        jit = np.array([delta * (-1) ** k for k in range(16)])
        clk = _pwl_clock(edge_jitter=jit)
        # Least squares with the slope fixed (TASK-418): the intercept is
        # the mean offset, 0 for 16 alternating errors, so TIE_k = δ(−1)^k.
        tie = M.tie(clk, ideal_period=P)
        assert np.allclose(tie.y, jit, rtol=0, atol=1e-22)
        # The pre-418 reference, anchored on edge[0] (= +δ):
        # TIE_k = δ((−1)^k − 1) = 0, −2δ, 0, −2δ …
        tie = M.tie(clk, ideal_period=P, reference='endpoints')
        expected = delta * ((-1.0) ** np.arange(16) - 1.0)
        assert np.allclose(tie.y, expected, rtol=0, atol=1e-22)

    @covers('long_term_jitter')
    def test_long_term_jitter_of_alternating_edges(self):
        dt = P / SPP
        delta = 2 * dt
        jit = np.array([delta * (-1) ** k for k in range(16)])
        expected = float(np.std(delta * ((-1.0) ** np.arange(16) - 1.0), ddof=1))
        got = M.long_term_jitter(_pwl_clock(edge_jitter=jit), ideal_period=P)
        assert got == pytest.approx(expected, rel=1e-9, abs=0)

    @covers('allan_deviation')
    def test_allan_deviation_alternating_periods(self):
        dt = P / SPP
        delta = 2 * dt
        jit = np.array([delta * (-1) ** k for k in range(32)])
        ad = M.allan_deviation(_pwl_clock(n_periods=32, edge_jitter=jit),
                               taus=np.array([P]))
        # periods alternate P∓2δ ⇒ y_i = ∓2δ/⟨T⟩ ⇒ σ_y²(⟨T⟩) = ½⟨(y_{i+1}−y_i)²⟩ = 8δ²/⟨T⟩²
        # (⟨T⟩ ≠ P: 31 periods, 16 of one sign — the metric normalises by the
        # OBSERVED mean period, exactly as its docstring states)
        t_avg = float(np.mean(np.diff(np.arange(32) * P + jit)))
        assert ad.xlabel == 'tau'
        assert ad.y[0] == pytest.approx(2 * math.sqrt(2) * delta / t_avg, rel=1e-9)

    @covers('cross_correlation')
    def test_cross_correlation_peak_locates_the_delay(self):
        dx, n, d = 1e-3, 64, 7
        x = np.arange(n) * dx
        b = np.zeros(n); b[10] = 1.0                 # reference impulse
        a = np.zeros(n); a[10 + d] = 1.0             # same impulse, delayed d samples
        c = M.cross_correlation(_tran(x, a), _tran(x, b))
        assert c.xlabel == 'lag'
        assert c.x[int(np.argmax(c.y))] == pytest.approx(d * dx, rel=1e-12)


# ===========================================================================
# spectral
# ===========================================================================

class TestSpectralCoherent:
    A, A2, A3, AN = 1.0, 1e-2, 3e-3, 1e-4     # fundamental, h2, h3, non-harmonic tone
    MB = 11                                   # fundamental bin

    def _distorted(self):
        return _tones((self.MB, self.A), (2 * self.MB, self.A2),
                      (3 * self.MB, self.A3), (97, self.AN))

    @covers('fft')
    def test_fft_bin_power_of_coherent_tone(self):
        spec = M.fft(_tones((self.MB, self.A)), tclk=TCLK, n_samples=N)
        assert spec.xlabel == 'frequency'
        assert spec.x[self.MB] == pytest.approx(self.MB * DF)
        assert spec.y[self.MB] == pytest.approx((self.A * N / 2) ** 2, rel=1e-12)
        assert spec.y[self.MB + 1] < 1e-20                     # coherent: no leakage

    @covers('sndr', 'sinad')
    def test_sndr_equals_signal_over_all_other_tones(self):
        s = self._distorted()
        nd = self.A2 ** 2 + self.A3 ** 2 + self.AN ** 2
        expected = 10 * math.log10(self.A ** 2 / nd)
        assert M.sndr(s, tclk=TCLK, n_samples=N) == pytest.approx(expected, abs=1e-9)
        assert M.sinad(s, tclk=TCLK, n_samples=N) == pytest.approx(expected, abs=1e-9)

    @covers('snr')
    def test_snr_excludes_harmonics(self):
        s = self._distorted()
        expected = 20 * math.log10(self.A / self.AN)
        assert M.snr(s, tclk=TCLK, n_samples=N) == pytest.approx(expected, abs=1e-6)   # FP leakage floor

    @covers('enob')
    def test_enob_formula(self):
        s = self._distorted()
        nd = self.A2 ** 2 + self.A3 ** 2 + self.AN ** 2
        expected = (10 * math.log10(self.A ** 2 / nd) - 1.76) / 6.02
        assert M.enob(s, tclk=TCLK, n_samples=N) == pytest.approx(expected, abs=1e-9)

    @covers('sfdr', 'SFDR')
    def test_sfdr_is_the_h2_and_reports_bin_resolution(self):
        r = M.sfdr(self._distorted(), tclk=TCLK, n_samples=N)
        assert isinstance(r, M.SFDR)
        assert r.sfdr_db == pytest.approx(20 * math.log10(self.A / self.A2), abs=1e-9)
        assert r.fundamental_hz == pytest.approx(self.MB * DF)
        assert r.spur_hz == pytest.approx(2 * self.MB * DF)
        assert r.bin_hz == pytest.approx(DF)          # frequencies are quantized to one bin

    @covers('thd', 'thd_percent')
    def test_thd_h2_h3(self):
        s = self._distorted()
        ratio = (self.A2 ** 2 + self.A3 ** 2) / self.A ** 2
        assert M.thd(s, tclk=TCLK, n_samples=N) == pytest.approx(10 * math.log10(ratio), abs=1e-9)
        assert M.thd_percent(s, tclk=TCLK, n_samples=N) == pytest.approx(100 * math.sqrt(ratio), rel=1e-9)

    @covers('noise_floor_dbfs')
    def test_noise_floor_deterministic_single_tone_floor(self):
        s = _tones((self.MB, self.A), (97, self.AN))
        # 513 rfft bins − DC − fundamental = 511 bins; all power sits in one of them
        expected = 10 * math.log10(self.AN ** 2 / (511 * self.A ** 2))
        assert M.noise_floor_dbfs(s, tclk=TCLK, n_samples=N) == pytest.approx(expected, abs=1e-9)

    @covers('harmonic_levels_db')
    def test_harmonic_levels(self):
        lv = M.harmonic_levels_db(self._distorted(), tclk=TCLK, n_samples=N, n_harmonics=4)
        assert lv[0] == pytest.approx(20 * math.log10(self.A2 / self.A), abs=1e-9)
        assert lv[1] == pytest.approx(20 * math.log10(self.A3 / self.A), abs=1e-9)
        # h4 absent: an empty bin bottoms at the eps floor (TASK-52),
        # 10*log10(eps / (A*N/2)^2) = -210.7 dB here — not a measurement.
        assert lv[2] < -200.0

    @covers('imd3_db')
    def test_imd3_two_tone_with_known_products(self):
        b1, b2, a = 40, 43, 1e-3
        s = _tones((b1, 1.0), (b2, 1.0), (2 * b1 - b2, a), (2 * b2 - b1, a))
        got = M.imd3_db(s, tclk=TCLK, f1_hz=b1 * DF, f2_hz=b2 * DF, n_samples=N)
        assert got == pytest.approx(20 * math.log10(a), abs=1e-9)

    @covers('oip3_dbm', 'iip3_dbm')
    def test_ip3_dbm_with_known_products(self):
        # 1 V tones into 50 ohm = +10 dBm; IM3 at -60 dBc -> OIP3 = 40 dBm.
        # A 0.1 V input tone is -10 dBm, gain 20 dB -> IIP3 = 20 dBm.
        b1, b2, a = 40, 43, 1e-3
        out = _tones((b1, 1.0), (b2, 1.0), (2 * b1 - b2, a), (2 * b2 - b1, a))
        kw = dict(tclk=TCLK, f1_hz=b1 * DF, f2_hz=b2 * DF, n_samples=N)
        assert M.oip3_dbm(out, **kw) == pytest.approx(40.0, abs=1e-9)
        assert M.iip3_dbm(out, _tones((b1, 0.1), (b2, 0.1)), **kw) \
            == pytest.approx(20.0, abs=1e-9)

    @covers('harmonic')
    def test_harmonic_complex_bin(self):
        X1 = M.harmonic(_tones((self.MB, self.A)), 1, tclk=TCLK, n_samples=N)
        # rfft of A·sin(2πmn/N) at bin m is −j·A·N/2
        assert X1 == pytest.approx(-1j * self.A * N / 2, rel=1e-12)

    @covers('mix_product')
    def test_mix_product_picks_the_named_bin(self):
        b1, b2, a = 40, 43, 1e-3
        s = _tones((b1, 1.0), (b2, 1.0), (2 * b1 - b2, a))
        X = M.mix_product(s, 2, -1, f1_hz=b1 * DF, f2_hz=b2 * DF, tclk=TCLK, n_samples=N)
        assert X == pytest.approx(-1j * a * N / 2, rel=1e-9)

    @covers('harmonic_freqs')
    def test_harmonic_freqs_true_and_observed(self):
        # TASK-79: true k·f0 for all n (600·DF > Nyquist is still H6's
        # frequency); observed=True gives the folded bin centre instead.
        hf = M.harmonic_freqs(_tones((100, 1.0)), tclk=TCLK, n_samples=N, n=10)
        assert np.allclose(hf, 100 * DF * np.arange(1, 11))
        obs = M.harmonic_freqs(_tones((100, 1.0)), tclk=TCLK, n_samples=N, n=10, observed=True)
        # k=6..10: nominal 600/700/800/900/1000 -> mirrored 424/324/224/124/24
        assert np.allclose(obs, DF * np.array([100, 200, 300, 400, 500, 424, 324, 224, 124, 24]))


class TestSpectralNonCoherentWindowed:
    """A non-coherent tone, offset from zero, in white noise (TASK-436):
    with a window and guard bins the ratio is the closed-form
    ``(A²/2)/σ²`` — DC's window lobe is neither noise nor the spur."""
    A, SIGMA, OFFSET, CYCLES, NN = 0.5, 5e-4, 0.3, 100.5, 4096

    def _record(self, seed):
        k = np.arange(self.NN)
        y = (self.OFFSET + self.A * np.sin(2 * np.pi * self.CYCLES * k / self.NN + 0.3)
             + np.random.default_rng(seed).normal(0, self.SIGMA, self.NN))
        return _tran(k * TCLK, y)

    @pytest.mark.parametrize('seed', range(3))
    def test_blackman_guarded_snr_sndr_match_the_closed_form(self, seed):
        expected = 10 * math.log10(self.A ** 2 / 2 / self.SIGMA ** 2)
        kw = dict(tclk=TCLK, n_samples=self.NN, window='blackman', guard_bins=10)
        s = self._record(seed)
        # Statistical resolution ≈ 4.3/√(bins) dB; 0.5 dB is > 4σ here.
        assert M.snr(s, **kw) == pytest.approx(expected, abs=0.5)
        assert M.sndr(s, **kw) == pytest.approx(expected, abs=0.5)
        assert M.sfdr(s, **kw).spur_hz > 12 / (self.NN * TCLK)   # not DC's band


class TestSpectralDensity:
    A, MB = 1.0, 11

    @covers('psd')
    def test_psd_rectangular_coherent_tone(self):
        p = M.psd(_tones((self.MB, self.A)), tclk=TCLK, n_samples=N, window='rectangular')
        assert p.ylabel == 'V^2/Hz'
        # 2·(A·N/2)² / (fs·N) = A²N/(2fs)
        assert p.y[self.MB] == pytest.approx(self.A ** 2 * N / (2 * FS), rel=1e-12)
        assert float(np.sum(p.y) * DF) == pytest.approx(self.A ** 2 / 2, rel=1e-9)   # Parseval

    @covers('spectral_power')
    def test_band_power_rect_and_hann(self):
        s = _tones((self.MB, self.A))
        lo, hi = (self.MB - 2) * DF, (self.MB + 2) * DF
        rect = M.spectral_power(s, lo, hi, tclk=TCLK, n_samples=N, window='rectangular')
        hann = M.spectral_power(s, lo, hi, tclk=TCLK, n_samples=N, window='hann')
        assert rect == pytest.approx(self.A ** 2 / 2, rel=1e-9)
        assert hann == pytest.approx(self.A ** 2 / 2, rel=1e-6)    # 3-bin main lobe, Σw² normalised (np.hanning is symmetric ⇒ ~1e-7 leakage)

    @covers('welch_psd')
    def test_welch_single_segment_reduces_to_periodogram(self):
        s = _tones((self.MB, self.A))
        w = M.welch_psd(s, tclk=TCLK, n_samples=N, segments=1, overlap=0.0, window='rectangular')
        assert w.y[self.MB] == pytest.approx(self.A ** 2 * N / (2 * FS), rel=1e-12)

    def test_welch_white_noise_level_statistical(self):
        sigma = 1e-3
        rng = np.random.default_rng(7)
        s = _tran(np.arange(8192) * TCLK, sigma * rng.standard_normal(8192))
        w = M.welch_psd(s, tclk=TCLK, n_samples=8192, segments=16, overlap=0.5)
        # one-sided white PSD = 2σ²/fs; 16 segments of ~900 bins ⇒ few-% statistics
        assert float(np.mean(w.y[1:-1])) == pytest.approx(2 * sigma ** 2 / FS, rel=0.05)

    @covers('coherence')
    def test_coherence_of_linearly_related_signals_is_one(self):
        rng = np.random.default_rng(3)
        a = _tran(np.arange(N) * TCLK, rng.standard_normal(N))
        b = _tran(a.x, 3.0 * a.y)
        c = M.coherence(a, b, tclk=TCLK, n_samples=N)
        assert np.allclose(c.y, 1.0, atol=1e-9)

    @covers('evm_constellation', 'EVMResult')
    def test_evm_of_scaled_qpsk_is_exact(self):
        ideals = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / math.sqrt(2)
        sym, n_sym, spp = 1e-9, 64, 20
        rng = np.random.default_rng(1)
        tx = ideals[rng.integers(0, 4, n_sym)] * 1.1              # 10 % radial error
        t = np.arange(n_sym * spp) * (sym / spp)
        i = _tran(t, np.repeat(tx.real, spp))
        q = _tran(t, np.repeat(tx.imag, spp))
        r = M.evm_constellation(i, q, symbol_period=sym, ideal_points=ideals, n_symbols=n_sym)
        assert isinstance(r, M.EVMResult)
        assert r.evm_percent == pytest.approx(10.0, rel=1e-9)


# ===========================================================================
# ac  (silk.metrics.ac)
# ===========================================================================

class TestAcMetrics:
    A0, FP = 100.0, 1e4
    PPD = 200                                        # points per decade
    R = 10 ** (1.0 / PPD)                            # grid ratio: 1.16 % per point

    def _single_pole(self, ppd=None):
        ppd = ppd or self.PPD
        f = np.logspace(1, 7, 6 * ppd + 1)
        return f, _freq(f, self.A0 / (1 + 1j * f / self.FP))

    @covers('passband_ripple_db')
    def test_passband_ripple(self):
        f = np.logspace(1, 5, 41)
        mag = np.where(f <= 1e3, 1.0 + 0.05 * np.cos(np.arange(41)), 0.01)
        band = mag[f <= 1e3]
        expected = 20 * math.log10(band.max() / band.min())
        assert M.passband_ripple_db(_freq(f, mag), f_pass=1e3) == pytest.approx(expected, rel=1e-12)

    @covers('stopband_attenuation_db')
    def test_stopband_attenuation(self):
        f = np.logspace(1, 5, 41)
        mag = np.where(f < 1e3, 1.0, 0.01)
        assert M.stopband_attenuation_db(_freq(f, mag), f_stop=1e3) == pytest.approx(40.0, abs=1e-9)

    @covers('q_from_peak')
    def test_q_of_second_order_bandpass_within_grid_resolution(self):
        Q, f0, ppd = 10.0, 1e4, 1000
        f = np.logspace(3, 5, 2 * ppd + 1)
        mag = 1.0 / np.sqrt(1 + Q ** 2 * (f / f0 - f0 / f) ** 2)
        r = 10 ** (1.0 / ppd)
        got = M.q_from_peak(_freq(f, mag))
        # both −3 dB edges snap INWARD to the grid ⇒ BW under-read by ≤ 2 steps
        # ⇒ ΔQ/Q ≤ 2·(r−1)·f0/BW = 2·(r−1)·Q
        assert Q <= got <= Q * (1 + 2 * (r - 1) * Q) + 1e-9

    @covers('group_delay')
    def test_group_delay_of_pure_delay(self):
        # complex H = exp(-j w tau) (TASK-99 contract); 100/dec keeps the
        # per-step phase move (2*pi*tau*df = 0.043 rad at the top) well
        # under the unwrap limit of pi
        tau = 3e-9
        f = np.logspace(3, 8, 501)
        gd = M.group_delay(_freq(f, np.exp(-2j * np.pi * f * tau)))
        assert np.allclose(gd.y, tau, rtol=1e-9)

    def test_group_delay_single_pole_within_documented_grid_bound(self):
        """FIXED (TASK-99): pre-fix the complex y was cast to float and
        Re(H) differentiated; on this fixture the value at the pole was
        7.96e-4 s (true 7.96e-6 s). Now every sweep point is within the
        docstring's finite-difference bound: (r-1)^2/r interior, r-1 at
        the two ends."""
        f, s = self._single_pole()
        r = 10 ** (1.0 / self.PPD)
        tau = (1 / (2 * np.pi * self.FP)) / (1 + (f / self.FP) ** 2)
        rel = np.abs(M.group_delay(s).y / tau - 1)
        assert rel[1:-1].max() <= (r - 1) ** 2 / r * 1.1
        assert max(rel[0], rel[-1]) <= r - 1

    @covers('gain_bw_product')
    def test_gbw_single_pole_within_one_grid_step(self):
        f, s = self._single_pole()
        mag = _freq(f, np.abs(s.y))                      # real magnitude data
        got = M.gain_bw_product(mag)
        assert self.A0 * self.FP / self.R <= got <= self.A0 * self.FP * (1 + 1e-12)

    def test_gbw_on_complex_ac_signal_matches_magnitude_form(self):
        """FIXED (TASK-63): gain_bw_product no longer casts complex y to
        float before taking the magnitude, so a complex Signal (the shape
        every simulator .ac sweep produces) agrees with the real-magnitude
        form."""
        f, s = self._single_pole()
        assert M.gain_bw_product(s) == pytest.approx(M.gain_bw_product(_freq(f, np.abs(s.y))), rel=1e-12)

    @covers('psrr_db')
    def test_psrr_constant_ratio(self):
        f = np.logspace(1, 5, 41)
        sup = _freq(f, np.ones(41, dtype=complex))
        out = _freq(f, 1e-3 * np.ones(41, dtype=complex))
        p = M.psrr_db(sup, out)
        assert np.allclose(p.y, 60.0, atol=1e-9)

    # --- AC/noise helpers in silk.metrics -------------------------

    @covers('dc_gain_dB')
    def test_dc_gain_at_lowest_frequency(self):
        f, s = self._single_pole()
        expected = 20 * math.log10(self.A0 / math.sqrt(1 + (f[0] / self.FP) ** 2))
        assert M.dc_gain_dB(s) == pytest.approx(expected, abs=1e-9)

    @covers('peak_gain_dB')
    def test_peak_gain_of_second_order_peaking(self):
        zeta, fn = 0.2, 1e4
        f = np.logspace(3, 5, 4001)
        h = 1.0 / (1 - (f / fn) ** 2 + 2j * zeta * f / fn)
        expected = 20 * math.log10(1 / (2 * zeta * math.sqrt(1 - zeta ** 2)))
        # grid step 0.115 % in f; the peak is quadratic ⇒ error ≪ 1e-3 dB
        assert M.peak_gain_dB(_freq(f, h)) == pytest.approx(expected, abs=1e-3)

    @covers('gain_at_freq', 'phase_at_freq')
    def test_gain_and_phase_at_a_grid_frequency(self):
        f, s = self._single_pole()
        fq = f[400]
        assert M.gain_at_freq(s, fq) == pytest.approx(self.A0 / math.sqrt(1 + (fq / self.FP) ** 2), rel=1e-12)
        assert M.phase_at_freq(s, fq) == pytest.approx(-math.degrees(math.atan(fq / self.FP)), rel=1e-12)

    @covers('bandwidth_3dB')
    def test_bandwidth_is_the_last_grid_point_at_or_below_the_pole(self):
        _, s = self._single_pole()
        got = M.bandwidth_3dB(s)
        # resolution = one grid step BELOW the true −3 dB frequency
        assert self.FP / self.R < got <= self.FP * (1 + 1e-12)
        # and at 20 pts/decade the same signal reads up to 12 % low
        _, coarse = self._single_pole(ppd=20)
        got20 = M.bandwidth_3dB(coarse)
        assert self.FP / 10 ** (1 / 20) < got20 <= self.FP * (1 + 1e-12)


    @covers('unity_gain_freq')
    def test_unity_gain_freq_single_pole(self):
        _, s = self._single_pole()
        fu = self.FP * math.sqrt(self.A0 ** 2 - 1)
        assert M.unity_gain_freq(s) == pytest.approx(fu, rel=1e-4)      # interpolated

    @covers('phase_margin')
    def test_phase_margin_single_pole(self):
        _, s = self._single_pole()
        fu = self.FP * math.sqrt(self.A0 ** 2 - 1)
        expected = 180.0 - math.degrees(math.atan(fu / self.FP))
        # phase is read at the NEAREST grid sample: error ≤ slope·(ln r)/2
        assert M.phase_margin(s) == pytest.approx(expected, abs=0.01)

    @covers('gain_margin', 'gain_margins')
    def test_gain_margin_three_pole(self):
        f = np.logspace(1, 7, 1201)
        h = self.A0 / (1 + 1j * f / self.FP) ** 3
        # phase −180° at f = fp·√3 where |L| = A0/8
        fc = self.FP * math.sqrt(3)
        gm = -20 * math.log10(self.A0 / 8)
        margins = M.gain_margins(_freq(f, h))
        assert len(margins) == 1
        assert margins[0][0] == pytest.approx(fc, rel=1e-4)
        assert margins[0][1] == pytest.approx(gm, abs=0.01)
        assert M.gain_margin(_freq(f, h)) == pytest.approx(gm, abs=0.01)

    @covers('cmrr_dB')
    def test_cmrr(self):
        f = np.logspace(1, 5, 41)
        d = _freq(f, 100.0 * np.ones(41, dtype=complex))
        c = _freq(f, 0.01 * np.ones(41, dtype=complex))
        assert M.cmrr_dB(d, c) == pytest.approx(80.0, abs=1e-9)

    @covers('integrated_noise')
    def test_integrated_white_noise_and_log_grid_bound(self):
        f = np.linspace(1.0, 1001.0, 1001)
        s = _freq(f, np.full(1001, 2e-9))
        assert M.integrated_noise(s, 1.0, 1001.0) == pytest.approx(2e-9 * math.sqrt(1000.0), rel=1e-12)
        # 1/f PSD on a 10 pt/decade log grid: trapezoid over-reads each segment by
        # ½(r − 1/r)/ln r − 1 = 0.9 % in power ⇒ ≤ 0.45 % in V_rms
        fl = np.logspace(0, 4, 41)
        s1 = _freq(fl, 1e-9 / np.sqrt(fl))           # V/√Hz ⇒ PSD ∝ 1/f
        exact = 1e-9 * math.sqrt(math.log(1e4))
        got = M.integrated_noise(s1, 1.0, 1e4)
        assert exact < got <= exact * 1.0045

    @covers('noise_density_at_freq')
    def test_noise_density_at_freq_nearest_grid_point(self):
        f = np.array([1.0, 10.0, 100.0, 1000.0])
        s = _freq(f, np.array([1e-9, 2e-9, 3e-9, 4e-9]))
        assert M.noise_density_at_freq(s, 10.0) == pytest.approx(2e-9, rel=1e-12)
        # nearer 1000 than 100 (linear grid): interpolate=False reads the nearest sample
        assert M.noise_density_at_freq(s, 600.0, interpolate=False) == pytest.approx(4e-9, rel=1e-12)
        # strictly outside the sweep: unavailable, not extrapolated
        assert math.isnan(M.noise_density_at_freq(s, 1001.0))

    @covers('noise_figure_at_freq', 'band_noise_figure', 'noise_figure_vs_freq')
    def test_noise_figure_closed_form(self):
        # TASK-414: source at T_sim plus added white noise of 3·4kT0·Rs ⇒ F = 4
        # (6.02 dB) at every frequency, whatever T_sim, and in the band.
        rs, t0, tsim = 1e3, 290.0, 300.15
        k = 1.380649e-23
        f = np.logspace(0, 6, 61)
        s_in = 4 * k * tsim * rs + 3 * 4 * k * t0 * rs
        nin = _freq(f, np.full(61, math.sqrt(s_in)))
        nout = _freq(f, np.sqrt(s_in * 100.0 / (1 + (f / 1e4) ** 2)))
        nf = 10 * math.log10(4.0)
        assert M.noise_figure_at_freq(nin, 1234.0, rs, tsim) == pytest.approx(nf, rel=1e-12)
        assert M.band_noise_figure(nin, nout, 1.0, 1e6, rs, tsim) == pytest.approx(nf, rel=1e-12)
        np.testing.assert_allclose(M.noise_figure_vs_freq(nin, rs, tsim).y, nf, rtol=1e-12)


class TestBandwidth3dBReference:
    """TASK-326 — ``bandwidth_3dB(reference=...)`` on the zeta=0.2 low-pass
    fixture (tests/fixtures/task326_zeta02_lowpass.py)."""

    @covers('bandwidth_3dB')
    def test_reference_peak_reproduces_todays_value_exactly(self):
        sig = ZETA02.signal()
        assert M.bandwidth_3dB(sig, reference='peak') == ZETA02.PEAK_BW_HZ

    @covers('bandwidth_3dB')
    def test_reference_dc_is_the_default_and_matches_analytic(self):
        sig = ZETA02.signal()
        # rel=1e-12 allows last-bit platform rounding (CI run 37536524664);
        # abs=0 keeps this a strictly relative comparison to the pinned value.
        expected = pytest.approx(ZETA02.DC_BW_HZ, rel=1e-12, abs=0)
        assert M.bandwidth_3dB(sig) == expected
        assert M.bandwidth_3dB(sig, reference='dc') == expected
        # grid-snapped: reads low by up to one grid step (200 pts/decade).
        grid_ratio = 10 ** (1 / 200)
        assert (ZETA02.DC_ANALYTIC_BW_HZ / grid_ratio < ZETA02.DC_BW_HZ
                <= ZETA02.DC_ANALYTIC_BW_HZ * (1 + 1e-9))

    @covers('bandwidth_3dB')
    def test_reference_dc_equals_gain_bw_product_over_dc_gain(self):
        # bandwidth_3dB(reference='dc') is, by construction, the same
        # DC-relative grid point gain_bw_product already scans for.
        sig = ZETA02.signal()
        dc_lin = 10 ** (M.dc_gain_dB(sig) / 20.0)
        assert M.bandwidth_3dB(sig, reference='dc') == pytest.approx(
            M.gain_bw_product(sig) / dc_lin, rel=1e-12)

    @covers('bandwidth_3dB')
    def test_invalid_reference_raises_value_error_naming_the_choices(self):
        sig = ZETA02.signal()
        with pytest.raises(ValueError, match=r"reference must be 'dc'\|'peak', got 'bogus'"):
            M.bandwidth_3dB(sig, reference='bogus')

    @covers('bandwidth_3dB')
    def test_band_pass_near_zero_dc_gain_returns_nan_suggesting_peak(self):
        import warnings
        from silk.metrics.diagnostics import MetricWarning

        f = np.logspace(1, 7, 1201)
        w0 = 2 * math.pi * 1e4
        x = 1j * 2 * math.pi * f / w0
        h = x / (1 + x + x ** 2)          # band-pass: a hard zero at DC
        sig = Signal('h', 'voltage', 'frequency', 'V', f, h)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            got = M.bandwidth_3dB(sig)
        assert math.isnan(got)
        [w] = [c.message for c in caught if isinstance(c.message, MetricWarning)]
        assert w.code == 'upper_rolloff_not_observed'
        assert "reference='peak'" in w.suggestion

    @covers('bandwidth_3dB')
    def test_exact_zero_dc_gain_raises_no_dc_reference_warning(self):
        import warnings
        from silk.metrics.diagnostics import MetricWarning

        f = np.array([1e3, 1e4, 1e5, 1e6])
        h = np.array([0.0, 1.0, 0.5, 0.1], dtype=complex)
        sig = Signal('h', 'voltage', 'frequency', 'V', f, h)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            got = M.bandwidth_3dB(sig)
        assert math.isnan(got)
        [w] = [c.message for c in caught if isinstance(c.message, MetricWarning)]
        assert w.code == 'no_dc_reference'
        assert "reference='peak'" in w.suggestion


# ===========================================================================
# dc
# ===========================================================================

class TestDcMetrics:

    @covers('inl', 'INL')
    def test_inl_endpoint_of_bowed_curve(self):
        n, lsb, eps = 16, 0.1, 0.03
        k = np.arange(n)
        bow = eps * np.sin(np.pi * k / (n - 1))
        r = M.inl(_code(k, k * lsb + bow))
        assert isinstance(r, M.INL)
        assert np.allclose(r.inl_lsb, bow / lsb, atol=1e-12)
        assert r.max_abs_lsb == pytest.approx(float(np.max(np.abs(bow))) / lsb, rel=1e-9)

    @covers('dnl', 'DNL')
    def test_dnl_of_known_step_errors(self):
        lsb = 0.1
        d = np.array([0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0])   # Σ = 0 keeps endpoint LSB
        v = np.concatenate([[0.0], np.cumsum(lsb * (1 + d))])
        r = M.dnl(_code(np.arange(8), v))
        assert isinstance(r, M.DNL)
        assert np.allclose(r.dnl_lsb, d, atol=1e-12)

    @covers('is_monotonic', 'missing_codes')
    def test_monotonic_and_zero_width_step(self):
        assert M.is_monotonic(_code([0, 1, 2, 3], [0.0, 0.1, 0.2, 0.3])) is True
        assert M.is_monotonic(_code([0, 1, 2, 3], [0.0, 0.2, 0.1, 0.3])) is False
        # T(1) == T(2): code 1's bin has zero width, so missing_codes
        # reports code 1 (DNL = −1), not the code after the gap.
        assert list(M.missing_codes(_code([0, 1, 2, 3], [0.0, 0.1, 0.1, 0.3]))) == [1.0]

    @covers('extract_code_transitions')
    def test_transitions_of_ideal_adc_within_ramp_step(self):
        n = 4001
        t = np.linspace(0, 1, n)
        v = t.copy()                                   # 0 → 1 V ramp
        step = 1.0 / (n - 1)
        code = np.floor(v * 8).clip(0, 7)
        tr = M.extract_code_transitions(_tran(t, v), Signal('c', 'code', 'time', '', t, code))
        # lowest observed code (0) has no transition level (IEEE 1241):
        # only codes 1..7 get a row.
        assert list(tr.x) == list(range(1, 8))
        # resolution: the ramp step (first sample AT or after the true level)
        for i, k in enumerate(range(1, 8)):
            assert k / 8 - 1e-12 <= tr.y[i] < k / 8 + step

    @covers('p1db')
    def test_p1db_pwl_compression(self):
        # gain G0 up to p_c, then falling at s dB per dB ⇒ P1dB = p_c + 1/s
        p_in = np.linspace(-30, 10, 41)
        G0, pc, s = 20.0, -5.0, 0.25
        gain = G0 - s * np.clip(p_in - pc, 0, None)
        sweep = Signal('psweep', 'power', 'power', 'dBm', p_in, p_in + gain)
        assert M.p1db(sweep) == pytest.approx(pc + 1 / s, rel=1e-12)


# ===========================================================================
# noise
# ===========================================================================

class TestNoiseMetrics:

    @covers('noise_figure', 'noise_figure_from_factor', 'input_referred_noise',
            'sum_uncorrelated', 'kT_over_C', 'aperture_jitter_snr_db')
    def test_closed_form_scalars(self):
        assert M.noise_figure(20.0, 17.0) == 3.0
        assert M.noise_figure_from_factor(2.0) == pytest.approx(10 * math.log10(2), rel=1e-12)
        assert M.input_referred_noise(1e-3, -100.0) == pytest.approx(1e-5, rel=1e-12)
        assert M.sum_uncorrelated(3.0, 4.0) == 5.0
        assert M.kT_over_C(1e-12, 300.0) == pytest.approx(math.sqrt(1.380649e-23 * 300 / 1e-12), rel=1e-12)
        assert M.aperture_jitter_snr_db(1e6, 1e-12) == pytest.approx(-20 * math.log10(2 * math.pi * 1e-6), rel=1e-12)

    @covers('one_over_f_corner')
    def test_corner_within_one_grid_step(self):
        w, fc, ppd = 1e-9, 1e3, 100
        f = np.logspace(0, 6, 6 * ppd + 1)
        y = w * np.sqrt(1 + fc / f)                  # equal 1/f and white power at fc
        got = M.one_over_f_corner(_freq(f, y))
        r = 10 ** (1 / ppd)
        # Stated definition: white floor = median of the upper half (f >= √(f_min·f_max))
        # of the LOG grid, i.e. y at f_med = 10^4.5 here — which still carries a
        # little 1/f, so the floor is over-read by ε and the corner reads LOW:
        # y < √2·w·(1+ε)  ⇔  f > fc / (2(1+ε)² − 1).
        eps = math.sqrt(1 + fc / 10 ** 4.5) - 1
        f_star = fc / (2 * (1 + eps) ** 2 - 1)
        assert f_star < got <= f_star * r            # first grid point above f*
        assert 0.95 * fc < got < fc                 # ≈ 4.5 % low on this sweep

    @covers('cascade_noise_figure')
    def test_friis_two_stage(self):
        F = 2.0 + (10.0 - 1.0) / 10.0                # NF1 3 dB, G1 10 dB, NF2 10 dB
        got = M.cascade_noise_figure([10 * math.log10(2.0), 10.0], [10.0, 0.0])
        assert got == pytest.approx(10 * math.log10(F), rel=1e-12)


# ===========================================================================
# fom
# ===========================================================================

class TestFom:

    @covers('walden_fom', 'schreier_fom', 'iip3_extrapolation', 'pae', 'dc_to_rf_efficiency')
    def test_closed_form_scalars(self):
        assert M.walden_fom(1e-3, 1e8, 10.0) == pytest.approx(1e-3 / (1024 * 1e8), rel=1e-12)
        assert M.schreier_fom(60.0, 1e6, 1e-3) == pytest.approx(150.0, rel=1e-12)
        assert M.iip3_extrapolation(-10.0, -20.0, -80.0) == pytest.approx(20.0, rel=1e-12)
        assert M.pae(1.0, 0.1, 2.0) == pytest.approx(0.45, rel=1e-12)
        assert M.dc_to_rf_efficiency(1.0, 2.0) == 0.5


# ===========================================================================
# completeness guard
# ===========================================================================

def _public_metric_names():
    names = set(M.__all__) - {'waveform', 'statistics', 'timing', 'jitter', 'eye', 'power',
                            'spectral', 'rf', 'converter', 'fom', 'ac', 'noise'}
    return names


def test_every_public_metric_has_a_fixture():
    missing = sorted(_public_metric_names() - COVERED)
    assert not missing, f'public metrics without an analytic fixture: {missing}'


@covers('sfdr_db', 'harmonic_level_dbc')
def test_scalar_spectral_wrappers_on_coherent_harmonic():
    sig = _tones((11, 1.0), (22, 0.1))
    assert M.sfdr_db(sig, tclk=TCLK, n_samples=N) == pytest.approx(20.0, abs=1e-9)
    assert M.harmonic_level_dbc(sig, n=2, tclk=TCLK, n_samples=N) == pytest.approx(-20.0, abs=1e-9)


@covers('eye_height', 'eye_width')
def test_scalar_eye_wrappers_on_triangle():
    sig = _triangle()
    assert M.eye_height(sig, period=P/2, threshold=0.) == pytest.approx(2.0, abs=1e-12)
    assert M.eye_width(sig, period=P/2, threshold=0.) == pytest.approx(P/2, abs=P/SPP)


@covers('evm_percent', 'modulation_grid')
def test_evm_percent_fits_scale_and_rotation_of_16qam():
    # TASK-369: a unit-RMS 16-QAM grid, scaled by 0.3 and rotated 20 degrees,
    # with every received point pushed 5 % (of |gain|) along +I: EVM is 5 %.
    grid = M.modulation_grid('16qam')
    assert np.mean(np.abs(grid) ** 2) == pytest.approx(1.0, rel=1e-12)
    gain = 0.3 * np.exp(1j * math.radians(20))
    sym = gain * grid[np.arange(160) % 16] + 0.05 * abs(gain)
    t = np.arange(sym.size * 4) / 4.0
    i, q = _tran(t, np.repeat(sym.real, 4)), _tran(t, np.repeat(sym.imag, 4))
    # The common offset is not a gain, so the least-squares fit leaves it as
    # the error: exactly 5 % of the fitted RMS.
    assert M.evm_percent(i, q, symbol_period=1.0, modulation='16qam') == \
        pytest.approx(5.0, rel=1e-9)


@covers('static_errors', 'StaticErrors', 'offset_error_lsb', 'gain_error_lsb',
        'full_scale_error_lsb')
def test_static_errors_known_offset_and_gain():
    # 3-bit, 1 V full scale: ideal transition into code k is k * 0.125 V.
    lsb = 0.125
    k = np.arange(1, 8)
    ideal = _code(k, k * lsb)
    r = M.static_errors(ideal, full_scale=1.0, n_bits=3)
    assert (r.offset_error_lsb, r.gain_error_lsb, r.full_scale_error_lsb) == (0.0, 0.0, 0.0)
    # +0.5 LSB offset and a 1 % gain stretch about the first transition.
    v = k * lsb + 0.5 * lsb
    v = v[0] + (v - v[0]) * 1.01
    sig = _code(k, v)
    span = 6 * lsb
    assert M.offset_error_lsb(sig, full_scale=1.0, n_bits=3) == pytest.approx(0.5)
    assert M.gain_error_lsb(sig, full_scale=1.0, n_bits=3) == pytest.approx(0.01 * span / lsb)
    assert M.full_scale_error_lsb(sig, full_scale=1.0, n_bits=3) == pytest.approx(0.5 + 0.06)


@covers('inl_max_lsb', 'dnl_max_lsb', 'missing_code_count', 'non_monotonic_steps')
def test_scalar_code_wrappers_with_known_errors():
    bowed = _code([0, 1, 2], [0, 1.5, 2])
    assert M.inl_max_lsb(bowed) == 0.5
    assert M.dnl_max_lsb(bowed) == 0.5
    assert M.missing_code_count(_code([0, 1, 2, 3], [0, 1, 1, 3])) == 1.0
    assert M.non_monotonic_steps(_code([0, 1, 2, 3], [0, 2, 1, 3])) == 1.0
