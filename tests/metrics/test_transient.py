"""Tests for the waveform, statistics, timing, jitter, eye and power metrics."""
import math
import re
import warnings
import numpy as np
import pytest

from silk.signals.base import Signal
from silk.utils._compat import trapezoid
from silk.circuit.spec_report import SpecReport
from silk.metrics.diagnostics import MetricWarning
from silk.metrics.power import (
    _VOLTAGE_TYPES,
    _CURRENT_TYPES,
    _recognised_quantity,
    average_power,
    dynamic_energy_per_cycle,
)
from silk.metrics.waveform import (
    _all_crossings,
    crossings,
    peak,
    peak_to_peak,
    minimum,
    maximum,
    argmin_x,
    argmax_x,
    integral,
    running_integral,
    derivative,
    value_at,
    crossing,
    convolve,
    lshift,
    rshift,
    clip,
    window,
    flip,
    sample,
    cross_correlation,
)
from silk.metrics.timing import (
    rise_time,
    fall_time,
    slew_rate,
    settling_time,
    propagation_delay,
    propagation_delays,
    overshoot,
    undershoot,
    average_duty,
    average_period,
    pulse_width,
    hysteresis,
    cycle_frequency,
    on_time,
    off_time,
)
from silk.metrics.jitter import (
    period_jitter,
    cycle_to_cycle_jitter,
    tie,
    long_term_jitter,
    allan_deviation,
)
from silk.metrics.statistics import (
    average,
    rms,
    stddev,
    error_norm,
    relative_error_norm,
)
from silk.metrics.eye import (
    eye_diagram,
    EyeResult,
)


def _trans(t, y, name='s'):
    return Signal(name, 'voltage', 'time', 'V',
                  np.asarray(t, dtype=float), np.asarray(y, dtype=float))


def _v(t, y):
    return Signal('v', 'voltage', 'time', 'V',
                  np.asarray(t, dtype=float), np.asarray(y, dtype=float))


def _i(t, y):
    return Signal('i', 'current', 'time', 'A',
                  np.asarray(t, dtype=float), np.asarray(y, dtype=float))


def _step(rise_const=1.0, n=1001, t_end=10.0):
    t = np.linspace(0.0, t_end, n)
    y = 1.0 - np.exp(-t / rise_const)
    return _trans(t, y)


# ---------------------------------------------------------------------------
# crossings
# ---------------------------------------------------------------------------

class TestCrossings:
    def test_known_clock(self):
        t = np.linspace(0, 1, 10001)
        y = (np.sin(2 * np.pi * 5 * t) > 0).astype(float)
        edges = crossings(_trans(t, y), edge='rising')
        assert len(edges) == 5
        assert math.isclose(np.mean(np.diff(edges)), 0.2, rel_tol=0.05)

    def test_falling(self):
        t = np.linspace(0, 1, 10001)
        y = (np.sin(2 * np.pi * 5 * t) > 0).astype(float)
        assert len(crossings(_trans(t, y), edge='falling')) == 5

    def test_any_returns_both(self):
        t = np.linspace(0, 1, 10001)
        y = (np.sin(2 * np.pi * 5 * t) > 0).astype(float)
        assert len(crossings(_trans(t, y), edge='any')) == 10

    def test_explicit_threshold(self):
        t = np.linspace(0, 10, 1001)
        ed = crossings(_trans(t, t.copy()), threshold=5.0, edge='rising')
        assert len(ed) == 1
        assert math.isclose(ed[0], 5.0, abs_tol=0.02)

    def test_no_crossings(self):
        t = np.linspace(0, 1, 100)
        assert len(crossings(_trans(t, np.zeros_like(t)), threshold=0.5)) == 0

    def test_accepts_monotonic_frequency_coordinates(self):
        sig = Signal('f', 'voltage', 'frequency', 'V',
                     np.array([1.0, 10.0]), np.array([0.0, 1.0]))
        np.testing.assert_array_equal(crossings(sig, threshold=.5), [5.5])

    def test_invalid_edge(self):
        with pytest.raises(ValueError, match='rising|falling|any'):
            crossings(_step(), edge='sideways')


# ---------------------------------------------------------------------------
# Edge metrics
# ---------------------------------------------------------------------------

class TestRiseTime:
    def test_first_order_step(self):
        assert math.isclose(rise_time(_step(1.0)), math.log(9.0), rel_tol=0.01)

    def test_constant_returns_nan(self):
        t = np.linspace(0, 1, 100)
        assert math.isnan(rise_time(_trans(t, np.ones_like(t))))

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match='low < high'):
            rise_time(_step(), low=0.9, high=0.1)

    # -- TASK-55: one-interval / sample-rate-floor cases -------------------
    #
    # rise_time restarted its 90% search at searchsorted(t_low), which is
    # one segment PAST the segment the 10% crossing came from (searchsorted
    # returns the index strictly after an interpolated interior point).
    # When both crossings share that one segment, the restart skips past
    # it entirely. The fix backs the restart index up by one segment
    # (start_idx = max(low_idx - 1, 0)) so the shared segment is rescanned.

    def test_one_interval_edge_hits_the_sample_rate_floor(self):
        """dt=1e-9, edge fully inside segment [2,3] (10% and 90% both in
        one interval). Pre-fix: nan — the 90% search started at segment
        3->4, one past the segment that held it, and never found a rising
        crossing there. Post-fix: 0.8*dt, the sample-rate floor (this is
        case A from the TASK-55 investigation).
        """
        dt = 1e-9
        t = np.arange(8) * dt
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        assert rise_time(_trans(t, y)) == pytest.approx(0.8 * dt, rel=1e-9, abs=0)

    def test_ringing_inside_one_interval_was_silently_wrong_not_nan(self):
        """The HIGH-severity case: with overshoot that dips back below the
        90% level, skipping the shared segment does not fail loud — it
        finds a LATER 90% crossing during the ringing. Pre-fix this
        returned 2.2333*dt (2.8x too large), a confidently wrong number
        with no error. Post-fix: 0.8*dt, same sample-rate floor as the
        clean one-interval edge above (case B from the investigation).
        """
        dt = 1e-9
        t = np.arange(8) * dt
        y = np.array([0, 0, 0, 1.0, 0.85, 1.0, 1.0, 1.0], dtype=float)
        assert rise_time(_trans(t, y)) == pytest.approx(0.8 * dt, rel=1e-9, abs=0)

    def test_multi_interval_edge_is_unchanged(self):
        """Regression guard: an edge whose 10% and 90% crossings land in
        DIFFERENT segments was already correct before the fix and must
        stay that way — backing the restart up by one segment must not
        walk it into an earlier, wrong crossing (case E, 4 intervals).
        """
        dt = 1e-9
        t = np.arange(8) * dt
        y = np.array([0, 0, 0.25, 0.5, 0.75, 1, 1, 1], dtype=float)
        assert rise_time(_trans(t, y)) == pytest.approx(3.2 * dt, rel=1e-9, abs=0)


class TestFallTime:
    def test_falling_step(self):
        t = np.linspace(0.0, 10.0, 1001)
        y = np.exp(-t / 1.0)
        assert math.isclose(fall_time(_trans(t, y)), math.log(9.0), rel_tol=0.01)

    # -- TASK-55: mirror image of the rise_time cases above -----------------

    def test_one_interval_edge_hits_the_sample_rate_floor(self):
        """Mirrors TestRiseTime's case A with rising/falling roles
        reversed. Pre-fix: nan (fall_time's high_idx restart had the
        identical off-by-one). Post-fix: 0.8*dt (case C).
        """
        dt = 1e-9
        t = np.arange(8) * dt
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
        assert fall_time(_trans(t, y)) == pytest.approx(0.8 * dt, rel=1e-9, abs=0)

    def test_ringing_inside_one_interval_was_silently_wrong_not_nan(self):
        """Mirrors TestRiseTime's ringing case: undershoot back above the
        10% level during the fall skips the shared segment and (pre-fix)
        finds a later, wrong 10% crossing instead of failing loud.
        Post-fix: 0.8*dt, the sample-rate floor (case D).
        """
        dt = 1e-9
        t = np.arange(8) * dt
        y = np.array([1, 1, 1, 0.0, 0.15, 0.0, 0.0, 0.0], dtype=float)
        assert fall_time(_trans(t, y)) == pytest.approx(0.8 * dt, rel=1e-9, abs=0)


class TestSlewRate:
    def test_linear_ramp(self):
        t = np.linspace(0, 1, 1001)
        assert math.isclose(slew_rate(_trans(t, t.copy())), 1.0, rel_tol=0.01)

    # -- TASK-55: slew_rate delegates to rise_time and inherited both of
    # its failure modes (nan on a clean one-interval edge, a silently
    # ~2.8x-too-slow number on a ringing one). Both must now be finite
    # and correct, computed straight from the sample-rate-floor rise time.

    def test_finite_on_one_interval_edge(self):
        dt = 1e-9
        t = np.arange(8) * dt
        y = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        # rise_time floor is 0.8*dt over a 0.8 swing fraction (0.9-0.1) of
        # a unit swing -> slew = 0.8 * 1.0 / (0.8*dt) = 1/dt.
        assert slew_rate(_trans(t, y)) == pytest.approx(1.0 / dt, rel=1e-9)

    def test_finite_on_ringing_edge(self):
        dt = 1e-9
        t = np.arange(8) * dt
        y = np.array([0, 0, 0, 1.0, 0.85, 1.0, 1.0, 1.0], dtype=float)
        assert slew_rate(_trans(t, y)) == pytest.approx(1.0 / dt, rel=1e-9)


class TestSettlingTime:
    def test_first_order(self):
        sig = _step(1.0, n=10001, t_end=20.0)
        assert 4.5 < settling_time(sig, initial=0.0, target=1.0, tol=0.01) < 4.8

    def test_unsettled(self):
        t = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        assert math.isnan(settling_time(_trans(t, y), initial=0.0, target=10.0, tol=0.01))

    def test_zero_swing_raises(self):
        with pytest.raises(ValueError, match='settling_time'):
            settling_time(_trans([0, 1, 2], [1.4, 1.5, 1.5]), initial=1.5, target=1.5)

    def test_band_is_fraction_of_swing_not_of_level(self):
        # A 1.4 -> 1.5 V step (100 mV swing) settles into a +-1 mV band
        # (1 % of the swing), not +-15 mV (1 % of the 1.5 V level).
        t = np.linspace(0, 1, 10001)
        y = 1.4 + 0.1 * (1.0 - np.exp(-t / 0.05))
        got = settling_time(_trans(t, y), initial=1.4, target=1.5, tol=0.01)
        # closed-form 1 % (of swing) settling instant for a first-order
        # step: tau * ln(1/tol).
        expect = 0.05 * math.log(100.0)
        assert expect <= got < expect + (t[1] - t[0])


class TestPropagationDelay:
    def test_simple_delay(self):
        t = np.linspace(0, 10, 1001)
        sig_in  = _trans(t, (t > 2.0).astype(float))
        sig_out = _trans(t, (t > 3.0).astype(float))
        d = propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5)
        assert math.isclose(d, 1.0, abs_tol=0.02)

    def test_auto_threshold(self):
        t = np.linspace(0, 10, 1001)
        sig_in  = _trans(t, 5.0 * (t > 2.0).astype(float))
        sig_out = _trans(t, 1.8 * (t > 3.5).astype(float))
        assert math.isclose(propagation_delay(sig_in, sig_out), 1.5, abs_tol=0.02)

    def test_no_crossing(self):
        t = np.linspace(0, 10, 1001)
        sig_in  = _trans(t, (t > 2.0).astype(float))
        sig_out = _trans(t, np.zeros_like(t))
        assert math.isnan(propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5))

    # TASK-192: an unrecognised edge used to fall through all three
    # direction tests in _first_crossing, so EVERY sample interval was
    # accepted and the first one won — edge='rsing' returned 0.0 on this
    # 1 ns pair, a finite number indistinguishable from a measurement.
    # decision-1 classes a malformed request as caller error, and the
    # three sibling edge-takers (crossings, crossing, and the two jitter
    # metrics through crossings) already raised on the same value.
    @pytest.mark.parametrize('bad', ['rsing', 'up', '', 'RISING', 'any ',
                                     None, 0])
    def test_invalid_edge(self, bad):
        t = np.linspace(0, 10, 1001)
        sig_in  = _trans(t, (t > 2.0).astype(float))
        sig_out = _trans(t, (t > 3.0).astype(float))
        with pytest.raises(ValueError, match='rising|falling|any'):
            propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5, in_edge=bad)

    def test_invalid_edge_message_names_metric_and_value(self):
        t = np.linspace(0, 10, 1001)
        sig_in  = _trans(t, (t > 2.0).astype(float))
        sig_out = _trans(t, (t > 3.0).astype(float))
        with pytest.raises(ValueError) as exc:
            propagation_delay(sig_in, sig_out, in_edge='rsing')
        msg = str(exc.value)
        assert 'propagation_delay' in msg      # which metric
        assert "'rsing'" in msg                # what was received
        for permitted in ("'rising'", "'falling'", "'any'"):
            assert permitted in msg            # what was available

    @pytest.mark.parametrize('edge', ['rising', 'falling', 'any'])
    def test_valid_edges_still_measure(self, edge):
        """The TASK-192 guard must not touch any permitted value."""
        t = np.linspace(0, 10, 1001)
        sig_in  = _trans(t, (t > 2.0).astype(float))
        sig_out = _trans(t, (t > 3.0).astype(float))
        got = propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5, in_edge=edge)
        if edge == 'falling':
            assert math.isnan(got)             # no falling edge in either
        else:
            assert math.isclose(got, 1.0, abs_tol=0.02)


class TestPropagationDelayOutEdge:
    """TASK-223 — ``out_edge`` decouples the output's crossing direction from
    the input's, so an inverting stage's t_pHL/t_pLH become requestable."""

    @staticmethod
    def _inverting_signals():
        """An inverting DUT: input HIGH from t=2 to t=6 (rising@2, falling@6);
        output inverts with a real delay — falling@2.3 (t_pHL=0.3 s),
        rising@6.4 (t_pLH=0.4 s)."""
        t = np.linspace(0, 10, 100_001)
        vin = ((t > 2.0) & (t <= 6.0)).astype(float)
        vout = 1.0 - (((t > 2.3) & (t <= 6.4)).astype(float))
        return _trans(t, vin), _trans(t, vout)

    def test_shared_edge_rising_is_the_half_period_trap(self):
        """Pinned so a later change to the pairing rule is deliberate
        (TASK-223 AC#1): ``edge='rising'`` on both sides pairs the input's
        rising crossing with the output's NEXT rising crossing, 4.4 s away —
        not the 0.3 s t_pHL."""
        sig_in, sig_out = self._inverting_signals()
        d = propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5, in_edge='rising')
        assert math.isclose(d, 4.4, abs_tol=1e-3)

    def test_out_edge_none_matches_edge_numerically(self):
        """AC#3 — ``out_edge=None`` means 'same as edge', so a pre-existing
        call is numerically unchanged."""
        sig_in, sig_out = self._inverting_signals()
        for edge in ('rising', 'falling', 'any'):
            plain = propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5, in_edge=edge)
            explicit = propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5,
                                         in_edge=edge, out_edge=None)
            if math.isnan(plain):
                assert math.isnan(explicit)
            else:
                assert plain == explicit

    def test_out_edge_falling_recovers_t_phl(self):
        """AC#4 — ``edge='rising', out_edge='falling'`` returns t_pHL."""
        sig_in, sig_out = self._inverting_signals()
        d = propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5,
                              in_edge='rising', out_edge='falling')
        assert math.isclose(d, 0.3, abs_tol=1e-3)

    def test_out_edge_rising_recovers_t_plh(self):
        """AC#4 — ``edge='falling', out_edge='rising'`` returns t_pLH."""
        sig_in, sig_out = self._inverting_signals()
        d = propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5,
                              in_edge='falling', out_edge='rising')
        assert math.isclose(d, 0.4, abs_tol=1e-3)

    @pytest.mark.parametrize('bad', ['rsing', 'up', '', 'RISING', 'any ',
                                     0])
    def test_invalid_out_edge(self, bad):
        sig_in, sig_out = self._inverting_signals()
        with pytest.raises(ValueError, match='rising|falling|any'):
            propagation_delay(sig_in, sig_out, in_threshold=0.5, out_threshold=0.5, out_edge=bad)

    def test_invalid_out_edge_message_names_metric_and_value(self):
        sig_in, sig_out = self._inverting_signals()
        with pytest.raises(ValueError) as exc:
            propagation_delay(sig_in, sig_out, out_edge='rsing')
        msg = str(exc.value)
        assert 'propagation_delay' in msg
        assert "'rsing'" in msg
        for permitted in ("'rising'", "'falling'", "'any'"):
            assert permitted in msg


class TestOvershoot:
    def test_no_overshoot(self):
        assert overshoot(_trans([0, 1, 2], [0.0, 0.5, 1.0]), initial=0.0, target=1.0) == 0.0

    def test_known(self):
        assert math.isclose(
            overshoot(_trans([0, 1, 2], [0.0, 1.2, 1.0]), initial=0.0, target=1.0),
            0.2, rel_tol=1e-12)

    def test_zero_target_nonzero_swing(self):
        # target == 0 V no longer takes an absolute-value branch: it is
        # normalised by the swing like any other target.
        assert math.isclose(
            overshoot(_trans([0, 1, 2, 3], [-0.5, 0.0, 0.3, 0.0]),
                     initial=-0.5, target=0.0),
            0.6, rel_tol=1e-12)

    def test_swing_normalisation_nonzero_start(self):
        # 1.4 -> 1.5 V step peaking at 1.519 V is 19 % of the 100 mV swing,
        # not 1.3 % of the 1.5 V level (TASK-285 AC#4).
        assert math.isclose(
            overshoot(_trans([0, 1, 2], [1.4, 1.519, 1.5]), initial=1.4, target=1.5),
            0.19, rel_tol=1e-9)

    def test_falling_step(self):
        # mirror of the rising-step case: excursion BELOW target.
        assert math.isclose(
            overshoot(_trans([0, 1, 2], [1.5, 1.381, 1.4]), initial=1.5, target=1.4),
            0.19, rel_tol=1e-9)

    def test_zero_swing_raises(self):
        with pytest.raises(ValueError, match='overshoot'):
            overshoot(_trans([0, 1, 2], [1.0, 1.2, 1.0]), initial=1.0, target=1.0)


class TestUndershoot:
    def test_known(self):
        # target (0.0) is reached at the very first sample, so the whole
        # trace is inside the post-transition search window.
        assert math.isclose(
            undershoot(_trans([0, 1, 2], [0.0, -0.2, 1.0]), initial=-1.0, target=0.0),
            0.2, rel_tol=1e-12)

    def test_post_transition_worked_example(self):
        # IEEE 181-2003 worked check (TASK-285 owner ruling): 1.4 -> 1.5 V
        # ringing up to 1.519 then dipping to 1.488 gives 0.12, never 1.0
        # from the pre-edge starting level.
        assert math.isclose(
            undershoot(_trans([0, 1, 2, 3, 4], [1.4, 1.519, 1.488, 1.5, 1.5]),
                      initial=1.4, target=1.5),
            0.12, rel_tol=1e-9)

    def test_falling_step(self):
        # mirror of the worked example: first reaches target at 1.381 (19 %
        # below, the falling step's own overshoot), rings back up to 1.412
        # (12 % above target) before settling.
        assert math.isclose(
            undershoot(_trans([0, 1, 2, 3, 4], [1.5, 1.381, 1.412, 1.4, 1.4]),
                      initial=1.5, target=1.4),
            0.12, rel_tol=1e-9)

    def test_preshoot_excluded(self):
        # a dip BEFORE the trace first reaches target is preshoot, not
        # undershoot (IEEE 181-2003), and must not be counted.
        assert undershoot(_trans([0, 1, 2, 3], [0.0, -0.5, 0.2, 1.0]),
                          initial=0.0, target=1.0) == 0.0

    def test_never_reaches_target_is_nan(self):
        assert math.isnan(
            undershoot(_trans([0, 1, 2], [0.0, 0.3, 0.4]), initial=0.0, target=1.0))

    def test_zero_swing_raises(self):
        with pytest.raises(ValueError, match='undershoot'):
            undershoot(_trans([0, 1, 2], [1.0, 1.2, 1.0]), initial=1.0, target=1.0)


class TestDutyCycle:
    def test_50pct(self):
        t = np.linspace(0, 1, 10001)
        y = (np.sin(2 * np.pi * 5 * t) > 0).astype(float)
        assert math.isclose(average_duty(_trans(t, y)), 0.5, abs_tol=0.02)

    def test_constant(self):
        t = np.linspace(0, 1, 100)
        assert math.isnan(average_duty(_trans(t, np.zeros_like(t))))


class TestPeriodZeroCross:
    def test_known(self):
        t = np.linspace(0, 1, 10001)
        y = np.sin(2 * np.pi * 5 * t)
        assert math.isclose(average_period(_trans(t, y)), 0.2, rel_tol=0.01)

    def test_too_few(self):
        t = np.linspace(0, 1, 10)
        assert math.isnan(average_period(_trans(t, np.zeros_like(t))))


class TestPulseWidth:
    def test_simple_pulse(self):
        t = np.linspace(0, 10, 10001)
        y = ((t > 2.0) & (t < 4.5)).astype(float)
        assert math.isclose(pulse_width(_trans(t, y)), 2.5, abs_tol=0.01)

    def test_immune_to_the_one_interval_bug_by_construction(self):
        """TASK-55 non-goal: pulse_width uses the identical searchsorted
        restart pattern as rise_time/fall_time did, but is IMMUNE — under
        linear interpolation a single sample interval is monotonic, so it
        can never contain both a rising and a falling crossing of the
        SAME threshold the way a one-interval rise/fall edge could contain
        both the 10% and 90% crossings. This is a regression guard proving
        the fix was correctly NOT applied here (pulse_width is unchanged).
        """
        dt = 1e-9
        t = np.arange(8) * dt
        # Pulse rises and falls within a couple of intervals of each other,
        # same scale as the TASK-55 rise/fall repros above.
        y = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=float)
        assert pulse_width(_trans(t, y)) == pytest.approx(2.0 * dt, rel=1e-9, abs=0)


# ---------------------------------------------------------------------------
# Clock / oscillator metrics
# ---------------------------------------------------------------------------

class TestPeriodJitter:
    def test_uniform_clock(self):
        t = np.linspace(0, 1.0, 100001)
        y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
        assert period_jitter(_trans(t, y)) < 1e-5

    def test_too_few(self):
        t = np.linspace(0, 1, 10)
        assert math.isnan(period_jitter(_trans(t, np.zeros_like(t))))

    def test_rejects_freq(self):
        sig = Signal('s', 'voltage', 'frequency', 'V',
                     np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="xlabel='time'"):
            period_jitter(sig)


class TestCycleToCycleJitter:
    def test_uniform_clock(self):
        t = np.linspace(0, 1.0, 100001)
        y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
        assert cycle_to_cycle_jitter(_trans(t, y)) < 1e-5

    def test_too_few(self):
        t = np.linspace(0, 1, 10)
        assert math.isnan(cycle_to_cycle_jitter(_trans(t, np.zeros_like(t))))


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

class TestAveragePower:
    def test_dc(self):
        t = np.linspace(0, 1, 1001)
        v = _v(t, np.full_like(t, 1.8))
        i = _i(t, np.full_like(t, 1e-3))
        assert math.isclose(average_power(v, i), 1.8e-3, rel_tol=1e-6)

    def test_sinusoidal(self):
        t = np.linspace(0, 1, 100001)
        c = np.cos(2 * np.pi * 5 * t)
        assert math.isclose(average_power(_v(t, c), _i(t, c)), 0.5, rel_tol=0.01)

    def test_mismatched_axes(self):
        v = _v(np.array([0.0, 1.0]), [1.0, 1.0])
        i = _i(np.array([0.0, 2.0]), [1.0, 1.0])
        with pytest.raises(ValueError, match='same time axis'):
            average_power(v, i)


class TestDynamicEnergyPerCycle:
    @pytest.mark.parametrize('t', [
        np.array([0., 2.]),
        np.array([0., 1., 2.]),
        np.arange(0., 2., .007),
        np.array([4., 6.]),
    ])
    def test_integrates_constant_power_to_exact_cycle_end(self, t):
        v = _v(t, np.ones_like(t))
        i = _i(t, np.ones_like(t))
        assert dynamic_energy_per_cycle(v, i, 1.5) == pytest.approx(1.5)

    def test_interpolates_power_rather_than_voltage_and_current(self):
        # Power rises linearly from 0 to 4 W over 2 s; its integral
        # over 0..1.5 s is 2.25 J. Multiplying interpolated v and i
        # instead would incorrectly give 1.6875 J.
        t = np.array([0., 2.])
        assert dynamic_energy_per_cycle(_v(t, [0., 2.]),
                                        _i(t, [0., 2.]), 1.5) == pytest.approx(2.25)

    def test_interpolated_energy_is_divided_by_complete_cycle_count(self):
        t = np.array([0., 4.])
        assert dynamic_energy_per_cycle(_v(t, [1., 1.]),
                                        _i(t, [1., 1.]), 1.5) == pytest.approx(1.5)

    @pytest.mark.parametrize('values', [[np.nan, 1.], [1., np.nan]])
    def test_nan_consumed_by_endpoint_interpolation_warns(self, values):
        t = np.array([0., 2.])
        with pytest.warns(MetricWarning, match='Integrated voltage-current'):
            assert math.isnan(dynamic_energy_per_cycle(
                _v(t, values), _i(t, [1., 1.]), 1.5))

    def test_constant_power(self):
        t = np.linspace(0, 1e-6, 1001)
        v = _v(t, np.full_like(t, 1.8))
        i = _i(t, np.full_like(t, 1e-3))
        assert math.isclose(dynamic_energy_per_cycle(v, i, period=1e-6),
                            1.8e-9, rel_tol=0.01)

    def test_invalid_period(self):
        t = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match='period must be > 0'):
            dynamic_energy_per_cycle(_v(t, [1, 1]), _i(t, [1, 1]), period=0)

    # -- TASK-61: whole-periods-only integration -------------------------
    #
    # A fractional trace must measure its complete cycles exactly, for
    # BOTH a uniform-power waveform and the pulsed (switching) waveform
    # this metric is actually for — the pulsed case is the headline
    # because it is what broke worst under the old round()-based count
    # (up to +50 % at 2.4 periods). Tolerances below account only for
    # trapezoidal discretisation of the pulse edges against the sample
    # grid, not for any bias from the fix itself.

    PERIOD = 1e-9
    RAIL = 1.8
    IDC = 1e-3      # uniform-power current
    IPULSE = 2e-3   # pulsed current
    DUTY = 0.25
    PTS_PER_PERIOD = 20_000

    @classmethod
    def _uniform_trace(cls, n_periods):
        t = np.linspace(0.0, n_periods * cls.PERIOD,
                        int(cls.PTS_PER_PERIOD * n_periods) + 1)
        v = _v(t, np.full_like(t, cls.RAIL))
        i = _i(t, np.full_like(t, cls.IDC))
        return v, i

    @classmethod
    def _pulsed_trace(cls, n_periods):
        t = np.linspace(0.0, n_periods * cls.PERIOD,
                        int(cls.PTS_PER_PERIOD * n_periods) + 1)
        phase = (t % cls.PERIOD) / cls.PERIOD
        y = np.where(phase < cls.DUTY, cls.IPULSE, 0.0)
        v = _v(t, np.full_like(t, cls.RAIL))
        i = _i(t, y)
        return v, i

    @pytest.mark.parametrize('n_periods', [2.4, 2.6, 3.4, 10.4])
    def test_fractional_trace_uniform_power_is_exact(self, n_periods):
        v, i = self._uniform_trace(n_periods)
        true_energy_per_cycle = self.RAIL * self.IDC * self.PERIOD
        got = dynamic_energy_per_cycle(v, i, self.PERIOD)
        assert got == pytest.approx(true_energy_per_cycle, rel=1e-6, abs=0)

    @pytest.mark.parametrize('n_periods', [2.4, 2.6, 3.4, 10.4])
    def test_fractional_trace_pulsed_switching_is_exact(self, n_periods):
        v, i = self._pulsed_trace(n_periods)
        true_energy_per_cycle = self.RAIL * self.IPULSE * self.DUTY * self.PERIOD
        got = dynamic_energy_per_cycle(v, i, self.PERIOD)
        # trapezoid discretises the pulse edges against the sample grid;
        # at 20,000 pts/period that residual is ~5e-5 relative, three
        # orders of magnitude below the old +/-50% bug.
        assert got == pytest.approx(true_energy_per_cycle, rel=2e-4, abs=0)

    def test_sub_period_trace_raises(self):
        t = np.linspace(0.0, 0.4 * self.PERIOD, 401)
        v = _v(t, np.full_like(t, self.RAIL))
        i = _i(t, np.full_like(t, self.IDC))
        with pytest.raises(ValueError) as exc:
            dynamic_energy_per_cycle(v, i, self.PERIOD)
        msg = str(exc.value)
        # message must name the trace length, the period, and the cycle
        # count found (decision-1: actionable caller-error diagnostics)
        assert '4e-10' in msg          # trace length (0.4 * 1 ns)
        assert '1e-09' in msg          # period
        assert '0.4 cycles' in msg     # cycle count found

    def test_whole_period_traces_unchanged_and_match_average_power(self):
        # exact identity, atol=0: these values are ~1e-12 magnitude, and
        # pytest.approx's default atol=1e-12 (or numpy's 1e-8) would make
        # any two such values compare equal regardless of the fix.
        t = np.linspace(0.0, 3 * self.PERIOD, 3 * 1000 + 1)
        v = _v(t, self.RAIL * np.sin(2 * np.pi * t / self.PERIOD) + 2.0)
        i = _i(t, self.IDC * np.sin(2 * np.pi * t / self.PERIOD + 0.3) + 5e-4)
        got = dynamic_energy_per_cycle(v, i, self.PERIOD)
        expected = average_power(v, i) * self.PERIOD
        assert got == pytest.approx(expected, rel=1e-9, abs=0)


class TestPowerSlotTypes:
    """TASK-201 — the v and i slots refuse a RECOGNISED wrong type.

    ``average_power(vout, vin)`` used to return V² labelled W in silence:
    ``_check_v_i`` checked the axis and never read ``Signal.type``, which
    the simulator fills in verbatim. It now does, under the three-valued
    rule ``silk.signals.kinds`` states for the frequency family — a
    recognised-and-wrong type CONFLICTS and raises (``decision-1``
    category 2, caller error), an unrecognised one is UNDECIDABLE and
    passes, because silk normalises nothing and a hand-built Signal
    carries whatever its author typed.
    """

    T = np.linspace(0.0, 1e-6, 101)

    @classmethod
    def _sig(cls, type_):
        """A perfectly good transient trace carrying *type_*."""
        return Signal('probe', type_, 'time', 'X',
                      cls.T, np.ones_like(cls.T))

    # -- the vocabulary, asserted as a table ----------------------------

    def test_the_recogniser_is_a_table(self):
        assert _VOLTAGE_TYPES == {'voltage', 'volt', 'volts', 'v'}
        assert _CURRENT_TYPES == {'current', 'amp', 'amps', 'i', 'a'}
        assert not (_VOLTAGE_TYPES & _CURRENT_TYPES)

    @pytest.mark.parametrize('spelling', sorted(_VOLTAGE_TYPES))
    def test_every_voltage_spelling_reads_as_voltage(self, spelling):
        assert _recognised_quantity(self._sig(spelling)) == 'voltage'

    @pytest.mark.parametrize('spelling', sorted(_CURRENT_TYPES))
    def test_every_current_spelling_reads_as_current(self, spelling):
        assert _recognised_quantity(self._sig(spelling)) == 'current'

    @pytest.mark.parametrize('written', ['V', ' Voltage ', 'CURRENT', 'I'])
    def test_case_and_whitespace_do_not_hide_a_spelling(self, written):
        assert _recognised_quantity(self._sig(written)) is not None

    # -- what is NOT in the tables passes --------------------------------

    #: Types the vocabulary has never seen. ``'voltage-density'`` is what a
    #: ``.noise`` Signal carries (V/√Hz) and ``'voltage*current'`` is what
    #: ``v * i`` composes (W): a PREFIX recogniser would call both of them
    #: voltages, which is why membership is exact.
    UNRECOGNISED = ['voltage-density', 'current-density', 'voltage*current',
                    'current*voltage', 'voltage/current', 'psd', 'signal',
                    'power', 'time', '', 'sig', None]

    @pytest.mark.parametrize('type_', UNRECOGNISED)
    def test_an_unrecognised_type_is_undecidable_and_passes(self, type_):
        """Not a lie of convenience: refusing here would break every
        hand-built Signal whose author typed something else."""
        odd = self._sig(type_)
        assert _recognised_quantity(odd) is None
        # both slots, and both metrics, take it without complaint
        assert average_power(odd, self._sig('current')) == pytest.approx(1.0)
        assert average_power(self._sig('voltage'), odd) == pytest.approx(1.0)
        assert dynamic_energy_per_cycle(odd, odd, period=1e-7) \
            == pytest.approx(1e-7)

    def test_a_composed_type_is_not_read_as_its_first_factor(self):
        """``v * i`` composes ``'voltage*current'``. Exact membership, so
        it is undecidable in BOTH slots — never a voltage in sig_i."""
        product = self._sig('voltage') * self._sig('current')
        assert product.type == 'voltage*current'
        assert _recognised_quantity(product) is None

    # -- what IS in the tables, and wrong, raises ------------------------

    @pytest.mark.parametrize('metric_name, call', [
        ('average_power', lambda v, i: average_power(v, i)),
        ('dynamic_energy_per_cycle',
         lambda v, i: dynamic_energy_per_cycle(v, i, period=1e-7)),
    ])
    def test_two_voltages_blame_sig_i(self, metric_name, call):
        """The reported defect: ``average_power(vout, vin)``."""
        with pytest.raises(ValueError) as exc:
            call(self._sig('voltage'), self._sig('voltage'))
        msg = str(exc.value)
        assert msg.startswith(f'{metric_name}: sig_i must be a current Signal')
        assert "type='voltage'" in msg           # what was found
        assert 'a recognised voltage type' in msg
        assert 'the current as sig_i' in msg     # what was wanted

    @pytest.mark.parametrize('metric_name, call', [
        ('average_power', lambda v, i: average_power(v, i)),
        ('dynamic_energy_per_cycle',
         lambda v, i: dynamic_energy_per_cycle(v, i, period=1e-7)),
    ])
    def test_two_currents_blame_sig_v(self, metric_name, call):
        with pytest.raises(ValueError) as exc:
            call(self._sig('current'), self._sig('current'))
        msg = str(exc.value)
        assert msg.startswith(f'{metric_name}: sig_v must be a voltage Signal')
        assert "type='current'" in msg
        assert 'a recognised current type' in msg
        assert 'the voltage as sig_v' in msg

    def test_a_swapped_pair_is_refused(self):
        with pytest.raises(ValueError, match='sig_v must be a voltage Signal'):
            average_power(self._sig('current'), self._sig('voltage'))

    def test_the_short_spellings_conflict_too(self):
        with pytest.raises(ValueError, match="type='V'"):
            average_power(self._sig('voltage'), self._sig('V'))
        with pytest.raises(ValueError, match="type='I'"):
            average_power(self._sig('I'), self._sig('current'))

    def test_the_axis_guard_still_runs_first(self):
        """A wrong axis is refused before the type is ever read — the
        pre-existing messages are untouched."""
        bad_axis = Signal('f', 'current', 'frequency', 'A',
                          self.T, np.ones_like(self.T))
        with pytest.raises(ValueError, match="xlabel='time'"):
            average_power(self._sig('voltage'), bad_axis)

    # -- correct input is numerically untouched --------------------------

    def test_correct_inputs_are_bit_for_bit_the_pre_fix_numbers(self):
        """MEASURED on the fixture BEFORE the guard existed and pinned
        here: ``v(out)`` -> ``'voltage'``, ``i(v1)`` -> ``'current'``,
        loaded through the real rawfile path. Exact equality, no
        tolerance — a tolerance would hide exactly the change this
        asserts is absent."""
        from pathlib import Path
        from silk.signals.rawfile import read_rawfile
        from silk.signals.spice import NgspiceSignal

        fixtures = Path(__file__).resolve().parent.parent / 'fixtures' / 'rawfiles'
        plot = read_rawfile(fixtures / 'a_tran_ascii.raw').plots[0]
        v = NgspiceSignal.from_rawfile(plot, 'out', 'V')
        i = NgspiceSignal.from_rawfile(plot, 'v1', 'I')
        assert (v.type, i.type) == ('voltage', 'current')

        period = float(v.x[-1] - v.x[0]) / 2
        assert period == 2.5e-06
        assert average_power(v, i) == -9.695880744986019e-05
        assert dynamic_energy_per_cycle(v, i, period) == -2.423970186246505e-10


# ---------------------------------------------------------------------------
# Comparator metrics
# ---------------------------------------------------------------------------

class TestHysteresis:
    def test_no_hysteresis(self):
        v_in_r = np.linspace(-1, 1, 1001)
        v_in_f = np.linspace(1, -1, 1001)
        x = np.arange(1001, dtype=float)
        h = hysteresis(
            _trans(x, v_in_r), _trans(x, np.where(v_in_r > 0, 1.8, 0.0)),
            _trans(x, v_in_f), _trans(x, np.where(v_in_f > 0, 1.8, 0.0)),
        )
        assert abs(h) < 0.01

    def test_known_hysteresis(self):
        v_in_r = np.linspace(-1, 1, 1001)
        v_in_f = np.linspace(1, -1, 1001)
        x = np.arange(1001, dtype=float)
        h = hysteresis(
            _trans(x, v_in_r), _trans(x, np.where(v_in_r > 0.1, 1.8, 0.0)),
            _trans(x, v_in_f), _trans(x, np.where(v_in_f > -0.1, 1.8, 0.0)),
        )
        assert math.isclose(h, 0.2, abs_tol=0.01)


class TestHysteresisPairing:
    """TASK-438 — the trip is an output crossing inside [start, end], by
    default the first one (n_rising/n_falling=1). When the output crosses
    more than once in a searched direction, one ``multiple_crossings``
    MetricWarning per side flags it; ``start=`` past the transient or
    ``n=-1`` (the last crossing) selects the real decision."""

    T = np.linspace(0.0, 1.0, 10001)

    def _sweeps(self, glitch=None, glitch_fall=None):
        t = self.T
        up, dn = t.copy(), 1.0 - t            # 0->1 V and 1->0 V
        out_up = np.where(up > 0.6, 1.0, 0.0)
        out_dn = np.where(dn > 0.4, 1.0, 0.0)
        if glitch is not None:                 # 20 ms high pulse
            out_up = np.where((t >= glitch) & (t < glitch + 0.02), 1.0, out_up)
        if glitch_fall is not None:            # 20 ms low pulse
            out_dn = np.where((t >= glitch_fall) & (t < glitch_fall + 0.02),
                              0.0, out_dn)
            out_dn[:1] = 1.0
        return (_trans(t, up), _trans(t, out_up),
                _trans(t, dn), _trans(t, out_dn))

    def test_clean_sweeps(self):
        assert hysteresis(*self._sweeps()) == pytest.approx(0.2, abs=1e-3)

    def test_output_glitch_default_warns_and_reads_the_glitch(self):
        sigs = self._sweeps(glitch=0.05, glitch_fall=0.45)
        with pytest.warns(MetricWarning) as rec:
            h = hysteresis(*sigs)
        assert h == pytest.approx(0.05 - 0.55, abs=1e-3)
        multi = [w for w in rec if w.message.code == 'multiple_crossings']
        assert sorted(w.message.details['signal_role'] for w in multi) == [
            'sig_out_falling', 'sig_out_rising']

    def test_glitch_resolved_by_last_crossing_or_window(self):
        sigs = self._sweeps(glitch=0.05, glitch_fall=0.45)
        assert hysteresis(*sigs, n_rising=-1, n_falling=-1) == pytest.approx(
            0.2, abs=1e-3)
        rising_only = self._sweeps(glitch=0.05)
        assert hysteresis(*rising_only, start=0.1) == pytest.approx(
            0.2, abs=1e-3)

    def test_clean_sweeps_do_not_warn(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            hysteresis(*self._sweeps())

    def test_window_excludes_startup_transient(self):
        sigs = self._sweeps(glitch=0.05)
        assert hysteresis(*sigs, start=0.1) == pytest.approx(0.2, abs=1e-3)
        assert hysteresis(*sigs, end=0.04) != pytest.approx(0.2)

    def test_missing_crossing_in_window_warns(self):
        with pytest.warns(MetricWarning) as rec:
            h = hysteresis(*self._sweeps(), end=0.3)
        assert math.isnan(h)
        assert rec[0].message.code == 'missing_crossing'

    def test_bad_arguments_raise(self):
        sigs = self._sweeps()
        for bad in (0, None, 1.5, True):
            with pytest.raises(ValueError, match='n_rising'):
                hysteresis(*sigs, n_rising=bad)
        with pytest.raises(ValueError, match='n_falling'):
            hysteresis(*sigs, n_falling=0)
        with pytest.raises(ValueError, match='end >= start'):
            hysteresis(*sigs, start=0.5, end=0.1)


class TestPropagationDelayPairing:
    """TASK-363 — the target is searched only AFTER the selected trigger, so a
    delay is never negative; every selection argument of the trigger/target
    contract, and the plural ``propagation_delays`` on the same core."""

    @staticmethod
    def _glitch():
        """TASK-295 T1: output glitch at 50-250 ps, input steps at 5 ns,
        output's real step at 5.1 ns — true delay +100 ps."""
        t = np.arange(0, 10e-9, 1e-12)
        vin = np.where(t > 5e-9, 1.0, 0.0)
        vout = np.where(t > 5.1e-9, 1.0, 0.0)
        vout[(t > 0.05e-9) & (t < 0.25e-9)] = 1.0
        return _trans(t, vin), _trans(t, vout)

    @staticmethod
    def _clock_pair(delay=0.1):
        """Input square wave, period 2, rising at 1, 3, 5, 7 (falling at 2,
        4, 6, 8); output the same shifted by ``delay``."""
        t = np.linspace(0, 8.5, 85_001)
        vin = (np.floor(t) % 2 == 1).astype(float)
        vout = (np.floor(t - delay) % 2 == 1).astype(float)
        vout[t < delay] = 0.0
        return _trans(t, vin), _trans(t, vout)

    @pytest.mark.parametrize('in_edge', ['any', 'rising'])
    def test_t1_pre_edge_glitch_is_not_paired(self, in_edge):
        sig_in, sig_out = self._glitch()
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            got = propagation_delay(sig_in, sig_out, in_edge=in_edge)
        assert got == pytest.approx(1e-10, abs=2e-12)   # 1 ps grid

    def test_no_output_crossing_after_the_trigger_is_nan_with_reason(self):
        # The output's only rising edge precedes the input's.
        t = np.linspace(0, 10, 1001)
        sig_in = _trans(t, (t > 5.0).astype(float))
        sig_out = _trans(t, ((t > 1.0) & (t < 2.0)).astype(float))
        with pytest.warns(MetricWarning) as caught:
            got = propagation_delay(sig_in, sig_out, in_threshold=.5,
                                    out_threshold=.5)
        assert math.isnan(got)
        assert len(caught) == 1
        d = caught[0].message
        assert d.code == 'missing_crossing'
        assert 'No output crossing follows the input crossing' in d.message
        assert d.details['signal_role'] == 'sig_out'
        assert d.details['trigger_time_s'] == pytest.approx(5.0, abs=0.01)

    def test_no_output_transition_is_nan_with_reason(self):
        t = np.linspace(0, 10, 1001)
        sig_in = _trans(t, (t > 5.0).astype(float))
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(propagation_delay(sig_in, _trans(t, 0 * t),
                                                in_threshold=.5,
                                                out_threshold=.5))
        assert caught[0].message.code == 'missing_crossing'
        assert caught[0].message.details['signal_role'] == 'sig_out'

    def test_default_edge_is_rising(self):
        sig_in, sig_out = self._clock_pair()
        # rising@1 -> 1.1; 'falling' first edge is at 2 -> 2.1, same delay
        assert propagation_delay(sig_in, sig_out, in_threshold=.5,
                                 out_threshold=.5) == pytest.approx(.1, abs=1e-3)

    @pytest.mark.parametrize('in_n,expected_trigger', [
        (1, 1.0), (2, 3.0), (4, 7.0), (-1, 7.0), (-4, 1.0)])
    def test_in_n_selects_the_trigger_including_from_the_end(
            self, in_n, expected_trigger):
        sig_in, sig_out = self._clock_pair()
        # out_n=2: the second rising output edge after the trigger, two
        # periods minus... i.e. trigger + 2 + delay; absent for the last.
        assert propagation_delay(sig_in, sig_out, in_threshold=.5,
                                 out_threshold=.5, in_n=in_n) == \
            pytest.approx(.1, abs=1e-3)
        if expected_trigger < 7.0:
            assert propagation_delay(
                sig_in, sig_out, in_threshold=.5, out_threshold=.5,
                in_n=in_n, out_n=2) == pytest.approx(2.1, abs=1e-3)
        else:
            with pytest.warns(MetricWarning):
                assert math.isnan(propagation_delay(
                    sig_in, sig_out, in_threshold=.5, out_threshold=.5,
                    in_n=in_n, out_n=2))

    def test_out_n_counts_after_the_trigger_not_from_t0(self):
        sig_in, sig_out = self._clock_pair()
        # Trigger = 3rd rising input edge (t=5). out_n=1 is the output's
        # edge at 5.1 — its THIRD edge from t=0, not its first (1.1).
        assert propagation_delay(sig_in, sig_out, in_threshold=.5,
                                 out_threshold=.5, in_n=3) == \
            pytest.approx(.1, abs=1e-3)
        # out_n=-1: the last output edge after the trigger (7.1).
        assert propagation_delay(sig_in, sig_out, in_threshold=.5,
                                 out_threshold=.5, in_n=3, out_n=-1) == \
            pytest.approx(2.1, abs=1e-3)
        # -2 from the end of [5.1, 7.1] is 5.1; -3 does not exist.
        assert propagation_delay(sig_in, sig_out, in_threshold=.5,
                                 out_threshold=.5, in_n=3, out_n=-2) == \
            pytest.approx(.1, abs=1e-3)
        with pytest.warns(MetricWarning):
            assert math.isnan(propagation_delay(
                sig_in, sig_out, in_threshold=.5, out_threshold=.5,
                in_n=3, out_n=-3))

    def test_edges_are_selected_per_side(self):
        sig_in, sig_out = self._clock_pair()
        kw = dict(in_threshold=.5, out_threshold=.5)
        assert propagation_delay(sig_in, sig_out, in_edge='falling', **kw) \
            == pytest.approx(.1, abs=1e-3)              # 2 -> 2.1
        assert propagation_delay(sig_in, sig_out, in_edge='rising',
                                 out_edge='falling', **kw) \
            == pytest.approx(1.1, abs=1e-3)             # 1 -> 2.1
        assert propagation_delay(sig_in, sig_out, in_edge='any', in_n=2,
                                 **kw) == pytest.approx(.1, abs=1e-3)
        assert propagation_delay(sig_in, sig_out, in_edge='falling',
                                 out_edge='any', out_n=2, **kw) \
            == pytest.approx(1.1, abs=1e-3)             # 2 -> 2.1, 3.1

    def test_window_bounds_both_searches(self):
        sig_in, sig_out = self._clock_pair()
        kw = dict(in_threshold=.5, out_threshold=.5)
        # The first rising input edge inside [2.5, 9] is at 3.
        assert propagation_delay(sig_in, sig_out, start=2.5, **kw) == \
            pytest.approx(.1, abs=1e-3)
        assert propagation_delay(sig_in, sig_out, start=2.5, in_n=-1,
                                 **kw) == pytest.approx(.1, abs=1e-3)
        # end=7.05 keeps the trigger at 7 but cuts its target at 7.1.
        with pytest.warns(MetricWarning) as caught:
            assert math.isnan(propagation_delay(sig_in, sig_out, end=7.05,
                                                in_n=-1, **kw))
        assert caught[0].message.details['window_end'] == 7.05
        # The automatic threshold is the swing inside the window.
        assert propagation_delay(sig_in, sig_out, start=0.5, end=3.5) == \
            pytest.approx(.1, abs=1e-3)

    def test_a_delay_is_never_negative(self):
        sig_in, sig_out = self._clock_pair(delay=-0.3)  # output leads
        for in_n in (1, 2, 3, -1):
            got = propagation_delay(sig_in, sig_out, in_threshold=.5,
                                    out_threshold=.5, in_n=in_n)
            assert math.isnan(got) or got >= 0

    def test_explicit_thresholds_are_independent(self):
        t = np.linspace(0, 10, 10_001)
        sig_in = _trans(t, np.clip(t - 2.0, 0, 1))            # 0.2 at 2.2
        sig_out = _trans(t, 1.8 * (t > 3.0).astype(float))    # 0.9 at 3.0
        assert propagation_delay(sig_in, sig_out, in_threshold=0.2,
                                 out_threshold=0.9) == \
            pytest.approx(0.8, abs=2e-3)

    @pytest.mark.parametrize('kwargs,match', [
        ({'in_n': 0}, 'in_n must be a nonzero integer'),
        ({'in_n': None}, 'in_n must be a nonzero integer'),
        ({'in_n': 1.0}, 'in_n must be a nonzero integer'),
        ({'out_n': 0}, 'out_n must be a nonzero integer'),
        ({'out_n': True}, 'out_n must be a nonzero integer'),
        ({'start': 5.0, 'end': 4.0}, 'end >= start'),
        ({'start': 100.0}, 'does not overlap'),
        ({'in_edge': 'up'}, 'in_edge must be'),
        ({'out_edge': 'up'}, 'out_edge must be'),
    ])
    def test_caller_errors_raise(self, kwargs, match):
        sig_in, sig_out = self._clock_pair()
        with pytest.raises(ValueError, match=match):
            propagation_delay(sig_in, sig_out, **kwargs)

    def test_old_arguments_are_gone(self):
        sig_in, sig_out = self._clock_pair()
        for old in ({'threshold': .5}, {'edge': 'rising'}):
            with pytest.raises(TypeError):
                propagation_delay(sig_in, sig_out, **old)

    def test_comparator_prop_delay_is_removed(self):
        import silk.metrics as M
        import silk.metrics.timing as T
        from silk.signals.schema import export_metric_schema
        assert not hasattr(M, 'comparator_prop_delay')
        assert not hasattr(T, 'comparator_prop_delay')
        assert 'comparator_prop_delay' not in export_metric_schema()


class TestPropagationDelays:
    """The plural: every trigger edge, same core, unregistered."""

    _pair = staticmethod(TestPropagationDelayPairing._clock_pair)

    def test_one_delay_per_trigger_edge(self):
        sig_in, sig_out = self._pair()
        got = propagation_delays(sig_in, sig_out, in_threshold=.5,
                                 out_threshold=.5)
        assert isinstance(got, np.ndarray) and got.shape == (4,)
        assert got == pytest.approx([.1] * 4, abs=1e-3)
        # element k belongs to crossings(sig_in, ...)[k]
        assert len(crossings(sig_in, threshold=.5)) == len(got)

    def test_each_element_equals_the_scalar_with_that_in_n(self):
        sig_in, sig_out = self._pair()
        kw = dict(in_threshold=.5, out_threshold=.5, in_edge='any',
                  out_n=2, start=0.5, end=8.5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', MetricWarning)
            many = propagation_delays(sig_in, sig_out, **kw)
            one = [propagation_delay(sig_in, sig_out, in_n=k + 1, **kw)
                   for k in range(len(many))]
        np.testing.assert_array_equal(many, one)

    def test_unpaired_trigger_is_nan_in_its_slot_with_one_warning(self):
        sig_in, sig_out = self._pair()
        with pytest.warns(MetricWarning) as caught:
            got = propagation_delays(sig_in, sig_out, in_threshold=.5,
                                     out_threshold=.5, out_n=2)
        assert got[:3] == pytest.approx([2.1] * 3, abs=1e-3)
        assert math.isnan(got[3])
        assert len(caught) == 1
        d = caught[0].message.details
        assert d['unpaired_trigger_count'] == 1 and d['trigger_count'] == 4
        assert d['trigger_time_s'] == pytest.approx(7.0, abs=1e-3)

    def test_pre_edge_glitch_is_not_paired(self):
        sig_in, sig_out = TestPropagationDelayPairing._glitch()
        got = propagation_delays(sig_in, sig_out, in_edge='any')
        assert got == pytest.approx([1e-10], abs=2e-12)

    def test_no_trigger_is_empty_and_quiet(self):
        t = np.linspace(0, 1, 11)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            got = propagation_delays(_trans(t, 0 * t), _trans(t, t))
        assert isinstance(got, np.ndarray) and got.shape == (0,)

    def test_consumed_nan_is_the_crossings_sentinel(self):
        t = np.linspace(0, 9, 91)
        y = (np.floor(t) % 2 == 1).astype(float)
        y[50] = np.nan
        with pytest.warns(MetricWarning) as caught:
            got = propagation_delays(_trans(t, y), _trans(t, y),
                                     in_threshold=.5, out_threshold=.5)
        assert got.shape == (1,) and math.isnan(got[0])
        assert caught[0].message.code == 'nan_waveform'

    def test_ignores_no_argument_and_has_no_in_n(self):
        sig_in, sig_out = self._pair()
        with pytest.raises(TypeError):
            propagation_delays(sig_in, sig_out, in_n=1)

    def test_scalar_is_registered_and_plural_is_not(self):
        from silk.signals.schema import export_metric_schema
        schema = export_metric_schema()
        assert 'propagation_delay' in schema
        assert 'propagation_delays' not in schema
        params = {p['name'] for p in schema['propagation_delay']['parameters']}
        assert params == {'in_threshold', 'out_threshold', 'in_edge',
                          'out_edge', 'in_n', 'out_n', 'start', 'end'}


class TestRemovedMetrics:
    """TASK-412: three metrics removed outright — no alias, no deprecation."""

    NAMES = ('input_referred_offset', 'kickback', 'hold_droop')

    @pytest.mark.parametrize('name', NAMES)
    def test_gone_everywhere(self, name):
        import silk.metrics
        import silk.metrics.timing
        from silk.signals.schema import export_metric_schema
        assert not hasattr(silk.metrics, name)
        assert not hasattr(silk.metrics.timing, name)
        assert name not in silk.metrics.__all__
        assert name not in export_metric_schema()


class TestImports:
    def test_top_level(self):
        from silk.metrics import (
            crossings, rise_time, period_jitter, average_power,
        )
        assert callable(crossings)


# ---------------------------------------------------------------------------
# General waveform primitives (vendor-parity additions)
# ---------------------------------------------------------------------------

class TestAverage:
    def test_constant(self):
        t = np.linspace(0, 1, 1001)
        assert math.isclose(average(_trans(t, np.full_like(t, 2.5))),
                            2.5, rel_tol=1e-12)

    def test_sine_zero(self):
        # Full integer number of periods → time-weighted mean ≈ 0.
        t = np.linspace(0, 1, 100001)
        y = np.sin(2 * np.pi * 5 * t)
        assert abs(average(_trans(t, y))) < 1e-3

    def test_window_subset(self):
        t = np.linspace(0, 10, 10001)
        y = np.where(t < 5, 0.0, 4.0)
        # Window [5, 10]: y is constant 4 → average = 4
        assert math.isclose(average(_trans(t, y), start=5.0, end=10.0),
                            4.0, rel_tol=1e-3)

    def test_invalid_window(self):
        t = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match='end > start'):
            average(_trans(t, t.copy()), start=0.5, end=0.5)

    def test_works_on_freq_signal(self):
        # Axis-agnostic: works on any Signal regardless of xlabel.
        sig = Signal('h', 'voltage', 'frequency', 'V',
                     np.array([1.0, 2.0, 3.0]),
                     np.array([1.0, 2.0, 3.0]))
        # Time-weighted mean of y=x over [1,3] = 2.0
        assert math.isclose(average(sig), 2.0, rel_tol=1e-12)


class TestRms:
    def test_constant(self):
        t = np.linspace(0, 1, 1001)
        assert math.isclose(rms(_trans(t, np.full_like(t, 3.0))),
                            3.0, rel_tol=1e-12)

    def test_sine(self):
        # RMS of unit-amplitude sine = 1/√2.
        t = np.linspace(0, 1, 100001)
        y = np.sin(2 * np.pi * 5 * t)
        assert math.isclose(rms(_trans(t, y)), 1.0 / math.sqrt(2.0),
                            rel_tol=0.01)


class TestStddev:
    def test_constant_zero(self):
        t = np.linspace(0, 1, 1001)
        assert stddev(_trans(t, np.full_like(t, 5.0))) < 1e-12

    def test_sine(self):
        # std of unit sine over many periods = 1/√2 (same as RMS for zero-mean).
        t = np.linspace(0, 1, 100001)
        y = np.sin(2 * np.pi * 5 * t)
        assert math.isclose(stddev(_trans(t, y)), 1.0 / math.sqrt(2.0),
                            rel_tol=0.01)


class TestPeak:
    def test_max_abs(self):
        sig = _trans([0, 1, 2, 3], [-3.0, 1.0, 2.0, -0.5])
        assert peak(sig) == 3.0

    def test_window(self):
        sig = _trans([0, 1, 2, 3], [-3.0, 1.0, 2.0, -0.5])
        # Window [1,3] excludes the −3
        assert peak(sig, start=1.0, end=3.0) == 2.0


class TestPeakToPeak:
    def test_basic(self):
        sig = _trans([0, 1, 2, 3], [-1.0, 0.5, 2.0, -0.5])
        assert peak_to_peak(sig) == 3.0   # 2 − (−1)


class TestWindowedMinMax:
    def test_min_max(self):
        sig = _trans([0, 1, 2, 3], [-1.0, 0.5, 2.0, -0.5])
        assert minimum(sig) == -1.0
        assert maximum(sig) == 2.0


class TestArgminArgmax:
    def test_argmax_x(self):
        sig = _trans([0, 1, 2, 3], [0.0, 0.5, 2.0, 1.0])
        assert argmax_x(sig) == 2.0

    def test_argmin_x(self):
        sig = _trans([0, 1, 2, 3], [1.0, -2.0, 0.0, 0.5])
        assert argmin_x(sig) == 1.0


class TestIntegral:
    def test_constant(self):
        # ∫ 1 dx from 0 to 5 = 5
        t = np.linspace(0, 5, 5001)
        assert math.isclose(integral(_trans(t, np.ones_like(t))), 5.0,
                            rel_tol=1e-9)

    def test_sine_full_cycles_zero(self):
        t = np.linspace(0, 1, 100001)
        y = np.sin(2 * np.pi * 5 * t)
        assert abs(integral(_trans(t, y))) < 1e-3

    def test_window(self):
        t = np.linspace(0, 10, 10001)
        y = np.full_like(t, 2.0)
        # Window [2, 7] → ∫ 2 dx = 10
        assert math.isclose(integral(_trans(t, y), start=2.0, end=7.0),
                            10.0, rel_tol=1e-3)


class TestRunningIntegral:
    def test_returns_signal(self):
        t = np.linspace(0, 5, 5001)
        sig = _trans(t, np.ones_like(t))
        ri = running_integral(sig)
        assert isinstance(ri, Signal)
        assert ri.xlabel == sig.xlabel
        assert ri.x[0] == 0.0
        assert ri.y[0] == 0.0
        assert math.isclose(float(ri.y[-1]), 5.0, rel_tol=1e-9)

    def test_against_definite_integral(self):
        # Last value of the running integral = definite integral.
        t = np.linspace(0, 1, 10001)
        y = t.copy() * 2.0     # ramp; ∫₀¹ 2t dt = 1
        sig = _trans(t, y)
        assert math.isclose(float(running_integral(sig).y[-1]),
                            integral(sig), rel_tol=1e-9)


class TestDerivative:
    def test_empty_keeps_shared_empty_input_error(self):
        sig = _trans([], [], name='vout')
        with pytest.raises(
                ValueError,
                match=(r"derivative: Signal 'vout' has no samples "
                       r"\(len\(x\) == 0\)")):
            derivative(sig)

    def test_one_sample_raises_actionable_value_error(self):
        sig = _trans([2.0], [3.0], name='vout')
        with pytest.raises(
                ValueError,
                match=(r"derivative: Signal 'vout' has 1 sample; "
                       r"a derivative needs at least 2 points")):
            derivative(sig)

    def test_two_samples_keep_endpoint_difference(self):
        d = derivative(_trans([0.0, 2.0], [1.0, 5.0]))
        assert np.array_equal(d.x, np.array([0.0, 2.0]))
        assert np.array_equal(d.y, np.array([2.0, 2.0]))

    def test_constant_zero(self):
        t = np.linspace(0, 5, 5001)
        d = derivative(_trans(t, np.full_like(t, 3.0)))
        assert np.allclose(d.y, 0.0, atol=1e-9)

    def test_linear_ramp_unity(self):
        t = np.linspace(0, 5, 5001)
        d = derivative(_trans(t, t.copy()))
        # dy/dx = 1 for y=x; check interior
        assert np.allclose(d.y[5:-5], 1.0, rtol=1e-6)

    def test_returns_signal(self):
        t = np.linspace(0, 1, 11)
        d = derivative(_trans(t, t.copy()))
        assert isinstance(d, Signal)
        assert d.xlabel == 'time'

    def test_complex_y_differentiates_real_and_imaginary_together(self):
        """FIXED (TASK-106): pre-fix, ``derivative`` did
        ``np.asarray(sig.y, dtype=float)`` unconditionally, so a complex
        Signal raised ``ComplexWarning`` and came back as d/dx(Re(y))
        only — here that would have been the derivative of the ramp
        (1.0 everywhere) with the ``-2*t`` imaginary part silently
        dropped. Now the complex gradient comes back whole, real part
        included, no warning."""
        t = np.linspace(0, 5, 5001)
        y = t.copy() + 1j * (-2.0 * t)
        sig = Signal('s', 'voltage', 'time', 'V', t, y)
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            d = derivative(sig)
        assert np.iscomplexobj(d.y)
        assert np.allclose(d.y.real[5:-5], 1.0, rtol=1e-6)
        assert np.allclose(d.y.imag[5:-5], -2.0, rtol=1e-6)

    def test_real_y_stays_real(self):
        t = np.linspace(0, 5, 5001)
        d = derivative(_trans(t, t.copy()))
        assert d.y.dtype == np.float64


class TestValueAt:
    def test_interpolation(self):
        sig = _trans([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        assert math.isclose(value_at(sig, 0.5), 0.5, rel_tol=1e-12)
        assert math.isclose(value_at(sig, 1.5), 2.5, rel_tol=1e-12)

    def test_outside_range_nan(self):
        sig = _trans([0.0, 1.0], [0.0, 1.0])
        assert math.isnan(value_at(sig, 5.0))
        assert math.isnan(value_at(sig, -1.0))


class TestCrossing:
    def test_first_rising(self):
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, t.copy())
        assert math.isclose(crossing(sig, threshold=5.0, edge='rising'),
                            5.0, abs_tol=0.02)

    def test_nth_crossing(self):
        # sin(2π·t/2) starts at exactly 0 (not a crossing — must come from
        # below). Detected rising 0-crossings are at t = 2, 4, 6, 8.
        t = np.linspace(0, 10, 100001)
        sig = _trans(t, np.sin(2 * np.pi * t / 2.0))
        assert math.isclose(crossing(sig, threshold=0.0, edge='rising', n=1),
                            2.0, abs_tol=0.01)
        assert math.isclose(crossing(sig, threshold=0.0, edge='rising', n=2),
                            4.0, abs_tol=0.01)
        assert math.isclose(crossing(sig, threshold=0.0, edge='rising', n=4),
                            8.0, abs_tol=0.01)

    def test_no_crossing_returns_nan(self):
        t = np.linspace(0, 1, 100)
        sig = _trans(t, np.zeros_like(t))
        assert math.isnan(crossing(sig, threshold=5.0, edge='any'))

    def test_invalid_edge(self):
        sig = _trans([0, 1], [0.0, 1.0])
        with pytest.raises(ValueError, match='rising|falling|any'):
            crossing(sig, threshold=0.5, edge='sideways')

    def test_invalid_n(self):
        sig = _trans([0, 1], [0.0, 1.0])
        with pytest.raises(ValueError, match='n must be a nonzero integer'):
            crossing(sig, threshold=0.5, n=0, edge='any')

    def test_works_on_freq_signal(self):
        # Axis-agnostic: gain crosses 0 dB at some frequency.
        sig = Signal('mag', 'voltage', 'frequency', 'dB',
                     np.array([1.0, 10.0, 100.0, 1000.0]),
                     np.array([20.0, 10.0, 0.0, -10.0]))
        # Falling crossing of 0 dB
        f = crossing(sig, threshold=0.0, edge='falling')
        assert math.isclose(f, 100.0, abs_tol=1.0)


class TestComplexAnyConvention:
    """Ordering needs projection; defined complex metrics preserve their values."""

    CASES = [
        ('average', {}, 1), ('rms', {}, math.sqrt(1.5)),
        ('stddev', {}, math.sqrt(0.5)), ('variance', {}, 0.5),
        ('peak', {}, 2), ('peak_to_peak', {}, 2),
        ('minimum', {}, 0), ('maximum', {}, 2),
        ('argmin_x', {'refine': True}, 1),
        ('argmax_x', {'refine': True}, 3),
        ('integral', {}, 2), ('value_at', {'x': 2}, 1),
        ('final_value', {}, 2), ('crossing', {}, 2),
    ]

    @staticmethod
    def ac_signal():
        return Signal('h', 'voltage', 'frequency', 'V',
                      np.array([1., 2., 3.]), np.array([0j, 3+4j, 6+8j]))

    @pytest.mark.parametrize('name,params,expected', [c for c in CASES if c[0] in ('peak', 'peak_to_peak', 'minimum', 'maximum', 'argmin_x', 'argmax_x', 'crossing')])
    @pytest.mark.parametrize('zero_imag', [False, True])
    def test_complex_input_refused(self, name, params, expected, zero_imag):
        import silk.metrics as M
        sig = self.ac_signal()
        if zero_imag:
            sig = Signal('h', 'voltage', 'frequency', 'V', sig.x,
                         sig.y.real.astype(complex))
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            with pytest.raises(ValueError) as caught:
                getattr(M, name)(sig, **params)
        assert type(caught.value) is ValueError
        message = str(caught.value)
        assert name in message
        assert 'complex' in message
        for choice in ('.real()', '.mag()', '.imag()'):
            assert choice in message

    @pytest.mark.parametrize('name,params,expected', CASES)
    @pytest.mark.parametrize('projection,scale', [('real', 3), ('imag', 4), ('mag', 5)])
    @pytest.mark.parametrize('deferred', [False, True])
    def test_explicit_projection(self, name, params, expected, projection, scale, deferred):
        import silk.metrics as M
        from types import SimpleNamespace
        sig = self.ac_signal()
        if deferred:
            # Resolve actual probe arithmetic against a loaded AC result.
            class Bench:
                test_name = 'ac'
            bench = Bench()
            unit = Signal('one', 'voltage', 'frequency', 'V', sig.x,
                          np.ones(3, dtype=complex))
            bench.netlist = SimpleNamespace(
                sim_results_loaded=True,
                load_signal=lambda name, selector: {'a': sig, 'b': unit}[name])
            a = Signal._unbound('a', bench=bench)
            b = Signal._unbound('b', bench=bench)
            projected = getattr(a / b, projection)()
        else:
            projected = getattr(sig, projection)()
        # Hand-integrated three-point ramp; the trapezoid gives E[y²]=1.5*k².
        if name == 'variance':
            expected *= scale ** 2
        elif name not in ('argmin_x', 'argmax_x', 'crossing'):
            expected *= scale
        params = dict(params)
        if name == 'crossing':
            params['threshold'] = scale
        assert getattr(M, name)(projected, **params) == pytest.approx(expected)

    def test_complex_rms_preserves_mean_squared_magnitude(self):
        # |y| = [0, 5, 10]; trapezoidal mean square is 37.5.
        assert rms(self.ac_signal()) == pytest.approx(math.sqrt(37.5))
        assert rms(self.ac_signal()) == rms(self.ac_signal().mag())

    def test_real_rms_interpolates_signed_samples(self):
        # Signed linear edge [-1, 1] has midpoint zero, not magnitude one.
        sig = Signal('r', 'voltage', 'frequency', 'V',
                     np.array([1., 3.]), np.array([-1., 1.]))
        assert rms(sig, start=2) == pytest.approx(math.sqrt(0.5))

    def test_out_of_range_complex_point_returns_nan(self):
        with pytest.warns(UserWarning, match='outside'):
            assert np.isnan(value_at(self.ac_signal(), 100))

    @pytest.mark.parametrize('kind', ['max', 'rms', 'mean_abs'])
    def test_error_norms_preserve_complex_difference(self, kind):
        # Equal magnitudes, opposite phases: |sig-goal|=10, not zero.
        x = np.array([1., 2., 3.])
        sig = Signal('a', 'voltage', 'frequency', 'V', x,
                     np.full(3, 3+4j))
        goal = Signal('b', 'voltage', 'frequency', 'V', x[[0, 2]],
                      np.full(2, -3-4j))
        assert error_norm(sig, goal, kind=kind) == pytest.approx(10)
        assert relative_error_norm(sig, goal, kind=kind, floor=0) == pytest.approx(2)


class TestNewPrimitivesImports:
    def test_top_level(self):
        from silk.metrics import (
            average, rms, stddev, peak, peak_to_peak,
            minimum, maximum, argmin_x, argmax_x,
            integral, running_integral, derivative, value_at, crossing,
        )
        assert callable(average)
        assert callable(running_integral)


# ---------------------------------------------------------------------------
# Tier 2 — vendor-parity transient additions
# ---------------------------------------------------------------------------

class TestInstFrequency:
    def test_constant_frequency_clock(self):
        # 100 Hz clock for 1 second → instantaneous freq ≈ 100 throughout.
        t = np.linspace(0, 1.0, 100001)
        y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
        f_sig = cycle_frequency(_trans(t, y))
        assert isinstance(f_sig, Signal)
        assert f_sig.xlabel == 'time'
        # Most samples should be near 100 Hz (some quantization from grid).
        assert math.isclose(np.median(f_sig.y), 100.0, rel_tol=0.02)

    def test_few_edges_returns_empty(self):
        t = np.linspace(0, 1, 100)
        f_sig = cycle_frequency(_trans(t, np.zeros_like(t)))
        assert len(f_sig.x) == 0


class TestOnTimeOffTime:
    def test_50pct_square(self):
        t = np.linspace(0, 1, 100001)
        y = (np.sin(2 * np.pi * 5 * t) > 0).astype(float)
        on  = on_time(_trans(t, y), threshold=0.5)
        off = off_time(_trans(t, y), threshold=0.5)
        assert math.isclose(on,  0.5, abs_tol=0.01)
        assert math.isclose(off, 0.5, abs_tol=0.01)

    def test_always_high(self):
        t = np.linspace(0, 1, 1001)
        y = np.full_like(t, 2.0)
        assert math.isclose(on_time(_trans(t, y), threshold=0.5), 1.0, rel_tol=1e-6)
        assert math.isclose(off_time(_trans(t, y), threshold=0.5), 0.0, abs_tol=1e-6)

    def test_always_low(self):
        t = np.linspace(0, 1, 1001)
        y = np.zeros_like(t)
        assert on_time(_trans(t, y), threshold=0.5) == 0.0
        assert math.isclose(off_time(_trans(t, y), threshold=0.5), 1.0, rel_tol=1e-6)


# --- BEGIN commented-out (TASK-226.11): eye_diagram fixtures and tests,
# superseded by TASK-226.12+. Kept for reference until the replacement
# is accepted; do not delete. ---
# # ---------------------------------------------------------------------------
# # eye_diagram — fixtures shared by every eye test below.
# # ---------------------------------------------------------------------------
#
# _PS = 1e-12
#
#
# def _ramp(x, t0, t1):
#     """A finite 0 -> 1 linear edge: 0 before ``t0``, 1 after ``t1``."""
#     return np.clip((x - t0) / (t1 - t0), 0.0, 1.0)
#
#
# def _edge_sig(y_of_x, span=400 * _PS, n=8001):
#     x = np.linspace(0.0, span, n)
#     return _trans(x, y_of_x(x))
#
#
# def _pwl_sig(edges, span, t_origin=0.0, edge_w=4 * _PS, dt=0.125 * _PS,
#              base=0.0):
#     """A piecewise-linear trace with one finite edge per requested
#     mid-rail crossing.
#
#     ``edges`` is a sequence of ``(t_cross, 'rise'|'fall')`` in absolute
#     time. Each edge is a linear ramp of width ``edge_w`` centred on
#     ``t_cross``, so the 0.5 crossing sits exactly on ``t_cross``. Every
#     ramp knot is a multiple of ``dt``, so the eye's own resampling reads
#     the ramp on its own straight line and recovers ``t_cross`` exactly.
#     ``base`` is the level the trace starts from, so a trace that idles
#     HIGH and dips is expressible too.
#     """
#     n = int(round(span / dt)) + 1
#     x = t_origin + np.linspace(0.0, span, n)
#     y = np.full_like(x, float(base))
#     for t_cross, direction in edges:
#         r = _ramp(x, t_cross - edge_w / 2, t_cross + edge_w / 2)
#         y = y + r if direction == 'rise' else y - r
#     return _trans(x, y)
#
#
# def _prbs7(n_bits):
#     """PRBS7, Fibonacci form, cleared state — the same stream
#     ``silk.examples.prbs_generator_example.prbs_reference`` produces.
#     Reimplemented in six lines so the metric tests depend on no example.
#     """
#     q = [0] * 7
#     out = []
#     for _ in range(n_bits):
#         fb = 1 - (q[6] ^ q[5])
#         q = [fb] + q[:6]
#         out.append(q[6])
#     return out
#
#
# def _rc_prbs_state(tau_r_ui, tau_f_ui, ui, n_bits):
#     """Per-bit state of a PRBS7 NRZ stream driving a single real pole.
#
#     Returns ``(bits, v0, tau)``: the transmitted level of each bit, the
#     channel output at the START of each bit, and the time constant in
#     force during it. Within bit ``n`` the response is then exactly
#     ``b_n + (v0_n − b_n)·e^(−(t−n·UI)/τ_n)``. Both the sampled fixture
#     and the closed-form reference below are built from this one
#     function, so they cannot drift apart into two different waveforms.
#
#     ``tau_f_ui`` gives the falling edge its own time constant. Left
#     ``None`` the channel is symmetric; set to something other than
#     ``tau_r_ui`` and the one and zero distributions stop being mirror
#     images, which is what moves the decision threshold off the mid-rail.
#     """
#     bits = np.asarray(_prbs7(n_bits), dtype=float)
#     tau_r = tau_r_ui * ui
#     tau_f = tau_r if tau_f_ui is None else tau_f_ui * ui
#     v0 = np.empty(n_bits)
#     tau = np.empty(n_bits)
#     v = 0.0
#     for i, b in enumerate(bits):
#         v0[i] = v
#         tau[i] = tau_r if b > v else tau_f
#         v = b + (v - b) * math.exp(-ui / tau[i])
#     return bits, v0, tau
#
#
# def _rc_prbs_sig(*, tau_r_ui=0.35, tau_f_ui=None, ui=200 * _PS,
#                  n_bits=127, src_per_ui=256, t_origin=0.0):
#     """A PRBS7 NRZ stream through a single real pole, sampled from the
#     closed form rather than from a discretised filter, so the trace is
#     exact at every sample.
#
#     This is the inter-symbol-interference fixture the eye tests measure:
#     the opening closes as ``τ`` grows, and — the property that matters
#     here — the opening profile rises MONOTONICALLY through the bit right
#     up to the next transition, because the channel is causal and
#     lag-dominated.
#     """
#     bits, v0, tau = _rc_prbs_state(tau_r_ui, tau_f_ui, ui, n_bits)
#     n = n_bits * src_per_ui + 1
#     x = np.arange(n) * (ui / src_per_ui)
#     k = np.clip((x / ui).astype(int), 0, n_bits - 1)
#     y = bits[k] + (v0[k] - bits[k]) * np.exp(-(x - k * ui) / tau[k])
#     return _trans(x + t_origin, y)
#
#
# def _rc_prbs_reference(*, tau_r_ui=0.35, tau_f_ui=None, ui=200 * _PS,
#                        n_bits=127, threshold=None, max_iter=200):
#     """The eye of :func:`_rc_prbs_sig`, solved in CLOSED FORM.
#
#     This is the expectation the metric is checked against, and it is
#     computed without resampling anything and without calling into
#     ``eye_diagram``. Every bit's segment is monotone, so its crossing of
#     a level ``L`` is one logarithm::
#
#         s = τ_n · ln((v0_n − b_n) / (L − b_n))     when 0 < ratio <= 1
#
#     and the value at any phase is one exponential. The eye is then the
#     widest gap between those phases on the circle, the instant is its
#     midpoint, and the threshold is solved with them exactly as the
#     metric solves it — the same definition, on exact data instead of a
#     128-point fold.
#
#     ``threshold`` pins the level instead of solving for it, which is how
#     the "one pass at the mid-rail" comparison is produced.
#
#     Returns a dict with ``threshold``, ``width``, ``instant``,
#     ``height``, ``jitter`` and ``n_windows``.
#     """
#     bits, v0, tau = _rc_prbs_state(tau_r_ui, tau_f_ui, ui, n_bits)
#
#     def phases_at(level):
#         out = []
#         for i in range(n_bits):
#             b, a = bits[i], v0[i]
#             if a == b:
#                 continue
#             ratio = (level - b) / (a - b)
#             if 0.0 < ratio <= 1.0:
#                 s = tau[i] * math.log(1.0 / ratio)
#                 if 0.0 <= s < ui:
#                     out.append(s)
#         return np.asarray(out, dtype=float)
#
#     def column_at(phase):
#         return bits + (v0 - bits) * np.exp(-phase / tau)
#
#     def widest_gap(ph):
#         p = np.sort(ph)
#         if len(p) == 1:
#             return float(p[0]), float(ui)
#         g = np.empty(len(p))
#         g[:-1] = p[1:] - p[:-1]
#         g[-1] = p[0] + ui - p[-1]
#         j = int(np.argmax(g))
#         return float(p[j]), float(g[j])
#
#     level = 0.5 if threshold is None else float(threshold)
#     if threshold is None:
#         for _ in range(max_iter):
#             ph = phases_at(level)
#             if len(ph) == 0:
#                 break
#             left, gap = widest_gap(ph)
#             col = column_at((left + 0.5 * gap) % ui)
#             ones, zeros = col[col > level], col[col <= level]
#             if ones.size == 0 or zeros.size == 0:
#                 break
#             nxt = 0.5 * (float(ones.min()) + float(zeros.max()))
#             if abs(nxt - level) <= 1e-15:
#                 level = nxt
#                 break
#             level = nxt
#
#     ph = phases_at(level)
#     left, gap = widest_gap(ph)
#     instant = (left + 0.5 * gap) % ui
#     col = column_at(instant)
#     ones, zeros = col[col > level], col[col <= level]
#     centre = (instant + 0.5 * ui) % ui
#     err = (ph - centre + 0.5 * ui) % ui - 0.5 * ui
#     return {
#         'threshold': level,
#         'width': gap,
#         'instant': instant,
#         'height': float(ones.min() - zeros.max()),
#         'jitter': float(np.std(err, ddof=1)) if len(ph) > 1 else 0.0,
#         'n_windows': len(ph),
#     }
#
#
# def _square_sig(*, span=8.0, n=32001, first=0.125, step=0.25, width=0.01):
#     """A finite-edge square wave whose crossings land on EXACTLY
#     representable times.
#
#     ``0.125 + k/4`` and the ``period/128`` resampling pitch of a period
#     of 1.0 are all binary-exact, so the folded crossing phases come out
#     as the exact values 0.125, 0.375, 0.625 and 0.875 and the four gaps
#     between them are exactly equal floats. That is what makes a genuine
#     argmax TIE testable; gaps that merely look equal in decimal are not
#     equal in binary and resolve on rounding noise instead.
#     """
#     x = np.linspace(0.0, span, n)
#     y = np.zeros_like(x)
#     level = 0
#     t_cross = first
#     while t_cross < span:
#         r = _ramp(x, t_cross - width / 2, t_cross + width / 2)
#         y = y + r if level == 0 else y - r
#         level = 1 - level
#         t_cross += step
#     return _trans(x, y)
#
#
# def _clip(sig, t_from):
#     """The same waveform with everything before ``t_from`` cut away.
#
#     Absolute times are untouched, so the waveform is identical and only
#     ``sig.x[0]`` — the fold origin — moves.
#     """
#     i = int(np.searchsorted(np.asarray(sig.x, dtype=float), t_from))
#     return _trans(np.asarray(sig.x)[i:], np.asarray(sig.y)[i:])
#
#
# def _fold_rows(sig, result, period):
#     """The folded rows the metric measured: the returned ``traces`` plus
#     each window's closing boundary sample (TASK-226.1), interpolated
#     by hand from the original waveform without calling into the metric.
#     This helper belongs to the retired resampled-eye tests below."""
#     endpoint = np.array([
#         float(np.interp(float(sig.x[0]) + (k + 1) * period, sig.x, sig.y))
#         for k in range(result.traces.shape[0])
#     ])
#     return np.hstack([result.traces, endpoint[:, None]])
#
#
# def _column_at(rows, search_t, t_sample):
#     """One value per window, linearly interpolated at ``t_sample``."""
#     i = int(np.searchsorted(search_t, t_sample, side='right')) - 1
#     i = min(max(i, 0), len(search_t) - 2)
#     f = (t_sample - search_t[i]) / (search_t[i + 1] - search_t[i])
#     return rows[:, i] * (1.0 - f) + rows[:, i + 1] * f
#
#
# def _opening_profile(rows, search_t):
#     """Mid-rail opening at every column of the fold, for locating the
#     WIDEST column — which is deliberately not what the metric samples."""
#     out = []
#     for i in range(rows.shape[1]):
#         v = rows[:, i]
#         mid = 0.5 * (float(v.min()) + float(v.max()))
#         hi, lo = v[v > mid], v[v <= mid]
#         out.append(float(hi.min() - lo.max()) if hi.size and lo.size
#                    else float('nan'))
#     return np.asarray(out)
#
#
# def _recovered_threshold(rows, search_t, result):
#     """The decision threshold the metric converged on, read back out of
#     the returned fold.
#
#     ``height`` is symmetric about that threshold by construction, so the
#     threshold is the midpoint of the one gap in the sorted column values
#     whose size is ``height``.
#     """
#     col = np.sort(_column_at(rows, search_t, result.sampling_instant_s))
#     gaps = np.diff(col)
#     i = int(np.argmin(np.abs(gaps - result.height)))
#     return 0.5 * (col[i] + col[i + 1])
#
#
# class TestEyeDiagram:
#     """TASK-226.8 — the eye is what lies between the two crossing
#     clusters, and every number comes from those crossings."""
#
#     def test_clean_clock(self):
#         """A 100 Hz square wave folded on its BIT period.
#
#         Rewritten for TASK-226.8. It used to fold this clock on 0.01 s,
#         which is TWO bits of a 1010 stream and therefore puts two
#         transition clusters and two openings in one window; see
#         :meth:`test_folding_at_two_unit_intervals_takes_the_widest_opening`
#         for what the metric now does with that. The unit interval is
#         0.005 s, and on it the clock has one transition per window and
#         one eye.
#
#         The edges of ``(sin > 0)`` land exactly on the input grid, so
#         each resolves within one resampling cell of the boundary and the
#         gap comes out one cell short of the full period:
#
#             width = 0.005 − time_resolution_s = 4.921875e-03
#             sampling instant = 2.5e-03, i.e. the middle of the bit
#         """
#         t = np.linspace(0, 1.0, 100001)
#         y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
#         result = eye_diagram(_trans(t, y), period=0.005,
#                              samples_per_period=64)
#         assert isinstance(result, EyeResult)
#         assert result.traces.shape == (200, 64)
#         assert result.height == pytest.approx(1.0)
#         assert result.width == pytest.approx(0.005 - result.time_resolution_s)
#         assert result.sampling_instant_s == pytest.approx(2.5e-03, rel=1e-9)
#
#     def test_clean_clock_fold_is_invariant(self):
#         """The same clean clock clipped half a bit later.
#
#         This fixture used to be
#         ``test_clean_clock_folded_half_a_period_off_is_unsupported``, and
#         it asserted the exact mirror of :meth:`test_clean_clock` on the
#         reasoning that the fold phase is part of the measurement. That
#         reasoning is what TASK-226.8 removes: the crossings define the
#         eye, so moving the window moves nothing but the window. The
#         rewrite therefore inverts the assertion rather than relaxing it.
#
#         ``sampling_instant_s`` is a phase on ``period_t``, and
#         ``period_t`` is a coordinate on the window that the clip DOES
#         move, so the phase rotates by exactly the clip while the
#         absolute instant it names stays where it was.
#         """
#         t = np.linspace(0, 1.0, 100001)
#         y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
#         period = 0.005
#         base = eye_diagram(_trans(t, y), period=period,
#                            samples_per_period=64)
#         shifted = eye_diagram(_clip(_trans(t, y), 0.0025), period=period,
#                               samples_per_period=64)
#
#         assert shifted.height == pytest.approx(base.height, rel=1e-9)
#         assert shifted.width == pytest.approx(base.width, rel=1e-9)
#         # Same physical instant: the phase rotated by the clip, and
#         # origin + phase is the same point of the waveform modulo the UI.
#         assert (0.0025 + shifted.sampling_instant_s) % period == \
#             pytest.approx(base.sampling_instant_s % period, abs=1e-12)
#         # The pre-TASK-226.8 expectation, stated so this can never be
#         # mistaken for a tolerance that drifted: the two folds used to
#         # disagree about width by a factor of 63.
#         assert shifted.width != pytest.approx(shifted.time_resolution_s,
#                                               rel=1e-3)
#
#     def test_folding_at_two_unit_intervals_takes_the_widest_opening(self):
#         """``period`` is the UNIT INTERVAL, and this is what happens when
#         it is not.
#
#         The same clock folded on 0.01 s, two bits. Its crossings then
#         form clusters at phase ~0 and ~T/2, the window holds two equal
#         openings, and ``np.argmax`` resolves the exact tie
#         deterministically to the one with the earlier left edge. The
#         sampling instant lands a quarter of the window in — the middle of
#         the FIRST opening — where every window is on the same level, so
#         no threshold sits midway between a one and a zero distribution
#         and ``height`` is ``nan``. That is the honest report for a fold
#         the metric cannot read as an eye, and it is documented rather
#         than repaired.
#         """
#         t = np.linspace(0, 1.0, 100001)
#         y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
#         r = eye_diagram(_trans(t, y), period=0.01, samples_per_period=64)
#         assert r.width == pytest.approx(0.005 - r.time_resolution_s)
#         assert r.sampling_instant_s == pytest.approx(2.5e-03, rel=1e-9)
#         assert math.isnan(r.height)
#
#     def test_invalid_period(self):
#         t = np.linspace(0, 1, 100)
#         with pytest.raises(ValueError, match='period must be > 0'):
#             eye_diagram(_trans(t, t.copy()), period=0)
#
#     def test_too_few_periods(self):
#         t = np.linspace(0, 0.005, 100)
#         with pytest.raises(ValueError, match='fewer than 2 periods'):
#             eye_diagram(_trans(t, t.copy()), period=0.01)
#
#     def test_n_traces_too_high(self):
#         t = np.linspace(0, 1.0, 1001)
#         y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
#         with pytest.raises(ValueError, match='only .* fit'):
#             eye_diagram(_trans(t, y), period=0.01, n_traces=200)
#
#     def test_invalid_samples_per_period(self):
#         t = np.linspace(0, 1.0, 1001)
#         y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
#         with pytest.raises(ValueError, match='samples_per_period must be >= 4'):
#             eye_diagram(_trans(t, y), period=0.01, samples_per_period=2)
#
#
# class TestEyeDiagramFoldInvariance:
#     """TASK-226.8 AC #5 and AC #6 — THE HEADLINE.
#
#     Everything derives from the crossings, so the result cannot depend on
#     where the caller clipped. These tests clip ONE waveform at ten
#     origins spread over a full unit interval — including one on the
#     transitions and one half a period off them — and require the same
#     answer every time.
#     """
#
#     UI = 200 * _PS
#     SPP = 128
#
#     def _origins(self, sig, reference):
#         """Ten clip origins across one UI: eight on an even grid, plus
#         the two that matter physically — the centre of the transition
#         cluster, and the sampling instant half a period from it."""
#         t0 = float(sig.x[0])
#         cluster = (reference.sampling_instant_s + 0.5 * self.UI) % self.UI
#         return ([t0 + j * self.UI / 8 for j in range(8)]
#                 + [t0 + cluster, t0 + reference.sampling_instant_s])
#
#     def test_height_width_and_instant_are_fold_invariant(self):
#         """AC #5 — one waveform, ten fold origins, one answer.
#
#         The waveform is PRBS7 through a 0.35 UI pole. ``height`` and
#         ``width`` must be equal outright; the sampling instant is a phase
#         on a window the clip moves, so what must be equal is the
#         ABSOLUTE instant it names, modulo the UI.
#
#         Tolerances are tied to the resampling grid, which is the only
#         thing a clip still moves: a clip changes which points of the
#         source the fold samples, so the interpolated crossings and the
#         interpolated column both shift by an interpolation error rather
#         than not at all.
#         """
#         sig = _rc_prbs_sig()
#         ref = eye_diagram(sig, period=self.UI, samples_per_period=self.SPP)
#         heights, widths, instants = [], [], []
#         for t_from in self._origins(sig, ref):
#             clipped = _clip(sig, t_from)
#             r = eye_diagram(clipped, period=self.UI,
#                             samples_per_period=self.SPP)
#             heights.append(r.height)
#             widths.append(r.width)
#             instants.append((float(clipped.x[0]) + r.sampling_instant_s)
#                             % self.UI)
#             # The fold really did move: this is not ten copies of one call.
#             assert r.traces.shape[0] in (126, 127)
#
#         assert len(heights) == 10
#         # Invariance first: the ten agree with each other to float noise,
#         # thousands of times finer than the 1.5625 ps resampling grid.
#         # MEASURED on task/task-226.8: 9.1e-13 mV, 6.5e-13 ps, 1.1e-15 UI.
#         assert max(heights) - min(heights) < 1e-9          # volts
#         assert max(widths) - min(widths) < 0.01 * _PS
#         assert max(instants) - min(instants) < 0.01 * _PS
#
#         # Then correctness: all ten agree with the CLOSED FORM, which is
#         # computed from the same bits without resampling anything and
#         # without calling the metric. The residual is the metric's own
#         # 128-point fold reading an exponential on straight lines --
#         # MEASURED at 0.0033 ps of width, 0.0025 ps of instant and
#         # 0.0032 mV of height, i.e. 0.2 % of one resampling cell, and
#         # unchanged when the SOURCE rate is varied 128..1024 per UI,
#         # which is what identifies the fold as its origin.
#         ref = _rc_prbs_reference()
#         cell = self.UI / self.SPP
#         assert ref['n_windows'] == 64
#         for h, w, inst in zip(heights, widths, instants):
#             assert abs(w - ref['width']) < 0.02 * cell
#             assert abs(inst - ref['instant']) < 0.02 * cell
#             assert abs(h - ref['height']) < 2e-5           # volts
#
#     def test_the_fold_dependence_measured_on_main_is_gone(self):
#         """AC #6 — the same ten origins, read the old way.
#
#         The superseded implementation sampled ``traces[:, spp // 2]``,
#         the middle of the WINDOW, and took its threshold from that same
#         column. Reproduced here in six lines so the regression carries
#         its own before-picture, it reports between 20.9 mV and 885.1 mV
#         on this waveform depending only on where the caller clipped --
#         a factor of 42 -- while the metric returns one number.
#
#         PROVENANCE. The 527.50 / 881.59 / 768.69 mV figures quoted on
#         TASK-226.8 came from an ad hoc probe the task's author ran while
#         assessing the metric, not from a fixture: 400 random bits from
#         ``default_rng(3)``, a 1 ns UI, and a discrete one-pole IIR
#         stepped at 200 samples per UI, with the 881.59 mV taken from a
#         per-column scan rather than from ``eye_diagram`` at all. That
#         probe is not reproducible from anything in the repo and its
#         numbers are deliberately NOT pinned here; the author's own
#         conclusion from it, that height should become the widest
#         opening, was refuted under TASK-226.7 and is what
#         :meth:`test_sampling_instant_is_the_cluster_midpoint_not_the_widest_column`
#         now guards against. The defect it illustrated is real and is
#         re-measured above on a fixture that IS reproducible.
#         """
#         sig = _rc_prbs_sig()
#         ref = eye_diagram(sig, period=self.UI, samples_per_period=self.SPP)
#         old, new = [], []
#         for t_from in self._origins(sig, ref):
#             clipped = _clip(sig, t_from)
#             r = eye_diagram(clipped, period=self.UI,
#                             samples_per_period=self.SPP)
#             new.append(r.height)
#             # The superseded measurement: fixed column, threshold from it.
#             centre = r.traces[:, self.SPP // 2]
#             mid = 0.5 * (float(centre.min()) + float(centre.max()))
#             hi, lo = centre[centre > mid], centre[centre <= mid]
#             old.append(float(hi.min() - lo.max()) if hi.size and lo.size
#                        else 0.0)
#
#         assert max(old) / min(old) > 40.0
#         assert min(old) < 0.025 and max(old) > 0.88
#         assert max(new) - min(new) < 1e-9
#
#     def test_sampling_instant_is_the_cluster_midpoint_not_the_widest_column(self):
#         """AC #3 — derived from the crossings, and deliberately NOT the
#         maximum of the opening.
#
#         On a causal lag-dominated channel the opening rises monotonically
#         through the bit right up to the next transition, so the widest
#         column is the one with no timing margin at all (measured under
#         TASK-226.7). Here the widest column is the fold boundary itself
#         at 885.1 mV, which is where the next transition begins; the
#         metric samples at 0.7323 UI for 753.2 mV, the instant with equal
#         timing margin on both sides.
#         """
#         sig = _rc_prbs_sig()
#         r = eye_diagram(sig, period=self.UI, samples_per_period=self.SPP)
#         rows = _fold_rows(sig, r, self.UI)
#         search_t = np.append(r.period_t, self.UI)
#         profile = _opening_profile(rows, search_t)
#         widest = int(np.nanargmax(profile))
#
#         ref = _rc_prbs_reference()
#         assert r.height == pytest.approx(ref['height'], abs=2e-5)
#         assert r.height < profile[widest]
#         # The widest column is 17 % taller and is computed here from the
#         # returned fold, independently of the metric.
#         assert profile[widest] / r.height == pytest.approx(1.175, rel=1e-3)
#
#         # The widest column is phase 0, the bit boundary -- where the
#         # next transition BEGINS. The eye runs from 48.52 ps to 244.39
#         # ps, so timing margin, the distance from a column to the nearer
#         # end of that gap, is 97.93 ps at the derived instant and only
#         # 44.39 ps at the widest column: the widest opening buys 17 % more
#         # volts for 55 % less time.
#         assert search_t[widest] == pytest.approx(0.0, abs=1e-18)
#         left = (r.sampling_instant_s - 0.5 * r.width) % self.UI
#         right = (r.sampling_instant_s + 0.5 * r.width) % self.UI
#
#         def margin(phase):
#             to_left = (phase - left) % self.UI
#             to_right = (right - phase) % self.UI
#             return min(to_left, to_right)
#
#         assert margin(r.sampling_instant_s) == pytest.approx(0.5 * r.width,
#                                                              rel=1e-9)
#         assert margin(search_t[widest]) == pytest.approx(44.39 * _PS,
#                                                          rel=1e-3)
#         assert margin(search_t[widest]) < 0.5 * margin(r.sampling_instant_s)
#         # And the instant really is half a period from the cluster centre.
#         cluster = (r.sampling_instant_s + 0.5 * self.UI) % self.UI
#         assert abs(r.sampling_instant_s - cluster) == \
#             pytest.approx(0.5 * self.UI, abs=1e-15)
#
#     def test_threshold_is_solved_at_the_instant_not_read_from_a_column(self):
#         """AC #4 — the threshold and the instant are solved together.
#
#         An ASYMMETRIC channel (0.55 UI rising, 0.18 UI falling) is what
#         makes this observable: the one and zero distributions stop being
#         mirror images, so the decision threshold leaves the mid-rail, and
#         because the crossing search runs at that threshold the eye's
#         clusters move with it.
#
#         MEASURED on task/task-226.8:
#
#         =================================  =========  ==========  =========
#         threshold                          width      instant     height
#         =================================  =========  ==========  =========
#         the fold's mid-rail, 500.0 mV      142.33 ps  147.41 ps   721.5 mV
#         solved at the instant, 370.3 mV    178.50 ps  140.14 ps   699.9 mV
#         =================================  =========  ==========  =========
#
#         A 25 % difference in width, so the joint solve is not cosmetic.
#         The fixed centre column's own mid-rail is a third level again,
#         and its opening a fourth number: 535.0 mV.
#         """
#         sig = _rc_prbs_sig(tau_r_ui=0.55, tau_f_ui=0.18)
#         r = eye_diagram(sig, period=self.UI, samples_per_period=self.SPP)
#         rows = _fold_rows(sig, r, self.UI)
#         search_t = np.append(r.period_t, self.UI)
#         cell = self.UI / self.SPP
#
#         ref = _rc_prbs_reference(tau_r_ui=0.55, tau_f_ui=0.18)
#         midrail = 0.5 * (float(rows.min()) + float(rows.max()))
#         solved = _recovered_threshold(rows, search_t, r)
#         assert midrail == pytest.approx(0.5, abs=1e-3)
#         assert solved == pytest.approx(ref['threshold'], abs=1e-4)
#         assert abs(solved - midrail) > 0.1
#
#         # height is symmetric about the SOLVED threshold -- that is what
#         # "midway between the one and zero distributions" means.
#         col = _column_at(rows, search_t, r.sampling_instant_s)
#         ones, zeros = col[col > solved], col[col <= solved]
#         assert float(ones.min()) - solved == \
#             pytest.approx(solved - float(zeros.max()), rel=1e-9)
#         assert r.height == pytest.approx(float(ones.min() - zeros.max()),
#                                          rel=1e-9)
#         assert r.height == pytest.approx(ref['height'], abs=2e-5)
#         assert r.width == pytest.approx(ref['width'], abs=0.02 * cell)
#         assert r.sampling_instant_s == pytest.approx(ref['instant'],
#                                                      abs=0.02 * cell)
#
#         # One pass at the mid-rail -- what "not solved together" would
#         # have produced -- gives a different eye. Solved in closed form
#         # too, by pinning the level instead of iterating for it.
#         one_pass = _rc_prbs_reference(tau_r_ui=0.55, tau_f_ui=0.18,
#                                       threshold=0.5)
#         assert one_pass['width'] == pytest.approx(142.33 * _PS, rel=1e-3)
#         assert abs(r.width - one_pass['width']) > 30 * _PS
#
#     def test_asymmetric_channel_is_fold_invariant_too(self):
#         """AC #5 — invariance is not a property of the symmetric fixture.
#
#         The same ten origins on the asymmetric channel, whose threshold
#         is 130 mV off the mid-rail and therefore moves the crossings.
#         """
#         sig = _rc_prbs_sig(tau_r_ui=0.55, tau_f_ui=0.18)
#         ref = eye_diagram(sig, period=self.UI, samples_per_period=self.SPP)
#         heights, widths, instants = [], [], []
#         for t_from in self._origins(sig, ref):
#             clipped = _clip(sig, t_from)
#             r = eye_diagram(clipped, period=self.UI,
#                             samples_per_period=self.SPP)
#             heights.append(r.height)
#             widths.append(r.width)
#             instants.append((float(clipped.x[0]) + r.sampling_instant_s)
#                             % self.UI)
#         assert max(heights) - min(heights) < 1e-6
#         assert max(widths) - min(widths) < 0.01 * _PS
#         assert max(instants) - min(instants) < 0.01 * _PS
#
#     def test_common_translation_is_invariant(self):
#         """Translating the whole trace translates the fold with it.
#
#         Kept from TASK-226.2, where it guarded a grid anchored to
#         ``sig.x[0]``. There is no grid now, but the property is still
#         worth pinning and it is now exact rather than approximate: a
#         37.5 ps shift is not a whole number of periods, so a hardcoded
#         zero origin would break here.
#         """
#         base = _pwl_sig([(398 * _PS, 'rise'), (602 * _PS, 'fall'),
#                          (802 * _PS, 'rise'), (1198 * _PS, 'fall')],
#                         1400 * _PS)
#         shift = 37.5 * _PS
#         moved = _pwl_sig([(398 * _PS + shift, 'rise'),
#                           (602 * _PS + shift, 'fall'),
#                           (802 * _PS + shift, 'rise'),
#                           (1198 * _PS + shift, 'fall')],
#                          1400 * _PS, t_origin=shift)
#         a = eye_diagram(base, period=self.UI, samples_per_period=self.SPP)
#         b = eye_diagram(moved, period=self.UI, samples_per_period=self.SPP)
#         assert b.height == pytest.approx(a.height, rel=1e-9)
#         assert b.width == pytest.approx(a.width, rel=1e-9)
#         assert b.jitter_rms == pytest.approx(a.jitter_rms, abs=1e-18)
#         assert b.sampling_instant_s == pytest.approx(a.sampling_instant_s,
#                                                      rel=1e-9)
#         np.testing.assert_allclose(b.traces, a.traces, atol=1e-12)
#         np.testing.assert_array_equal(b.period_t, a.period_t)
#
#
# class TestEyeDiagramTwoClusters:
#     """TASK-226.8 AC #1, #2 and #3 — the arithmetic on the clusters.
#
#     Replaces ``TestEyeDiagramNearestIdealEdge``. That class pinned
#     ``width = period − (max − min)`` of one cluster of first RISING
#     crossings, measured against the ideal grid ``sig.x[0] + n·period``.
#     Both halves are superseded: the population is now every crossing of
#     either direction, and the statistic is the gap BETWEEN the clusters
#     rather than a spread within one of them. Its fixtures are reused
#     where they still say something, with the falling edges they used to
#     ignore made explicit.
#     """
#
#     PERIOD = 200 * _PS
#     SPP = 128
#
#     def _eye(self, sig, **kw):
#         return eye_diagram(sig, period=self.PERIOD,
#                            samples_per_period=self.SPP, **kw)
#
#     def test_width_is_the_gap_between_the_two_clusters(self):
#         """AC #2 — the review's 398/802 ps example, with its falling
#         edges in the population.
#
#         Rising crossings at 398 and 802 ps, falling at 602 and 1198 ps.
#         Folded on 200 ps the phases are 198, 2, 2 and 198: a left cluster
#         at 2 ps, a right cluster at 198 ps, and the eye between them.
#
#             width = 198 − 2                    = 196 ps
#             sampling instant = 2 + 196/2       = 100 ps
#
#         196 ps is the same number TASK-226.2's ``period − (max error −
#         min error)`` produced from the two rising edges alone, which is
#         the continuity check: the two readings agree when the edges
#         jitter alike. They part company in
#         :meth:`test_rising_and_falling_edges_are_both_in_the_population`,
#         where they do not.
#         """
#         sig = _pwl_sig([(398 * _PS, 'rise'), (602 * _PS, 'fall'),
#                         (802 * _PS, 'rise'), (1198 * _PS, 'fall')],
#                        1400 * _PS)
#         r = self._eye(sig)
#         assert r.width == pytest.approx(196 * _PS, rel=1e-9)
#         assert r.sampling_instant_s == pytest.approx(100 * _PS, rel=1e-9)
#         assert r.height == pytest.approx(1.0, rel=1e-9)
#         assert r.n_crossing_windows == 4
#         # NOT period minus the spread of one cluster, which for the two
#         # rising crossings alone would be the same 196 ps by coincidence
#         # of this fixture -- and NOT the raw folded-phase spread, 4 ps,
#         # which the pre-TASK-226.2 code produced.
#         assert r.width != pytest.approx(4 * _PS, rel=1e-6)
#
#     def test_rising_and_falling_edges_are_both_in_the_population(self):
#         """AC #1 — the crossing search is not restricted to one
#         direction, and it changes the answer.
#
#         Rising crossings at phase 2 ps, FALLING crossings at phase 12 ps.
#         The two directions do not coincide, so the transition arc runs
#         from 2 to 12 ps and the eye is the other 190 ps:
#
#             width = 2 + 200 − 12               = 190 ps
#             sampling instant = 12 + 95         = 107 ps
#
#         A rising-only search sees one phase, calls the whole period open
#         and samples at 102 ps. Both numbers are asserted against.
#         """
#         sig = _pwl_sig([(402 * _PS, 'rise'), (612 * _PS, 'fall'),
#                         (802 * _PS, 'rise'), (1012 * _PS, 'fall')],
#                        1200 * _PS)
#         r = self._eye(sig)
#         assert r.width == pytest.approx(190 * _PS, rel=1e-9)
#         assert r.sampling_instant_s == pytest.approx(107 * _PS, rel=1e-9)
#         # What rising-only would have said.
#         assert r.width != pytest.approx(self.PERIOD, rel=1e-6)
#         assert r.sampling_instant_s != pytest.approx(102 * _PS, rel=1e-6)
#
#     def test_jitter_is_the_spread_about_the_derived_cluster_centre(self):
#         """AC #1 — ``jitter_rms`` comes from the same population.
#
#         The reference is derived, not assumed: the transition arc is the
#         complement of the eye, so its centre is the sampling instant plus
#         half a period, and no crossing sits anywhere near the wrap. On
#         the 398/602/802/1198 fixture the four phases are 198, 2, 2, 198
#         and the centre is phase 0, giving errors −2, +2, +2, −2 and
#         ``std(ddof=1) = sqrt(16/3) = 2.309401 ps``.
#         """
#         sig = _pwl_sig([(398 * _PS, 'rise'), (602 * _PS, 'fall'),
#                         (802 * _PS, 'rise'), (1198 * _PS, 'fall')],
#                        1400 * _PS)
#         r = self._eye(sig)
#         assert r.jitter_rms == pytest.approx(math.sqrt(16 / 3) * _PS,
#                                              rel=1e-9)
#
#     def test_missing_transition_costs_nothing(self):
#         """AC #2 — a skipped bit does not accumulate a period of error.
#
#         Kept from TASK-226.2 in substance: the two crossings are 804 ps
#         apart rather than 204, so four ideal edges carry no transition,
#         as PRBS data routinely does. Reducing modulo the period is
#         per-crossing and independent, so the answer is the same
#         196 ps / 100 ps eye. Anything indexed off the previous OBSERVED
#         crossing would call the second edge 804 ps late.
#         """
#         sig = _pwl_sig([(398 * _PS, 'rise'), (1202 * _PS, 'fall')],
#                        1600 * _PS)
#         r = self._eye(sig)
#         assert r.width == pytest.approx(196 * _PS, rel=1e-9)
#         assert r.sampling_instant_s == pytest.approx(100 * _PS, rel=1e-9)
#         assert r.jitter_rms == pytest.approx(2 * math.sqrt(2) * _PS,
#                                              rel=1e-9)
#
#     def test_equal_gaps_resolve_to_the_earliest_left_edge(self):
#         """AC #2 — an exact tie between equal gaps is deterministic.
#
#         A square wave whose transitions sit at ``0.125 + k/4`` folded on
#         a period of 1.0: the crossing phases are the exactly
#         representable 0.125, 0.375, 0.625 and 0.875, so all four gaps are
#         the identical float 0.25 and the tie is genuine rather than a
#         decimal coincidence. ``np.argmax`` takes the FIRST maximum, so
#         the chosen gap is the one with the earliest left edge, 0.125, and
#         the instant is 0.25. An arbitrary but fixed choice for an
#         undecidable population, not recovered phase.
#
#         ``height`` is ``nan`` here: every window is HIGH at 0.25, so no
#         level sits midway between a one and a zero distribution. Four
#         crossings per window is a fold on four unit intervals, and this
#         is what the metric does with one.
#         """
#         r = eye_diagram(_square_sig(), period=1.0, samples_per_period=128)
#         assert r.width == 0.25
#         assert r.sampling_instant_s == 0.25
#         assert r.n_crossing_windows == 8
#         assert math.isnan(r.height)
#
#     def test_public_numbers_match_an_independent_reimplementation(self):
#         """AC #2 and #3 — the returned fold, width, instant and height
#         are reproduced from the public fields alone.
#
#         ``period_t``, ``traces`` and ``time_resolution_s`` are rebuilt
#         from the Signal; the crossings are re-found with the PUBLIC
#         ``crossings()`` metric rather than the private helper; the gap,
#         instant and opening are then recomputed by hand.
#         """
#         sig = _rc_prbs_sig()
#         r = self._eye(sig)
#
#         n_traces = int(float(sig.x[-1] - sig.x[0]) / self.PERIOD)
#         period_t = np.linspace(0.0, self.PERIOD, self.SPP, endpoint=False)
#         traces = np.vstack([
#             np.interp(float(sig.x[0]) + k * self.PERIOD + period_t,
#                       sig.x, sig.y)
#             for k in range(n_traces)
#         ])
#         np.testing.assert_array_equal(r.period_t, period_t)
#         np.testing.assert_array_equal(r.traces, traces)
#         assert r.time_resolution_s == self.PERIOD / self.SPP
#
#         rows = _fold_rows(sig, r, self.PERIOD)
#         search_t = np.append(period_t, self.PERIOD)
#         threshold = _recovered_threshold(rows, search_t, r)
#
#         phases = []
#         for k in range(n_traces):
#             ui_trace = Signal(f'ui{k}', 'V', 'time', 's', search_t, rows[k])
#             for c in crossings(ui_trace, threshold=threshold, edge='any'):
#                 phases.append(float(c) % self.PERIOD)
#         phases = np.sort(np.asarray(phases))
#         gaps = np.append(np.diff(phases),
#                          phases[0] + self.PERIOD - phases[-1])
#         j = int(np.argmax(gaps))
#         assert r.width == pytest.approx(float(gaps[j]), rel=1e-9)
#         instant = (phases[j] + 0.5 * gaps[j]) % self.PERIOD
#         assert r.sampling_instant_s == pytest.approx(instant, rel=1e-9)
#
#         col = _column_at(rows, search_t, instant)
#         ones, zeros = col[col > threshold], col[col <= threshold]
#         assert r.height == pytest.approx(float(ones.min() - zeros.max()),
#                                          rel=1e-9)
#
#
# class TestEyeDiagramJointSolveConvergence:
#     """TASK-226.8 AC #4 — the threshold/instant fixed point terminates,
#     and the eye it finds tracks the physics.
#
#     The solve is iterated because the threshold changes which crossings
#     exist and the crossings change where the threshold is read. It
#     normally settles in two to four passes; a badly closed eye can flip
#     its own classification under the update instead, which is what the
#     iteration cap and the cycle guard are for.
#     """
#
#     UI = 200 * _PS
#     SPP = 128
#
#     def test_the_eye_closes_as_the_channel_gets_lossier(self):
#         """Height and width both fall monotonically as the pole slows,
#         and the sampling instant walks later through the bit as the
#         channel's lag grows. MEASURED on task/task-226.8:
#
#         =========  ==========  ==========  =========
#         tau / UI   height      width       instant
#         =========  ==========  ==========  =========
#         0.10       993.26 mV   199.999 ps  113.87 ps
#         0.35       753.15 mV   195.863 ps  146.45 ps
#         0.60       517.46 mV   174.881 ps  170.62 ps
#         1.00       237.90 mV   108.552 ps  192.72 ps
#         =========  ==========  ==========  =========
#         """
#         rows = [eye_diagram(_rc_prbs_sig(tau_r_ui=tau), period=self.UI,
#                             samples_per_period=self.SPP)
#                 for tau in (0.10, 0.35, 0.60, 1.00)]
#         heights = [r.height for r in rows]
#         widths = [r.width for r in rows]
#         instants = [r.sampling_instant_s for r in rows]
#         assert heights == sorted(heights, reverse=True)
#         assert widths == sorted(widths, reverse=True)
#         assert instants == sorted(instants)
#         for r, tau in zip(rows, (0.10, 0.35, 0.60, 1.00)):
#             ref = _rc_prbs_reference(tau_r_ui=tau)
#             assert r.height == pytest.approx(ref['height'], abs=5e-5)
#             assert r.width == pytest.approx(ref['width'],
#                                             abs=0.05 * self.UI / self.SPP)
#
#     def test_a_badly_closed_eye_terminates_and_repeats(self):
#         """A 5 UI pole leaves an 8.7 mV opening, and the solve runs into
#         its iteration cap rather than settling.
#
#         What the cap guarantees is not convergence but DETERMINISM: the
#         final iterate is returned, the call finishes, and two calls on
#         the same data agree exactly. The 8.7 mV and 20.6 ps it reports
#         fail any eye spec worth writing, which is the honest outcome for
#         a link this closed.
#         """
#         sig = _rc_prbs_sig(tau_r_ui=5.0)
#         a = eye_diagram(sig, period=self.UI, samples_per_period=self.SPP)
#         b = eye_diagram(sig, period=self.UI, samples_per_period=self.SPP)
#         assert a.height == b.height
#         assert a.width == b.width
#         assert a.jitter_rms == b.jitter_rms
#         assert a.sampling_instant_s == b.sampling_instant_s
#         assert math.isfinite(a.height) and 0.0 < a.height < 0.02
#         assert 0.0 < a.width < 0.2 * self.UI
#
#
# class TestEyeDiagramWindowEndpoints:
#     """TASK-226.1 — the crossing search covers the whole unit interval.
#
#     ``period_t`` is built with ``endpoint=False``, so the last
#     ``period / samples_per_period`` of every window used to sit between
#     the final sample of one folded row and the first sample of the next
#     and was searched by neither. The fix hands the search one extra
#     sample per window -- the value at the window's closing boundary --
#     while leaving ``period_t``, ``traces`` and ``time_resolution_s`` on
#     the original grid. TASK-226.8 keeps that fix and widens what it
#     feeds: the same extended rows now carry the crossing search in BOTH
#     directions and the interpolated column behind ``height``.
#
#     With ``period = 200 ps`` and ``samples_per_period = 128`` the grid
#     step is 1.5625 ps, the last grid point of a window is at phase
#     198.4375 ps, and the formerly unchecked interval is
#     198.4375..200 ps.
#     """
#
#     PERIOD = 200 * _PS
#     SPP = 128
#
#     def _eye(self, sig, **kw):
#         return eye_diagram(sig, period=self.PERIOD,
#                            samples_per_period=self.SPP, **kw)
#
#     def test_crossing_in_final_grid_interval_of_first_window(self):
#         """AC #8 — the 199.3 ps crossing in window 0's tail is found.
#
#         Window 0 carries a 0 -> 1 edge crossing at 199.3 ps, inside the
#         formerly unchecked 198.4375..200 ps interval; window 1 carries a
#         falling edge at 240.7 ps and a rising one at 299.3 ps. Resolved
#         on the 1.5625 ps grid the three phases are 199.21875, 40.71875
#         and 99.21875 ps, and the widest gap runs from 99.21875 to
#         199.21875 ps.
#
#             width = exactly 100 ps, half the period
#
#         Lose the tail crossing and the widest gap instead runs from
#         99.21875 ps round to 40.71875 ps and width reads 141.5 ps, so
#         the assertion below is red without the endpoint sample.
#         """
#         sig = _edge_sig(lambda x:
#                         _ramp(x, 198.6 * _PS, 200.0 * _PS)
#                         - _ramp(x, 240.0 * _PS, 241.4 * _PS)
#                         + _ramp(x, 298.6 * _PS, 300.0 * _PS))
#         r = self._eye(sig)
#         assert r.traces.shape == (2, self.SPP)
#         # The tail interval this regression targets really is unsampled.
#         assert r.period_t[-1] == pytest.approx(198.4375 * _PS)
#         assert r.period_t[-1] < 199.3 * _PS < self.PERIOD
#
#         assert r.width == pytest.approx(100 * _PS, rel=1e-9)
#         assert r.sampling_instant_s == pytest.approx(149.21875 * _PS,
#                                                      rel=1e-9)
#         # What the pre-TASK-226.1 search, blind to the tail, would give.
#         assert r.width != pytest.approx(141.5 * _PS, rel=1e-3)
#
#     def test_crossing_in_final_grid_interval_of_last_window(self):
#         """AC #8 — the last complete window is covered from real data.
#
#         The trace ends at exactly 400 ps, which is also the closing
#         boundary of the final window, so its endpoint sample is the last
#         supplied sample rather than anything read past the waveform. The
#         edge crossing at 399.3 ps sits in that window's tail.
#         """
#         sig = _edge_sig(lambda x:
#                         _ramp(x, 98.6 * _PS, 100.0 * _PS)
#                         - _ramp(x, 150.0 * _PS, 151.4 * _PS)
#                         + _ramp(x, 398.6 * _PS, 400.0 * _PS))
#         assert float(sig.x[-1]) == pytest.approx(
#             float(sig.x[0]) + 2 * self.PERIOD)
#         r = self._eye(sig)
#         assert r.traces.shape == (2, self.SPP)
#         assert r.width == pytest.approx(100 * _PS, rel=1e-9)
#         assert math.isfinite(r.sampling_instant_s)
#
#     @pytest.mark.parametrize('tag,t0,t1', [
#         # Crossing at 199.3 ps: just BEFORE the internal boundary, in the
#         # formerly unchecked tail of window 0.
#         ('just_before', 198.6 * _PS, 200.0 * _PS),
#         # Crossing at exactly 200.0 ps: the boundary shared by window 0's
#         # closing endpoint and window 1's opening sample.
#         ('exactly_on', 199.0 * _PS, 201.0 * _PS),
#         # Crossing at 200.7 ps: just AFTER the boundary, inside window 1.
#         ('just_after', 200.0 * _PS, 201.4 * _PS),
#         # Crossing at 290.7 ps: an ordinary within-window crossing.
#         ('mid_window', 290.0 * _PS, 291.4 * _PS),
#     ])
#     def test_single_edge_counted_exactly_once(self, tag, t0, t1):
#         """AC #8 — one physical edge contributes one crossing.
#
#         The waveform carries exactly one edge, so the population holds
#         one phase, the gap runs from it round to itself and ``width`` is
#         the full period with ``jitter_rms`` 0.0 (the owner's R3 ruling,
#         below). Those two numbers no longer distinguish a double count,
#         because two crossings at the SAME phase also leave one distinct
#         phase -- reducing modulo the period is what absorbs it. What does
#         distinguish it is ``n_crossing_windows``: a boundary edge
#         double-counted would be seen by two windows instead of one.
#         """
#         sig = _edge_sig(lambda x: _ramp(x, t0, t1))
#         r = self._eye(sig)
#         assert r.width == pytest.approx(self.PERIOD, rel=1e-9)
#         assert r.jitter_rms == 0.0
#         assert r.n_crossing_windows == 1
#
#     def test_endpoint_does_not_move_the_returned_fold(self):
#         """AC #8 — ``period_t``, ``traces`` and ``time_resolution_s``
#         are bit-identical to the documented ``endpoint=False`` fold,
#         recomputed here independently.
#
#         ``height`` is no longer on that list. It used to be read from a
#         column of ``traces``; it is now interpolated at the derived
#         instant on the extended rows, so the closing sample can reach it.
#         The fold the caller PLOTS is unchanged, which is what this test
#         is for.
#         """
#         sig = _edge_sig(lambda x:
#                         _ramp(x, 198.6 * _PS, 200.0 * _PS)
#                         - _ramp(x, 240.0 * _PS, 241.4 * _PS)
#                         + _ramp(x, 298.6 * _PS, 300.0 * _PS))
#         r = self._eye(sig)
#
#         expected_period_t = np.linspace(0.0, self.PERIOD, self.SPP,
#                                         endpoint=False)
#         expected_traces = np.vstack([
#             np.interp(float(sig.x[0]) + k * self.PERIOD + expected_period_t,
#                       sig.x, sig.y)
#             for k in range(2)
#         ])
#         np.testing.assert_array_equal(r.period_t, expected_period_t)
#         assert r.traces.shape == (2, self.SPP)
#         np.testing.assert_array_equal(r.traces, expected_traces)
#         assert r.time_resolution_s == self.PERIOD / self.SPP
#
#     def test_public_surface_unchanged(self):
#         """The signature is untouched; ``EyeResult`` grows one additive,
#         defaulted, last-position field for the derived sampling
#         instant (TASK-226.8)."""
#         import dataclasses
#         import inspect
#
#         params = inspect.signature(eye_diagram).parameters
#         assert list(params) == ['sig', 'period', 'n_traces',
#                                 'samples_per_period']
#         assert params['sig'].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
#         for name in ('period', 'n_traces', 'samples_per_period'):
#             assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
#         assert params['n_traces'].default is None
#         assert params['samples_per_period'].default == 128
#
#         fields = dataclasses.fields(EyeResult)
#         assert [f.name for f in fields] == [
#             'height', 'width', 'jitter_rms', 'traces', 'period_t',
#             'time_resolution_s', 'n_crossing_windows', 'sampling_instant_s',
#         ]
#         assert math.isnan(fields[-1].default)
#
#
# class TestEyeDiagramCrossingCount:
#     """TASK-226.3 — ``EyeResult.n_crossing_windows``, restated by
#     TASK-226.8 for the two-cluster population.
#
#     It used to count the windows whose FIRST RISING crossing fed width
#     and jitter. The population is now every crossing of either direction
#     in every window, so the field counts the windows that contributed at
#     least one: a window with a rising and a falling crossing puts two
#     crossings in the population and counts once. What it is for is
#     unchanged -- it says what the numbers rest on, so ``width = period``
#     and ``jitter_rms = 0.0`` from one observation are distinguishable
#     from the same two numbers out of ten thousand.
#     """
#
#     PERIOD = 200 * _PS
#     SPP = 128
#
#     def test_count_matches_an_independent_hand_count(self):
#         """AC #8 — recount the contributing windows without calling the
#         metric's own search.
#
#         The PRBS fixture: 127 windows, 64 of which carry a transition
#         (PRBS7 over 127 bits has exactly 64 transitions). The hand count
#         rebuilds the endpoint-extended rows and uses the PUBLIC
#         ``crossings()`` metric at the threshold read back out of the
#         result.
#         """
#         sig = _rc_prbs_sig()
#         r = eye_diagram(sig, period=self.PERIOD, samples_per_period=self.SPP)
#         assert r.traces.shape[0] == 127
#         assert r.n_crossing_windows == 64
#
#         rows = _fold_rows(sig, r, self.PERIOD)
#         search_t = np.append(r.period_t, self.PERIOD)
#         threshold = _recovered_threshold(rows, search_t, r)
#         hand = 0
#         for k in range(rows.shape[0]):
#             ui_trace = Signal(f'ui{k}', 'V', 'time', 's', search_t, rows[k])
#             if len(crossings(ui_trace, threshold=threshold, edge='any')):
#                 hand += 1
#         assert hand == 64
#         assert r.n_crossing_windows == hand
#
#     def test_count_equals_n_traces_when_every_window_contributes(self):
#         """AC #8 — every window carries a transition, and the count says
#         so.
#
#         The same square wave folded on its own quarter-period: one
#         crossing per window, all at the exactly representable phase
#         0.125, in all 32 windows. ``width`` is the full period and
#         ``jitter_rms`` is 0.0 -- the identical pair of numbers the
#         single-crossing case in
#         :class:`TestEyeDiagramDegenerateData` returns -- and this field
#         is the only thing that separates 32 clean observations from one.
#         """
#         r = eye_diagram(_square_sig(), period=0.25, samples_per_period=64)
#         assert r.traces.shape[0] == 32
#         assert r.n_crossing_windows == 32
#         assert r.width == 0.25
#         assert r.jitter_rms == 0.0
#         assert r.height == pytest.approx(1.0, rel=1e-9)
#
#     def test_a_window_with_two_crossings_counts_once(self):
#         """AC #8 — the field counts WINDOWS, while the population counts
#         CROSSINGS, and both facts are visible at once.
#
#         Window 0 carries a rising crossing at 40.5 ps and a falling one
#         at 60.5 ps; window 1 carries a rising crossing at 240.5 ps.
#         Phases 40.5, 60.5 and 40.5: two windows, three crossings, two
#         distinct phases. The count is 2, and ``width`` is 180 ps rather
#         than the full period, which is only possible if the second
#         crossing of window 0 reached the population.
#         """
#         sig = _pwl_sig([(40.5 * _PS, 'rise'), (60.5 * _PS, 'fall'),
#                         (240.5 * _PS, 'rise')], 600 * _PS)
#         r = eye_diagram(sig, period=self.PERIOD, samples_per_period=self.SPP)
#         assert r.traces.shape[0] == 3
#         assert r.n_crossing_windows == 2
#         assert r.width == pytest.approx(180 * _PS, rel=1e-9)
#
#     def test_nonpow2_samples_per_period_explicit_n_traces(self):
#         """AC #8 — non-power-of-two ``samples_per_period`` with an
#         explicit ``n_traces``: the fold is still bit-identical to an
#         independent reimplementation and the count lands in
#         ``[0, n_traces]``."""
#         sig = _rc_prbs_sig(n_bits=100)
#         period = 200 * _PS
#         spp = 97
#         n_traces = 37
#
#         r = eye_diagram(sig, period=period, samples_per_period=spp,
#                         n_traces=n_traces)
#         assert r.traces.shape == (n_traces, spp)
#         assert r.period_t.shape == (spp,)
#         assert r.time_resolution_s == pytest.approx(period / spp)
#         assert math.isfinite(r.height)
#         assert math.isfinite(r.width)
#         assert math.isfinite(r.jitter_rms)
#         assert math.isfinite(r.sampling_instant_s)
#         assert 0 <= r.n_crossing_windows <= n_traces
#
#         expected_period_t = np.linspace(0.0, period, spp, endpoint=False)
#         expected_traces = np.vstack([
#             np.interp(float(sig.x[0]) + k * period + expected_period_t,
#                       sig.x, sig.y)
#             for k in range(n_traces)
#         ])
#         np.testing.assert_array_equal(r.period_t, expected_period_t)
#         np.testing.assert_array_equal(r.traces, expected_traces)
#
#
# class TestEyeDiagramDegenerateData:
#     """TASK-226.8 AC #7 — degenerate data returns a number, never a raise
#     (Backlog decision-1), and what it returns is checked through
#     ``SpecReport.check`` rather than asserted to be "sensible".
#
#     The specs an eye carries point one way each: height and width are
#     wanted LARGE, jitter SMALL. A degenerate return that passes one of
#     those is a wrong number that looks right, which is the failure mode
#     this class exists to prevent.
#     """
#
#     PERIOD = 200 * _PS
#     SPP = 128
#
#     def _eye(self, sig, **kw):
#         return eye_diagram(sig, period=self.PERIOD,
#                            samples_per_period=self.SPP, **kw)
#
#     @staticmethod
#     def _specs(r):
#         report = SpecReport('eye', 'link', {
#             'eye_height': r.height,
#             'eye_width': r.width,
#             'eye_jitter': r.jitter_rms,
#         })
#         report.check('height', 'eye_height', '>', 0.0)
#         report.check('width', 'eye_width', '>', 0.0)
#         report.check('jitter', 'eye_jitter', '<', 1e-9)
#         return {res.name: res.passed for res in report.results}
#
#     def test_no_crossing_anywhere_returns_nan_and_passes_nothing(self):
#         """A flat trace never crosses the threshold, so there is no
#         cluster, no gap and no instant: everything is ``nan`` and
#         ``n_crossing_windows`` is 0.
#
#         This is a CHANGE. The superseded code returned ``width = 0.0``
#         and ``jitter_rms = 0.0`` here. 0.0 width fails a width spec, but
#         0.0 jitter PASSES a jitter spec -- a perfect-jitter result out of
#         a trace in which nothing was measured at all. The owner's R3
#         ruling is about ONE observed crossing, where 0.0 is what was
#         measured; with zero crossings nothing was.
#         """
#         r = self._eye(_pwl_sig([], 1000 * _PS))
#         assert math.isnan(r.height)
#         assert math.isnan(r.width)
# #         assert math.isnan(r.sampling_instant_s)
#         assert r.n_crossing_windows == 0
#         assert r.traces.shape == (5, self.SPP)
#         assert self._specs(r) == {'height': False, 'width': False,
#                                   'jitter': False}
#
#     def test_one_cluster_reports_what_one_observation_yields(self):
#         """A single crossing: the population holds one distinct phase, so
#         the gap runs from it round to itself.
#
#             width = period,  jitter_rms = 0.0,  n_crossing_windows = 1
#
#         This is the owner's ruling on R3, preserved verbatim in effect:
#         "if someone has one clock edge the jitter is 0 by definition",
#         and by the same reasoning one crossing leaves the whole unit
#         interval open. Neither value is a lie and neither is withheld.
#
#         Both therefore PASS their specs, and that is asserted rather than
#         hidden: the remedy the owner chose is disclosure, not a NaN
#         low-count policy, and ``n_crossing_windows == 1`` is the
#         disclosure. ``height`` is a real measurement here -- the opening
#         at the derived instant -- so it passes honestly.
#         """
#         r = self._eye(_pwl_sig([(398 * _PS, 'rise')], 1000 * _PS))
#         assert r.width == pytest.approx(self.PERIOD, rel=1e-9)
#         assert r.jitter_rms == 0.0
#         assert r.n_crossing_windows == 1
#         assert r.sampling_instant_s == pytest.approx(98 * _PS, rel=1e-9)
#         assert r.height == pytest.approx(1.0, rel=1e-9)
#         assert self._specs(r) == {'height': True, 'width': True,
#                                   'jitter': True}
#
#     def test_unresolvable_threshold_returns_nan_height(self):
#         """Every window on the same side of the threshold at the derived
#         instant: no level sits midway between a one and a zero
#         distribution, so ``height`` is ``nan``.
#
#         The trace idles HIGH and dips once, at 104 to 112 ps. Two
#         crossings, both in window 0, so the gap runs from 112 ps round to
#         104 ps and the instant lands at 8 ps -- where all five windows
#         are at 1.0 V. The TIMING measurement resolved and is returned;
#         only the vertical one did not.
#         """
#         sig = _pwl_sig([(104 * _PS, 'fall'), (112 * _PS, 'rise')],
#                        1000 * _PS, base=1.0)
#         r = self._eye(sig)
#         assert math.isnan(r.height)
#         assert r.width == pytest.approx(192 * _PS, rel=1e-9)
#         assert r.sampling_instant_s == pytest.approx(8 * _PS, rel=1e-9)
#         assert r.jitter_rms == pytest.approx(4 * math.sqrt(2) * _PS,
#                                              rel=1e-9)
#         assert r.n_crossing_windows == 1
#         specs = self._specs(r)
#         assert specs['height'] is False
#
#     def test_nan_input_policy_unchanged(self):
#         """TASK-78 — a NaN sample anywhere in ``sig.y`` makes every
#         measurement ``nan`` with the count at 0, and the fold is still
#         returned so it can be plotted. Unchanged by TASK-226.8 apart from
#         the new field, which joins the others at ``nan``."""
#         sig = _edge_sig(lambda x: _ramp(x, 198.6 * _PS, 200.0 * _PS))
#         y = np.array(sig.y, dtype=float)
#         y[len(y) // 2] = float('nan')
#         r = self._eye(_trans(sig.x, y))
#         assert math.isnan(r.height)
#         assert math.isnan(r.width)
# #         assert math.isnan(r.sampling_instant_s)
#         assert r.n_crossing_windows == 0
#         assert r.traces.shape == (2, self.SPP)
#         assert r.time_resolution_s == self.PERIOD / self.SPP
#         assert self._specs(r) == {'height': False, 'width': False,
#                                   'jitter': False}
# --- END commented-out (TASK-226.11) ---


# ---------------------------------------------------------------------------
# eye_diagram — the REPLACEMENT (TASK-226.12): the fold, and only the fold.
#
# The contract these tests pin is deliberately narrow: this function folds
# and measures NOTHING. So every test below is about the partition — which
# sample lands in which window, and whether it survives the trip unchanged —
# and the one test that touches a number asserts that the number was NOT
# computed.
# ---------------------------------------------------------------------------

_UI = 1e-9


def _uniform_eye_sig(n_windows=10, per_window=100, ui=_UI, t0=0.0):
    """A clean 1010… NRZ trace on a uniform grid, ``n_windows`` UI long.

    Built window by window — ``t0 + w*ui + j*(ui/per_window)`` — so the
    sample opening each window is bit-identical to that window's origin.
    A single ``arange`` over the whole trace is NOT the same number at the
    edges (``10*(ui/10) != ui`` in binary), and this fixture is used by the
    boundary test, which is about exactly that sample.
    """
    rows = [t0 + w * ui + np.arange(per_window) * (ui / per_window)
            for w in range(n_windows)]
    rows.append(np.array([t0 + n_windows * ui]))
    t = np.concatenate(rows)
    y = np.where(((t - t0) // ui).astype(int) % 2 == 0, 1.0, -1.0)
    return _trans(t, y, name='vrx')


#: A deliberately NON-UNIFORM phase pattern, of the shape a SPICE transient
#: produces: dense through the transition, sparse across the flat. Written
#: out rather than generated so a failure is reproducible by eye.
_SPICE_PHASES = np.array([
    0.0, 0.004, 0.009, 0.017, 0.031, 0.06, 0.11, 0.2, 0.35,
    0.47, 0.483, 0.491, 0.497, 0.503, 0.52, 0.58, 0.69, 0.8, 0.91, 0.97,
])


def _nonuniform_eye_sig(n_windows=12, ui=_UI, t0=0.0):
    """Non-uniform times, and a DIFFERENT set of them in every window.

    Each window slides its dense cluster by a window-dependent amount, so
    no two windows share sample positions. Any resampling onto a common
    grid would therefore be visible as two windows agreeing on ``x``.
    """
    out = []
    for k in range(n_windows):
        shift = 0.013 * ((k % 5) - 2)
        p = np.clip(_SPICE_PHASES + shift * (_SPICE_PHASES > 0.3), 0.0, 0.999)
        out.append(t0 + k * ui + ui * p)
    t = np.unique(np.concatenate(out))
    y = np.tanh(5.0 * np.sin(2 * np.pi * (t - t0) / ui))
    return _trans(t, y, name='vrx')


def _reconstruct(result, sig, period):
    """Undo the fold: window k's phases plus its own start time.

    This is the AC#3 instrument. It uses only what the result carries (the
    windows, in order) plus the two inputs, and it adds the window origin
    back with ``+``, so an exactly-equal comparison against the input is a
    statement about the samples, not about a tolerance.
    """
    t0 = float(sig.x[0])
    xs, ys = [], []
    for k, w in enumerate(result.traces):
        xs.append(np.asarray(w.x) + (t0 + k * period))
        ys.append(np.asarray(w.y))
    return np.concatenate(xs), np.concatenate(ys)


class TestEyeDiagramFold:
    """TASK-226.12 — ``eye_diagram`` folds with NO resampling."""

    def test_signature_keeps_lossless_fold_options_and_optional_threshold(self):
        """``samples_per_period`` described a resampling grid; there is no
        grid now, so the argument is gone rather than ignored."""
        import inspect
        params = inspect.signature(eye_diagram).parameters
        assert list(params) == ['sig', 'period', 'n_traces', 'auto_center', 'threshold']
        assert params['threshold'].default is None
        assert params['threshold'].kind is inspect.Parameter.KEYWORD_ONLY
        assert params['auto_center'].default is False
        assert params['auto_center'].kind is inspect.Parameter.KEYWORD_ONLY
        assert params['period'].kind is inspect.Parameter.KEYWORD_ONLY
        assert params['period'].default is inspect.Parameter.empty
        assert params['n_traces'].kind is inspect.Parameter.KEYWORD_ONLY
        assert params['n_traces'].default is None

    def test_it_is_importable_from_silk_metrics(self):
        from silk.metrics import eye_diagram as top_level
        assert top_level is eye_diagram

    def test_only_traces_is_filled(self):
        """AC#1 — nothing is measured in this step, and the result says so."""
        r = eye_diagram(_uniform_eye_sig(), period=_UI)
        assert math.isnan(r.height)
        assert math.isnan(r.width)
        assert math.isnan(r.sampling_instant_s)

    def test_traces_is_a_list_of_signals_one_per_window(self):
        """AC#2 — a list of Signals, not a rectangular array."""
        sig = _uniform_eye_sig(n_windows=10)
        r = eye_diagram(sig, period=_UI)
        assert isinstance(r.traces, list)
        assert len(r.traces) == 10
        assert all(isinstance(w, Signal) for w in r.traces)
        for w in r.traces:
            assert w.xlabel == 'time' and w.ylabel == sig.ylabel
            assert w.type == sig.type
            assert len(w.x) == len(w.y)
            assert float(np.min(w.x)) >= 0.0
            assert float(np.max(w.x)) < _UI

    def test_there_is_no_shared_grid(self):
        """AC#2 — every window carries its OWN sample positions. On
        non-uniform input no two windows agree, which is precisely what a
        shared ``period_t`` would have destroyed."""
        r = eye_diagram(_nonuniform_eye_sig(), period=_UI)
        first = np.asarray(r.traces[0].x)
        assert any(len(np.asarray(w.x)) != len(first)
                   or not np.array_equal(np.asarray(w.x), first)
                   for w in r.traces[1:])
        # and the spacing inside a single window is itself irregular
        d = np.diff(np.asarray(r.traces[0].x))
        assert float(np.max(d)) > 3.0 * float(np.min(d))

    def test_fold_is_lossless_on_a_uniform_grid(self):
        """AC#3 — reconstruct the input from the windows, compare EXACTLY."""
        sig = _uniform_eye_sig(n_windows=10, per_window=100)
        r = eye_diagram(sig, period=_UI)
        x, y = _reconstruct(r, sig, _UI)
        covered = len(x)
        assert covered == 1000                      # the whole covered span
        assert np.array_equal(x, np.asarray(sig.x)[:covered])
        assert np.array_equal(y, np.asarray(sig.y)[:covered])

    def test_fold_is_lossless_on_non_uniform_input(self):
        """AC#4 — the SPICE case, verified sample by sample."""
        sig = _nonuniform_eye_sig(n_windows=12)
        r = eye_diagram(sig, period=_UI)
        x, y = _reconstruct(r, sig, _UI)
        covered = len(x)
        t_in = np.asarray(sig.x)
        y_in = np.asarray(sig.y)
        # every whole window is covered; only the trailing partial one is not
        assert covered == int(np.count_nonzero(t_in < 11 * _UI))
        for i in range(covered):
            assert x[i] == t_in[i], f'sample {i}: time changed'
            assert y[i] == y_in[i], f'sample {i}: value changed'

    def test_fold_is_lossless_when_the_trace_starts_late(self):
        """The fold origin is ``sig.x[0]``; a non-zero, large start time is
        where a phase-computed-by-modulo would drift off the sample."""
        sig = _nonuniform_eye_sig(n_windows=12, t0=1.0e-3)
        r = eye_diagram(sig, period=_UI)
        x, y = _reconstruct(r, sig, _UI)
        covered = len(x)
        assert np.array_equal(x, np.asarray(sig.x)[:covered])
        assert np.array_equal(y, np.asarray(sig.y)[:covered])

    def test_every_covered_sample_lands_in_exactly_one_window(self):
        """AC#3, the other half: no sample is dropped inside the span and
        none is duplicated across windows."""
        sig = _nonuniform_eye_sig(n_windows=12)
        r = eye_diagram(sig, period=_UI)
        t_in = np.asarray(sig.x)
        seen = []
        for k, w in enumerate(r.traces):
            for xv in np.asarray(w.x):
                hits = np.flatnonzero(t_in == xv + k * _UI)
                assert len(hits) == 1
                seen.append(int(hits[0]))
        assert sorted(seen) == list(range(len(seen)))      # a prefix, once each
        assert len(seen) == len(set(seen))

    def test_a_boundary_sample_belongs_to_the_later_window_only(self):
        """Nothing is duplicated, so the sample sitting exactly on a window
        edge is the phase-0 sample of the NEXT window, not a closing sample
        of the previous one."""
        sig = _uniform_eye_sig(n_windows=4, per_window=10)
        r = eye_diagram(sig, period=_UI)
        for w in r.traces:
            assert float(w.x[0]) == 0.0                    # opens on the edge
            assert float(w.x[-1]) < _UI                    # never closes on it
        assert all(len(w.x) == 10 for w in r.traces)

    def test_a_sample_a_hair_below_the_edge_keeps_a_phase_in_range(self):
        """The window assignment is settled against the window ORIGIN, not
        against the division that finds it.

        On ``np.arange(n) * (ui/10)`` the sample numbered 30 lands a
        fraction of an ulp BELOW ``3·ui``, yet ``(t − t0)/period`` for it
        rounds to exactly 3.0. Taking that floor at face value files it
        under window 3 with a phase of −4e-25 — a point drawn a hair left
        of the frame. It belongs to window 2, at a phase just under the
        period, and the fold is lossless either way."""
        t = np.arange(101) * (_UI / 10)
        sig = _trans(t, np.sign(np.sin(2 * np.pi * t / _UI)))
        r = eye_diagram(sig, period=_UI)
        for w in r.traces:
            assert float(np.min(w.x)) >= 0.0
            assert float(np.max(w.x)) < _UI
        # window 2 keeps the edge sample; window 3 is one short of a
        # nominal ten. Honest, and it is what the trace actually says.
        assert [len(w.x) for w in r.traces] == [10, 10, 11, 9, 10,
                                                11, 9, 10, 11, 9]
        assert float(r.traces[2].x[-1]) + 2 * _UI == t[30]
        x, y = _reconstruct(r, sig, _UI)
        assert np.array_equal(x, t[:len(x)])

    def test_values_are_bit_identical_including_a_nan_sample(self):
        """No interpolation touches the samples, so a NaN travels into its
        window as a NaN and nothing else is disturbed (decision-1: there is
        no statistic here for it to poison)."""
        sig = _uniform_eye_sig(n_windows=4, per_window=10)
        y = np.array(sig.y, copy=True)
        y[25] = np.nan
        sig = _trans(sig.x, y)
        r = eye_diagram(sig, period=_UI)
        x, got = _reconstruct(r, sig, _UI)
        assert np.array_equal(got, y[:len(got)], equal_nan=True)
        assert np.flatnonzero(np.isnan(got)).tolist() == [25]
        assert bool(np.isnan(r.traces[2].y[5]))

    def test_n_traces_returns_the_first_n_windows(self):
        sig = _uniform_eye_sig(n_windows=10, per_window=100)
        r = eye_diagram(sig, period=_UI, n_traces=3)
        assert len(r.traces) == 3
        x, y = _reconstruct(r, sig, _UI)
        assert np.array_equal(x, np.asarray(sig.x)[:300])
        assert np.array_equal(y, np.asarray(sig.y)[:300])

    def test_a_window_with_no_samples_is_still_a_window(self):
        """A gap longer than the period leaves an empty window. It stays in
        the list so ``traces[k]`` is always window k."""
        t = np.array([0.0, 0.2e-9, 0.6e-9, 2.1e-9, 2.6e-9, 3.4e-9])
        sig = _trans(t, np.arange(len(t), dtype=float))
        r = eye_diagram(sig, period=_UI, n_traces=3)
        assert [len(w.x) for w in r.traces] == [3, 0, 2]
        assert isinstance(r.traces[1], Signal)

    def test_period_must_be_positive(self):
        sig = _uniform_eye_sig()
        with pytest.raises(ValueError, match='period must be > 0'):
            eye_diagram(sig, period=0)
        with pytest.raises(ValueError, match='period must be > 0'):
            eye_diagram(sig, period=-_UI)
        with pytest.raises(ValueError, match='period must be > 0'):
            eye_diagram(sig, period=float('nan'))

    def test_trace_shorter_than_two_periods_raises(self):
        sig = _uniform_eye_sig(n_windows=10)
        with pytest.raises(ValueError, match='fewer than 2 periods'):
            eye_diagram(sig, period=6e-9)

    def test_n_traces_out_of_range_raises(self):
        sig = _uniform_eye_sig(n_windows=10)
        with pytest.raises(ValueError, match='only 10 fit'):
            eye_diagram(sig, period=_UI, n_traces=11)
        with pytest.raises(ValueError, match='n_traces must be >= 1'):
            eye_diagram(sig, period=_UI, n_traces=0)

    def test_empty_and_wrong_axis_raise(self):
        with pytest.raises(ValueError, match='eye_diagram'):
            eye_diagram(_trans([], []), period=_UI)
        ac = Signal('s', 'voltage', 'frequency', 'V',
                    np.linspace(1, 10, 100), np.ones(100))
        with pytest.raises(ValueError, match="xlabel='time'"):
            eye_diagram(ac, period=_UI)

    def test_the_eye_plots_from_the_result_with_no_helper(self):
        """AC#5 — the docstring example, executed: every window drawn onto
        one axis by ``Signal.plot``, and the drawn data is the window's own
        un-resampled samples."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        sig = _nonuniform_eye_sig(n_windows=12)
        result = eye_diagram(sig, period=_UI)
        fig, ax = plt.subplots()
        try:
            for w in result.traces:
                w.plot(ax=ax, color='steelblue', lw=0.5, alpha=0.3)
            assert len(ax.lines) == len(result.traces)
            for line, w in zip(ax.lines, result.traces):
                assert np.array_equal(line.get_xdata(), np.asarray(w.x))
                assert np.array_equal(line.get_ydata(), np.asarray(w.y))
        finally:
            plt.close(fig)

class TestTier2TransientImports:
    def test_top_level(self):
        # eye_diagram omitted: withdrawn pending replacement (TASK-226.11).
        from silk.metrics import (
            cycle_frequency, on_time, off_time, EyeResult,
        )
        assert callable(cycle_frequency)
        assert EyeResult is not None


# ---------------------------------------------------------------------------
# Tier 3 — vendor-parity transient additions
# ---------------------------------------------------------------------------

class TestErrorNorm:
    def test_zero_for_identical(self):
        t = np.linspace(0, 1, 1001)
        s = _trans(t, np.sin(2 * np.pi * t))
        g = _trans(t, np.sin(2 * np.pi * t))
        assert error_norm(s, g) < 1e-12

    def test_max(self):
        t = np.linspace(0, 1, 1001)
        s = _trans(t, t.copy())
        g = _trans(t, t.copy() + 0.1)
        # err = -0.1 everywhere; max abs = 0.1
        assert math.isclose(error_norm(s, g, kind='max'), 0.1, rel_tol=1e-9)

    def test_rms(self):
        t = np.linspace(0, 1, 1001)
        s = _trans(t, np.zeros_like(t))
        g = _trans(t, np.full_like(t, 2.0))
        # err = -2 everywhere → RMS = 2
        assert math.isclose(error_norm(s, g, kind='rms'), 2.0, rel_tol=1e-9)

    def test_resamples_goal(self):
        # Sig has 1001 samples, goal has 11 — should still work via interp
        t_s = np.linspace(0, 1, 1001)
        t_g = np.linspace(0, 1, 11)
        s = _trans(t_s, t_s.copy())
        g = _trans(t_g, t_g.copy())
        assert error_norm(s, g, kind='rms') < 1e-9

    def test_invalid_kind(self):
        t = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="must be 'max'"):
            error_norm(_trans(t, t), _trans(t, t), kind='quux')

    # TASK-65: emptiness is validated by silk, not left to numpy. On
    # numpy < 2.5 np.interp raised on an empty sample-point array; on
    # numpy >= 2.5 it returns an empty array and the norm became a silent
    # nan. An empty *sig* with a non-empty goal was a silent nan on EVERY
    # numpy version.
    @pytest.mark.parametrize('kind', ['max', 'rms'])
    def test_empty_sig_raises_typed(self, kind):
        t = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match=r'len\(sig\)=0, len\(goal\)=10'):
            error_norm(_trans([], []), _trans(t, t), kind=kind)

    @pytest.mark.parametrize('kind', ['max', 'rms'])
    def test_empty_goal_raises_typed(self, kind):
        t = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match=r'len\(sig\)=10, len\(goal\)=0'):
            error_norm(_trans(t, t), _trans([], []), kind=kind)

    @pytest.mark.parametrize('kind', ['max', 'rms'])
    def test_both_empty_raise_typed(self, kind):
        with pytest.raises(ValueError, match='at least one sample'):
            error_norm(_trans([], []), _trans([], []), kind=kind)


class TestConvolve:
    def test_delta_returns_other(self):
        # Convolving with a unit-area delta returns the original.
        dx = 0.001
        t = np.arange(0, 1.0 + dx/2, dx)
        # "Delta": single non-zero sample of height 1/dx (so area=1)
        delta_y = np.zeros_like(t)
        delta_y[0] = 1.0 / dx
        delta = _trans(t, delta_y)
        # Other signal: a Gaussian-shaped pulse
        center = 0.5
        sigma = 0.05
        y = np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
        sig = _trans(t, y)
        result = convolve(delta, sig)
        # Result should look like sig (shifted by delta's start at 0).
        # Check peak amplitude near where the gaussian peak landed.
        peak_idx = int(np.argmax(result.y))
        assert math.isclose(result.x[peak_idx], 0.5, abs_tol=2 * dx)

    def test_returns_signal(self):
        t = np.linspace(0, 1, 11)
        a = _trans(t, np.ones_like(t))
        b = _trans(t, np.ones_like(t))
        out = convolve(a, b)
        assert isinstance(out, Signal)
        assert len(out.x) == 21   # 11 + 11 - 1

    def test_mismatched_dx(self):
        a = _trans(np.linspace(0, 1, 11), np.ones(11))
        b = _trans(np.linspace(0, 1, 21), np.ones(21))
        with pytest.raises(ValueError, match='same dx'):
            convolve(a, b)

    def test_non_uniform_x_rejected(self):
        a_x = np.array([0.0, 0.1, 0.2, 0.5])
        a = _trans(a_x, np.ones_like(a_x))
        b = _trans(np.linspace(0, 1, 11), np.ones(11))
        with pytest.raises(ValueError, match='uniform'):
            convolve(a, b)


@pytest.mark.parametrize('transform', [convolve, cross_correlation])
class TestTransformSpacing:
    @pytest.mark.parametrize('operand', ['a', 'b'])
    @pytest.mark.parametrize('x', [
        np.r_[0.0, np.cumsum([5e-12, 3.6e-11, 5e-12, 3.6e-11, 5e-12])],
        np.array([0, 1, 2, 4, 5, 6]) * 1e-9,
    ], ids=['ps-adaptive', 'ns-missing-sample'])
    def test_nonuniform_operand_rejected(self, transform, operand, x):
        dx = x[1] - x[0]
        spacing = np.diff(x)
        # The default absolute tolerance hides variation far outside rtol.
        assert np.allclose(spacing, dx, rtol=1e-6)
        assert not np.allclose(spacing, dx, rtol=1e-6, atol=0.0)
        uneven = _trans(x.copy(), [0, 0, 0, 0, 1, 0])
        uniform = _trans(np.arange(len(x)) * dx, [0, 1, 0, 0, 0, 0])
        a, b = (uneven, uniform) if operand == 'a' else (uniform, uneven)
        before = [(s.x.copy(), s.y.copy()) for s in (a, b)]

        with pytest.raises(ValueError) as exc:
            transform(a, b)

        message = str(exc.value)
        assert message.startswith(f'{transform.__name__}:')
        assert f"input '{operand}' must have uniform x-spacing" in message
        assert 'Expected dx' in message
        assert 'observed dx ranges from' in message
        assert 'sample()' in message
        assert 'both inputs' in message
        assert 'uniform spacing' in message
        assert 'same dx' in message
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', message)
        np.testing.assert_allclose(
            np.array(numbers, dtype=float), [dx, spacing.min(), spacing.max()],
            rtol=1e-6, atol=0.0,
        )
        for sig, (old_x, old_y) in zip((a, b), before):
            np.testing.assert_array_equal(sig.x, old_x)
            np.testing.assert_array_equal(sig.y, old_y)

    @pytest.mark.parametrize('dx', [1e-12, 1e-9], ids=['ps', 'ns'])
    def test_unequal_uniform_intervals_rejected(self, transform, dx):
        a = _trans(np.arange(4) * dx, [1, 2, 3, 4])
        b = _trans(np.arange(4) * (2 * dx), [4, 3, 2, 1])
        assert np.isclose(dx, 2 * dx, rtol=1e-9)
        assert not np.isclose(dx, 2 * dx, rtol=1e-9, atol=0.0)

        with pytest.raises(ValueError) as exc:
            transform(a, b)

        message = str(exc.value)
        assert message.startswith(f'{transform.__name__}:')
        assert 'a and b must have the same dx' in message
        assert 'da=' in message and 'db=' in message
        assert 'sample()' in message
        assert 'both inputs' in message
        assert 'uniform spacing' in message
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', message)
        np.testing.assert_allclose(
            np.array(numbers, dtype=float), [dx, 2 * dx], rtol=1e-6, atol=0.0,
        )

    @pytest.mark.parametrize('dx,xlabel', [
        (1e-12, 'time'), (1e-9, 'time'), (0.1, 'time'), (1e6, 'frequency'),
    ], ids=['ps', 'ns', 'seconds', 'frequency'])
    @pytest.mark.parametrize('perturb', [False, True], ids=['uniform', 'roundoff'])
    def test_valid_grid_preserves_values_and_axes(self, transform, dx, xlabel,
                                                perturb):
        # Exercise both retained relative tolerances without changing da.
        a_x = np.array([2, 3, 4 + (5e-7 if perturb else 0)]) * dx
        b_x = np.array([4, 5 + (5e-10 if perturb else 0)]) * dx
        a = Signal('a', 'voltage', xlabel, 'V', a_x, np.array([1., 2., 3.]))
        b = Signal('b', 'voltage', xlabel, 'V', b_x, np.array([4., 5.]))
        before = [(s.x.copy(), s.y.copy()) for s in (a, b)]

        result = transform(a, b)

        x_atol = 0.0
        if transform is convolve:
            expected_y = np.array([4, 13, 22, 15]) * dx
            expected_x = np.array([6, 7, 8, 9]) * dx
            assert result.xlabel == xlabel
        else:
            expected_y = [5, 14, 23, 12]
            # a.x[0] - b.x[0] == 2*dx - 4*dx == -2*dx (unaffected by the
            # roundoff perturbation, which only touches each array's last
            # element), carried into the lag axis. The sum k*da + (a.x[0] -
            # b.x[0]) is no longer an exact multiple of dx, so allow a tiny
            # float-rounding tolerance relative to dx (not a real-world atol).
            expected_x = np.array([-1, 0, 1, 2]) * dx + (a_x[0] - b_x[0])
            x_atol = abs(dx) * 1e-9
            assert result.xlabel == 'lag'
        np.testing.assert_allclose(result.y, expected_y, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(result.x, expected_x, rtol=1e-12, atol=x_atol)
        for sig, (old_x, old_y) in zip((a, b), before):
            np.testing.assert_array_equal(sig.x, old_x)
            np.testing.assert_array_equal(sig.y, old_y)

    def test_explicit_sample_recovers_ns_peak_separation(self, transform):
        x = np.array([0, 1, 2, 4, 5, 6]) * 1e-9
        a = _trans(x.copy(), [0, 0, 0, 0, 1, 0])
        b = _trans(x.copy(), [0, 1, 0, 0, 0, 0])
        before = [(s.x.copy(), s.y.copy()) for s in (a, b)]
        with pytest.raises(ValueError, match='uniform x-spacing'):
            transform(a, b)

        grid = np.arange(7) * 1e-9
        sampled_a = sample(a, x_new=grid)
        sampled_b = sample(b, x_new=grid)
        sampled_before = [(s.x.copy(), s.y.copy())
                          for s in (sampled_a, sampled_b)]
        result = transform(sampled_a, sampled_b)

        expected_y = np.zeros(13)
        if transform is convolve:
            expected_y[6] = 1e-9
            expected_x = np.arange(13) * 1e-9
        else:
            expected_y[10] = 1.0
            expected_x = np.arange(-6, 7) * 1e-9
            assert result.x[np.argmax(result.y)] == pytest.approx(4e-9, abs=0)
        np.testing.assert_allclose(result.y, expected_y, rtol=1e-12, atol=0.0)
        np.testing.assert_allclose(result.x, expected_x, rtol=1e-12, atol=0.0)
        for sig, (old_x, old_y) in zip(
                (a, b, sampled_a, sampled_b), before + sampled_before):
            np.testing.assert_array_equal(sig.x, old_x)
            np.testing.assert_array_equal(sig.y, old_y)


class TestTier3TransientImports:
    def test_top_level(self):
        from silk.metrics import error_norm, convolve
        assert callable(error_norm)
        assert callable(convolve)


# ---------------------------------------------------------------------------
# Leftovers — waveform manipulation + TIE / long-term jitter
# ---------------------------------------------------------------------------

class TestLshift:
    def test_basic_shift(self):
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, t.copy())
        shifted = lshift(sig, 3.0)
        # x_new = x - 3 → goes from -3 to 7
        assert math.isclose(float(shifted.x[0]), -3.0, rel_tol=1e-12)
        assert math.isclose(float(shifted.x[-1]), 7.0, rel_tol=1e-12)
        # y unchanged
        assert np.allclose(shifted.y, sig.y)

    def test_negative_shift(self):
        t = np.linspace(0, 1, 11)
        shifted = lshift(_trans(t, t.copy()), -0.5)
        assert math.isclose(float(shifted.x[0]), 0.5, rel_tol=1e-12)


class TestRshift:
    def test_equals_negative_lshift(self):
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, np.sin(t))
        r, l = rshift(sig, 3.0), lshift(sig, -3.0)
        assert np.array_equal(r.x, l.x) and np.array_equal(r.y, l.y)
        assert math.isclose(float(r.x[0]), 3.0, rel_tol=1e-12)
        assert 'rshift' in r.name

    def test_empty_raises_naming_rshift(self):
        empty = _trans(np.array([]), np.array([]))
        with pytest.raises(ValueError, match='rshift'):
            rshift(empty, 1.0)


class TestClipY:
    def test_limits_y_keeps_x(self):
        t = np.linspace(0, 10, 11)
        out = clip(_trans(t, t.copy()), low=2.0, high=5.0)
        np.testing.assert_array_equal(out.x, t)
        np.testing.assert_array_equal(out.y, np.clip(t, 2.0, 5.0))

    def test_one_sided(self):
        t = np.arange(5.0)
        assert clip(_trans(t, t.copy()), high=2.0).y.max() == 2.0
        assert clip(_trans(t, t.copy()), low=2.0).y.min() == 2.0

    def test_old_positional_form_raises(self):
        t = np.arange(5.0)
        with pytest.raises(TypeError):
            clip(_trans(t, t.copy()), 1.0, 3.0)

    def test_bad_bounds(self):
        t = np.arange(5.0)
        with pytest.raises(ValueError):
            clip(_trans(t, t.copy()))
        with pytest.raises(ValueError):
            clip(_trans(t, t.copy()), low=3.0, high=1.0)


class TestWindowMatchesSignalWindow:
    def test_same_samples(self):
        t = np.linspace(0, 10, 1001)
        s = _trans(t, np.sin(t))
        a, b = window(s, 2.0, 5.0), s.window(2.0, 5.0)
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.y, b.y)


class TestClip:
    def test_basic(self):
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, t.copy())
        clipped = window(sig, start=2.0, end=5.0)
        assert isinstance(clipped, Signal)
        assert clipped.x[0] >= 2.0
        assert clipped.x[-1] <= 5.0
        # y values match the originals at those x
        assert np.allclose(clipped.y, clipped.x)

    def test_invalid_window(self):
        t = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match='end > start'):
            window(_trans(t, t), start=0.5, end=0.5)


class TestFlip:
    def test_y_reversed(self):
        sig = _trans([0.0, 1.0, 2.0, 3.0], [10.0, 20.0, 30.0, 40.0])
        flipped = flip(sig)
        assert np.allclose(flipped.x, sig.x)
        assert np.allclose(flipped.y, [40.0, 30.0, 20.0, 10.0])

    def test_double_flip_identity(self):
        t = np.linspace(0, 1, 50)
        y = np.sin(2 * np.pi * t)
        sig = _trans(t, y)
        twice = flip(flip(sig))
        assert np.allclose(twice.y, sig.y)

    def test_uniform_grid_bit_identical_to_unchanged_x(self):
        """TASK-313 AC#2: uniform grid -> x unchanged, bit-identical."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        for y in (
            np.array([10.0, 20.0, 30.0, 40.0]),
            np.array([1 + 2j, 3 - 4j, 5j, -6 + 1j]),
            np.array([1.0, np.nan, 3.0, 4.0]),
        ):
            sig = _trans(x, y) if not np.iscomplexobj(y) else Signal(
                's', 'voltage', 'time', 'V', x, y)
            flipped = flip(sig)
            assert np.array_equal(flipped.x, x)
            assert np.array_equal(flipped.y, y[::-1], equal_nan=True)

    def test_non_uniform_grid_mirrors_x(self):
        """TASK-313 AC#1: SPICE-like non-uniform grid, asymmetric case."""
        sig = _trans([0.0, 1.0, 1.1, 3.0], [0.0, 0.0, 1.0, 2.0])
        flipped = flip(sig)
        assert np.allclose(flipped.x, [0.0, 1.9, 2.0, 3.0])
        assert np.allclose(flipped.y, [2.0, 1.0, 0.0, 0.0])

    def test_uniform_grid_x_exactly_unchanged(self):
        """TASK-313 AC#2: exact (bit-identical) x on realistic uniform grids.

        The naive mirror formula ``x[0] + x[-1] - x[::-1]`` is NOT
        bit-identical to ``x`` on these grids (measured up to 1 ULP on
        ~90% of samples) — ``flip`` must detect the uniform grid and
        return it unchanged rather than recompute it.
        """
        for x in (
            np.linspace(0, 1e-6, 1001),
            np.arange(10) * 0.1,
            np.linspace(5e-9, 105e-9, 4097),
        ):
            sig = _trans(x, np.arange(len(x), dtype=float))
            flipped = flip(sig)
            assert np.array_equal(flipped.x, x)


class TestSample:
    def test_n_uniform(self):
        # Original sig is dense; resample to 11 uniform points.
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, t.copy())
        s = sample(sig, n=11)
        assert len(s.x) == 11
        assert math.isclose(float(s.x[0]), 0.0, abs_tol=1e-12)
        assert math.isclose(float(s.x[-1]), 10.0, abs_tol=1e-12)
        # Linear sig → resample reproduces values
        assert np.allclose(s.y, s.x)

    def test_explicit_x_new(self):
        t = np.linspace(0, 10, 1001)
        sig = _trans(t, t.copy() ** 2)
        x_new = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        s = sample(sig, x_new=x_new)
        assert np.allclose(s.x, x_new)
        # Quadratic resampled is approximately quadratic
        assert math.isclose(float(s.y[2]), 25.0, rel_tol=0.01)

    def test_neither_arg(self):
        t = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="exactly one"):
            sample(_trans(t, t))

    def test_both_args(self):
        t = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="exactly one"):
            sample(_trans(t, t), x_new=[0.5], n=10)

    def test_n_too_small(self):
        t = np.linspace(0, 1, 10)
        with pytest.raises(ValueError, match='n must be >= 2'):
            sample(_trans(t, t), n=1)


class TestTie:
    def test_perfect_clock_zero_tie(self):
        # 100 Hz clock for 1 second; period = 0.01 s
        t = np.linspace(0, 1.0, 100001)
        y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
        tie_sig = tie(_trans(t, y), ideal_period=0.01)
        # Should be very small (limited by t-grid resolution)
        assert np.max(np.abs(tie_sig.y)) < 1e-4

    def test_auto_period(self):
        t = np.linspace(0, 1.0, 100001)
        y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
        tie_sig = tie(_trans(t, y))   # auto-detect period
        assert isinstance(tie_sig, Signal)
        assert tie_sig.xlabel == 'time'
        # Auto-period removes constant offset, so TIE should have ≈zero mean
        assert abs(float(np.mean(tie_sig.y))) < 1e-4

    def test_too_few_edges_returns_empty(self):
        t = np.linspace(0, 1, 100)
        sig = _trans(t, np.zeros_like(t))
        result = tie(sig)
        assert len(result.x) == 0

    def test_invalid_period(self):
        t = np.linspace(0, 1, 1001)
        sig = _trans(t, np.sin(2 * np.pi * 5 * t))
        with pytest.raises(ValueError, match='ideal_period must be > 0'):
            tie(sig, ideal_period=0)


class TestLongTermJitter:
    def test_clean_clock(self):
        t = np.linspace(0, 1.0, 100001)
        y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
        ltj = long_term_jitter(_trans(t, y), ideal_period=0.01)
        # Should be tiny — limited by t-grid resolution.
        assert ltj < 1e-4

    def test_too_few_edges(self):
        t = np.linspace(0, 1, 100)
        sig = _trans(t, np.zeros_like(t))
        assert math.isnan(long_term_jitter(sig))


class TestLeftoverImports:
    def test_top_level(self):
        from silk.metrics import lshift, clip, window, flip, sample, tie, long_term_jitter
        assert callable(lshift)
        assert callable(tie)


# ---------------------------------------------------------------------------
# Beyond-vendor — Allan deviation, cross-correlation
# ---------------------------------------------------------------------------

class TestAllanDeviation:
    def test_clean_clock_low_sigma(self):
        # 100 Hz clock for 1 s → ~100 edges; ideal clock has tiny Allan dev
        t = np.linspace(0, 1.0, 100001)
        y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
        result = allan_deviation(_trans(t, y))
        assert isinstance(result, Signal)
        assert result.xlabel == 'tau'
        assert len(result.x) > 0
        # All sigmas should be small (limited by t-grid quantisation)
        finite = result.y[np.isfinite(result.y)]
        assert np.max(finite) < 1e-3

    def test_returns_empty_on_no_clock(self):
        t = np.linspace(0, 1, 100)
        result = allan_deviation(_trans(t, np.zeros_like(t)))
        assert len(result.x) == 0

    def test_explicit_taus(self):
        t = np.linspace(0, 1.0, 100001)
        y = (np.sin(2 * np.pi * 100 * t) > 0).astype(float)
        my_taus = np.array([0.01, 0.05, 0.1])
        result = allan_deviation(_trans(t, y), taus=my_taus)
        assert np.allclose(result.x, my_taus)


class TestCrossCorrelation:
    def test_self_correlation_peak_at_zero(self):
        # Auto-correlation of a pulse peaks at lag = 0
        dx = 0.001
        t = np.arange(0, 1.0, dx)
        # Gaussian pulse centered at 0.5
        y = np.exp(-((t - 0.5) ** 2) / (2 * 0.05 ** 2))
        sig = _trans(t, y)
        c = cross_correlation(sig, sig)
        peak_idx = int(np.argmax(c.y))
        # For autocorrelation in 'full' mode, lag 0 sits at index n-1
        # where n = len of one input.
        assert math.isclose(float(c.x[peak_idx]), 0.0, abs_tol=2 * dx)

    def test_shift_detection(self):
        # b is a copy of a shifted right by 0.1 s; xcorr peak at -0.1.
        # Wait — c = correlate(a, b) peaks where a aligns with shifted-b.
        # If b lags a by τ, then peak is at lag = +τ.
        dx = 0.001
        t = np.arange(0, 1.0, dx)
        center_a = 0.5
        center_b = 0.6   # b is 0.1 later than a
        y_a = np.exp(-((t - center_a) ** 2) / (2 * 0.02 ** 2))
        y_b = np.exp(-((t - center_b) ** 2) / (2 * 0.02 ** 2))
        c = cross_correlation(_trans(t, y_a), _trans(t, y_b))
        peak_idx = int(np.argmax(c.y))
        # numpy's correlate(a, b) puts peak at the lag where shifting b
        # by that amount maximises overlap with a; b is right-of-a so the
        # peak is at lag = -(center_b - center_a) = -0.1
        assert math.isclose(float(c.x[peak_idx]), -0.1, abs_tol=3 * dx)

    def test_mismatched_dx(self):
        a = _trans(np.linspace(0, 1, 11), np.ones(11))
        b = _trans(np.linspace(0, 1, 21), np.ones(21))
        with pytest.raises(ValueError, match='same dx'):
            cross_correlation(a, b)

    def test_start_time_offset_included_in_lag(self):
        # a and b are identical Gaussian pulses (same shape, same
        # position within their own sample arrays); b's x-axis simply
        # starts 0.2 s later than a's, so b's feature occurs 0.2 s
        # later in absolute time than a's — b is delayed relative to a.
        dx = 0.001
        n = 1000
        t = np.arange(n) * dx
        y = np.exp(-((t - 0.3) ** 2) / (2 * 0.02 ** 2))
        a = _trans(t, y)
        b = _trans(t + 0.2, y)
        c = cross_correlation(a, b)
        peak_idx = int(np.argmax(c.y))
        # Documented sign convention: b delayed relative to a -> negative
        # peak lag, equal to a.x[0] - b.x[0] = -0.2 here.
        assert math.isclose(float(c.x[peak_idx]), -0.2, abs_tol=3 * dx)

    def test_shared_start_time_unchanged(self):
        # When both inputs share a start time, a.x[0] - b.x[0] == 0 and
        # the lag axis must be bit-identical to the pre-fix behaviour
        # (plain sample-index lags scaled by dx).
        dx = 0.001
        t = np.arange(0, 1.0, dx)
        y_a = np.exp(-((t - 0.5) ** 2) / (2 * 0.05 ** 2))
        y_b = np.exp(-((t - 0.6) ** 2) / (2 * 0.05 ** 2))
        a = _trans(t, y_a)
        b = _trans(t, y_b)
        c = cross_correlation(a, b)
        n_a, n_b = len(y_a), len(y_b)
        expected_lags = np.arange(-(n_b - 1), n_a) * dx
        assert np.array_equal(np.asarray(c.x), expected_lags)


class TestBeyondVendorTransientImports:
    def test_top_level(self):
        from silk.metrics import allan_deviation, cross_correlation
        assert callable(allan_deviation)


class TestTimeGridTolerance310:
    """TASK-310: grids must agree within a fraction of a time step."""

    @staticmethod
    def _grid():
        t = np.arange(0.0, 100e-9, 0.1e-9)  # 100 MHz x 10 cycles, 0.1 ns step
        return t, np.sin(2 * np.pi * 100e6 * t)

    def test_quarter_period_shift_raises_average_power(self):
        t, s = self._grid()
        v = _v(t, s)
        i = lshift(_i(t, s), -2.5e-9)  # quarter period, 25 steps
        with pytest.raises(ValueError, match='same time axis'):
            average_power(v, i)

    def test_one_step_shift_raises_all_three(self):
        t, s = self._grid()
        v = _v(t, s)
        i = lshift(_i(t, s), -0.1e-9)
        with pytest.raises(ValueError, match='same time axis'):
            average_power(v, i)
        with pytest.raises(ValueError, match='same time axis'):
            dynamic_energy_per_cycle(v, i, period=10e-9)
        vi = _v(t, s)
        with pytest.raises(ValueError, match='x-axis'):
            hysteresis(vi, v, vi, lshift(_v(t, s), -0.1e-9))
        with pytest.raises(ValueError, match='x-axis'):
            hysteresis(vi, lshift(_v(t, s), -0.1e-9), vi, v)

    def test_float_noise_still_accepted(self):
        t, s = self._grid()
        noisy = t * (1 + 1e-13) + 1e-20
        assert math.isfinite(average_power(_v(t, s), _i(noisy, s)))

    def test_different_length_raises(self):
        t, s = self._grid()
        with pytest.raises(ValueError, match='same time axis'):
            average_power(_v(t, s), _i(t[:-1], s[:-1]))


class TestEdgePlurals:
    """rise_times / fall_times / slew_rates: every edge, same core (TASK-372)."""

    @staticmethod
    def _wave():
        # two rising and two falling ramps of different slopes, 0 -> 1 V
        t = np.linspace(0, 10, 2001)
        y = np.interp(t, [0, 1, 1.5, 3, 3.4, 5, 5.2, 7, 7.8, 10],
                      [0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        return _trans(t, y)

    def test_each_element_equals_the_singular_with_that_n(self):
        from silk.metrics import rise_times, fall_times, slew_rates
        sig = self._wave()
        cases = [
            (rise_times, rise_time, {}),
            (fall_times, fall_time, {}),
            (slew_rates, slew_rate, {'edge': 'any'}),
            (rise_times, rise_time, {'start': 0.5, 'end': 6.0}),
            (rise_times, rise_time, {'initial': 0.0, 'final': 1.0}),
            (fall_times, fall_time, {'initial': 1.0, 'final': 0.0}),
            (slew_rates, slew_rate, {'edge': 'any', 'initial': 0.0,
                                     'final': 1.0, 'start': 0.5}),
        ]
        for plural, single, kw in cases:
            many = plural(sig, **kw)
            assert isinstance(many, np.ndarray) and many.size >= 2, (plural, kw)
            one = [single(sig, n=k + 1, **kw) for k in range(many.size)]
            np.testing.assert_array_equal(many, one)

    def test_values_in_time_order(self):
        from silk.metrics import rise_times
        got = rise_times(self._wave())
        assert got == pytest.approx([0.4, 0.16], abs=2e-3)

    def test_no_edge_in_range_is_empty_with_the_singulars_warning(self):
        from silk.metrics import rise_times
        sig = self._wave()
        with pytest.warns(MetricWarning) as c:
            got = rise_times(sig, start=3.2, end=3.3, initial=0.0, final=1.0)
        assert isinstance(got, np.ndarray) and got.shape == (0,)
        with pytest.warns(MetricWarning) as c1:
            assert math.isnan(rise_time(sig, start=3.2, end=3.3,
                                        initial=0.0, final=1.0))
        assert c[0].message.code == c1[0].message.code == 'missing_crossing'

    def test_stated_level_never_crossed_is_empty_and_named(self):
        from silk.metrics import rise_times
        with pytest.warns(MetricWarning) as c:
            got = rise_times(self._wave(), initial=0.0, final=5.0)
        assert got.shape == (0,)
        assert c[0].message.code == 'missing_crossing'

    def test_plurals_take_no_n_and_are_unregistered(self):
        from silk.metrics import rise_times, fall_times, slew_rates
        from silk.signals.schema import export_metric_schema
        for f in (rise_times, fall_times, slew_rates):
            with pytest.raises(TypeError):
                f(self._wave(), n=1)
        schema = export_metric_schema()
        for name in ('rise_times', 'fall_times', 'slew_rates'):
            assert name not in schema
        for name in ('rise_time', 'fall_time', 'slew_rate'):
            assert name in schema


class TestErrorNormWindowAndWeighting:
    """TASK-385: mean_abs, start/end, time-weighted rms/mean_abs."""

    @pytest.mark.parametrize('kind', ['max', 'rms', 'mean_abs'])
    def test_explicit_one_sample_window_raises_for_every_kind(self, kind):
        sig = _trans([0., 1., 2.], [2., 3., 4.])
        goal = _trans([0., 1., 2.], [0., 0., 0.])
        with pytest.raises(ValueError, match='error_norm.*contains 1 samples'):
            error_norm(sig, goal, kind=kind, start=0.5, end=1.5)

    @pytest.mark.parametrize('kind', ['max', 'rms', 'mean_abs'])
    def test_single_sample_without_window_returns_absolute_error(self, kind):
        assert error_norm(_trans([0.], [3.]), _trans([0.], [1.]),
                          kind=kind) == 2.0

    @staticmethod
    def _grid():
        # Strongly non-uniform: dense near 0, sparse after. e(t) = t on [0, 1].
        return np.concatenate([np.linspace(0, 0.1, 400, endpoint=False),
                               np.linspace(0.1, 1, 10)])

    def test_nonuniform_grid_matches_analytic(self):
        t = self._grid()
        s, g = _trans(t, t.copy()), _trans(t, np.zeros_like(t))
        # e = t: L1 = 1/2, L2 = 1/sqrt(3); a plain sample mean would be ~0.1 / 0.2
        assert math.isclose(error_norm(s, g, kind='mean_abs'), 0.5, rel_tol=2e-2)
        assert math.isclose(error_norm(s, g, kind='rms'), 3 ** -0.5, rel_tol=2e-2)
        assert error_norm(s, g, kind='max') == 1.0

    def test_window(self):
        t = self._grid()
        s, g = _trans(t, t.copy()), _trans(t, np.zeros_like(t))
        # window [0.5, 1]: mean of t = 0.75, max = 1
        assert math.isclose(error_norm(s, g, kind='mean_abs', start=0.5),
                            0.75, rel_tol=1e-2)
        assert error_norm(s, g, kind='max', end=0.5) <= 0.5

    def test_bad_window_raises(self):
        t = np.linspace(0, 1, 11)
        s = _trans(t, t)
        with pytest.raises(ValueError, match='error_norm'):
            error_norm(s, s, start=0.6, end=0.4)
        with pytest.raises(ValueError, match='error_norm'):
            error_norm(s, s, start=2.0)


class TestRelativeErrorNorm:
    """TASK-386: dimensionless error norm with an exclusion floor."""

    @pytest.mark.parametrize('kind', ['max', 'rms', 'mean_abs'])
    def test_explicit_one_sample_window_raises_for_every_kind(self, kind):
        sig = _trans([0., 1., 2.], [2., 3., 4.])
        goal = _trans([0., 1., 2.], [1., 1., 1.])
        with pytest.raises(ValueError, match='relative_error_norm.*contains 1 samples'):
            relative_error_norm(sig, goal, kind=kind, floor=1.,
                                start=0.5, end=1.5)

    @staticmethod
    def _pair(rel=0.1):
        t = np.linspace(0, 1, 1001)
        g = 2.0 + t                       # goal in [2, 3]
        return _trans(t, g * (1 + rel)), _trans(t, g), t, g

    def test_constant_relative_error_every_kind(self):
        s, g, _, _ = self._pair(0.1)
        for kind in ('max', 'rms', 'mean_abs'):
            assert math.isclose(relative_error_norm(s, g, kind=kind, floor=1.0),
                                0.1, rel_tol=1e-9)

    def test_varying_error_analytic(self):
        t = np.linspace(0, 1, 100001)
        g = _trans(t, np.ones_like(t) * 2.0)
        s = _trans(t, 2.0 + t)            # rel err = t / 2
        assert math.isclose(relative_error_norm(s, g, kind='max', floor=1.0),
                            0.5, rel_tol=1e-9)
        assert math.isclose(relative_error_norm(s, g, kind='mean_abs', floor=1.0),
                            0.25, rel_tol=1e-6)
        assert math.isclose(relative_error_norm(s, g, kind='rms', floor=1.0),
                            math.sqrt(1 / 12), rel_tol=1e-6)

    def test_floor_excludes_near_zero_goal(self):
        t = np.linspace(0, 1, 1001)
        g = _trans(t, t.copy())           # goal 0 -> 1
        s = _trans(t, 1.1 * t)            # 10 % everywhere except t=0
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            v = relative_error_norm(s, g, kind='max', floor=0.5)
        assert math.isclose(v, 0.1, rel_tol=1e-9)

    def test_window_applies(self):
        t = np.linspace(0, 1, 1001)
        g = _trans(t, np.ones_like(t))
        s = _trans(t, np.where(t < 0.5, 1.0, 1.2))
        assert relative_error_norm(s, g, kind='max', floor=0.1, end=0.4) == 0.0
        assert math.isclose(relative_error_norm(s, g, kind='max', floor=0.1,
                                                start=0.6), 0.2, rel_tol=1e-9)

    def test_all_excluded_is_nan_with_warning_naming_floor(self):
        s, g, _, _ = self._pair()
        with pytest.warns(MetricWarning, match='floor'):
            v = relative_error_norm(s, g, kind='rms', floor=10.0)
        assert math.isnan(v)

    def test_bad_floor_and_kind_raise(self):
        s, g, _, _ = self._pair()
        with pytest.raises(ValueError, match='floor'):
            relative_error_norm(s, g, kind='rms', floor=-1.0)
        with pytest.raises(ValueError, match='relative_error_norm'):
            relative_error_norm(s, g, kind='bad', floor=1.0)

    def test_floor_and_kind_are_required(self):
        s, g, _, _ = self._pair()
        with pytest.raises(TypeError):
            relative_error_norm(s, g, kind='rms')
        with pytest.raises(TypeError):
            relative_error_norm(s, g, floor=1.0)

    def test_registered_dimensionless(self):
        from silk.signals.schema import export_metric_schema
        entry = export_metric_schema()['relative_error_norm']
        assert entry['unit'] == ''


class TestArgExtremaRefine380:
    """TASK-380: optional parabolic sub-sample refinement of argmin_x/argmax_x."""

    @staticmethod
    def _cos(t0, n=21, span=2.0):
        t = np.linspace(0.0, span, n)
        return _trans(t, np.cos(np.pi * (t - t0)), 'c')

    def test_default_returns_sample_x(self):
        sig = self._cos(0.93)
        assert argmax_x(sig) == argmax_x(sig, refine=False)
        assert argmax_x(sig) == pytest.approx(0.9)

    def test_refine_recovers_off_grid_peak_and_trough(self):
        sig = self._cos(0.93)                      # peak at 0.93, grid step 0.1
        assert abs(argmax_x(sig, refine=True) - 0.93) < 2e-3
        assert abs(argmin_x(sig, refine=True) - 1.93) < 2e-3   # 0.93 + 1 period/2

    def test_refine_falls_back_at_window_edge(self):
        sig = self._cos(0.0)                       # peak on first sample
        assert argmax_x(sig, refine=True) == 0.0
        sig = self._cos(0.93)
        assert argmax_x(sig, start=0.0, end=0.5, refine=True) == pytest.approx(0.5)

    def test_refine_clamped_to_neighbours(self):
        sig = _trans([0., 1., 2.], [0., 5., 4.9])
        assert 0.0 <= argmax_x(sig, refine=True) <= 2.0


class TestComplexDefinedMetrics:
    """Hand quadrature on log-spaced frequencies catches magnitude projection."""

    @staticmethod
    def signal():
        return Signal('h', 'voltage', 'frequency', 'V',
                      np.array([1., 10., 100.]),
                      np.array([1+0j, 0+1j, -1+0j]))

    @pytest.mark.parametrize('name,params,expected', [
        ('average', {}, -9/22 + 0.5j),
        ('integral', {}, -40.5 + 49.5j),
        ('value_at', {'x': 55}, -0.5 + 0.5j),
        ('value_at', {'x': 10}, 1j),
        ('final_value', {}, -1+0j),
        # Window [5.5, 55] inserts two Cartesian midpoints.
        ('integral', {'start': 5.5, 'end': 55}, -10.125 + 37.125j),
        ('average', {'start': 5.5, 'end': 55}, -9/44 + 0.75j),
        ('average', {'start': 5.5, 'end': 100, 'interpolate': False}, -0.5+0.5j),
    ])
    @pytest.mark.parametrize("descending", [False, True])
    def test_complex_result(self, name, params, expected, descending):
        import silk.metrics as M
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            sig = self.signal()
            if descending:
                sig = Signal('h', 'voltage', 'frequency', 'V', sig.x[::-1], sig.y[::-1])
                if name == 'final_value':
                    expected = 1+0j
            result = getattr(M, name)(sig, **params)
        assert type(result) is complex
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize('name,expected', [
        ('variance', 141/242), ('stddev', math.sqrt(141/242)),
    ])
    @pytest.mark.parametrize("descending", [False, True])
    def test_spread_about_complex_mean(self, name, expected, descending):
        import silk.metrics as M
        # |y| is constant 1; E[|y-mu|²] = 1 - |(-9/22)+i/2|².
        sig = self.signal()
        if descending:
            sig = Signal('h', 'voltage', 'frequency', 'V', sig.x[::-1], sig.y[::-1])
        result = getattr(M, name)(sig)
        assert type(result) is float
        assert result == pytest.approx(expected)
        assert getattr(M, name)(self.signal().mag()) == 0

    def test_spread_uses_cartesian_window_edges(self):
        import silk.metrics as M
        # [5.5, 55] gives 0.5+0.5j, 1j, -0.5+0.5j. Weighted mean
        # is -9/44 + 3j/4 and mean squared magnitude is 3/4.
        assert M.variance(self.signal(), start=5.5, end=55) == pytest.approx(141/968)
        assert M.stddev(self.signal(), start=5.5, end=55) == pytest.approx(math.sqrt(141/968))

    def test_complex_single_point_final_value(self):
        sig = Signal('h', 'voltage', 'frequency', 'V',
                     np.array([10.]), np.array([3+4j]))
        from silk.metrics import final_value
        assert final_value(sig) == 3+4j
