"""Tests for silk.metrics — analog performance metric functions."""
import math
import numpy as np
import pytest
from silk.signals.base import Signal
from silk.metrics import (
    dc_gain_dB, gain_at_freq, bandwidth_3dB, unity_gain_freq,
    phase_at_freq, phase_margin, gain_margin, gain_margins, cmrr_dB,
    integrated_noise, peak_gain_dB,
)


def _make_sig(x, y, name='test'):
    """Helper: build a Signal from numpy arrays."""
    return Signal(name, 'voltage', 'frequency', 'V', x, y)


def _ac_sig(freqs, magnitudes, phases_deg=None):
    """Build a complex AC signal."""
    if phases_deg is None:
        phases_deg = np.zeros_like(magnitudes)
    y = np.array(magnitudes) * np.exp(1j * np.radians(phases_deg))
    return _make_sig(np.array(freqs, dtype=float), y)


class TestDcGainDB:
    def test_flat_gain_of_10(self):
        sig = _ac_sig([1, 10, 100], [10, 10, 10])
        assert math.isclose(dc_gain_dB(sig), 20.0, rel_tol=1e-6)

    def test_flat_gain_of_100(self):
        sig = _ac_sig([1, 10, 100], [100, 100, 100])
        assert math.isclose(dc_gain_dB(sig), 40.0, rel_tol=1e-6)

    def test_peaked_response_reads_dc_not_peak(self):
        """THE semantic fix (2026-08-23): the name says DC, so a peaked
        response must report the lowest-frequency point — the old peak
        behavior (40 dB here) moved to peak_gain_dB."""
        sig = _ac_sig([1, 10, 100], [1, 100, 1])
        assert math.isclose(dc_gain_dB(sig), 0.0, abs_tol=1e-9)
        assert math.isclose(peak_gain_dB(sig), 40.0, rel_tol=1e-6)

    def test_dc_gain_is_order_independent(self):
        """The lowest FREQUENCY wins, not index 0 — a descending sweep
        must give the same answer as an ascending one."""
        up   = _ac_sig([1, 10, 100], [50, 10, 1])
        down = _ac_sig([100, 10, 1], [1, 10, 50])
        assert math.isclose(dc_gain_dB(up), dc_gain_dB(down), rel_tol=1e-9)
        assert math.isclose(dc_gain_dB(up), 20*math.log10(50), rel_tol=1e-6)

    def test_second_order_peaking_analytic(self):
        """Underdamped 2nd-order low-pass, zeta=0.05: true DC gain 0 dB,
        true peak 1/(2*zeta*sqrt(1-zeta^2)) = +20.01 dB. Both functions
        against the analytic values — this is the exact case the old
        behavior got wrong by the full 20 dB."""
        import numpy as np
        zeta, f0 = 0.05, 1e3
        f = np.logspace(0, 5, 2000)
        s = 1j * f / f0
        y = 1.0 / (1 + 2*zeta*s + s**2)
        sig = _make_sig(f, y)
        peak_expected = 20*math.log10(1/(2*zeta*math.sqrt(1-zeta**2)))
        assert math.isclose(dc_gain_dB(sig), 0.0, abs_tol=1e-3)
        assert math.isclose(peak_gain_dB(sig), peak_expected, abs_tol=0.01)

    def test_unity_gain(self):
        sig = _ac_sig([1, 10, 100], [1, 1, 1])
        assert math.isclose(dc_gain_dB(sig), 0.0, abs_tol=1e-9)


class TestGainAtFreq:
    def test_exact_match(self):
        sig = _ac_sig([1.0, 10.0, 100.0], [5.0, 10.0, 2.0])
        assert math.isclose(gain_at_freq(sig, 10.0), 10.0, rel_tol=1e-6)

    def test_nearest_sample(self):
        sig = _ac_sig([1.0, 10.0, 100.0], [5.0, 10.0, 2.0])
        # 9.0 is closer to 10.0 than to 1.0
        assert math.isclose(gain_at_freq(sig, 9.0, interpolate=False), 10.0,
                            rel_tol=1e-6)

    def test_interpolates_by_default(self):
        sig = _ac_sig([1.0, 10.0, 100.0], [5.0, 10.0, 2.0])
        # halfway in log frequency between 10 (|y|=10) and 100 (|y|=2): log-log mean
        assert math.isclose(gain_at_freq(sig, 10 ** 1.5), math.sqrt(20.0),
                            rel_tol=1e-9)


class TestBandwidth3dB:
    def test_single_pole_rolloff(self):
        # Gain of 1 at f=0 rolling off, 3dB point at f=1
        freqs = np.logspace(-1, 2, 200)
        mag = 1.0 / np.sqrt(1 + (freqs / 1.0) ** 2)
        sig = _ac_sig(freqs, mag)
        bw = bandwidth_3dB(sig)
        assert math.isclose(bw, 1.0, rel_tol=0.05)

    def test_flat_gain_returns_nan(self):
        """A response still at (or above) -3dB at the top of the sweep is
        indistinguishable from one that never rolls off — the true -3dB
        point, if any, is above the swept range and unknowable from this
        data (TASK-63, folded in from TASK-57). Was: returned the sweep's
        top frequency, indistinguishable from a real -3dB point there."""
        freqs = [1.0, 10.0, 100.0]
        sig = _ac_sig(freqs, [10.0, 10.0, 10.0])
        assert math.isnan(bandwidth_3dB(sig))


class TestUnityGainFreq:
    def test_crosses_at_known_freq(self):
        freqs = np.array([1.0, 10.0, 100.0, 1000.0])
        mag   = np.array([10.0, 2.0, 0.5, 0.1])
        sig = _ac_sig(freqs, mag)
        ugf = unity_gain_freq(sig)
        assert 10.0 < ugf < 100.0

    def test_gain_always_above_1_returns_nan(self):
        sig = _ac_sig([1, 10, 100], [10, 5, 2])
        assert math.isnan(unity_gain_freq(sig))

    def test_gain_always_below_1_returns_nan(self):
        sig = _ac_sig([1, 10, 100], [0.1, 0.05, 0.01])
        assert math.isnan(unity_gain_freq(sig))


class TestPhaseAtFreq:
    def test_known_phase(self):
        freqs = [1.0, 10.0, 100.0]
        mag   = [1.0, 1.0, 1.0]
        phases = [0.0, -45.0, -90.0]
        sig = _ac_sig(freqs, mag, phases)
        assert math.isclose(phase_at_freq(sig, 10.0), -45.0, abs_tol=1e-6)


class TestPhaseMargin:
    def test_typical_amplifier(self):
        # Unity gain at 10 Hz, phase = -120 deg → PM = 60 deg
        freqs = np.array([1.0, 10.0, 100.0])
        mag   = np.array([10.0, 1.0, 0.1])
        phases= np.array([-60.0, -120.0, -160.0])
        sig = _ac_sig(freqs, mag, phases)
        pm = phase_margin(sig)
        assert math.isclose(pm, 60.0, rel_tol=0.05)

    def test_no_ugf_returns_nan(self):
        sig = _ac_sig([1, 10, 100], [0.1, 0.05, 0.01])
        assert math.isnan(phase_margin(sig))


class TestGainMargin:
    def test_positive_gain_margin(self):
        # Phase goes from -135 to -200 (crosses -180 between i=1 and i=2)
        # gain at phase crossing is between 2.0 and 0.5 — unstable system has negative GM
        # Stable system: phase crosses -180 when gain < 1
        freqs = np.array([1.0, 10.0, 100.0, 1000.0])
        mag   = np.array([10.0, 0.2, 0.05, 0.01])   # gain already < 1 at phase crossing
        phases= np.array([-90.0, -135.0, -180.0, -200.0])
        sig = _ac_sig(freqs, mag, phases)
        gm = gain_margin(sig)
        # triggers at i=1: phase -135 > -180, phase -180 >= -180. gain=0.2
        # GM = -20*log10(0.2) = 13.98 dB (positive = stable)
        assert gm > 0

    def test_negative_gain_margin_unstable(self):
        # |gain| > 1 AT the interpolated -180° crossing → negative GM.
        # (The pre-2026-08-23 version of this test relied on the old
        # sample-before-the-crossing read; interpolation is now exact.)
        freqs = np.array([1.0, 10.0, 100.0, 1000.0])
        mag   = np.array([10.0, 4.0, 2.0, 0.1])
        phases= np.array([-90.0, -135.0, -181.0, -200.0])
        sig = _ac_sig(freqs, mag, phases)
        gm = gain_margin(sig)
        assert gm < 0   # unstable: gain > 1 where phase crosses -180

    def test_no_crossing_gain_still_high_returns_nan(self):
        """No crossover AND |gain| still >= 1 at the top of the sweep: the
        sweep ended before the answer was knowable — NaN, not a number."""
        sig = _ac_sig([1, 10, 100], [10, 5, 2], [0, -45, -90])
        assert math.isnan(gain_margin(sig))


class TestCmrrDB:
    def test_freq_argument_picks_nearest_sample(self):
        """2026-08-23: the measurement frequency is explicit. A rolling-off
        differential gain gives a different (lower) CMRR at high frequency —
        and the default is the LOWEST frequency, order-independent, not
        index 0."""
        diff = _ac_sig([1.0, 1e3, 1e6], [1000.0, 1000.0, 10.0])
        cm   = _ac_sig([1.0, 1e3, 1e6], [1.0, 1.0, 1.0])
        assert math.isclose(cmrr_dB(diff, cm), 60.0, rel_tol=1e-6)
        assert math.isclose(cmrr_dB(diff, cm, freq=1e6), 20.0, rel_tol=1e-6)
        # descending sweep: same default answer (lowest freq, not sample 0)
        diff_d = _ac_sig([1e6, 1e3, 1.0], [10.0, 1000.0, 1000.0])
        cm_d   = _ac_sig([1e6, 1e3, 1.0], [1.0, 1.0, 1.0])
        assert math.isclose(cmrr_dB(diff_d, cm_d), 60.0, rel_tol=1e-6)

    def test_typical_cmrr(self):
        diff = _ac_sig([1, 10], [1000, 1000])
        cm   = _ac_sig([1, 10], [1, 1])
        assert math.isclose(cmrr_dB(diff, cm), 60.0, rel_tol=1e-6)

    def test_zero_cm_gain_returns_inf(self):
        diff = _ac_sig([1, 10], [100, 100])
        cm   = _ac_sig([1, 10], [0, 0])
        assert cmrr_dB(diff, cm) == float('inf')


class TestIntegratedNoise:
    def test_white_noise(self):
        # Flat 1 nV/sqrt(Hz) noise from 1 Hz to 100 Hz → RMS = 1e-9 * sqrt(99) ≈ 9.95 nV
        freqs = np.linspace(1.0, 100.0, 1000)
        psd   = np.full_like(freqs, 1e-9)
        sig = _make_sig(freqs, psd)
        result = integrated_noise(sig, 1.0, 100.0)
        expected = 1e-9 * np.sqrt(99.0)
        assert math.isclose(result, expected, rel_tol=0.01)

    def test_out_of_band_raises(self):
        # TASK-56: an empty band is a malformed request, not a
        # measurement — raises rather than silently reading as 0 V
        # noise (a perfect result).
        freqs = np.array([1.0, 10.0, 100.0])
        sig = _make_sig(freqs, np.array([1e-9, 1e-9, 1e-9]))
        with pytest.raises(ValueError, match='integrated_noise'):
            integrated_noise(sig, 200.0, 1000.0)


class TestImports:
    def test_metrics_shim_is_removed(self):
        import importlib
        with pytest.raises(ModuleNotFoundError, match=r"silk\.signals\.metrics"):
            importlib.import_module('silk.signals.metrics')

    @pytest.mark.parametrize('name', [
        'dc_gain_dB', 'peak_gain_dB', 'gain_at_freq', 'bandwidth_3dB',
        'unity_gain_freq', 'phase_at_freq', 'phase_margin', 'gain_margins',
        'gain_margin', 'cmrr_dB', 'integrated_noise',
    ])
    def test_metrics_have_one_public_namespace(self, name):
        import silk.signals as signals
        import silk.metrics as metrics
        from silk.metrics import ac, noise
        assert not hasattr(signals, name)
        assert name not in signals.__all__
        owner = noise if name == 'integrated_noise' else ac
        assert getattr(metrics, name) is getattr(owner, name)

    def test_signals_does_not_export_metrics_module(self):
        import silk.signals as signals
        assert 'metrics' not in signals.__all__
        assert not hasattr(signals, 'metrics')


class TestGainMarginWrappedPhase:
    """The 2026-08-23 fix: np.angle returns WRAPPED phase, so before the
    np.unwrap in gain_margin, any system whose true phase continues past
    -180° (any 3+-pole loop — exactly what gain margin exists to catch)
    jumped -180 → +180 in the raw array, the crossing never fired, and a
    real, computable, often NEGATIVE (unstable) margin came back NaN."""

    @staticmethod
    def _three_pole(freqs, k=100.0, f1=1e3, f2=5e3, f3=2e4):
        """H(f) = k / ((1+jf/f1)(1+jf/f2)(1+jf/f3)) — true phase → -270°."""
        jf = 1j * np.asarray(freqs, dtype=float)
        return k / ((1 + jf / f1) * (1 + jf / f2) * (1 + jf / f3))

    def test_three_pole_unstable_margin_found_not_nan(self):
        # Analytic phase-crossover: atan(f/f1)+atan(f/f2)+atan(f/f3) = pi,
        # solved by bisection — independent of the metric under test.
        def total_phase(f):
            return math.atan(f / 1e3) + math.atan(f / 5e3) + math.atan(f / 2e4)
        lo, hi = 1e3, 1e6
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if total_phase(mid) < math.pi:
                lo = mid
            else:
                hi = mid
        f180 = math.sqrt(lo * hi)
        expected_gm = -20.0 * math.log10(abs(self._three_pole([f180])[0]))
        assert expected_gm < 0        # genuinely unstable case

        freqs = np.logspace(1, 6, 1500)          # dense grid, 300 pts/decade
        sig = _make_sig(freqs, self._three_pole(freqs))
        gm = gain_margin(sig)
        assert not math.isnan(gm)                # was nan before the unwrap
        # value taken at the sample before the crossing → grid-quantized
        assert math.isclose(gm, expected_gm, abs_tol=0.5)

    def test_two_pole_rolled_off_returns_inf(self):
        """A 2-pole loop never crosses -180 and its gain has rolled off
        well below 1 by the top of the sweep: genuinely unbounded margin —
        +inf, so a `gain_margin >= 10` spec PASSES the safest possible loop
        (the old NaN failed it)."""
        freqs = np.logspace(1, 7, 800)
        jf = 1j * freqs
        y = 100.0 / ((1 + jf / 1e3) * (1 + jf / 1e4))
        gm = gain_margin(_make_sig(freqs, y))
        assert math.isinf(gm) and gm > 0


class TestPhaseMarginWrappedPhase:
    """Same 2026-08-23 unwrap family as gain_margin, worse consequence:
    phase_at_freq reads WRAPPED np.angle, so an unstable loop (true phase
    past -180° at unity gain) reported PM + 360° — a true -62° came back
    +298° and silently PASSED a `phase_margin >= 45` spec."""

    @staticmethod
    def _loop(k):
        f = np.logspace(1, 7, 4000)
        jf = 1j * f
        return _make_sig(f, k / ((1 + jf/1e3) * (1 + jf/2e4) * (1 + jf/5e4)))

    def test_unstable_loop_reports_negative_pm(self):
        sig = self._loop(3000.0)
        pm = phase_margin(sig)
        # independent truth: unwrapped phase at the |H|=1 sample
        mag = np.abs(sig.y)
        ph = np.degrees(np.unwrap(np.angle(sig.y)))
        true_pm = 180.0 + ph[int(np.argmin(np.abs(mag - 1.0)))]
        assert true_pm < 0                       # genuinely unstable case
        assert math.isclose(pm, true_pm, abs_tol=1.0)
        assert pm < 0                            # was +298 before the fix

    def test_stable_loop_unchanged(self):
        sig = self._loop(20.0)                   # low gain → UGF before poles bite
        pm = phase_margin(sig)
        mag = np.abs(sig.y)
        ph = np.degrees(np.unwrap(np.angle(sig.y)))
        true_pm = 180.0 + ph[int(np.argmin(np.abs(mag - 1.0)))]
        assert true_pm > 0
        assert math.isclose(pm, true_pm, abs_tol=1.0)


class TestGainMarginCrossingSelection:
    """The 2026-08-23 redesign (research-backed: Åström & Murray §10.3,
    python-control stability_margins): gain_margin returns
    the crossover margin CLOSEST TO 0 dB across ALL negative-real-axis
    crossings — both directions, every -180°+k·360° level, interpolated —
    and gain_margins() exposes them all."""

    @staticmethod
    def _cond_stable(freqs, k=3e4):
        """Conditionally stable loop: triple pole at 10 Hz drives phase to
        -270°, double zero at 1 kHz recovers it back UP through -180°,
        double pole at 100 kHz sends it down through -180° again.
        Three crossings: down, up, down."""
        jf = 1j * np.asarray(freqs, dtype=float)
        return (k * (1 + jf / 1e3) ** 2
                / ((1 + jf / 10.0) ** 3 * (1 + jf / 1e5) ** 2))

    @classmethod
    def _analytic_crossings(cls):
        """Independent truth: solve phase(f) = -180 (mod 360) by bisection
        on the ANALYTIC phase function — no shared code with the metric."""
        def phase(f):
            return (2 * math.atan(f / 1e3)
                    - 3 * math.atan(f / 10.0)
                    - 2 * math.atan(f / 1e5))
        def solve(lo, hi, level):
            for _ in range(200):
                mid = math.sqrt(lo * hi)
                if (phase(mid) - level) * (phase(lo) - level) > 0:
                    lo = mid
                else:
                    hi = mid
            return math.sqrt(lo * hi)
        level = -math.pi
        # bracket the three crossings by scanning
        f_scan = np.logspace(0, 7, 20000)
        ph = np.array([phase(f) for f in f_scan])
        sign = np.sign(ph - level)
        idx = np.where(np.diff(sign) != 0)[0]
        roots = [solve(f_scan[i], f_scan[i + 1], level) for i in idx]
        gms = [-20 * math.log10(abs(cls._cond_stable([r])[0])) for r in roots]
        return list(zip(roots, gms))

    def test_all_three_crossings_found_and_binding_one_selected(self):
        truth = self._analytic_crossings()
        assert len(truth) == 3                       # down, up, down
        freqs = np.logspace(0, 7, 3000)
        sig = _make_sig(freqs, self._cond_stable(freqs))
        found = gain_margins(sig)
        assert len(found) == 3
        for (f_t, gm_t), (f_m, gm_m) in zip(truth, found):
            assert math.isclose(f_m, f_t, rel_tol=0.02)
            assert math.isclose(gm_m, gm_t, abs_tol=0.1)
        # conditionally stable: both negative (lower-limit) and positive
        # (upper-limit) margins exist
        signs = [gm > 0 for _, gm in found]
        assert True in signs and False in signs
        # the single number is the one closest to 0 dB — NOT the first
        expected = min(truth, key=lambda c: abs(c[1]))[1]
        assert math.isclose(gain_margin(sig), expected, abs_tol=0.1)
        first = truth[0][1]
        assert abs(first) > abs(expected)            # first ≠ binding here

    def test_interpolation_beats_sample_snap_at_dec10(self):
        """At `.ac dec 10` density the interpolated margin must stay within
        0.3 dB of analytic truth (the sample-before-the-crossing read was
        off by several dB at this density)."""
        truth = min(self._analytic_crossings(), key=lambda c: abs(c[1]))[1]
        freqs = np.logspace(0, 7, 71)                # 10 pts/decade
        sig = _make_sig(freqs, self._cond_stable(freqs))
        assert math.isclose(gain_margin(sig), truth, abs_tol=0.3)

    def test_deep_crossings_counted(self):
        """A 7-pole loop's phase passes -180° AND -540°: both levels are
        negative-real-axis crossings ('modulo 360°') and
        both must appear."""
        freqs = np.logspace(1, 8, 2000)
        jf = 1j * freqs
        y = 50.0 / (1 + jf / 1e3) ** 7               # phase → -630°
        found = gain_margins(_make_sig(freqs, y))
        assert len(found) == 2                       # -180° and -540°

    def test_phase_anchor_is_modulo_360_by_nature(self):
        """Documents the stated limitation rather than pretending to warn
        about it: a loop whose true phase starts at -200° produces the SAME
        complex data as one at +160° — unwrap must anchor on the principal
        value, so crossings below the sweep are structurally invisible.
        (The research proposal to warn on this case was dead code.)"""
        freqs = np.logspace(1, 3, 50)
        mags = np.linspace(10.0, 1.0, 50)
        below = _ac_sig(freqs, mags, np.linspace(-200, -300, 50))
        above = _ac_sig(freqs, mags, np.linspace(160, 60, 50))
        assert gain_margins(below) == gain_margins(above)

    def test_zero_magnitude_crossing_is_inf_not_log_domain_error(self):
        """TASK-111: the interpolated crossing magnitude mc is exactly 0.0
        when both bracketing samples have zero magnitude -- constructed
        here with signed zeros (0+0j reads angle 0 deg, -0+0j reads angle
        180 deg, both magnitude exactly 0.0, so the phase genuinely crosses
        a -180+k*360 level between them). Undocumented before this fix:
        the code already special-cases mc == 0.0 -> +inf rather than
        -20*log10(0) raising a domain error; +inf is the correct margin at
        that crossing (the loop gain there cannot be raised by any finite
        multiple and lose stability at this specific crossover)."""
        freqs = np.array([10.0, 20.0])
        y = np.array([complex(0.0, 0.0), complex(-0.0, 0.0)])
        sig = _make_sig(freqs, y)
        found = gain_margins(sig)
        assert found == [(20.0, float('inf'))]
