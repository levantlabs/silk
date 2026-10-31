"""Tests for silk.metrics.ac — frequency-response metrics."""
import math
import warnings

import numpy as np
import pytest

from silk.signals.base import Signal
from silk.metrics.ac import (
    passband_ripple_db, stopband_attenuation_db, q_from_peak, group_delay,
    gain_bw_product, psrr_db, cmrr_dB, gain_margin,
)


def _freq(freqs, mag):
    return Signal('h', 'voltage', 'frequency', 'V',
                  np.asarray(freqs, dtype=float),
                  np.asarray(mag, dtype=float))


def _freq_complex(freqs, y):
    return Signal('h', 'voltage', 'frequency', 'V',
                  np.asarray(freqs, dtype=float),
                  np.asarray(y, dtype=complex))


class TestPassbandRippleDb:
    def test_flat(self):
        f = np.logspace(0, 3, 100)
        assert passband_ripple_db(_freq(f, np.ones_like(f)), f_pass=100.0) == 0.0

    def test_known(self):
        sig = _freq([1.0, 10.0, 50.0, 100.0], [1.0, 0.9, 1.1, 1.0])
        expected = 20.0 * math.log10(1.1 / 0.9)
        assert math.isclose(passband_ripple_db(sig, f_pass=100.0),
                            expected, rel_tol=1e-6)

    def test_no_passband(self):
        # TASK-56: an empty band is a malformed request, not a
        # measurement — raises rather than silently reading as 0 dB
        # ripple (a perfect result).
        with pytest.raises(ValueError, match='passband_ripple_db'):
            passband_ripple_db(_freq([100.0, 200.0], [1.0, 0.5]), f_pass=10.0)

    def test_rejects_time(self):
        sig = Signal('s', 'voltage', 'time', 'V',
                     np.array([0.0, 1.0]), np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="xlabel='frequency'"):
            passband_ripple_db(sig, 1.0)


class TestStopbandAttenuationDb:
    def test_known(self):
        sig = _freq([1.0, 10.0, 100.0, 1000.0], [1.0, 1.0, 0.01, 0.001])
        assert math.isclose(stopband_attenuation_db(sig, f_stop=100.0),
                            40.0, abs_tol=1e-6)

    def test_no_stopband(self):
        # TASK-56: an empty stopband is a malformed request, not a
        # measurement — raises rather than silently reading as 0 dB
        # attenuation (a failing-safe but unexplained result).
        with pytest.raises(ValueError, match='stopband_attenuation_db'):
            stopband_attenuation_db(_freq([1.0, 10.0], [1.0, 0.5]), f_stop=100.0)

    def test_zero_stopband_inf(self):
        sig = _freq([1.0, 10.0, 100.0], [1.0, 0.5, 0.0])
        assert stopband_attenuation_db(sig, f_stop=100.0) == float('inf')

    def test_zero_passband_is_unavailable(self):
        """No passband reference exists even when the stopband is nonzero."""
        sig = _freq([1.0, 10.0, 100.0], [0.0, 0.0, 0.5])
        assert math.isnan(stopband_attenuation_db(sig, f_stop=100.0))

    def test_both_peaks_zero_is_nan(self):
        """Without a passband reference, zero/zero cannot measure rejection."""
        sig = _freq([1.0, 10.0, 100.0], [0.0, 0.0, 0.0])
        assert math.isnan(stopband_attenuation_db(sig, f_stop=100.0))


class TestRejectionAndMarginLimits:
    @pytest.mark.parametrize('freq', [None, 10.0])
    def test_cmrr_zero_over_zero_is_nan(self, freq):
        dead = _freq([1.0, 10.0, 100.0], [0.0, 0.0, 0.0])
        assert math.isnan(cmrr_dB(dead, dead, freq=freq))

    def test_zero_common_mode_with_real_differential_gain_is_infinite(self):
        diff = _freq([1.0, 10.0, 100.0], [10.0, 10.0, 10.0])
        cm = _freq([1.0, 10.0, 100.0], [0.0, 0.0, 0.0])
        assert cmrr_dB(diff, cm) == math.inf

    def test_margin_with_no_phase_crossing_and_rolled_off_gain_stays_infinite(self):
        f = np.logspace(0, 6, 100)
        sig = _freq_complex(f, 10 / (1 + 1j * f / 100))
        assert gain_margin(sig) == math.inf


class TestQFromPeak:
    def test_known_q(self):
        f = np.linspace(0.5, 1.5, 100001)
        q_target = 10.0
        m = 1.0 / np.sqrt(1.0 + (q_target ** 2) * (f / 1.0 - 1.0 / f) ** 2)
        assert math.isclose(q_from_peak(_freq(f, m)), 10.0, rel_tol=0.05)

    def test_zero_peak(self):
        f = np.linspace(1, 10, 100)
        assert math.isnan(q_from_peak(_freq(f, np.zeros_like(f))))


class TestGroupDelay:
    def test_constant_delay(self):
        tau = 1e-9
        f = np.linspace(1e6, 1e9, 1000)
        gd = group_delay(_freq_complex(f, np.exp(-2j * np.pi * tau * f)))
        assert isinstance(gd, Signal) and gd.xlabel == 'frequency' and gd.ylabel == 's'
        np.testing.assert_allclose(gd.x, f)
        assert np.allclose(gd.y, tau, rtol=1e-9)     # linear phase: exact

    def test_single_pole_closed_form_at_every_sweep_point(self):
        """FIXED (TASK-99): group_delay differentiates the unwrapped phase
        of the complex H(jw), not Re(H). Pre-fix, ``np.asarray(y,
        dtype=float)`` dropped the imaginary part (ComplexWarning) and the
        result at the pole read 7.96e-4 s for a true 7.96e-6 s — 100x, the
        A0 = 100 gain leaking into a "time" — and 3.2 µs at the sweep
        start for a true 15.9 µs."""
        A0, fp, ppd = 100.0, 1e4, 100
        f = np.logspace(1, 7, 6 * ppd + 1)
        wp = 2 * np.pi * fp
        gd = group_delay(_freq_complex(f, A0 / (1 + 1j * f / fp)))
        tau = (1.0 / wp) / (1 + (f / fp) ** 2)
        r = 10 ** (1.0 / ppd)
        rel = np.abs(gd.y / tau - 1)
        assert rel[1:-1].max() <= (r - 1) ** 2 / r * 1.1     # documented interior bound
        assert rel[0] <= r - 1 and rel[-1] <= r - 1           # documented end bound
        i = int(np.argmin(np.abs(f - fp)))
        assert gd.y[i] == pytest.approx(1 / (2 * wp), rel=1e-3)

    def test_real_y_is_rejected(self):
        """Pre-fix contract was an unwrapped-phase array; it is retired, not
        silently accepted next to the complex one (a phase array read as a
        real H would give an all-zero delay)."""
        f = np.logspace(1, 5, 41)
        with pytest.raises(ValueError, match='real-valued y'):
            group_delay(_freq(f, -2.0 * np.pi * 1e-9 * f))
        with pytest.raises(ValueError, match='real-valued y'):
            group_delay(_freq(f, np.abs(1 / (1 + 1j * f / 1e3))))

    def test_nan_sample_propagates_to_every_output_sample(self):
        f = np.logspace(1, 5, 41)
        y = 1 / (1 + 1j * f / 1e3)
        y[7] = np.nan
        gd = group_delay(_freq_complex(f, y))
        assert np.all(np.isnan(gd.y)) and len(gd.y) == len(f)

    def test_too_few(self):
        with pytest.raises(ValueError, match='at least 2'):
            group_delay(_freq_complex([1.0], [0.0 + 0j]))


class TestImports:
    def test_top_level(self):
        from silk.metrics import passband_ripple_db, q_from_peak, group_delay
        assert callable(passband_ripple_db)


# ---------------------------------------------------------------------------
# Tier 2 — vendor-parity AC additions
# ---------------------------------------------------------------------------

class TestGainBwProduct:
    def test_first_order_rolloff(self):
        # H(f) = A0 / sqrt(1 + (f/f_p)^2). DC gain = A0, BW = f_p.
        # NOTE: this hand-builds the REAL magnitude, not the complex
        # transfer function a simulator returns — it alone is not enough
        # to catch a complex-cast bug (TASK-63 F12); see
        # test_first_order_rolloff_complex_y below for the complex sibling.
        A0 = 100.0
        f_p = 1e6
        f = np.logspace(0, 9, 10000)
        mag = A0 / np.sqrt(1.0 + (f / f_p) ** 2)
        sig = _freq(f, mag)
        gbw = gain_bw_product(sig)
        # GBW ≈ A0 * f_p = 1e8
        assert math.isclose(gbw, A0 * f_p, rel_tol=0.05)

    def test_first_order_rolloff_complex_y(self):
        # H(f) = A0 / (1 + j*f/f_p) — the complex form a simulator's .ac
        # sweep actually loads as (Signal.y is complex128). Before TASK-63
        # this was cast to float first, so the search ran on |Re H| and
        # read the bandwidth 36.9% low (0.6436*f_p instead of f_p).
        A0 = 100.0
        f_p = 1e6
        f = np.logspace(0, 9, 10000)
        y = A0 / (1.0 + 1j * f / f_p)
        sig = _freq_complex(f, y)
        with warnings.catch_warnings():
            warnings.simplefilter('error', np.exceptions.ComplexWarning)
            gbw = gain_bw_product(sig)
        assert math.isclose(gbw, A0 * f_p, rel_tol=0.05)
        # real-magnitude input must give the same answer
        real_sig = _freq(f, np.abs(y))
        assert math.isclose(gbw, gain_bw_product(real_sig), rel_tol=1e-9)

    def test_zero_dc_gain_returns_nan(self):
        f = np.logspace(0, 6, 100)
        sig = _freq(f, np.zeros_like(f))
        assert math.isnan(gain_bw_product(sig))

    def test_rejects_time_signal(self):
        sig = Signal('s', 'voltage', 'time', 'V',
                     np.array([0.0, 1.0]), np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="xlabel='frequency'"):
            gain_bw_product(sig)


class TestTier2AcImports:
    def test_top_level(self):
        from silk.metrics import gain_bw_product
        assert callable(gain_bw_product)


# ---------------------------------------------------------------------------
# Beyond-vendor — PSRR
# ---------------------------------------------------------------------------

class TestPsrrDb:
    def test_known_60db(self):
        # Supply at 1 V noise; output sees 1 mV → 60 dB PSRR
        f = np.array([1.0, 10.0, 100.0])
        v_supply = _freq(f, [1.0, 1.0, 1.0])
        v_out    = _freq(f, [1e-3, 1e-3, 1e-3])
        result = psrr_db(v_supply, v_out)
        assert result.xlabel == 'frequency'
        assert result.ylabel == 'dB'
        assert np.allclose(result.y, 60.0, rtol=1e-9)

    def test_frequency_dependent(self):
        # Output gets noisier at high freq → PSRR rolls off
        f = np.array([1.0, 100.0])
        v_supply = _freq(f, [1.0, 1.0])
        v_out    = _freq(f, [1e-3, 1e-1])
        result = psrr_db(v_supply, v_out)
        assert math.isclose(float(result.y[0]),  60.0, rel_tol=1e-9)
        assert math.isclose(float(result.y[1]),  20.0, rel_tol=1e-9)

    def test_zero_output_floors_at_eps_not_tiny(self):
        # A perfectly rejected point (|V_out| == 0) is degenerate DATA
        # (decision-1 SS1), not a request error, so this must still
        # return a number. Before 2026-09-05 the zero was floored at
        # np.finfo(float).tiny, so a 1 V supply / 0 V output point read
        # 20*log10(1 / 2.225e-308) ~= +6153.05 dB -- a floor artifact
        # presented as a measurement, with no bound stated anywhere.
        # The fix floors |V_out| at eps * |V_supply| instead, capping
        # the reportable value at 20*log10(1/eps) ~= +313.07 dB
        # (TASK-105).
        f = np.array([1.0, 10.0])
        v_supply = _freq(f, [1.0, 1.0])
        v_out    = _freq(f, [0.0, 1e-3])
        result = psrr_db(v_supply, v_out)
        expected = 20.0 * math.log10(1.0 / np.finfo(float).eps)
        assert math.isclose(float(result.y[0]), expected, rel_tol=1e-12)
        assert math.isclose(float(result.y[0]), 313.07, rel_tol=1e-4)
        pre_fix_value = 20.0 * math.log10(1.0 / np.finfo(float).tiny)
        assert math.isclose(pre_fix_value, 6153.05, rel_tol=1e-4)
        assert result.y[0] < pre_fix_value
        # unaffected point still reads the ordinary 60 dB
        assert math.isclose(float(result.y[1]), 60.0, rel_tol=1e-9)

    def test_zero_supply_and_output_is_nan(self):
        # No supply ripple was injected at this point (|V_supply| == 0
        # too), so the floor itself is zero and the ratio is 0/0: PSRR
        # is genuinely unknowable here, not measurable-and-zero
        # (decision-1 SS1's nan branch), so this must NOT silently read
        # as some finite value.
        f = np.array([1.0, 10.0])
        v_supply = _freq(f, [0.0, 1.0])
        v_out    = _freq(f, [0.0, 1e-3])
        result = psrr_db(v_supply, v_out)
        assert math.isnan(result.y[0])
        # unaffected point still reads the ordinary 60 dB
        assert math.isclose(float(result.y[1]), 60.0, rel_tol=1e-9)

    def test_mismatched_axes(self):
        a = _freq([1.0, 10.0], [1.0, 1.0])
        b = _freq([1.0, 20.0], [1.0, 1.0])
        with pytest.raises(ValueError, match='same frequency axis'):
            psrr_db(a, b)

    def test_rejects_time_signal(self):
        sig_t = Signal('s', 'voltage', 'time', 'V',
                       np.array([0.0, 1.0]), np.array([1.0, 1.0]))
        sig_f = _freq([1.0, 10.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="xlabel='frequency'"):
            psrr_db(sig_t, sig_f)


class TestBeyondVendorAcImports:
    def test_top_level(self):
        from silk.metrics import psrr_db
        assert callable(psrr_db)


@pytest.mark.parametrize('metric_name', [
    'phase_margin', 'phase_at_freq', 'gain_margin', 'gain_margins',
])
@pytest.mark.parametrize('signed', [False, True])
def test_real_phase_input_warns_once_and_preserves_result(metric_name, signed):
    """Losing complex phase must be diagnosed without rejecting real-axis data."""
    from silk.metrics import ac
    from silk.metrics.diagnostics import MetricWarning

    f = np.logspace(0, 8, 801)
    h = 1e3 / ((1 + 1j*f/1e3) * (1 + 1j*f/1e6) * (1 + 1j*f/2e6))
    y = -np.abs(h) if signed else np.abs(h)
    sig = _freq(f, y)
    args = (1e6,) if metric_name == 'phase_at_freq' else ()
    with pytest.warns(MetricWarning, match='no phase') as caught:
        result = getattr(ac, metric_name)(sig, *args)
    assert len(caught) == 1
    warning = caught[0].message
    assert warning.metric == metric_name
    assert 'complex' in warning.suggestion
    expected = {
        'phase_margin': 0.0 if signed else -180.0,
        'phase_at_freq': 180.0 if signed else 0.0,
        'gain_margin': float('inf'),
        'gain_margins': [],
    }
    assert result == expected[metric_name]


def test_three_pole_complex_phase_metrics_unchanged():
    """Complex loop measurements retain phase and emit no missing-phase warning."""
    from silk.metrics.ac import phase_margin, phase_at_freq, gain_margins

    f = np.logspace(0, 8, 801)
    h = 1e3 / ((1 + 1j*f/1e3) * (1 + 1j*f/1e6) * (1 + 1j*f/2e6))
    sig = _freq_complex(f, h)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert phase_margin(sig) == pytest.approx(33.19, abs=0.01)
        assert phase_at_freq(sig, 1e6) == pytest.approx(-161.5, abs=0.1)
        assert gain_margin(sig) == pytest.approx(9.56, abs=0.01)
        crossings = gain_margins(sig)
    assert len(crossings) == 1
    assert crossings[0][0] == pytest.approx(1.415e6, rel=0.001)
    assert crossings[0][1] == pytest.approx(9.56, abs=0.01)
    assert not caught
