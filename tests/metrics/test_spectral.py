"""Tests for silk.metrics.spectral — fft, sndr/sinad/sfdr/enob/thd,
noise floor, harmonic levels, IMD3, plus the shared
sample-extraction chokepoint.
"""
import math
import warnings
import numpy as np
import pytest

from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal
from silk.metrics.spectral import (
    fft,
    sndr,
    sinad,
    sfdr,
    thd,
    thd_percent,
    noise_floor_dbfs,
    harmonic_levels_db,
    SFDR,
    _sample_at_clock,
    psd,
    spectral_power,
    harmonic,
    snr,
    harmonic_freqs,
    welch_psd,
    coherence,
)
from silk.metrics.converter import (
    enob,
)
from silk.metrics.rf import (
    imd3_db,
    mix_product,
    evm_constellation,
    EVMResult,
)


TCLK = 1e-9
TSTART = 0.0
N = 1024
M_COH = 11


def _coherent_sine_signal(n=N, m=M_COH, amp=1.0, tclk=TCLK, tstart=TSTART):
    t = tstart + np.arange(n) * tclk
    y = amp * np.sin(2.0 * np.pi * m * np.arange(n) / n)
    return Signal('vh', 'voltage', 'time', 'V', t, y)


def _trans(y):
    t = np.arange(len(y)) * TCLK
    return Signal('s', 'voltage', 'time', 'V', t, np.asarray(y, dtype=float))


# ---------------------------------------------------------------------------
# Sample-extraction chokepoint
# ---------------------------------------------------------------------------

class TestSampleAtClock:
    def test_basic(self):
        sig = _coherent_sine_signal()
        s = _sample_at_clock(sig, metric='probe', tclk=TCLK, tstart=0.0, n_samples=N)
        assert len(s) == N
        assert math.isclose(s[0], 0.0, abs_tol=1e-12)

    def test_auto_n_samples(self):
        sig = _coherent_sine_signal()
        s = _sample_at_clock(sig, metric='probe', tclk=TCLK, tstart=0.0)
        assert len(s) == N

    def test_tstart_offset(self):
        sig = _coherent_sine_signal()
        s = _sample_at_clock(sig, metric='probe', tclk=TCLK, tstart=10 * TCLK, n_samples=8)
        assert math.isclose(s[0], sig.y[10], rel_tol=1e-12)

    def test_rejects_non_transient(self):
        sig = Signal('s', 'voltage', 'frequency', 'V',
                     np.array([1.0, 10.0]), np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="xlabel='time'"):
            _sample_at_clock(sig, metric='probe', tclk=TCLK)

    def test_overflow_raises(self):
        sig = _coherent_sine_signal()
        with pytest.raises(ValueError, match='exceeds trace end'):
            _sample_at_clock(sig, metric='probe', tclk=TCLK, n_samples=N + 100)

    def test_tstart_before(self):
        sig = _coherent_sine_signal()
        with pytest.raises(ValueError, match='before the start'):
            _sample_at_clock(sig, metric='probe', tclk=TCLK, tstart=-1e-6)

    def test_invalid_tclk(self):
        sig = _coherent_sine_signal()
        with pytest.raises(ValueError, match='tclk must be > 0'):
            _sample_at_clock(sig, metric='probe', tclk=-1e-9)

    def test_too_few_samples(self):
        sig = _coherent_sine_signal()
        with pytest.raises(ValueError, match='must be >= 4'):
            _sample_at_clock(sig, metric='probe', tclk=TCLK, n_samples=2)


# ---------------------------------------------------------------------------
# fft primitive
# ---------------------------------------------------------------------------

class TestFft:
    def test_returns_signal(self):
        spec = fft(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        assert isinstance(spec, Signal)
        assert spec.xlabel == 'frequency'
        assert len(spec.x) == N // 2 + 1

    def test_freq_axis(self):
        spec = fft(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        assert math.isclose(spec.x[1], 1.0 / (N * TCLK), rel_tol=1e-12)
        assert math.isclose(spec.x[-1], 0.5 / TCLK, rel_tol=1e-12)

    def test_fundamental_at_expected_bin(self):
        spec = fft(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        peak_bin = int(np.argmax(spec.y[1:]) + 1)
        assert peak_bin == M_COH

    def test_plottable(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        spec = fft(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        fig, ax = plt.subplots()
        spec.plot(ax=ax)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------

class TestSndr:
    def test_clean_sine(self):
        v = sndr(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        assert math.isfinite(v)
        assert v == pytest.approx(186.65, abs=0.01)

    def test_white_noise_drops(self):
        rng = np.random.default_rng(42)
        sig = _coherent_sine_signal()
        sig.y = sig.y + 0.01 * rng.standard_normal(N)
        v = sndr(sig, tclk=TCLK, n_samples=N)
        assert 30.0 < v < 60.0

    def test_rejects_freq(self):
        sig = Signal('ac', 'voltage', 'frequency', 'V',
                     np.array([1, 10, 100], dtype=float),
                     np.array([1, 1, 1], dtype=float))
        with pytest.raises(ValueError, match="xlabel='time'"):
            sndr(sig, tclk=TCLK)


class TestSpectralPrecisionFloor:
    """Residual subtraction loses low-power bins beside a strong tone."""

    def _signal(self, case):
        t = np.arange(0, 256e-9, 1e-10)
        y = 0.5 + 0.4 * np.sin(2 * np.pi * 7 / 256e-9 * t)
        if case == 'dead':
            y = np.full_like(t, 0.5)
        elif case == 'noisy':
            y += 1e-3 * np.random.default_rng(7).standard_normal(len(t))
        elif case == 'distorted':
            y += 0.01 * np.sin(2 * np.pi * 21 / 256e-9 * t)
        return Signal('v', 'voltage', 'time', 'V', t, y)

    @pytest.mark.parametrize('metric, expected', [
        (sndr, 172.69), (sinad, 172.69), (snr, 173.01),
        (enob, 28.39), (thd, -187.71), (noise_floor_dbfs, -193.73),
    ])
    def test_reproducer(self, metric, expected):
        sig = self._signal('near-perfect')
        value = metric(sig, tclk=TCLK, tstart=0.0, n_samples=256)
        assert math.isfinite(value)
        assert value == pytest.approx(expected, abs=0.01)

    @pytest.mark.parametrize('case', ['dead', 'near-perfect', 'noisy', 'distorted'])
    def test_spectral_ratios_are_finite(self, case):
        sig = self._signal(case)
        kw = dict(tclk=TCLK, tstart=0.0, n_samples=256)
        for metric in (sndr, sinad, snr, enob, thd, thd_percent,
                       noise_floor_dbfs, harmonic_levels_db):
            assert np.all(np.isfinite(metric(sig, **kw))), metric.__name__
        result = sfdr(sig, **kw)
        assert np.all(np.isfinite([
            result.sfdr_db, result.fundamental_hz, result.spur_hz, result.bin_hz,
        ]))

    @pytest.mark.parametrize('case', ['dead', 'near-perfect', 'noisy', 'distorted'])
    def test_residual_sum_is_at_least_the_largest_spur(self, case):
        sig = self._signal(case)
        kw = dict(tclk=TCLK, tstart=0.0, n_samples=256)
        assert sndr(sig, **kw) <= sfdr(sig, **kw).sfdr_db

    def test_reproducer_spur_and_harmonics_share_the_floor(self):
        sig = self._signal('near-perfect')
        kw = dict(tclk=TCLK, tstart=0.0, n_samples=256)
        spur = sfdr(sig, **kw)
        assert spur.sfdr_db == pytest.approx(193.73, abs=0.01)
        # 127 residual bins and four harmonic bins each sit at eps.
        assert sndr(sig, **kw) == pytest.approx(
            spur.sfdr_db - 10 * math.log10(127), abs=1e-9)
        assert thd(sig, **kw) == pytest.approx(
            -spur.sfdr_db + 10 * math.log10(4), abs=1e-9)

    @pytest.mark.parametrize('metric', [sndr, sinad, snr, enob, sfdr])
    @pytest.mark.parametrize('amp', [1.0, 1e150])
    def test_exact_tone_with_no_noise_bins(self, metric, amp):
        # Exact FFT: P[1] = 8*amp**2; the only other non-DC bin is H2.
        # Excluding H2 leaves SNR no noise bins. At large amplitude the
        # powers remain finite but their quotient would overflow.
        sig = Signal('v', 'voltage', 'time', 'V', np.arange(4),
                     amp * np.array([1., 0., -1., 0.]))
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            result = metric(sig, tclk=1.0, n_samples=4)
        value = result.sfdr_db if isinstance(result, SFDR) else result
        if metric is snr:
            assert math.isnan(value)
            return
        expected = 165.56649761519 + 20 * math.log10(amp)
        if metric is enob:
            expected = (expected - 1.76) / 6.02
        assert math.isfinite(value)
        assert value == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize('metric', [sndr, sinad, snr, enob, sfdr])
    @pytest.mark.parametrize('amp', [1e154, 1e200])
    def test_unrepresentable_power_is_nan_not_infinite_quality(self, metric, amp):
        sig = Signal('v', 'voltage', 'time', 'V', np.arange(4),
                     amp * np.array([1., 0., -1., 0.]))
        with np.errstate(over='ignore', divide='raise', invalid='raise'):
            result = metric(sig, tclk=1.0, n_samples=4)
        if isinstance(result, SFDR):
            assert math.isnan(result.sfdr_db)
            assert math.isnan(result.fundamental_hz)
            assert math.isnan(result.spur_hz)
            assert result.bin_hz == 0.25
        else:
            assert math.isnan(result)


class TestSinad:
    def test_matches_sndr(self):
        rng = np.random.default_rng(0)
        sig = _coherent_sine_signal()
        sig.y = sig.y + 0.01 * rng.standard_normal(N)
        a = sinad(sig, tclk=TCLK, n_samples=N)
        b = sndr(sig,  tclk=TCLK, n_samples=N)
        assert math.isclose(a, b, rel_tol=1e-12)


class TestSfdr:
    def test_clean(self):
        result = sfdr(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        assert isinstance(result, SFDR)
        assert result.sfdr_db > 200.0
        assert math.isclose(result.fundamental_hz,
                            M_COH / (N * TCLK), rel_tol=1e-9)

    def test_with_h2(self):
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 11 * n_arr / N)
             + 0.01 * np.sin(2 * np.pi * 22 * n_arr / N))
        result = sfdr(_trans(y), tclk=TCLK, n_samples=N)
        assert math.isclose(result.sfdr_db, 40.0, abs_tol=0.5)


class TestEnob:
    def test_matches_formula(self):
        rng = np.random.default_rng(3)
        sig = _coherent_sine_signal()
        sig.y = sig.y + 0.001 * rng.standard_normal(N)
        s = sndr(sig, tclk=TCLK, n_samples=N)
        e = enob(sig, tclk=TCLK, n_samples=N)
        assert math.isclose(e, (s - 1.76) / 6.02, rel_tol=1e-12)

    def test_perfect_huge(self):
        e = enob(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        assert e > 30.0


class TestThd:
    def test_clean_sine(self):
        v = thd(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        assert v == float('-inf') or v < -200.0

    def test_with_h2(self):
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 11 * n_arr / N)
             + 0.01 * np.sin(2 * np.pi * 22 * n_arr / N))
        v = thd(_trans(y), tclk=TCLK, n_samples=N, n_harmonics=2)
        assert math.isclose(v, -40.0, abs_tol=0.5)

    def test_invalid_n_harmonics(self):
        with pytest.raises(ValueError, match='>= 2'):
            thd(_coherent_sine_signal(), tclk=TCLK, n_samples=N, n_harmonics=1)


class TestThdPercent:
    def test_clean_sine(self):
        assert thd_percent(_coherent_sine_signal(), tclk=TCLK, n_samples=N) < 1e-6

    def test_with_h2(self):
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 11 * n_arr / N)
             + 0.01 * np.sin(2 * np.pi * 22 * n_arr / N))
        v = thd_percent(_trans(y), tclk=TCLK, n_samples=N, n_harmonics=2)
        assert math.isclose(v, 1.0, abs_tol=0.05)


class TestNoiseFloorDbfs:
    def test_negative_with_noise(self):
        rng = np.random.default_rng(7)
        sig = _coherent_sine_signal()
        sig.y = sig.y + 0.01 * rng.standard_normal(N)
        assert noise_floor_dbfs(sig, tclk=TCLK, n_samples=N) < 0


class TestHarmonicLevelsDb:
    def test_known_h2_h3(self):
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 11 * n_arr / N)
             + 0.01  * np.sin(2 * np.pi * 22 * n_arr / N)
             + 0.001 * np.sin(2 * np.pi * 33 * n_arr / N))
        levels = harmonic_levels_db(_trans(y), tclk=TCLK, n_samples=N,
                                    n_harmonics=4)
        assert math.isclose(levels[0], -40.0, abs_tol=0.5)
        assert math.isclose(levels[1], -60.0, abs_tol=0.5)


class TestImd3Db:
    def test_two_tone_with_imd(self):
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 100 * n_arr / N)
             + np.sin(2 * np.pi * 110 * n_arr / N)
             + 0.01 * np.sin(2 * np.pi * 90 * n_arr / N))
        f1_hz = 100 / (N * TCLK)
        f2_hz = 110 / (N * TCLK)
        v = imd3_db(_trans(y), tclk=TCLK, f1_hz=f1_hz, f2_hz=f2_hz, n_samples=N)
        assert math.isclose(v, -40.0, abs_tol=0.5)

    def test_no_imd(self):
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 100 * n_arr / N)
             + np.sin(2 * np.pi * 110 * n_arr / N))
        f1_hz = 100 / (N * TCLK)
        f2_hz = 110 / (N * TCLK)
        v = imd3_db(_trans(y), tclk=TCLK, f1_hz=f1_hz, f2_hz=f2_hz, n_samples=N)
        assert v == float('-inf') or v < -100.0

    def test_invalid_freqs(self):
        with pytest.raises(ValueError, match='> 0 and distinct'):
            imd3_db(_coherent_sine_signal(), tclk=TCLK,
                    f1_hz=1.0, f2_hz=1.0, n_samples=N)

    def test_products_aliased_below_dc_and_above_nyquist_both_fold_in_band(self):
        # S1 (TASK-311): 64 samples, tones at bins 10 and 25. The products
        # 2*10-25=-5 (below DC) and 2*25-10=40 (above Nyquist) both alias
        # into the sampled record, at bins 5 and 24 respectively. Put the
        # IMD3 energy at the lower product's aliased bin (5); imd3_db must
        # read it there, not return -inf by dropping it.
        n = 64
        n_arr = np.arange(n)
        a = 0.01
        y = (np.sin(2 * np.pi * 10 * n_arr / n)
             + np.sin(2 * np.pi * 25 * n_arr / n)
             + a * np.sin(2 * np.pi * 5 * n_arr / n))
        f1_hz = 10 / (n * TCLK)
        f2_hz = 25 / (n * TCLK)
        v = imd3_db(_trans(y), tclk=TCLK, f1_hz=f1_hz, f2_hz=f2_hz, n_samples=n)
        assert math.isfinite(v)
        assert math.isclose(v, 20 * math.log10(a), abs_tol=0.5)

    def test_below_dc_product_alone_is_read_at_its_folded_bin(self):
        # f1_bin=8, f2_bin=20: only the lower product (2*8-20=-4) aliases
        # below DC, folding to bin 4; the upper (2*20-8=32) is already
        # in-band (the Nyquist bin itself, unaliased).
        n = 64
        n_arr = np.arange(n)
        a = 0.02
        y = (np.sin(2 * np.pi * 8 * n_arr / n)
             + np.sin(2 * np.pi * 20 * n_arr / n)
             + a * np.sin(2 * np.pi * 4 * n_arr / n))
        f1_hz = 8 / (n * TCLK)
        f2_hz = 20 / (n * TCLK)
        v = imd3_db(_trans(y), tclk=TCLK, f1_hz=f1_hz, f2_hz=f2_hz, n_samples=n)
        assert math.isclose(v, 20 * math.log10(a), abs_tol=0.5)

    def test_above_nyquist_product_alone_is_read_at_its_folded_bin(self):
        # f1_bin=15, f2_bin=24: only the upper product (2*24-15=33)
        # aliases above Nyquist, folding to bin 31; the lower
        # (2*15-24=6) is already in-band, unaliased.
        n = 64
        n_arr = np.arange(n)
        a = 0.02
        y = (np.sin(2 * np.pi * 15 * n_arr / n)
             + np.sin(2 * np.pi * 24 * n_arr / n)
             + a * np.sin(2 * np.pi * 31 * n_arr / n))
        f1_hz = 15 / (n * TCLK)
        f2_hz = 24 / (n * TCLK)
        v = imd3_db(_trans(y), tclk=TCLK, f1_hz=f1_hz, f2_hz=f2_hz, n_samples=n)
        assert math.isclose(v, 20 * math.log10(a), abs_tol=0.5)

    def test_product_folding_onto_dc_returns_nan_with_warning(self):
        # f1_bin=8, f2_bin=16: the lower product 2*8-16=0 sits exactly at
        # DC -- inseparable from DC regardless of the data.
        n = 64
        n_arr = np.arange(n)
        y = (np.sin(2 * np.pi * 8 * n_arr / n)
             + np.sin(2 * np.pi * 16 * n_arr / n))
        f1_hz = 8 / (n * TCLK)
        f2_hz = 16 / (n * TCLK)
        with pytest.warns(MetricWarning) as caught:
            v = imd3_db(_trans(y), tclk=TCLK, f1_hz=f1_hz, f2_hz=f2_hz,
                        n_samples=n)
        assert math.isnan(v)
        warning = caught[-1].message
        assert warning.code == 'unattributable_imd_product'
        assert warning.details['f1_bin'] == 8 and warning.details['f2_bin'] == 16
        assert warning.details['lower_product_bin_raw'] == 0
        assert warning.details['lower_product_bin_folded'] == 0

    def test_product_folding_onto_a_tone_bin_returns_nan_with_warning(self):
        # f1_bin=2, f2_bin=22: the upper product 2*22-2=42 aliases (folds
        # to bin 22) onto f2 itself -- inseparable from that tone.
        n = 64
        n_arr = np.arange(n)
        y = (np.sin(2 * np.pi * 2 * n_arr / n)
             + np.sin(2 * np.pi * 22 * n_arr / n))
        f1_hz = 2 / (n * TCLK)
        f2_hz = 22 / (n * TCLK)
        with pytest.warns(MetricWarning) as caught:
            v = imd3_db(_trans(y), tclk=TCLK, f1_hz=f1_hz, f2_hz=f2_hz,
                        n_samples=n)
        assert math.isnan(v)
        warning = caught[-1].message
        assert warning.code == 'unattributable_imd_product'
        assert warning.details['upper_product_bin_raw'] == 42
        assert warning.details['upper_product_bin_folded'] == 22

    def test_in_band_two_tone_case_is_unchanged_by_the_fold(self):
        # Regression for AC#4: an already-in-band case (no aliasing, no
        # collision) must read identically to before TASK-311.
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 100 * n_arr / N)
             + np.sin(2 * np.pi * 110 * n_arr / N)
             + 0.01 * np.sin(2 * np.pi * 90 * n_arr / N))
        f1_hz = 100 / (N * TCLK)
        f2_hz = 110 / (N * TCLK)
        v = imd3_db(_trans(y), tclk=TCLK, f1_hz=f1_hz, f2_hz=f2_hz, n_samples=N)
        assert math.isclose(v, -40.0, abs_tol=0.5)


class TestImports:
    def test_top_level(self):
        from silk.metrics import sndr, sfdr, enob, fft, thd_percent, harmonic_levels_db
        assert callable(fft)


# ---------------------------------------------------------------------------
# Tier 2 — vendor-parity spectral additions
# ---------------------------------------------------------------------------

class TestPsd:
    def test_returns_signal_in_v2_per_hz(self):
        spec = psd(_coherent_sine_signal(), tclk=TCLK, n_samples=N)
        assert isinstance(spec, Signal)
        assert spec.xlabel == 'frequency'
        assert len(spec.x) == N // 2 + 1
        # All values non-negative
        assert np.all(spec.y >= 0)

    def test_window_options(self):
        sig = _coherent_sine_signal()
        for w in ('hann', 'hamming', 'blackman', 'rectangular', 'rectangular', None):
            p = psd(sig, tclk=TCLK, n_samples=N, window=w)
            assert p.x.shape == p.y.shape

    def test_invalid_window(self):
        with pytest.raises(ValueError, match='window must be'):
            psd(_coherent_sine_signal(), tclk=TCLK, n_samples=N, window='kaiser')


class TestPsdBasebandRemoved:
    def test_psd_baseband_is_gone(self):
        import silk.metrics
        import silk.metrics.spectral
        assert not hasattr(silk.metrics, 'psd_baseband')
        assert not hasattr(silk.metrics.spectral, 'psd_baseband')
        assert 'psd_baseband' not in silk.metrics.__all__


class TestSpectralPower:
    def test_band_around_fundamental(self):
        sig = _coherent_sine_signal()
        f_fund = M_COH / (N * TCLK)
        df = 1.0 / (N * TCLK)
        # Narrow band around fundamental should capture most signal power
        p_band = spectral_power(sig, f_fund - 5 * df, f_fund + 5 * df,
                                tclk=TCLK, n_samples=N, window='rectangular')
        assert p_band > 0.0

    def test_invalid_band(self):
        with pytest.raises(ValueError, match='f2_hz > f1_hz'):
            spectral_power(_coherent_sine_signal(), 100, 50,
                           tclk=TCLK, n_samples=N)


class TestHarmonic:
    def test_fundamental_complex(self):
        sig = _coherent_sine_signal()
        h1 = harmonic(sig, 1, tclk=TCLK, n_samples=N)
        # |h1| should be the largest non-DC bin
        assert isinstance(h1, complex)
        assert abs(h1) > 0.0

    def test_second_harmonic_at_known_level(self):
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 11 * n_arr / N)
             + 0.01 * np.sin(2 * np.pi * 22 * n_arr / N))
        sig = _trans(y)
        h1 = harmonic(sig, 1, tclk=TCLK, n_samples=N)
        h2 = harmonic(sig, 2, tclk=TCLK, n_samples=N)
        # h2 / h1 ≈ 0.01 in magnitude
        assert math.isclose(abs(h2) / abs(h1), 0.01, rel_tol=0.1)

    def test_invalid_k(self):
        with pytest.raises(ValueError, match='k must be >= 1'):
            harmonic(_coherent_sine_signal(), 0, tclk=TCLK, n_samples=N)

    def test_above_nyquist_is_read_at_the_folded_bin(self):
        # M_COH=11 in N=1024 → Nyquist bin = 512. k=100 sits at nominal
        # bin 1100, which sampling folds to (1100 mod 1024) = 76 — an
        # empty bin on a pure tone. TASK-79: the folded bin is READ
        # (≈0, the honest "no energy there"), no longer reported as nan
        # — nan is reserved for a harmonic that folds onto DC or onto
        # the fundamental (unattributable), matching harmonic_levels_db.
        sig = _coherent_sine_signal()
        h = harmonic(sig, 100, tclk=TCLK, n_samples=N)
        assert not math.isnan(h.real)
        assert abs(h) < 1e-9 * abs(harmonic(sig, 1, tclk=TCLK, n_samples=N))


class TestMixProduct:
    def test_imd3_lower(self):
        # Two tones at bins 100 and 110; inject IMD3 at 2·100 - 110 = 90.
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 100 * n_arr / N)
             + np.sin(2 * np.pi * 110 * n_arr / N)
             + 0.01 * np.sin(2 * np.pi * 90 * n_arr / N))
        sig = _trans(y)
        f1 = 100 / (N * TCLK)
        f2 = 110 / (N * TCLK)
        # Lower IMD3 = 2f1 - f2
        imd_lower = mix_product(sig, 2, -1, f1_hz=f1, f2_hz=f2,
                                tclk=TCLK, n_samples=N)
        fund = mix_product(sig, 1, 0, f1_hz=f1, f2_hz=f2,
                           tclk=TCLK, n_samples=N)
        assert math.isclose(abs(imd_lower) / abs(fund), 0.01, rel_tol=0.1)


class TestSnr:
    def test_excludes_harmonics(self):
        # Build a signal with both noise and harmonics; SNR (excluding
        # harmonics) should be HIGHER than SNDR (which includes them).
        rng = np.random.default_rng(99)
        n_arr = np.arange(N)
        y = (np.sin(2 * np.pi * 11 * n_arr / N)
             + 0.05 * np.sin(2 * np.pi * 22 * n_arr / N)   # h2 distortion
             + 0.005 * rng.standard_normal(N))             # white noise
        sig = _trans(y)
        snr_db  = snr(sig, tclk=TCLK, n_samples=N, n_harmonics=10)
        sndr_db = sndr(sig, tclk=TCLK, n_samples=N)
        assert snr_db > sndr_db

    def test_invalid_n_harmonics(self):
        with pytest.raises(ValueError, match='>= 2'):
            snr(_coherent_sine_signal(), tclk=TCLK, n_samples=N, n_harmonics=1)


class TestTier2Imports:
    def test_top_level(self):
        from silk.metrics import psd, spectral_power, harmonic, mix_product, snr
        assert callable(psd)
        assert callable(snr)


# ---------------------------------------------------------------------------
# Tier 3 — vendor-parity additions
# ---------------------------------------------------------------------------

QPSK_IDEAL = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex) / math.sqrt(2.0)


class TestEvmConstellation:
    def test_perfect_qpsk_zero_evm(self):
        # Build a clean QPSK transmission: 8 symbols, each held flat
        # for one symbol_period.
        symbol_period = 1e-6
        n_sym = 8
        rng = np.random.default_rng(0)
        choices = rng.integers(0, 4, size=n_sym)
        # Time grid: 100 samples per symbol
        n_samples_per_sym = 100
        t = np.linspace(0, n_sym * symbol_period,
                        n_sym * n_samples_per_sym, endpoint=False)
        i_y = np.empty_like(t)
        q_y = np.empty_like(t)
        for k in range(n_sym):
            sym = QPSK_IDEAL[choices[k]]
            sl = slice(k * n_samples_per_sym, (k + 1) * n_samples_per_sym)
            i_y[sl] = sym.real
            q_y[sl] = sym.imag
        i_sig = Signal('i', 'voltage', 'time', 'V', t, i_y)
        q_sig = Signal('q', 'voltage', 'time', 'V', t, q_y)
        result = evm_constellation(
            i_sig, q_sig, symbol_period=symbol_period,
            ideal_points=QPSK_IDEAL, n_symbols=n_sym,
        )
        assert isinstance(result, EVMResult)
        assert result.evm_percent < 0.01
        assert result.received.shape == (n_sym,)

    def test_evm_with_added_noise(self):
        symbol_period = 1e-6
        n_sym = 64
        rng = np.random.default_rng(42)
        choices = rng.integers(0, 4, size=n_sym)
        n_samples_per_sym = 50
        t = np.linspace(0, n_sym * symbol_period,
                        n_sym * n_samples_per_sym, endpoint=False)
        i_y = np.empty_like(t)
        q_y = np.empty_like(t)
        for k in range(n_sym):
            sym = QPSK_IDEAL[choices[k]]
            sl = slice(k * n_samples_per_sym, (k + 1) * n_samples_per_sym)
            i_y[sl] = sym.real
            q_y[sl] = sym.imag
        # Add 1% RMS noise on each axis
        i_y = i_y + 0.01 * rng.standard_normal(len(t))
        q_y = q_y + 0.01 * rng.standard_normal(len(t))
        i_sig = Signal('i', 'voltage', 'time', 'V', t, i_y)
        q_sig = Signal('q', 'voltage', 'time', 'V', t, q_y)
        result = evm_constellation(
            i_sig, q_sig, symbol_period=symbol_period,
            ideal_points=QPSK_IDEAL, n_symbols=n_sym,
        )
        # Noise of σ=0.01 on real and imag → |error|² ~ 2·σ² = 2e-4
        # rms_err ≈ √2·σ ≈ 0.014; rms_ref = 1; EVM ≈ 1.4%
        assert 0.5 < result.evm_percent < 3.0

    def test_invalid_period(self):
        t = np.linspace(0, 1e-6, 100)
        i_sig = Signal('i', 'voltage', 'time', 'V', t, np.zeros_like(t))
        q_sig = Signal('q', 'voltage', 'time', 'V', t, np.zeros_like(t))
        with pytest.raises(ValueError, match='symbol_period must be > 0'):
            evm_constellation(i_sig, q_sig, symbol_period=0,
                              ideal_points=QPSK_IDEAL)

    def test_mismatched_axes(self):
        t1 = np.linspace(0, 1e-6, 100)
        t2 = np.linspace(0, 2e-6, 100)
        i_sig = Signal('i', 'voltage', 'time', 'V', t1, np.zeros_like(t1))
        q_sig = Signal('q', 'voltage', 'time', 'V', t2, np.zeros_like(t2))
        with pytest.raises(ValueError, match='same time axis'):
            evm_constellation(i_sig, q_sig, symbol_period=1e-7,
                              ideal_points=QPSK_IDEAL)

    def test_empty_ideals(self):
        t = np.linspace(0, 1e-6, 100)
        i_sig = Signal('i', 'voltage', 'time', 'V', t, np.zeros_like(t))
        q_sig = Signal('q', 'voltage', 'time', 'V', t, np.zeros_like(t))
        with pytest.raises(ValueError, match='ideal_points must not be empty'):
            evm_constellation(i_sig, q_sig, symbol_period=1e-7,
                              ideal_points=[])


class TestEvmConstellationReference:
    """TASK-227: reference= scores each symbol against its own known
    transmitted point instead of the nearest ideal point, so a wrong
    decision costs the full error instead of being hidden by re-scoring
    to the wrong reference (TASK-187.11 finding 3).
    """

    def _qpsk_trace(self, choices, symbol_period=1e-6, n_samples_per_sym=50):
        n_sym = len(choices)
        t = np.linspace(0, n_sym * symbol_period,
                        n_sym * n_samples_per_sym, endpoint=False)
        i_y = np.empty_like(t)
        q_y = np.empty_like(t)
        for k in range(n_sym):
            sym = QPSK_IDEAL[choices[k]]
            sl = slice(k * n_samples_per_sym, (k + 1) * n_samples_per_sym)
            i_y[sl] = sym.real
            q_y[sl] = sym.imag
        i_sig = Signal('i', 'voltage', 'time', 'V', t, i_y)
        q_sig = Signal('q', 'voltage', 'time', 'V', t, q_y)
        return i_sig, q_sig, symbol_period, n_sym

    def test_reference_matches_nearest_when_all_decisions_correct(self):
        # Clean signal, no noise: nearest-point and reference-scored EVM
        # must agree exactly, since every symbol is assigned to the same
        # ideal point either way (AC#3's "the two agree" cross-check).
        rng = np.random.default_rng(1)
        choices = rng.integers(0, 4, size=16)
        i_sig, q_sig, sp, n_sym = self._qpsk_trace(choices)
        nearest = evm_constellation(i_sig, q_sig, symbol_period=sp,
                                    ideal_points=QPSK_IDEAL, n_symbols=n_sym)
        reference = QPSK_IDEAL[choices]
        scored = evm_constellation(i_sig, q_sig, symbol_period=sp,
                                   ideal_points=QPSK_IDEAL, n_symbols=n_sym,
                                   reference=reference)
        assert scored.evm_percent == pytest.approx(nearest.evm_percent)
        assert np.array_equal(scored.assigned_ideals, nearest.assigned_ideals)

    def test_reference_length_mismatch_raises(self):
        i_sig, q_sig, sp, n_sym = self._qpsk_trace([0, 1, 2, 3])
        with pytest.raises(ValueError,
                           match='evm_constellation:.*reference has.*expected 4'):
            evm_constellation(i_sig, q_sig, symbol_period=sp,
                              ideal_points=QPSK_IDEAL, n_symbols=n_sym,
                              reference=QPSK_IDEAL[[0, 1, 2]])

    def test_reference_membership_mismatch_raises(self):
        i_sig, q_sig, sp, n_sym = self._qpsk_trace([0, 1, 2, 3])
        bad_reference = list(QPSK_IDEAL[[0, 1, 2]]) + [3 + 3j]
        with pytest.raises(ValueError,
                           match='evm_constellation:.*reference\\[3\\].*'
                                 'does not match any of the 4 ideal_points'):
            evm_constellation(i_sig, q_sig, symbol_period=sp,
                              ideal_points=QPSK_IDEAL, n_symbols=n_sym,
                              reference=bad_reference)

    def test_reference_float_noise_scores_against_exact_ideal_point(self):
        # A caller-built reference value with floating-point noise (e.g.
        # reconstructed from bits via arithmetic rather than indexed
        # straight out of ideal_points) must be snapped to the exact
        # ideal_points entry it matches, not used verbatim -- otherwise
        # the noise would leak into the reported error.
        i_sig, q_sig, sp, n_sym = self._qpsk_trace([0])
        noisy = QPSK_IDEAL[0] + 1e-12
        result = evm_constellation(i_sig, q_sig, symbol_period=sp,
                                   ideal_points=QPSK_IDEAL, n_symbols=n_sym,
                                   reference=[noisy])
        assert result.assigned_ideals[0] == QPSK_IDEAL[0]
        assert result.evm_percent < 1e-6

    def test_closed_eye_reference_aware_reads_higher_than_nearest_point(self):
        """TASK-227 AC#4 / TASK-187.11 finding 3, ported into the test
        rather than into silk/examples/metrics/prbs_eye.py (paused with
        the eye work, per the owner ruling in this task's notes).

        Synthetic port of prbs_eye.py's construction: two NRZ baseband
        lanes carrying a PRBS7 sequence (I) and the same sequence
        cyclically shifted by 63 bits (Q, an m-sequence's near-orthogonal
        shift), each driven through a single-pole RC channel with
        tau = symbol_period (tau/UI = 1.0 -- a closed eye by construction,
        same knob prbs_eye.py sweeps). No SPICE: the channel is the exact
        closed-form ZOH response of tau*dv/dt + v = vtx, which is what
        prbs_eye.py's own convolution model proves an R-C node obeys.

        At this closed eye, scoring the received symbols against their
        OWN transmitted point (reference=) reads a meaningfully HIGHER
        EVM than nearest-point scoring, because more than a third of the
        decisions are wrong and nearest-point re-scores each of them to
        the wrong (closer) point -- the exact bias TASK-187.11 found and
        this task's docstring update now states. The percentages here are
        this synthetic channel's own numbers, not the sky130 example's;
        what matters is the same failure mode: nearest read LOW, and
        reference-aware reads the larger, correct number.
        """
        from silk.examples.prbs_generator_example import prbs_reference

        n_lead, n_bits, ui, spp = 12, 254, 1.0, 80
        bits_i = prbs_reference(n_bits, order=7)
        shift = 63
        bits_q = bits_i[shift:] + bits_i[:shift]
        symbols_i = [0] * n_lead + [1] * n_lead + bits_i
        symbols_q = [0] * n_lead + [1] * n_lead + bits_q
        n_total = len(symbols_i)

        t = np.arange(n_total * spp) * (ui / spp)

        def make_vtx(symbols):
            vtx = np.empty(n_total * spp)
            for k, bit in enumerate(symbols):
                vtx[k * spp:(k + 1) * spp] = 1.0 if bit else -1.0
            return vtx

        def rc_response(vtx, tau):
            # Exact zero-order-hold response of tau*dv/dt + v = vtx at a
            # piecewise-constant vtx: v[n] = a*v[n-1] + (1-a)*vtx[n],
            # a = exp(-dt/tau) -- the same single-pole model prbs_eye.py
            # verifies against a real sky130 R-C node to a few mV.
            dt = ui / spp
            a = math.exp(-dt / tau)
            v = np.empty_like(vtx)
            v[0] = vtx[0] * (1 - a)
            for n in range(1, len(vtx)):
                v[n] = a * v[n - 1] + (1 - a) * vtx[n]
            return v

        vrx_i = rc_response(make_vtx(symbols_i), tau=ui)
        vrx_q = rc_response(make_vtx(symbols_q), tau=ui)
        i_sig = Signal('i', 'voltage', 'time', 'V', t, vrx_i)
        q_sig = Signal('q', 'voltage', 'time', 'V', t, vrx_q)

        ideals = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=complex)
        t_start = 2 * n_lead * ui   # mid-slot fold: origin at the TX symbol boundary

        nearest = evm_constellation(i_sig, q_sig, symbol_period=ui,
                                    ideal_points=ideals, tstart=t_start,
                                    n_symbols=n_bits)

        data_bits_i = np.asarray(symbols_i[2 * n_lead:2 * n_lead + n_bits])
        data_bits_q = np.asarray(symbols_q[2 * n_lead:2 * n_lead + n_bits])
        reference = (np.where(data_bits_i == 1, 1.0, -1.0)
                    + 1j * np.where(data_bits_q == 1, 1.0, -1.0))
        reference_aware = evm_constellation(i_sig, q_sig, symbol_period=ui,
                                            ideal_points=ideals, tstart=t_start,
                                            n_symbols=n_bits, reference=reference)

        n_wrong = int(np.sum(nearest.assigned_ideals != reference_aware.assigned_ideals))
        assert n_wrong > n_bits // 4, (
            f'expected a genuinely closed eye with many wrong decisions, got '
            f'{n_wrong}/{n_bits}'
        )
        assert nearest.evm_percent == pytest.approx(64.19218753741484, abs=1e-6)
        assert reference_aware.evm_percent == pytest.approx(72.90950403290519, abs=1e-6)
        assert reference_aware.evm_percent - nearest.evm_percent > 5.0


class TestTier3SpectralImports:
    def test_top_level(self):
        from silk.metrics import evm_constellation, EVMResult
        assert callable(evm_constellation)


# ---------------------------------------------------------------------------
# Leftovers — harmonic frequency list
# ---------------------------------------------------------------------------

class TestHarmonicFreqs:
    def test_known_fundamental(self):
        sig = _coherent_sine_signal()   # M_COH=11 in N=1024 with TCLK=1ns
        f_fund = M_COH / (N * TCLK)
        freqs = harmonic_freqs(sig, tclk=TCLK, n_samples=N, n=5)
        assert len(freqs) == 5
        assert math.isclose(freqs[0], f_fund, rel_tol=1e-9)
        assert math.isclose(freqs[1], 2 * f_fund, rel_tol=1e-9)
        assert math.isclose(freqs[4], 5 * f_fund, rel_tol=1e-9)

    def test_not_truncated_at_nyquist(self):
        # M_COH=11 in N=1024 → fund_bin=11; Nyquist bin=512. Asking for
        # n=100 used to return only the 46 harmonics <= Nyquist; TASK-79:
        # every harmonic's TRUE frequency k·f0 is reported (length n),
        # and the ones above fs/2 are exactly the aliased ones.
        sig = _coherent_sine_signal()
        f_fund = M_COH / (N * TCLK)
        freqs = harmonic_freqs(sig, tclk=TCLK, n_samples=N, n=100)
        assert len(freqs) == 100
        assert np.allclose(freqs, f_fund * np.arange(1, 101))
        assert np.sum(freqs > 0.5 / TCLK) == 100 - 46
        # observed=True: where sampling put them — always <= Nyquist
        obs = harmonic_freqs(sig, tclk=TCLK, n_samples=N, n=100, observed=True)
        assert len(obs) == 100
        assert np.all(obs[np.isfinite(obs)] <= 0.5 / TCLK)
        assert np.allclose(obs[:46], freqs[:46])          # in-band: identical

    def test_invalid_n(self):
        with pytest.raises(ValueError, match='n must be >= 1'):
            harmonic_freqs(_coherent_sine_signal(), tclk=TCLK, n_samples=N, n=0)


class TestLeftoverSpectralImports:
    def test_top_level(self):
        from silk.metrics import harmonic_freqs
        assert callable(harmonic_freqs)


# ---------------------------------------------------------------------------
# Beyond-vendor — Welch PSD, coherence
# ---------------------------------------------------------------------------

class TestWelchPsd:
    def test_returns_signal(self):
        spec = welch_psd(_coherent_sine_signal(), tclk=TCLK, n_samples=N,
                         segments=4)
        assert isinstance(spec, Signal)
        assert spec.xlabel == 'frequency'

    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match='overlap must be in'):
            welch_psd(_coherent_sine_signal(), tclk=TCLK, n_samples=N,
                      overlap=1.0)

    def test_invalid_segments(self):
        with pytest.raises(ValueError, match='segments must be >= 1'):
            welch_psd(_coherent_sine_signal(), tclk=TCLK, n_samples=N,
                      segments=0)

    def test_lower_variance_than_single_psd(self):
        # On a noisy signal, Welch PSD should be smoother than the
        # single-FFT version (smaller std-dev across bins in a flat noise
        # region).
        rng = np.random.default_rng(123)
        N_long = 8192
        t = np.arange(N_long) * TCLK
        white = rng.standard_normal(N_long)
        sig = Signal('n', 'voltage', 'time', 'V', t, white)
        from silk.metrics.spectral import psd as plain_psd
        single = plain_psd(sig, tclk=TCLK, n_samples=N_long, window='hann')
        welch = welch_psd(sig, tclk=TCLK, n_samples=N_long,
                          segments=8, overlap=0.5, window='hann')
        # Compare std-dev in mid-band (avoid endpoints where windowing
        # affects both).
        mid_single = single.y[len(single.y)//4:3*len(single.y)//4]
        mid_welch  = welch.y[len(welch.y)//4:3*len(welch.y)//4]
        assert np.std(mid_welch) < np.std(mid_single)


class TestCoherence:
    def test_perfect_correlation(self):
        # b = a → coherence ≡ 1 (within numerical noise) at all freqs
        rng = np.random.default_rng(0)
        N_long = 4096
        t = np.arange(N_long) * TCLK
        y = rng.standard_normal(N_long)
        a = Signal('a', 'voltage', 'time', 'V', t, y)
        b = Signal('b', 'voltage', 'time', 'V', t, y.copy())
        coh = coherence(a, b, tclk=TCLK, n_samples=N_long, segments=8)
        assert np.allclose(coh.y, 1.0, atol=1e-9)

    def test_uncorrelated_low(self):
        # Two independent white-noise sources → coherence ≈ 1/segments
        # (statistical floor; smaller for more segments)
        rng = np.random.default_rng(7)
        N_long = 8192
        t = np.arange(N_long) * TCLK
        a = Signal('a', 'voltage', 'time', 'V', t, rng.standard_normal(N_long))
        b = Signal('b', 'voltage', 'time', 'V', t, rng.standard_normal(N_long))
        coh = coherence(a, b, tclk=TCLK, n_samples=N_long, segments=16)
        # Mean coherence should be much less than 1
        assert float(np.mean(coh.y)) < 0.4

    def test_mismatched_axes(self):
        t1 = np.linspace(0, 1, 100)
        t2 = np.linspace(0, 2, 100)
        a = Signal('a', 'voltage', 'time', 'V', t1, np.zeros_like(t1))
        b = Signal('b', 'voltage', 'time', 'V', t2, np.zeros_like(t2))
        with pytest.raises(ValueError, match='same time axis'):
            coherence(a, b, tclk=TCLK)

    def test_zero_variance_segments_is_nan(self):
        """TASK-111: a and b constant (zero-variance) over every averaged
        segment means S_xx = S_yy = S_xy = 0 at every frequency, so gamma^2
        is 0/0 -- degenerate DATA with no defined coherence (decision-1
        Section 1), not "uncorrelated". Before this fix the denominator was
        floored at np.finfo(float).tiny, so the result read as a confident
        0.0 at every frequency; the floor never guarded a nonzero numerator
        (S_xy is 0 in exactly the cases S_xx*S_yy is, by Cauchy-Schwarz), so
        it is dropped and the honest nan falls out."""
        N = 256
        t = np.arange(N) * TCLK
        a = Signal('a', 'voltage', 'time', 'V', t, np.full(N, 3.0))
        b = Signal('b', 'voltage', 'time', 'V', t, np.full(N, -1.0))
        coh = coherence(a, b, tclk=TCLK, n_samples=N, segments=4)
        assert np.all(np.isnan(coh.y))


class TestBeyondVendorSpectralImports:
    def test_top_level(self):
        from silk.metrics import welch_psd, coherence
        assert callable(welch_psd)
        assert callable(coherence)


class TestSampleAtClockEndTolerance:
    """TASK-335: the end-of-trace check shares max_n's relative tolerance."""

    @staticmethod
    def _sig(tstop, tclk):
        t = np.linspace(0.0, tstop, int(round(tstop / tclk)) * 4 + 1)
        return Signal('s', 'voltage', 'time', 'V', t, np.sin(t))

    def test_max_n_request_succeeds_on_decimal_stop(self):
        sig = self._sig(0.7, 0.01)
        max_n = int(math.floor(0.7 / 0.01 + 1e-9)) + 1
        s = _sample_at_clock(sig, metric='probe', tclk=0.01, n_samples=max_n)
        assert len(s) == max_n

    def test_default_n_samples_never_raises(self):
        sig = self._sig(0.7, 0.1)
        assert len(_sample_at_clock(sig, metric='probe', tclk=0.1)) == 8

    def test_explicit_overrun_names_n_samples(self):
        sig = self._sig(0.7, 0.1)
        with pytest.raises(ValueError, match='reduce n_samples=64'):
            _sample_at_clock(sig, metric='probe', tclk=0.1, n_samples=64)

    def test_real_overrun_still_raises(self):
        sig = self._sig(0.7, 0.01)
        with pytest.raises(ValueError, match='exceeds trace end'):
            _sample_at_clock(sig, metric='probe', tclk=0.01, n_samples=72)


class TestInterpolatedClock:
    def test_equivalent_timestamp_arithmetic(self):
        fs = 1e6
        k = np.arange(1024)
        y = np.sin(2 * np.pi * 11 * k / 1024)
        mul = Signal('mul', 'voltage', 'time', 'V', k * (1 / fs), y)
        div = Signal('div', 'voltage', 'time', 'V', k / fs, y)
        kw = dict(tclk=1 / fs, n_samples=1024)
        for sig in (mul, div):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', MetricWarning)
                sampled = _sample_at_clock(sig, metric='control', **kw)
            np.testing.assert_allclose(sampled, y, atol=2e-14, rtol=0)
            # These equivalent grids differ by much less than half a period,
            # so the same-index recorded time is independently the nearest.
            distances = np.abs(sig.x - k * (1 / fs))
            if np.any(distances):
                diagnostic, = [w.message for w in caught]
                assert diagnostic.details['interpolated_sample_count'] == np.count_nonzero(distances)
                assert diagnostic.details['max_nearest_time_distance_s'] == distances.max()
            else:
                assert not caught
            # Coherent unit sine: N²/2 fundamental power, 511 eps bins.
            expected = 10 * np.log10((1024**2 / 2) / (511 * np.finfo(float).eps))
            assert sndr(sig, **kw) == pytest.approx(expected, abs=1e-9)
            assert enob(sig, **kw) == pytest.approx((expected - 1.76) / 6.02, abs=1e-9)

    @pytest.mark.parametrize('grid,count,distance', [
        ([0, 1, 2, 3, 4, 5, 6], 3, .375),
        ([0, .125, .75, 1.25, 2.125, 3, 3.75, 4.125, 5, 6], 4, .375),
        ([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3], 2, .125),
    ])
    def test_linear_waveform_at_requested_clock(self, grid, count, distance):
        x = np.array(grid)
        sig = Signal('ramp', 'voltage', 'time', 'V', x, 3 * x - 2)
        with pytest.warns(MetricWarning) as caught:
            actual = _sample_at_clock(sig, metric='control', tclk=.625,
                                      tstart=.375, n_samples=4)
        np.testing.assert_allclose(actual, [-.875, 1, 2.875, 4.75], atol=1e-14)
        w, = [w.message for w in caught]
        assert w.details['interpolated_sample_count'] == count
        assert w.details['max_nearest_time_distance_s'] == distance
