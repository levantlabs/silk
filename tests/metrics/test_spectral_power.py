"""Sampled-record power checks independent of the rfft weighting code."""
import math

import numpy as np
import pytest

from silk import metrics as M
from silk.metrics.spectral import _rfft_spectrum
from silk.signals.base import Signal


def _signal(y):
    return Signal('v', 'voltage', 'time', 'V', np.arange(len(y)), y)


@pytest.mark.parametrize('n', [255, 256])
def test_one_sided_power_matches_time_domain_and_full_fft(n):
    k = np.arange(n)
    y = (1.7 + np.cos(2 * np.pi * 13 * k / n)
         + 0.3 * np.cos(2 * np.pi * (n // 2) * k / n))
    centered = y - np.mean(y)
    full_power = np.abs(np.fft.fft(centered)) ** 2
    _, power, fundamental = _rfft_spectrum(y)
    assert fundamental == 13
    assert np.sum(power) / n**2 == pytest.approx(np.mean(centered**2))
    for b in (13, n // 2):
        # Distinct positive/negative frequencies; Nyquist is its own partner.
        expected = sum(full_power[i] for i in {b, n - b})
        assert power[b] == pytest.approx(expected)


@pytest.mark.parametrize('n', [255, 256])
def test_ratios_include_highest_bin_residual_power(n):
    k = np.arange(n)
    fundamental = np.cos(2 * np.pi * 13 * k / n)
    harmonic = 0.03 * np.cos(2 * np.pi * 26 * k / n)
    noise = 0.04 * np.cos(2 * np.pi * (n // 2) * k / n)
    sig = _signal(fundamental + harmonic + noise)
    kw = dict(tclk=1.0, n_samples=n)
    signal_power = np.mean(fundamental**2)
    harmonic_power = np.mean(harmonic**2)
    noise_power = np.mean(noise**2)
    residual_power = harmonic_power + noise_power
    sndr_db = 10 * math.log10(signal_power / residual_power)
    assert M.sndr(sig, **kw) == pytest.approx(sndr_db)
    assert M.sinad(sig, **kw) == pytest.approx(sndr_db)
    assert M.enob(sig, **kw) == pytest.approx((sndr_db - 1.76) / 6.02)
    assert M.snr(sig, **kw) == pytest.approx(
        10 * math.log10(signal_power / noise_power))
    assert M.sfdr(sig, **kw).sfdr_db == pytest.approx(
        10 * math.log10(signal_power / max(harmonic_power, noise_power)))
    assert M.noise_floor_dbfs(sig, **kw) == pytest.approx(
        10 * math.log10(residual_power / (n // 2 - 1) / signal_power))
    assert M.thd_percent(sig, **kw) == pytest.approx(3.0)
    assert M.thd(sig, **kw) == pytest.approx(20 * math.log10(0.03))
    assert M.harmonic_levels_db(sig, **kw)[0] == pytest.approx(
        20 * math.log10(0.03))


@pytest.mark.parametrize('phase', [0.0, math.pi / 4, math.pi / 2])
def test_nyquist_harmonic_measures_sampled_power_not_analog_amplitude(phase):
    n = 256
    k = np.arange(n)
    fundamental = np.cos(2 * np.pi * 64 * k / n)
    harmonic = 0.2 * np.cos(np.pi * k + phase)
    sig = _signal(fundamental + harmonic)
    kw = dict(tclk=1.0, n_samples=n, n_harmonics=2)
    ratio = np.mean(harmonic**2) / np.mean(fundamental**2)
    assert M.thd_percent(sig, **kw) == pytest.approx(
        100 * math.sqrt(ratio), abs=1e-7)
    if phase != math.pi / 2:
        expected_db = 10 * math.log10(ratio)
        assert M.thd(sig, **kw) == pytest.approx(expected_db)
        assert M.harmonic_levels_db(sig, **kw)[0] == pytest.approx(expected_db)
    else:
        assert math.isfinite(M.thd(sig, **kw))
        assert M.thd(sig, **kw) < -150


def test_nyquist_fundamental_and_ordinary_spur_use_sampled_rms():
    n = 256
    k = np.arange(n)
    sig = _signal((-1.0)**k + 0.01 * np.cos(2 * np.pi * 13 * k / n))
    kw = dict(tclk=1.0, n_samples=n)
    # Nyquist samples have 1 V RMS; the ordinary spur has .01/sqrt(2) V RMS.
    expected = 10 * math.log10(1 / 0.00005)
    assert M.sfdr(sig, **kw).sfdr_db == pytest.approx(expected)
    assert M.sndr(sig, **kw) == pytest.approx(expected)
    assert M.snr(sig, **kw) == pytest.approx(expected)


def test_fundamental_election_uses_power_and_preserves_raw_coefficients():
    n = 256
    k = np.arange(n)
    sig = _signal(np.cos(2 * np.pi * 13 * k / n) + 0.6 * (-1.0)**k)
    kw = dict(tclk=1.0, n_samples=n)
    # Ordinary component: .5 V². Nyquist: .36 V² (but larger raw |X|²).
    assert M.sfdr(sig, **kw).fundamental_hz == 13 / n
    assert M.harmonic_freqs(sig, **kw, n=1)[0] == 13 / n
    assert M.harmonic(sig, 1, **kw) == pytest.approx(128 + 0j)
    raw = np.abs(np.fft.rfft(sig.y - np.mean(sig.y))) ** 2
    np.testing.assert_allclose(M.fft(sig, **kw).y, raw)


def test_weighting_keeps_uniform_dead_floor_and_adds_eps_after_weighting():
    n = 256
    _, power, _ = _rfft_spectrum(np.ones(n))
    assert np.all(power == np.finfo(float).eps)
    sig = _signal(0.4 * np.cos(2 * np.pi * 13 * np.arange(n) / n))
    kw = dict(tclk=1.0, n_samples=n)
    # P_fund = N² * time-domain mean square = N² * .4² / 2.
    expected = 10 * math.log10(n**2 * 0.4**2 / 2 / (127 * np.finfo(float).eps))
    assert M.sndr(sig, **kw) == pytest.approx(expected)
