"""TASK-436 — the tone-spectrum core's guard bands: an empty residual is a
caller error, DC owns a band like every tone, and the noise sums are
referred back to the default selection's bin count."""
import math

import numpy as np
import pytest

from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal
import silk.metrics.spectral as S
from silk.metrics import converter


def _sig(y, tclk=1.0):
    return Signal('v', 'voltage', 'time', 'V', np.arange(len(y)) * tclk, y)


def _quantised_tone(bits, cycles, n):
    lsb = 2.0 / 2 ** bits
    y = np.sin(2 * np.pi * cycles * np.arange(n) / n + 0.3)
    return np.round(y / lsb) * lsb


NOISE_METRICS = {
    'sndr': S.sndr, 'sinad': S.sinad, 'enob': converter.enob, 'snr': S.snr,
    'noise_floor_dbfs': S.noise_floor_dbfs,
    'sfdr': lambda s, **kw: S.sfdr(s, **kw).sfdr_db,
}


# --- AC#1: a guard band that empties the residual is a caller error ---------

@pytest.mark.parametrize('guard', [64, 127, 128])
@pytest.mark.parametrize('cycles', [13, 100])
@pytest.mark.parametrize('name', sorted(NOISE_METRICS))
def test_a_guard_band_that_swallows_the_spectrum_raises(name, cycles, guard):
    n = 256
    y = np.sin(2 * np.pi * cycles * np.arange(n) / n) + \
        1e-3 * np.random.default_rng(0).standard_normal(n)
    with pytest.raises(ValueError, match=r'guard_bins=\d+.*non-DC bins 1\.\.128'):
        NOISE_METRICS[name](_sig(y), tclk=1.0, n_samples=n, guard_bins=guard)


@pytest.mark.parametrize('name', sorted(NOISE_METRICS))
def test_a_guard_band_that_fits_stays_finite(name):
    n = 256
    y = np.sin(2 * np.pi * 100 * np.arange(n) / n) + \
        1e-3 * np.random.default_rng(0).standard_normal(n)
    v = NOISE_METRICS[name](_sig(y), tclk=1.0, n_samples=n, guard_bins=5)
    assert math.isfinite(v)


@pytest.mark.parametrize('name', sorted(NOISE_METRICS))
def test_a_window_lobe_that_swallows_a_tiny_record_is_nan_not_inf(name):
    """No guard_bins: the empty residual is degenerate data (decision-1)."""
    y = np.sin(2 * np.pi * np.arange(4) / 4)
    with pytest.warns(MetricWarning, match='no_residual_bins'):
        v = NOISE_METRICS[name](_sig(y), tclk=1.0, n_samples=4,
                                window='blackman')
    assert math.isnan(v)


# --- AC#2: mean removal + window does not plant a bin-1 spur -----------------

@pytest.mark.parametrize('window, guard, expected', [
    # Artefact-free references from the review (DC band excluded by hand).
    ('hann', 10, 67.6), ('blackman', 10, 73.3),
    # At guard_bins=3 the tone's own leakage skirt limits the reading.
    ('hann', 3, 47.5), ('blackman', 3, 61.3),
])
def test_windowed_non_coherent_12_bit_tone(window, guard, expected):
    n = 1024
    sig = _sig(_quantised_tone(12, 37.5, n))
    kw = dict(tclk=1.0, n_samples=n, window=window, guard_bins=guard)
    assert S.sndr(sig, **kw) == pytest.approx(expected, abs=1.5)
    lobe = S._WINDOW_LOBE_BINS[window]
    spur_bin = S.sfdr(sig, **kw).spur_hz * n
    assert spur_bin > lobe + guard          # never DC's window lobe


def test_dc_offset_does_not_reach_the_noise_sum():
    n = 1024
    y = _quantised_tone(12, 37.5, n)
    kw = dict(tclk=1.0, n_samples=n, window='hann', guard_bins=10)
    assert S.sndr(_sig(y + 0.37), **kw) == pytest.approx(
        S.sndr(_sig(y), **kw), abs=0.05)


# --- AC#3: excluded bins are accounted for ------------------------------------

@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('metric', [S.snr, S.sndr])
def test_guard_bins_do_not_bias_white_noise_high(metric, seed):
    n = 16384
    k = np.arange(n)
    y = np.sin(2 * np.pi * (n // 20 + 1) * k / n) + \
        1e-2 * np.random.default_rng(seed).standard_normal(n)
    sig = _sig(y)
    base = metric(sig, tclk=1.0, n_samples=n, guard_bins=0)
    assert metric(sig, tclk=1.0, n_samples=n, guard_bins=10) == \
        pytest.approx(base, abs=0.05)
