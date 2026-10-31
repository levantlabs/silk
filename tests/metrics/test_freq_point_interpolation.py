"""TASK-413: gain_at_freq / phase_at_freq / noise_density_at_freq interpolate."""
import numpy as np
import pytest

from silk.metrics import ac, noise
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal

F = np.logspace(0, 6, 61)                 # 10 points/decade
FP = 1e3
OFF_GRID = [1.1e3, 891.25, 3.3e3, 10 ** 3.05, 12.3, 4.4e5]


def one_pole(f=F):
    return Signal('h', 'voltage', 'frequency', 'V/V', f,
                  1.0 / (1.0 + 1j * f / FP))


def density(f, y):
    return Signal('d', 'noise', 'frequency', 'V/sqrt(Hz)', f, y)


def h(f):
    return 1.0 / (1.0 + 1j * f / FP)


def db(x):
    return 20 * np.log10(x)


@pytest.mark.parametrize('k', [-1.0, -0.5, 0.0, 1.0])
def test_power_law_recovered_exactly(k):
    sig = density(F, 3e-9 * F ** k)
    for fq in OFF_GRID:
        got = noise.noise_density_at_freq(sig, fq)
        assert got == pytest.approx(3e-9 * fq ** k, rel=1e-9)


def test_noise_nearest_is_off_on_power_law():
    sig = density(F, 3e-9 / F ** 0.5)
    fq = 10 ** 0.45
    near = noise.noise_density_at_freq(sig, fq, interpolate=False)
    assert abs(near / (3e-9 * fq ** -0.5) - 1) > 0.01


def test_one_pole_gain_phase_within_tolerance_and_nearest_error_recorded():
    sig = one_pole()
    worst_near_db = worst_near_deg = 0.0
    for fq in OFF_GRID + [FP]:
        g = ac.gain_at_freq(sig, fq)
        p = ac.phase_at_freq(sig, fq)
        assert abs(db(g) - db(abs(h(fq)))) < 0.05
        assert abs(p - np.degrees(np.angle(h(fq)))) < 0.5
        gn = ac.gain_at_freq(sig, fq, interpolate=False)
        pn = ac.phase_at_freq(sig, fq, interpolate=False)
        worst_near_db = max(worst_near_db, abs(db(gn) - db(abs(h(fq)))))
        worst_near_deg = max(worst_near_deg,
                             abs(pn - np.degrees(np.angle(h(fq)))))
    # nearest-sample error beside the interpolated one (measured: ~1.3 dB, ~6 deg)
    assert worst_near_db > 0.5 and worst_near_deg > 2.0


def test_on_grid_returns_sample_both_modes():
    sig = one_pole()
    nd = density(F, 1e-9 / np.sqrt(F))
    for i in (0, 7, 30, 60):
        for mode in (True, False):
            assert ac.gain_at_freq(sig, F[i], interpolate=mode) == float(np.abs(sig.y[i]))
            assert ac.phase_at_freq(sig, F[i], interpolate=mode) == \
                float(np.degrees(np.angle(sig.y[i])))
            assert noise.noise_density_at_freq(nd, F[i], interpolate=mode) == nd.y[i]


def test_interpolate_false_is_nearest_bit_for_bit():
    sig = one_pole()
    nd = density(F, 1e-9 / np.sqrt(F))
    for fq in OFF_GRID:
        i = int(np.argmin(np.abs(F - fq)))
        # NumPy 1.x scalar __abs__ can round differently from np.abs.
        assert ac.gain_at_freq(sig, fq, interpolate=False) == float(np.abs(sig.y[i]))
        assert ac.phase_at_freq(sig, fq, interpolate=False) == \
            float(np.degrees(np.angle(sig.y[i])))
        assert noise.noise_density_at_freq(nd, fq, interpolate=False) == float(nd.y[i])


def test_phase_does_not_interpolate_across_wrap():
    # +170 deg and -170 deg neighbours: true path crosses 180, not 0.
    f = np.array([1.0, 10.0, 100.0])
    y = np.exp(1j * np.radians([170.0, -170.0, -150.0]))
    sig = Signal('w', 'voltage', 'frequency', 'V/V', f, y)
    mid = ac.phase_at_freq(sig, np.sqrt(10.0))
    assert abs(abs(mid) - 180.0) < 1e-9          # naive linear gives 0
    q = ac.phase_at_freq(sig, 10 ** 0.25)        # a quarter of the way
    assert q == pytest.approx(175.0, abs=1e-9)
    q = ac.phase_at_freq(sig, 10 ** 0.75)        # three quarters
    assert q == pytest.approx(-175.0, abs=1e-9)


def test_unsorted_sweep_matches_sorted():
    sig = one_pole()
    rev = Signal('h', 'voltage', 'frequency', 'V/V', F[::-1], sig.y[::-1])
    assert ac.gain_at_freq(rev, 1.1e3) == ac.gain_at_freq(sig, 1.1e3)


def test_outside_sweep_still_nan_with_warning():
    sig = one_pole()
    nd = density(F, 1e-9 / np.sqrt(F))
    for fn, s in ((ac.gain_at_freq, sig), (ac.phase_at_freq, sig),
                  (noise.noise_density_at_freq, nd)):
        with pytest.warns(MetricWarning):
            assert np.isnan(fn(s, 1e7))
