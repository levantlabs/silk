"""TASK-395: fft window= and scaling=."""
import numpy as np
import pytest

from silk.metrics import spectral as M
from silk.signals.base import Signal

N, CYC = 256, 7
WINDOWS = ['rectangular', None, 'hann', 'hamming', 'blackman']


def _sine(amp=1.0, cyc=CYC, dc=0.0):
    t = np.arange(N) * 1.0
    y = dc + amp * np.sin(2 * np.pi * cyc * t / N + 0.3)
    return Signal('v', 'voltage', 'time', 'V', t, y)


def _kw():
    return dict(tclk=1.0, n_samples=N)


def test_default_is_raw_rectangular_exactly():
    s = _sine(0.7, dc=0.2)
    ref = np.abs(np.fft.rfft(s.y - np.mean(s.y))) ** 2
    out = M.fft(s, **_kw())
    assert np.array_equal(out.y, ref)
    assert np.array_equal(M.fft(s, window='rectangular', **_kw()).y, ref)
    assert np.array_equal(M.fft(s, window=None, **_kw()).y, ref)


@pytest.mark.parametrize('window', WINDOWS)
def test_amplitude_reads_peak_volts(window):
    out = M.fft(_sine(1.0, dc=0.4), window=window, scaling='amplitude',
                **_kw())
    # np.hanning etc. are symmetric (not periodic) windows, so the peak
    # is within ~1% of 1 V, not exact; compare against the DFT directly.
    w = M._make_window(window, N, metric='t')
    s = _sine(1.0, dc=0.4)
    X = np.fft.rfft((s.y - s.y.mean()) * w)
    assert out.y[CYC] == pytest.approx(2 * abs(X[CYC]) / w.sum())
    assert out.y[CYC] == pytest.approx(1.0, abs=0.02)


def test_rectangular_exact_amplitude_and_power():
    s = _sine(1.0)
    assert M.fft(s, scaling='amplitude', **_kw()).y[CYC] == pytest.approx(1.0)
    assert M.fft(s, scaling='power', **_kw()).y[CYC] == pytest.approx(0.5)
    assert M.fft(s, scaling='amplitude', **_kw()).ylabel == 'V'
    assert M.fft(s, scaling='power', **_kw()).ylabel == 'V^2'


@pytest.mark.parametrize('window', WINDOWS)
def test_power_is_half_amplitude_squared(window):
    s = _sine(1.0)
    a = M.fft(s, window=window, scaling='amplitude', **_kw()).y
    p = M.fft(s, window=window, scaling='power', **_kw()).y
    assert p[CYC] == pytest.approx(a[CYC] ** 2 / 2)
    assert p[CYC] == pytest.approx(0.5, abs=0.02)


def test_nyquist_bin_unpaired():
    t = np.arange(N) * 1.0
    s = Signal('v', 'voltage', 'time', 'V', t, 0.5 * (-1.0) ** np.arange(N))
    assert M.fft(s, scaling='amplitude', **_kw()).y[-1] == pytest.approx(0.5)
    assert M.fft(s, scaling='power', **_kw()).y[-1] == pytest.approx(0.25)


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match='scaling'):
        M.fft(_sine(), scaling='rms', **_kw())
    with pytest.raises(ValueError, match='window'):
        M.fft(_sine(), window='kaiser', **_kw())


@pytest.mark.parametrize('scaling,unit', [('amplitude', 'A'), ('power', 'A^2')])
def test_output_unit_follows_input_unit(scaling, unit):
    t = np.arange(N) * 1.0
    s = Signal('i', 'current', 'time', 'A', t,
               1e-3 * np.sin(2 * np.pi * CYC * t / N))
    assert M.fft(s, scaling=scaling, **_kw()).ylabel == unit
