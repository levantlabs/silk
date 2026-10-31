"""TASK-437: spectral windows are periodic (DFT-even), so a coherent tone
does not leak and ratio metrics are not capped by the window."""
import numpy as np
import pytest

from silk.metrics import spectral as S
from silk.metrics import converter
from silk.signals.base import Signal

N, CYC, TCLK = 1024, 101, 1.0
WINDOWS = ['hann', 'hamming', 'blackman']


def _sig(y):
    t = np.arange(N) * TCLK
    return Signal('v', 'voltage', 'time', 'V', t, y)


def _tone(bits=None):
    y = 0.5 * np.sin(2 * np.pi * CYC * np.arange(N) / N + 0.3)
    if bits:
        lsb = 1.0 / 2 ** bits
        y = np.round(y / lsb) * lsb
    return _sig(y)


@pytest.mark.parametrize('window', WINDOWS)
def test_window_is_periodic(window):
    w = S._make_window(window, 16, metric='t')
    sym = getattr(np, {'hann': 'hanning'}.get(window, window))(17)
    assert np.array_equal(w, sym[:-1])


@pytest.mark.parametrize('window', WINDOWS)
def test_clean_coherent_tone_is_not_capped(window):
    assert S.sndr(_tone(), tclk=TCLK, window=window) > 150


@pytest.mark.parametrize('window', WINDOWS)
def test_14bit_enob_matches_rectangular(window):
    ref = converter.enob(_tone(14), tclk=TCLK, window='rectangular')
    got = converter.enob(_tone(14), tclk=TCLK, window=window)
    assert abs(got - ref) < 0.1, (window, got, ref)
