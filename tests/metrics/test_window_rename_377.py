"""TASK-377: windowed_min/max -> minimum/maximum; tstart/tstop -> start/end."""
import numpy as np
import pytest

import silk.metrics as M
from silk.signals import Signal

PRIMITIVES = ['average', 'rms', 'stddev', 'peak', 'peak_to_peak', 'minimum',
              'maximum', 'argmin_x', 'argmax_x', 'integral']


def _sig():
    x = np.linspace(0.0, 2.0, 201)
    return Signal('v', 'voltage', 'time', 'V', x, np.sin(2 * np.pi * x) - 0.5 * x)


def test_old_names_removed():
    assert not hasattr(M, 'windowed_min') and not hasattr(M, 'windowed_max')
    from silk.metrics import waveform
    assert not hasattr(waveform, 'windowed_min')


def test_minimum_maximum_match_numpy_on_window():
    s = _sig()
    m = (s.x >= 0.5) & (s.x <= 1.5)
    assert M.minimum(s, start=0.5, end=1.5) == pytest.approx(s.y[m].min())
    assert M.maximum(s, start=0.5, end=1.5) == pytest.approx(s.y[m].max())
    assert M.minimum(s) == pytest.approx(s.y.min())
    assert M.maximum(s) == pytest.approx(s.y.max())


@pytest.mark.parametrize('name', PRIMITIVES)
def test_old_window_keywords_raise_and_new_work(name):
    s, f = _sig(), getattr(M, name)
    with pytest.raises(TypeError):
        f(s, tstart=0.5)
    with pytest.raises(TypeError):
        f(s, tstop=1.5)
    assert f(s, start=0.5, end=1.5) == f(s, start=0.5, end=1.5)
