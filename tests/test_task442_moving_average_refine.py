"""TASK-442: moving_average cut segment is piecewise-linear; argmin/argmax
refine=True never fits through a synthetic window-edge point."""
import numpy as np
import pytest

from silk.metrics.waveform import argmax_x, moving_average
from silk.metrics.statistics import average
from silk.signals import Signal


def _sig(x, y):
    return Signal('s', 'voltage', 'time', 'V', np.asarray(x, float), np.asarray(y, float))


def _check(x, y, w):
    sig = _sig(x, y)
    out = moving_average(sig, window=w)
    n = 0
    for k, xk in enumerate(x):
        if xk - w < x[0]:
            assert np.isnan(out.y[k])
            continue
        n += 1
        assert out.y[k] == pytest.approx(
            average(sig, start=xk - w, end=xk), abs=1e-12)
    assert n


def test_moving_average_nonuniform_reproducer():
    x = [0, 1, 3, 3.5, 6, 6.2, 9]
    y = [0, 2, 1, 5, 5, 0, 3]
    _check(x, y, 2.0)
    out = moving_average(_sig(x, y), window=2.0)
    assert out.y[-1] == pytest.approx(1.9286, abs=1e-4)


def test_moving_average_random_grid():
    rng = np.random.default_rng(442)
    x = np.cumsum(rng.uniform(0.05, 1.0, 60))
    y = rng.normal(size=60)
    _check(x, y, 2.3)


def test_argmax_refine_ignores_synthetic_edge():
    x = np.round(np.arange(0, 2.0001, 0.1), 10)
    sig = _sig(x, np.cos(2 * np.pi * (x - 0.93) / 3.0 * 1.0))
    base = argmax_x(sig, refine=True)
    for start in (0.85, 0.89):
        assert argmax_x(sig, start=start, refine=True) == pytest.approx(base, abs=1e-12)
