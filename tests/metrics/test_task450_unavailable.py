"""TASK-450: unavailable cases return nan WITH a MetricWarning (decision-1)."""
import numpy as np
import pytest

from silk.metrics import ac, noise
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def freq(y, x, unit=''):
    return Signal('s', 'ac', 'frequency', unit, np.asarray(x, float), np.asarray(y))


def test_group_delay_all_zero_is_nan_with_one_warning():
    with pytest.warns(MetricWarning) as w:
        out = ac.group_delay(freq([0j, 0j, 0j], [1, 2, 3]))
    assert len(w) == 1
    assert np.isnan(out.y).all()


def test_group_delay_partly_zero_marks_only_zero_points():
    f = np.logspace(0, 3, 8)
    h = 1 / (1 + 1j * f / 100)
    h[3] = 0
    with pytest.warns(MetricWarning) as w:
        out = ac.group_delay(freq(h, f))
    assert len(w) == 1
    assert np.isnan(out.y[3]) and np.isfinite(np.delete(out.y, 3)).all()


def test_group_delay_nonzero_no_warning(recwarn):
    f = np.logspace(0, 3, 8)
    ac.group_delay(freq(1 / (1 + 1j * f / 100), f))
    assert not [w for w in recwarn if issubclass(w.category, MetricWarning)]


def test_q_from_peak_two_flat_points_warns():
    with pytest.warns(MetricWarning, match='half-power bandwidth'):
        assert np.isnan(ac.q_from_peak(freq([2.0, 2.0], [1, 10])))


def dens(y, f):
    return Signal('d', 'noise', 'frequency', 'V/sqrt(Hz)',
                  np.asarray(f, float), np.asarray(y, float))


def test_white_only_is_nan_with_warning():
    f = np.logspace(1, 7, 61)
    with pytest.warns(MetricWarning, match='1/f region'):
        assert np.isnan(noise.one_over_f_corner(dens(np.full(61, 1e-6), f)))


def test_corner_at_lower_edge_stays_finite():
    f = np.logspace(1, 7, 61)
    y = 1e-6 * np.sqrt(1 + 100.0 / f)   # corner at 100 Hz, well inside sweep
    y[0] = 1e-6 * 2.0                    # lowest sample clearly 1/f-dominated
    assert np.isfinite(noise.one_over_f_corner(dens(y, f)))


def test_three_samples_warns():
    with pytest.warns(MetricWarning, match='fewer than four'):
        assert np.isnan(noise.one_over_f_corner(dens([1, 1, 1], [1, 10, 100])))
