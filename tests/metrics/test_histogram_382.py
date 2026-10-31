"""histogram: time-weighted distribution of a waveform's values (TASK-382)."""
import hashlib
import json
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics import HistogramResult, histogram
from silk.signals.base import Signal
from silk.signals.schema import export_metric_schema


def _tran(t, y, name='v'):
    return Signal(name, 'transient', 'time', 'V', np.asarray(t, float),
                  np.asarray(y, float))


def test_returns_one_fixed_type_whatever_bins_is():
    s = _tran([0, 1, 2], [0, 1, 0])
    for bins in (4, [0.0, 0.5, 1.0], np.array([0.0, 1.0])):
        h = histogram(s, bins=bins)
        assert type(h) is HistogramResult
        assert len(h.edges) == len(h.fractions) + 1


def test_uniform_ramp_gives_equal_fractions():
    # Non-uniform sampling must not matter: the ramp is linear.
    t = np.sort(np.r_[0.0, 1.0, np.random.RandomState(1).uniform(0, 1, 37)])
    h = histogram(_tran(t, 2.0 * t), bins=8)
    np.testing.assert_allclose(h.edges, np.linspace(0, 2, 9))
    np.testing.assert_allclose(h.fractions, np.full(8, 1 / 8), rtol=1e-12)
    assert h.fractions.sum() == pytest.approx(1.0, abs=1e-12)


def test_segment_crossing_several_bins_splits_its_time():
    # One linear segment 0 -> 1 over 1 s; the bins are unequal.
    h = histogram(_tran([0, 1], [0, 1]), bins=[0.0, 0.1, 0.4, 1.0])
    np.testing.assert_allclose(h.fractions, [0.1, 0.3, 0.6], rtol=1e-12)


def test_sample_crowding_on_edges_is_not_counted():
    # A square wave with many samples on its edges and few on its plateaus:
    # a sample count would put most weight mid-swing; time puts ~none there.
    edge = np.linspace(0, 1e-3, 200)
    t = np.r_[edge, 0.5 + edge, 1.0]
    y = np.r_[edge / 1e-3, 1 - edge / 1e-3, 0.0]
    h = histogram(_tran(t, y), bins=4)
    assert h.fractions[0] + h.fractions[-1] > 0.99
    assert h.fractions.sum() == pytest.approx(1.0, abs=1e-12)


def test_sine_follows_the_arcsine_distribution():
    t = np.linspace(0, 3.0, 30001)          # three full periods
    a = 1.0
    h = histogram(_tran(t, a * np.sin(2 * np.pi * t)), bins=20)
    expected = np.diff(np.arcsin(np.clip(h.edges / a, -1, 1)) / np.pi)
    np.testing.assert_allclose(h.fractions, expected, atol=2e-4)
    assert h.fractions.sum() == pytest.approx(1.0, abs=1e-12)


def test_window_bounds_are_interpolated():
    # Ramp 0..10 over 10 s; window [2.5, 7.5] holds values 2.5..7.5.
    t = np.arange(11.0)
    h = histogram(_tran(t, t), bins=[0, 5, 10], start=2.5, end=7.5)
    np.testing.assert_allclose(h.fractions, [0.5, 0.5])
    h = histogram(_tran(t, t), bins=5, start=2.5, end=7.5)
    np.testing.assert_allclose(h.edges, np.linspace(2.5, 7.5, 6))
    np.testing.assert_allclose(h.fractions, np.full(5, 0.2))


def test_constant_intervals_go_to_the_bin_holding_their_value():
    # 1 s at 0, 1 s ramp 0->1, 2 s at 1. Value 1 is the last edge (closed).
    h = histogram(_tran([0, 1, 2, 4], [0, 0, 1, 1]), bins=2)
    np.testing.assert_allclose(h.fractions, [1.5 / 4, 2.5 / 4])
    # A flat value on an interior edge belongs to the bin it starts.
    h = histogram(_tran([0, 1], [0.5, 0.5]), bins=[0, 0.5, 1])
    np.testing.assert_allclose(h.fractions, [0, 1])


def test_constant_signal_uses_numpy_range():
    h = histogram(_tran([0, 1, 2], [3, 3, 3]), bins=1)
    np.testing.assert_allclose(h.edges, [2.5, 3.5])
    np.testing.assert_allclose(h.fractions, [1.0])


def test_time_outside_explicit_edges_is_not_counted():
    h = histogram(_tran([0, 1], [0, 1]), bins=[0.0, 0.5])
    np.testing.assert_allclose(h.fractions, [0.5])


def test_nan_sample_gives_nan_fractions_with_warning():
    s = _tran([0, 1, 2], [0, np.nan, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        h = histogram(s, bins=3)
    assert h.edges.shape == (4,) and np.all(np.isnan(h.fractions))
    assert any(getattr(w.message, 'code', None) == 'nan_waveform'
               for w in caught)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = histogram(s, bins=[0, 1])
    np.testing.assert_array_equal(h.edges, [0, 1])
    assert np.isnan(h.fractions).all()


@pytest.mark.parametrize('bins', [0, -2, True, 'auto', [1.0], [0, 0, 1],
                                  [1, 0], [0, np.inf], [[0, 1]], 2.5])
def test_bad_bins_raise(bins):
    with pytest.raises(ValueError, match='histogram'):
        histogram(_tran([0, 1], [0, 1]), bins=bins)


def test_bad_window_and_axis_raise():
    s = _tran([0, 1], [0, 1])
    with pytest.raises(ValueError, match='histogram'):
        histogram(s, bins=2, start=5.0)
    with pytest.raises(ValueError, match='histogram'):
        histogram(s, bins=2, start=0.5, end=0.5)
    ac = Signal('v', 'ac', 'frequency', 'V', np.array([1.0, 2.0]),
                np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match='histogram'):
        histogram(ac, bins=2)


def test_not_registered_and_schema_unchanged():
    assert 'histogram' not in export_metric_schema()
    canonical = json.dumps(export_metric_schema(), sort_keys=True,
                           separators=(',', ':'), allow_nan=False)
    # Catalog digest after TASK-385, 377, 399, 378, 386, 396, 397, 380, 379 and 412;
    # histogram adds nothing to it.
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        # TASK-226.22: required scalar eye threshold; centering default True.
        'f47ca45cb4a0666deb6a955dfe495e17888bfa63308be96cb15e27966e65c040')
    assert M.histogram is histogram
