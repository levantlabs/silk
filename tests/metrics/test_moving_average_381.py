"""moving_average (TASK-381): analytic ramp / square-wave checks."""
import numpy as np
import pytest

from silk.signals.base import Signal
from silk.metrics import moving_average


def _sig(t, y):
    return Signal('v', 'tran', 'time', 'V', np.asarray(t, float), np.asarray(y, float))


def test_ramp_lags_by_half_window():
    t = np.linspace(0, 10, 1001)
    out = moving_average(_sig(t, 2 * t), window=1.0)
    ok = t >= 1.0
    assert np.allclose(out.y[ok], 2 * (t[ok] - 0.5))
    assert np.isnan(out.y[t < 1 - 1e-9]).all()
    assert np.array_equal(out.x, t)


def test_square_wave_triangle():
    t = np.linspace(0, 4, 4001)
    y = ((t % 2) < 1).astype(float)  # 1 on [0,1), 0 on [1,2) ...
    out = moving_average(_sig(t, y), window=2.0)
    ok = t >= 2.0
    assert np.allclose(out.y[ok], 0.5, atol=2e-3)  # full period averages to duty


def test_first_window_is_nan():
    t = np.linspace(0, 1, 101)
    out = moving_average(_sig(t, np.ones_like(t)), window=0.5)
    assert np.isnan(out.y[t < 0.5 - 1e-9]).all()
    assert np.allclose(out.y[t >= 0.5], 1.0)


@pytest.mark.parametrize('w', [0, -1.0, float('nan'), float('inf')])
def test_bad_window(w):
    with pytest.raises(ValueError):
        moving_average(_sig([0, 1, 2], [0, 1, 2]), window=w)


def test_unregistered_schema_unchanged():
    from silk.signals.schema import export_metric_schema
    import json
    # TASK-417: the string 'moving_average' is now a permitted value of the
    # jitter metrics' ``reference`` param, so check the registry keys.
    assert 'moving_average' not in export_metric_schema()
    assert '"moving_average":' not in json.dumps(export_metric_schema())


def test_fractional_window_integrates_cut_segment():
    out = moving_average(_sig([0., 1., 2.], [0., 2., 4.]), window=0.5)
    assert np.isnan(out.y[0])
    np.testing.assert_allclose(out.y[1:], [1.5, 3.5])
