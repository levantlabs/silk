"""TASK-379: variance is stddev squared on the shared core."""
import numpy as np
import pytest

import silk.metrics as M
from silk.signals import Signal
from silk.signals.schema import export_metric_schema

# Deliberately non-uniform grid.
X = np.array([0.0, 0.1, 0.5, 0.6, 2.0, 2.1, 5.0])
Y = np.array([0.0, 1.0, 3.0, 2.0, -1.0, 4.0, 1.5])


def _sig():
    return Signal('v', 'voltage', 'time', 'V', X, Y)


@pytest.mark.parametrize('kw', [{}, {'start': 0.3, 'end': 3.7},
                                {'start': 0.3, 'end': 3.7, 'interpolate': False}])
def test_variance_is_stddev_squared(kw):
    s = _sig()
    assert M.variance(s, **kw) == pytest.approx(M.stddev(s, **kw) ** 2, rel=1e-12)
    assert M.variance(s, **kw) > 0


def test_stddev_rms_mean_identity():
    s = _sig()
    assert M.variance(s) == pytest.approx(M.rms(s) ** 2 - M.average(s) ** 2, rel=1e-9)


def test_registered_with_squared_unit():
    entry = export_metric_schema()['variance']
    assert entry['unit'] == '[y]^2'
    assert [p['name'] for p in entry['parameters']] == ['start', 'end', 'interpolate']
