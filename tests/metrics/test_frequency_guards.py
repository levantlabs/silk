"""Frequency guard errors identify the public metric without changing policy."""
import math
import warnings

import numpy as np
import pytest

from silk.metrics import bandwidth_3dB, gain_bw_product, integrated_noise
from silk.metrics import unity_gain_freq, noise_density_at_freq
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal
from silk.signals.kinds import SignalKindError


@pytest.fixture(params=[
    (bandwidth_3dB, ()),
    (gain_bw_product, ()),
    (unity_gain_freq, ()),
    (integrated_noise, (1, 3)),
    (noise_density_at_freq, (2,)),
], ids=lambda case: case[0].__name__)
def frequency_metric(request):
    return request.param


def sweep(y, axis='frequency'):
    return Signal('response', 'voltage', axis, 'V',
                  np.arange(1, len(y) + 1, dtype=float), np.asarray(y))


@pytest.mark.parametrize('metric', [bandwidth_3dB, gain_bw_product, unity_gain_freq])
@pytest.mark.parametrize('y', [[], [np.nan], [1., np.nan]])
def test_wrong_axis_precedes_empty_short_and_nan_data(metric, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(SignalKindError) as error:
            metric(sweep(y, axis='time'))
    assert type(error.value) is SignalKindError
    assert str(error.value) == (
        f"{metric.__name__}: sig must have xlabel='frequency', got 'time'"
    )
    assert caught == []


def test_empty_frequency_trace_keeps_its_actionable_error(frequency_metric):
    metric, args = frequency_metric
    with pytest.raises(ValueError) as error:
        metric(sweep([]), *args)
    assert type(error.value) is ValueError
    assert str(error.value) == (
        f"{metric.__name__}: Signal 'response' has no samples (len(x) == 0); "
        'a metric cannot measure an empty trace'
    )


@pytest.mark.parametrize('sample', [1., np.nan])
def test_one_point_names_metric_and_counts_before_nan_data(frequency_metric, sample):
    metric, args = frequency_metric
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError) as error:
            metric(sweep([sample]), *args)
    assert type(error.value) is ValueError
    assert str(error.value) == (
        f'{metric.__name__}: need at least 2 frequency points, got 1'
    )
    assert caught == []


def test_nan_data_remains_an_unavailable_measurement(frequency_metric):
    metric, args = frequency_metric
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        value = metric(sweep([1., np.nan, .1]), *args)
    assert type(value) is float
    assert math.isnan(value)
