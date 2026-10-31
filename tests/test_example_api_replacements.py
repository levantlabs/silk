"""Shared-grid decoding and monotonic transfer-curve contracts for examples."""
from types import SimpleNamespace

import numpy as np
import pytest

from silk.examples import flash_adc_example as adc
from silk.examples.metrics.comparator_and_sampler import peak_gain
from silk.signals import Signal


def _signal(name, y, x=None):
    return Signal(name, 'voltage', 'time', 'V',
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x),
                  np.asarray(y, dtype=float))


@pytest.mark.parametrize('reverse', [False, True])
def test_peak_gain_preserves_monotonic_transfer_orientation(reverse):
    vin, vout = np.array([0., 1., 2., 4.]), np.array([0., 1., 4., 8.])
    if reverse:
        vin, vout = vin[::-1], vout[::-1]
    assert peak_gain(_signal('in', vin), _signal('out', vout)) == pytest.approx(8 / 3)


@pytest.mark.parametrize('vin', [[0, 1, 0, 2], [0, 1, 1, 2]])
def test_peak_gain_refuses_nonmonotonic_or_repeated_input(vin):
    with pytest.raises(ValueError, match='monotonic'):
        peak_gain(_signal('in', vin), _signal('out', [0, 1, 4, 8]))


def test_thermometer_trace_counts_bits_and_checks_input_grid(monkeypatch):
    monkeypatch.setattr(adc, 'N_COMP', 3)
    traces = {
        'vin': _signal('vin', [0, .2, .4, .2]),
        'vcmp1': _signal('vcmp1', [0, 1.8, 1.8, 1.8]),
        'vcmp2': _signal('vcmp2', [0, 0, 1.8, 0]),
        'vcmp3': _signal('vcmp3', [0, 0, 0, 0]),
    }
    ckt = SimpleNamespace(load_signal=traces.__getitem__)
    code = adc._code_trace(ckt)
    np.testing.assert_array_equal(code.y, [0, 1, 2, 1])
    np.testing.assert_array_equal(code.x, traces['vin'].x)
    traces['vin'] = _signal('vin', [0, .2, .4, .2], [0, 1, 2, 4])
    with pytest.raises(ValueError, match='vin.*grid'):
        adc._code_trace(ckt)
    traces['vin'] = _signal('vin', [0, .2, .4, .2])
    traces['vcmp2'] = _signal('vcmp2', [0, 0, 1.8, 0], [0, 1, 2, 4])
    with pytest.raises(ValueError, match='axis|grid'):
        adc._code_trace(ckt)
