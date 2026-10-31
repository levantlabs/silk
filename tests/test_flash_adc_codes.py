"""Simulator-free regressions for the flash ADC example's static model."""

import numpy as np
import pytest

from silk.examples.metrics.flash_adc_codes import static_model_code
from silk.metrics import extract_code_transitions
from silk.signals import Signal


@pytest.mark.parametrize('measured_codes', [
    [0, 0, 1, 1, 3, 3, 4],
    [2, 2, 3, 3, 5, 5, 6],
])
def test_static_model_reproduces_measured_code_numbering(measured_codes):
    # The skipped code gives coincident transitions; equality must select
    # the code reached at the transition, including both coincident levels.
    time = np.arange(len(measured_codes), dtype=float)
    vin = Signal('vin', 'voltage', 'time', 'V', time, time / 10)
    measured = Signal('code', 'count', 'time', 'code', time, measured_codes)
    transitions = extract_code_transitions(vin, measured)

    model = static_model_code(vin, transitions)

    np.testing.assert_array_equal(model.y, measured_codes)
    np.testing.assert_array_equal(model.x, time)
