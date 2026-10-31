"""Physical references and transform identities from the RLC step recipe."""
import math

import numpy as np
import pytest

from silk.examples.metrics import rlc_step as rlc
from silk.metrics import (argmax_x, convolve, cross_correlation, derivative,
                          flip, lshift, running_integral, sample, window)
from silk.signals.base import Signal
from tests._env import needs_ngspice


@pytest.mark.slow
@needs_ngspice
def test_rlc_step_physical_references_and_transform_identities(tmp_path):
    mod = rlc.build_rlc(str(tmp_path))
    mod.simulate()
    vout, vin = mod.load_signal('vout'), mod.load_signal('vin')
    core = rlc.measure_core(vout, rlc.ideal_step(vout.x))
    mp = math.exp(-rlc.ZETA * math.pi / math.sqrt(1 - rlc.ZETA ** 2))
    envelope_ns = math.log(1 / (rlc.TOL * math.sqrt(1 - rlc.ZETA ** 2))) / (rlc.ZETA * rlc.WN) * 1e9
    assert core['overshoot'] == pytest.approx(mp, rel=0.01)
    assert core['undershoot'] == pytest.approx(mp ** 2, rel=0.01)
    # The envelope crossing bounds the grid-snapped last actual band exit.
    assert 0.85 * envelope_ns < core['settling_time_ns'] < 1.01 * envelope_ns
    assert core['error_norm_rms_mV'] < 5
    derived = rlc.measure_derived(vout, vin)
    assert derived['charge_delivered_nC'] == pytest.approx(rlc.C_F * rlc.V0 * 1e9, rel=0.001)
    assert derived['edge_to_peak_delay_ns'] == pytest.approx(rlc.T_PEAK * 1e9, rel=0.02)
    assert derived['ring_period_ns'] == pytest.approx(rlc.T_RING * 1e9, rel=0.05)

    dv = derivative(vout)
    current = Signal('i', 'current', 'time', 'A', dv.x, rlc.C_F * dv.y)
    charge = running_integral(current)
    np.testing.assert_allclose(charge.y, rlc.C_F * (vout.y - vout.y[0]), atol=2e-13)
    ring_win = window(vout, rlc.TD_S + 0.5 * rlc.T_PEAK,
                      rlc.TD_S + 0.5 * rlc.T_PEAK + 2 * rlc.T_RING)
    ring = sample(ring_win, n=4 * len(ring_win.x))
    ring0 = Signal(ring.name, ring.type, ring.xlabel, ring.ylabel,
                   ring.x - ring.x[0], ring.y - np.mean(ring.y))
    composed = convolve(ring0, flip(ring0))
    direct = cross_correlation(ring0, ring0)
    np.testing.assert_allclose(composed.y, direct.y * (ring0.x[1] - ring0.x[0]), rtol=1e-12, atol=1e-22)
    # Convolution starts at summed origins; correlation starts at negative lag.
    np.testing.assert_allclose(composed.x - composed.x[len(composed.x) // 2],
                               direct.x, rtol=1e-12, atol=1e-21)
    assert argmax_x(direct, interpolate=False) == pytest.approx(0, abs=1e-21)
    shifted = lshift(ring, float(ring.x[0]))
    np.testing.assert_allclose(shifted.x, ring0.x, rtol=0, atol=1e-21)
    np.testing.assert_array_equal(shifted.y, ring.y)
