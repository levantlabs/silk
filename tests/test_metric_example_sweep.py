"""The amplitude sweep publishes its stimulus and preserves compression."""
import re

import numpy as np
import pytest

from silk.examples.metrics import cs_amp_power_sweep as example
from silk.metrics import p1db
from silk.utils import parse_spice_value
from tests._env import needs_ngspice, needs_pdk

pytestmark = [pytest.mark.slow, needs_ngspice, needs_pdk]


def test_sweep_stimulus_and_compression_reference(tmp_path):
    mod, tb, amplitude = example.build_cs_amp(str(tmp_path))
    points = example.sweep(mod, tb, amplitude)
    assert len(points) == example.N_POINTS
    expected = np.linspace(example.AMPL_MIN_V, example.AMPL_MAX_V, example.N_POINTS)
    np.testing.assert_allclose([p.vampl_v for p in points], expected)
    for point in points:
        assert 'Variable(' not in point.sine_line
        values = re.search(r'SIN\(([^)]+)\)', point.sine_line).group(1).split()
        assert parse_spice_value(values[1]) == pytest.approx(point.vampl_v)
        assert point.a_in_v == pytest.approx(point.vampl_v, rel=1e-3)
    assert points[0].sine_line != points[-1].sine_line
    m = example.measure(points)
    assert np.isfinite(m['p1db_dbm'])
    load_referenced = example.power_sweep_signal(points, z0_out_ohm=example.RD_OHM)
    assert m['p1db_dbm'] == pytest.approx(p1db(load_referenced), abs=1e-10)
    gain = m['sweep'].y - m['sweep'].x
    assert abs(gain[0] - gain[1]) < 0.02
    assert gain[0] - gain[-1] > 1
