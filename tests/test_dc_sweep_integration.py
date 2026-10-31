"""End-to-end: a real ngspice DC sweep must be readable via load_signal.

Guards against the DC-sweep readback bug (the sweep axis is named neither
'time' nor 'frequency', so a name-based x-axis whitelist made every .dc result
unreadable). The fix treats rawfile vector 0 as the independent axis, which
holds for ngspice ('temp-sweep' / 'v(v-sweep)') and Xyce ('sweep') alike — both
verified manually; this test pins the ngspice path in CI.
"""
import shutil
import numpy as np
import pytest
from silk.simulation.backend import SimulatorBackend
from silk.simulation.runner import ngspice_executable

pytestmark = pytest.mark.slow

ngspice_missing = shutil.which(ngspice_executable()) is None


@pytest.mark.skipif(ngspice_missing, reason="ngspice not installed")
def test_dc_voltage_sweep_readback(tmp_path):
    """.dc vin 0 1 0.1 through a divider: x is the swept voltage, y = vin/2."""
    deck = tmp_path / 'dcv.cir'
    deck.write_text(
        "* dc voltage sweep of a divider\n"
        ".dc vin 0 1 0.1\n"
        "vin 1 0 0\n"
        "r1 1 2 1k\n"
        "r2 2 0 1k\n"
        ".end\n"
    )
    backend = SimulatorBackend.get('ngspice')
    raw = backend.run(str(deck), str(tmp_path))
    sig = backend.load_signal(raw, '2', type='V')     # node 2 = divider output
    assert len(sig.x) == 11
    np.testing.assert_allclose([sig.x[0], sig.x[-1]], [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(sig.y, sig.x / 2.0, atol=1e-6)   # divider


@pytest.mark.skipif(ngspice_missing, reason="ngspice not installed")
def test_dc_temp_sweep_readback(tmp_path):
    """.dc temp -40 125 5: x is the swept temperature (bare 'temp-sweep' axis)."""
    deck = tmp_path / 'dct.cir'
    deck.write_text(
        "* dc temperature sweep\n"
        ".dc temp -40 125 5\n"
        "vin 1 0 1\n"
        "r1 1 2 1k\n"
        "r2 2 0 1k\n"
        ".end\n"
    )
    backend = SimulatorBackend.get('ngspice')
    raw = backend.run(str(deck), str(tmp_path))
    sig = backend.load_signal(raw, '2', type='V')
    assert len(sig.x) == 34
    np.testing.assert_allclose([sig.x[0], sig.x[-1]], [-40.0, 125.0], atol=1e-6)
