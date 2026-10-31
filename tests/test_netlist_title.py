"""Generated deck titles identify runs in real simulator rawfiles."""

from pathlib import Path
import shutil

import numpy as np
import pytest

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.simulation import Transient
from silk.simulation.xyce_runner import _xyce_executable
from tests._env import needs_ngspice


def _circuit(tmp_path, simulator='ngspice', name=None):
    mod = Module('sky130', str(tmp_path), 'lna_bias.cir', load_pdk('sky130'))
    mod.circuit_name = name
    mod.simulator = simulator
    mod.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1))
    mod.add(elements.resistor('load', p='out', n='0', value=1000))
    mod.add_sim_type(Transient(step=1e-7, stop=1e-6))
    return mod


@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('name', [None, ''])
def test_unnamed_module_title_uses_netlist_filename(tmp_path, simulator, name):
    mod = _circuit(tmp_path, simulator, name)
    for _ in range(2):
        mod_deck_path = mod.generate_netlist()
        assert Path(mod_deck_path).read_text().splitlines()[0] == (
            '*** Circuit: lna_bias.cir')


@pytest.mark.slow
@pytest.mark.parametrize('simulator', [
    pytest.param('ngspice', marks=needs_ngspice),
    pytest.param('xyce', marks=[
        pytest.mark.requires_xyce,
        pytest.mark.skipif(shutil.which(_xyce_executable()) is None,
                           reason='needs Xyce ($SILK_XYCE_BIN or Xyce on PATH)'),
    ]),
])
@pytest.mark.parametrize('name, title_name', [
    ('my_lna', 'my_lna'),
    ('My LNA (rev. 2): gain=20dB / in-out; $bias #1',
     'My LNA (rev. 2): gain=20dB / in-out; $bias #1'),
    (None, 'lna_bias.cir'),
])
def test_real_rawfile_title(tmp_path, simulator, name, title_name):
    mod = _circuit(tmp_path, simulator, name)
    mod.simulate()
    # Read the simulator's bytes, independently of silk's in-memory metadata.
    with Path(mod.results_file).open('rb') as raw:
        title = raw.readline().decode('ascii').rstrip('\r\n')
    expected = f'*** Circuit: {title_name}'
    # ngspice lowercases deck titles; Xyce preserves their spelling.
    if simulator == 'ngspice':
        expected = expected.lower()
    assert title == f'Title: {expected}'
    np.testing.assert_allclose(mod.load_signal('out').y, 1)
