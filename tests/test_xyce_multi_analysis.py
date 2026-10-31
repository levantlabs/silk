"""TASK-220: refuse Xyce multi-analysis decks before writing or launching them."""

import os
from itertools import permutations
from pathlib import Path
import shutil

import numpy as np
import pytest

from silk import Module, load_pdk, sources
from silk.circuit.xyce_netlister import XyceNetlister
from silk.elements import elements
from silk.exceptions import SimulationError
from silk.simulation import AC, DC, Noise, Transient
from silk.simulation.xyce_runner import _xyce_executable


def _analyses():
    return {
        'tran': Transient(step=1e-8, stop=1e-5),
        'ac': AC('dec', 10, 1, 1e9),
        'dc': DC(source='vin', start=0, stop=1, step=0.1),
        'noise': Noise('vout', '0', 'vin', 'dec', 10, 1, 1e9),
    }


@pytest.mark.parametrize('names', list(permutations(('tran', 'ac', 'dc', 'noise'), 2)))
def test_every_analysis_pair_is_refused_at_output_emission(names, tmp_path):
    analyses = _analyses()
    requested = {name: analyses[name] for name in names}
    with pytest.raises(SimulationError, match='Xyce.*one analysis') as exc:
        XyceNetlister(None).generate_output_block(
            str(tmp_path), requested, ['vin', 'vout'])
    message = str(exc.value)
    for analysis in requested.values():
        assert type(analysis).__name__ in message
    assert 'split' in message.lower()
    assert 'ngspice' in message


def _rc_testbench(root, names):
    mod = Module('sky130', str(root), 'rc.cir', load_pdk('sky130'),
                 module_name='rc_lpf')
    mod.simulator = 'xyce'
    mod.define_module_ports({'vin': 'input', 'vout': 'output'})
    mod.add(elements.resistor('r1', p='vin', n='vout', value=1e3))
    mod.add(elements.capacitor('c1', p='vout', n='0', value=1e-9))
    tb = mod.create_test('rc')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add(sources.VoltageSource('vin', positive='vin', negative='0', dc=1, ac=1))
    analyses = _analyses()
    for name in names:
        tb.add_sim_type(analyses[name])
    return tb


@pytest.mark.parametrize('names', [('tran', 'ac'), ('ac', 'tran')])
def test_testbench_refuses_before_writing_a_multi_analysis_deck(names, tmp_path):
    tb = _rc_testbench(tmp_path, names)
    tb.netlist.simulator = 'xyce'
    with pytest.raises(SimulationError, match='Xyce.*one analysis'):
        tb.netlist.generate_netlist()
    assert not Path(os.path.join(tb.netlist.dir_folder, tb.netlist.netlistname)).exists()
    assert not tb.netlist.sim_results_loaded


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.parametrize('names', [('tran', 'ac'), ('ac', 'tran')])
def test_real_xyce_refusal_and_split_test_remedy(names, tmp_path):
    """The rejected request becomes two real, loadable Xyce runs when split."""
    if shutil.which(_xyce_executable()) is None:
        pytest.skip('Xyce unavailable; set SILK_XYCE_BIN or put Xyce on PATH')
    combined = _rc_testbench(tmp_path / 'combined', names)
    with pytest.raises(SimulationError, match='Xyce.*one analysis'):
        combined.run()
    assert not Path(os.path.join(combined.netlist.dir_folder, combined.netlist.netlistname)).exists()
    assert not combined.netlist.sim_results_loaded

    for name, axis in [('tran', 'time'), ('ac', 'frequency')]:
        split = _rc_testbench(tmp_path / name, [name])
        split.run()
        assert split.netlist.sim_results_loaded
        signal = split.netlist.load_signal('vout', analysis=name)
        assert signal.xlabel == axis
        assert len(signal.y) > 1
        assert np.all(np.isfinite(signal.y))
        assert signal.y[0] == pytest.approx(1, abs=1e-4)
