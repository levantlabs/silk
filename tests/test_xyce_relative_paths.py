"""TASK-445: Xyce runs with cwd=results_dir, so every path in the deck and on
its command line must be absolute. A relative ``dir_folder`` used to emit
``.print ... file=<relative>/rawfile.raw`` that Xyce resolved against the
results directory itself and failed to open."""
import os
import shutil

import pytest

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.simulation import Transient
from silk.simulation.xyce_runner import _xyce_executable

_have_xyce = bool(os.environ.get('SILK_XYCE_BIN') or shutil.which('Xyce'))


def _bench(dir_folder):
    mod = Module('sky130', dir_folder, 'rc.cir', load_pdk('sky130'),
                 module_name='rc')
    mod.simulator = 'xyce'
    mod.define_module_ports({'vin': 'input', 'vout': 'output'})
    mod.add(elements.resistor('r1', p='vin', n='vout', value=1e3))
    mod.add(elements.capacitor('c1', p='vout', n='0', value=1e-9))
    tb = mod.create_test('rc')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add(sources.VoltageSource('vin', positive='vin', negative='0', dc=1))
    tb.add_sim_type(Transient(step=1e-7, stop=1e-5))
    return mod, tb


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not _have_xyce, reason='Xyce unavailable; set $SILK_XYCE_BIN')
def test_relative_directory_runs_on_real_xyce(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod, tb = _bench('rel_out')
    probe = tb.probe('vout')
    mod.run_all_tests()
    assert float(probe.y[-1]) == pytest.approx(1.0)
    deck = next((tmp_path / 'rel_out').rglob('*.cir')).read_text()
    printed = [l for l in deck.splitlines() if l.startswith('.print')]
    assert printed and all(os.path.isabs(l.split('file=')[1].split()[0])
                           for l in printed)
