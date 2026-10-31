"""Generator for the ``e_bus_array_*.raw`` fixtures (TASK-608).

REAL output of ngspice-42 (``_ng``), Xyce 7.10 (``_xy``) and the behavioral
engine (``_bh``) for a PDK-free bench holding a 2-bit bus net ``d[1:0]`` and an
arrayed copy ``amp1[k]`` whose copies hold different values, so identity can
be checked numerically. They record which vector names each producer writes
for a bus bit / array-copy net: always the wrapped form (``v(d[0])`` ngspice,
``V(D[0])`` / ``V(XDUT:XAMP1<0>:MID)`` Xyce), never a bare ``d[0]``. The
SPICE benches request the copies' internal ``mid`` nets with a save list.
Re-run to regenerate (from the repo root)::

    .venv/bin/python tests/fixtures/rawfiles/gen_e_bus.py
"""
import shutil
import tempfile
from pathlib import Path

from silk import Module, sources
from silk.elements import elements
from silk.simulation import Transient

OUT = Path(__file__).resolve().parent


def build(simulator, tmp):
    """Build in the directory *tmp*. vout = 2 V; amp1[0].mid = 1.0 V (r2 = 1k), amp1[1].mid = 1.5 V (r2 = 3k)."""
    dut = Module(None, tmp, 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'out': 'output'})
    for k, r2 in enumerate((1000, 3000)):
        leaf = Module(None, tmp, f'leaf{k}.cir', None, module_name=f'leaf{k}')
        leaf.define_module_ports({'out': 'output'})
        leaf.add(elements.resistor('r1', p='out', n='mid', value=1000))
        leaf.add(elements.resistor('r2', p='mid', n='0', value=r2))
        dut.add_module(leaf, f'amp1[{k}]', {'out': 'out'})
    tb = dut.create_test('v')
    tb.add_module_under_test({'out': 'vout'})
    tb.netlist.add_net('d', bits=2)
    for k in range(2):
        tb.add_source(sources.VoltageSource(f'sd{k}', positive=f'd[{k}]', negative='0', dc=1 + k))
    tb.add_source(sources.VoltageSource('supply', positive='vout', negative='0', dc=2))
    tb.add_sim_type(Transient(step=1e-6, stop=3e-6))
    tb.netlist.simulator = simulator
    tb.netlist.set_save_list(['vout', 'd[0]', 'd[1]', 'dut.amp1[0].mid', 'dut.amp1[1].mid'])
    return tb


def build_behavioral(tmp):
    """Build in the directory *tmp*. d[k] = 1 + k V; copy amp1[k] scales its input by 2 (k=0) or 3 (k=1) into mid."""
    dut = Module(None, tmp, 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'a': 'input', 'b': 'input'})
    for k, (gain, pin) in enumerate(((2.0, 'a'), (3.0, 'b'))):
        scale = Module(None, tmp, f'scale{k}.cir', None, module_name=f'scale{k}')
        scale.define_module_ports({'vin': 'input', 'vout': 'output'})
        def scale_model(ports, t, params):
            ports.vout.v = params.gain * ports.vin.v
        scale.set_behavioral_model(scale_model, params={'gain': gain}, execution='vector')
        unity = Module(None, tmp, f'unity{k}.cir', None, module_name=f'unity{k}')
        unity.define_module_ports({'vin': 'input', 'vout': 'output'})
        def unity_model(ports, t, params):
            ports.vout.v = ports.vin.v
        unity.set_behavioral_model(unity_model, execution='vector')
        cell = Module(None, tmp, f'cell{k}.cir', None, module_name=f'cell{k}')
        cell.define_module_ports({'vin': 'input', 'vout': 'output'})
        cell.add_module(scale, 's', nets={'vin': 'vin', 'vout': 'mid'})
        cell.add_module(unity, 'u', nets={'vin': 'mid', 'vout': 'vout'})
        dut.add_module(cell, f'amp1[{k}]', nets={'vin': pin, 'vout': f'o{k}'})
    tb = dut.create_test('v')
    tb.add_module_under_test({'a': 'd[0]', 'b': 'd[1]'})
    for k in range(2):
        tb.add_source(sources.VoltageSource(f'sd{k}', positive=f'd[{k}]', negative='0', dc=1 + k))
    tb.add_sim_type(Transient(step=1e-6, stop=3e-6))
    tb.netlist.simulator = 'behavioral'
    return tb


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        for simulator, name in (('ngspice', 'e_bus_array_ng.raw'), ('xyce', 'e_bus_array_xy.raw')):
            tb = build(simulator, tmp)
            tb.run()
            shutil.copy(tb.netlist.results_file, OUT / name)
        tb = build_behavioral(tmp)
        tb.run()
        shutil.copy(tb.netlist.results_file, OUT / 'e_bus_array_bh.raw')
