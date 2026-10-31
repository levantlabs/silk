"""Real ModuleTestbench → Optimizer regression, including device linkage."""

import math
from pathlib import Path

import pytest

from silk import Module, ModuleTestbench, Variable, Port, load_pdk, sources
from silk.optimize import Optimizer, VariableSpace
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk


@pytest.mark.slow
@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('mode', ['overwrite', 'new'])
def test_optimizer_measures_two_linked_widths(tmp_path, mode):
    pdk = load_pdk('sky130')
    width = Variable('w_n', value=2.0, bounds=(1.0, 8.0))
    dut = Module('PDK_skywater130', str(tmp_path), 'nmos.cir',
                 pdk_details=pdk, module_name='nmos')
    dut.define_module_ports({
        'D': Port.analog_output(), 'G': Port.analog_input(),
        'S': Port.ground(),
    })
    with dut:
        pdk.nmos1p8('m1', d='D', g='G', s='S', b='S', w=width, l=0.5)
    tb = dut.create_test('current')
    tb.add_module_under_test({'D': 'drain', 'G': 'gate', 'S': '0'})
    # Embedding copies the definition used for emission; link the variable
    # to that copy so VariableSpace.apply changes the device that SPICE simulates.
    emitted_dut = tb.netlist.module_dict['nmos']['device']
    emitted_dut.update_element('m1', ['w'], [width])
    tb.add_source(sources.VoltageSource(
        'drain', positive='drain', negative='0', dc=0.8))
    tb.add_source(sources.VoltageSource(
        'gate', positive='gate', negative='0', dc=1.0))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    tb.add_metric('drain_current_A',
                  lambda ckt: -float(ckt.load_signal('vdrain', 'I').y[-1]))
    tb.add_spec('conducts', 'drain_current_A', '>', 0.0)

    dut.set_output_options(mode=mode)
    paths = []
    opt = Optimizer(tb, VariableSpace([width]))
    opt.add_objective('drain_current_A', maximize=True)
    opt.add_constraint('drain_current_A', min=0.0)
    assert isinstance(opt.test, ModuleTestbench)
    records = []
    for value in (2.0, 4.0):
        rec = opt.evaluate({'w_n': value}, corner='tt')
        assert width.value == value
        assert rec['status'] == 'success', rec
        assert rec['feasible'] is True
        assert rec['error'] is rec['error_type'] is None
        assert rec['params'] == {'w_n': value}
        assert rec['corner'] == 'tt'
        current = rec['metrics']['drain_current_A']
        assert math.isfinite(current) and current > 0
        records.append(rec)
        paths.append(tb.results_file)

    # A large physical change proves the live width reaches SPICE and the
    # second evaluation reads fresh measurements, without a model golden.
    assert (records[1]['metrics']['drain_current_A'] >
            1.5 * records[0]['metrics']['drain_current_A'])
    assert [rec['id'] for rec in records] == [0, 1]
    assert opt.history == opt.successes == records
    assert opt.failures == []
    assert opt.best is records[1]

    assert (paths[0] != paths[1]) == (mode == 'new')
    assert all(Path(path).is_file() for path in paths)
