"""TASK-520: designer branch paths, physical PWL tokens and ammeter paths."""
import os
import numpy as np
import pytest

from silk import sources, load_pdk
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk.elements import elements
from silk.simulation import Transient
from tests._env import needs_ngspice, needs_pdk
from tests.test_metric_branch_current_paths import _build, _avg


@needs_ngspice
def test_arrayed_designer_probe_and_metric_read_same_branch(tmp_path):
    tb = _build(tmp_path, [('vs[0]', 'vin', 1.2)], [_avg('current', 'i(vs[0])')])
    probe = tb.probe('i(vs[0])')
    report = tb.run('tt')
    assert report.metrics['current'] == pytest.approx(-.0006)
    np.testing.assert_allclose(probe.y, -.0006)
    raw = tb.netlist.load_signal('vvs<0>', 'I')
    np.testing.assert_array_equal(probe.y, raw.y)


@needs_ngspice
def test_raw_bracket_branch_spelling_is_an_alias_of_the_emitted_vector(tmp_path):
    """TASK-527 replaces the TASK-520 no-alias contract: i(vvs[0]) is the
    bracket spelling of the emitted raw vector i(vvs<0>)."""
    tb = _build(tmp_path, [('vs[0]', 'vin', 1.2)], [_avg('old', 'i(vvs[0])')])
    report = tb.run('tt')
    assert report.metrics['old'] == pytest.approx(-.0006)
    np.testing.assert_array_equal(tb.probe('i(vvs[0])').y, tb.probe('i(vs[0])').y)


@needs_ngspice
def test_pwl_private_net_and_both_physical_branches_roundtrip(tmp_path):
    top = Module(None, str(tmp_path), 'pwl.cir', None)
    top.add_source(sources.VoltageSource('bias[0]', positive='out[0]', negative='0',
                                        dc=.2, pwl=[(0, 0), (1e-9, 1)]))
    top.add(elements.resistor('R[0]', p='out[0]', n='0', value=1000))
    top.add_sim_type(Transient(step=.1e-9, stop=2e-9))
    deck_path = top.generate_netlist()
    top.run_simulator()
    y = top.load_signal('out[0]').y
    assert y[-1] == pytest.approx(1.2)
    np.testing.assert_allclose(top.load_signal('__silk_src_Vbias[0]').y, .2)
    for name in ('Vbias<0>__silk_pwl', 'Vbias<0>__silk_dc'):
        np.testing.assert_allclose(top.load_signal(name, 'I').y, -y/1000)
    assert 'out[0]_pdc_Vbias<0>' not in open(deck_path).read()


@needs_ngspice
@needs_pdk
def test_arrayed_ammeter_path_matches_plain(tmp_path):
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk, module_name='dut')
    dut.define_module_ports({'d': PortDir.OUT, 's': PortDir.GROUND})
    dut.add(pdk.nmos1p8('M[0]', d='d', g='d', s='s', b='s', w=1, l=.15))
    top = Module('sky130', str(tmp_path), 'top.cir', pdk)
    top.add_module(dut, 'X[0]', nets={'d': 'd', 's': '0'})
    top.add_module(dut, 'plain', nets={'d': 'd', 's': '0'})
    top.add_source(sources.VoltageSource('bias', positive='d', negative='0', dc=1))
    top.add_sim_type(Transient(step=1e-9, stop=3e-9))
    top.save_device_currents = True
    top.terminal_current_method = 'ammeter_wrapper'
    top.simulate()
    angled = top.load_signal('v.xx<0>.xmm<0>.vamm_d', 'I')
    plain = top.load_signal('v.xplain.xmm<0>.vamm_d', 'I')
    assert np.all(angled.y > 0)
    np.testing.assert_array_equal(angled.y, plain.y)
    report = top.annotate_currents()
    assert report.pins_annotated > 0
