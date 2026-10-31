"""TASK-527: probes and load_signal accept the bracket spelling of a raw path;
i(<name>) considers designer and normalised-emitted candidates."""
import os
import shutil

import numpy as np
import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.elements import elements
from silk.exceptions import AmbiguousBranchCurrentError
from silk.simulation import Transient
from tests._env import HAVE_XYCE, needs_ngspice
from tests.test_metric_branch_current_paths import _build, _avg


def test_branch_vector_name_designer_and_emitted_candidates():
    from silk.circuit.module_test import _branch_vector_name
    src = lambda n: sources.VoltageSource(n, positive='a', negative='0', dc=1)
    one = {'vs[0]': src('vs[0]')}
    assert _branch_vector_name(one, 'vs[0]') == 'Vvs<0>'
    # a raw name passes through with its ORIGINAL spelling; the one collector
    # considers it together with its simulator spelling (code review P2-1)
    assert _branch_vector_name(one, 'vvs[0]') == 'vvs[0]'
    assert _branch_vector_name(one, 'vvs<0>') == 'vvs<0>'


def test_vs0_beside_vvs0_is_ambiguous_and_v_vv_unchanged():
    from silk.circuit.module_test import _branch_vector_name
    src = lambda n: sources.VoltageSource(n, positive='a', negative='0', dc=1)
    both = {'vs[0]': src('vs[0]'), 'vvs[0]': src('vvs[0]')}
    with pytest.raises(AmbiguousBranchCurrentError):
        _branch_vector_name(both, 'vvs[0]')
    two = {'v': src('v'), 'vv': src('vv')}
    with pytest.raises(AmbiguousBranchCurrentError):
        _branch_vector_name(two, 'vv')


def test_same_entity_reached_two_ways_is_deduplicated():
    from silk.circuit.module_test import _branch_vector_name
    src = sources.VoltageSource('a', positive='n', negative='0', dc=1)
    # i(Va) names source 'a' by emitted instance and no source 'Va' exists
    assert _branch_vector_name({'a': src}, 'Va').lower() == 'va'


@needs_ngspice
def test_top_level_source_current_bracket_and_angle_spellings_agree(tmp_path):
    top = Module(None, str(tmp_path), 'n.cir', None)
    top.add_source(sources.VoltageSource('bias[0]', positive='out[0]', negative='0', dc=1))
    top.add(elements.resistor('R[0]', p='out[0]', n='0', value=1000))
    top.add_sim_type(Transient(step=1e-9, stop=2e-9))
    top.simulate()
    np.testing.assert_allclose(top.load_signal('out[0]').y[-1], 1.0)   # the net keeps its brackets
    a = top.load_signal('Vbias[0]', 'I')
    b = top.load_signal('Vbias<0>', 'I')
    c = top.load_signal('vbias[0]', 'I')          # case-insensitive as today
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.y, c.y)


@needs_ngspice
def test_raw_current_escape_hatch_and_alias_same_vector(tmp_path):
    tb = _build(tmp_path, [('vs[0]', 'vin', 1.2)], [_avg('cur', 'i(vs[0])')])
    tb.run('tt')
    raw = tb.probe('vvs<0>', 'I')
    alias = tb.probe('vvs[0]', 'I')
    designer = tb.probe('i(vs[0])')
    np.testing.assert_array_equal(raw.y, alias.y)
    np.testing.assert_array_equal(raw.y, designer.y)


@needs_ngspice
def test_pwl_two_aliases_and_reserved_private_net_preserves_brackets(tmp_path):
    top = Module(None, str(tmp_path), 'pwl.cir', None)
    top.add_source(sources.VoltageSource('bias[0]', positive='out[0]', negative='0',
                                        dc=.2, pwl=[(0, 0), (1e-9, 1)]))
    top.add(elements.resistor('R[0]', p='out[0]', n='0', value=1000))
    top.add_sim_type(Transient(step=.1e-9, stop=2e-9))
    deck_path = top.generate_netlist()
    top.run_simulator()
    pwl = top.load_signal('Vbias[0]__silk_pwl', 'I')
    dc = top.load_signal('Vbias[0]__silk_dc', 'I')
    np.testing.assert_array_equal(pwl.y, top.load_signal('Vbias<0>__silk_pwl', 'I').y)
    np.testing.assert_array_equal(dc.y, top.load_signal('Vbias<0>__silk_dc', 'I').y)
    assert not np.array_equal(pwl.y, dc.y) or pwl is not dc
    np.testing.assert_allclose(top.load_signal('__silk_src_Vbias[0]').y, .2)
    assert '__silk_src_Vbias<0>' not in open(deck_path).read()


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='requires Xyce (SILK_XYCE_BIN setting)')
def test_xyce_colon_upper_case_resolves_bracket_request(tmp_path):
    """Xyce prints only listed vectors: the branch current is printed because a
    testbench probe asks for it (TASK-346), so this is a testbench."""
    from pathlib import Path
    from silk.circuit.circuit_elements import PortDir
    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'p': PortDir.IN, 'g': PortDir.GROUND})
    dut.add(elements.resistor('R[0]', p='p', n='g', value=1000))
    tb = dut.create_test('xy')
    tb.add_module_under_test({'p': 'out[0]', 'g': '0'})
    tb.add_source(sources.VoltageSource('bias[0]', positive='out[0]', negative='0', dc=1))
    tb.add_sim_type(Transient(step=1e-9, stop=2e-9))
    tb.netlist.simulator = 'xyce'
    probe = tb.probe('i(bias[0])')
    report = tb.run()
    deck = Path(report.run.deck_path).read_text()
    assert 'I(VBIAS<0>)' in deck
    np.testing.assert_array_equal(tb.netlist.load_signal('Vbias[0]', 'I').y,
                                  tb.netlist.load_signal('Vbias<0>', 'I').y)
    np.testing.assert_array_equal(probe.y, tb.netlist.load_signal('Vbias<0>', 'I').y)


def _nested_top(tmp_path, simulator=None):
    from silk.circuit.circuit_elements import PortDir
    inner = Module(None, str(tmp_path), 'inner.cir', None, module_name='inner')
    inner.define_module_ports({'a': PortDir.IN, 'z': PortDir.GROUND})
    inner.add(elements.resistor('r1', p='a', n='g[0]', value=1000))
    inner.add(elements.resistor('r2', p='g[0]', n='z', value=1000))
    mid = Module(None, str(tmp_path), 'mid.cir', None, module_name='mid')
    mid.define_module_ports({'a': PortDir.IN, 'z': PortDir.GROUND})
    mid.add_module(inner, 'y[1]', nets={'a': 'a', 'z': 'z'})
    top = Module(None, str(tmp_path), 'top.cir', None)
    if simulator:
        top.simulator = simulator
    top.add_source(sources.VoltageSource('vin', positive='in', negative='0', dc=1))
    top.add_module(mid, 'x1[0]', nets={'a': 'in', 'z': '0'})
    top.add_sim_type(Transient(step=1e-9, stop=2e-9))
    return top


@pytest.mark.slow
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='requires Xyce (SILK_XYCE_BIN setting)')
def test_xyce_nested_dot_request_against_colon_output(tmp_path):
    from pathlib import Path
    top = _nested_top(tmp_path, 'xyce')
    top.set_save_list(['xx1<0>.xy<1>.g[0]'])   # Xyce prints only listed vectors
    deck_path = top.generate_netlist()
    top.run_simulator()
    assert 'v(xx1<0>:xy<1>:g[0])' in Path(deck_path).read_text()
    bracket = top.load_signal('xx1[0].xy[1].g[0]')       # dot request, colon-spelled output
    spice = top.load_signal('xx1<0>.xy<1>.g[0]')
    np.testing.assert_array_equal(bracket.y, spice.y)
    np.testing.assert_allclose(bracket.y, .5)


def test_behavioral_bracket_alias_resolves_entered_instance_net(tmp_path):
    from silk.circuit.circuit_elements import PortDir
    gain = Module(None, str(tmp_path), 'gain.cir', None, module_name='gain')
    gain.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    def model_1(ports, t, params):
        ports.vout.v = 2 * ports.vin.v
    gain.set_behavioral_model(model_1, execution='vector')
    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    dut.add_module(gain, 'g1', nets={'vin': 'vin', 'vout': 'mid'})
    dut.add_module(gain, 'g2', nets={'vin': 'mid', 'vout': 'vout'})
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.simulator = 'behavioral'
    top.add_source(sources.VoltageSource('vin', positive='a', negative='0', dc=1))
    top.add_module(dut, 'X[0]', nets={'vin': 'a', 'vout': 'o'})
    top.add_sim_type(Transient(step=1e-9, stop=4e-9))
    top.simulate()
    spice = top.load_signal('xX<0>.mid')
    bracket = top.load_signal('xX[0].mid')
    np.testing.assert_array_equal(spice.y, bracket.y)
    np.testing.assert_allclose(bracket.y, 2.0)


@needs_ngspice
def test_reported_depth2_nested_node_before_and_after(tmp_path):
    """The reported regression: since TASK-520 'xx1[0].xy[1].g[0]' was not found
    while 'xx1<0>.xy<1>.g[0]' was. Both spellings read one vector now."""
    top = _nested_top(tmp_path)
    top.simulate()
    spice = top.load_signal('xx1<0>.xy<1>.g[0]')
    bracket = top.load_signal('xx1[0].xy[1].g[0]')
    np.testing.assert_array_equal(spice.y, bracket.y)
    np.testing.assert_allclose(bracket.y, .5)
    np.testing.assert_array_equal(top.load_signal('XX1[0].XY[1].G[0]').y, bracket.y)  # case-insensitive
