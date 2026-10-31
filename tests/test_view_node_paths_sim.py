"""Successful real runs publish viewer proof; failed launches never do."""
import subprocess

import numpy as np
import pytest

from silk import Module, sources
from silk.circuit import view_node_context
from silk.elements import elements
from silk.exceptions import SimulationError
from silk.signals.view import emit_view_data
from silk.simulation import Transient
from tests._env import needs_ngspice, HAVE_XYCE

needs_xyce = pytest.mark.skipif(not HAVE_XYCE, reason='configured Xyce unavailable')


def bench(tmp_path, simulator):
    leaf = Module(None,str(tmp_path),'leaf.cir',None,module_name='leaf')
    leaf.define_module_ports({'out':'output'})
    leaf.add(elements.resistor('r1',p='out',n='Mid',value=1000))
    leaf.add(elements.resistor('r2',p='Mid',n='0',value=1000))
    dut = Module(None,str(tmp_path),'dut.cir',None,module_name='dut')
    dut.define_module_ports({'out':'output'})
    dut.add_module(leaf,'x1[0]',{'out':'out'})
    tb = dut.create_test('viewer')
    tb.add_module_under_test({'out':'vout'})
    tb.add_source(sources.VoltageSource('supply',positive='vout',negative='0',dc=2))
    tb.add_sim_type(Transient(step=1e-6,stop=3e-6))
    tb.netlist.set_save_list(['dut.x1[0].Mid', 'vout'])
    tb.netlist.simulator = simulator
    return tb


@pytest.mark.slow
@pytest.mark.parametrize('simulator', [pytest.param('ngspice',marks=needs_ngspice),
                                      pytest.param('xyce',marks=[needs_xyce,pytest.mark.requires_xyce])])
def test_simulated_designer_view_and_copied_probe_roundtrip(tmp_path,simulator):
    tb = bench(tmp_path,simulator)
    tb.run()
    nodes = view_node_context(tb)
    assert nodes is not None
    p = emit_view_data(nodes.source,node_context=nodes,labels='designer',signals=['dut.x1[0].Mid'])
    sig, = p['plots'][0]['signals']
    assert sig['label']=='v(dut.x1[0].Mid)'
    np.testing.assert_allclose(tb.probe(sig['node_probe']['name']).y,1.0)
    assert view_node_context(tb.netlist) is not None


def test_launch_success_then_failure_clears_record(tmp_path,monkeypatch):
    from tests.test_simulation_freshness import _success, _failure
    tb = bench(tmp_path,'ngspice')
    mod = tb.netlist
    mod.generate_netlist()
    monkeypatch.setattr(subprocess,'run',_success(mod._run_files))
    mod.run_simulator()
    assert view_node_context(mod) is not None
    mod.generate_netlist()
    monkeypatch.setattr(subprocess,'run',_failure(mod._run_files,'missing'))
    with pytest.raises(SimulationError):
        mod.run_simulator()
    assert view_node_context(mod) is None


def test_external_deck_keeps_legacy_labels(tmp_path,monkeypatch):
    from tests.test_simulation_freshness import _success
    mod = Module(None,str(tmp_path),'external.cir',None)
    mod.set_output_options(mode='overwrite')
    (tmp_path/'external.cir').write_text('* external\n.end\n')
    files = mod.backend._resolve_run_files('external.cir',str(tmp_path))
    monkeypatch.setattr(subprocess,'run',_success(files))
    mod.run_simulator()
    assert view_node_context(mod) is None
