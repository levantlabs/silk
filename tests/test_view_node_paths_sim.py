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


def device_run_bench(tmp_path, simulator, *, bulk_saves=False):
    from silk import load_pdk
    from silk.circuit.circuit_elements import PortDir
    from silk.simulation import Op
    pdk = load_pdk('sky130')
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN, 's': PortDir.GROUND})
    for name in ('MN', 'MX'):
        dut.add(pdk.nmos1p8(name, d='d', g='g', s='s', b='s', w=1, l=.15))
    dut.simulator = simulator
    dut.save_device_params = bulk_saves
    tb = dut.create_test('devices')
    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0'})
    tb.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1))
    tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1))
    tb.add_sim_type(Op())
    return tb


@pytest.mark.slow
@pytest.mark.parametrize('simulator', [pytest.param('ngspice',marks=needs_ngspice),
                                      pytest.param('xyce',marks=[needs_xyce,pytest.mark.requires_xyce])])
def test_simulated_device_vectors_use_designer_names_and_probe_roundtrip(tmp_path, simulator):
    from tests._env import HAVE_PDK
    if not HAVE_PDK:
        pytest.skip('sky130 models unavailable')
    tb = device_run_bench(tmp_path, simulator)
    gm, idrain = tb.probe('MN:gm'), tb.probe('MN:id')
    tb.run()
    nodes = view_node_context(tb)
    assert nodes is not None
    for name, probe in (('gm', gm), ('id', idrain)):
        for selector in (f'dut.MN:{name}', f'MN:{name}'):
            p = emit_view_data(nodes.source, node_context=nodes, labels='designer', signals=[selector])
            sig, = [s for s in p['plots'][0]['signals'] if s.get('role') != 'coordinate']
            assert sig['label'] == f'dut.MN:{name}'
            assert sig['node_probe'] == {'name': f'MN:{name}', 'type': 'D'}
            np.testing.assert_allclose(tb.probe(sig['node_probe']['name']).y, probe.y)
    with pytest.raises(Exception, match='matched no signal'):
        emit_view_data(nodes.source, node_context=nodes, labels='designer', signals=['MX:gm'])


@pytest.mark.slow
@needs_ngspice
def test_bulk_saved_unrequested_devices_stay_raw_in_real_run(tmp_path):
    from tests._env import HAVE_PDK
    if not HAVE_PDK:
        pytest.skip('sky130 models unavailable')
    tb = device_run_bench(tmp_path, 'ngspice', bulk_saves=True)
    tb.probe('MN:gm')
    tb.run()
    nodes = view_node_context(tb)
    p = emit_view_data(nodes.source, node_context=nodes, labels='designer')
    labels = {s.get('label') for s in p['plots'][0]['signals']}
    assert 'dut.MN:gm' in labels and 'dut.MX:gm' not in labels
    raw_mx = [s for s in p['plots'][0]['signals'] if 'xmmx' in s['name']]
    assert raw_mx and all(not (s.get('node_probe') or {}).get('type') == 'D' for s in raw_mx)
