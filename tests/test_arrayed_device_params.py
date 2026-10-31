"""Real simulator and retained-result regressions for TASK-520."""
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from silk import sources, load_pdk
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import PortDir
from silk.elements import elements
from silk.simulation import Op, Transient, DC
from silk.simulation.backend import NgspiceBackend, select_plot
from tests._env import needs_ngspice, needs_pdk, HAVE_XYCE


def _bench(tmp_path, backend='ngspice', *, transient=False):
    pdk = load_pdk('sky130')
    def block(name):
        mod = Module('sky130', str(tmp_path), name + '.cir', pdk_details=pdk, module_name=name)
        mod.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN, 's': PortDir.GROUND})
        return mod
    leaf = block('leaf')
    for name in ('MA', 'M1[0]', 'M1[1]', 'M1_0'):
        leaf.add(pdk.nmos1p8(name, d='d', g='g', s='s', b='s', w=1, l=.15))
    middle = block('middle')
    for name in ('Y[0]', 'Y[1]', 'plain'):
        middle.add_module(leaf, name, nets={'d': 'd', 'g': 'g', 's': 's'})
    dut = block('dut')
    for name in ('X1[0]', 'X1[1]', 'X1_0', 'plain'):
        dut.add_module(middle, name, nets={'d': 'd', 'g': 'g', 's': 's'})
    dut.simulator = backend
    dut.save_device_params = True
    tb = dut.create_test('params')
    tb.add_module_under_test({'d': 'd', 'g': 'g', 's': '0'})
    tb.add_source(sources.VoltageSource('vd', positive='d', negative='0', dc=1))
    tb.add_source(sources.VoltageSource('vg', positive='g', negative='0', dc=1))
    tb.add_sim_type(Transient(step=1e-9, stop=3e-9) if transient else Op())
    return dut, tb


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('transient', [False, True])
def test_arrayed_params_in_op_and_transient_plots(tmp_path, transient):
    dut, tb = _bench(tmp_path, transient=transient)
    probe = tb.probe('X1[0].Y[1].M1[0]:gm')
    report = dut.run_test('params')
    rows = {row.name: row for row in report.op_table()}
    control = rows['dut.plain.plain.MA']
    assert len(rows) == 48
    for row in rows.values():
        for param, origin in control.sources.items():
            if origin != 'sim':
                continue
            assert getattr(row, param) == pytest.approx(getattr(control, param), rel=1e-8), row.name
    np.testing.assert_allclose(probe.y, control.gm, rtol=1e-8)
    assert 'dut.X1[0].Y[1].M1[0]' in rows
    assert '<0>' in rows['dut.X1[0].Y[1].M1[0]'].path


@needs_pdk
@pytest.mark.requires_xyce
@pytest.mark.skipif(not HAVE_XYCE, reason='Xyce unavailable through silk settings')
def test_bracketed_names_on_xyce(tmp_path):
    dut, tb = _bench(tmp_path, 'xyce')
    probe = tb.probe('X1[1].Y[0].M1[1]:gm')
    report = dut.run_test('params')
    rows = {row.name: row for row in report.op_table()}
    assert rows['dut.X1[1].Y[0].M1[1]'].gm == pytest.approx(rows['dut.plain.plain.MA'].gm)
    assert '<1>' in rows['dut.X1[1].Y[0].M1[1]'].path
    np.testing.assert_allclose(probe.y, rows['dut.plain.plain.MA'].gm)


def _node_bench(tmp_path, backend):
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=load_pdk('sky130'), module_name='child')
    child.define_module_ports({'p': PortDir.IN})
    child.add_source(sources.VoltageSource('bias[0]', positive='g[0]', negative='0', dc=.75))
    parent = Module('sky130', str(tmp_path), 'parent.cir', pdk_details=load_pdk('sky130'), module_name='parent')
    parent.define_module_ports({'p': PortDir.IN})
    parent.add_module(child, 'Y[1]', nets={'p': 'p'})
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=load_pdk('sky130'))
    top.add_module(parent, 'X1[0]', nets={'p': '0'})
    top.simulator = backend
    top.add_sim_type(Transient(step=1e-9, stop=3e-9))
    return top


@pytest.mark.parametrize('backend', ['ngspice', 'behavioral'])
@needs_ngspice
def test_nested_angled_parent_bracketed_net_and_viewer(tmp_path, backend):
    top = _node_bench(tmp_path, backend)
    if backend == 'ngspice':
        top.set_save_list(['xX1<0>.xY<1>.g[0]'])
    top.simulate()
    sig = top.load_signal('xx1<0>.xy<1>.g[0]')
    np.testing.assert_allclose(sig.y, .75)
    if backend == 'ngspice':
        from tests.test_view_trace_identity import _dump
        from silk.signals.view import emit_view_data
        entries = _dump(emit_view_data(top.results_file))
        entry = next(e for e in entries if e['identity']['vector'].lower() == 'v(xx1<0>.xy<1>.g[0])')
        assert entry['code']['ok']
        spec = entry['identity']['probe']
        np.testing.assert_allclose(top.load_signal(spec['name'], spec['type']).y, .75)


@needs_ngspice
def test_arrayed_source_branch_dc_and_controller_references(tmp_path):
    top = Module('sky130', str(tmp_path), 'sources.cir', pdk_details=load_pdk('sky130'))
    top.add_source(sources.VoltageSource('bias[0]', positive='v', negative='0', dc=1))
    top.add(elements.resistor('R[0]', p='v', n='0', value=1000))
    top.add_source(sources.CCCS('copy[0]', 'f', '0', 2, 'bias[0]'))
    top.add_source(sources.CCVS('trans[0]', 'h', '0', 1000, 'bias[0]'))
    top.add(elements.resistor('rf', p='f', n='0', value=1000))
    top.add_sim_type(DC(source='bias[0]', start=1, stop=2, step=1))
    top.simulate()
    np.testing.assert_allclose(top.load_signal('Vbias<0>', 'I').y, [-.001, -.002])
    np.testing.assert_allclose(top.load_signal('f').y, [2, 4])
    np.testing.assert_allclose(top.load_signal('h').y, [-1, -2])


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('damage', ['missing', 'empty', 'absent-op', 'nan'])
def test_standalone_requested_parameter_diagnostic(tmp_path, monkeypatch, damage):
    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'missing.cir', pdk_details=pdk)
    top.add(pdk.nmos1p8('MA', d='d', g='d', s='0', b='0', w=1, l=.15))
    top.add_source(sources.VoltageSource('bias', positive='d', negative='0', dc=1))
    top.add_sim_type(Transient(step=1e-9, stop=3e-9))
    top.save_device_params = True
    launch = NgspiceBackend._launch_prepared
    def damaged(self, *args, **kwargs):
        raw, log, reason = launch(self, *args, **kwargs)
        plot = select_plot(raw.native, 'op')
        if damage == 'absent-op':
            raw.native.plots[:] = [p for p in raw.native.plots if p is not plot]
        else:
            vec = next(v for v in plot.vectors if v.name.endswith('[gm]'))
            if damage == 'missing':
                plot.vectors.remove(vec)
            elif damage == 'empty':
                vec.data = np.array([])
            else:
                vec.data = np.full_like(vec.data, np.nan)
        return raw, log, reason
    monkeypatch.setattr(NgspiceBackend, '_launch_prepared', damaged)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        top.simulate()
    missing = [w.message for w in seen if getattr(w.message, 'code', '') == 'op_parameter_missing']
    if damage == 'nan':
        assert not missing
    else:
        assert missing, 'requested missing data must not be silent without op_table()'
        assert 'MA' in str(missing[0])
        if damage != 'absent-op':
            assert 'gm' in str(missing[0])
    assert getattr(top._raw, '_device_param_requests', ()), 'proof belongs to the actual run'
    from silk.circuit.test_execution import RunEvidence
    result = SimpleNamespace(netlist=top, corner_str='tt', conditions=None)
    frozen = RunEvidence.capture(result)
    restored = frozen.result(SimpleNamespace(invocation=None, cache_key=None))
    proof = top._raw._device_param_requests
    assert restored.netlist._raw._device_param_requests == proof
    from silk.simulation._worker_pickle import copy_simulation_netlist
    assert copy_simulation_netlist(top)._raw._device_param_requests == proof
    assert pickle.loads(pickle.dumps(top._raw))._device_param_requests == proof
    top.save_device_params = False
    monkeypatch.setattr(NgspiceBackend, '_launch_prepared', launch)
    top.simulate()
    assert not getattr(top._raw, '_device_param_requests', ())
    assert restored.netlist._raw._device_param_requests == proof


@needs_ngspice
@needs_pdk
def test_present_transient_requests_do_not_warn(tmp_path):
    dut, tb = _bench(tmp_path, transient=True)
    tb.probe('X1[0].Y[0].MA:gm')
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        dut.run_test('params')
    assert not [w for w in seen if getattr(w.message, 'code', '') == 'op_parameter_missing']


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('workers', [1, 2])
def test_shared_and_adopted_results_keep_original_request_proof(tmp_path, workers):
    dut, tb = _bench(tmp_path)
    top = Module('sky130', str(tmp_path), 'shared.cir', load_pdk('sky130'))
    for name in ('A[0]', 'A[1]'):
        top.add_module(dut, name, nets={'d': 'd', 'g': 'g', 's': '0'})
    try:
        reports = list(top.run_all_tests(n_workers=workers).values())
        assert len(reports) == 2
        raws = [r.sim_results._raw for r in reports]
        proof = raws[0]._device_param_requests
        assert proof and all(raw._device_param_requests == proof for raw in raws)
        assert len({r.rawfile_path for r in reports}) == 1
        # Mutating the source placement cannot rewrite retained evidence.
        dut.save_device_params = False
        tb.netlist.sig_sources_dict['vd'].positive = 'renamed[0]'
        assert all(raw._device_param_requests == proof for raw in raws)
        for entry in top.list_test_entries():
            assert entry.test.report.sim_results._raw._device_param_requests == proof
    finally:
        top.close()
        dut.close()


@needs_ngspice
@needs_pdk
def test_external_and_failed_runs_have_no_inferred_request_proof(tmp_path, monkeypatch):
    from silk.exceptions import SimulationError
    dut, tb = _bench(tmp_path)
    report = dut.run_test('params')
    proof = report.sim_results._raw._device_param_requests
    assert proof
    # Explicitly adopting an external rawfile cannot inherit current inventory.
    tb.evaluate(rawfile=report.rawfile_path)
    assert not getattr(tb.netlist._raw, '_device_param_requests', ())
    def fail(*args, **kwargs):
        raise SimulationError('injected launch failure')
    monkeypatch.setattr(NgspiceBackend, '_launch_prepared', fail)
    with pytest.raises(SimulationError, match='injected launch failure'):
        tb.netlist.simulate()
    assert tb.netlist._raw is None
    assert not getattr(tb.netlist, '_prepared_device_param_requests', ())


@needs_ngspice
@needs_pdk
def test_added_device_replaces_only_new_run_request_proof(tmp_path):
    pdk = load_pdk('sky130')
    top = Module('sky130', str(tmp_path), 'rename.cir', pdk)
    top.add(pdk.nmos1p8('MA', d='d', g='d', s='0', b='0', w=1, l=.15))
    top.add_source(sources.VoltageSource('bias', positive='d', negative='0', dc=1))
    top.add_sim_type(Op())
    top.save_device_params = True
    top.simulate()
    old = top._raw
    old_proof = old._device_param_requests
    # Placed names are read-only: change this design by adding a new part.
    top.add(pdk.nmos1p8('MB[0]', d='d', g='d', s='0', b='0', w=1, l=.15))
    top.simulate()
    new_proof = top._raw._device_param_requests
    assert new_proof and new_proof != old_proof
    assert {request[1] for request in new_proof} == {'MA', 'MB[0]'}
    assert all('.XMMB<0>.' in request[2] for request in new_proof if request[1] == 'MB[0]')
    assert old._device_param_requests == old_proof


@needs_ngspice
@needs_pdk
@pytest.mark.parametrize('fail_first', [False, True])
def test_repeated_prepared_launch_keeps_request_proof(tmp_path, monkeypatch, fail_first):
    dut, tb = _bench(tmp_path)
    spec = tb._prepare_run(corner='tt')
    bench = spec.netlist
    bench.set_output_options(mode='overwrite')
    bench.generate_netlist(corner=spec.corner_str)
    if fail_first:
        from silk.exceptions import SimulationError
        def fail(*args, **kwargs):
            raise SimulationError('injected launch failure')
        with monkeypatch.context() as patch:
            patch.setattr(NgspiceBackend, '_launch_prepared', fail)
            with pytest.raises(SimulationError, match='injected launch failure'):
                bench.run_simulator()
        assert bench._raw is None
        assert not bench._prepared_device_param_requests
    bench.run_simulator()
    first = bench._raw
    proof = first._device_param_requests
    assert proof
    bench.run_simulator()
    assert bench._raw._device_param_requests == proof
    assert first._device_param_requests == proof
