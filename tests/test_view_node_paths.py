"""Checked graph labels must follow the successful result, never a pathname."""
import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from silk import Module
from silk.signals.rawfile import RawFile, RawPlot, RawVector, read_rawfile, write_rawfile
from silk.signals.view import emit_view_data, ViewError


def circuit(tmp_path):
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({'out': 'output'})
    child.add_net('Mid')
    top = Module(None, str(tmp_path), 'top.cir', None)
    top.add_net('vout')
    top.add_module(child, 'dut', {'out': 'vout'})
    return top


def rawfile(tmp_path, names=('v(vout)', 'v(xdut.mid)'), *, ac=False):
    plots = []
    for title in (['Transient Analysis', 'AC Analysis'] if ac else ['Transient Analysis']):
        complex_ = title == 'AC Analysis'
        vectors = [RawVector('frequency' if complex_ else 'time', data=np.array([1., 2.]))]
        vectors += [RawVector(n, 'voltage', data=np.array([i+1., i+2.], dtype=complex if complex_ else float)) for i,n in enumerate(names)]
        plots.append(RawPlot({'plotname': title}, vectors, 2, not complex_, True))
    path = tmp_path / 'result.raw'
    write_rawfile(path, RawFile(plots))
    return read_rawfile(path)


def associated(tmp_path, names=('v(vout)', 'v(xdut.mid)'), *, ac=False):
    from silk.circuit.view_nodes import _capture_addresses
    from silk.simulation.backend import RawResult
    mod = circuit(tmp_path)
    raw = rawfile(tmp_path, names, ac=ac)
    mod._raw = RawResult(raw, raw.source_path)
    mod._raw._view_addresses = _capture_addresses(mod)
    mod.results_file = raw.source_path
    return mod


def context(mod):
    from silk.circuit import view_node_context
    return view_node_context(mod)


def payload(mod, **kwargs):
    ctx = context(mod)
    assert ctx is not None
    return emit_view_data(ctx.source, node_context=ctx, labels='designer', **kwargs)


def test_checked_context_labels_selects_and_preserves_raw_identity(tmp_path):
    mod = associated(tmp_path)
    p = payload(mod, signals=['dut.Mid'])
    plot = p['plots'][0]
    sig, = plot['signals']
    assert sig['name'] == 'v(xdut.mid)'
    assert sig['label'] == 'v(dut.Mid)'
    assert sig['node_probe'] == {'name': 'v(dut.Mid)', 'type': 'V'}
    assert sig['source_row'] == 2
    assert len(plot['node_paths']['vectors']) == 3
    assert any(x['name']=='dut.out' for x in plot['node_paths']['paths'])


def test_same_path_replacement_and_wiring_change_withhold_context(tmp_path):
    mod = associated(tmp_path)
    assert context(mod) is not None
    mod.elem_dict['dut']['pin_list']['out'].net = 'other'
    assert context(mod) is None
    mod.elem_dict['dut']['pin_list']['out'].net = 'vout'
    path = Path(mod.results_file)
    path.write_bytes(path.read_bytes().replace(b'xdut.mid', b'xdut.bad'))
    assert context(mod) is None


def test_mutated_vector_inventory_withholds_metadata(tmp_path):
    mod = associated(tmp_path)
    ctx = context(mod)
    ctx.source.plots[0].vectors[2].name = 'v(changed)'
    with pytest.warns(UserWarning, match='inventory'):
        p = emit_view_data(ctx.source, node_context=ctx, labels='designer')
    assert 'node_paths' not in p['plots'][0]
    assert all('node_probe' not in s for s in p['plots'][0]['signals'])


def test_same_target_transient_ac_labels_survive(tmp_path):
    mod = associated(tmp_path, ac=True)
    p = payload(mod)
    assert [s['label'] for s in p['plots'][0]['signals']] == ['v(vout)', 'v(dut.Mid)']
    assert [s['label'] for s in p['plots'][1]['signals']] == ['v(vout) (mag)', 'v(vout) (phase)', 'v(dut.Mid) (mag)', 'v(dut.Mid) (phase)']


def test_hidden_duplicate_refuses_copy(tmp_path):
    mod = associated(tmp_path, ('v(xdut.mid)', 'v(xdut.mid)'))
    p = payload(mod)
    assert all(s['node_probe'] is None for s in p['plots'][0]['signals'])
    assert all('label' not in s for s in p['plots'][0]['signals'])


def test_copy_pickle_and_missing_association(tmp_path):
    mod = associated(tmp_path)
    assert context(copy.deepcopy(mod)) is not None
    assert context(pickle.loads(pickle.dumps(mod))) is None  # Module drops _raw
    restored = pickle.loads(pickle.dumps(mod._raw))
    assert restored._view_addresses == mod._raw._view_addresses
    del mod._raw._view_addresses
    assert context(mod) is None


def test_raw_fingerprint_is_the_parsed_read(tmp_path):
    import hashlib
    raw = rawfile(tmp_path)
    expected = hashlib.sha256(Path(raw.source_path).read_bytes()).hexdigest()
    assert raw._content_sha256 == expected
    Path(raw.source_path).write_bytes(b'changed')
    assert raw._content_sha256 == expected


def test_metadata_none_is_byte_identical_and_cap_retains_selected_rows(tmp_path, monkeypatch):
    from silk.signals.view import emit as emitter
    from silk.signals.view.serialize import build_data_js
    monkeypatch.setattr(emitter, '_now_iso', lambda: 'fixed')
    mod = associated(tmp_path)
    raw = read_rawfile(mod.results_file)
    assert build_data_js(emit_view_data(raw)) == build_data_js(emit_view_data(raw, node_context=None))
    checked = payload(mod, signals=['dut.Mid'])
    monkeypatch.setattr(emitter, 'NODE_METADATA_MAX_BYTES', 0)
    with pytest.warns(UserWarning, match='metadata size'):
        legacy = payload(mod, signals=['dut.Mid'])
    assert [s['name'] for s in legacy['plots'][0]['signals']] == ['v(xdut.mid)']
    assert 'node_paths' not in legacy['plots'][0]
    assert checked['run']['id'] != legacy['run']['id']
    assert checked['run']['state_key'] == legacy['run']['state_key']


def test_metadata_cap_exact_serialized_bound(tmp_path, monkeypatch):
    from silk.signals.view import emit as emitter
    from silk.signals.view.serialize import build_data_js
    mod = associated(tmp_path)
    p = payload(mod)
    extra = [{'node_paths': plot['node_paths'], 'signals': [
        {k:s[k] for k in ('source_row','node_probe','label') if k in s} for s in plot['signals']]} for plot in p['plots']]
    bound = len(build_data_js(extra).encode('utf-8'))
    monkeypatch.setattr(emitter, 'NODE_METADATA_MAX_BYTES', bound)
    assert 'node_paths' in payload(mod)['plots'][0]
    monkeypatch.setattr(emitter, 'NODE_METADATA_MAX_BYTES', bound-1)
    with pytest.warns(UserWarning, match='metadata size'):
        assert 'node_paths' not in payload(mod)['plots'][0]


def test_total_payload_budget_omits_metadata_not_waveforms(tmp_path, monkeypatch):
    from silk.signals.view import emit as emitter
    mod = associated(tmp_path)
    ctx = context(mod)
    original = emitter._project_bytes
    # Observe the existing waveform projection to set a budget that fits just
    # those samples. The assertion is on emitted data, not on a mock call.
    plans, _ = emitter._plan_all(ctx.source)
    budget = original(plans, dict(min_points=2000,base=4,min_bucket=2))
    with pytest.warns(UserWarning, match='metadata size'):
        output = emitter.emit_view(ctx.source, tmp_path/'cap.html', node_context=ctx, max_bytes=budget)
    assert 'node_paths' not in Path(output.data_path).read_text()
    with pytest.raises(ViewError, match='projected payload size'):
        emitter.emit_view(ctx.source, tmp_path/'too-small.html', max_bytes=0)


@pytest.mark.parametrize('count', [40, 400])
def test_proof_walk_counts_independent_of_alias_count(tmp_path, monkeypatch, count):
    import silk.circuit.view_nodes as nodes
    import silk.circuit.modules_netlist as netlist
    mod = associated(tmp_path)
    for i in range(count):
        mod.add_net(f'n{i}')
    calls = {'candidate': 0, 'physical': 0}
    candidates, physical = nodes.node_candidates, netlist._save_list_inventory
    def counted_candidates(mod, **kwargs):
        calls['candidate'] += 1
        return candidates(mod, **kwargs)
    def counted_physical(mod):
        calls['physical'] += 1
        return physical(mod)
    monkeypatch.setattr(nodes,'node_candidates',counted_candidates)
    monkeypatch.setattr(netlist,'_save_list_inventory',counted_physical)
    mod._raw._view_addresses = nodes._capture_addresses(mod)
    assert calls == {'candidate':1,'physical':1}
    calls.update(candidate=0,physical=0)
    p = payload(mod, signals=['dut.Mid'])
    assert len(p['plots'][0]['signals']) == 1
    assert calls == {'candidate':1,'physical':1}


def test_excluded_conflicting_name_blocks_label_and_emitted_set_wins(tmp_path):
    mod = associated(tmp_path, ('v(xdut.mid)', 'v(dut.Mid)'))
    p = payload(mod, signals=['xdut.mid'])
    sig, = p['plots'][0]['signals']
    assert 'label' not in sig
    assert sig['node_probe']['name'] == 'v(xdut.mid)'
    with pytest.warns(UserWarning, match='precedence'):
        p = payload(mod, signals=['dut.mid'])
    assert p['plots'][0]['signals'][0]['name'] == 'v(dut.Mid)'


def test_unsaved_physical_node_keeps_precedence(tmp_path):
    import silk.circuit.view_nodes as nodes
    mod = associated(tmp_path, ('v(xdut.mid)',))
    from tests._legacy_names import add_legacy_net
    add_legacy_net(mod, 'dut.Mid')
    mod._raw._view_addresses = nodes._capture_addresses(mod)
    sig, = payload(mod)['plots'][0]['signals']
    assert 'label' not in sig
    assert sig['node_probe'] == {'name':'v(xdut.mid)','type':'V'}


def test_external_or_corner_reader_never_manufactures_proof(tmp_path):
    from silk.circuit import view_node_context
    from silk.circuit.modules_sim import _invalidate_current_results
    mod = associated(tmp_path)
    _invalidate_current_results(mod)
    assert context(mod) is None
    with pytest.raises(TypeError, match='requires a Module'):
        view_node_context(SimpleNamespace(results_file='external.raw'))


def test_view_reads_disk_once_even_when_replaced(tmp_path, monkeypatch):
    import builtins
    mod = associated(tmp_path)
    path = Path(mod.results_file)
    path.write_bytes(path.read_bytes().replace(b'xdut.mid',b'xdut.bad'))
    opened, original = [], builtins.open
    def counted(file, *args, **kwargs):
        if str(file) == str(path):
            opened.append(str(file))
        return original(file,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',counted)
    mod.view(out_path=tmp_path/'changed.html',open=False)
    assert opened == [str(path)]


def test_bench_context_uses_existing_result_route(tmp_path, monkeypatch):
    from tests.test_probe_node_paths import bench
    from silk.circuit import view_node_context
    from silk.circuit.view_nodes import _capture_addresses
    from silk.simulation.backend import RawResult
    tb = bench(tmp_path)
    raw = rawfile(tmp_path,('v(xdut.mid)',))
    reader = copy.deepcopy(tb.netlist)
    reader.results_file = raw.source_path
    reader._raw = RawResult(raw,raw.source_path)
    reader._raw._view_addresses = _capture_addresses(reader)
    monkeypatch.setattr(tb,'_result_netlist',lambda:reader)
    assert view_node_context(tb) is not None
    assert tb.netlist.results_file is None
    monkeypatch.setattr(tb,'_result_netlist',lambda:SimpleNamespace(results_file=raw.source_path))
    assert view_node_context(tb) is None


def test_checked_metadata_browser_state_copy_and_local_file(tmp_path):
    import json
    import shutil
    import subprocess
    from tests.test_view_e2e import _reason_to_skip, _find_chrome, _BROWSER_DIR
    from silk.signals.view import emit_view
    reason = _reason_to_skip()
    if reason:
        pytest.skip(reason)
    mod = associated(tmp_path)
    ctx = context(mod)
    emit_view(ctx.source,tmp_path/'nodes.html',node_context=ctx,labels='designer',poll_ms=100)
    p = emit_view_data(ctx.source,node_context=ctx,labels='designer')
    (tmp_path/'nodes.json').write_text(json.dumps(p))
    (tmp_path/'dev').mkdir()
    tb, dev = device_associated(tmp_path/'dev', ['MN:gm'])
    dctx = context(dev)
    emit_view(dctx.source,tmp_path/'devices.html',node_context=dctx,labels='designer',poll_ms=100)
    result = subprocess.run(['node',str(Path(_BROWSER_DIR)/'node_paths_e2e.mjs'),
                             str(tmp_path/'nodes.html'),str(tmp_path/'nodes.json'),
                             mod.results_file,_find_chrome(),str(tmp_path/'devices.html'),
                             device_vector(tb, 'MN', 'gm')],cwd=_BROWSER_DIR,
                            capture_output=True,text=True,timeout=90)
    assert result.returncode == 0,result.stdout+result.stderr


def test_distinct_targets_with_flattened_spelling_keep_unsaved_candidate(tmp_path):
    from silk.circuit.view_nodes import _capture_addresses
    mod = associated(tmp_path,('v(xdut.mid)',))
    from tests._legacy_names import add_legacy_net
    add_legacy_net(mod, 'dut.Mid')
    # This top-level spelling competes with the nested designer name even
    # though only the nested emitted row was saved.
    mod._raw._view_addresses = _capture_addresses(mod)
    with pytest.raises(ViewError,match='ambiguous'):
        payload(mod,signals=['dut.Mid'])
    p = payload(mod,signals=['dut.*'])
    path = next(p for p in p['plots'][0]['node_paths']['paths'] if p['name']=='dut.Mid')
    assert len(path['targets'])==2
    assert sorted(t['rows'] for t in path['targets'])==[[],[1]]


def test_conflicting_raw_spelling_in_other_analysis_blocks_label(tmp_path):
    mod = associated(tmp_path,ac=True)
    raw = mod._raw.native
    raw.plots[1].vectors[2].name = 'v(dut.Mid)'
    write_rawfile(mod.results_file,raw)
    mod._raw.native = read_rawfile(mod.results_file)
    sig = payload(mod)['plots'][0]['signals'][1]
    assert 'label' not in sig
    assert sig['node_probe']=={'name':'v(xdut.mid)','type':'V'}


def test_existing_run_evidence_carries_immutable_address_record(tmp_path):
    from dataclasses import FrozenInstanceError
    from silk.circuit.test_execution import RunEvidence
    mod = associated(tmp_path)
    run = SimpleNamespace(netlist=mod,corner_str='tt',conditions=None)
    evidence = RunEvidence.capture(run)
    transported = dict(evidence.state)['_raw']
    assert transported is not mod._raw
    assert transported._view_addresses == mod._raw._view_addresses
    with pytest.raises(FrozenInstanceError):
        transported._view_addresses.physical = ()


@pytest.mark.parametrize('mutation', ['reorder', 'replace'])
def test_borrowed_inventory_reorder_and_replacement_drop_metadata(tmp_path,mutation):
    mod = associated(tmp_path)
    ctx = context(mod)
    rows = ctx.source.plots[0].vectors
    if mutation=='reorder':
        rows[1],rows[2]=rows[2],rows[1]
    else:
        rows[2]=RawVector('v(other)','voltage',data=np.array([8.,9.]))
    with pytest.warns(UserWarning,match='inventory'):
        p = emit_view_data(ctx.source,node_context=ctx)
    assert 'node_paths' not in p['plots'][0]


def test_one_point_analysis_keeps_its_axis_and_context(tmp_path):
    """A one-point transient keeps `time` as the x axis, not a coordinate row."""
    mod = associated(tmp_path)
    raw = mod._raw.native
    for plot in raw.plots:
        plot.npoints=1
        for vector in plot.vectors:
            vector.data=vector.data[:1]
    write_rawfile(mod.results_file,raw)
    mod._raw.native=read_rawfile(mod.results_file)
    p=payload(mod)
    assert p['plots'][0]['x']['name']=='time'
    assert [s['name'] for s in p['plots'][0]['signals']]==['v(vout)','v(xdut.mid)']
    assert all('role' not in s for s in p['plots'][0]['signals'])


def test_metadata_budget_stops_serialization_before_tail(tmp_path, monkeypatch):
    from silk.signals.view import emit as emitter
    ctx = context(associated(tmp_path))
    # Checked facts are immutable and reused, never serialized in the context.
    with pytest.raises(TypeError):
        ctx._metadata[0]['paths'][0]['name'] = 'changed'
    assert ctx._for_source(ctx.source) is ctx._metadata

    class Unreachable(dict):
        def items(self):
            raise AssertionError('serializer traversed past metadata budget')
    assert emitter._bounded_metadata_bytes([{'first': 'x' * 100}, Unreachable()], 200) > 200
    from silk.signals.view.serialize import build_data_js
    escaped = {'name': '</ScRiPt>\\\"\nµ'}
    exact = len(build_data_js(escaped).encode('utf-8'))
    assert emitter._bounded_metadata_bytes(escaped, exact) == exact
    assert emitter._bounded_metadata_bytes(escaped, exact - 1) > exact - 1
    monkeypatch.setattr(emitter, 'NODE_METADATA_MAX_BYTES', 1)
    with pytest.warns(UserWarning, match='cap 1 bytes.*Browser graph selectors and checked copy'):
        result = emit_view_data(ctx.source, node_context=ctx, signals=['dut.Mid'])
    assert [s['name'] for s in result['plots'][0]['signals']] == ['v(xdut.mid)']
    assert 'node_paths' not in result['plots'][0]


def test_selected_bench_does_not_borrow_authored_result(tmp_path):
    from tests.test_probe_node_paths import bench
    from silk import Module
    from silk.circuit import view_node_context
    from silk.circuit.view_nodes import _capture_addresses
    from silk.simulation.backend import RawResult
    from silk.circuit.test_placement import SelectedTestbench
    tb = bench(tmp_path)
    raw = rawfile(tmp_path, ('v(xdut.mid)',))
    tb.netlist.results_file = raw.source_path
    tb.netlist._raw = RawResult(raw, raw.source_path)
    tb.netlist._raw._view_addresses = _capture_addresses(tb.netlist)
    assert view_node_context(tb) is not None
    parent = Module(None, str(tmp_path), 'parent.cir', None)
    parent.add_module(tb.module, 'one', {'out': 'a'})
    selected = parent.list_test_entries()[0].test
    assert isinstance(selected, SelectedTestbench)
    assert view_node_context(selected) is None


@pytest.mark.parametrize('dialect', ['ngspice', 'xyce'])
@pytest.mark.parametrize('mixed', [False, True])
def test_generated_copy_evaluates(tmp_path, dialect, mixed):
    """Execute the actual JS-generated Python against synthetic run evidence."""
    from tests.test_probe_node_paths import bench
    from tests.test_view_trace_identity import _dump
    from silk.circuit import view_node_context
    from silk.circuit.view_nodes import _capture_addresses

    tb = bench(tmp_path)
    # Wired port, local node, nested arrayed placement, and helper placement.
    names = ['v(vout)', 'v(xdut.mid)', 'v(xdut.xx1<0>.mid)', 'v(xstage.mid)']
    if dialect == 'xyce':
        names = [name.upper().replace('.', ':') for name in names]
    plots = []
    for ac in ([False, True] if mixed else [False]):
        axis = 'frequency' if ac else 'time'
        rows = [RawVector(axis, axis, data=np.array([10., 100.] if ac else [1., 2.]))]
        for index, name in enumerate(names):
            samples = np.array([index + 1., index + 1.5])
            if ac:
                samples = samples * 3 + np.array([1j, -2j])
            rows.append(RawVector(name, 'voltage', data=samples))
        plots.append(RawPlot({'plotname': 'AC Analysis' if ac else 'Transient Analysis'},
                             rows, npoints=2, real=not ac, padded=True))
    filename = tmp_path / 'checked-copy.raw'
    write_rawfile(filename, RawFile(plots))
    tb.netlist.load_file(str(filename))
    tb.netlist.read_raw_file()
    from silk.simulation.backend import RawResult
    tb.netlist._raw = RawResult(read_rawfile(filename), str(filename))
    tb.netlist.results_file = str(filename)
    # Simulate only successful-result association; emission, JS and probes are real.
    tb.netlist._raw._view_addresses = _capture_addresses(tb.netlist)
    ctx = view_node_context(tb)
    assert ctx is not None
    payload = emit_view_data(ctx.source, node_context=ctx, labels='designer')
    entries = _dump(payload)
    assert len(entries) == (12 if mixed else 4)
    for item in entries:
        assert item['code']['ok'], item
        signal = eval(item['code']['code'], {'__builtins__': {}, 'tb': tb})
        sig = next(s for s in payload['plots'][item['plot']]['signals']
                   if s['name'] == item['name'])
        source_plot = ctx.source.plots[item['plot']]
        expected = source_plot.vectors[sig['source_row']].data
        if sig['part'] == 'mag':
            expected = np.abs(expected)
        elif sig['part'] == 'phase':
            expected = np.rad2deg(np.unwrap(np.angle(expected)))
        np.testing.assert_allclose(signal.y, expected)
        np.testing.assert_allclose(signal.x, source_plot.vectors[0].data.real)


@pytest.mark.parametrize('options', [{'signals': ['i(Vone)']}, {'max_depth': 1},
                                     {'exclude': ['i(split)']}])
def test_existing_view_selection_does_not_walk_single_card_aliases(tmp_path, options, monkeypatch):
    from unittest.mock import MagicMock
    from silk import sources
    from silk.circuit import view_node_context
    from silk.circuit.view_nodes import _capture_addresses
    from tests.test_raw_current_paths import _bench
    tb = _bench(tmp_path, ['Vone', '@Isplit__silk_pwl[current]', '@Isplit__silk_dc[current]'],
                [sources.VoltageSource('one', 'a', '0', dc=1),
                 sources.CurrentSource('split', 'b', '0', dc=.2, pwl=[0, .5, 1, 1])])
    mod = tb.netlist
    mod.results_file = str(tmp_path / 's.raw')
    mod._raw._view_addresses = _capture_addresses(mod)
    ctx = view_node_context(mod)
    assert ctx is not None
    monkeypatch.setattr('silk.signals.view.emit._now_iso', lambda: '2000-01-01T00:00:00Z')
    records = ctx._source_currents[0]
    composites = [r for r in records if not r[3] or r[2]]
    aliases = [r for r in records if r[3] and not r[2]]
    assert composites and aliases
    assert records == tuple(composites + aliases)
    expected = emit_view_data(ctx.source, node_context=ctx, **options)
    def prefix_only():
        yield from composites
        yield aliases[0]
        raise AssertionError('selection unnecessarily traversed the single-card aliases')
    guarded = MagicMock()
    guarded.__iter__.side_effect = prefix_only
    object.__setattr__(ctx, '_source_currents', (guarded,))
    assert emit_view_data(ctx.source, node_context=ctx, **options) == expected


# ---------------------------------------------------------------------------
# Device-parameter vectors named by designer (requested probes only).
# ---------------------------------------------------------------------------

def device_bench(tmp_path, probes, simulator='ngspice'):
    """A prepared (never simulated) bench holding several MOS devices."""
    from silk import load_pdk, sources
    from silk.circuit.circuit_elements import PortDir
    from silk.simulation import Op
    pdk = load_pdk('sky130')
    ports = {'d': PortDir.OUT, 'g': PortDir.IN, 's': PortDir.GROUND}
    core = Module('sky130', str(tmp_path), 'core.cir', pdk_details=pdk, module_name='core')
    core.define_module_ports(ports)
    core.add(pdk.nmos1p8('MN', d='d', g='g', s='s', b='s', w=1, l=.15))
    dut = Module('sky130', str(tmp_path), 'dut.cir', pdk_details=pdk, module_name='dut')
    dut.define_module_ports(ports)
    for name in ('MN', 'MX', 'M1[0]'):
        dut.add(pdk.nmos1p8(name, d='d', g='g', s='s', b='s', w=1, l=.15))
    dut.add(pdk.pmos1p8('MP', d='d', g='g', s='s', b='s', w=1, l=.15))
    dut.add_module(core, 'core', nets={'d': 'd', 'g': 'g', 's': 's'})
    dut.simulator = simulator
    tb = dut.create_test('v')
    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0'})
    tb.add_source(sources.VoltageSource('vd', positive='vd', negative='0', dc=1))
    tb.add_source(sources.VoltageSource('vg', positive='vg', negative='0', dc=1))
    tb.add_sim_type(Op())
    for probe in probes:
        tb.probe(probe)
    tb._prepare_run(corner='tt')
    return tb


def device_vector(tb, designer, param, *, wrapper=''):
    """The simulator's own spelling of one device-parameter vector."""
    from silk.circuit.modules_results import (
        resolve_device_probe_address, ngspice_device_op_line, xyce_device_op_term)
    from silk.circuit.modules_results import dut_module_from_elem_dict
    address = resolve_device_probe_address(
        dut_module_from_elem_dict(tb.netlist.elem_dict), designer, param)
    args = (address.instance_path, address.leaf, address.internal, param)
    if tb.netlist.simulator == 'xyce':
        return xyce_device_op_term(*args)
    line = ngspice_device_op_line(*args).lower()
    return f'{wrapper}({line})' if wrapper else line


def device_associated(tmp_path, probes, extra=(), *, simulator='ngspice', ac=False,
                      wrapper_for=None):
    from silk.circuit.view_nodes import _capture_addresses
    from silk.simulation.backend import RawResult
    tb = device_bench(tmp_path, probes, simulator)
    wrapper_for = wrapper_for or {}
    names = ['v(vd)'] + [device_vector(tb, *p.split(':'), wrapper=wrapper_for.get(p, ''))
                         for p in probes]
    names += list(extra)
    raw = rawfile(tmp_path, tuple(names), ac=ac)
    mod = tb.netlist
    mod._raw = RawResult(raw, raw.source_path)
    mod._raw._view_addresses = _capture_addresses(mod)
    mod.results_file = raw.source_path
    return tb, mod


def sigs(p, plot=0):
    return {s['name']: s for s in p['plots'][plot]['signals']}


def test_requested_device_vector_labelled_and_selected_by_both_spellings(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm'])
    vector = device_vector(tb, 'MN', 'gm')
    for selector in ('dut.MN:gm', 'MN:gm'):
        p = payload(mod, signals=[selector])
        sig, = p['plots'][0]['signals']
        assert sig['name'] == vector
        assert sig['label'] == 'dut.MN:gm'
        assert sig['node_probe'] == {'name': 'MN:gm', 'type': 'D'}
        assert sig['source_row'] == 2
    path = next(x for x in p['plots'][0]['node_paths']['paths'] if x['name'] == 'dut.MN:gm')
    assert path['kind'] == 'device' and path['targets'][0]['rows'] == [2]
    net_ids = {t['id'] for x in p['plots'][0]['node_paths']['paths']
               if x.get('kind') != 'device' for t in x['targets']}
    assert path['targets'][0]['id'] not in net_ids
    assert all('kind' not in x for x in p['plots'][0]['node_paths']['paths']
               if x['name'] != 'dut.MN:gm')


def test_device_label_applies_to_every_wrapper_form(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:id'], wrapper_for={'MN:id': 'i'})
    sig, = payload(mod, signals=['MN:id'])['plots'][0]['signals']
    assert sig['name'].startswith('i(@m.') and sig['label'] == 'dut.MN:id'
    assert sig['node_probe'] == {'name': 'MN:id', 'type': 'D'}


def test_nested_and_arrayed_devices_use_probe_spelling(tmp_path):
    tb, mod = device_associated(tmp_path, ['core.MN:gm', 'M1[0]:vth', 'MN:gm'])
    p = payload(mod)
    labels = {s['label']: s['node_probe']['name'] for s in p['plots'][0]['signals']
              if s['node_probe'] and s['node_probe'].get('type') == 'D'}
    assert labels == {'dut.core.MN:gm': 'core.MN:gm', 'dut.M1[0]:vth': 'M1[0]:vth',
                      'dut.MN:gm': 'MN:gm'}
    for selector in ('core.MN:gm', 'dut.core.MN:gm'):
        sig, = payload(mod, signals=[selector])['plots'][0]['signals']
        assert sig['label'] == 'dut.core.MN:gm'
    for selector in ('M1[0]:vth', 'dut.M1[0]:vth'):
        sig, = payload(mod, signals=[selector])['plots'][0]['signals']
        assert sig['label'] == 'dut.M1[0]:vth'
    # The bare name is exact for the top-level device, not the nested one.
    sig, = payload(mod, signals=['dut.MN:gm'])['plots'][0]['signals']
    assert sig['node_probe']['name'] == 'MN:gm'


def test_exclusion_by_designer_device_name(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm', 'MX:gm'])
    names = [s['label'] for s in payload(mod, exclude=['dut.MN:gm'])['plots'][0]['signals']
             if s.get('label', '').endswith(':gm')]
    assert names == ['dut.MX:gm']


def test_unrequested_device_vector_stays_raw(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm'])
    other = device_vector(tb, 'MX', 'gm')
    mod._raw.native = rawfile(tmp_path, ('v(vd)', device_vector(tb, 'MN', 'gm'), other))
    from silk.circuit.view_nodes import _capture_addresses
    mod._raw._view_addresses = _capture_addresses(mod)
    mod.results_file = mod._raw.native.source_path
    p = payload(mod)
    raw_row = sigs(p)[other]
    assert raw_row.get('label') in (None, other) and 'dut.MX:gm' not in str(p['plots'][0]['signals'])
    assert not (raw_row.get('node_probe') or {}).get('type') == 'D'
    with pytest.raises(ViewError):
        payload(mod, signals=['MX:gm'])
    assert not any(x['name'] == 'dut.MX:gm' for x in p['plots'][0]['node_paths']['paths'])


def test_bulk_device_saves_never_expand_eligibility(tmp_path):
    tb, mod = device_associated(tmp_path, [])
    tb.netlist.save_device_params = True
    vector = device_vector(tb, 'MN', 'gm')
    mod._raw.native = rawfile(tmp_path, ('v(vd)', vector))
    from silk.circuit.view_nodes import _capture_addresses
    mod._raw._view_addresses = _capture_addresses(mod)
    mod.results_file = mod._raw.native.source_path
    with pytest.raises(ViewError):
        payload(mod, signals=['MN:gm'])


def test_ambiguous_and_duplicate_rows_never_labelled(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm'])
    vector = device_vector(tb, 'MN', 'gm')
    mod._raw.native = rawfile(tmp_path, ('v(vd)', vector, f'v({vector})'))
    from silk.circuit.view_nodes import _capture_addresses
    mod._raw._view_addresses = _capture_addresses(mod)
    mod.results_file = mod._raw.native.source_path
    p = payload(mod)
    assert not any(s.get('label') == 'dut.MN:gm' for s in p['plots'][0]['signals'])
    with pytest.raises(ViewError):
        payload(mod, signals=['MN:gm'])


def test_cross_plot_ambiguity_blocks_label_everywhere(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm'], ac=True)
    vector = device_vector(tb, 'MN', 'gm')
    control = payload(mod)
    assert all(any(s.get('label') == 'dut.MN:gm' for s in plot['signals'])
               for plot in control['plots'][:1])
    raw = mod._raw.native
    extra = RawVector(f'i({vector})', 'current', data=np.array([1., 2.], dtype=complex))
    raw.plots[1].vectors.append(extra)
    write_rawfile(mod.results_file, raw)
    mod._raw.native = read_rawfile(mod.results_file)
    p = payload(mod)
    for plot in p['plots']:
        assert not any(s.get('label') == 'dut.MN:gm' for s in plot['signals'])


def test_stale_overwritten_and_late_declared_device_get_no_context(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm'])
    assert context(mod) is not None
    tb.probe('MX:gm')
    tb._prepare_run(corner='tt')
    assert context(mod) is None
    (tmp_path / 'again').mkdir()
    tb, mod = device_associated(tmp_path / 'again', ['MN:gm'])
    path = Path(mod.results_file)
    path.write_bytes(path.read_bytes().replace(b'msky130', b'msky131'))
    assert context(mod) is None


def test_ngspice_ac_device_probe_keeps_label_but_refuses_copy(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm'], ac=True)
    p = payload(mod)
    for plot in p['plots']:
        sig = next(s for s in plot['signals'] if s['name'] == device_vector(tb, 'MN', 'gm')
                   or s.get('group') == device_vector(tb, 'MN', 'gm'))
        assert sig['label'].startswith('dut.MN:gm')
    ac_plot = p['plots'][1]
    assert all(s['node_probe'] is None for s in ac_plot['signals']
               if (s.get('group') or s['name']) == device_vector(tb, 'MN', 'gm'))
    assert p['plots'][0]['signals'][1]['node_probe'] == {'name': 'MN:gm', 'type': 'D'}


def test_xyce_device_terms_label_and_pmos_id_sign_caveat(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm', 'MP:id', 'MP:gm'], simulator='xyce')
    p = payload(mod)
    by_label = {s['label']: s for s in p['plots'][0]['signals'] if 'label' in s}
    assert by_label['dut.MN:gm']['node_probe'] == {'name': 'MN:gm', 'type': 'D'}
    assert by_label['dut.MP:gm']['node_probe'] == {'name': 'MP:gm', 'type': 'D'}
    # Silk's read negates a PMOS id; the raw row would not match: no copy.
    assert by_label['dut.MP:id']['node_probe'] is None
    sig, = payload(mod, signals=['MP:id'])['plots'][0]['signals']
    assert sig['label'] == 'dut.MP:id'


def test_payload_without_device_probes_carries_no_device_kind(tmp_path):
    mod = associated(tmp_path)
    p = payload(mod)
    assert all(set(x) == {'name', 'targets'} for x in p['plots'][0]['node_paths']['paths'])
    tb, mod = device_associated(tmp_path, [])
    assert all(set(x) == {'name', 'targets'}
               for x in payload(mod)['plots'][0]['node_paths']['paths'])


def test_device_copy_as_code_in_explorer_module(tmp_path):
    """The explorer module (run under Node) renders the checked device kind."""
    from tests.test_view_trace_identity import _dump
    tb, mod = device_associated(tmp_path, ['MN:gm', 'core.MN:gm'], ac=True)
    p = payload(mod)
    entries = _dump(p)
    by = {(e['plot'], e['name']): e for e in entries}
    top, nested = (device_vector(tb, n, 'gm') for n in ('MN', 'core.MN'))
    assert by[(0, top)]['code'] == {'ok': True, 'code': "tb.probe('MN:gm', analysis='tran')"}
    assert by[(0, nested)]['code']['code'].startswith("tb.probe('core.MN:gm'")
    # AC copy would read the real part, not the displayed complex row.
    ac = [e for e in entries if e['plot'] == 1 and e['name'].startswith(top)]
    assert ac and all(not e['code']['ok'] for e in ac)
    assert all('v(MN:gm)' not in e['code'].get('code', '') for e in entries)


@pytest.mark.parametrize('where', ['raw', 'netlister'])
def test_refused_device_parameter_gets_no_label_and_no_copy(tmp_path, where):
    tb, mod = device_associated(tmp_path, ['MN:gm', 'MX:gm'])
    refused = {('MN', 'gm'): object()}
    if where == 'raw':
        mod._raw._preflight_unavailable_params = refused
    else:
        mod.netlister._preflight_unavailable_params = refused
    p = payload(mod)
    labels = {s.get('label') for s in p['plots'][0]['signals']}
    assert 'dut.MN:gm' not in labels and 'dut.MX:gm' in labels
    refused_row = sigs(p)[device_vector(tb, 'MN', 'gm')]
    assert (refused_row.get('node_probe') or {}).get('type') != 'D'
    with pytest.raises(ViewError):
        payload(mod, signals=['MN:gm'])
    from tests.test_view_trace_identity import _dump
    entries = {e['name']: e for e in _dump(p)}
    assert "MN:gm" not in entries[device_vector(tb, 'MN', 'gm')]['code'].get('code', '')


def test_device_selectors_use_existing_grammar_with_globs_and_brackets(tmp_path):
    tb, mod = device_associated(tmp_path, ['MN:gm', 'core.MN:gm', 'M1[0]:vth'])
    def labels(**kw):
        return sorted(s['label'] for s in payload(mod, **kw)['plots'][0]['signals']
                      if s.get('label', '').count(':') == 1)
    assert labels(signals=['MN:gm']) == ['dut.MN:gm']
    assert labels(signals=['*:gm']) == ['dut.MN:gm', 'dut.core.MN:gm']
    assert labels(signals=['dut.*:gm']) == ['dut.MN:gm', 'dut.core.MN:gm']
    assert labels(signals=['M1[0]:*']) == ['dut.M1[0]:vth']
    assert labels(signals=['dut.M1[0]:vth']) == ['dut.M1[0]:vth']
    assert 'dut.MN:gm' not in labels(exclude=['dut.*:gm'])
    assert 'dut.core.MN:gm' not in labels(exclude=['core.MN:gm'])
