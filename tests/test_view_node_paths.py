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
    def counted_candidates(mod):
        calls['candidate'] += 1
        return candidates(mod)
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
    result = subprocess.run(['node',str(Path(_BROWSER_DIR)/'node_paths_e2e.mjs'),
                             str(tmp_path/'nodes.html'),str(tmp_path/'nodes.json'),
                             mod.results_file,_find_chrome()],cwd=_BROWSER_DIR,
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
