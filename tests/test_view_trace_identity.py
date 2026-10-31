"""TASK-375: an explorer trace's identity record and its "Copy as silk code".

The mapping from a simulator vector name to a ``tb.probe`` spelling lives in
``explorer.mjs`` (it must work for a rawfile opened in the browser, with no
Python behind it). These tests run that JavaScript under Node and hold it to
silk's own Python:

* over every fixture rawfile, each probe spelling the explorer offers is
  resolved with the loader ``tb.probe`` reads through
  (``NgspiceSignal.from_rawfile`` on ``select_plot``'s plot) and must land on
  the very vector the trace was drawn from;
* on a real ngspice testbench, every copied snippet is executed against that
  testbench and its data compared with the rawfile's vectors.
"""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from silk.signals.analysis_kind import plot_silk_analysis
from silk.signals.rawfile import read_rawfile
from silk.signals.spice import NgspiceSignal
from silk.signals.view import emit_view_data
from silk.simulation.backend import select_plot
from tests._env import needs_ngspice

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/rawfiles'
DUMP = ROOT / 'tests/js/trace_identity_dump.mjs'


def _node():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required to run the explorer trace-identity code')
    return node


def _dump(payload, derived=()):
    with tempfile.TemporaryDirectory(prefix='silk_trace_identity_') as tmp:
        path = Path(tmp) / 'dump.json'
        path.write_text(json.dumps({'payload': payload,
                                    'derived': list(derived)}))
        result = subprocess.run([_node(), str(DUMP), str(path)], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _vector(raw_plot, name):
    matches = [v for v in raw_plot.vectors if v.name == name]
    assert len(matches) == 1, (name, [v.name for v in raw_plot.vectors])
    return np.asarray(matches[0].data)


def test_node_trace_identity_contract():
    result = subprocess.run([_node(), '--test', str(ROOT / 'tests/js/explorer_trace_identity.test.mjs')],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('rawfile', sorted(p.name for p in FIXTURES.glob('*.raw')))
def test_offered_probe_reads_the_traced_vector(rawfile):
    raw = read_rawfile(str(FIXTURES / rawfile))
    payload = emit_view_data(str(FIXTURES / rawfile))
    entries = _dump(payload)
    assert len(payload['plots']) == len(raw.plots)
    for i, plot in enumerate(payload['plots']):
        # The simulator's constants plot (left by an aborted run) is not an
        # analysis; the explorer offers no probe for it (see below).
        expected = None if plot['kind'] == 'constants' else plot_silk_analysis(raw.plots[i])
        got = {e['identity']['source']['analysis'] for e in entries if e['plot'] == i}
        assert got <= {expected}, (rawfile, i, got, expected)
    for entry in entries:
        identity, code = entry['identity'], entry['code']
        assert identity['kind'] == 'raw' and identity['identity'] == 1
        assert identity['source']['plot'] == entry['plot']
        if identity['probe'] is None:
            assert identity['probe_reason'] and code['ok'] is False
            assert code['reason'] == identity['probe_reason']
            continue
        probe = identity['probe']
        assert code['ok'] is True and code['code'].startswith('tb.probe(')
        assert (probe['analysis'] is not None) == (len(raw.plots) > 1)
        loaded = NgspiceSignal.from_rawfile(raw, probe['name'], probe['type'],
                                            plot=select_plot(raw, probe['analysis']))
        target = _vector(raw.plots[entry['plot']], identity['vector'])
        np.testing.assert_array_equal(np.asarray(loaded.y), target, err_msg=f'{rawfile}: {code["code"]}')


def test_unmappable_vectors_are_reported_not_guessed():
    entries = _dump(emit_view_data(str(FIXTURES / 'e_ng_noise.raw')))
    assert entries and all(e['code']['ok'] is False for e in entries)
    assert all('noise' in e['code']['reason'] for e in entries)
    constants = _dump(emit_view_data(str(FIXTURES / 'd_multiplot_constants_first.raw')))
    refused = [e for e in constants if e['identity']['source']['plot_name'] == 'constants']
    assert refused and all(e['code']['ok'] is False for e in refused)
    assert all('constants' in e['code']['reason'] for e in refused)
    single = _dump(emit_view_data(str(FIXTURES / 'd_constants_only.raw')))
    assert single and all(e['code']['ok'] is False for e in single)


def _rc_testbench(directory):
    from silk import sources
    from silk.circuit.circuit_elements import PortDir
    from silk.circuit.modules import Module
    from silk.elements import elements
    from silk.pdk import load_pdk
    from silk.simulation import analysis

    dut = Module('PDK_skywater130', str(directory), 'rc.cir', pdk_details=load_pdk('sky130'),
                 module_name='rc')
    dut.define_module_ports({'a': PortDir.IN, 'b': PortDir.OUT})
    with dut:
        elements.resistor('r1', p='a', n='m', value=1e3)
        elements.resistor('r2', p='m', n='b', value=1e3)
        elements.capacitor('c1', p='b', n='0', value=1e-9)
    tb = dut.create_test('rc_tb')
    tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
    tb.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0.0, ac=1.0,
        pulse=sources.Pulse(v1=0.0, v2=1.0, td=0.0, tr=1e-9, tf=1e-9, pw=5e-6, per=10e-6)))
    tb.add_sim_type(analysis.Transient(step=1e-8, stop=1e-5))
    tb.add_sim_type(analysis.AC('dec', 10, 1e3, 1e8))
    return tb


@needs_ngspice
@pytest.mark.filterwarnings('ignore:invalid value encountered in divide:RuntimeWarning')  # d3 at t=0
def test_copied_code_runs_against_a_real_testbench(tmp_path):
    tb = _rc_testbench(tmp_path)
    tb.run()
    raw = read_rawfile(tb.results_file)
    payload = emit_view_data(tb.results_file)
    kinds = [plot_silk_analysis(p) for p in raw.plots]
    tran, ac = kinds.index('tran'), kinds.index('ac')

    derived = [
        {'plot': tran, 'name': 'd1', 'operation': 'difference', 'operands': ['v(vin)', 'v(vout)']},
        {'plot': tran, 'name': 'd2', 'operation': 'mean', 'operands': ['v(vin)', 'v(vout)', 'v(xdut.m)']},
        {'plot': tran, 'name': 'd3', 'operation': 'quotient', 'operands': ['d1', 'i(vvs__silk_wave)']},
        {'plot': tran, 'name': 'd4', 'operation': 'scale', 'operands': ['v(xdut.m)'],
         'params': {'gain': 2.5, 'offset': -0.25}},
        {'plot': tran, 'name': 'd5', 'operation': 'abs', 'operands': ['d1']},
        {'plot': tran, 'name': 'd6', 'operation': 'derivative', 'operands': ['v(vout)']},
        {'plot': tran, 'name': 'd7', 'operation': 'integral', 'operands': ['i(vvs__silk_wave)']},
        {'plot': tran, 'name': 'd8', 'operation': 'product', 'operands': ['v(vin)', 'i(vvs__silk_wave)']},
        {'plot': tran, 'name': 'd9', 'operation': 'sum', 'operands': ['d4', 'v(vout)']},
        {'plot': ac, 'name': 'g', 'operation': 'quotient', 'operands': ['v(vout) (mag)', 'v(vin) (mag)']},
    ]
    entries = _dump(payload, derived)

    tv = {v.name: np.asarray(v.data) for v in raw.plots[tran].vectors}
    av = {v.name: np.asarray(v.data) for v in raw.plots[ac].vectors}
    t = tv['time']
    d1 = tv['v(vin)'] - tv['v(vout)']
    integral = np.concatenate([[0.0], np.cumsum(0.5 * (tv['i(vvs__silk_wave)'][1:] + tv['i(vvs__silk_wave)'][:-1]) * np.diff(t))])
    expected = {
        (tran, 'v(vin)'): tv['v(vin)'], (tran, 'v(vout)'): tv['v(vout)'],
        (tran, 'v(xdut.m)'): tv['v(xdut.m)'], (tran, 'i(vvs__silk_wave)'): tv['i(vvs__silk_wave)'],
        (tran, 'v(__silk_src_vvs)'): tv['v(__silk_src_vvs)'],
        (tran, 'i(vvs__silk_dc)'): tv['i(vvs__silk_dc)'],
        (tran, 'd1'): d1,
        (tran, 'd2'): (tv['v(vin)'] + tv['v(vout)'] + tv['v(xdut.m)']) / 3,
        (tran, 'd3'): d1 / tv['i(vvs__silk_wave)'],
        (tran, 'd4'): 2.5 * tv['v(xdut.m)'] - 0.25,
        (tran, 'd5'): np.abs(d1),
        (tran, 'd6'): np.gradient(tv['v(vout)'], t),
        (tran, 'd7'): integral,
        (tran, 'd8'): tv['v(vin)'] * tv['i(vvs__silk_wave)'],
        (tran, 'd9'): 2.5 * tv['v(xdut.m)'] - 0.25 + tv['v(vout)'],
        (ac, 'g'): np.abs(av['v(vout)']) / np.abs(av['v(vin)']),
    }
    for name in ('v(vin)', 'v(vout)', 'v(xdut.m)', 'i(vvs__silk_wave)',
                 'v(__silk_src_vvs)', 'i(vvs__silk_dc)'):
        expected[(ac, f'{name} (mag)')] = np.abs(av[name])
        z = av[name]
        expected[(ac, f'{name} (phase)')] = np.degrees(np.unwrap(np.angle(z)))

    codes = {(e['plot'], e['name']): e['code'] for e in entries}
    assert set(codes) == set(expected)
    assert codes[(tran, 'v(xdut.m)')]['code'] == "tb.probe('xdut.m', analysis='tran')"
    assert codes[(tran, 'i(vvs__silk_wave)')]['code'] == "tb.probe('vvs__silk_wave', 'I', analysis='tran')"
    assert codes[(tran, 'd1')]['code'] == ("tb.probe('vin', analysis='tran') - "
                                           "tb.probe('vout', analysis='tran')")
    for key, want in expected.items():
        code = codes[key]
        assert code['ok'], (key, code)
        signal = eval(code['code'], {'tb': tb})  # the snippet exactly as copied
        np.testing.assert_allclose(np.asarray(signal.y), want, rtol=1e-9, atol=1e-15,
                                   err_msg=f'{key}: {code["code"]}')


def _browser_skip_reason():
    from tests.test_view_e2e import _reason_to_skip
    return _reason_to_skip()


@pytest.mark.skipif(_browser_skip_reason() is not None, reason=_browser_skip_reason() or '')
def test_trace_menu_and_host_hook_in_a_browser(tmp_path):
    from tests.test_view_e2e import _find_chrome
    assets = ROOT / 'silk/signals/view/assets'
    for name in ('selectors.mjs', 'explorer.mjs', 'view.css', 'uplot.iife.min.js', 'waveform-core.mjs'):
        shutil.copyfile(assets / name, tmp_path / name)
    payload = emit_view_data(str(FIXTURES / 'e_ng_tran.raw'))
    (tmp_path / 'payload.json').write_text(json.dumps(payload), encoding='utf-8')
    result = subprocess.run(
        ['node', str(ROOT / 'tests/browser/explorer_trace_menu_e2e.mjs'), str(tmp_path), _find_chrome()],
        cwd=ROOT / 'tests/browser', capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_silk_code_covers_exactly_the_exported_operation_list():
    """Every operation TASK-373 exports renders, and nothing else does."""
    from silk.signals import export_signal_operations
    operations = export_signal_operations()
    script = """
import {silkCode} from './silk/signals/view/assets/explorer.mjs';
const ops = JSON.parse(process.argv[1]);
const raw = n => ({kind: 'raw', name: n, probe: {name: n, type: 'V', transform: null, analysis: null}});
const out = {};
for (const op of [...ops, {name: 'mystery', min_inputs: 1, parameters: []}]) {
    const operands = Array.from({length: op.min_inputs}, (_, i) => raw('n' + i));
    const parameters = Object.fromEntries(op.parameters.map(p => [p.name, p.default + 1]));
    out[op.name] = silkCode({kind: 'derived', name: 'd', operation: op.name, parameters, operands});
}
process.stdout.write(JSON.stringify(out));
"""
    result = subprocess.run([_node(), '--input-type=module', '-e', script, json.dumps(operations)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    codes = json.loads(result.stdout)
    assert codes.pop('mystery')['ok'] is False
    assert set(codes) == {op['name'] for op in operations}
    for name, code in codes.items():
        assert code['ok'], (name, code)
        compile(code['code'], name, 'eval')


@pytest.mark.parametrize('headers', [
    ['AC Analysis', 'Operating Point'],
    ['AC Analysis', 'DC transfer characteristic'],
    ['Operating Point', 'DC transfer characteristic'],
    ['Transient Analysis'],
])
def test_scalar_copy_code_roundtrips_the_declared_plot(tmp_path, headers):
    """Display shape op must not select a different analysis's values."""
    from silk.simulation.backend import RawResult
    from tests.test_view_op_e2e import raw_plot

    source = tmp_path / 'scalar.raw'
    chunks = []
    for i, header in enumerate(headers):
        axis, axis_type = ('v(vin)', 'voltage') if header == 'Operating Point' else (
            ('frequency', 'frequency') if header == 'AC Analysis' else
            ('time', 'time') if header == 'Transient Analysis' else ('sweep', 'voltage'))
        chunks.append(raw_plot(header, [(axis, axis_type), ('v(vout)', 'voltage')],
                               [[1, (i + 2) * 10]], complex=header == 'AC Analysis'))
    source.write_text(''.join(chunks))
    raw = read_rawfile(source)
    tb = _rc_testbench(tmp_path / 'bench')
    tb.netlist._raw = RawResult(raw, str(source))
    tb.netlist.simResultsAll = raw
    tb.netlist.sim_results_loaded = True
    entries = _dump(emit_view_data(source))
    outputs = [e for e in entries if e['identity']['vector'] == 'v(vout)' and e['identity']['part'] != 'phase']
    assert len(outputs) == len(headers)
    for entry in outputs:
        i = entry['plot']
        assert entry['identity']['source']['analysis'] == plot_silk_analysis(raw.plots[i])
        assert entry['code']['ok']
        signal = eval(entry['code']['code'], {'tb': tb})
        np.testing.assert_array_equal(signal.y, [(i + 2) * 10])
    for entry in entries:
        if entry['name'] in ('time', 'sweep'):
            assert entry['identity']['probe'] is None
            assert entry['code']['ok'] is False
            with pytest.raises(ValueError, match='has no net'):
                tb.probe(entry['name'])
    op_indices = [i for i,h in enumerate(headers) if h == 'Operating Point']
    for i in op_indices:
        first = next(e for e in entries if e['plot'] == i and e['name'] == 'v(vin)')
        np.testing.assert_array_equal(eval(first['code']['code'], {'tb': tb}).y, [1])


@pytest.mark.parametrize('filename,values', [
    ('combined.raw', (1., 9.)), ('reversed.raw', (9., 1.)),
])
def test_same_kind_plots_keep_distinct_viewer_index_identity(filename, values):
    path = FIXTURES.parent / 'plot_selection' / filename
    raw = read_rawfile(path)
    payload = emit_view_data(str(path))
    entries = _dump(payload)
    assert payload['default_plot'] == 1
    assert len(payload['plots']) == 2
    out = [e for e in entries if e['identity']['vector'] == 'v(out)']
    assert len(out) == 2
    for entry, value in zip(out, values):
        index = entry['plot']
        assert entry['identity']['source']['plot'] == index
        assert entry['identity']['source']['analysis'] == 'tran'
        np.testing.assert_allclose(_vector(raw.plots[index], 'v(out)'), value)
