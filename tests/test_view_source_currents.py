"""Checked physical source cards retain probe and trace identity."""
import json
from pathlib import Path
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import threading

import numpy as np
import pytest

from silk import Module, sources
from silk.circuit import view_node_context
from silk.elements import elements
from silk.signals.view import emit_view_data, emit_view
from silk.simulation import Transient
from tests._env import needs_ngspice
from tests.test_view_trace_identity import _dump


def source_bench(tmp_path):
    child = Module(None, str(tmp_path), 'child.cir', None, module_name='child')
    child.define_module_ports({'out': 'output'})
    child.add_source(sources.CurrentSource('bias[0]', positive='out', negative='0',
                                          dc=.002, pwl=[0, .001, 2e-6, .003]))
    child.add_source(sources.VoltageSource('inner[1]', positive='local', negative='0',
                                          dc=.1, pwl=[0, .2, 2e-6, .4]))
    child.add(elements.resistor('local_load', p='local', n='0', value=1000))
    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'out': 'output'})
    dut.add_module(child, 'leaf[1]', {'out': 'out'})
    tb = dut.create_test('cards')
    tb.add_module_under_test({'out': 'out'})
    tb.netlist.add(elements.resistor('load', p='out', n='0', value=1000))
    tb.add_source(sources.VoltageSource('stim[0]', positive='vin', negative='0',
                                       dc=.2, pulse=sources.Pulse(0, 1, tr=1e-8, tf=1e-8,
                                                                 pw=1e-6, per=2e-6)))
    tb.netlist.add(elements.resistor('input_load', p='vin', n='0', value=1000))
    tb.add_source(sources.VoltageSource('plain[1]', positive='fixed', negative='0', dc=1))
    tb.netlist.add(elements.resistor('fixed_load', p='fixed', n='0', value=1000))
    tb.add_sim_type(Transient(step=1e-8, stop=3e-6))
    probes = [tb.probe('i(' + name + ')') for name in (
        'Vstim[0]__silk_wave', 'Vstim[0]__silk_dc', 'Vplain[1]',
        'xdut.xleaf[1].Ibias[0]__silk_pwl', 'xdut.xleaf[1].Ibias[0]__silk_dc',
        'v.xdut.xleaf[1].Vinner[1]__silk_pwl', 'v.xdut.xleaf[1].Vinner[1]__silk_dc')]
    tb.run()
    return tb, probes


@pytest.mark.slow
@needs_ngspice
def test_real_source_cards_prove_each_row_and_copied_identity(tmp_path):
    tb, probes = source_bench(tmp_path)
    ctx = view_node_context(tb)
    payload = emit_view_data(ctx.source, node_context=ctx, labels='designer')
    signals = payload['plots'][0]['signals']
    currents = [s for s in signals if (s.get('node_probe') or {}).get('type') == 'I'
                and not (s.get('node_probe') or {}).get('terms')]
    assert len(currents) == 7
    entries = {e['name']: e for e in _dump(payload)}
    for signal in currents:
        probe = signal['node_probe']
        assert '<' not in probe['name'] and '[' in probe['name']
        assert signal['label'] == probe['name']
        row = ctx.source.plots[0].vectors[signal['source_row']]
        np.testing.assert_array_equal(tb.probe(probe['name']).y, row.data)
        entry = entries[signal['name']]
        assert entry['identity']['probe']['name'] == probe['name']
        assert entry['identity']['probe']['type'] == 'I'
        assert entry['code']['ok']
        np.testing.assert_array_equal(eval(entry['code']['code'], {'tb': tb}).y, row.data)
    assert len(probes) == len(currents)
    # These two series cards remain separate rows, with equal currents.
    np.testing.assert_allclose(probes[0].y, probes[1].y, rtol=1e-12, atol=1e-15)
    assert any(s.get('label') == 'v(vin)' for s in signals)
    emit_view(ctx.source, tmp_path / 'sources.html', node_context=ctx, labels='designer', inline=True)


@pytest.mark.slow
@needs_ngspice
def test_logical_source_rows_match_named_probes(tmp_path):
    from tests.test_view_emit import _decoded_y
    tb, _ = source_bench(tmp_path)
    ctx = view_node_context(tb)
    payload = emit_view_data(ctx.source, node_context=ctx, labels='designer')
    signals = payload['plots'][0]['signals']
    entries = {entry['name']: entry for entry in _dump(payload)}
    for name in ('stim[0]', 'xdut.xleaf[1].bias[0]', 'xdut.xleaf[1].inner[1]'):
        probe = 'i(' + name + ')'
        signal = next((s for s in signals if (s.get('node_probe') or {}).get('name') == probe), None)
        assert signal is not None, probe
        np.testing.assert_allclose(_decoded_y(signal), tb.probe(probe).y, rtol=1e-6, atol=1e-12)
        assert entries[signal['name']]['identity']['probe']['name'] == probe
        code = entries[signal['name']]['code']
        assert code['ok']
        np.testing.assert_array_equal(eval(code['code'], {'tb': tb}).y, tb.probe(probe).y)
    assert len([s for s in signals if '__silk_' in (s.get('node_probe') or {}).get('name', '')]) == 6


def saved_source_context(tmp_path, names, source_class=sources.CurrentSource):
    from silk.circuit.view_nodes import _capture_addresses
    from silk.simulation.backend import RawResult
    from tests.test_view_node_paths import rawfile
    mod = Module(None, str(tmp_path), 'saved.cir', None)
    mod.add_source(source_class('stim[0]', positive='out', negative='0',
                                dc=.2, pwl=[0, .5, 1, 1]))
    raw = rawfile(tmp_path, names)
    mod._raw = RawResult(raw, raw.source_path)
    mod._raw._view_addresses = _capture_addresses(mod)
    mod.results_file = raw.source_path
    return view_node_context(mod)


@pytest.mark.slow
@needs_ngspice
def test_unprobed_waveform_current_source_views_without_logical_row(tmp_path):
    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({'out': 'output'})
    dut.add(elements.resistor('load', p='out', n='0', value=1000))
    tb = dut.create_test('unprobed')
    tb.add_module_under_test({'out': 'out'})
    tb.add_source(sources.CurrentSource('bias', positive='out', negative='0',
                                       dc=1e-3, pwl=[0, 0, 1e-6, 1e-3]))
    tb.add_sim_type(Transient(step=1e-8, stop=2e-6))
    tb.run()
    tb.netlist.view(open=False)
    ctx = view_node_context(tb)
    payload = emit_view_data(ctx.source, node_context=ctx, labels='designer')
    names = [s['name'] for s in payload['plots'][0]['signals']]
    assert 'v(out)' in names
    assert 'i(bias)' not in names


@pytest.mark.parametrize('missing', ['pwl', 'dc', 'both', None])
def test_missing_saved_current_card_refuses_logical_current(tmp_path, missing):
    from silk.signals.view import ViewError
    names = tuple('i(@istim<0>__silk_' + suffix + '[current])'
                  for suffix in ('pwl', 'dc') if missing not in (suffix, 'both'))
    ctx = saved_source_context(tmp_path, names)
    if missing:
        payload = emit_view_data(ctx.source, node_context=ctx)
        assert [s['name'] for s in payload['plots'][0]['signals']] == list(names)
        with pytest.raises(ViewError, match=r'i\(stim\[0\]\).*missing.*__silk_') as error:
            emit_view_data(ctx.source, node_context=ctx, signals=['i(stim[0])'])
        for suffix in ('pwl', 'dc') if missing == 'both' else (missing,):
            assert '__silk_' + suffix in str(error.value)
        assert 'save' in str(error.value).lower()
        # Display filtering is not missing saved data: the surviving card is usable.
        if names:
            payload = emit_view_data(ctx.source, node_context=ctx, signals=[names[0]])
            assert [s['name'] for s in payload['plots'][0]['signals']] == list(names)
            payload = emit_view_data(ctx.source, node_context=ctx, exclude=['i(stim[0])'])
            assert [s['name'] for s in payload['plots'][0]['signals']] == list(names)
            from silk.signals.view import EmptySelectionError
            with pytest.raises(EmptySelectionError):
                emit_view_data(ctx.source, node_context=ctx, exclude=list(names))
        return
    payload = emit_view_data(ctx.source, node_context=ctx)
    logical = next(s for s in payload['plots'][0]['signals'] if s['name'] == 'i(stim[0])')
    from tests.test_view_emit import _decoded_y
    np.testing.assert_array_equal(_decoded_y(logical), np.array([3., 5.]))


@pytest.mark.parametrize('filters', [{'exclude': []}, {'max_depth': 10}])
def test_unsaved_logical_current_does_not_keep_filtered_selection_alive(tmp_path, filters):
    from silk.signals.view import EmptySelectionError
    ctx = saved_source_context(tmp_path, ())
    with pytest.raises(EmptySelectionError):
        emit_view_data(ctx.source, node_context=ctx, **filters)


@pytest.mark.parametrize('selector', ['*', 'i(*)'])
def test_explicit_glob_refuses_incomplete_logical_current(tmp_path, selector):
    from silk.signals.view import ViewError
    ctx = saved_source_context(tmp_path, ('i(@istim<0>__silk_pwl[current])',))
    with pytest.raises(ViewError, match=r'i\(stim\[0\]\).*missing.*__silk_dc'):
        emit_view_data(ctx.source, node_context=ctx, signals=[selector])


@pytest.mark.parametrize('selection,excluded,expected', [
    (None, None, {'i(vstim<0>__silk_pwl)', 'i(vstim<0>__silk_dc)', 'i(stim[0])'}),
    (['i(vstim<0>__silk_pwl)'], None, {'i(vstim<0>__silk_pwl)'}),
    (['i(stim[0])'], None, {'i(stim[0])'}),
    (None, ['i(stim[0])'], {'i(vstim<0>__silk_pwl)', 'i(vstim<0>__silk_dc)'}),
    (None, ['i(vstim<0>__silk_pwl)'], {'i(vstim<0>__silk_dc)', 'i(stim[0])'}),
    (['i(stim*)'], None, {'i(stim[0])'}),
])
def test_logical_source_rows_follow_existing_selectors(tmp_path, selection, excluded, expected):
    ctx = saved_source_context(tmp_path, ('i(vstim<0>__silk_pwl)', 'i(vstim<0>__silk_dc)'),
                               sources.VoltageSource)
    payload = emit_view_data(ctx.source, node_context=ctx, signals=selection, exclude=excluded)
    assert {s['name'] for s in payload['plots'][0]['signals']} == expected


def test_logical_current_selection_reads_hidden_saved_operands(tmp_path):
    from tests.test_view_emit import _decoded_y
    ctx = saved_source_context(tmp_path, ('i(@istim<0>__silk_pwl[current])',
                                          'i(@istim<0>__silk_dc[current])'))
    payload = emit_view_data(ctx.source, node_context=ctx, signals=['i(stim[0])'])
    signals = payload['plots'][0]['signals']
    assert [s['name'] for s in signals] == ['i(stim[0])']
    np.testing.assert_array_equal(_decoded_y(signals[0]), np.array([3., 5.]))


@pytest.mark.parametrize('mode', [
    'ac_current', 'behavioral_voltage', 'ambiguous_current',
    'uncaptured_card', 'constants_current',
])
def test_missing_cards_do_not_certify_unsupported_or_unproved_currents(tmp_path, mode):
    from silk.circuit.view_nodes import _capture_addresses, _proof, _source_current_proofs
    from silk.simulation.behavioral_backend import BEHAVIORAL_COMMAND
    from tests.test_view_node_paths import rawfile
    mod = Module(None, str(tmp_path), 'unsupported.cir', None)
    cls = sources.VoltageSource if mode == 'behavioral_voltage' else sources.CurrentSource
    mod.add_source(cls('stim[0]', positive='out', negative='0', dc=.2, pwl=[0, .5, 1, 1]))
    names = (('i(@istim<0>__silk_pwl[current])',) * 2
             if mode == 'ambiguous_current' else ('v(out)',))
    raw = rawfile(tmp_path, names)
    if mode == 'ac_current':
        raw.plots[0].header['plotname'] = 'AC Analysis'
    if mode == 'constants_current':
        raw.plots[0].header['plotname'] = 'constants'
    if mode == 'behavioral_voltage':
        raw.plots[0].header['command'] = BEHAVIORAL_COMMAND
    frozen = _capture_addresses(mod)
    _, proofs = _proof(raw, frozen)
    if mode == 'uncaptured_card':
        from dataclasses import replace
        frozen = replace(frozen, currents=())
    assert _source_current_proofs(raw, frozen, proofs) == ((),)


@pytest.mark.parametrize('budget_kind', ['metadata', 'payload'])
@pytest.mark.parametrize('selection_kind', ['logical', 'mixed', 'physical'])
def test_budget_fallback_never_discards_selected_logical_currents(
        tmp_path, monkeypatch, budget_kind, selection_kind):
    from silk.signals.view import emit as emitter
    from silk.signals.view import ViewError
    card = 'i(vstim<0>__silk_dc)'
    ctx = saved_source_context(tmp_path, ('i(vstim<0>__silk_pwl)', card), sources.VoltageSource)
    selected = ([] if selection_kind == 'physical' else ['i(stim[0])'])
    if selection_kind != 'logical':
        selected.append(card)
    options = {}
    if budget_kind == 'metadata':
        monkeypatch.setattr(emitter, 'NODE_METADATA_MAX_BYTES', 0)
    else:
        physical, _ = emitter._plan_all(ctx.source, masks=[[False, card in selected]])
        options['max_bytes'] = emitter._project_bytes(
            physical, dict(min_points=2000, base=4, min_bucket=2))

    def export():
        if budget_kind == 'metadata':
            return emit_view_data(ctx.source, node_context=ctx, signals=selected)
        return emit_view(ctx.source, tmp_path / 'budget.html', node_context=ctx,
                         signals=selected, **options)

    if selection_kind == 'physical':
        with pytest.warns(UserWarning, match='metadata size'):
            result = export()
        if budget_kind == 'metadata':
            assert [s['name'] for s in result['plots'][0]['signals']] == [card]
            assert 'node_paths' not in result['plots'][0]
        else:
            assert result.n_signals == 1
            assert '"node_paths"' not in Path(result.data_path).read_text()
        return
    with pytest.raises(ViewError) as error:
        export()
    message = str(error.value)
    assert 'i(stim[0])' in message
    assert 'metadata cap' in message
    assert '0 bytes' in message
    if budget_kind == 'payload':
        assert 'payload budget' in message and 'max_bytes' in message
    assert 'reduce' in message.lower() or 'increase' in message.lower()


def test_behavioral_current_cards_only(tmp_path):
    from tests.test_behavioral_dc import _bench
    tb = _bench(tmp_path, analysis=Transient(step=.25, stop=1))
    tb.add_source(sources.CurrentSource('bias[0]', positive='bias', negative='0',
                                       dc=.2, pwl=[0, .5, 1, 1]))
    tb.run()
    from silk.circuit.view_nodes import _capture_addresses, _proof
    from silk.signals.rawfile import read_rawfile
    reader = tb._result_netlist()
    raw = read_rawfile(reader.results_file)
    _, proofs = _proof(raw, _capture_addresses(reader))
    currents = [(row, label, probe) for row, label, probe in proofs[0]
                if probe and probe.startswith('i(')]
    assert len(currents) == 3  # existing total plus its two physical cards
    assert all('Ibias[0]' in label for _, label, _ in currents)
    for row, label, probe in currents:
        np.testing.assert_array_equal(tb.probe(probe).y, raw.plots[0].vectors[row].data)


def exercise_source_browser(payload, executable, screenshots):
    """Drive the installed browser with its matching optional test driver."""
    from playwright.sync_api import sync_playwright
    assets = Path(__file__).resolve().parents[1] / 'silk/signals/view/assets'

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/':
                body = ('<!doctype html><link rel="stylesheet" href="/view.css">'
                        '<link rel="stylesheet" href="/uplot.min.css">'
                        '<script src="/uplot.iife.min.js"></script>'
                        '<div id="viewer" style="width:1400px;height:900px"></div>').encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

    server = ThreadingHTTPServer(('127.0.0.1', 0),
                                 functools.partial(Handler, directory=str(assets)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    screenshots = Path(screenshots)
    screenshots.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as driver:
            firefox = 'firefox' in Path(executable).name.lower()
            engine = driver.firefox if firefox else driver.chromium
            env = dict(os.environ)
            if firefox:
                env['MOZ_DISABLE_CONTENT_SANDBOX'] = '1'
            browser = engine.launch(executable_path=executable, headless=True, env=env)
            try:
                for sections in (False, True):
                    page = browser.new_page(viewport={'width': 1450, 'height': 1000})
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto(f'http://127.0.0.1:{server.server_port}/')
                    page.evaluate("""async ({payload, sections}) => {
                        const {createExplorer} = await import('/explorer.mjs');
                        const core = await import('/waveform-core.mjs');
                        window.copied = []; window.kept = [];
                        Object.defineProperty(navigator, 'clipboard', {
                            value: {writeText: async text => copied.push(text)}});
                        window.viewer = createExplorer(document.getElementById('viewer'), {
                            uPlot, waveformCore: core, groupSignalsByKind: sections,
                            contextMenu: {items: [{id:'keep', label:'Keep trace',
                                onSelect: identity => kept.push(identity)}]}});
                        await viewer.load(payload);
                    }""", {'payload': payload, 'sections': sections})
                    page.wait_for_selector('.u-over')

                    def command(cmd, args=None):
                        return page.evaluate("""async ({cmd, args}) => {
                            const result = await viewer.run({cmd, args});
                            if (!result.ok) throw new Error(JSON.stringify(result));
                            return result.results[0];
                        }""", {'cmd': cmd, 'args': args or {}})

                    if sections:
                        page.locator('[data-silk="picker-list"]').click()
                        page.locator('.group-row[title="kind:current"] .ptwisty').click()
                    page.wait_for_function("""() => [...document.querySelectorAll('.group-row .pname')]
                        .some(e => e.textContent.toLowerCase().includes('stim[0]'))""")
                    logical = next((s for s in payload['plots'][0]['signals']
                                    if (s.get('node_probe') or {}).get('name') == 'i(stim[0])'), None)
                    assert logical is not None
                    logical_row = '.picker-row[data-name=' + json.dumps(logical['name']) + ']'
                    assert page.locator(logical_row).is_visible()
                    assert 'hybrid-row' in page.locator(logical_row).get_attribute('class')
                    # Initial state, before any expansion or manual collapse.
                    for card in payload['plots'][0]['signals']:
                        if (card.get('node_probe') or {}).get('name', '').startswith('i(Vstim[0]__silk_'):
                            selector = '.picker-row[data-name=' + json.dumps(card['name']) + ']'
                            assert page.locator(selector).count() == 0
                    page.locator(logical_row).click(button='right')
                    page.locator('[data-item="copy-silk-code"]').click()
                    page.wait_for_function('copied.length === 1')
                    assert page.evaluate('copied.pop()') == "tb.probe('i(stim[0])', 'I')"
                    page.locator(logical_row).click(button='right')
                    page.locator('[data-item="keep"]').click()
                    assert page.evaluate('kept.pop().probe.name') == 'i(stim[0])'
                    logical_sources = [s for s in payload['plots'][0]['signals']
                                       if (s.get('node_probe') or {}).get('terms')]
                    assert len(logical_sources) == 3
                    command('expandAll')
                    for source in logical_sources:
                        selector = '.picker-row[data-name=' + json.dumps(source['name']) + ']'
                        assert page.locator(selector).is_visible()
                        page.locator(selector + ' .ptwisty').click()
                    command('addPanel', {'signals': [s['name'] for s in logical_sources]})
                    command('removePanel', {'panel': 0})
                    page.screenshot(path=str(screenshots / ('opened-sections.png' if sections
                                                           else 'opened-hierarchy.png')))
                    current = next(s for s in payload['plots'][0]['signals']
                                   if (s.get('node_probe') or {}).get('name') == 'i(Vstim[0]__silk_dc)')
                    row = '.picker-row[data-name=' + json.dumps(current['name']) + ']'
                    assert page.locator(row).count() == 0
                    command('expandAll')
                    page.wait_for_selector(row)
                    keys = page.locator('.group-row').evaluate_all('(rows) => rows.map(row => row.title)')
                    physical = [key.lower().replace('<', '[').replace('>', ']') for key in keys]
                    assert len(set(physical)) == len(physical), keys
                    group_text = page.locator('.group-row .pname').all_text_contents()
                    for source in ('stim[0]', 'bias[0]', 'inner[1]'):
                        assert any(source.lower() in text.lower() for text in group_text)
                    cards = [s for s in payload['plots'][0]['signals']
                             if (s.get('node_probe') or {}).get('type') == 'I'
                             and '__silk_' in s['node_probe']['name']]
                    assert len(cards) == 6
                    for card in cards:
                        selector = '.picker-row[data-name=' + json.dumps(card['name']) + ']'
                        assert page.locator(selector).is_visible(), card['name']
                        assert page.locator(selector + ' .pindent').evaluate(
                            'e => parseFloat(e.style.width) > 0')
                    page.locator(row).click(button='right')
                    page.locator('[data-silk="derive-menu"]:not([hidden])').wait_for()
                    page.locator('[data-item="keep"]').click()
                    identity = page.evaluate('kept[0]')
                    assert identity['vector'] == (current.get('raw') or current['name'])
                    assert identity['probe'] == {'name': current['node_probe']['name'],
                                                 'type': 'I', 'transform': None, 'analysis': None}
                    page.locator(row).click(button='right')
                    page.locator('[data-silk="derive-menu"]:not([hidden])').wait_for()
                    page.locator('[data-item="copy-silk-code"]').click()
                    page.wait_for_function('copied.length === 1')
                    assert page.evaluate('copied[0]') == "tb.probe('i(Vstim[0]__silk_dc)', 'I')"
                    command('addPanel', {'signals': [card['name'] for card in cards]})
                    command('removePanel', {'panel': 0})
                    page.evaluate('() => new Promise(requestAnimationFrame)')
                    page.screenshot(path=str(screenshots / ('source-currents-sections.png' if sections
                                                           else 'source-currents-hierarchy.png')))
                    command('setTreeMode', {'on': False})
                    for card in cards:
                        assert page.locator('.picker-row[data-name=' + json.dumps(card['name']) + ']').is_visible()
                    assert errors == []
                    page.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.slow
@needs_ngspice
def test_source_groups_and_trace_menu_in_browser(tmp_path):
    from tests._env import browser_executable
    executable = browser_executable()
    if not executable or not Path(executable).is_file():
        pytest.skip('configured unpacked browser required')
    pytest.importorskip('playwright.sync_api', reason='optional Playwright browser driver required')
    tb, probes = source_bench(tmp_path)
    ctx = view_node_context(tb)
    payload = emit_view_data(ctx.source, node_context=ctx, labels='designer')
    exercise_source_browser(payload, executable, tmp_path)


@pytest.mark.parametrize('names,proven', [
    (('i(vstim<0>__silk_pwl)',), True),
    (('i(vstim<0>__silk_pwl)', 'i(vstim[0]__silk_pwl)'), False),
    (('i(vstim<0>__silk_pwl)', 'vstim[0]__silk_pwl'), False),
    (('i(vstim<0>__silk_pwl)', 'i(vstim<0>__silk_pwl)'), False),
])
def test_card_probe_refuses_literal_alias_and_duplicate_rows(tmp_path, names, proven):
    from silk.circuit.view_nodes import _capture_addresses, _proof
    from tests.test_view_node_paths import rawfile
    mod = Module(None, str(tmp_path), 'source.cir', None)
    mod.add_source(sources.VoltageSource('stim[0]', positive='in', negative='0',
                                        dc=.2, pwl=[0, .5, 1, 1]))
    raw = rawfile(tmp_path, names)
    _, proofs = _proof(raw, _capture_addresses(mod))
    current_probes = [probe for _, _, probe in proofs[0] if probe and probe.startswith('i(')]
    assert bool(current_probes) is proven
