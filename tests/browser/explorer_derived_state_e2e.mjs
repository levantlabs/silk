// Persist definitions through the same browser state/file paths as panel layouts.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {pathToFileURL} from 'node:url';
import puppeteer from 'puppeteer-core';

const [html, raw, chrome] = process.argv.slice(2);
const browser = await puppeteer.launch({executablePath: chrome, headless: true,
    args: ['--no-sandbox', '--disable-gpu']});
const server = http.createServer((req, res) => {
    if (req.url === '/') {
        res.setHeader('Content-Type', 'text/html');
        res.end('<!doctype html><link rel="stylesheet" href="/view.css"><script src="/uplot.iife.min.js"></script><div id="mount" style="width:1400px;height:1000px"></div>');
        return;
    }
    const file = path.join(path.dirname(html), path.basename(req.url));
    if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.setHeader('Content-Type', file.endsWith('.css') ? 'text/css' : 'text/javascript');
    res.end(fs.readFileSync(file));
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const errors = [];
async function newPage(context = browser) {
    const page = await context.newPage();
    await page.setViewport({width: 1400, height: 1000});
    page.on('pageerror', e => errors.push(e.message));
    await page.evaluateOnNewDocument(() => {
        let ctor;
        window.charts = [];
        Object.defineProperty(window, 'uPlot', {get: () => ctor, set: value => {
            ctor = new Proxy(value, {construct(target, args) {
                const u = new target(...args); charts.push(u); return u;
            }});
        }});
        window.command = async (cmd, args = {}) => {
            const result = await silkView.run({cmd, args});
            if (!result.ok) throw new Error(JSON.stringify(result));
            return result.results[0];
        };
        window.panelData = () => Array.from(document.querySelectorAll('.panel'), panel => {
            const u = charts.find(u => u.root.isConnected && panel.contains(u.root));
            return u ? {names: u.series.slice(1).map(s => s.label), data: u.data.map(a => Array.from(a))} : null;
        });
        window.downloads = [];
        const create = URL.createObjectURL.bind(URL);
        URL.createObjectURL = blob => { downloads.push(blob); return create(blob); };
        HTMLAnchorElement.prototype.click = function () {};
    });
    return page;
}
const stateOf = page => page.evaluate(() => command('getState'));
const dataOf = page => page.evaluate(() => panelData());
async function drop(page, name, bytes) {
    await page.evaluate(({name, bytes}) => {
        const transfer = new DataTransfer();
        transfer.items.add(new File([Uint8Array.from(bytes)], name));
        document.querySelector('.silk-explorer').dispatchEvent(new DragEvent('drop', {bubbles: true, dataTransfer: transfer}));
    }, {name, bytes: Array.from(bytes)});
}
async function loadView(page, state) {
    await page.evaluate(state => command('loadViewState', {state}), state);
}
try {
    const page = await newPage();
    const url = pathToFileURL(html).href;
    await page.goto(url);
    await page.waitForSelector('.u-over');
    const payload = await page.evaluate(() => SILK_VIEW_DATA);
    const names = payload.plots[0].signals.map(s => s.name);
    const [a, b] = names;
    const setup = await page.evaluate(async ({a, b}) => {
        const first = await command('deriveSignal', {operation: 'difference', operands: [a, b]});
        const nested = await command('deriveSignal', {operation: 'product', operands: [first.name, a]});
        const unused = await command('deriveSignal', {operation: 'mean', operands: [a, b]});
        await command('clearPanel', {panel: 0});
        await command('addSignal', {panel: 0, signals: [first.name, a]});
        await command('addPanel', {signals: [nested.name, b]});
        await command('setCols', {n: 2});
        await command('setYRange', {panel: 0, min: -1, max: 2});
        await command('setSeriesStyle', {panel: 0, signal: first.name, color: '#ff8800', width: 3});
        await command('placeCursor', {which: 'A', x: 0});
        await command('saveViewState');
        return {first, nested, unused, envelope: JSON.parse(await downloads.at(-1).text())};
    }, {a, b});
    const state = await stateOf(page), data = await dataOf(page);
    assert.deepEqual(setup.envelope.view, state, 'Download and getState share one serializer');
    assert.deepEqual(state.plots['0'].derived, [
        {name: setup.first.name, operation: 'difference', operands: [a, b]},
        {name: setup.nested.name, operation: 'product', operands: [setup.first.name, a].sort()},
        {name: setup.unused.name, operation: 'mean', operands: [a, b].sort()},
    ]);
    await page.waitForFunction(expected => {
        const saved = JSON.parse(localStorage.getItem('silk.view.' + SILK_VIEW_DATA.run.state_key));
        return JSON.stringify(saved?.plots) === JSON.stringify(expected.plots);
    }, {}, state);
    // A genuinely fresh page reads localStorage and resolves every panel name.
    const fresh = await newPage();
    await fresh.goto(url);
    await fresh.waitForSelector('.u-over');
    assert.deepEqual(await stateOf(fresh), state);
    assert.deepEqual(await dataOf(fresh), data);
    assert.equal((await fresh.evaluate(args => command('deriveSignal', args), {
        operation: 'difference', operands: [a, b],
    })).created, false, 'Restored definitions remain deduplicated');

    await fresh.close();

    // Import the actual exported file in an isolated browser context, with no
    // localStorage fallback. Deliberately shuffle definitions child-before-parent.
    const context = await browser.createBrowserContext();
    const imported = await newPage(context);
    await imported.goto(url);
    await imported.waitForSelector('.u-over');
    const envelope = structuredClone(setup.envelope);
    envelope.view.plots['0'].derived.reverse();
    const file = path.join(path.dirname(html), 'roundtrip.silkview.json');
    fs.writeFileSync(file, JSON.stringify(envelope));
    await (await imported.$('[data-silk="view-file-input"]')).uploadFile(file);
    await imported.waitForFunction(() => document.querySelector('[data-silk="toast-container"]').textContent.includes('view loaded from roundtrip'));
    assert.deepEqual(await dataOf(imported), data);
    const importedState = await stateOf(imported);
    assert.deepEqual(importedState.plots['0'].panels, state.plots['0'].panels);
    assert.equal(importedState.cols, 2);
    assert.deepEqual(importedState.plots['0'].derived.map(d => d.name).sort(), state.plots['0'].derived.map(d => d.name).sort());

    // A local filename owns its definitions, even when the next file has the
    // exact same bytes and operand names. Returning restores that file's view.
    const bytes = fs.readFileSync(raw);
    await drop(imported, 'first.raw', bytes);
    await imported.waitForFunction(() => SILK_VIEW_DATA.run.source === 'first.raw');
    await loadView(imported, setup.envelope);
    const localState = await stateOf(imported), localData = await dataOf(imported);
    await drop(imported, 'second.raw', bytes);
    await imported.waitForFunction(() => SILK_VIEW_DATA.run.source === 'second.raw');
    assert.equal((await stateOf(imported)).plots['0'].derived, undefined);
    assert.equal(await imported.$eval('[data-silk="picker-list"]', e => e.textContent.includes('Derived signals')), false);
    await drop(imported, 'first.raw', bytes);
    await imported.waitForFunction(() => SILK_VIEW_DATA.run.source === 'first.raw');
    assert.deepEqual(await stateOf(imported), localState);
    assert.deepEqual(await dataOf(imported), localData);
    await imported.reload();
    await imported.waitForSelector('.u-over');
    await drop(imported, 'first.raw', bytes);
    await imported.waitForFunction(() => SILK_VIEW_DATA.run.source === 'first.raw');
    assert.deepEqual(await stateOf(imported), localState);
    assert.deepEqual(await dataOf(imported), localData);
    await context.close();

    await page.bringToFront();

    // A host lets us change the save list and core capability while preserving
    // the public payload format. The source remains free of view definitions.
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async payload => {
        const {createExplorer} = await import('/explorer.mjs');
        window.core = await import('/waveform-core.mjs');
        window.input = payload;
        input.plots.push(structuredClone(input.plots[0]));
        window.mount = async (viewState, withCore = true) => {
            window.silkView?.destroy();
            window.silkView = createExplorer(document.getElementById('mount'), {
                uPlot, waveformCore: withCore ? core : undefined, viewState, stateKey: 'derived-state-test',
            });
            await silkView.load(input);
        };
        await mount();
    }, payload);
    const offplot = await page.evaluate(async ({a, b}) => {
        const result = await command('deriveSignal', {plot: 1, operation: 'sum', operands: [a, b]});
        return result;
    }, {a, b});
    // No getState or panel action to trigger the save: creation alone persists.
    await page.waitForFunction(name => JSON.parse(localStorage.getItem('silk.view.derived-state-test'))?.plots['1']?.derived?.[0]?.name === name, {polling: 100}, offplot.name);
    await page.evaluate(() => mount());
    await page.evaluate(async name => {
        await command('setPlot', {index: 1});
        await command('clearPanel', {panel: 0});
        await command('addSignal', {panel: 0, signals: [name]});
    }, offplot.name);
    assert.deepEqual((await stateOf(page)).plots['1'].panels[0].signals, [offplot.name]);
    await loadView(page, state);
    await page.evaluate(() => silkView.reload(input));
    assert.deepEqual(await dataOf(page), data, 'Host reload recomputes before panels');
    assert.equal(await page.evaluate(() => input.plots.some(p => p.signals.some(s => s.derived))), false);

    // Renaming is not a creation feature; a saved collision suffix must still
    // be honored when the conflicting payload trace has disappeared.
    const renamed = structuredClone(state);
    const def = renamed.plots['0'].derived[0], oldName = def.name;
    def.name += ' [2]';
    renamed.plots['0'].derived[1].operands = renamed.plots['0'].derived[1].operands.map(n => n === oldName ? def.name : n);
    renamed.plots['0'].panels = [{signals: [def.name, renamed.plots['0'].derived[1].name], yMode: 'linear'}];
    await loadView(page, renamed);
    assert.deepEqual((await stateOf(page)).plots['0'].panels[0].signals, renamed.plots['0'].panels[0].signals);
    assert.deepEqual((await dataOf(page))[0].data.slice(1), [data[0].data[1], data[1].data[1]]);

    // Changed operands drop the definition and dependent, while an independent
    // definition and ordinary panel reference survive with a visible diagnosis.
    const stale = structuredClone(state);
    stale.plots['0'].derived[0].operands[0] = 'missing-from-save-list';
    await loadView(page, stale);
    assert.deepEqual((await stateOf(page)).plots['0'].derived.map(d => d.name), [setup.unused.name]);
    assert.deepEqual((await stateOf(page)).plots['0'].panels.map(p => p.signals), [[a], [b]]);
    assert.match(await page.$eval('[data-silk="toast-container"]', e => e.textContent), /dropped: missing or cyclic operands: missing-from-save-list/);
    await loadView(page, state);
    await page.evaluate(a => {
        const changed = structuredClone(input);
        changed.plots[0].signals = changed.plots[0].signals.filter(s => s.name !== a);
        return silkView.reload(changed);
    }, a);
    assert.equal((await stateOf(page)).plots['0'].derived?.length || 0, 0);
    assert.match(await page.$eval('[data-silk="toast-container"]', e => e.textContent), /Saved derived signal.*dropped:/);

    // Unsupported engines, malformed lists, invalid operations and cycles all
    // degrade visibly and leave valid ordinary traces usable.
    await page.evaluate(async state => {
        await mount(undefined, false);
        await command('loadViewState', {state});
    }, state);
    assert.deepEqual((await stateOf(page)).plots['0'].panels.map(p => p.signals), [[a], [b]]);
    assert.match(await page.$eval('[data-silk="toast-container"]', e => e.textContent), /matching waveformCore/);
    await page.evaluate(() => mount(undefined, true));
    for (const [definitions, message] of [
        ['invalid', /expected an array/],
        [null, /expected an array/],
        [[null], /invalid definition/],
        [[{name: 'bad-op', operation: 'unknown', operands: [a, b]}], /operation must be/],
        [[{name: 'bad-units', operation: 'sum', operands: [a, names[2]]}], /same unit/],
        [[{name: 'cycle-a', operation: 'sum', operands: ['cycle-b', a]},
          {name: 'cycle-b', operation: 'sum', operands: ['cycle-a', b]}], /cyclic operands/],
        [[{name: a, operation: 'sum', operands: [a, b]}], /name already exists/],
    ]) {
        await page.$eval('[data-silk="toast-container"]', e => e.replaceChildren());
        await loadView(page, {v: 1, plots: {'0': {derived: definitions, panels: [{signals: [a, b]}]}}});
        assert.deepEqual((await stateOf(page)).plots['0'].panels[0].signals, [a, b]);
        assert.equal((await stateOf(page)).plots['0'].derived?.length || 0, 0);
        assert.match(await page.$eval('[data-silk="toast-container"]', e => e.textContent), message);
    }
    await loadView(page, {v: 1, plots: {'8': {derived: [{name: 'gone', operation: 'sum', operands: [a, b]}]}}});
    assert.match(await page.$eval('[data-silk="toast-container"]', e => e.textContent), /plot is not in this run/);

    // Boot-time sidecar precedence and its undo restore complete definitions.
    await loadView(page, state);
    const prior = await stateOf(page);
    const embedded = structuredClone(state);
    embedded.cols = 1;
    embedded.plots['0'].derived = [embedded.plots['0'].derived[2]];
    embedded.plots['0'].panels = [{signals: [setup.unused.name], yMode: 'linear'}];
    const untouched = await page.evaluate(async embedded => {
        const before = JSON.stringify(embedded);
        await mount(embedded);
        return before === JSON.stringify(embedded);
    }, embedded);
    assert.equal(untouched, true, 'Restoration leaves host-owned view state intact');
    assert.deepEqual((await stateOf(page)).plots['0'].derived, embedded.plots['0'].derived);
    await page.click('.toast-action');
    assert.deepEqual(await stateOf(page), prior);
    assert.deepEqual(await dataOf(page), data);

    const legacy = {v: 1, cols: 2, plots: {'0': {panels: [{signals: [b, a], yMode: 'linear'}]}}};
    await loadView(page, state);
    await loadView(page, legacy);
    const legacyState = await stateOf(page);
    assert.deepEqual(legacyState.plots['0'].panels[0].signals, [b, a]);
    assert.equal(legacyState.plots['0'].derived, undefined);
    assert.equal(await page.$eval('[data-silk="picker-list"]', e => e.textContent.includes('Derived signals')), false);
    await loadView(page, state);
    await page.click('[data-silk="view-btn"]');
    await page.$$eval('[data-silk="view-menu"] button', buttons => buttons.find(b => b.textContent === 'Reset view to defaults').click());
    assert.equal((await stateOf(page)).plots['0'].derived, undefined);
    assert.deepEqual(errors, [], 'No uncaught browser errors');
    console.log('PASS: derived state localStorage/file round trips, dependency order, names, stale/legacy state, host reloads and rawfile isolation');
} finally {
    await browser.close();
    server.close();
}
