// TASK-375: the trace context menu, "Copy as silk code" and the host menu hook.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import puppeteer from 'puppeteer-core';

const [directory, chrome] = process.argv.slice(2);
const server = http.createServer((req, res) => {
    if (req.url === '/') {
        res.setHeader('Content-Type', 'text/html');
        res.end('<!doctype html><link rel="stylesheet" href="/view.css"><script src="/uplot.iife.min.js"></script>' +
            '<div id="one" style="width:1200px;height:600px"></div><div id="two" style="width:1200px;height:600px"></div>');
        return;
    }
    const file = path.join(directory, path.basename(req.url));
    if (!fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.setHeader('Content-Type', file.endsWith('.css') ? 'text/css' : file.endsWith('.json') ? 'application/json' : 'text/javascript');
    res.end(fs.readFileSync(file));
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await puppeteer.launch({executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-gpu']});
try {
    const page = await browser.newPage();
    await page.setViewport({width: 1300, height: 1300});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.evaluateOnNewDocument(() => {
        window.copied = [];
        Object.defineProperty(navigator, 'clipboard', {value: {writeText: text => { copied.push(text); return Promise.resolve(); }}});
    });
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async () => {
        const {createExplorer} = await import('/explorer.mjs');
        window.createExplorer = createExplorer;
        window.core = await import('/waveform-core.mjs');
        window.payload = await (await fetch('/payload.json')).json();
        window.kept = [];
        window.one = createExplorer(document.getElementById('one'), {uPlot, waveformCore: core, groupSignalsByKind: false, contextMenu: {
            items: [{id: 'keep', label: 'Keep in test', onSelect: identity => kept.push(identity)},
                {label: 'Broken host item', onSelect: () => { throw new Error('host bug'); }}],
        }});
        await one.load(payload);
        window.two = createExplorer(document.getElementById('two'), {uPlot, contextMenu: {hide: ['copy-silk-code']}});
        await two.load(payload);
    });
    const row = (mount, name) => `#${mount} [data-silk="picker-list"] .picker-row[data-name=${JSON.stringify(name)}]`;
    // Trace items live in the derive menu, below its entries, for a selected trace.
    const traceMenu = mount => `#${mount} [data-silk="derive-menu"]`;
    async function select(mount, names) {
        await page.focus(`#${mount} [data-silk="picker-list"]`);
        await page.keyboard.press('Escape');
        await page.keyboard.down('Control');
        for (const name of names) await page.click(row(mount, name));
        await page.keyboard.up('Control');
    }
    async function openTraceMenu(mount, name, names = [name]) {
        await select(mount, names);
        await page.click(row(mount, name), {button: 'right'});
        await page.waitForSelector(traceMenu(mount) + ':not([hidden])', {timeout: 2000});
        return page.$$eval(traceMenu(mount) + ' button[data-item]', bs => bs.map(b => ({id: b.dataset.item, text: b.textContent})));
    }
    async function choose(mount, id) {
        await page.click(`${traceMenu(mount)} button[data-item="${id}"]`);
    }
    const toasts = mount => page.$$eval(`#${mount} .toast`, ts => ts.map(t => t.textContent));

    const selectedNames = () => page.$$eval('#one .picker-row.selected', rs => rs.map(r => r.dataset.name).sort());
    const expectedItems = ['copy-silk-code', 'keep', 'host-1'];
    async function checkMenu(names, multi = false, clicked = 'v(out)') {
        await page.waitForSelector(traceMenu('one') + ':not([hidden])', {timeout: 2000});
        assert.deepEqual(await selectedNames(), [...names].sort());
        const buttons = await page.$$eval(traceMenu('one') + ' button', bs => bs.map(b => b.dataset.item || b.textContent));
        assert.match(buttons[0], multi ? /^Sum: / : /^Abs: /);
        assert.deepEqual(buttons.slice(-3), expectedItems, 'Copy precedes host items after derive entries');
        assert.equal(await page.$eval(traceMenu('one'), m => m.contains(document.activeElement)), true);
        // Both selection and menu focus re-render rows; position at the row, not root top-left.
        assert.equal(await page.$eval(traceMenu('one'), (m, clicked) => {
            const rect = m.getBoundingClientRect();
            const root = document.getElementById('one').getBoundingClientRect();
            const row = document.querySelector('#one .picker-row[data-name=' + JSON.stringify(clicked) + ']').getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 && Math.abs(rect.top - row.bottom - 4) < 1 && rect.bottom <= root.bottom;
        }, clicked), true, 'Menu is anchored below the clicked row');
    }
    // TASK-536: empty or unrelated selection becomes exactly the clicked row.
    for (const previous of [[], ['v(step)', 'i(v1)']]) {
        await select('one', previous);
        await page.click(row('one', 'v(out)'), {button: 'right'});
        await checkMenu(['v(out)']);
        await page.keyboard.press('Escape');
        // Inspect cancellation too: opening a viewer menu must suppress native UI.
        await select('one', previous);
        assert.equal(await page.$eval(row('one', 'v(out)'), r => {
            const event = new MouseEvent('contextmenu', {bubbles: true, cancelable: true});
            r.dispatchEvent(event); return event.defaultPrevented;
        }), true);
        await checkMenu(['v(out)']);
        await page.keyboard.press('Escape');
    }
    // Already-selected rows retain the whole selection and clicked-trace identity.
    await openTraceMenu('one', 'v(step)', ['v(out)', 'v(step)']);
    await checkMenu(['v(out)', 'v(step)'], true, 'v(step)');
    await choose('one', 'keep');
    assert.equal(await page.evaluate(() => kept.pop().vector), 'v(step)');

    // Both keyboard gestures select a focused unselected row, or preserve its selection.
    for (const key of ['ContextMenu', 'Shift+F10']) {
        for (const previous of [[], ['v(step)', 'i(v1)'], ['v(out)', 'v(step)']]) {
            await select('one', previous);
            await page.keyboard.press('Home');
            const steps = await page.$eval(row('one', 'v(out)'), r => Number(r.dataset.index));
            for (let i = 0; i < steps; i++) await page.keyboard.press('ArrowDown');
            if (key === 'Shift+F10') await page.keyboard.down('Shift');
            await page.keyboard.press(key === 'Shift+F10' ? 'F10' : key);
            if (key === 'Shift+F10') await page.keyboard.up('Shift');
            const multi = previous.includes('v(out)');
            await checkMenu(multi ? previous : ['v(out)'], multi);
            await choose('one', 'keep');
            assert.equal(await page.evaluate(() => kept.pop().vector), 'v(out)');
        }
    }
    // Built-in item first, then the host's items under their own labels,
    // all after the single-trace derive entries.
    assert.deepEqual(await openTraceMenu('one', 'v(out)'), [
        {id: 'copy-silk-code', text: 'Copy as silk code'}, {id: 'keep', text: 'Keep in test'},
        {id: 'host-1', text: 'Broken host item'}]);
    const all = await page.$$eval(traceMenu('one') + ' button', bs => bs.map(b => b.dataset.item || b.textContent));
    assert.match(all[0], /^Abs: /);
    assert.deepEqual(all.slice(-3), ['copy-silk-code', 'keep', 'host-1'], 'Trace items come last');
    await choose('one', 'copy-silk-code');
    await page.waitForFunction(() => copied.length === 1);
    assert.deepEqual(await page.evaluate(() => copied), ["tb.probe('out')"]);
    assert.equal(await page.$eval(traceMenu('one'), m => m.hidden), true);

    // The host callback receives the same identity record the API returns.
    await openTraceMenu('one', 'i(v1)');
    await choose('one', 'keep');
    const kept = await page.evaluate(() => ({kept, api: one.traceIdentity('i(v1)')}));
    assert.deepEqual(kept.kept, [kept.api]);
    assert.equal(kept.api.kind, 'raw');
    assert.equal(kept.api.vector, 'i(v1)');
    assert.deepEqual(kept.api.probe, {name: 'v1', type: 'I', transform: null, analysis: null});
    assert.equal(kept.api.source.analysis, 'tran');
    assert.equal(kept.api.source.plot, 0);
    assert.equal(await page.evaluate(() => one.silkCode(one.traceIdentity('i(v1)')).code), "tb.probe('v1', 'I')");

    // A throwing host callback is reported, not fatal.
    await openTraceMenu('one', 'v(out)');
    await choose('one', 'host-1');
    assert.ok((await toasts('one')).some(t => /Broken host item failed: host bug/.test(t)));

    // Derived trace: its identity carries operation and operand identities.
    const derived = await page.evaluate(async () => {
        const result = await one.run({cmd: 'deriveSignal', args: {operation: 'difference', operands: ['v(step)', 'v(out)']}});
        return result.results[0].name;
    });
    // Empty list space and section headers keep native menus, even with a selection.
    for (const selector of ['#one [data-silk="picker-list"]', '#one .flat-header']) {
        assert.equal(await page.$eval(selector, el => {
            const event = new MouseEvent('contextmenu', {bubbles: true, cancelable: true});
            el.dispatchEvent(event); return event.defaultPrevented;
        }), false);
    }
    await openTraceMenu('one', derived);
    await choose('one', 'copy-silk-code');
    await page.waitForFunction(() => copied.length === 2);
    assert.equal(await page.evaluate(() => copied[1]), "tb.probe('step') - tb.probe('out')");
    await openTraceMenu('one', derived);
    await choose('one', 'keep');
    const record = await page.evaluate(() => kept.at(-1));
    assert.equal(record.kind, 'derived');
    assert.equal(record.operation, 'difference');
    assert.deepEqual(record.parameters, {});
    assert.deepEqual(record.operands.map(o => o.vector), ['v(step)', 'v(out)']);
    assert.ok(JSON.parse(JSON.stringify(record)), 'identity is JSON-safe');

    // A multi-selection appends the clicked trace's items the same way.
    const items = await openTraceMenu('one', 'v(step)', ['v(out)', 'v(step)']);
    assert.deepEqual(items.map(i => i.id), ['copy-silk-code', 'keep', 'host-1']);
    assert.ok(await page.$eval(traceMenu('one'), m => m.textContent.includes('Difference:')));
    await page.click('#one [data-silk="derive-menu"] button[data-item="copy-silk-code"]');
    await page.waitForFunction(() => copied.length === 3);
    assert.equal(await page.evaluate(() => copied[2]), "tb.probe('step')", 'Copies the clicked trace');

    // Hidden built-in and no host items: only the derive entries remain.
    assert.deepEqual(await openTraceMenu('two', 'v(out)'), []);
    assert.match(await page.$eval(traceMenu('two'), m => m.textContent), /Abs: /);
    await page.keyboard.press('Escape');

    // Mount-time validation: built-ins cannot be renamed or replaced.
    const refusals = await page.evaluate(() => {
        const bad = [
            {items: [{id: 'copy-silk-code', label: 'Copy as code', onSelect() {}}]},
            {hide: ['copy-python']},
            {items: [{label: '', onSelect() {}}]},
            {items: [{label: 'x'}]},
            {items: [{label: 'x', onSelect() {}, title: 'y'}]},
            {labels: {'copy-silk-code': 'Copy'}},
            {items: [{id: 'a', label: 'x', onSelect() {}}, {id: 'a', label: 'y', onSelect() {}}]},
            [],
        ];
        return bad.map(contextMenu => {
            const div = document.createElement('div');
            try { createExplorer(div, {uPlot, contextMenu}); return 'mounted'; } catch (e) {
                return e.constructor.name + ': ' + e.message + (div.className ? ' (dirty)' : '');
            }
        });
    });
    assert.match(refusals[0], /TypeError: .*built-in item.*cannot be replaced or renamed/);
    for (const r of refusals) assert.match(r, /^TypeError: (?!.*dirty)/, r);

    // The no-equivalent path reports and copies nothing.
    const reason = await page.evaluate(async () => {
        const noisy = structuredClone(payload);
        noisy.plots[0].signals[0].name = 'onoise_spectrum';
        await one.load(noisy);
        return one.silkCode(one.traceIdentity('onoise_spectrum'));
    });
    assert.equal(reason.ok, false);
    await openTraceMenu('one', 'onoise_spectrum');
    await choose('one', 'copy-silk-code');
    await page.waitForFunction(() => document.querySelectorAll('#one .toast').length >= 2);
    assert.ok((await toasts('one')).some(t => /No silk code for onoise_spectrum: .*noise/.test(t)));
    assert.equal(await page.evaluate(() => copied.length), 3, 'Nothing copied for an unmappable trace');
    assert.deepEqual(errors, []);
} finally {
    await browser.close();
    server.close();
}
