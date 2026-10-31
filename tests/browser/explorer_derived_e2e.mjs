// Derived traces use full-resolution operands and the ordinary viewer consumers.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';

const [html, chrome] = process.argv.slice(2);
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
const browser = await puppeteer.launch({executablePath: chrome, headless: true,
    args: ['--no-sandbox', '--disable-gpu']});
try {
    const page = await browser.newPage();
    await page.setViewport({width: 1400, height: 1000});
    const errors = [];
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
        window.chart = () => charts.find(u => u.root.isConnected);
        window.downloads = [];
        const create = URL.createObjectURL.bind(URL);
        URL.createObjectURL = blob => { downloads.push(blob); return create(blob); };
        HTMLAnchorElement.prototype.click = function () {};
    });
    await page.goto(pathToFileURL(html).href);
    await page.waitForSelector('.u-over');
    assert.ok(await page.evaluate(() => silkView.commands().some(c => c.name === 'deriveSignal')),
        'deriveSignal is registered on the command bus');
    // Exercise the standalone shim's additive core wiring before the host tests.
    const standalone = await page.evaluate(async () => {
        const names = SILK_VIEW_DATA.plots[SILK_VIEW_DATA.default_plot].signals.map(s => s.name);
        return command('deriveSignal', {operation: 'product', operands: names.slice(0, 2)});
    });
    assert.equal(standalone.created, true);
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async () => {
        const {createExplorer} = await import('/explorer.mjs');
        window.SILK_WAVEFORM_CORE = {...await import('/waveform-core.mjs')};
        window.silkView = createExplorer(document.getElementById('mount'), {uPlot, waveformCore: SILK_WAVEFORM_CORE});
        // Observe the actual pyramid supplied to Store without exposing test hooks.
        const build = SILK_WAVEFORM_CORE.build_pyramid;
        window.built = [];
        SILK_WAVEFORM_CORE.build_pyramid = (...args) => {
            const result = build(...args); built.push(result); return result;
        };
        window.loadFixture = async (n = 4) => {
            const x = Float64Array.from({length: n}, (_, i) => i);
            const signal = (name, unit, values) => ({name, unit, type: 'fixture', part: null,
                group: null, nonfinite: values.filter(v => !Number.isFinite(v)).length,
                levels: [{stride: 1, n, dt: 'f64', y: Float64Array.from({length: n}, (_, i) => values[i % values.length])}]});
            window.input = {format: 1, endian: 'le', run: {id: 'derived-' + n, title: 'Derived test'},
                default_plot: 0, annotations: {markers: [], bands: [], notes: []}, plots: [{
                    name: 'Transient', kind: 'transient', npoints: n, complex: false, x_log_default: false,
                    x: {name: 'time', unit: 's', levels: [{stride: 1, n, dt: 'f64', x}]}, signals: [
                        signal('v(top.a)', 'V', [2, -4, 6, 0]), signal('v(top.b)', 'V', [1, 2, -3, 0]),
                        signal('v(c)', 'V', [3, 2, 1, 4]), signal('i(load)', 'A', [2, 3, 4, 5]),
                        signal('tiny', 'A', [1e-30]), signal('huge', 'V', [1e30]),
                        signal('unknown', '', [2]), signal('phase', 'deg', [90]),
                        signal('sum(v(top.a), v(top.b))', 'V', [99]),
                    ]}, {
                    name: 'Other', kind: 'transient', npoints: n, complex: false, x_log_default: false,
                    x: {name: 'time', unit: 's', levels: [{stride: 1, n, dt: 'f64', x}]},
                    signals: [signal('other', 'V', [7])],
                }]};
            await silkView.load(input);
            await command('clearPanel', {panel: 0});
        };
        await loadFixture();
        window.derive = (operation, operands, extra = {}) => command('deriveSignal', {operation, operands, ...extra});
    });
    const a = 'v(top.a)', b = 'v(top.b)', c = 'v(c)';
    for (const [op, operands, want, unit] of [
        ['sum', [a, b], [3, -2, 3, 0], 'V'], ['difference', [a, b], [1, -6, 9, 0], 'V'],
        ['product', [a, b], [2, -8, -18, 0], ''], ['quotient', [a, b], [2, -2, -2, 'NaN'], ''],
        ['mean', [a, b], [1.5, -1, 1.5, 0], 'V'],
        ['sum', [a, b, c], [6, 0, 4, 4], 'V'], ['product', [a, b, c], [6, -16, -18, 0], ''],
        ['mean', [a, b, c], [2, 0, Math.fround(4 / 3), Math.fround(4 / 3)], 'V'],
        ['product', [a, 'i(load)'], [4, -12, 24, 0], 'W'],
        ['quotient', [a, 'i(load)'], [1, Math.fround(-4 / 3), 1.5, 0], 'Ω'],
    ]) {
        const result = await page.evaluate(async ({op, operands}) => {
            const meta = await derive(op, operands);
            await command('clearPanel', {panel: 0});
            await command('addSignal', {panel: 0, signals: [meta.name]});
            const level = built.at(-1).levels[0];
            return {meta, y: Array.from(level.y, v => Number.isNaN(v) ? 'NaN' : v),
                plotted: Array.from(chart().data[1], v => v === null ? 'NaN' : v)};
        }, {op, operands});
        assert.deepEqual(result.y, want, op + operands);
        assert.deepEqual(result.plotted, want, 'Store/rendered ' + op);
        assert.equal(result.meta.unit, unit);
        assert.equal(result.meta.created, true);
        if (op === 'sum' && operands.length === 2) assert.notEqual(result.meta.name, 'sum(v(top.a), v(top.b))');
    }
    for (const op of ['sum', 'product', 'mean']) {
        const result = await page.evaluate(async ({op, a, b, c}) => {
            const first = await derive(op, [a, b, c]);
            const again = await derive(op, [c, a, b]);
            const repeated = await derive(op, [a, a, b]);
            return {first, again, repeated};
        }, {op, a, b, c});
        assert.equal(result.again.name, result.first.name);
        assert.equal(result.again.created, false);
        assert.notEqual(result.repeated.name, result.first.name, 'Operand multiplicity is part of identity');
    }
    for (const op of ['difference', 'quotient']) {
        const names = await page.evaluate(async ({op, a, b}) => [await derive(op, [a, b]), await derive(op, [b, a])], {op, a, b});
        assert.notEqual(names[0].name, names[1].name);
    }
    for (const [args, error] of [
        ...['sum', 'difference', 'mean'].map(operation => [{operation, operands: [a, 'i(load)']}, /same unit/i]),
        [{operation: 'difference', operands: [a, b, c]}, /exactly two/i],
        [{operation: 'quotient', operands: [a]}, /exactly two/i],
        [{operation: 'sum', operands: []}, /at least two/i],
        [{operation: 'sum', operands: [a, 2]}, /names|strings/i],
        [{operation: 'sum', operands: [a, 'other']}, /unknown signal.*other/i],
        [{operation: 'sum', operands: [a, b], plot: 7}, /plot.*7/i],
        [{operation: 'mystery', operands: [a, b]}, /operation/i],
    ]) {
        const result = await page.evaluate(args => silkView.run({cmd: 'deriveSignal', args}), args);
        assert.equal(result.ok, false, JSON.stringify(args));
        assert.match(result.error, error);
    }
    const edges = await page.evaluate(async () => {
        const ratio = await derive('quotient', ['huge', 'tiny']);
        const large = built.at(-1);
        const small = await derive('product', ['tiny', 'tiny']);
        const tiny = built.at(-1);
        const zero = await derive('quotient', ['v(c)', 'v(top.a)']);
        const nonfinite = built.at(-1).nonfinite;
        const nested = await derive('quotient', [ratio.name, ratio.name]);
        return {ratio, small, zero, nested, nonfinite, dt: large.levels[0].dt,
            tinyDt: tiny.levels[0].dt, unknown: await derive('product', ['unknown', 'huge']),
            phase: await derive('quotient', ['phase', 'phase'])};
    });
    assert.equal(edges.dt, 'f64');
    assert.equal(edges.tinyDt, 'f64');
    assert.equal(edges.nonfinite, 1);
    assert.equal(edges.nested.unit, '');
    assert.equal(edges.unknown.unit, '');
    assert.equal(edges.phase.unit, '');

    const chained = await page.evaluate(async () => {
        const squared = await derive('product', ['v(top.a)', 'v(top.b)']);
        const gain = await derive('quotient', ['v(top.a)', 'v(top.b)']);
        const mixed = await derive('sum', [squared.name, gain.name]);
        return {
            voltage: await derive('quotient', [squared.name, 'v(c)']),
            current: await derive('product', [gain.name, 'i(load)']),
            unknown: await derive('product', [mixed.name, 'i(load)']),
        };
    });
    assert.equal(chained.voltage.unit, 'V', 'Recover a known unit after an intermediate V squared');
    assert.equal(chained.current.unit, 'A', 'Dimensionless derived gain times current');
    assert.equal(chained.unknown.unit, '', 'Equal empty display units do not assert equal dimensions');

    // An explicitly chosen plot owns its definitions and can be visited later.
    const offplot = await page.evaluate(async () => {
        const meta = await derive('sum', ['other', 'other'], {plot: 1});
        const wrong = await silkView.run({cmd: 'addSignal', args: {signals: [meta.name]}});
        await command('setPlot', {index: 1});
        await command('addPanel', {signals: [meta.name]});
        const again = await derive('sum', ['other', 'other']);
        await command('setPlot', {index: 0});
        return {meta, wrong, again};
    });
    assert.equal(offplot.meta.unit, 'V');
    assert.equal(offplot.wrong.ok, false);
    assert.equal(offplot.again.created, false);

    // Literal derived names retain dots, even in filtered and flat pickers.
    for (const mode of [true, false]) {
        await page.evaluate(async mode => {
            await command('setTreeMode', {on: mode});
            await command('setFilter', {text: 'difference(v(top.a), v(top.b))'});
        }, mode);
        const picker = await page.$eval('[data-silk="picker-list"]', e => e.textContent);
        assert.ok(picker.includes('Derived signals'));
        if (mode) assert.match(picker, /— 1 of /, 'Filter count includes the derived match');
        assert.equal(await page.$eval('.picker-row .pname', e => e.textContent), 'difference(v(top.a), v(top.b))');
        await page.click('.picker-row', {clickCount: 2});
    }
    await page.evaluate(async () => {
        await command('setFilter', {text: ''});
        await command('setTreeMode', {on: true});
    });
    assert.ok(await page.$eval('[data-silk="picker-list"]', e => e.textContent.includes('Derived signals')));

    // Actual cursor/statistics/CSV and PNG consumers, plus independent axis units.
    await page.evaluate(async () => {
        await command('clearPanel', {panel: 0});
        const diff = await derive('difference', ['v(top.a)', 'v(top.b)']);
        const power = await derive('product', ['v(top.a)', 'i(load)']);
        const gain = await derive('quotient', ['v(top.a)', 'v(top.b)']);
        await command('addSignal', {panel: 0, signals: [diff.name, power.name, gain.name]});
        await command('placeCursor', {which: 'A', x: 1});
        await command('exportCsv', {scope: 'full', panel: 0});
        await command('exportPng', {scope: 'panel', panel: 0});
    });
    const rows = await page.$$eval('.panel-table tr', rows => rows.slice(1).map(r => Array.from(r.querySelectorAll('td'), c => c.textContent)));
    assert.ok(rows[0].includes('-6.000 V') && rows[0].includes('9.000 V') && rows[0].includes('1.000 V'), JSON.stringify(rows));
    assert.deepEqual(await page.$$eval('.panel-ymode', es => es.map(e => e.dataset.unit)), ['V', 'W', '']);
    const csv = await page.evaluate(async () => await downloads.find(b => b.type.startsWith('text/csv')).text());
    assert.match(csv, /1,-6,-12,-2/);
    await page.waitForFunction(() => downloads.some(b => b.type === 'image/png'));
    assert.ok(await page.evaluate(() => downloads.find(b => b.type === 'image/png').size > 1000));
    assert.equal(await page.evaluate(() => input.plots[0].signals.length), 9, 'Host payload has no derived signals');
    await page.screenshot({path: path.join(path.dirname(html), 'derived-signals.png')});

    // Guard ALL operand decimated level properties. Then compare the displayed
    // envelope to a separate, sorting-based reference over level-0 differences.
    const pyramid = await page.evaluate(async () => {
        await loadFixture(65536);
        window.decimatedReads = 0;
        for (const sig of input.plots[0].signals) {
            const decimated = {};
            for (const prop of ['stride', 'n', 'dt', 'x', 'y']) Object.defineProperty(decimated, prop, {
                get() { decimatedReads++; throw new Error('Read decimated operand ' + prop); },
            });
            sig.levels.push(decimated);
        }
        const result = await derive('difference', ['v(top.a)', 'v(top.b)']);
        const levels = built.at(-1).levels;
        await command('addSignal', {panel: 0, signals: [result.name]});
        const u = chart();
        const level = levels.find(l => l.n === u.data[0].length && l.stride > 1);
        if (!level) throw new Error('Expected a decimated stride');
        const rawA = input.plots[0].signals[0].levels[0].y;
        const rawB = input.plots[0].signals[1].levels[0].y;
        const expectedX = [], expectedY = [];
        for (let lo = 0; lo < rawA.length; lo += level.stride) {
            const bucket = Array.from({length: Math.min(level.stride, rawA.length - lo)}, (_, j) =>
                ({i: lo + j, v: rawA[lo + j] - rawB[lo + j]}));
            const min = bucket.slice().sort((a, b) => a.v - b.v || a.i - b.i)[0];
            const max = bucket.slice().sort((a, b) => b.v - a.v || a.i - b.i)[0];
            for (const p of [min, max].sort((a, b) => a.i - b.i)) { expectedX.push(p.i); expectedY.push(p.v); }
        }
        return {reads: decimatedReads, x: Array.from(u.data[0]), y: Array.from(u.data[1]), expectedX, expectedY};
    });
    assert.equal(pyramid.reads, 0);
    assert.deepEqual(pyramid.x, pyramid.expectedX);
    assert.deepEqual(pyramid.y, pyramid.expectedY);
    await page.evaluate(() => loadFixture());
    const missing = await page.evaluate(() => silkView.run({cmd: 'addSignal', args: {signals: ['difference(v(top.a), v(top.b))']}}));
    assert.equal(missing.ok, false, 'A fresh load without a state key discards unpersisted definitions');

    // Real picker selection must preserve count, direction and plotted values.
    const menu = '[data-silk="derive-menu"]';
    const pickerRow = name => '[data-silk="picker-list"] .picker-row[data-name=' + JSON.stringify(name) + ']';
    // TASK-159: the natural idiom -- plain-click the first row, ctrl-click the
    // rest. Before that fix a plain click cleared without selecting, so this
    // helper had to ctrl-click every name including the first.
    // Establish a selection deterministically. Consecutive clicks at one
    // position inside the double-click interval arrive with detail >= 2, which
    // handleRowClick deliberately ignores so double-click-to-plot cannot
    // disturb the selection -- so this helper clears with Escape and ctrl-clicks
    // rather than relying on a plain click landing. The click/ctrl-click idiom
    // itself is covered by its own regression below.
    async function select(names) {
        await page.focus('[data-silk="picker-list"]');
        await page.keyboard.press('Escape');
        await page.keyboard.down('Control');
        for (const name of names) await page.click(pickerRow(name));
        await page.keyboard.up('Control');
    }
    const selectedNames = () => page.$$eval(
        '[data-silk="picker-list"] .picker-row.selected',
        rows => rows.map(r => r.dataset.name).sort());
    async function openMenu(names) {
        await select(names);
        await page.click(pickerRow(names[0]), {button: 'right'});
        await page.waitForSelector(menu + ':not([hidden])', {timeout: 2000});
        // TASK-375 appends the clicked trace's items ([data-item]) below the derive entries.
        return page.$$eval(menu + ' button:not([data-item])', buttons => buttons.map(b => ({text: b.textContent, disabled: b.disabled})));
    }
    async function choose(label) {
        for (const button of await page.$$(menu + ' button')) {
            if (await button.evaluate(b => b.textContent) === label) { await button.click(); return; }
        }
        assert.fail('Missing menu entry: ' + label);
    }
    await page.evaluate(() => command('setTreeMode', {on: false}));
    // TASK-159 regression: a plain click must SELECT the clicked row, so both
    // orderings of the click/ctrl-click idiom behave the way a user expects.
    {
        await page.mouse.move(0, 0);
        await page.click(pickerRow(a));
        assert.deepEqual(await selectedNames(), [a], 'A plain click selects the clicked row');
        await page.click(pickerRow(b));
        assert.deepEqual(await selectedNames(), [b], 'A second plain click replaces the selection');
        await page.click(pickerRow(a));
        await page.keyboard.down('Control');
        await page.click(pickerRow(b));
        await page.keyboard.up('Control');
        assert.deepEqual(await selectedNames(), [a, b].sort(), 'click then ctrl-click selects both');
        await page.mouse.move(0, 0);
        await page.keyboard.down('Control');
        await page.click(pickerRow(a));
        await page.keyboard.up('Control');
        assert.deepEqual(await selectedNames(), [b], 'ctrl-click toggles the clicked row off');
        await new Promise(resolve => setTimeout(resolve, 600)); // outlive the dblclick interval
        await page.click(pickerRow(a));
        assert.deepEqual(await selectedNames(), [a], 'a plain click after ctrl-click replaces, not clears');
        await page.focus('[data-silk="picker-list"]');
        await page.keyboard.press('Escape');
        assert.deepEqual(await selectedNames(), [], 'Escape in the picker deselects all');
    }

    const binary = [
        [`Sum: ${a} + ${b}`, [3, -2, 3, 0]],
        [`Difference: ${a} − ${b}`, [1, -6, 9, 0]],
        [`Difference: ${b} − ${a}`, [-1, 6, -9, 0]],
        [`Product: ${a} × ${b}`, [2, -8, -18, 0]],
        [`Quotient: ${a} / ${b}`, [2, -2, -2, null]],
        [`Quotient: ${b} / ${a}`, [0.5, -0.5, -0.5, null]],
        [`Mean: (${a} + ${b}) / 2`, [1.5, -1, 1.5, 0]],
    ];
    for (const [label, expected] of binary) {
        await page.evaluate(() => command('clearPanel', {panel: 0}));
        const entries = await openMenu([a, b]);
        assert.deepEqual(entries, binary.map(([text]) => ({text, disabled: false})));
        await choose(label);
        assert.deepEqual(await page.evaluate(() => Array.from(chart().data[1])), expected, label);
        assert.equal(await page.$eval(menu, e => e.hidden), true);
    }
    const ternary = [
        [`Sum: ${a} + ${b} + ${c}`, [6, 0, 4, 4]],
        [`Product: ${a} × ${b} × ${c}`, [6, -16, -18, 0]],
        [`Mean: (${a} + ${b} + ${c}) / 3`, [2, 0, Math.fround(4 / 3), Math.fround(4 / 3)]],
    ];
    for (const [label, expected] of ternary) {
        await page.evaluate(() => command('clearPanel', {panel: 0}));
        const entries = await openMenu([a, b, c]);
        assert.deepEqual(entries, ternary.map(([text]) => ({text, disabled: false})));
        await choose(label);
        assert.deepEqual(await page.evaluate(() => Array.from(chart().data[1])), expected, label);
    }
    for (const names of [[a, 'i(load)'], [a, b, 'i(load)']]) {
        const entries = await openMenu(names);
        for (const entry of entries) assert.equal(entry.disabled, /^(Sum|Difference|Mean):/.test(entry.text));
        const reason = await page.$eval(menu, e => e.innerText);
        assert.match(reason, /same unit/i);
        assert.ok(reason.includes('i(load) [A]') && reason.includes(a + ' [V]'), reason);
        assert.match(await page.evaluate(() => document.activeElement.textContent), /^Product:/, 'Keyboard skips disabled entries');
        await page.$eval(menu + ' button:disabled', b => b.click());
        assert.equal(await page.$eval(menu, e => e.hidden), false, 'Disabled entry cannot create a trace');
        if (names.length === 3) await page.screenshot({path: path.join(path.dirname(html), 'derive-menu-units.png')});
        await page.keyboard.press('Escape');
    }

    // Compose, then repeat in reverse selection order: reveal the same trace
    // in the focused panel without changing the first panel or signal count.
    await page.evaluate(() => command('addPanel', {signals: []}));
    const diff = `difference(${a}, ${b})`;
    await openMenu([diff, c]);
    await choose(`Sum: ${diff} + ${c}`);
    const nested = await page.evaluate(() => {
        const u = charts.filter(u => u.root.isConnected).at(-1);
        return {name: u.series[1].label, values: Array.from(u.data[1])};
    });
    assert.deepEqual(nested.values, [4, -4, 10, 4]);
    assert.deepEqual(await page.evaluate(() => Array.from(chart().data[1])), ternary[2][1], 'First panel stays unchanged');
    const count = await page.$eval('[data-silk="picker-count"]', e => e.textContent);
    await page.evaluate(() => command('clearPanel', {panel: 1}));
    await openMenu([c, diff]);
    await choose(`Sum: ${c} + ${diff}`);
    assert.equal(await page.$eval('[data-silk="picker-count"]', e => e.textContent), count, 'No duplicate definition');
    assert.deepEqual(await page.evaluate(() => {
        const u = charts.filter(u => u.root.isConnected).at(-1);
        return {name: u.series[1].label, values: Array.from(u.data[1])};
    }), nested, 'Existing derived trace is plotted again in the focused panel');

    await select([a, b]);
    await page.focus('[data-silk="picker-list"]');
    await page.keyboard.down('Shift');
    await page.keyboard.press('F10');
    await page.keyboard.up('Shift');
    assert.equal(await page.$eval(menu, e => !e.hidden && e.contains(document.activeElement)), true, 'Keyboard opens and focuses menu');
    await page.keyboard.press('ArrowDown');
    assert.equal(await page.evaluate(() => document.activeElement.textContent), binary[1][0]);
    await page.keyboard.press('End');
    assert.equal(await page.evaluate(() => document.activeElement.textContent), 'Copy as silk code', 'Trace items follow the derive entries');
    await page.keyboard.press('ArrowUp');
    assert.equal(await page.evaluate(() => document.activeElement.textContent), binary[6][0]);
    await page.keyboard.press('Home');
    assert.equal(await page.evaluate(() => document.activeElement.textContent), binary[0][0]);
    await page.keyboard.press('Escape');
    assert.equal(await page.$eval(menu, e => e.hidden), true);
    assert.equal(await page.evaluate(() => document.activeElement.dataset.silk), 'picker-list');
    await openMenu([a, b]);
    await page.click('[data-silk="picker-filter"]');
    assert.equal(await page.$eval(menu, e => e.hidden), true, 'Click outside menu dismisses it');
    await select([]);
    assert.equal(await page.$eval(pickerRow(a), row => {
        const event = new MouseEvent('contextmenu', {bubbles: true, cancelable: true});
        row.dispatchEvent(event); return event.defaultPrevented;
    }), true, 'An empty selection selects the row and opens the viewer menu');
    await page.keyboard.press('Escape');

    // TASK-374: one selected trace offers silk's single-trace operations.
    await page.evaluate(() => command('removePanel', {panel: 1}));
    const scaleLabel = name => `Scale: gain × ${name} + offset…`;
    const single = [
        [`Abs: |${a}|`, [2, 4, 6, 0]],
        [`Derivative: d(${a})/d(time)`, [-6, 2, 2, -6]],
        [`Integral: ∫ ${a} d(time)`, [0, -1, 0, 3]],
    ];
    for (const [label, expected] of single) {
        await page.evaluate(() => command('clearPanel', {panel: 0}));
        const entries = await openMenu([a]);
        assert.deepEqual(entries, [...single.map(([text]) => ({text, disabled: false})),
            {text: scaleLabel(a), disabled: false}]);
        // TASK-375: the clicked trace's items follow the single-trace entries.
        assert.deepEqual(await page.$$eval(menu + ' button[data-item]', bs => bs.map(b => b.textContent)),
            ['Copy as silk code']);
        await choose(label);
        assert.deepEqual(await page.evaluate(() => Array.from(chart().data[1])), expected, label);
        assert.equal(await page.$eval(menu, e => e.hidden), true);
    }
    const scaleForm = menu + ' [data-silk="derive-scale"]';
    async function scaleWith(name, gain, offset) {
        await openMenu([name]);
        await choose(scaleLabel(name));
        await page.waitForSelector(scaleForm);
        assert.equal(await page.$eval(menu, e => e.hidden), false, 'Scale keeps the menu open for its parameters');
        assert.equal(await page.evaluate(() => document.activeElement.dataset.silk), 'derive-scale-gain');
        await page.keyboard.type(gain);
        await page.keyboard.press('Tab');
        assert.equal(await page.evaluate(() => document.activeElement.dataset.silk), 'derive-scale-offset',
            'Tab moves between parameters without closing the menu');
        await page.$eval(menu + ' [data-silk="derive-scale-offset"]', e => e.select());
        await page.keyboard.type(offset);
        await page.keyboard.press('Enter');
    }
    await page.evaluate(() => command('clearPanel', {panel: 0}));
    await scaleWith(a, '-2', '1');
    assert.deepEqual(await page.evaluate(() => Array.from(chart().data[1])), [-3, 9, -11, 1], 'gain × a + offset');
    assert.equal(await page.$eval(menu, e => e.hidden), true);
    const scaled = `scale(${a}, gain=-2, offset=1)`;
    assert.equal(await page.evaluate(() => chart().series[1].label), scaled);
    await scaleWith(a, 'x', '0');
    assert.match(await page.$eval('[data-silk="toast-container"]', e => e.textContent), /gain must be a finite number/);
    assert.equal(await page.$eval(menu, e => e.hidden), false, 'Invalid parameters keep the form open');
    await page.keyboard.press('Escape');
    assert.equal(await page.$eval(menu, e => e.hidden), true);
    // A derived trace is an operand like any other.
    await page.evaluate(() => command('clearPanel', {panel: 0}));
    await openMenu([scaled]);
    await choose(`Derivative: d(${scaled})/d(time)`);
    assert.deepEqual(await page.evaluate(() => Array.from(chart().data[1])), [12, -4, -4, 12]);
    // A refused operation explains itself and creates nothing.
    const refused = await page.evaluate(() => silkView.run({cmd: 'deriveSignal', args: {operation: 'derivative', operands: ['v(top.a)', 'v(top.b)']}}));
    assert.equal(refused.ok, false);
    assert.match(refused.error, /exactly one operand/);
    await openMenu([a, b]);
    await page.evaluate(() => command('setPlot', {index: 1}));
    assert.equal(await page.$eval(menu, e => e.hidden), true, 'Plot changes dismiss stale operands');

    // A collapsed group uses its selected descendants. With every panel
    // removed, choosing from the menu creates a panel just like Add selected.
    await page.evaluate(async () => {
        await loadFixture();
        await command('setTreeMode', {on: true});
        await command('collapseAll');
        while (document.querySelector('.panel')) await command('removePanel', {panel: 0});
    });
    const group = '[data-silk="picker-list"] .group-row[title="top"]';
    await page.keyboard.down('Control');
    await page.click(group);
    await page.keyboard.up('Control');
    await page.click(group, {button: 'right'});
    await page.waitForSelector(menu + ':not([hidden])', {timeout: 2000});
    assert.equal((await page.$$(menu + ' button')).length, 7);
    await page.keyboard.press('Enter');
    assert.deepEqual(await page.evaluate(() => Array.from(chart().data[1])), [3, -2, 3, 0], 'Keyboard activation creates the missing panel');
    assert.equal((await page.$$('.panel')).length, 1);
    // A native AC payload: both output parts are drawable, the context menu
    // adds both. Numerical code-export parity is exercised by the Node gate.
    const ac = await page.evaluate(async () => {
        await silkView.load(await (await fetch('/ac.json')).json());
        await command('clearPanel', {panel: 0});
        const result = await derive('difference', ['v(in) (phase)', 'v(out)']);
        await command('addSignal', {panel: 0, signals: [result.name + ' (mag)', result.name + ' (phase)']});
        return result;
    });
    await page.waitForFunction(() => chart().series.length === 3);
    assert.deepEqual(await page.evaluate(() => chart().series.slice(1).map(s => s.label)),
        [ac.name + ' (mag)', ac.name + ' (phase)']);
    assert.ok(await page.evaluate(() => Array.from(chart().data[1]).every(Number.isFinite)));
    await page.evaluate(() => command('setYMode', {panel: 0, unit: 'V', mode: 'dB'}));
    await page.evaluate(() => command('setTreeMode', {on: false}));
    const mixedPartsMenu = await openMenu(['v(in) (mag)', 'v(out) (phase)']);
    assert.ok(mixedPartsMenu.filter(e => /Sum|Mean|Difference/.test(e.text)).every(e => !e.disabled),
        'AC menu uses vector units even for a phase part');
    await page.keyboard.press('Escape');
    await page.evaluate(() => command('setFilter', {text: 'v(out) (phase)'}));
    // Use the same single-trace menu path as real data: it must expand the
    // returned group name into the two actual trace names before drawing.
    const phaseRow = '[data-silk="picker-list"] .picker-row[data-name="v(out) (phase)"]';
    await page.waitForSelector(phaseRow);
    await openMenu(['v(out) (phase)']);
    const integralButton = await page.$$(menu + ' button');
    let clicked = false;
    for (const button of integralButton) {
        if ((await button.evaluate(e => e.textContent)).toLowerCase().includes('integral')) {
            await button.click(); clicked = true; break;
        }
    }
    assert.ok(clicked, 'AC part exposes integral context-menu operation');
    await page.waitForFunction(() => charts.some(u => u.root.isConnected &&
        u.series.some(s => s.label === 'integral(v(out)) (phase)')));
    assert.deepEqual(errors, []);
    console.log('Derived signal arithmetic, identity, units, picker, readouts, exports and level-0 pyramid: PASS');
} finally {
    await browser.close();
    server.close();
}
