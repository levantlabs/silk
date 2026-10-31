// Real AC charts: unit grouping, independent scales, persisted settings and exports.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const [html, chrome, unitless_html] = process.argv.slice(2);
const browser = await puppeteer.launch({
    executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-gpu'],
});
try {
    const page = await browser.newPage();
    await page.setViewport({width: 1400, height: 1100});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.evaluateOnNewDocument(() => {
        let constructor;
        window.charts = [];
        Object.defineProperty(window, 'uPlot', {
            configurable: true,
            get: () => constructor,
            set: value => {
                constructor = new Proxy(value, {construct(target, args) {
                    const chart = new target(...args);
                    charts.push(chart);
                    return chart;
                }});
            },
        });
        window.command = async (cmd, args = {}) => {
            const result = await silkView.run([{cmd, args: {panel: 0, ...args}}]);
            if (!result.ok) throw new Error(JSON.stringify(result));
            return result.results[0];
        };
        window.chart = () => {
            const panel = document.querySelector('.panel');
            return charts.find(chart => chart.root.isConnected && panel.contains(chart.root));
        };
        window.snapshot = () => {
            const panel = document.querySelector('.panel'), u = chart();
            return u.axes.slice(1).map(axis => {
                const group = panel.querySelectorAll('.panel-axis-range')[u.axes.indexOf(axis) - 1] || panel;
                const select = group.querySelector('.panel-ymode');
                const scale = u.scales[axis.scale];
                return {
                    ticks: axis._values.filter(Boolean), width: axis._size, side: axis.side,
                    unit: select.dataset.unit, mode: select.value,
                    range: [scale.min, scale.max],
                    readout: Array.from(group.querySelectorAll('.editable-value'), el => el.textContent),
                    modes: Array.from(select.options, opt => opt.value),
                    signals: u.series.slice(1).filter(s => s.scale === axis.scale).map(s => s.label),
                };
            });
        };
        window.plot_signals = async signals => {
            await command('clearPanel');
            await command('addSignal', {signals});
            await command('autoscaleY');
        };
        // Intercept download blobs at the browser boundary, leaving production
        // CSV serialization and PNG compositing untouched.
        window.downloads = [];
        const create_url = URL.createObjectURL.bind(URL);
        URL.createObjectURL = blob => {
            downloads.push(blob);
            return create_url(blob);
        };
        HTMLAnchorElement.prototype.click = function () {};
    });
    await page.goto(pathToFileURL(html).href);
    await page.waitForSelector('.u-over');

    // Keep the single-unit tick/gutter regression: overlaid identical units
    // must render exactly as a lone signal with the same mode and range.
    const uniform = new Map();
    for (const [signals, unit, db] of [
        [['v(out) (mag)'], 'V', true],
        [['v(out) (mag)', 'v(in) (mag)'], 'V', true],
        [['v(out) (phase)'], 'deg', false],
        [['v(out) (phase)', 'v(in) (phase)'], 'deg', false],
    ]) {
        await page.evaluate(async signals => {
            await plot_signals(signals);
            await command('setYRange', {min: -0.002, max: 2000});
        }, signals);
        const [axis] = await page.evaluate(() => snapshot());
        assert.ok(axis.ticks.length > 1 && axis.ticks.every(tick => tick.endsWith(unit)));
        assert.deepEqual(axis.readout, ['-2.000 m' + unit, '2.000 k' + unit]);
        assert.equal(axis.modes.includes('dB'), db);
        if (uniform.has(unit)) {
            assert.deepEqual(axis.ticks, uniform.get(unit).ticks);
            assert.equal(axis.width, uniform.get(unit).width);
        } else uniform.set(unit, axis);
    }

    // Bode fixture: independent ranges must expose the tiny magnitude change
    // near 1 V, even though the phase spans almost two degrees below zero.
    await page.evaluate(() => plot_signals(['v(out) (mag)', 'v(out) (phase)']));
    let axes = await page.evaluate(() => snapshot());
    assert.deepEqual(axes.map(a => [a.unit, a.side]), [['V', 3], ['deg', 1]]);
    assert.ok(axes.every(a => a.ticks.length > 1 && a.ticks.every(t => t.endsWith(a.unit))));
    assert.ok(axes.every(a => a.readout.every(value => value !== '—')));
    const spans = await page.evaluate(() => {
        const u = chart();
        return u.series.slice(1).map((series, i) => {
            const ys = u.data[i + 1].filter(Number.isFinite);
            return Math.abs(u.valToPos(Math.max(...ys), series.scale) - u.valToPos(Math.min(...ys), series.scale)) /
                (u.bbox.height / devicePixelRatio);
        });
    });
    assert.ok(spans.every(span => span > 0.5), 'Both traces occupy >50% of plot height: ' + spans);
    assert.ok(axes[0].range[0] > 0 && axes[1].range[1] < 0, JSON.stringify(axes));
    await page.evaluate(() => command('setYMode', {unit: 'V', mode: 'dB'}));
    const db_span = await page.evaluate(() => {
        const u = chart(), ys = u.data[1].filter(Number.isFinite);
        return Math.abs(u.valToPos(Math.max(...ys), 'y') - u.valToPos(Math.min(...ys), 'y')) / (u.bbox.height / devicePixelRatio);
    });
    assert.ok(db_span > 0.5, 'dB magnitude occupies >50% of plot height: ' + db_span + ' ' + JSON.stringify(await page.evaluate(() => snapshot())));
    await page.evaluate(async () => {
        await command('setYMode', {unit: 'V', mode: 'dB'});
        await command('setYRange', {unit: 'V', min: 0.999, max: 1.001});
        await command('setYRange', {unit: 'deg', min: -2, max: 0});
    });
    axes = await page.evaluate(() => snapshot());
    assert.deepEqual(axes.map(a => a.mode), ['dB', 'linear']);
    assert.deepEqual(axes.map(a => a.range), [[0.999, 1.001], [-2, 0]]);
    assert.ok(new Set(axes[0].ticks).size > 1, 'dB ticks must distinguish small changes');
    assert.ok(axes[0].ticks.every(t => t.endsWith(' dB')));
    assert.ok(axes[1].ticks.every(t => t.endsWith('deg')));
    for (const args of [{unit: 'deg', mode: 'dB'}, {unit: 'missing', mode: 'linear'}]) {
        const result = await page.evaluate(args => silkView.run([{cmd: 'setYMode', args: {panel: 0, ...args}}]), args);
        assert.equal(result.ok, false, JSON.stringify(args));
    }

    // Real controls target their own axis, and Y autoscale leaves X intact.
    await page.select('.panel-axis-range[data-unit="V"] select', 'linear');
    await page.evaluate(() => command('setXRange', {min: 2000, max: 4000}));
    await page.click('.panel-axis-range[data-unit="deg"] .panel-yauto-btn');
    axes = await page.evaluate(() => snapshot());
    assert.deepEqual(axes[0].range, [0.999, 1.001]);
    assert.notDeepEqual(axes[1].range, [-2, 0]);
    assert.deepEqual(await page.evaluate(() => [chart().scales.x.min, chart().scales.x.max]), [2000, 4000]);
    await page.evaluate(async () => {
        await command('resetZoom');
        await command('setYRange', {unit: 'deg', min: -2, max: 0});
        await command('placeCursor', {which: 'A', x: 1000});
        await command('placeCursor', {which: 'B', x: 5000});
    });
    const cells = await page.$$eval('.panel:first-child .panel-table tr', rows => rows.slice(1).map(
        row => Array.from(row.querySelectorAll('td'), td => td.textContent),
    ));
    assert.ok(cells[0].slice(1).every(value => value.endsWith('V')), JSON.stringify(cells));
    assert.ok(cells[1].slice(1).every(value => value.endsWith('deg')), JSON.stringify(cells));
    assert.notEqual(cells[0][6], cells[0][7], 'Magnitude cursor values retain their small variation');
    assert.notEqual(cells[1][6], cells[1][7], 'Phase cursor values differ');

    // Observe actual annotation draw coordinates and compare them with the
    // appropriate axis. Markers and bands are supplied by emit_view.
    const annotations = await page.evaluate(async () => {
        const u = chart(), ctx = u.ctx, calls = [];
        const fill_text = ctx.fillText, fill_rect = ctx.fillRect;
        ctx.fillText = function (text, x, y, ...args) {
            if (text.startsWith('test-')) calls.push({text, y});
            return fill_text.call(this, text, x, y, ...args);
        };
        ctx.fillRect = function (x, y, w, h) {
            if (x === u.bbox.left && w === u.bbox.width) calls.push({band: true, y, h});
            return fill_rect.call(this, x, y, w, h);
        };
        u.redraw();
        await new Promise(requestAnimationFrame);
        ctx.fillText = fill_text;
        ctx.fillRect = fill_rect;
        return {calls, expected: [u.valToPos(0.9997, 'y', true) - 3, u.valToPos(-1, 'y2', true) - 3],
            band: [u.valToPos(-0.5, 'y2', true), Math.abs(u.valToPos(-1.5, 'y2', true) - u.valToPos(-0.5, 'y2', true))]};
    });
    // One redraw() can coincide with a frame uPlot had already queued, so a
    // label may be painted more than once inside the patch window. The claim
    // under test is which axis positions the labels land on, not how many
    // paints happened, so collapse repeats by label (first paint wins).
    const drawn = new Map();
    annotations.calls.filter(c => c.text).forEach(c => {
        if (!drawn.has(c.text)) drawn.set(c.text, c.y);
    });
    assert.deepEqual(Array.from(drawn.values()), annotations.expected);
    assert.ok(annotations.calls.some(c => c.band && Math.abs(c.y - annotations.band[0]) < 1e-8 && Math.abs(c.h - annotations.band[1]) < 1e-8));

    await page.evaluate(async () => {
        await command('exportCsv', {scope: 'full'});
        await command('exportPng', {scope: 'panel', mode: 'download'});
    });
    await page.waitForFunction(() => downloads.some(b => b.type === 'image/png'));
    const csv = await page.evaluate(() => downloads.find(b => b.type.startsWith('text/csv')).text());
    assert.match(csv.split('\n')[0], /frequency.*v\(out\) \(mag\).*v\(out\) \(phase\)/);
    const first = csv.trim().split('\n')[1].split(',').map(Number);
    assert.ok(first[1] > 0.999 && first[1] < 1 && first[2] < -0.3 && first[2] > -0.4, csv);
    // The exported chart region must preserve *all* canvas pixels, including
    // both gutters, with only the documented title and padding around it.
    const png = await page.evaluate(async () => {
        const bitmap = await createImageBitmap(downloads.find(b => b.type === 'image/png'));
        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width; canvas.height = bitmap.height;
        const ctx = canvas.getContext('2d'); ctx.drawImage(bitmap, 0, 0);
        const u = chart(), source = u.ctx.canvas, dpr = devicePixelRatio;
        const pad = Math.round(8 * dpr), top = pad + Math.round(22 * dpr);
        const actual = ctx.getImageData(pad, top, source.width, source.height).data;
        const expected = u.ctx.getImageData(0, 0, source.width, source.height).data;
        let gutters = 0, mismatch = 0;
        for (let y = u.bbox.top; y < source.height; y++) for (let x = 0; x < source.width; x++) {
            if (x >= u.bbox.left && x <= u.bbox.left + u.bbox.width) continue;
            const i = (y * source.width + x) * 4;
            if (expected[i + 3] !== 255) continue; // transparent canvas is composited over the export background
            gutters++;
            if (actual[i] !== expected[i] || actual[i + 1] !== expected[i + 1] || actual[i + 2] !== expected[i + 2]) mismatch++;
        }
        return {size: [bitmap.width, bitmap.height], expected_size: [source.width + 2 * pad, source.height + 2 * pad + Math.round(22 * dpr)], gutters, mismatch};
    });
    assert.deepEqual(png.size, png.expected_size);
    assert.ok(png.gutters > 50 && png.mismatch === 0, JSON.stringify(png));

    // State round trip, then old layouts without yAxes. The additive legacy
    // fields continue to describe the first axis for older consumers.
    await page.evaluate(() => command('setYMode', {unit: 'V', mode: 'dB'}));
    await page.waitForFunction(() => Array.from(document.querySelectorAll('.panel:first-child .panel-table tr')).slice(1)
        .every(row => row.querySelectorAll('td')[1].textContent !== '—'));
    const contained = await page.evaluate(() => {
        const u = chart(), rect = u.ctx.canvas.getBoundingClientRect();
        return rect.right <= document.querySelector('.panel').getBoundingClientRect().right;
    });
    assert.ok(contained, 'The entire canvas, including right tick labels, fits inside the panel');
    await (await page.$('.panel')).screenshot({path: path.join(path.dirname(html), 'bode.png')});
    const state = await page.evaluate(() => command('getState'));
    assert.deepEqual(state.plots['0'].panels[0].yAxes.map(a => [a.unit, a.signals]), [
        ['V', ['v(out) (mag)']], ['deg', ['v(out) (phase)']],
    ]);
    await page.waitForFunction(state => Object.keys(localStorage).some(key => {
        const value = JSON.parse(localStorage.getItem(key));
        return JSON.stringify(value.plots) === JSON.stringify(state.plots);
    }), {}, state);
    await page.reload();
    await page.waitForSelector('.u-over');
    assert.deepEqual(await page.evaluate(() => command('getState')), state);
    assert.deepEqual((await page.evaluate(() => snapshot())).map(a => [a.mode, a.range]), [
        ['dB', [0.999, 1.001]], ['linear', [-2, 0]],
    ]);
    await page.evaluate(() => {
        for (const key of Object.keys(localStorage)) {
            const value = JSON.parse(localStorage.getItem(key));
            if (!value.plots) continue;
            for (const plot of Object.values(value.plots)) for (const panel of plot.panels) delete panel.yAxes;
            localStorage.setItem(key, JSON.stringify(value));
        }
    });
    await page.reload();
    await page.waitForSelector('.u-over');
    assert.deepEqual((await page.evaluate(() => snapshot())).map(a => [a.unit, a.side]), [['V', 3], ['deg', 1]]);

    // Shift-drag zoom uses each scale's coordinates, then shift-wheel
    // narrows both ranges. The panel reset restores every axis together.
    await page.evaluate(async () => {
        await command('setYMode', {unit: 'V', mode: 'linear'});
        await command('autoscaleY');
    });
    const before_drag = await page.evaluate(() => snapshot().map(a => a.range));
    const over = await page.$('.panel:first-child .u-over');
    const bounds = await over.boundingBox();
    const x = bounds.x + bounds.width * 0.5;
    await page.mouse.move(x, bounds.y + bounds.height * 0.25);
    await page.keyboard.down('Shift');
    await page.mouse.down();
    await page.mouse.move(x, bounds.y + bounds.height * 0.75, {steps: 8});
    await page.mouse.up();
    await page.keyboard.up('Shift');
    const after_drag = await page.evaluate(() => snapshot().map(a => a.range));
    for (let i = 0; i < before_drag.length; i++) {
        const ratio = (after_drag[i][1] - after_drag[i][0]) / (before_drag[i][1] - before_drag[i][0]);
        assert.ok(ratio > 0.45 && ratio < 0.55, 'Y drag narrows each scale: ' + ratio);
    }
    await page.keyboard.down('Shift');
    await page.mouse.wheel({deltaY: -100});
    await page.keyboard.up('Shift');
    await page.waitForFunction(before => snapshot().every((axis, i) =>
        axis.range[1] - axis.range[0] < before[i][1] - before[i][0]), {}, after_drag);
    await page.click('.panel:first-child .panel-header .panel-yauto-btn');
    assert.deepEqual(await page.evaluate(() => snapshot().map(a => a.range)), before_drag);

    // Deterministic group order follows first appearance, including phase
    // first and V/A magnitude groups sharing a panel with phase.
    for (const [signals, expected] of [
        [['v(out) (phase)', 'v(out) (mag)'], [['deg', 3], ['V', 1]]],
        [['i(v1) (mag)', 'v(out) (phase)', 'v(out) (mag)', 'v(in) (mag)'], [['A', 3], ['deg', 3], ['V', 3]]],
    ]) {
        await page.evaluate(signals => plot_signals(signals), signals);
        axes = await page.evaluate(() => snapshot());
        assert.deepEqual(axes.map(a => [a.unit, a.side]), expected);
        assert.ok(axes.every(a => a.ticks.length > 1 && a.width > 0));
        assert.deepEqual(axes.map(a => a.modes.includes('dB')), expected.map(([unit]) => unit !== 'deg'));
    }
    await page.evaluate(() => command('removeSignal', {signal: 'v(out) (phase)'}));
    assert.deepEqual((await page.evaluate(() => snapshot())).map(a => [a.unit, a.side]), [['A', 3], ['V', 1]]);
    await page.evaluate(() => command('removeSignal', {signal: 'i(v1) (mag)'}));
    assert.equal((await page.evaluate(() => snapshot())).length, 1);
    assert.equal(await page.$eval('.panel:first-child .panel-header .panel-ymode', el => el.dataset.unit), 'V');

    await page.goto(pathToFileURL(unitless_html).href);
    await page.waitForSelector('.u-over');
    for (const [signals, expected] of [
        [['unknown', 'also_unknown'], [['', 3]]],
        [['unknown', 'v(out)'], [['', 3], ['V', 1]]],
        [['v(out)', 'unknown'], [['V', 3], ['', 1]]],
        [['unknown', 'i(v1)', 'v(out)', 'also_unknown'], [['', 3], ['A', 3], ['V', 3]]],
        [['unknown', 'p(out)', 'i(v1)', 'v(out)', 'also_unknown'], [['', 3], ['W', 3], ['A', 3], ['V', 3]]],
    ]) {
        await page.evaluate(signals => plot_signals(signals), signals);
        axes = await page.evaluate(() => snapshot());
        assert.deepEqual(axes.map(a => [a.unit, a.side]), expected);
        assert.ok(axes.find(a => a.unit === '').ticks.every(t => !/V|A|deg/.test(t)));
        assert.ok(axes.every(a => !a.modes.includes('dB')), 'Real signals never offer dB');
    }
    await page.evaluate(() => command('setYRange', {unit: '', min: 0, max: 9000}));
    assert.deepEqual((await page.evaluate(() => snapshot()))[0].range, [0, 9000]);
    assert.deepEqual(errors, [], 'No uncaught browser errors');
    console.log('PASS: independent Bode axes, unit grouping/order, controls, annotations, exports, and saved layouts');
} finally {
    await browser.close();
}
