import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import puppeteer from 'puppeteer-core';

const [html, raw, chrome] = process.argv.slice(2);
const bytes = fs.readFileSync(raw);
const file_name = path.basename(raw);
const run_id = createHash('sha256').update(bytes).update('\0[2,8000,8,1000]').digest('hex').slice(0, 16);
const state_key = createHash('sha256').update(file_name).digest('hex').slice(0, 8);
const browser = await puppeteer.launch({
    executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-gpu'],
});
try {
    const page = await browser.newPage();
    await page.setViewport({width: 1400, height: 900});
    const errors = [], network = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => network.push(request.url()));
    // Observe the real renderer's input without exposing viewer internals.
    await page.evaluateOnNewDocument(() => {
        let constructor;
        window.charts = [];
        Object.defineProperty(window, 'uPlot', {
            configurable: true,
            get: () => constructor,
            set: value => {
                constructor = new Proxy(value, {
                    construct(target, args) {
                        const chart = new target(...args);
                        window.charts.push(chart);
                        return chart;
                    },
                });
            },
        });
    });
    const url = pathToFileURL(path.resolve(html)).href;
    await page.goto(url, {waitUntil: 'load'});
    await page.waitForSelector('.u-over');
    await page.evaluate(() => {
        window.original_payload = window.SILK_VIEW_DATA;
        window.equal = function equal(a, b, at = 'data') {
            if (Object.is(a, b)) return;
            if (a && b && typeof a === 'object' && typeof b === 'object') {
                const ka = Object.keys(a), kb = Object.keys(b);
                if (ka.join('|') !== kb.join('|')) throw new Error(at + ': keys differ');
                for (const key of ka) equal(a[key], b[key], at + '.' + key);
            } else if (!Object.is(a, b)) throw new Error(at + ': ' + a + ' !== ' + b);
        };
        window.command = async (cmd, args = {}) => {
            const result = await window.silkView.run([{cmd, args}]);
            if (!result.ok) throw new Error(JSON.stringify(result));
            return result.results[0];
        };
        window.panel_data = () => window.charts.filter(u => u.root.isConnected)
            .map(u => u.data.map(column => Array.from(column)));
        window.normalize = payload => payload.plots.map(plot => ({
            ...plot,
            x: {...plot.x, levels: plot.x.levels.map(level => ({
                ...level, x: decode(level.x, 'f64'),
            }))},
            signals: plot.signals.map(signal => ({
                ...signal, levels: signal.levels.map(level => ({
                    ...level, ...(level.x === undefined ? {} : {x: decode(level.x, 'f64')}),
                    y: decode(level.y, level.dt),
                })),
            })),
        }));
        function decode(value, dt) {
            if (ArrayBuffer.isView(value)) return value;
            const buffer = Uint8Array.from(atob(value), c => c.charCodeAt(0)).buffer;
            return dt === 'f64' ? new Float64Array(buffer) : new Float32Array(buffer);
        }
        window.expected = window.normalize(window.SILK_VIEW_DATA);
        window.snapshots = [];
    });
    // Record full and zoomed real uPlot data for every plot, before JS opens it.
    const count = await page.evaluate(() => window.SILK_VIEW_DATA.plots.length);
    // TASK-238: the default layout opens on one empty panel now, so add
    // every signal explicitly to keep exercising real plotted data (raw
    // series values are unaffected by which panel/yMode holds them).
    for (let index = 0; index < count; index++) {
        await page.evaluate(async index => {
            await window.command('setPlot', {index});
            const names = window.expected[index].signals.map(s => s.name);
            if (names.length) await window.command('addSignal', {signals: names});
            const full = window.panel_data();
            const x = window.expected[index].x.levels[0].x;
            const range = {min: x[0], max: x[Math.min(x.length - 1, 200)]};
            if (range.max > range.min) await window.command('setXRange', range);
            window.snapshots.push({full, zoom: window.panel_data(), range});
        }, index);
    }
    if (file_name === 'pyramid.raw') {
        assert.deepEqual(await page.evaluate(() => window.expected[0].signals.map(s =>
            s.levels.map(l => l.stride))), [[1, 8, 64], [1, 8, 64], [1, 8, 64]]);
        assert.deepEqual(await page.evaluate(() => window.expected[0].signals.map(s => s.levels[0].dt)),
            ['f32', 'f64', 'f32']);
    }

    async function open_file() {
        const chooser_promise = page.waitForFileChooser();
        await page.click('#open-raw-btn');
        const chooser = await chooser_promise;
        await chooser.accept([raw]);
        await page.waitForFunction(() => !document.getElementById('open-raw-btn').disabled);
        assert.equal(await page.$eval('#fatal-banner', el => el.hidden), true);
    }
    async function drop_file(name, content) {
        await page.evaluate((name, content) => {
            const transfer = new DataTransfer();
            transfer.items.add(new File([Uint8Array.from(atob(content), c => c.charCodeAt(0))], name));
            document.querySelector('.u-over')?.dispatchEvent(new DragEvent('dragover', {
                bubbles: true, cancelable: true, dataTransfer: transfer,
            }));
            const target = document.querySelector('.u-over') || document.body;
            target.dispatchEvent(new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: transfer}));
        }, name, Buffer.from(content).toString('base64'));
        await page.waitForFunction(() => !document.getElementById('open-raw-btn').disabled);
    }
    await page.setOfflineMode(true);
    await open_file();
    assert.deepEqual(await page.evaluate(() => {
        const p = window.SILK_VIEW_DATA;
        window.equal(window.expected, window.normalize(p), 'plots and all decimation levels');
        for (const plot of p.plots) {
            if (!(plot.x.levels[0].x instanceof Float64Array)) throw new Error('X is not typed');
            for (const signal of plot.signals) for (const level of signal.levels) {
                if (!(level.y instanceof (level.dt === 'f64' ? Float64Array : Float32Array))) {
                    throw new Error('Y has wrong dtype');
                }
            }
        }
        return [p.run.id, p.run.state_key, p.run.source];
    }), [run_id, state_key, file_name]);
    for (let index = 0; index < count; index++) {
        await page.evaluate(async index => {
            await window.command('setPlot', {index});
            const names = window.expected[index].signals.map(s => s.name);
            if (names.length) await window.command('addSignal', {signals: names});
            window.equal(window.snapshots[index].full, window.panel_data(), 'full panel');
            const {range} = window.snapshots[index];
            if (range.max > range.min) await window.command('setXRange', range);
            window.equal(window.snapshots[index].zoom, window.panel_data(), 'zoomed panel');
        }, index);
    }
    assert.equal(await page.evaluate(() => window.command('getContext')), null);
    console.log('PASS: exact dual-producer plot levels and full/zoomed panel data, offline');

    // A same-file reload preserves chart objects and the entire saved view.
    await page.click('#layout-toggle');
    await page.select('#font-scale-select', 'large');
    const state = await page.evaluate(async () => {
        window.current_charts = window.charts.filter(u => u.root.isConnected);
        return window.command('getState');
    });
    await open_file();
    await page.evaluate(() => {
        const current = window.charts.filter(u => u.root.isConnected);
        if (current.length !== window.current_charts.length ||
            current.some((chart, i) => chart !== window.current_charts[i])) {
            throw new Error('same-file reopen rebuilt charts');
        }
    });
    assert.deepEqual(await page.evaluate(() => window.command('getState')), state);
    console.log('PASS: same file is a no-op reload');

    const requests_before = network.length;
    await page.evaluate(() => {
        window.__silkViewStamp?.('stale-sidecar');
        window.__silkViewData(window.original_payload);
    });
    assert.equal(await page.evaluate(() => window.SILK_VIEW_DATA.run.id), run_id);
    await new Promise(resolve => setTimeout(resolve, 500));
    assert.equal(network.length, requests_before, 'local mode must stop sidecar polling');

    // The filename is the state seed even when the bytes (hence run ID) match.
    await drop_file('different.raw', bytes);
    assert.equal(await page.evaluate(() => window.SILK_VIEW_DATA.run.id), run_id);
    assert.notEqual(await page.evaluate(() => window.SILK_VIEW_DATA.run.state_key), state_key);
    await drop_file(file_name, bytes);
    assert.deepEqual(await page.evaluate(() => window.command('getState')), state);
    assert.equal(await page.$eval('#fatal-banner', el => el.hidden), true);
    console.log('PASS: drag-drop, filename-specific layouts, stale sidecar ignored');

    // New bytes at the same filename are a fresh run with the same layout.
    const changed = Buffer.from(bytes);
    changed[changed.indexOf('Title: ') + 7] = 'Z'.charCodeAt(0);
    await drop_file(file_name, changed);
    assert.notEqual(await page.evaluate(() => window.SILK_VIEW_DATA.run.id), run_id);
    assert.equal(await page.evaluate(() => window.SILK_VIEW_DATA.run.state_key), state_key);
    assert.deepEqual(await page.evaluate(() => window.command('getState')), state);
    await drop_file(file_name, bytes);

    // Out-of-order reads must not replace the last file the user selected.
    await page.evaluate(() => {
        window.read_file = File.prototype.arrayBuffer;
        File.prototype.arrayBuffer = function () {
            if (this.name !== 'slow.raw') return window.read_file.call(this);
            return new Promise(resolve => { window.release_file = () => resolve(window.read_file.call(this)); });
        };
        const transfer = new DataTransfer();
        transfer.items.add(new File(['delayed malformed input'], 'slow.raw'));
        document.body.dispatchEvent(new DragEvent('drop', {bubbles: true, dataTransfer: transfer}));
    });
    await drop_file(file_name, bytes);
    await page.evaluate(async () => {
        window.release_file();
        File.prototype.arrayBuffer = window.read_file;
        await new Promise(resolve => setTimeout(resolve, 30));
    });
    assert.equal(await page.evaluate(() => window.SILK_VIEW_DATA.run.id), run_id);
    assert.equal(await page.$eval('#fatal-banner', el => el.hidden), true);

    // The existing view-state file drop continues through its original path.
    await drop_file('layout.silkview.json', Buffer.from(JSON.stringify({silkview: 1, view: state})));
    await page.waitForFunction(() => document.getElementById('toast-container').textContent
        .includes('view loaded from layout.silkview.json'));
    assert.deepEqual(await page.evaluate(() => window.command('getState')), state);
    console.log('PASS: changed bytes preserve layout, latest selection wins, JSON view drop preserved');

    // Reopening after a page refresh reads the filename's durable localStorage.
    await page.reload({waitUntil: 'load'});
    await page.waitForSelector('.u-over');
    await open_file();
    const restored = await page.evaluate(async () => {
        const result = await window.silkView.run([{cmd: 'getState'}]);
        return result.results[0];
    });
    assert.deepEqual(restored, state);
    console.log('PASS: saved layout survives page refresh and reopening');

    for (const content of ['not a rawfile <img src=x onerror=alert(1)>',
        'Title: aborted\nPlotname: Transient Analysis\nNo. Variables: 1\nNo. Points: 9\n']) {
        await drop_file('broken.raw', Buffer.from(content));
        assert.equal(await page.$eval('#fatal-banner', el => el.hidden), false);
        assert.match(await page.$eval('#fatal-banner', el => el.textContent), /^RawfileError:/);
        assert.equal(await page.$eval('#app', el => el.style.display), 'none');
        assert.equal(await page.$eval('#fatal-banner', el => el.querySelector('img')), null);
        await drop_file(file_name, bytes);
        assert.equal(await page.$eval('#fatal-banner', el => el.hidden), true);
    }
    assert.deepEqual(errors, []);
    assert(network.every(url => url.startsWith('file:') || url.startsWith('data:text/javascript;base64,')),
        'only local files and embedded modules are used; no server or network resource');
    console.log('PASS: malformed/aborted inputs show the module fatal error and valid drops recover');
} finally {
    await browser.close();
}
