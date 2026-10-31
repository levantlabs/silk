import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createInterface} from 'node:readline';
import puppeteer from 'puppeteer-core';

const [directory, chrome] = process.argv.slice(2);
const browser = await puppeteer.launch({executablePath: chrome, headless: true,
    args: ['--no-sandbox', '--disable-gpu']});
try {
    const page = await browser.newPage();
    await page.evaluateOnNewDocument(() => {
        window.charts = [];
        Object.defineProperty(window, 'uPlot', {configurable: true,
            set(target) {
                Object.defineProperty(window, 'uPlot', {configurable: true, writable: true,
                    value: new Proxy(target, {construct(target, args) {
                        const chart = new target(...args);
                        charts.push(chart);
                        return chart;
                    }})});
            }});
        window.chartData = () => charts.filter(chart => chart.root.isConnected)
            .map(chart => chart.data.map(values => Array.from(values)));
    });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const url = pathToFileURL(path.join(directory, 'main.html'));
    url.searchParams.set('commands', JSON.stringify([{cmd: 'setCols', args: {n: 2}}]));
    await page.goto(url.href);
    await page.waitForSelector('.u-over');
    await page.evaluate(() => {
        window.command = async (cmd, args = {}) => {
            const result = await silkView.run([{cmd, args}]);
            if (!result.ok) throw new Error(JSON.stringify(result));
            return result.results[0];
        };
    });
    assert.equal(await page.evaluate(() => silkView.version), '1.4');
    assert.equal(await page.evaluate(() => silkView.commands().some(command => command.name === 'getState')), true);
    assert.equal((await page.evaluate(() => command('getState'))).cols, 2,
        'Query commands run after embedded state and commands');
    const original = await page.evaluate(() => SILK_VIEW_DATA);
    assert.equal(await page.title(), 'Initial title');
    await page.evaluate(async () => {
        await command('addSignal', {signals: SILK_VIEW_DATA.plots[0].signals[0].name, panel: 0});
        await command('setXRange', {min: 0, max: 100e-9});
        await command('placeCursor', {which: 'A', x: 10e-9});
        await command('addMarker', {name: 'Preserved marker', x: 20e-9});
    });
    const state = await page.evaluate(() => command('getState'));
    await page.waitForFunction(() => {
        const saved = JSON.parse(localStorage.getItem('silk.view.' + SILK_VIEW_DATA.run.state_key));
        return saved?.plots?.['0']?.markers?.length > 0;
    });
    const saved = await page.evaluate(() => localStorage.getItem('silk.view.' + SILK_VIEW_DATA.run.state_key));
    const samples = await page.evaluate(() => chartData());
    assert.ok(samples.some(chart => chart.length > 1 && chart[1].length > 1));
    const annotation = await page.$eval('.pbadge', el => el.title);
    assert.equal(annotation, 'Preserved annotation');
    const staticPages = [];
    for (const filename of ['paused.html', 'inline.html']) {
        const staticPage = await browser.newPage();
        await staticPage.goto(pathToFileURL(path.join(directory, filename)).href);
        await staticPage.waitForSelector('.u-over');
        assert.equal(await staticPage.title(), 'Initial title');
        staticPages.push(staticPage);
    }
    await page.bringToFront();
    const input = createInterface({input: process.stdin});
    const reemitted = new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Python re-emission timed out')), 45000);
        input.once('line', line => {
            clearTimeout(timer);
            if (line === 'REEMITTED') resolve();
            else reject(new Error(`Unexpected handshake: ${line}`));
        });
        input.once('close', () => { clearTimeout(timer); reject(new Error('Python input closed')); });
    });
    console.log('READY');
    try { await reemitted; } finally { input.close(); }
    await page.waitForFunction(() => document.getElementById('run-title').textContent === 'Polled refresh');
    assert.equal(await page.title(), 'Polled refresh');
    assert.deepEqual(await page.evaluate(() => command('getState')), state,
        'Title polling preserves zoom, selected signals, cursors, markers and layout');
    assert.equal(await page.evaluate(() => localStorage.getItem('silk.view.' + SILK_VIEW_DATA.run.state_key)), saved);
    assert.deepEqual(await page.evaluate(() => chartData()), samples);
    assert.equal(await page.$eval('.pbadge', el => el.title), annotation);
    for (const staticPage of staticPages) {
        await staticPage.bringToFront();
        // Exceed both the explicit 50 ms poll and the default 2000 ms poll
        // with this control foregrounded, so throttling cannot hide a reload.
        await staticPage.evaluate(() => new Promise(resolve => setTimeout(resolve, 2500)));
        assert.equal(await staticPage.title(), 'Initial title', 'Inline and poll_ms=0 do not refresh automatically');
        assert.equal(await staticPage.$eval('#run-title', el => el.textContent), 'Initial title');
        await staticPage.close();
    }
    await page.bringToFront();
    await page.evaluate(() => command('setScope', {key: 'sibling'}));
    assert.equal((await page.evaluate(() => command('listScopes'))).current, 'sibling');
    assert.equal((await page.evaluate(() => command('getContext'))).title, 'Sibling context');
    assert.equal((await page.evaluate(() => command('getState'))).cols, 2, 'Scope switch retains the layout');
    // A loaded scope must poll its own stamp, not the original page's stamp.
    fs.writeFileSync(path.join(directory, 'sibling.stamp.js'), 'window.__silkViewStamp("sibling-refresh");');
    const sibling = {...original, run: {...original.run, id: 'sibling-refresh', title: 'Sibling refreshed'},
        context: {title: 'Sibling refreshed', subtitle: '', fields: []}};
    fs.writeFileSync(path.join(directory, 'sibling.data.js'), `window.__silkViewData(${JSON.stringify(sibling)});`);
    await page.waitForFunction(() => document.getElementById('run-title').textContent === 'Sibling refreshed');
    await page.evaluate(() => command('loadViewState', {state: {silkview: 1, view: {v: 1, cols: 1}}}));
    assert.equal((await page.evaluate(() => command('getState'))).cols, 1);
    assert.equal(await page.evaluate(() => typeof __silkViewData === 'function' && typeof __silkViewStamp === 'function'), true);
    assert.deepEqual(errors, []);
    console.log('PASS: standalone globals, command precedence, stamp/data polling, scopes and view-state loading');
} finally { await browser.close(); }
