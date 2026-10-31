import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import puppeteer from 'puppeteer-core';

const [directory, chrome] = process.argv.slice(2);
const html = `<!doctype html><title>Foreign application</title>
<style>body{margin:17px;background:#e6dccb;color:#281c13;font:17px Georgia}
button,input{background:#ffeebb;color:#521b0f;border:3px solid #804000;padding:11px}
*{box-sizing:content-box}main{margin:0} .mount{width:1400px;height:490px;margin:12px 0}</style>
<main id="foreign"><button id="host-button">Host action</button><input id="host-input" value="Host input"></main>
<div id="one" class="mount"></div><div id="two" class="mount"></div>`;
const server = http.createServer((req, res) => {
    const name = decodeURIComponent(req.url.split('?')[0]);
    if (name === '/') { res.setHeader('Content-Type', 'text/html'); res.end(html); return; }
    const file = path.join(directory, path.basename(name));
    if (!fs.existsSync(file)) { res.writeHead(404); res.end('missing asset'); return; }
    res.setHeader('Content-Type', name.endsWith('.mjs') || name.endsWith('.js') ? 'text/javascript'
        : name.endsWith('.css') ? 'text/css' : 'application/octet-stream');
    res.end(fs.readFileSync(file));
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await puppeteer.launch({executablePath: chrome, headless: true,
    args: ['--no-sandbox', '--disable-gpu']});
try {
    const page = await browser.newPage();
    await page.setViewport({width: 1520, height: 1160});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(() => {
        window.hostStyles = () => ['body', '#foreign', '#host-button', '#host-input'].map(selector => {
            const styles = getComputedStyle(document.querySelector(selector));
            return Object.fromEntries(Array.from(styles, key => [key, styles.getPropertyValue(key)]));
        });
        window.before = hostStyles();
    });
    await page.addStyleTag({url: '/view.css'});
    assert.deepEqual(await page.evaluate(() => hostStyles()), await page.evaluate(() => before),
        'Loading the explorer stylesheet must not restyle the host');
    await page.evaluate(() => {
        window.globalListeners = [];
        const add = EventTarget.prototype.addEventListener, remove = EventTarget.prototype.removeEventListener;
        EventTarget.prototype.addEventListener = function (type, handler, options) {
            const capture = typeof options === 'boolean' ? options : !!options?.capture;
            if ((this === window || this === document) && !globalListeners.some(item =>
                item.target === this && item.type === type && item.handler === handler && item.capture === capture)) {
                globalListeners.push({target: this, type, handler, capture});
            }
            return add.call(this, type, handler, options);
        };
        EventTarget.prototype.removeEventListener = function (type, handler, options) {
            const capture = typeof options === 'boolean' ? options : !!options?.capture;
            window.globalListeners = globalListeners.filter(item => !(item.target === this && item.type === type && item.handler === handler && item.capture === capture));
            return remove.call(this, type, handler, options);
        };
    });
    await page.addScriptTag({url: '/uplot.iife.min.js'});
    const baselineListeners = await page.evaluate(() => globalListeners.map(item => item.type).sort());
    const available = await page.evaluate(async () => {
        try { window.factory = (await import('/explorer.mjs')).createExplorer; return typeof factory === 'function'; }
        catch { return false; }
    });
    assert.equal(available, true, 'The vendored module must expose createExplorer');
    await page.evaluate(async () => {
        window.payload = await (await fetch('/payload.json')).json();
        window.core = await import('/waveform-core.mjs');
        const raw = await (await fetch('/fixture.raw')).arrayBuffer();
        window.typed = await core.decimate(core.plan(core.read(raw, {file_name: 'fixture.raw'})));
        window.charts = [];
        window.chartPeer = new Proxy(uPlot, {construct(target, args) {
            const chart = new target(...args); charts.push(chart); return chart;
        }});
        const theme = {'--bg-0':'#171923','--bg-1':'#202431','--bg-2':'#292e3e','--bg-3':'#353c51',
            '--text-0':'#f2f4fa','--text-1':'#c0c7dd','--text-2':'#959db5','--border':'#454d65',
            '--border-soft':'#353c51','--accent':'#a0aaff','--accent-dim':'#6975cd','--trace-1':'#b7a1ff',
            '--trace-2':'#6ae6c7','--cursor-a':'#ff79bc','--cursor-b':'#68def5','--marker-1':'#ffd479',
            '--selection-row':'rgba(160,170,255,0.18)','--selection-row-hover':'rgba(160,170,255,0.26)'};
        for (const [key, value] of Object.entries(theme)) document.getElementById('one').style.setProperty(key, value);
        window.a = factory(document.getElementById('one'), {uPlot: chartPeer, waveformCore: core, stateKey: 'project/module/a'});
        window.b = factory(document.getElementById('two'), {uPlot: chartPeer, waveformCore: core, stateKey: 'project/module/b'});
        await a.load(payload);
        await b.load(typed);
        window.command = async (instance, cmd, args = {}) => {
            const result = await instance.run([{cmd, args}]);
            if (!result.ok) throw new Error(JSON.stringify(result));
            return result.results[0];
        };
        window.liveCharts = id => charts.filter(chart => chart.root.isConnected && document.getElementById(id).contains(chart.root));
        window.data = id => liveCharts(id).map(chart => chart.data.map(column => Array.from(column)));
        // TASK-238: the default layout opens on an empty panel now, so plot
        // a signal explicitly in both mounts to keep exercising real
        // dual-producer data and the themed trace palette below.
        const plotted = payload.plots[payload.default_plot].signals[0].name;
        await command(a, 'addSignal', {signals: [plotted]});
        await command(b, 'addSignal', {signals: [plotted]});
    });
    assert.equal(await page.evaluate(() => document.querySelectorAll('#one .u-over').length > 0 &&
        document.querySelectorAll('#two .u-over').length > 0), true);
    assert.deepEqual(await page.evaluate(() => data('one')), await page.evaluate(() => data('two')),
        'Both payload producers must render the same numerical samples');
    assert.equal(await page.evaluate(() => {
        const ids = Array.from(document.querySelectorAll('[id]'), el => el.id);
        return ids.length === new Set(ids).size;
    }), true, 'Mounts must not introduce duplicate IDs');
    assert.deepEqual(await page.evaluate(() => ['silkView','SILK_VIEW_DATA','__silkViewData','__silkViewStamp',
        'SILK_VIEW_CONFIG','SILK_VIEW_STATE','SILK_VIEW_COMMANDS','SILK_VIEW_SCOPES','SILK_WAVEFORM_CORE']
        .filter(name => name in window)), [], 'Host mounts must not install legacy globals');
    assert.equal(await page.evaluate(() => liveCharts('one')[0].series[1].stroke(liveCharts('one')[0], 1)), '#b7a1ff', 'Trace palette comes from the root');
    assert.equal(await page.$eval('#one [data-silk="toolbar"]', el => getComputedStyle(el).backgroundColor), 'rgb(32, 36, 49)');
    assert.equal(await page.$eval('#one button', el => getComputedStyle(el).borderTopWidth), '1px', 'Foreign button styles do not override the scoped controls');
    const second = await page.evaluate(() => command(b, 'getState'));
    await page.evaluate(async () => {
        const x = typed.plots[typed.default_plot].x.levels[0].x;
        await command(a, 'setXRange', {min: x[0], max: x[Math.floor(x.length / 2)]});
        await command(a, 'placeCursor', {which: 'A', x: x[1]});
        await command(a, 'addMarker', {name: 'Themed marker', x: x[2]});
    });
    assert.deepEqual(await page.evaluate(() => command(b, 'getState')), second, 'uPlot zoom and cursor sync must not cross instances');
    await page.waitForSelector('#one .marker-badge');
    assert.equal(await page.$eval('#one .marker-badge', el => getComputedStyle(el).backgroundColor), 'rgb(255, 212, 121)');
    await page.evaluate(async () => { await command(a, 'setCols', {n: 2}); await command(a, 'setFilter', {text: 'absent'}); });
    assert.deepEqual(await page.evaluate(() => command(b, 'getState')), second, 'Commands must stay in their instance');
    await page.focus('#host-input');
    await page.keyboard.press('/');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'host-input', 'Foreign input keeps shortcut keys');
    await page.focus('#two');
    await page.keyboard.press('/');
    assert.equal(await page.evaluate(() => document.activeElement === document.querySelector('#two [data-silk="picker-filter"]')), true);
    assert.equal(await page.$eval('#one [data-silk="picker-filter"]', el => el.value), 'absent');
    const outside = await page.evaluate(() => {
        const transfer = new DataTransfer();
        transfer.items.add(new File(['bad'], 'outside.raw'));
        const event = new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: transfer});
        document.getElementById('foreign').dispatchEvent(event);
        return event.defaultPrevented;
    });
    assert.equal(outside, false, 'Foreign drops are not intercepted');

    await page.evaluate(() => {
        const names = Array.from(document.querySelectorAll('#two .picker-row[data-name]'), row => row.dataset.name).slice(0, 2);
        for (const name of names) document.querySelector('#two .picker-row[data-name=' + JSON.stringify(name) + ']')
            .dispatchEvent(new MouseEvent('click', {bubbles: true, ctrlKey: true}));
    });
    await page.click('#two .picker-row.selected', {button: 'right'});
    await page.waitForSelector('#two [data-silk="derive-menu"]:not([hidden])', {timeout: 2000});
    const context = await page.evaluate(() => {
        const menu = document.querySelector('#two [data-silk="derive-menu"]');
        const bounds = menu.getBoundingClientRect(), root = document.getElementById('two').getBoundingClientRect();
        const event = new MouseEvent('contextmenu', {bubbles: true, cancelable: true});
        let reachedHost = false;
        const handler = () => { reachedHost = true; };
        document.addEventListener('contextmenu', handler);
        document.getElementById('foreign').dispatchEvent(event);
        document.removeEventListener('contextmenu', handler);
        return {prevented: event.defaultPrevented, reachedHost, open: !menu.hidden,
            global: globalListeners.filter(item => item.type === 'contextmenu').length,
            contained: bounds.left >= root.left && bounds.right <= root.right && bounds.top >= root.top && bounds.bottom <= root.bottom,
            otherOpen: !!document.querySelector('#one [data-silk="derive-menu"]:not([hidden])')};
    });
    assert.deepEqual(context, {prevented: false, reachedHost: true, open: true, global: 0, contained: true, otherOpen: false});
    await page.keyboard.press('Escape');
    assert.deepEqual(await page.evaluate(() => command(b, 'getState')), second);
    const width = await page.evaluate(() => liveCharts('two')[0].width);
    await page.$eval('#two', el => { el.style.width = '1000px'; el.style.height = '430px'; });
    await page.waitForFunction(previous => liveCharts('two')[0].width < previous, {}, width);
    assert.equal(await page.$eval('#two [data-silk="app"]', el => Math.round(el.getBoundingClientRect().height)), 430);
    await page.evaluate(async () => { await command(a, 'setFilter', {text: ''}); await a.reload({...payload, run: {...payload.run, id: 'refresh'}}); });
    assert.equal((await page.evaluate(() => command(a, 'getState'))).cols, 2, 'Reload preserves current layout');
    await page.evaluate(() => { document.getElementById('one').classList.add('host-updated'); a.destroy(); a.destroy(); });
    assert.equal(await page.$eval('#one', el => el.classList.contains('host-updated')), true, 'Destroy preserves host class updates');
    assert.equal(await page.$eval('#one', el => el.children.length), 0);
    assert.deepEqual(await page.evaluate(() => command(b, 'getState')), second);
    await page.evaluate(async () => {
        window.a = factory(document.getElementById('one'), {uPlot: chartPeer, waveformCore: core, stateKey: 'project/module/a'});
        await a.load(payload);
    });
    assert.equal((await page.evaluate(() => command(a, 'getState'))).cols, 2, 'Remount restores its own host state key');
    assert.deepEqual(await page.evaluate(() => hostStyles().slice(1)), await page.evaluate(() => before.slice(1)));
    await page.screenshot({path: path.join(directory, 'explorer-host.png')});
    const firstState = await page.evaluate(() => command(a, 'getState'));
    const secondState = await page.evaluate(() => command(b, 'getState'));
    // A drag from one picker must not act as a foreign device/name drop in another mount.
    await page.evaluate(async () => {
        await command(b, 'clearPanel');
        const transfer = new DataTransfer();
        const row = document.querySelector('#one .picker-row[draggable="true"]');
        row.dispatchEvent(new DragEvent('dragstart', {bubbles: true, dataTransfer: transfer}));
        document.querySelector('#two .panel-chart').dispatchEvent(new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: transfer}));
        row.dispatchEvent(new DragEvent('dragend', {bubbles: true, dataTransfer: transfer}));
    });
    assert.equal(await page.evaluate(() => (command(b, 'getState')).then(state => state.plots['0'].panels[0].signals.length)), 0,
        'Picker drags are private to the source mount');
    await page.evaluate(async state => { await command(b, 'loadViewState', {state}); }, secondState);
    await page.evaluate(async () => {
        a.destroy();
        window.a = factory(document.getElementById('one'), {uPlot: chartPeer, stateKey: 'project/module/a',
            scopes: {current: 'current', scopes: [{key:'current',label:'Current'}, {key:'later',label:'Later'}]},
            loadScope: (scope, {signal}) => new Promise(resolve => { window.finishScope = resolve; window.scopeSignal = signal; })});
        await a.load(payload);
        window.pendingScope = a.run([{cmd:'setScope',args:{key:'later'}}]);
    });
    await page.waitForFunction(() => !!window.finishScope);
    await page.evaluate(() => command(a, 'setScope', {key: 'current'}));
    assert.equal(await page.evaluate(() => scopeSignal.aborted), true, 'Selecting the current scope cancels an in-flight different scope');
    assert.equal(await page.evaluate(async () => (await pendingScope).ok), false);
    await page.evaluate(() => {
        window.finishScope = null;
        window.pendingScope = a.run([{cmd:'setScope',args:{key:'later'}}]);
    });
    await page.waitForFunction(() => !!window.finishScope);
    await page.evaluate(async () => {
        a.destroy();
        window.a = factory(document.getElementById('one'), {uPlot: chartPeer, stateKey: 'project/module/a'});
        await a.load(payload);
        finishScope({...payload, run: {...payload.run, title:'STALE TITLE'}});
    });
    assert.equal(await page.evaluate(() => scopeSignal.aborted), true, 'Destroy aborts host scope work');
    assert.equal(await page.evaluate(async () => (await pendingScope).ok), false, 'Pending commands settle on destroy');
    assert.notEqual(await page.$eval('#one [data-silk="run-title"]', el => el.textContent), 'STALE TITLE');
    assert.deepEqual(await page.evaluate(() => command(b, 'getState')), secondState);
    assert.deepEqual(await page.evaluate(() => command(a, 'getState')), firstState);
    const over = await page.$eval('#one .u-over', el => { const r = el.getBoundingClientRect(); return {x:r.x + 20,y:r.y + 20}; });
    await page.mouse.move(over.x, over.y);
    await page.mouse.down();
    assert.equal(await page.evaluate(() => globalListeners.some(item => item.type === 'mousemove')), true, 'A drag owns temporary move handlers');
    await page.evaluate(() => { a.destroy(); b.destroy(); });
    await page.mouse.up();
    assert.deepEqual(await page.evaluate(() => hostStyles().slice(1)), await page.evaluate(() => before.slice(1)));
    assert.deepEqual(await page.evaluate(() => globalListeners.map(item => item.type).sort()), baselineListeners,
        'All per-mount document/window handlers, including the peer chart handlers, are removed on destroy');
    // Final-review regressions run independently so each defect is visible on the red run.
    const regressions = [];
    async function regression(name, check) {
        try { await check(); }
        catch (error) { regressions.push(name + ': ' + error.message); }
        finally { await page.evaluate(() => a.destroy()); }
    }
    for (const replacement of ['load', 'reload']) {
        await regression('local open superseded by ' + replacement, async () => {
            const result = await page.evaluate(async replacement => {
                window.a = factory(document.getElementById('one'), {uPlot: chartPeer, waveformCore: core});
                await a.load(payload);
                const input = document.querySelector('#one [data-silk="raw-file-input"]');
                const button = document.querySelector('#one [data-silk="open-raw-btn"]');
                const state = () => ({disabled: button.disabled, text: button.textContent});
                function pendingFile(name) {
                    let resolve;
                    const bytes = new Promise(done => { resolve = done; });
                    Object.defineProperty(input, 'files', {configurable: true,
                        value: [{name, arrayBuffer: () => bytes}]});
                    input.dispatchEvent(new Event('change', {bubbles: true}));
                    return resolve;
                }
                const old = pendingFile('obsolete.raw');
                const during = state();
                await a[replacement]({...payload, run: {...payload.run, title: 'HOST REPLACEMENT'}});
                const replaced = state();
                const newer = pendingFile('newer.raw');
                // A malformed obsolete read must neither show a fatal banner nor reset the newer button.
                old(new ArrayBuffer(0));
                await new Promise(resolve => setTimeout(resolve, 30));
                const whileNewer = state();
                const bytes = await (await fetch('/fixture.raw')).arrayBuffer();
                newer(bytes);
                for (let i = 0; i < 100 && button.disabled; i++) await new Promise(resolve => setTimeout(resolve, 10));
                const recovered = state();
                const fatal = !document.querySelector('#one [data-silk="fatal-banner"]').hidden;
                const pendingDestroy = pendingFile('destroyed.raw');
                a.destroy();
                pendingDestroy(new ArrayBuffer(0));
                await new Promise(resolve => setTimeout(resolve, 20));
                return {during, replaced, whileNewer, recovered, fatal,
                    remainingChildren: document.getElementById('one').children.length};
            }, replacement);
            assert.deepEqual(result, {
                during: {disabled: true, text: 'Opening…'},
                replaced: {disabled: false, text: 'Open .raw…'},
                whileNewer: {disabled: true, text: 'Opening…'},
                recovered: {disabled: false, text: 'Open .raw…'},
                fatal: false, remainingChildren: 0,
            });
        });
    }
    await regression('failed scope commands settle and preserve prior scope', async () => {
        const result = await page.evaluate(async () => {
            let signal;
            window.a = factory(document.getElementById('one'), {uPlot: chartPeer,
                scopes: {current: 'current', scopes: [{key: 'current', label: 'Current'}, {key: 'bad', label: 'Bad'}]},
                loadScope: (scope, request) => {
                    signal = request.signal;
                    return {...payload, plots: payload.plots.map(plot => ({...plot, signals: null}))};
                }});
            await a.load(payload);
            const before = data('one');
            const pending = a.run([{cmd: 'setScope', args: {key: 'bad'}}]);
            const bounded = promise => Promise.race([promise,
                new Promise(resolve => setTimeout(() => resolve('UNSETTLED'), 100))]);
            const commandResult = await bounded(pending);
            const scopeLabel = document.querySelector('#one [data-silk="scope-btn"]').textContent;
            const after = data('one');
            const current = await command(a, 'setScope', {key: 'current'});
            await a.load({...payload, run: {...payload.run, title: 'RECOVERED'}});
            const recoveredTitle = document.querySelector('#one [data-silk="run-title"]').textContent;
            a.destroy();
            return {commandResult, afterDestroy: await bounded(pending), aborted: signal.aborted,
                scopeLabel, before, after, current, recoveredTitle};
        });
        assert.notEqual(result.commandResult, 'UNSETTLED', 'Failed setScope must finish before teardown');
        assert.equal(result.commandResult.ok, false);
        assert.match(JSON.stringify(result.commandResult), /plot.*signals.*array/i, 'Malformed input identifies the invalid plot field');
        assert.deepEqual(result.afterDestroy, result.commandResult);
        assert.equal(result.aborted, true);
        assert.match(result.scopeLabel, /Current/);
        assert.deepEqual(result.current, {key: 'current', changed: false});
        assert.deepEqual(result.after, result.before, 'Malformed payload must leave prior chart data usable');
        assert.equal(result.recoveredTitle, 'RECOVERED');
    });
    await regression('scope application exceptions settle', async () => {
        const result = await page.evaluate(async () => {
            let signal;
            window.a = factory(document.getElementById('one'), {uPlot: chartPeer,
                onTitle: title => { if (title === 'THROW') throw new Error('Host title callback failed'); },
                scopes: {current: 'current', scopes: [{key:'current',label:'Current'}, {key:'bad',label:'Bad'}]},
                loadScope: (scope, request) => { signal = request.signal;
                    return {...payload, run: {...payload.run, title: 'THROW'}}; }});
            await a.load(payload);
            const pending = a.run([{cmd:'setScope',args:{key:'bad'}}]);
            const bounded = () => Promise.race([pending, new Promise(resolve => setTimeout(() => resolve('UNSETTLED'), 100))]);
            const settled = await bounded();
            const scopeLabel = document.querySelector('#one [data-silk="scope-btn"]').textContent;
            const current = await command(a, 'setScope', {key: 'current'});
            const restoredTitle = document.querySelector('#one [data-silk="run-title"]').textContent;
            await a.load(payload);
            a.destroy();
            return {settled, afterDestroy: await bounded(), aborted: signal.aborted, scopeLabel, current,
                restoredTitle, originalTitle: payload.run.title || ''};
        });
        assert.notEqual(result.settled, 'UNSETTLED');
        assert.equal(result.settled.ok, false);
        assert.match(JSON.stringify(result.settled), /Host title callback failed/);
        assert.deepEqual(result.afterDestroy, result.settled);
        assert.equal(result.aborted, true);
        assert.match(result.scopeLabel, /Current/);
        assert.deepEqual(result.current, {key: 'current', changed: false});
        assert.equal(result.restoredTitle, result.originalTitle);
    });
    await regression('host form buttons never submit', async () => {
        await page.evaluate(async () => {
            window.formSubmits = 0;
            const form = document.createElement('form');
            form.id = 'host-form';
            form.addEventListener('submit', event => { event.preventDefault(); window.formSubmits++; });
            const root = document.getElementById('one');
            root.before(form); form.append(root);
            window.a = factory(root, {uPlot: chartPeer});
            await a.load(payload);
        });
        await page.click('#one [data-silk="add-panel-btn"]');
        await page.click('#one [data-silk="view-btn"]');
        await page.waitForSelector('#one [data-silk="view-menu"] button', {visible: true});
        await page.evaluate(() => {
            window.resetButton = [...document.querySelectorAll('#one [data-silk="view-menu"] button')]
                .find(button => button.textContent === 'Reset view to defaults');
            resetButton.setAttribute('data-regression-reset', '');
        });
        await page.click('#one [data-regression-reset]');
        const submits = await page.evaluate(() => formSubmits);
        const wrongTypes = await page.$$eval('#one button', buttons => buttons.filter(button => button.type !== 'button').map(button => button.textContent));
        await page.evaluate(() => { const form = document.getElementById('host-form'); form.before(document.getElementById('one')); form.remove(); });
        assert.equal(submits, 0, 'Actual static and dynamic control clicks must not submit an ancestor form');
        assert.deepEqual(wrongTypes, [], 'Every rendered explorer button explicitly has type=button');
    });
    assert.deepEqual(regressions, [], 'Final-review host regressions');
    assert.deepEqual(errors, [], 'No browser errors');
    console.log('PASS: vendored module, both producers, two instances, focus/drop isolation, container resize, persistence and destroy');
} finally {
    await browser.close();
    await new Promise(resolve => server.close(resolve));
}
