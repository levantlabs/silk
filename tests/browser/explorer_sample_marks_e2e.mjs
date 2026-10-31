// TASK-293: the "Samples" toggle draws a dot at every simulated sample of each
// plotted trace. Off by default; the `s` key and the button both toggle it;
// the state survives zoom and pan; while the samples would be denser than one
// per 4 px the dots are hidden and the plot says so; toggling never moves any
// layout box. Runs against a standalone emitted page ("standalone") and a
// createExplorer mount ("embedded").
//
// Usage: node explorer_sample_marks_e2e.mjs standalone <view.html> <chrome>
//        node explorer_sample_marks_e2e.mjs embedded <consumer-dir> <chrome>
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {pathToFileURL} from 'node:url';
import puppeteer from 'puppeteer-core';

const [mode, target, chrome] = process.argv.slice(2);
let server = null;
const browser = await puppeteer.launch({executablePath: chrome, headless: true,
    args: ['--no-sandbox', '--disable-gpu']});
try {
    const page = await browser.newPage();
    await page.setViewport({width: 1400, height: 1000});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));

    if (mode === 'standalone') {
        await page.goto(pathToFileURL(target).href);
    } else {
        const html = '<!doctype html><style>*{box-sizing:content-box}body{margin:0}</style>'
            + '<div id="mount" style="width:1300px;height:800px"></div>';
        server = http.createServer((req, res) => {
            const name = decodeURIComponent(req.url.split('?')[0]);
            if (name === '/') { res.setHeader('Content-Type', 'text/html'); res.end(html); return; }
            const file = path.join(target, path.basename(name));
            if (!fs.existsSync(file)) { res.writeHead(404); res.end('missing'); return; }
            res.setHeader('Content-Type', name.endsWith('.mjs') || name.endsWith('.js') ? 'text/javascript'
                : name.endsWith('.css') ? 'text/css' : name.endsWith('.json') ? 'application/json'
                : 'application/octet-stream');
            res.end(fs.readFileSync(file));
        });
        await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
        await page.goto(`http://127.0.0.1:${server.address().port}/`);
        await page.addStyleTag({url: '/view.css'});
        await page.addScriptTag({url: '/uplot.iife.min.js'});
        await page.evaluate(async () => {
            const {createExplorer} = await import('/explorer.mjs');
            const core = await import('/waveform-core.mjs');
            const payload = await (await fetch('/payload.json')).json();
            window.explorer = createExplorer(document.getElementById('mount'), {uPlot, waveformCore: core});
            await window.explorer.load(payload);
            const name = payload.plots[payload.default_plot].signals[0].name;
            const result = await window.explorer.run([{cmd: 'addSignal', args: {signals: [name]}}]);
            if (!result.ok) throw new Error(JSON.stringify(result));
        });
    }
    await page.waitForSelector('.u-over');
    if (mode === 'standalone' && !(await page.evaluate(() => document.querySelector('[data-silk="panels"]').textContent.includes('v(out)')))) {
        // Nothing plotted yet on a fresh page: plot the one signal.
        await page.click('.picker-row:not(.group-row):not(.flat-header)', {clickCount: 2});
    }
    await page.waitForFunction(() => document.querySelectorAll('[data-silk="panels"] canvas').length > 0);

    const button = '[data-silk="samples-toggle"]';
    const marks = () => page.evaluate(() => {
        const host = document.querySelector('[data-silk="panels"] [data-sample-marks]');
        return host ? host.getAttribute('data-sample-marks') : 'off';
    });
    const canvasHash = () => page.evaluate(() => {
        const c = document.querySelector('[data-silk="panels"] canvas');
        const url = c.toDataURL();
        let h = 0;
        for (let i = 0; i < url.length; i++) h = (Math.imul(h, 31) + url.charCodeAt(i)) | 0;
        return url.length + ':' + h;
    });
    const layout = () => page.evaluate(() => {
        const q = s => Array.from(document.querySelectorAll(s)).map(e => {
            const r = e.getBoundingClientRect();
            return [r.x, r.y, r.width, r.height].map(v => Math.round(v * 100) / 100);
        });
        return JSON.stringify([q('[data-silk="toolbar"]'), q('[data-silk="sidebar"]'),
            q('[data-silk="panels"]'), q('[data-silk="panels"] .u-wrap')]);
    });
    const settle = () => new Promise(resolve => setTimeout(resolve, 250));
    // X zoom through the real keyboard: '+' zooms about the cursor, '0' resets.
    const overBox = async () => (await page.$('.u-over')).boundingBox();
    const zoomIn = async times => {
        const box = await overBox();
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
        for (let i = 0; i < times; i++) { await page.keyboard.press('+'); await settle(); }
    };
    const pan = async () => {
        const box = await overBox();
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
        await page.mouse.wheel({deltaX: 300}); await settle();
    };

    // --- off by default, full range is dense
    assert.equal(await page.$eval(button, e => e.getAttribute('aria-pressed')), 'false');
    assert.equal(await marks(), 'off');
    const hashOff = await canvasHash();
    const layoutOff = await layout();

    // --- ON via the button, dense at full range: hidden with an indication
    await page.click(button); await settle();
    assert.equal(await page.$eval(button, e => e.getAttribute('aria-pressed')), 'true');
    assert.equal(await marks(), 'hidden', 'full range is denser than 1 sample / 4 px');
    assert.notEqual(await canvasHash(), hashOff, 'the dense note is drawn on the plot');
    assert.equal(await layout(), layoutOff, 'toggling on moves no layout box');

    // --- zoom in until sparse: dots are drawn
    await zoomIn(9);
    assert.equal(await marks(), 'dots', 'zoomed in far enough that dots are drawn');
    assert.equal(await page.$eval(button, e => e.getAttribute('aria-pressed')), 'true');
    const hashDots = await canvasHash();
    await pan();
    assert.equal(await marks(), 'dots', 'state survives a pan');
    assert.equal(await page.$eval(button, e => e.getAttribute('aria-pressed')), 'true');

    // --- OFF via the key (focus a non-input first)
    await page.focus('[data-silk="picker-list"]');
    await page.keyboard.press('s'); await settle();
    assert.equal(await page.$eval(button, e => e.getAttribute('aria-pressed')), 'false');
    assert.equal(await marks(), 'off');
    const hashOffZoomed = await canvasHash();
    assert.notEqual(hashOffZoomed, hashDots, 'dots leave the canvas when off');

    // --- ON via the key again, then reset zoom: back to dense/hidden
    await page.keyboard.press('s'); await settle();
    assert.equal(await marks(), 'dots');
    await page.keyboard.press('0'); await settle();
    assert.equal(await marks(), 'hidden', 'state survives zoom reset');
    assert.equal(await layout(), layoutOff, 'layout unchanged with markers on at full range');

    // --- OFF via the button: canvas returns to the untouched full-range render
    await page.click(button); await settle();
    assert.equal(await marks(), 'off');
    assert.equal(await canvasHash(), hashOff, 'off restores the original render exactly');
    assert.equal(await layout(), layoutOff, 'toggling off moves no layout box');

    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
    if (server) server.close();
}
