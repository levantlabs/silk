// TASK-360: "Expand all" in the picker must open every kind section (Voltages /
// Currents) and every group inside each, plus the ungrouped rest; "Collapse
// all" closes them again. Runs against a standalone emitted page
// ("standalone") and a createExplorer mount ("embedded").
//
// Usage: node explorer_expand_all_e2e.mjs standalone <view.html> <chrome>
//        node explorer_expand_all_e2e.mjs embedded <consumer-dir> <chrome>
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
        const html = '<!doctype html><style>body{margin:0}</style>'
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
        });
    }
    await page.waitForSelector('.picker-row');
    const settle = () => new Promise(resolve => setTimeout(resolve, 200));
    const click = async name => {
        await page.evaluate(n => document.querySelector(`[data-silk="${n}"]`).click(), name);
        await settle();
    };
    const rows = () => page.evaluate(() => ({
        leaves: document.querySelectorAll('.picker-row:not(.group-row):not(.flat-header)').length,
        groups: document.querySelectorAll('.picker-row.group-row').length,
    }));

    // Fixture: 3 voltage + 3 current signals, dotted so each kind has groups.
    await click('tree-collapse-all');
    const closed = await rows();
    assert.equal(closed.leaves, 0, 'Collapse all leaves no signal rows');
    assert.equal(closed.groups, 2, 'only the two kind section headers remain');

    await click('tree-expand-all');
    const open = await rows();
    assert.equal(open.leaves, 6, `Expand all shows every signal, saw ${open.leaves}`);
    assert.ok(open.groups >= 4, `Expand all opens sections and inner groups, saw ${open.groups} group rows`);

    await click('tree-collapse-all');
    assert.deepEqual(await rows(), closed, 'Collapse all still closes everything');

    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
    if (server) server.close();
}
