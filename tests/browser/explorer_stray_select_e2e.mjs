// TASK-347: no stray pixel at the plot's top-left corner. uPlot leaves an
// empty selection (.u-select) as a 0x0 box at the plot origin; a CSS border
// on it painted a 2x2 accent square there. Runs against a standalone emitted
// page ("standalone") and a createExplorer mount ("embedded"), with the
// pointer parked off the plot, before and after a zoom.
//
// Usage: node explorer_stray_select_e2e.mjs standalone <view.html> <chrome>
//        node explorer_stray_select_e2e.mjs embedded <consumer-dir> <chrome>
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
        await page.click('.picker-row:not(.group-row):not(.flat-header)', {clickCount: 2});
    }
    await page.waitForFunction(() => document.querySelectorAll('[data-silk="panels"] canvas').length > 0);
    const settle = () => new Promise(resolve => setTimeout(resolve, 250));
    await settle();

    // The empty selection must not occupy any pixels, and nothing painted in
    // the accent colour may sit in the plot's top-left 12x12 px corner.
    const strayInfo = () => page.evaluate(() => {
        const over = document.querySelector('.u-over').getBoundingClientRect();
        const sel = document.querySelector('.u-select');
        const box = sel.getBoundingClientRect();
        const cs = getComputedStyle(sel);
        return {
            corner: [over.x, over.y],
            selPaints: box.width > 0 && box.height > 0,
            selBox: [box.width, box.height],
            border: cs.borderTopWidth + ' ' + cs.borderLeftWidth,
        };
    });
    const cornerAccentPixels = async () => {
        const {corner} = await strayInfo();
        const png = await page.screenshot({clip: {x: corner[0] - 2, y: corner[1] - 2, width: 14, height: 14}});
        const bitmap = await page.evaluate(async b64 => {
            const img = new Image(); img.src = 'data:image/png;base64,' + b64; await img.decode();
            const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
            const x = c.getContext('2d'); x.drawImage(img, 0, 0);
            return Array.from(x.getImageData(0, 0, c.width, c.height).data);
        }, Buffer.from(png).toString('base64'));
        let n = 0;
        for (let i = 0; i < bitmap.length; i += 4) {
            // accent teal is (79,209,197): blue and green both far above red
            if (bitmap[i + 1] - bitmap[i] > 60 && bitmap[i + 2] - bitmap[i] > 60) n++;
        }
        return n;
    };
    const overBox = async () => (await page.$('.u-over')).boundingBox();

    // Pointer parked off the plot, full range.
    await page.mouse.move(5, 5); await settle();
    let info = await strayInfo();
    assert.equal(info.selPaints, false, `empty selection occupies ${info.selBox} px (border ${info.border})`);
    assert.equal(await cornerAccentPixels(), 0, 'no accent pixel at the plot top-left, full range');

    // Zoom in 9x about the plot centre, park the pointer off the plot again.
    const box = await overBox();
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    for (let i = 0; i < 9; i++) { await page.keyboard.press('+'); await settle(); }
    await page.mouse.move(5, 5); await settle();
    info = await strayInfo();
    assert.equal(info.selPaints, false, 'empty selection still has no box after zoom');
    assert.equal(await cornerAccentPixels(), 0, 'no accent pixel at the plot top-left after zoom');

    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
    if (server) server.close();
}
