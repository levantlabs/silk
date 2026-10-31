// TASK-338: selecting a leaf row must never move the picker list under the
// pointer. add-selected-btn used to sit `hidden` (display:none) above
// picker-tree-actions; selecting a row un-hid it, which grew picker-controls
// and pushed every row in picker-list down by more than one row height. A
// double-click's second click then landed on the wrong row (or on nothing).
// Fixed by reserving the button's slot always (visibility, not display) --
// see the [data-silk="add-selected-btn"][hidden] rule in view.css.
//
// Covers both a full-size standalone mount and a short, narrow mount
// matching how a host embeds this explorer in a panel.
//
// Usage: node explorer_row_stability_e2e.mjs <consumer-dir> <chrome-executable>
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import puppeteer from 'puppeteer-core';

const [directory, chrome] = process.argv.slice(2);
const html = `<!doctype html><style>
*{box-sizing:content-box}body{margin:0}.mount{position:relative}
</style>
<div id="standalone" class="mount" style="width:1400px;height:900px"></div>
<div id="embedded" class="mount" style="width:360px;height:420px"></div>`;
const server = http.createServer((req, res) => {
    const name = decodeURIComponent(req.url.split('?')[0]);
    if (name === '/') { res.setHeader('Content-Type', 'text/html'); res.end(html); return; }
    const file = path.join(directory, path.basename(name));
    if (!fs.existsSync(file)) { res.writeHead(404); res.end('missing asset'); return; }
    res.setHeader('Content-Type', name.endsWith('.mjs') || name.endsWith('.js') ? 'text/javascript'
        : name.endsWith('.css') ? 'text/css' : name.endsWith('.json') ? 'application/json'
        : 'application/octet-stream');
    res.end(fs.readFileSync(file));
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await puppeteer.launch({executablePath: chrome, headless: true,
    args: ['--no-sandbox', '--disable-gpu']});
const results = [];
function check(name, condition, detail = '') {
    const pass = Boolean(condition);
    results.push({name, pass});
    console.log(`${pass ? 'PASS' : 'FAIL'} :: ${name}${detail ? `  (${detail})` : ''}`);
}

try {
    const page = await browser.newPage();
    await page.setViewport({width: 1520, height: 1000});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.addStyleTag({url: '/view.css'});
    await page.addScriptTag({url: '/uplot.iife.min.js'});
    await page.evaluate(async () => {
        const {createExplorer} = await import('/explorer.mjs');
        window.core = await import('/waveform-core.mjs');
        window.payload = await (await fetch('/payload.json')).json();
        window.explorers = {};
        for (const id of ['standalone', 'embedded']) {
            const mount = document.getElementById(id);
            const explorer = createExplorer(mount, {uPlot, waveformCore: core});
            await explorer.load(payload);
            // Flat picker, no kind sections/tree groups -- a deterministic
            // list of one leaf row per signal, in file order.
            const result = await explorer.run({cmd: 'setTreeMode', args: {on: false}});
            if (!result.ok) throw new Error(JSON.stringify(result));
            explorers[id] = explorer;
        }

        // A real double-click is two independent mousedown/mouseup cycles at
        // ONE fixed screen point (the pointer never moves between them) --
        // the second cycle hit-tests whatever is under that point AT THAT
        // TIME, so if the first click's selection shifted the list, the
        // second lands on a different row (or nothing). Puppeteer's own
        // `clickCount` option can't reproduce this: with `count` left at its
        // default of 1, `Mouse.click` skips its priming loop entirely and
        // sends a SINGLE down/up pair tagged `clickCount:2`, so the DOM
        // never sees an intervening render() to shift anything under the
        // second press. Dispatch both cycles by hand instead, synchronously,
        // exactly reproducing what two real presses at a fixed point do
        // (the first click's handler-driven render() runs and completes
        // before the second cycle's elementFromPoint is even read).
        window.dblClickAt = function (x, y) {
            function fire(type, el, detail) {
                el.dispatchEvent(new MouseEvent(type, {
                    bubbles: true, cancelable: true, view: window,
                    clientX: x, clientY: y, detail: detail, button: 0,
                }));
            }
            const el1 = document.elementFromPoint(x, y);
            fire('mousedown', el1, 1);
            fire('mouseup', el1, 1);
            fire('click', el1, 1);
            const el2 = document.elementFromPoint(x, y);
            fire('mousedown', el2, 2);
            fire('mouseup', el2, 2);
            fire('click', el2, 2);
            fire('dblclick', el2, 2);
        };
    });

    async function rowStability(mountId) {
        const listSel = `#${mountId} [data-silk="picker-list"]`;
        const btnSel = `#${mountId} [data-silk="add-selected-btn"]`;
        const rowSel = `${listSel} .picker-row[data-name]`;

        // The embedded mount is narrow enough to trip the responsive
        // sidebar breakpoint, which starts the sidebar (and its picker)
        // slid off-canvas behind a toggle button. Open it first, same as a
        // real embedded user would.
        const toggleSel = `#${mountId} [data-silk="sidebar-toggle"]`;
        const toggleVisible = await page.evaluate(sel => {
            const el = document.querySelector(sel);
            return !!el && getComputedStyle(el).display !== 'none';
        }, toggleSel);
        if (toggleVisible) {
            await page.click(toggleSel);
            await new Promise(resolve => setTimeout(resolve, 250)); // outlive the 0.15s slide-in transition
        }

        await page.waitForSelector(rowSel);

        const rowTops = () => page.$$eval(rowSel, rows =>
            rows.map(r => ({name: r.dataset.name, top: r.getBoundingClientRect().top})));
        const btnHidden = () => page.$eval(btnSel, b => b.hidden);

        // Same pre-existing focus/click priming used throughout this
        // directory: the very first click a fresh picker list has ever seen
        // can be swallowed by a known focus-render quirk (mousedown moves
        // DOM focus into the list, whose 'focus' listener re-renders
        // mid-gesture and detaches the just-clicked element before mouseup).
        // Prime it once, then force an empty selection regardless of what
        // that priming click happened to land on.
        await page.click(listSel);
        await page.focus(listSel);
        await page.keyboard.press('Escape');
        check(`${mountId}: starts with an empty selection (add-selected-btn hidden)`, await btnHidden());

        const rows = await rowTops();
        assert.ok(rows.length >= 3, `${mountId}: fixture needs at least 3 leaf rows, got ${rows.length}`);
        const before = rows;

        // AC #1: one plain click on a leaf, from an empty selection, must
        // not move ANY row's position -- not just the clicked one.
        await page.click(`${listSel} .picker-row[data-name=${JSON.stringify(before[0].name)}]`);
        check(`${mountId}: selecting a row un-hides add-selected-btn`, !(await btnHidden()));
        const after = await rowTops();
        check(`${mountId}: single click on a leaf leaves every row's position unchanged`,
            before.length === after.length && before.every((b, i) => b.name === after[i].name
                && Math.abs(b.top - after[i].top) < 0.5),
            JSON.stringify({before, after}));

        // Back to an empty selection for the double-click check.
        await page.focus(listSel);
        await page.keyboard.press('Escape');
        check(`${mountId}: Escape restores an empty selection`, await btnHidden());

        // AC #2: a double-click on a leaf from an empty selection plots
        // exactly that leaf. Target the SECOND row (not the first): with a
        // 32px shift and a 24px row height, the row now under the second
        // row's original point is provably the first row, a real, different
        // row -- the clearest demonstration that the wrong signal gets
        // plotted (not merely that nothing does).
        const target = before[1];
        const point = await page.$eval(`${listSel} .picker-row[data-name=${JSON.stringify(target.name)}]`, e => {
            const r = e.getBoundingClientRect();
            return {x: r.left + r.width / 2, y: r.top + r.height / 2};
        });
        await page.evaluate((x, y) => dblClickAt(x, y), point.x, point.y);
        await new Promise(resolve => setTimeout(resolve, 100));
        const plotted = await page.$$eval(`#${mountId} [data-silk="panels"] .chip .cname`,
            cs => cs.map(c => c.textContent));
        check(`${mountId}: double-click on ${target.name} plots exactly that leaf`,
            plotted.length === 1 && plotted[0] === target.name, JSON.stringify(plotted));
    }

    await rowStability('standalone');
    await rowStability('embedded');

    check('no uncaught page errors', errors.length === 0, errors.join('; '));
} finally {
    await browser.close();
    server.close();
}

const failed = results.filter(({pass}) => !pass).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
