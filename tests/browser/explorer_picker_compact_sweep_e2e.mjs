// TASK-341: below a container-query-derived breakpoint, the picker's action
// rows (Plot visible / Clear panel / Add / Tree / Expand all / Collapse all)
// collapse into a single anchored "more actions" menu and picker-count moves
// inline into the filter row -- see the @container silk-sidebar block in
// view.css and Picker.installPickerMoreMenu/fillPickerMoreMenu in
// explorer.mjs. This sweeps the sidebar's own rendered height from 150px to
// 420px in 10px steps (crossing the 250px breakpoint) and at every step
// asserts:
//   - at least 5 fully visible picker-list rows (AC #2)
//   - TASK-338's no-layout-shift guarantee still holds: a plain click from
//     an empty selection leaves every row's position unchanged, and a
//     double-click on a leaf plots exactly that leaf (AC #3) -- checked both
//     with the compact "more actions" menu closed and, below the breakpoint,
//     with it open (an anchored, position:absolute overlay, so it must never
//     move picker-list either)
//
// Usage: node explorer_picker_compact_sweep_e2e.mjs <consumer-dir> <chrome-executable>
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import puppeteer from 'puppeteer-core';

const [directory, chrome] = process.argv.slice(2);
const ROW_H = 24;
const BREAKPOINT = 250; // view.css: @container silk-sidebar (max-height: 250px)
const html = `<!doctype html><style>
*{box-sizing:content-box}body{margin:0}.mount{position:relative}
</style>
<div id="m" class="mount" style="width:900px;height:900px"></div>`;
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
const sweepTable = [];
function check(name, condition, detail = '') {
    const pass = Boolean(condition);
    results.push({name, pass});
    console.log(`${pass ? 'PASS' : 'FAIL'} :: ${name}${detail ? `  (${detail})` : ''}`);
}

try {
    const page = await browser.newPage();
    await page.setViewport({width: 1000, height: 1000});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.addStyleTag({url: '/view.css'});
    await page.addScriptTag({url: '/uplot.iife.min.js'});
    await page.evaluate(async () => {
        const {createExplorer} = await import('/explorer.mjs');
        window.core = await import('/waveform-core.mjs');
        window.payload = await (await fetch('/sweep_payload.json')).json();
        const mount = document.getElementById('m');
        window.explorer = createExplorer(mount, {uPlot, waveformCore: core});
        await explorer.load(payload);
        const result = await explorer.run({cmd: 'setTreeMode', args: {on: false}});
        if (!result.ok) throw new Error(JSON.stringify(result));

        // Same hand-dispatched real-double-click helper as
        // explorer_row_stability_e2e.mjs -- see that file's comment for why
        // Puppeteer's own {clickCount:2} can't reproduce a real double-click.
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

    const listSel = '#m [data-silk="picker-list"]';
    const rowSel = `${listSel} .picker-row[data-name]`;
    const sidebarSel = '#m [data-silk="sidebar"]';
    const actionsSel = '#m [data-silk="picker-actions"]';
    const moreWrapSel = '#m [data-silk="picker-more-wrap"]';
    const moreBtnSel = '#m [data-silk="picker-more-btn"]';
    const moreMenuSel = '#m [data-silk="picker-more-menu"]';
    const countInlineSel = '#m [data-silk="picker-count-inline"]';
    const countSel = '#m [data-silk="picker-count"]';

    // 900px mount width trips the responsive breakpoint (view.css
    // @container silk-explorer (max-width: 900px)): the sidebar starts
    // off-canvas behind a toggle button, same as a real narrow embed. Open
    // it once, same as explorer_row_stability_e2e.mjs's embedded mount.
    const toggleSel = '#m [data-silk="sidebar-toggle"]';
    const toggleVisible = await page.evaluate(sel => {
        const el = document.querySelector(sel);
        return !!el && getComputedStyle(el).display !== 'none';
    }, toggleSel);
    if (toggleVisible) {
        await page.click(toggleSel);
        await new Promise(resolve => setTimeout(resolve, 250)); // outlive the 0.15s slide-in transition
    }

    // Priming click for the known first-click focus-render quirk (see
    // explorer_first_click_e2e.mjs), done once up front, same as
    // explorer_row_stability_e2e.mjs.
    await page.waitForSelector(rowSel);
    await page.click(listSel);
    await page.focus(listSel);
    await page.keyboard.press('Escape');

    async function setSidebarHeight(targetH) {
        // The mount is the only element under direct control; the sidebar's
        // height is mount height minus the toolbar's, and the toolbar's own
        // height isn't perfectly stable in advance (TASK-241: it can wrap
        // depending on width/state). Converge with a couple of
        // measure-and-correct passes instead of trusting a single guess.
        let mountH = targetH + 75;
        for (let attempt = 0; attempt < 3; attempt++) {
            await page.evaluate(h => { document.getElementById('m').style.height = h + 'px'; }, Math.round(mountH));
            await new Promise(resolve => setTimeout(resolve, 80));
            const sidebarH = await page.$eval(sidebarSel, el => el.getBoundingClientRect().height);
            if (Math.abs(sidebarH - targetH) < 1) return sidebarH;
            mountH += (targetH - sidebarH);
        }
        return page.$eval(sidebarSel, el => el.getBoundingClientRect().height);
    }

    const rowTops = () => page.$$eval(rowSel, rows =>
        rows.map(r => ({name: r.dataset.name, top: r.getBoundingClientRect().top})));

    async function assertNoShift(label) {
        await page.focus(listSel);
        await page.keyboard.press('Escape');
        const before = await rowTops();
        assert.ok(before.length >= 2, `${label}: fixture needs at least 2 leaf rows, got ${before.length}`);
        await page.click(`${listSel} .picker-row[data-name=${JSON.stringify(before[0].name)}]`);
        const after = await rowTops();
        check(`${label}: single click on a leaf leaves every row's position unchanged`,
            before.length === after.length && before.every((b, i) => b.name === after[i].name
                && Math.abs(b.top - after[i].top) < 0.5),
            JSON.stringify({before, after}));
        await page.focus(listSel);
        await page.keyboard.press('Escape');
        return before; // pre-selection row list, reused for the double-click target
    }

    async function assertDoubleClickPlotsTarget(label, rows) {
        const target = rows[1];
        const point = await page.$eval(`${listSel} .picker-row[data-name=${JSON.stringify(target.name)}]`, e => {
            const r = e.getBoundingClientRect();
            return {x: r.left + r.width / 2, y: r.top + r.height / 2};
        });
        await page.evaluate((x, y) => dblClickAt(x, y), point.x, point.y);
        await new Promise(resolve => setTimeout(resolve, 60));
        const plotted = await page.$$eval('#m [data-silk="panels"] .chip .cname', cs => cs.map(c => c.textContent));
        check(`${label}: double-click on ${target.name} plots exactly that leaf`,
            plotted.length === 1 && plotted[0] === target.name, JSON.stringify(plotted));
        await page.evaluate(async () => { await explorer.run({cmd: 'clearPanel', args: {}}); });
    }

    for (let targetH = 150; targetH <= 420; targetH += 10) {
        const sidebarH = await setSidebarHeight(targetH);
        const compact = await page.$eval(actionsSel, el => getComputedStyle(el).display === 'none');
        const label = `sidebarH=${sidebarH.toFixed(0)} (target ${targetH}, ${compact ? 'compact' : 'full'})`;

        check(`${label}: matches the ${BREAKPOINT}px breakpoint (compact iff sidebarH<=${BREAKPOINT})`,
            compact === (sidebarH <= BREAKPOINT), `sidebarH=${sidebarH}`);

        // DOMRect's fields are inherited getters, not own properties -- a
        // bare `getBoundingClientRect()` return value serializes across the
        // page.$eval boundary as `{}`. Pull out plain numbers instead.
        const listRect = await page.$eval(listSel, el => {
            const r = el.getBoundingClientRect();
            return {top: r.top, bottom: r.bottom};
        });
        const rowRects = await page.$$eval(rowSel, rows => rows.map(r => {
            const b = r.getBoundingClientRect();
            return {top: b.top, bottom: b.bottom};
        }));
        const fullyVisible = rowRects.filter(r => r.top >= listRect.top - 0.5 && r.bottom <= listRect.bottom + 0.5).length;
        check(`${label}: at least 5 fully visible rows`, fullyVisible >= 5, `fullyVisible=${fullyVisible}/${rowRects.length}`);

        if (compact) {
            const countInlineVisible = await page.$eval(countInlineSel, el => getComputedStyle(el).display !== 'none');
            const countRowVisible = await page.$eval(countSel, el => getComputedStyle(el).display !== 'none');
            const countInlineText = await page.$eval(countInlineSel, el => el.textContent);
            check(`${label}: count is inline in the filter row, not the bottom row`,
                countInlineVisible && !countRowVisible && /^\d+\/\d+$/.test(countInlineText), countInlineText);
        } else {
            const moreVisible = await page.$eval(moreWrapSel, el => getComputedStyle(el).display !== 'none');
            const countRowVisible = await page.$eval(countSel, el => getComputedStyle(el).display !== 'none');
            check(`${label}: more-actions menu hidden, bottom count row shown (full mode)`,
                !moreVisible && countRowVisible);
        }

        sweepTable.push({sidebarH: Math.round(sidebarH), mode: compact ? 'compact' : 'full', fullyVisible});

        // TASK-338 guarantee, menu closed.
        const rowsBeforeDbl = await assertNoShift(`${label} menu-closed`);
        await assertDoubleClickPlotsTarget(`${label} menu-closed`, rowsBeforeDbl);

        if (compact) {
            // The overlay must never move picker-list, opened or closed.
            await page.focus(listSel);
            await page.keyboard.press('Escape');
            const beforeOpen = await rowTops();
            await page.click(moreBtnSel);
            const menuOpen = await page.$eval(moreMenuSel, el => !el.hidden);
            check(`${label}: picker-more-menu opens`, menuOpen);
            const afterOpen = await rowTops();
            check(`${label}: opening picker-more-menu leaves every row's position unchanged`,
                beforeOpen.length === afterOpen.length && beforeOpen.every((b, i) => b.name === afterOpen[i].name
                    && Math.abs(b.top - afterOpen[i].top) < 0.5));
            const rowRectsOpen = await page.$$eval(rowSel, rows => rows.map(r => {
                const b = r.getBoundingClientRect();
                return {top: b.top, bottom: b.bottom};
            }));
            const fullyVisibleOpen = rowRectsOpen.filter(r => r.top >= listRect.top - 0.5 && r.bottom <= listRect.bottom + 0.5).length;
            check(`${label}: still at least 5 fully visible rows with the menu open`, fullyVisibleOpen >= 5,
                `fullyVisibleOpen=${fullyVisibleOpen}`);

            // Proceed with the standard guarantee checks starting from the
            // OPEN state: selecting a row closes picker-more-menu as a side
            // effect (Picker.updateSelectionUI -> close_picker_more_menu,
            // the same idiom already used for derive_menu) -- confirms that
            // interaction doesn't reintroduce a shift or leave the menu
            // stuck open over the list it would otherwise occlude.
            const rowsWithMenuOpen = await assertNoShift(`${label} menu-open`);
            check(`${label}: picker-more-menu auto-closes once a row is selected`,
                await page.$eval(moreMenuSel, el => el.hidden));
            await assertDoubleClickPlotsTarget(`${label} menu-open`, rowsWithMenuOpen);
        }
    }

    check('no uncaught page errors', errors.length === 0, errors.join('; '));
} finally {
    await browser.close();
    server.close();
}

console.log('\nsweep table (sidebarH -> mode, fully-visible rows):');
console.table(sweepTable);

const failed = results.filter(({pass}) => !pass).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
