// TASK-319 AC#3: with groupSignalsByKind off, the rendered picker rows are
// exactly what main (pre-TASK-319) rendered -- captured once, by hand,
// against an unmodified checkout, and pinned here as BASELINE_ROWS. See the
// task notes for the exact capture command.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import puppeteer from 'puppeteer-core';

const [html, chrome] = process.argv.slice(2);

// Captured from git rev 21dc5a9b (main, before any TASK-319 change), with the
// TASK-318 change (no wrapper badge, pwrapper always null) applied, against
// the same fixture this script is given (fully expanded via `expandAll`, so
// the comparison does not depend on default-collapsed state), reading each
// visible .picker-row's classes/name/pname/pwrapper/pcount/ptype/indent
// width in DOM order.
const BASELINE_ROWS = [
    {
        cls: 'picker-row group-row', name: null, pname: 'xdut', pwrapper: null,
        pcount: '2', ptype: null, indentPx: null,
    },
    {
        cls: 'picker-row', name: 'v(xdut.vsup)', pname: 'vsup', pwrapper: null,
        pcount: null, ptype: 'V', indentPx: '12px',
    },
    {
        cls: 'picker-row', name: 'i(xdut.vsup)', pname: 'i(vsup)', pwrapper: null,
        pcount: null, ptype: 'A', indentPx: '12px',
    },
    {
        cls: 'picker-row', name: 'i(vvclk)', pname: 'i(vvclk)', pwrapper: null,
        pcount: null, ptype: 'A', indentPx: null,
    },
];

const browser = await puppeteer.launch({
    executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-gpu'],
});
try {
    const page = await browser.newPage();
    await page.setViewport({width: 1400, height: 1000});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(html).href);
    await page.waitForSelector('.u-over');
    await page.evaluate(() => silkView.run({cmd: 'expandAll'}));

    const rows = await page.evaluate(() => Array.from(
        document.querySelectorAll('[data-silk="picker-list"] .picker-row'),
    ).map(el => ({
        cls: el.className,
        name: el.dataset.name || null,
        pname: el.querySelector('.pname') ? el.querySelector('.pname').textContent : null,
        pwrapper: el.querySelector('.pwrapper') ? el.querySelector('.pwrapper').textContent : null,
        pcount: el.querySelector('.pcount') ? el.querySelector('.pcount').textContent : null,
        ptype: el.querySelector('.ptype') ? el.querySelector('.ptype').textContent : null,
        indentPx: el.querySelector('.pindent') ? el.querySelector('.pindent').style.width : null,
    })));

    assert.deepEqual(rows, BASELINE_ROWS,
        'groupSignalsByKind:false must restore today\'s (pre-TASK-319) list exactly');

    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
}
