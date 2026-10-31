// TASK-241: the toolbar must wrap onto additional rows as the explorer
// narrows -- a host embeds this explorer in a panel routinely narrower than
// a browser window, with no visible scrollbar to discover a horizontally
// scrolling toolbar. This drives the REAL explorer (not a markup stand-in)
// at embedded-panel widths and checks every control stays reachable, at a
// consistent size, and the plot area yields the height wrapped rows need.
//
// Usage: node explorer_toolbar_wrap_e2e.mjs <consumer-dir> <chrome-executable>
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import puppeteer from 'puppeteer-core';

const [directory, chrome] = process.argv.slice(2);
const html = `<!doctype html><style>
*{box-sizing:content-box}body{margin:0}.mount{height:900px}
</style><div id="mount" class="mount" style="width:1400px"></div>`;
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
        window.mount = document.getElementById('mount');
        window.a = createExplorer(mount, {uPlot, waveformCore: core});
        await a.load(payload);
        const plotted = payload.plots[payload.default_plot].signals[0].name;
        const result = await a.run([{cmd: 'addSignal', args: {signals: [plotted]}}]);
        if (!result.ok) throw new Error(JSON.stringify(result));
    });

    // Every direct control in the always-visible toolbar groups, keyed by a
    // stable attribute -- excludes hidden file inputs and badges.
    const CONTROL_SELECTOR = '[data-silk="toolbar-left"] > :not([hidden]), '
        + '[data-silk="toolbar-right"] > :not([hidden])';

    async function measure(width, selector) {
        await page.evaluate((w) => { mount.style.width = w + 'px'; }, width);
        // Let the container query and flex layout settle.
        await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
        return page.evaluate((selector) => {
            const root = document.querySelector('.silk-explorer');
            const toolbar = document.querySelector('[data-silk="toolbar"]');
            const body = document.querySelector('[data-silk="body"]');
            const rootRect = root.getBoundingClientRect();
            const toolbarRect = toolbar.getBoundingClientRect();
            const bodyRect = body.getBoundingClientRect();
            const controls = Array.from(document.querySelectorAll(selector)).map((el) => {
                const r = el.getBoundingClientRect();
                return {name: el.getAttribute('data-silk') || el.tagName, left: r.left, right: r.right, width: r.width};
            });
            return {
                rootWidth: rootRect.width,
                toolbarHeight: toolbarRect.height,
                toolbarScrollWidth: toolbar.scrollWidth,
                toolbarClientWidth: toolbar.clientWidth,
                bodyTop: bodyRect.top,
                toolbarBottom: toolbarRect.bottom,
                controls,
                byName: Object.fromEntries(controls.map((c) => [c.name, c])),
            };
        }, selector);
    }

    const wide = await measure(1400, CONTROL_SELECTOR);
    const embedded = await measure(400, CONTROL_SELECTOR);
    const tight = await measure(300, CONTROL_SELECTOR);

    for (const [label, snap] of [['wide (1400px)', wide], ['embedded panel (400px)', embedded], ['tight panel (300px)', tight]]) {
        check(`${label}: toolbar has no horizontal overflow`,
            snap.toolbarScrollWidth <= snap.toolbarClientWidth + 1,
            `scrollWidth=${snap.toolbarScrollWidth} clientWidth=${snap.toolbarClientWidth}`);
        const outOfBounds = snap.controls.filter((c) => c.left < -0.5 || c.right > snap.rootWidth + 0.5);
        check(`${label}: every toolbar control stays within the explorer bounds (reachable)`,
            outOfBounds.length === 0,
            JSON.stringify(outOfBounds));
        check(`${label}: plot area starts at or below the toolbar's bottom edge (no overlap)`,
            snap.bodyTop >= snap.toolbarBottom - 0.5,
            `bodyTop=${snap.bodyTop} toolbarBottom=${snap.toolbarBottom}`);
    }

    check('narrowing wraps the toolbar onto additional rows (height grows)',
        embedded.toolbarHeight > wide.toolbarHeight && tight.toolbarHeight >= embedded.toolbarHeight,
        `wide=${wide.toolbarHeight} embedded=${embedded.toolbarHeight} tight=${tight.toolbarHeight}`);

    for (const name of ['xlog-toggle', 'layout-toggle', 'markers-toggle', 'shortcuts-btn']) {
        const w0 = wide.byName[name], w1 = embedded.byName[name], w2 = tight.byName[name];
        check(`control "${name}" keeps a consistent width as the panel narrows (not shrunk)`,
            w0 && w1 && w2 && Math.abs(w0.width - w1.width) < 1 && Math.abs(w0.width - w2.width) < 1,
            JSON.stringify({wide: w0 && w0.width, embedded: w1 && w1.width, tight: w2 && w2.width}));
    }

    check('no uncaught page errors', errors.length === 0, errors.join('; '));
} finally {
    await browser.close();
    server.close();
}

const failed = results.filter(({pass}) => !pass).length;
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
