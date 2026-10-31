// TASK-319: one collapsible section per signal kind (Voltages, Currents) in
// the picker, table-driven; a kind with no table entry (here: power) stays
// in the ordinary, unsectioned tree. Covers: defaults (Voltages open,
// Currents closed), one-click toggle, a closed header's count/plotted text,
// section+rest+derived ordering, nesting inside a section still matching
// today's tree, and -- the one AC that most needs a real browser -- that
// collapsing a section never touches what is plotted.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import puppeteer from 'puppeteer-core';

const [html, chrome] = process.argv.slice(2);
const browser = await puppeteer.launch({
    executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-gpu'],
});
try {
    const page = await browser.newPage();
    await page.setViewport({width: 1400, height: 1000});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.evaluateOnNewDocument(() => {
        let ctor;
        window.charts = [];
        Object.defineProperty(window, 'uPlot', {
            configurable: true, get: () => ctor,
            set: value => {
                ctor = new Proxy(value, {construct(target, args) {
                    const u = new target(...args);
                    charts.push(u);
                    return u;
                }});
            },
        });
        window.command = async (cmd, args = {}) => {
            const result = await silkView.run({cmd, args});
            if (!result.ok) throw new Error(JSON.stringify(result));
            return result.results[0];
        };
        window.chart = () => charts.find(u => u.root.isConnected);
    });
    await page.goto(pathToFileURL(html).href);
    await page.waitForSelector('.u-over');

    const sectionHeader = key => '[data-silk="picker-list"] .picker-row.group-row[title=' + JSON.stringify(key) + ']';
    const row = name => '[data-silk="picker-list"] .picker-row[data-name=' + JSON.stringify(name) + ']';
    const rowIds = () => page.evaluate(() =>
        Array.from(document.querySelectorAll('[data-silk="picker-list"] .picker-row'))
            .map(el => el.dataset.name || el.title));

    // ---- defaults: Voltages open, Currents closed -------------------------
    await page.waitForSelector(sectionHeader('kind:voltage'));

    // Pre-existing quirk, unrelated to TASK-319: the picker list's first-ever
    // click also moves DOM focus into it, whose 'focus' handler re-renders
    // mid-gesture and detaches the clicked element before mouseup -- the
    // browser then never synthesizes a 'click' event at all (verified this
    // affects an ordinary, already-existing group twisty just as much as a
    // new section one; reported separately). One harmless click primes
    // focus so every *real* click below (a twisty toggle) behaves like a
    // user's second-or-later click, which always works.
    await page.click('[data-silk="picker-list"]');
    assert.equal(await page.$eval(sectionHeader('kind:voltage') + ' .pname', e => e.textContent), 'Voltages');
    assert.equal(await page.$eval(sectionHeader('kind:voltage') + ' .ptwisty', e => e.textContent), '▾');
    assert.equal(await page.$eval(sectionHeader('kind:voltage') + ' .pcount', e => e.textContent), '(2)');

    assert.equal(await page.$eval(sectionHeader('kind:current') + ' .pname', e => e.textContent), 'Currents');
    assert.equal(await page.$eval(sectionHeader('kind:current') + ' .ptwisty', e => e.textContent), '▸');
    assert.equal(await page.$eval(sectionHeader('kind:current') + ' .pcount', e => e.textContent), '(2 · 0 plotted)');

    // Voltages' rows are visible (open by default); Currents' are not (closed).
    assert.ok(await page.$(row('v(vdd)')), 'v(vdd) should be visible under the open Voltages section');
    assert.equal(await page.$(row('i(vvclk)')), null, 'i(vvclk) should be hidden under the closed Currents section');

    // The "any other kind" signal (power, no table entry) stays in the
    // ordinary tree, unsectioned -- visible without opening anything.
    assert.ok(await page.$(row('p(out)')), 'p(out) (an unsectioned kind) should already be visible');

    // ---- one-click open ------------------------------------------------
    await page.click(sectionHeader('kind:current') + ' .ptwisty');
    await page.waitForSelector(row('i(vvclk)'));
    assert.equal(await page.$eval(sectionHeader('kind:current') + ' .ptwisty', e => e.textContent), '▾');
    assert.equal(await page.$eval(sectionHeader('kind:current') + ' .pcount', e => e.textContent), '(2)');

    // A section opening does not itself expand the ordinary hierarchy
    // groups inside it -- "xdut" (nested under both Voltages and Currents
    // here, same device name in both) still starts collapsed like any other
    // tree group, unrelated to the section's own default. Expand both to
    // check the full order below. Each is its OWN group, scoped by section
    // (fullKey "kind:voltage/xdut" vs "kind:current/xdut") specifically so
    // the two same-named groups never share one expand/collapse toggle.
    await page.click('.picker-row.group-row[title="kind:voltage/xdut"] .ptwisty');
    await page.waitForSelector(row('v(xdut.vsup)'));
    await page.click('.picker-row.group-row[title="kind:current/xdut"] .ptwisty');
    await page.waitForSelector(row('i(xdut.vsup)'));

    // Full row order, everything open: Voltages header, its tree (a
    // top-level leaf then a nested group), Currents header, its tree (same
    // shape -- nesting inside a section matches today's tree), then the
    // unsectioned "power" leaf last (no Derived signals here).
    assert.deepEqual(await rowIds(), [
        'kind:voltage', 'v(vdd)', 'kind:voltage/xdut', 'v(xdut.vsup)',
        'kind:current', 'i(vvclk)', 'kind:current/xdut', 'i(xdut.vsup)',
        'p(out)',
    ]);
    // The nested current still reads i(vsup) under its group (TASK-317/318
    // untouched), not the full i(xdut.vsup) repeated under "xdut".
    assert.equal(await page.$eval(row('i(xdut.vsup)') + ' .pname', e => e.textContent), 'i(vsup)');

    // ---- collapsing never changes the plot ------------------------------
    await page.evaluate(() => command('addSignal', {signals: ['i(vvclk)']}));
    await page.waitForFunction(() => window.chart() && window.chart().series.some(s => s.label === 'i(vvclk)'));

    await page.click(sectionHeader('kind:current') + ' .ptwisty'); // collapse again
    await page.waitForFunction(sel => !document.querySelector(sel), {}, row('i(vvclk)'));
    assert.equal(await page.$eval(sectionHeader('kind:current') + ' .pcount', e => e.textContent), '(2 · 1 plotted)');

    const seriesLabels = await page.evaluate(() => window.chart().series.map(s => s.label));
    assert.ok(seriesLabels.includes('i(vvclk)'),
        'collapsing the Currents section must not remove the already-plotted trace: ' + seriesLabels);

    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
}
