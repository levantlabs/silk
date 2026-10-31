// TASK-319, reviewer decision (c): a saved layout's kind-section open/closed
// state lives in its OWN 'sections' field, never in 'expanded'. A layout
// saved before this feature existed has 'expanded' but no 'sections' key at
// all -- that absence must apply the KIND_SECTIONS defaults (Voltages open,
// Currents closed), and restoring it must leave its ordinary tree-group
// expansion (an unsectioned "any other kind" group, here the power group
// "xdut") untouched, proving the two fields restore independently.
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
    const url = pathToFileURL(html).href;
    await page.goto(url);
    await page.waitForSelector('.u-over');

    const sectionHeader = key => '[data-silk="picker-list"] .picker-row.group-row[title=' + JSON.stringify(key) + ']';
    const row = name => '[data-silk="picker-list"] .picker-row[data-name=' + JSON.stringify(name) + ']';
    const xdutGroup = '.picker-row.group-row[title="xdut"]';

    // A pre-TASK-319 saved layout: 'expanded' names one ordinary tree group
    // (the unsectioned power group "xdut", whose fullKey has no kind-section
    // prefix, exactly as an old build would have computed it) and there is
    // deliberately no 'sections' key anywhere in the object -- an old build
    // never wrote one.
    const stateKey = await page.evaluate(() => SILK_VIEW_DATA.run.state_key);
    const preFeatureLayout = {
        v: 1,
        plots: {
            '0': {
                // A panel needs at least one valid signal, or
                // PanelManager.pruneSavedLayout drops the whole saved layout
                // (including 'expanded') as if none existed at all.
                panels: [{signals: ['v(vdd)'], yMode: 'linear'}],
                xRange: null, xLog: false, cursors: [], markers: [],
                expanded: ['xdut'],
                // no 'sections' key: this layout predates the feature.
            },
        },
        activePlot: 0, filter: '', cols: 1, treeMode: true, fontScale: 'normal',
    };
    await page.evaluate((key, layout) => localStorage.setItem(key, JSON.stringify(layout)),
        'silk.view.' + stateKey, preFeatureLayout);

    await page.reload();
    await page.waitForSelector('.u-over');
    await page.waitForSelector(sectionHeader('kind:voltage'));

    // Same pre-existing focus/click priming as explorer_kind_sections_e2e.mjs
    // -- not exercised by this test (no clicks below), kept only in case a
    // future edit adds one; harmless either way.
    await page.click('[data-silk="picker-list"]');

    // No 'sections' field -> defaults apply: Voltages open, Currents closed.
    assert.equal(await page.$eval(sectionHeader('kind:voltage') + ' .ptwisty', e => e.textContent), '▾',
        'Voltages must open by default when the saved layout has no sections field');
    assert.ok(await page.$(row('v(vdd)')), 'v(vdd) should be visible under the open Voltages section');

    assert.equal(await page.$eval(sectionHeader('kind:current') + ' .ptwisty', e => e.textContent), '▸',
        'Currents must stay closed by default when the saved layout has no sections field');
    assert.equal(await page.$(row('i(vvclk)')), null, 'i(vvclk) should be hidden under the closed Currents section');

    // The pre-existing 'expanded' entry for the ordinary "xdut" group (an
    // unsectioned "any other kind" -- power -- so its key was never
    // prefixed) restores exactly as saved: still expanded, independent of
    // sections restoring to their defaults.
    assert.equal(await page.$eval(xdutGroup + ' .ptwisty', e => e.textContent), '▾',
        'A pre-feature layout\'s ordinary group expansion must restore unchanged');
    assert.ok(await page.$(row('p(xdut.out)')), 'p(xdut.out) should be visible: its group "xdut" was saved expanded');

    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
}
