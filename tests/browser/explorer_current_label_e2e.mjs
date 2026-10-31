// TASK-317: a current signal's picker row shows wrapper(leaf) -- its full
// rawfile name at top level ("i(v1)"), not the bare leaf ("v1"); rebuilt
// from wrapper/leaf rather than the raw name so a NESTED current under a
// group ("i(xdut.vsup)") shows just "i(vsup)" there, not the whole path
// repeated under its own group row. The separate wrapper badge is left
// exactly as it is today for both voltages and currents (a follow-up
// decides that badge's fate) -- only the .pname text changes.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import puppeteer from 'puppeteer-core';

const [html, chrome, nested_html] = process.argv.slice(2);
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

    const row = name => '[data-silk="picker-list"] .picker-row[data-name=' + JSON.stringify(name) + ']';

    // The current row (rawfile vector "i(v1)") shows its full name; no
    // wrapper badge is rendered (TASK-318).
    await page.waitForSelector(row('i(v1)'));
    assert.equal(await page.$eval(row('i(v1)') + ' .pname', e => e.textContent), 'i(v1)');
    assert.equal(await page.$(row('i(v1)') + ' .pwrapper'), null);

    // A voltage row (rawfile vector "v(in)") is unchanged: leaf-only name,
    // no wrapper badge.
    await page.waitForSelector(row('v(in)'));
    assert.equal(await page.$eval(row('v(in)') + ' .pname', e => e.textContent), 'in');
    assert.equal(await page.$(row('v(in)') + ' .pwrapper'), null);

    // A current nested under a group ("i(xdut.vsup)", one dotted segment
    // "xdut") shows "i(vsup)" once the group is expanded -- not the whole
    // "i(xdut.vsup)" repeated right under the "xdut" group row.
    await page.goto(pathToFileURL(nested_html).href);
    await page.waitForSelector('.u-over');
    const expanded = await page.evaluate(() => silkView.run({cmd: 'expand', args: {path: 'xdut'}}));
    assert.equal(expanded.ok, true, JSON.stringify(expanded));
    await page.waitForSelector(row('i(xdut.vsup)'));
    assert.equal(await page.$eval(row('i(xdut.vsup)') + ' .pname', e => e.textContent), 'i(vsup)');
    assert.equal(await page.$(row('i(xdut.vsup)') + ' .pwrapper'), null);
    // The top-level current under the same page is unaffected by nesting.
    await page.waitForSelector(row('i(vvclk)'));
    assert.equal(await page.$eval(row('i(vvclk)') + ' .pname', e => e.textContent), 'i(vvclk)');

    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
}
