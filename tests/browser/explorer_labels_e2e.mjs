// TASK-527/530: a signal's display label is the PRIMARY text (textContent, owner
// D1); the raw spelling is secondary and the row's data-name identity; the
// filter box finds a row by raw name and by label.
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
    await page.goto(pathToFileURL(html).href);
    await page.waitForSelector('.u-over');
    const row = name => '[data-silk="picker-list"] .picker-row[data-name=' + JSON.stringify(name) + ']';
    await page.waitForSelector(row('v(in)'));
    assert.equal(await page.$eval(row('v(in)') + ' .pname', e => e.textContent), '<b>in[0]</b>');
    assert.equal(await page.$eval(row('v(in)') + ' .plabel', e => e.textContent), 'in');
    assert.equal(await page.$(row('v(in)') + ' .pname b'), null, 'label is text, not markup');
    const filter = '[data-silk="picker-filter"]';
    for (const [query, expected] of [['b>in[', true], ['v(in)', true], ['zzz', false]]) {
        await page.$eval(filter, e => { e.value = ''; });
        await page.type(filter, query);
        await page.waitForFunction(() => true);
        const found = !!(await page.$(row('v(in)')));
        assert.equal(found, expected, 'filter ' + query);
    }
    await page.$eval(filter, e => { e.value = ''; e.dispatchEvent(new Event('input', {bubbles: true})); });
    assert.equal(await page.$eval(row('v(in)'), e => e.dataset.name), 'v(in)');
    assert.match(await page.$eval(row('v(in)'), e => e.title), /<b>in\[0\]<\/b> — v\(in\)/);
    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
}
