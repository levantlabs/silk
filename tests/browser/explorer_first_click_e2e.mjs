// Regression gate (TASK-322): on a fresh page load the first, unprimed click on
// a picker twisty must toggle it. The list's focus handler used to re-render
// mid-gesture and detach the pressed element, so no click was ever fired.
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

    const group = '.picker-row.group-row[title="xdut"]';
    await page.waitForSelector(group);
    assert.equal(await page.$eval(group + ' .ptwisty', e => e.textContent), '▸',
        'xdut starts collapsed');

    // The very first click the page has ever seen -- deliberately no
    // priming click first, unlike every other e2e test in this directory.
    await page.click(group + ' .ptwisty');
    await new Promise(resolve => setTimeout(resolve, 300)); // let any click settle

    assert.equal(await page.$eval(group + ' .ptwisty', e => e.textContent), '▾',
        'a single, unprimed click on a twisty must expand its group');

    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
}
