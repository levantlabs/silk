// TASK-530: a label-only re-emit (run.id changes, state_key does not) restores
// saved view state and raw-key annotations; raw identities hold; group rows
// follow their own collision status.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import puppeteer from 'puppeteer-core';

const [html, chrome, reemit] = process.argv.slice(2);
const browser = await puppeteer.launch({
    executablePath: chrome, headless: true, args: ['--no-sandbox', '--disable-gpu'],
});
const errors = [];
async function open() {
    const page = await browser.newPage();
    await page.setViewport({width: 1400, height: 1000});
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(pathToFileURL(html).href);
    await page.waitForSelector('.u-over');
    return page;
}
const run = (page, cmd, args = {}) =>
    page.evaluate(async (cmd, args) => {
        const r = await window.silkView.run([{cmd, args}]);
        if (!r.ok) throw new Error(JSON.stringify(r));
        return r.results[0];
    }, cmd, args);
try {
    const first = await open();
    const before = await first.evaluate(() => ({id: SILK_VIEW_DATA.run.id, key: SILK_VIEW_DATA.run.state_key,
        names: SILK_VIEW_DATA.plots[0].signals.map(s => s.name),
        labels: SILK_VIEW_DATA.plots[0].signals.map(s => s.label ?? null),
        notes: JSON.stringify(SILK_VIEW_DATA.annotations || null)}));
    await run(first, 'setCols', {n: 2});
    await run(first, 'setTreeMode', {on: false});
    await run(first, 'addPanel', {signals: [before.names[0]]});
    await run(first, 'setFilter', {text: 'mid'});
    const saved = await run(first, 'getState');
    await first.waitForFunction(key => !!localStorage.getItem('silk.view.' + key), {}, before.key);
    await new Promise(r => setTimeout(r, 800));            // the 400 ms save debounce
    await first.close();

    const [cmd, ...cmdArgs] = JSON.parse(reemit);
    execFileSync(cmd, cmdArgs, {stdio: 'inherit'});   // re-emit, labels changed

    const second = await open();
    const after = await second.evaluate(() => ({id: SILK_VIEW_DATA.run.id, key: SILK_VIEW_DATA.run.state_key,
        names: SILK_VIEW_DATA.plots[0].signals.map(s => s.name),
        labels: SILK_VIEW_DATA.plots[0].signals.map(s => s.label ?? null),
        notes: JSON.stringify(SILK_VIEW_DATA.annotations || null)}));
    assert.notEqual(after.id, before.id, 'run.id changes with the labels');
    assert.equal(after.key, before.key, 'state_key does not');
    assert.notDeepEqual(after.labels, before.labels);
    assert.deepEqual(after.names, before.names, 'raw identities hold');
    assert.equal(after.notes, before.notes, 'annotations keep their raw keys');
    assert.match(before.notes, /v\(xx1<0>\.mid\)|v\(other<1>\.a\)/);
    const restored = await run(second, 'getState');
    assert.equal(restored.cols, saved.cols);
    assert.equal(restored.treeMode, saved.treeMode);
    assert.equal(restored.filter, saved.filter);
    const nonEmpty = s => s.plots['0'].panels.map(p => p.signals).filter(sigs => sigs.length);
    assert.deepEqual(nonEmpty(restored), nonEmpty(saved));
    // the annotated raw row still carries its restored note badge, and only it
    await run(second, 'setFilter', {text: ''});
    const rowSel = name => '[data-silk="picker-list"] .picker-row[data-name=' + JSON.stringify(name) + ']';
    await second.waitForSelector(rowSel('v(other<1>.a)') + ' .pbadge.sev-warn');
    assert.equal(await second.$eval(rowSel('v(other<1>.a)') + ' .pbadge', e => e.title), 'check me');
    assert.equal(await second.$(rowSel('v(xx1<0>.mid)') + ' .pbadge'), null, 'no badge on an unannotated raw row');
    await run(second, 'setTreeMode', {on: true});
    await run(second, 'expandAll');
    const groups = await second.$$eval('[data-silk="picker-list"] .picker-row .pname', els => els.map(e => e.textContent));
    assert.ok(groups.includes('xx1<0>') && groups.includes('xx1[0]'), 'collision groups keep their raw segments: ' + groups);
    assert.ok(groups.includes('other[1]'), 'an unrelated labelled group shows [k]: ' + groups);
    assert.deepEqual(errors, []);
    console.log('OK');
} finally {
    await browser.close();
}
