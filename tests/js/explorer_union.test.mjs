// SPDX-License-Identifier: MIT
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {compileFunction} from 'node:vm';

// The union-axis contract, provenance and finding
// rule are recorded beside the unchanged, independently hand-computed JSON.
const fixture = JSON.parse(await readFile(new URL('../fixtures/union-x-semantics.json', import.meta.url), 'utf8'));
const explorer_url = new URL('../../silk/signals/view/assets/explorer.mjs', import.meta.url);
const source = await readFile(explorer_url, 'utf8');

// These functions are private to createExplorer / PanelManager. Evaluate
// their actual source without mounting a DOM or copying the merge algorithm.
// Missing/ambiguous boundaries must fail loudly if the module is reorganized.
function source_block(start, end) {
    assert.equal(source.split(start).length, 2, `Expected one source boundary: ${start}`);
    assert.equal(source.split(end).length, 2, `Expected one source boundary: ${end}`);
    const first = source.indexOf(start), last = source.indexOf(end);
    assert.ok(last > first, `Source boundaries out of order: ${start}, ${end}`);
    return source.slice(first, last);
}

const numerical_source = [
    source_block('  function isFiniteNum(', '  function levenshtein('),
    source_block('  function b64ToBuffer(', '  var PAYLOAD ='),
    source_block('    function unionX(', '    function makeHooks('),
].join('\n');

function load_merge() {
    const load = compileFunction(
        numerical_source + '\nreturn {Store, unionX, buildData, makeSeries};',
        ['AxisModes', 'colorFor'], {filename: explorer_url.pathname},
    );
    return load({for_signal: () => ({key: 'y'})}, () => '#000');
}

for (const sample of fixture.cases) {
    test(`${sample.name}: union preserves repeated x multiplicity`, () => {
        const {unionX} = load_merge();
        const actual = unionX(sample.signals.map(signal => new Float64Array(signal.x)));
        assert.ok(actual instanceof Float64Array);
        assert.deepEqual(Array.from(actual), sample.expected.x);
    });

    test(`${sample.name}: owned y positions and both hole strategies`, () => {
        const {Store, buildData, makeSeries} = load_merge();
        const signals = sample.signals.map(signal => ({
            name: signal.name,
            nonfinite: signal.nonfinite,
            levels: [{
                stride: sample.stride, n: signal.x.length, dt: 'f64',
                x: new Float64Array(signal.x),
                // Convert only the input NaN/Infinity strings. The real Store
                // must nullify them; expectations stay literal and untouched.
                y: Float64Array.from(signal.y, Number),
            }],
        }));
        Store.init({plots: [{x: {name: 'time'}, signals}]});
        const panel = {plotIdx: 0, signals: signals.map(signal => signal.name)};
        const data = buildData(panel, sample.stride);
        assert.deepEqual(Array.from(data[0]), sample.expected.x, 'shared x');
        assert.equal(data.length, sample.signals.length + 1);
        for (let i = 0; i < sample.signals.length; i++) {
            assert.deepEqual(Array.from(data[i + 1]), sample.expected.y[i], sample.signals[i].name);
        }
        const series = makeSeries(panel).slice(1);
        assert.deepEqual(series.map(signal => signal.label), panel.signals);
        // uPlot defaults an omitted spanGaps to false.
        assert.deepEqual(series.map(signal => Boolean(signal.spanGaps)), sample.expected.spanGaps);
    });
}

// TASK-530 (owner D1): the series label is the signal's display label; the
// raw name stays the series' lookup key in panel.signals.
test('series label uses the display label, panel keys stay raw', () => {
    const {Store, makeSeries} = load_merge();
    const signal = (name, label) => ({name, label, unit: 'V', type: 'voltage', levels: [{stride: 1, n: 2,
        x: new Float64Array([0, 1]), y: new Float64Array([0, 1]), dt: 'f64'}]});
    Store.init({plots: [{x: {name: 'time'}, signals: [signal('v(xx1<0>.m)', 'v(xx1[0].m)'), signal('v(o)')]}]});
    const panel = {plotIdx: 0, signals: ['v(xx1<0>.m)', 'v(o)']};
    assert.deepEqual(makeSeries(panel).slice(1).map(s => s.label), ['v(xx1[0].m)', 'v(o)']);
    assert.deepEqual(panel.signals, ['v(xx1<0>.m)', 'v(o)']);
});
