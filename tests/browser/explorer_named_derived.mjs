// Named math uses the same state and command paths as an embedded host.
import {createExplorer} from './explorer.mjs';
import * as core from './waveform-core.mjs';
const root = document.createElement('div');
root.style.cssText = 'width:1200px;height:850px'; document.body.append(root);
const errors = [];
window.addEventListener('error', e => errors.push(e.message));
const check = (value, message) => { if (!value) throw Error(message); };
const same = (a, b, message) => check(JSON.stringify(a) === JSON.stringify(b), message + ': ' + JSON.stringify(a));
const charts = [];
const Plot = new Proxy(uPlot, {construct(target, args) {const u = new target(...args); charts.push(u); return u;}});
const raw = new TextEncoder().encode(`Title: named math
Plotname: AC Analysis
Flags: complex
No. Variables: 3
No. Points: 3
Variables:
0 frequency frequency
1 v(out) voltage
2 v(in) voltage
Values:
0 1,0
3,4
2,0
1 10,0
3,4
2,0
2 100,0
3,4
2,0
`);
let explorer;
const command = async (cmd, args = {}) => {
    const r = await explorer.run({cmd, args}); check(r.ok, JSON.stringify(r)); return r.results[0];
};
const state = () => command('getState');
const panelNames = async () => (await state()).plots['0'].panels.map(p => p.signals);
const displayed = () => charts.filter(u => u.root.isConnected).map(u => ({
    names: u.series.slice(1).map(s => s.label), values: u.data.slice(1).map(a => Array.from(a)),
}));
const original = {name: 'math1', operation: 'abs', operands: ['v(out)']};
const complex = {name: 'math1', operation: 'scale', operands: ['v(in)'], params: {gain: 2}, replace: true};
try {
    const payload = await core.decimate(core.plan(core.read(raw, {file_name: 'named.raw'})));
    const mount = async () => {
        explorer = createExplorer(root, {uPlot: Plot, waveformCore: core, stateKey: 'named-math'});
        await explorer.load(payload);
    };
    await mount();
    same(await command('deriveSignal', original), {name: 'math1', unit: 'V', created: true}, 'caller name');
    await command('addSignal', {signals: ['math1']});
    await command('addPanel', {signals: ['math1', 'v(in) (mag)']});
    check(root.querySelector('.panel .chip-name')?.textContent === 'math1' ||
        [...root.querySelectorAll('.panel [title]')].some(e => e.textContent === 'math1' && e.title === 'abs(v(out))'), 'legend name/tooltip');
    check([...root.querySelectorAll('.sig-name')].some(e => e.textContent === 'math1' && e.title === 'abs(v(out))'), 'monitor name/tooltip');
    await command('setFilter', {text: 'math1'});
    check([...root.querySelectorAll('[title]')].some(e => e.title === 'abs(v(out))' && e.closest('[data-silk="picker-list"]')), 'picker tooltip');
    const saved = await state();
    same(saved.plots['0'].derived[0].given_name, 'math1', 'persisted caller name');
    // Reload replays given_name; a fresh host derive is idempotent.
    await explorer.reload(payload);
    check(!(await command('deriveSignal', original)).created, 'reload duplicate');
    // Fresh instance restores the same name from the shared stateKey.
    explorer.destroy(); await mount();
    check(!(await command('deriveSignal', original)).created, 'stateKey duplicate');
    // Import uses the view-state path shared with .silkview.json files.
    await command('loadViewState', {state: saved});
    check(!(await command('deriveSignal', original)).created, 'import duplicate');
    await command('deriveSignal', complex);
    same(await panelNames(), [['math1 (mag)', 'math1 (phase)'], ['math1 (mag)', 'math1 (phase)', 'v(in) (mag)']], 'real to complex panels');
    check(displayed()[0].values[0].every(v => Math.abs(v - 4) < 1e-5), 'replacement chart samples');
    check(displayed()[0].values[1].every(v => Math.abs(v) < 1e-5), 'replacement phase');
    const identity = explorer.traceIdentity('math1 (phase)', 0);
    same([identity.given_name, identity.operation, identity.parameters], ['math1', 'scale', {gain: 2, offset: 0}], 'replacement identity');
    check(!(await command('deriveSignal', {...complex, operands: ['v(in) (phase)'], params: {offset: 0, gain: 2}})).created, 'normalized restored repeat');
    await command('removeSignal', {panel: 0, signal: 'math1 (mag)'});
    await command('deriveSignal', {...original, operands: ['v(in)'], replace: true});
    same(await panelNames(), [['math1'], ['math1', 'v(in) (mag)']], 'complex to real panels');
    check(displayed()[0].values[0].every(v => Math.abs(v - 2) < 1e-5), 'complex to real data');
    const final = await state();
    same(final.plots['0'].derived[0], {name: 'math1', operation: 'abs', operands: ['v(in)'], given_name: 'math1'}, 'updated persistence');
    // Reload into one sample (an ordinary one-point chart now): the named definition keeps its name and expression tooltip in the picker.
    const one = await core.decimate(core.plan(core.read(new TextEncoder().encode(new TextDecoder().decode(raw).replace('No. Points: 3', 'No. Points: 1').split('1 10,0')[0]), {file_name: 'named.raw'})));
    await explorer.reload(one);
    check([...root.querySelectorAll('[data-silk="picker-list"] .picker-row[data-name="math1"]')].some(e => e.title === 'abs(v(in))'), 'picker name/tooltip');
    await explorer.reload(payload);
    await command('deriveSignal', {name: 'math2', operation: 'scale', operands: ['math1'], params: {gain: 3}});
    await command('deriveSignal', {name: 'math3', operation: 'sum', operands: ['math2', 'v(in)']});
    await command('deriveSignal', {name: 'sibling', operation: 'abs', operands: ['v(out)']});
    await command('addPanel', {signals: ['math2', 'math3 (mag)', 'sibling']});
    const beforeFailure = await state(), beforeData = displayed();
    const beforeIdentity = explorer.traceIdentity('math3 (mag)');
    const failed = await explorer.run({cmd: 'deriveSignal', args: {
        name: 'math1', operation: 'derivative', operands: ['v(out)'], replace: true,
    }});
    check(!failed.ok && failed.error.includes('descendant "math3"') && failed.error.includes('unit'), 'descendant refusal diagnostic');
    same(await state(), beforeFailure, 'failed replace persisted state rollback');
    same(displayed(), beforeData, 'failed replace panel/data rollback');
    same(explorer.traceIdentity('math3 (mag)'), beforeIdentity, 'failed replace identity rollback');
    await command('deriveSignal', {...complex, params: {gain: 0, offset: 4}});
    same((await panelNames())[2], ['math2 (mag)', 'math2 (phase)', 'math3 (mag)', 'sibling'], 'descendant panel shape follows parent');
    const chain = displayed()[2];
    check(chain.values[0].every(v => Math.abs(v - 12) < 1e-5), 'direct descendant chart recomputed');
    check(chain.values[2].every(v => Math.abs(v - 14) < 1e-5), 'transitive descendant chart recomputed');
    same(chain.values[3], beforeData[2].values[2], 'unrelated sibling values untouched');
    const chainState = await state();
    await command('loadViewState', {state: {silkview: 1, view: chainState}});
    check(!(await command('deriveSignal', {...complex, params: {gain: 0, offset: 4}})).created, 'envelope restore repeat');
    const multi = {...payload, plots: [payload.plots[0], structuredClone(payload.plots[0])]};
    await explorer.reload(multi);
    await command('deriveSignal', {...original, plot: 1});
    await command('setPlot', {index: 1});
    await command('addSignal', {signals: ['math1']});
    await state(); // persist the now-visible plot before switching away
    await command('setPlot', {index: 0});
    await command('deriveSignal', {...complex, plot: 1});
    same((await state()).plots['1'].panels[0].signals, ['math1 (mag)', 'math1 (phase)'], 'inactive panel real-to-complex');
    await command('setPlot', {index: 1});
    check(displayed()[0].values[0].every(v => Math.abs(v - 4) < 1e-5), 'inactive panel replacement data');
    await state();
    await command('setPlot', {index: 0});
    await command('deriveSignal', {...original, plot: 1, replace: true});
    same((await state()).plots['1'].panels[0].signals, ['math1'], 'inactive panel complex-to-real');
    same(errors, [], 'browser errors');
    await fetch('/result', {method: 'POST', body: 'PASS'});
} catch (error) {
    await fetch('/result', {method: 'POST', body: error.message + '\n' + error.stack});
} finally { explorer?.destroy(); }
