// SPDX-License-Identifier: MIT
// Differential gate for the explorer's derive operations (TASK-374): the real
// Store.derive_signal against silk's Python Signal computation on the same
// payloads, dumped by tests/js/dump_derive_expected.py into the directory named
// by SILK_DERIVE_EXPECTED. tests/test_view_js_differential.py drives both.
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import {compileFunction} from 'node:vm';

const expected_dir = process.env.SILK_DERIVE_EXPECTED;
assert.ok(expected_dir, 'Set SILK_DERIVE_EXPECTED to the dump_derive_expected.py output directory');
const oracle = JSON.parse(await readFile(path.join(expected_dir, 'derive_expected.json'), 'utf8'));
const explorer_url = new URL('../../silk/signals/view/assets/explorer.mjs', import.meta.url);
const core = await import(new URL('../../silk/signals/view/assets/waveform-core.mjs', import.meta.url));
const source = await readFile(explorer_url, 'utf8');
const {traceIdentityOf, silkCode} = await import(explorer_url);

// Evaluate the actual private Store source without a DOM (same approach as
// explorer_union.test.mjs); a reorganised module fails loudly here.
function source_block(start, end) {
    assert.equal(source.split(start).length, 2, `Expected one source boundary: ${start}`);
    assert.equal(source.split(end).length, 2, `Expected one source boundary: ${end}`);
    const first = source.indexOf(start), last = source.indexOf(end);
    assert.ok(last > first, `Source boundaries out of order: ${start}, ${end}`);
    return source.slice(first, last);
}

const store_source = [
    source_block('  function bsearchLeft(', '  function levenshtein('),
    source_block('  function b64ToBuffer(', '  var PAYLOAD ='),
].join('\n');

// Returns the Store plus every (x, y) pair handed to build_pyramid, so the
// test reads the exact samples the explorer derived and checks the pyramid.
function load_store(payload) {
    const built = [];
    const waveformCore = {build_pyramid: (x, y) => {
        const pyramid = core.build_pyramid(x, y);
        built.push({x, y, pyramid});
        return pyramid;
    }};
    const toasts = [];
    const Store = compileFunction(store_source + '\nreturn Store;', ['options', 'showToast'],
        {filename: explorer_url.pathname})({waveformCore}, message => toasts.push(message));
    Store.init(payload);
    return {Store, built, toasts};
}

function f64(b64) {
    const bytes = Buffer.from(b64, 'base64');
    return new Float64Array(bytes.buffer, bytes.byteOffset, bytes.length / 8);
}

// Float tolerance: identical formulas in the same order are usually bit-exact;
// allow last-bit differences from fused or reordered arithmetic.
function assert_close(actual, want, label) {
    assert.equal(actual.length, want.length, label + ' length');
    let worst = 0;
    for (let i = 0; i < want.length; i++) {
        const a = actual[i], w = want[i];
        if (Number.isNaN(w) || Number.isNaN(a)) {
            assert.ok(Number.isNaN(w) && Number.isNaN(a), `${label}[${i}]: ${a} vs ${w}`);
            continue;
        }
        if (!Number.isFinite(w)) { assert.equal(a, w, `${label}[${i}]`); continue; }
        const error = Math.abs(a - w), bound = 1e-12 * Math.abs(w) + 1e-300;
        assert.ok(error <= bound, `${label}[${i}]: ${a} vs ${w} (error ${error})`);
        worst = Math.max(worst, error / Math.max(Math.abs(w), 1e-300));
    }
    return worst;
}

test('every single-trace result matches silk Signal maths', () => {
    const covered = new Set();
    for (const sample of oracle.cases) {
        const {Store, built} = load_store(oracle.payloads[sample.payload]);
        const label = `${sample.payload}: ${sample.operation}(${sample.signal}, ${JSON.stringify(sample.params)})`;
        const result = Store.derive_signal(sample.plot, sample.operation, [sample.signal], sample.params ?? undefined);
        assert.equal(result.created, true, label);
        assert.equal(built.length, 1, label + ' rebuilds one pyramid');
        assert_close(built[0].y, f64(sample.expected), label);
        // The derived trace is an ordinary Store signal with a fresh pyramid.
        const sig = Store.findSignal(sample.plot, result.name);
        assert.equal(sig.levels, built[0].pyramid.levels, label);
        assert.equal(sig.levels[0].n, built[0].y.length, label);
        covered.add(sample.operation + ':' + sample.payload);
    }
    for (const op of ['abs', 'scale', 'derivative', 'integral']) {
        for (const payload of ['e_ng_tran', 'uniform', 'nonuniform_nan', 'two_samples']) {
            assert.ok(covered.has(op + ':' + payload), `${op} on ${payload} is compared`);
        }
    }
});

test('the real SPICE grid is non-uniform', () => {
    const plot = oracle.payloads.e_ng_tran.plots[oracle.payloads.e_ng_tran.default_plot];
    const x = f64(plot.x.levels[0].x);
    const steps = new Set();
    for (let i = 1; i < x.length; i++) steps.add(x[i] - x[i - 1]);
    assert.ok(steps.size > 10, 'ngspice time steps vary');
});

test('refusals match Python and never leave an empty trace', () => {
    assert.ok(oracle.refusals.length >= 1);
    for (const refusal of oracle.refusals) {
        const {Store} = load_store(oracle.payloads[refusal.payload]);
        const before = Store.plot(refusal.plot).signals.length;
        assert.throws(() => Store.derive_signal(refusal.plot, refusal.operation, [refusal.signal]),
            /derivative: signal "v\(one\)" has 1 sample; derivative needs at least 2/);
        assert.equal(Store.plot(refusal.plot).signals.length, before);
    }
    const {Store} = load_store(oracle.payloads.one_sample);
    for (const op of ['abs', 'scale', 'integral']) {
        assert.equal(Store.derive_signal(0, op, ['v(one)']).created, true, op + ' of one sample is defined');
    }
});

test('the operation list is the exported silk list (export_signal_operations)', () => {
    const {Store} = load_store(oracle.payloads.uniform);
    const a = 'v(sin)', b = 'v(sin)';
    for (const op of oracle.operations) {
        const counts = op.inputs === '2+' ? {ok: [2, 3], bad: [0, 1]} : {ok: [op.inputs], bad: [op.inputs - 1, op.inputs + 1]};
        for (const n of counts.ok) {
            const result = Store.derive_signal(0, op.name, Array(n).fill(a));
            assert.ok(result.name.startsWith(op.name + '('), op.name);
        }
        for (const n of counts.bad) {
            assert.throws(() => Store.derive_signal(0, op.name, Array(n).fill(b)), /requires (exactly|at least)/, op.name + ' x' + n);
        }
        const keys = Object.keys(op.params);
        if (keys.length) {
            const meta = Store.derive_signal(0, op.name, [a], {});
            assert.deepEqual(Store.findSignal(0, meta.name).derived.params, op.params, op.name + ' defaults');
            assert.throws(() => Store.derive_signal(0, op.name, [a], {bogus: 1}), /unknown parameter "bogus"/);
            for (const key of keys) {
                assert.throws(() => Store.derive_signal(0, op.name, [a], {[key]: 'x'}), /finite number/);
                assert.throws(() => Store.derive_signal(0, op.name, [a], {[key]: Infinity}), /finite number/);
            }
        } else {
            assert.throws(() => Store.derive_signal(0, op.name, op.inputs === 1 ? [a] : [a, a], {gain: 2}),
                /takes no parameters/, op.name);
        }
    }
    assert.throws(() => Store.derive_signal(0, 'db', [a]), /operation must be/);
});

test('single-trace units, identity, chaining and persistence', () => {
    const {Store, built} = load_store(oracle.payloads.uniform);
    const v = 'v(sin)', i = 'i(ramp)';
    assert.equal(Store.derive_signal(0, 'abs', [v]).unit, 'V');
    assert.equal(Store.derive_signal(0, 'scale', [i], {gain: 1e6}).unit, 'A');
    const slew = Store.derive_signal(0, 'derivative', [v]);
    assert.equal(slew.unit, 'V/s');
    assert.equal(slew.name, 'derivative(v(sin))');
    const charge = Store.derive_signal(0, 'integral', [i]);
    assert.equal(charge.unit, 'C', 'integral of current over time is charge');
    // Chaining: the derived slew is an operand like any trace and recovers V.
    const back = Store.derive_signal(0, 'integral', [slew.name]);
    assert.equal(back.unit, 'V');
    const ratio = Store.derive_signal(0, 'quotient', [slew.name, v]);
    assert.equal(ratio.unit, 'Hz');
    assert.equal(Store.derive_signal(0, 'product', [charge.name, slew.name]).unit, 'W');

    // Parameters are part of identity; repeats return the existing trace.
    const scaled = Store.derive_signal(0, 'scale', [v], {gain: -1});
    assert.equal(scaled.name, 'scale(v(sin), gain=-1, offset=0)');
    assert.equal(Store.derive_signal(0, 'scale', [v], {gain: -1, offset: 0}).created, false);
    assert.equal(Store.derive_signal(0, 'scale', [v], {gain: -1, offset: 1}).created, true);
    assert.equal(Store.derive_signal(0, 'derivative', [v]).created, false);

    // Persisted definitions carry params only where they apply and restore.
    const saved = Store.derived_for_persist(0);
    assert.deepEqual(saved.find(d => d.name === scaled.name),
        {name: scaled.name, operation: 'scale', operands: [v], params: {gain: -1, offset: 0}});
    assert.equal('params' in saved.find(d => d.name === slew.name), false);
    const fresh = load_store(oracle.payloads.uniform);
    const plots = {'0': {derived: saved.slice().reverse()}};
    fresh.Store.restore_derived(plots);
    assert.deepEqual(fresh.toasts, []);
    assert.deepEqual(plots['0'].derived.map(d => d.name).sort(), saved.map(d => d.name).sort());
    const restored = fresh.Store.findSignal(0, scaled.name);
    const original = Store.findSignal(0, scaled.name);
    assert.deepEqual(Array.from(fresh.built.find(b => b.pyramid.levels === restored.levels).y),
        Array.from(built.find(b => b.pyramid.levels === original.levels).y));
});

// Polar reconstruction incurs absolute roundoff near cancellation. Compare
// against Python on reconstructed inputs at rtol=1e-10, atol=1e-12 (degrees too).
function complex_close(actual, want, label) {
    assert.equal(actual.length, want.length, label);
    for (let i = 0; i < want.length; i++) {
        assert.ok(Math.abs(actual[i] - want[i]) <= 1e-12 + 1e-10 * Math.abs(want[i]),
            `${label}[${i}]: ${actual[i]} vs ${want[i]}`);
    }
}

test('all AC operations use complex vectors and accept group or either part', () => {
    for (const sample of oracle.complex_cases) {
        for (const suffix of ['', ' (mag)', ' (phase)']) {
            const {Store, built} = load_store(oracle.payloads.ac);
            const result = Store.derive_signal(0, sample.operation,
                sample.operands.map(n => n + suffix), sample.params);
            const real = sample.operation === 'abs';
            assert.equal(built.length, real ? 1 : 2);
            const mag = Store.findSignal(0, result.name + (real ? '' : ' (mag)'));
            assert.ok(mag);
            complex_close(built[0].y, f64(sample.encoded_mag), sample.operation + ' magnitude');
            if (!real) {
                const phase = Store.findSignal(0, result.name + ' (phase)');
                assert.equal(mag.group, result.name);
                assert.equal(phase.group, result.name);
                assert.equal(phase.unit, 'deg');
                complex_close(built[1].y, f64(sample.encoded_phase), sample.operation + ' phase');
            }
            assert.equal(Store.derive_signal(0, sample.operation, sample.operands, sample.params).created, false);
        }
    }
});

test('AC units, chaining, one persisted definition per pair and shuffled restore', () => {
    const {Store} = load_store(oracle.payloads.ac);
    const sum = Store.derive_signal(0, 'sum', ['v(in)', 'v(out) (phase)']);
    assert.equal(sum.unit, 'V');
    const slew = Store.derive_signal(0, 'derivative', [sum.name]);
    assert.equal(slew.unit, 'V/Hz');
    const back = Store.derive_signal(0, 'integral', [slew.name + ' (phase)']);
    assert.equal(back.unit, 'V');
    const ratio = Store.derive_signal(0, 'quotient', [sum.name, 'v(in)']);
    assert.equal(ratio.unit, '');
    const abs = Store.derive_signal(0, 'abs', [ratio.name]);
    const scaledAbs = Store.derive_signal(0, 'scale', [abs.name], {offset: -2});
    assert.ok(Store.findSignal(0, scaledAbs.name), 'real abs output stays real when chained');
    const saved = Store.derived_for_persist(0);
    assert.equal(saved.length, 6);
    const fresh = load_store(oracle.payloads.ac);
    fresh.Store.restore_derived({'0': {derived: saved.slice().reverse()}});
    assert.deepEqual(fresh.toasts, []);
    for (const sig of Store.plot(0).signals) {
        const restored = fresh.Store.findSignal(0, sig.name);
        assert.ok(restored, sig.name);
        complex_close(fresh.Store.rawYLevel0(0, restored), Store.rawYLevel0(0, sig), sig.name);
    }
});

test('AC rawfile accuracy and exported Python expressions', async () => {
    const exports = [];
    let worstRelativeMagnitude = 0, worstPhaseDegrees = 0;
    for (const sample of oracle.complex_cases) {
        const {Store} = load_store(oracle.payloads.ac);
        const result = Store.derive_signal(0, sample.operation, sample.operands, sample.params);
        const payload = {run: oracle.payloads.ac.run, plots: [Store.plot(0)]};
        for (const part of sample.operation === 'abs' ? ['mag'] : ['mag', 'phase']) {
            const name = result.name + (sample.operation === 'abs' ? '' : ` (${part})`);
            const values = Store.rawYLevel0(0, Store.findSignal(0, name));
            const want = f64(sample[part]);
            for (let i = 0; i < want.length; i++) {
                const error = Math.abs(values[i] - want[i]);
                if (part === 'mag') {
                    worstRelativeMagnitude = Math.max(worstRelativeMagnitude, error / Math.max(1e-300, Math.abs(want[i])));
                    assert.ok(error <= 1e-10 + 1e-4 * Math.abs(want[i]), `${sample.operation} mag[${i}]: ${values[i]} vs ${want[i]}`);
                } else {
                    worstPhaseDegrees = Math.max(worstPhaseDegrees, error);
                    assert.ok(error <= (sample.operation === 'derivative' ? 0.2 : 1e-3), `${sample.operation} phase[${i}]: ${values[i]} vs ${want[i]}`);
                }
            }
            const code = silkCode(traceIdentityOf(payload, 0, name));
            assert.equal(code.ok, true, code.reason);
            exports.push({code: code.code, expected: sample[part]});
        }
        const whole = silkCode(traceIdentityOf(payload, 0, result.name));
        assert.equal(whole.ok, true, whole.reason);
        exports.push({code: whole.code, re: sample.re, im: sample.im});
    }
    console.log('AC float32 payload/display error:', {worstRelativeMagnitude, worstPhaseDegrees});
    await writeFile(path.join(expected_dir, 'derive_exports.json'), JSON.stringify(exports));
});

test('AC restore preserves renamed groups, dependents and identity reuse', () => {
    const {Store, toasts} = load_store(oracle.payloads.ac);
    const saved = {'0': {derived: [
        {name: 'later', operation: 'abs', operands: ['renamed (phase)']},
        {name: 'renamed', operation: 'sum', operands: ['v(in)', 'v(out)']},
    ]}};
    Store.restore_derived(saved);
    assert.deepEqual(toasts, []);
    assert.ok(Store.findSignal(0, 'renamed (mag)'));
    assert.ok(Store.findSignal(0, 'renamed (phase)'));
    assert.ok(Store.findSignal(0, 'later'));
    assert.equal(Store.derive_signal(0, 'sum', ['v(in)', 'v(out)']).name, 'renamed');
    const payload = {plots: [Store.plot(0)]};
    assert.equal(silkCode(traceIdentityOf(payload, 0, 'later')).code,
        "(tb.probe('in') + tb.probe('out')).mag()");
});

test('AC cancellation is complex, repeated operands survive, and mixed real operands promote', () => {
    const {Store, built} = load_store(oracle.payloads.ac);
    const zero = Store.derive_signal(0, 'difference', ['v(out) (mag)', 'v(out) (phase)']);
    assert.ok(built[0].y.every(v => v === 0), zero.name);
    const triple = Store.derive_signal(0, 'sum', ['v(out)', 'v(out) (mag)', 'v(out) (phase)']);
    const scale = Store.derive_signal(0, 'scale', ['v(out)'], {gain: 3});
    complex_close(Store.rawYLevel0(0, Store.findSignal(0, triple.name + ' (mag)')),
        Store.rawYLevel0(0, Store.findSignal(0, scale.name + ' (mag)')), 'repeated operands');
    const abs = Store.derive_signal(0, 'abs', ['v(out)']);
    const mixed = Store.derive_signal(0, 'sum', [abs.name, 'v(out)']);
    assert.ok(Store.findSignal(0, mixed.name + ' (phase)'));
    const payload = {plots: [Store.plot(0)]};
    assert.equal(silkCode(traceIdentityOf(payload, 0, mixed.name)).code,
        "tb.probe('out').mag() + tb.probe('out')");
});

test('restoring a renamed group does not rename another plot identity', () => {
    const payload = structuredClone(oracle.payloads.ac);
    payload.plots.push(structuredClone(payload.plots[0]));
    const {Store, toasts} = load_store(payload);
    const original = Store.derive_signal(0, 'sum', ['v(in)', 'v(out)']);
    Store.restore_derived({'1': {derived: [
        {name: 'renamed', operation: 'sum', operands: ['v(in)', 'v(out)']},
    ]}});
    assert.deepEqual(toasts, []);
    assert.equal(Store.derive_signal(0, 'sum', ['v(in)', 'v(out)']).name, original.name);
    assert.equal(Store.derive_signal(1, 'sum', ['v(in)', 'v(out)']).name, 'renamed');
});

test('a quotient singularity does not erase valid samples in chained complex math', () => {
    const payload = structuredClone(oracle.payloads.ac), p = payload.plots[0];
    p.npoints = 3;
    p.x.levels = [{stride: 1, n: 3, dt: 'f64', x: new Float64Array([1, 2, 3])}];
    p.signals = ['a', 'b'].flatMap((name, index) => ['mag', 'phase'].map(part => ({
        name: `${name} (${part})`, group: name, part, unit: part === 'mag' ? 'V' : 'deg',
        levels: [{stride: 1, n: 3, dt: 'f64', y: new Float64Array(
            part === 'phase' ? [0, 0, 0] : index ? [0, 1, 1] : [0, 1, 2])}],
    })));
    const {Store} = load_store(payload);
    const quotient = Store.derive_signal(0, 'quotient', ['a', 'b']);
    const abs = Store.derive_signal(0, 'abs', [quotient.name + ' (phase)']);
    assert.deepEqual(Array.from(Store.rawYLevel0(0, Store.findSignal(0, abs.name))), [NaN, 1, 2]);
    const scale = Store.derive_signal(0, 'scale', [quotient.name], {gain: 2});
    assert.deepEqual(Array.from(Store.rawYLevel0(0, Store.findSignal(0, scale.name + ' (mag)'))), [NaN, 2, 4]);
    const fresh = load_store(payload);
    fresh.Store.restore_derived({'0': {derived: Store.derived_for_persist(0).reverse()}});
    assert.deepEqual(fresh.toasts, []);
    assert.deepEqual(Array.from(fresh.Store.rawYLevel0(0, fresh.Store.findSignal(0, abs.name))), [NaN, 1, 2]);
});

test('caller names preserve identities and normalized repeat semantics through restore', () => {
    const {Store} = load_store(oracle.payloads.ac);
    const first = Store.derive_signal(0, 'sum', ['v(out) (phase)', 'v(in)'], undefined, 'math1');
    assert.deepEqual(first, {name: 'math1', unit: 'V', created: true});
    assert.equal(Store.derive_signal(0, 'sum', ['v(in) (mag)', 'v(out)'], {}, 'math1').created, false);
    assert.equal(Store.derive_signal(0, 'sum', ['v(in)', 'v(out)'], undefined, 'math2').created, true);
    const identity = traceIdentityOf({plots: [Store.plot(0)]}, 0, 'math1 (phase)');
    assert.equal(identity.given_name, 'math1');
    assert.equal(identity.operation, 'sum');
    assert.equal(identity.part, 'phase');
    assert.equal(identity.operands.length, 2);
    assert.equal(silkCode(identity).ok, true);
    const saved = {'0': {derived: Store.derived_for_persist(0)}};
    assert.equal(saved['0'].derived[0].given_name, 'math1');
    const restored = load_store(oracle.payloads.ac);
    restored.Store.restore_derived(saved);
    assert.deepEqual(restored.toasts, []);
    assert.equal(restored.Store.derive_signal(0, 'sum', ['v(out)', 'v(in)'], undefined, 'math1').created, false);
    assert.equal(restored.Store.derive_signal(0, 'difference', ['v(out)', 'v(in)'], undefined, 'math1', true).created, true);
    assert.equal(restored.Store.derived_for_persist(0)[0].operation, 'difference');
});

test('named scale defaults deduplicate and unnamed dedup remains independent', () => {
    const {Store} = load_store(oracle.payloads.ac);
    assert.equal(Store.derive_signal(0, 'scale', ['v(out)'], {gain: 2}, ' literal.name ').name, ' literal.name ');
    assert.equal(Store.derive_signal(0, 'scale', ['v(out) (mag)'], {offset: 0, gain: 2}, ' literal.name ').created, false);
    assert.equal(Store.derive_signal(0, 'scale', ['v(out)'], {gain: 2}).created, true);
    assert.equal(Store.derive_signal(0, 'scale', ['v(out)'], {gain: 2, offset: 0}).created, false);
});

test('explicit name collisions protect raw, unnamed and other complex parts', () => {
    const {Store} = load_store(oracle.payloads.ac);
    for (const name of [null, '', '  ', 4, {}]) {
        assert.throws(() => Store.derive_signal(0, 'abs', ['v(out)'], undefined, name), /name must be a non-empty string/);
    }
    for (const replace of [null, 1, 'true', {}]) {
        assert.throws(() => Store.derive_signal(0, 'abs', ['v(out)'], undefined, 'math', replace), /replace must be a boolean/);
    }
    const unnamed = Store.derive_signal(0, 'abs', ['v(out)']).name;
    Store.derive_signal(0, 'sum', ['v(in)', 'v(out)'], undefined, 'math');
    for (const name of ['v(out)', 'v(out) (mag)', unnamed, 'math (phase)']) {
        for (const replace of [false, true]) {
            assert.throws(() => Store.derive_signal(0, 'abs', ['v(in)'], undefined, name, replace),
                error => error.message.includes(JSON.stringify(name)) && /conflicts with existing signal or trace/.test(error.message));
        }
    }
    assert.throws(() => Store.derive_signal(0, 'difference', ['v(in)', 'v(out)'], undefined, 'math'), /conflicts/);
});

test('replacement changes cached samples, shape and persistence without duplicate rows', () => {
    const {Store} = load_store(oracle.payloads.ac);
    Store.derive_signal(0, 'abs', ['v(out)'], undefined, 'math', true);
    const before = Array.from(Store.rawYLevel0(0, Store.findSignal(0, 'math')));
    Store.derive_signal(0, 'scale', ['v(in)'], {gain: 2}, 'math', true);
    assert.equal(Store.findSignal(0, 'math'), null);
    assert.ok(Store.findSignal(0, 'math (phase)'));
    Store.rawYLevel0(0, Store.findSignal(0, 'math (mag)'));
    Store.derive_signal(0, 'abs', ['v(in)'], undefined, 'math', true);
    assert.equal(Store.findSignal(0, 'math (mag)'), null);
    assert.equal(Store.findSignal(0, 'math (phase)'), null);
    assert.notDeepEqual(Array.from(Store.rawYLevel0(0, Store.findSignal(0, 'math'))), before);
    assert.equal(Store.derived_for_persist(0).length, 1);
    assert.deepEqual(Store.derived_for_persist(0)[0], {name: 'math', given_name: 'math', operation: 'abs', operands: ['v(in)']});
    const unchanged = Store.derived_for_persist(0);
    assert.throws(() => Store.derive_signal(0, 'scale', ['missing'], {}, 'math', true), /unknown signal/);
    assert.deepEqual(Store.derived_for_persist(0), unchanged);
});

test('given_name origin tripwire protects replace safety', () => {
    const message = 'A new naming path (UI rename, Copy-as-code, an import) must revisit replace safety first';
    const deriveStart = source.indexOf('    function derive_signal(');
    const deriveEnd = source.indexOf('    function derived_for_persist(');
    const restore = source_block('    function restore_derived(', '    // Nearest level-0 sample');
    // Runtime definitions use `definition`; record/def are serialized output copies.
    const writes = [...source.matchAll(/\b(\w+)\.given_name\s*=(?!=)/g)];
    assert.ok(writes.some(m => m[1] === 'definition'), message);
    for (const match of writes) {
        if (['record', 'def'].includes(match[1])) continue;
        assert.ok(match.index >= deriveStart && match.index < deriveEnd, message);
    }
    assert.doesNotMatch(source, /(?:given_name\s*:|\[['"]given_name['"]\]\s*=)/, message);
    assert.match(restore, /derive_signal\(plotIdx, def\.operation, def\.operands, def\.params, def\.given_name\)/, message);
});

test('replace recomputes a two-level chain and unnamed descendants, leaving siblings untouched', () => {
    const {Store} = load_store(oracle.payloads.ac);
    Store.derive_signal(0, 'abs', ['v(out)'], undefined, 'math1');
    Store.derive_signal(0, 'scale', ['math1'], {gain: 3}, 'math2');
    Store.derive_signal(0, 'abs', ['math2'], undefined, 'math3');
    const unnamed = Store.derive_signal(0, 'scale', ['math3'], {gain: 2}).name;
    Store.derive_signal(0, 'abs', ['v(in)'], undefined, 'sibling');
    const sibling = Store.findSignal(0, 'sibling');
    const siblingData = Store.rawYLevel0(0, sibling);
    Store.derive_signal(0, 'scale', ['v(out)'], {gain: 0, offset: 2}, 'math1', true);
    for (const [name, value] of [['math1 (mag)', 2], ['math2 (mag)', 6], ['math3', 6], [unnamed, 12]]) {
        const values = Store.rawYLevel0(0, Store.findSignal(0, name));
        assert.ok(values.every(v => v === value), name + ' must use the new ancestor data');
    }
    assert.equal(Store.findSignal(0, 'math2'), null);
    assert.equal(Store.findSignal(0, 'sibling'), sibling, 'unaffected sibling is not rebuilt');
    assert.deepEqual(Store.rawYLevel0(0, sibling), siblingData);
    assert.equal(Store.derive_signal(0, 'scale', ['math1'], {gain: 3}, 'math2').created, false);
    assert.equal(Store.derive_signal(0, 'scale', ['math3'], {gain: 2}).created, false);
    const identity = traceIdentityOf({plots: [Store.plot(0)]}, 0, 'math3');
    assert.equal(identity.operands[0].operands[0].parameters.offset, 2);
    assert.equal(Store.derived_for_persist(0).find(d => d.name === unnamed).given_name, undefined);
});

test('invalid transitive descendant rolls back signals, identity, definitions and decoded caches', () => {
    const {Store} = load_store(oracle.payloads.ac);
    Store.derive_signal(0, 'abs', ['v(out)'], undefined, 'math1');
    Store.derive_signal(0, 'scale', ['math1'], {gain: 3}, 'math2');
    const ref = Store.derive_signal(0, 'abs', ['v(in)']).name;
    Store.derive_signal(0, 'sum', ['math2', ref], undefined, 'math3');
    const before = Store.plot(0).signals.slice();
    const defs = Store.derived_for_persist(0);
    const identity = traceIdentityOf({plots: [Store.plot(0)]}, 0, 'math3');
    const samples = before.map(sig => Array.from(Store.rawYLevel0(0, sig)));
    assert.throws(() => Store.derive_signal(0, 'derivative', ['v(out)'], undefined, 'math1', true),
        error => /descendant "math3"/.test(error.message) && /unit/.test(error.message));
    assert.deepEqual(Store.derived_for_persist(0), defs);
    assert.deepEqual(traceIdentityOf({plots: [Store.plot(0)]}, 0, 'math3'), identity);
    assert.deepEqual(Store.plot(0).signals, before);
    assert.deepEqual(before.map(sig => Array.from(Store.rawYLevel0(0, sig))), samples);
    assert.equal(Store.derive_signal(0, 'abs', ['v(out)'], undefined, 'math1').created, false);
    // A later valid transaction still works after rollback.
    Store.derive_signal(0, 'scale', [ref], {gain: 0}, 'math1', true);
    assert.ok(Store.rawYLevel0(0, Store.findSignal(0, 'math2')).every(v => v === 0));
});

test('descendant shape collision rolls back instead of overwriting an unrelated trace', () => {
    const {Store} = load_store(oracle.payloads.ac);
    Store.derive_signal(0, 'abs', ['v(out)'], undefined, 'math1');
    const dependent = Store.derive_signal(0, 'scale', ['math1'], {gain: 3}).name;
    Store.derive_signal(0, 'abs', ['v(in)'], undefined, dependent + ' (mag)');
    const before = Store.derived_for_persist(0);
    assert.throws(() => Store.derive_signal(0, 'scale', ['v(out)'], {gain: 0}, 'math1', true),
        error => error.message.includes('descendant ' + JSON.stringify(dependent)) && /conflict/.test(error.message));
    assert.deepEqual(Store.derived_for_persist(0), before);
});

test('an explicitly named real definition reserves both future complex-part names', () => {
    const {Store} = load_store(oracle.payloads.ac);
    Store.derive_signal(0, 'abs', ['v(out)'], undefined, 'math');
    for (const name of ['math (mag)', 'math (phase)']) {
        assert.throws(() => Store.derive_signal(0, 'abs', ['v(in)'], undefined, name, true),
            error => error.message.includes(JSON.stringify(name)) && error.message.includes('"math"') && /conflicts/.test(error.message));
    }
});

test('waveform payloads cannot import a caller-name definition outside derive_signal', () => {
    const payload = structuredClone(oracle.payloads.ac);
    const raw = payload.plots[0].signals[0];
    raw.derived = {given_name: raw.group || raw.name, operation: 'abs', operands: ['v(in)'], params: null};
    assert.throws(() => load_store(payload), /must not supply given_name.*deriveSignal/);
});

test('batch panel shape updates preserve unchanged unit axes on parents and descendants', () => {
    const names = [
        {name: 'math1 (mag)', unit: 'V', part: 'mag'},
        {name: 'math1 (phase)', unit: 'deg', part: 'phase'},
        {name: 'math2 (mag)', unit: 'V/Hz', part: 'mag'},
        {name: 'math2 (phase)', unit: 'deg', part: 'phase'},
    ];
    const Store = {findSignal: (_, name) => names.find(s => s.name === name)};
    const axesSource = source_block('  var AxisModes = (function () {', '  var PanelManager = (function () {');
    const axes = compileFunction(axesSource + '\nreturn AxisModes;', ['Store', 'isFiniteNum'])(Store, Number.isFinite);
    const panelSource = source_block('    function replaceDerived(', '    function refreshValues(');
    const changes = [
        {before: ['math1'], after: ['math1 (mag)', 'math1 (phase)']},
        {before: ['math2'], after: ['math2 (mag)', 'math2 (phase)']},
    ];
    const initial = {plotIdx: 0, signals: ['math1', 'math2'], yAxes: [
        {unit: 'V', signals: ['math1'], yMode: 'linear', yRange: null},
        {unit: 'V/Hz', signals: ['math2'], yMode: 'log', yRange: [1, 100]},
    ]};
    const mounted = structuredClone(initial), saved = structuredClone(initial);
    const replace = compileFunction(panelSource + '\nreturn replaceDerived;',
        ['panelsForPlot', 'StateStore', 'AxisModes', 'renderPanelHeader', 'buildUplot', 'updateMonitorTable'])(
        () => [mounted], {get: () => ({plots: {'0': {panels: [saved]}}})}, axes, axes.sync_axes, () => {}, () => {});
    replace(0, changes);
    for (const panel of [mounted, saved]) {
        assert.deepEqual(panel.signals, names.map(s => s.name));
        const axis = panel.yAxes.find(a => a.unit === 'V/Hz');
        assert.equal(axis.yMode, 'log');
        assert.deepEqual(axis.yRange, [1, 100]);
    }
});
