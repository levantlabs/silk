// SPDX-License-Identifier: MIT
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {decimate, plan, read} from '../../silk/signals/view/assets/waveform-core.mjs';
import {compare, normalise} from './differential.mjs';

const fixture = await readFile(new URL('../fixtures/rawfiles/f_nonascii_title.raw', import.meta.url));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const encode = text => new TextEncoder().encode(text);

function ascii(values, {plotname = 'Transient Analysis', type = 'voltage', extra = '', count = values.length} = {}) {
    return encode(`Title: test\nPlotname: ${plotname}\nFlags: real\nNo. Variables: 2\nNo. Points: ${count}\n${extra}Variables:\n\t0\ttime\ttime\n\t1\tv(out)\t${type}\nValues:\n` +
        values.map((value, i) => `${i}\t${i}\n\t${value}\n`).join(''));
}

test('read accepts buffers and offset views, snapshots bytes, and decodes real latin-1', () => {
    const surrounded = new Uint8Array(fixture.length + 7).fill(65);
    surrounded.set(fixture, 3);
    const input = surrounded.subarray(3, fixture.length + 3);
    const raw = read(input, {file_name: 'µ.raw'});
    const from_buffer = read(input.slice().buffer, {file_name: 'µ.raw'});
    assert.deepEqual(raw, from_buffer);
    assert.equal(raw.plots[0].title, 'UTF-8 micro \xc2\xb5; lone high byte \xe9; latin-1 \x80\x9f');
    assert.ok(Object.is(raw.plots[0].vectors[0].data[0], -0));
    input.fill(0);
    assert.deepEqual(raw.bytes, new Uint8Array(fixture));
    assert.throws(() => read('raw text'), TypeError);
    assert.throws(() => read(fixture, {file_name: 3}), TypeError);
});

test('rawfile failures are actionable; footers and incomplete data keep complete points', () => {
    for (const [source, message] of [
        ['', /no plot header found/],
        ['Title: broken\nPlotname: Transient\n', /data section/],
        ['Plotname: Transient\ngarbage\n', /Keyword: value/],
        ['Variables:\n', /No. Variables/],
        ['No. Variables: 2\nVariables:\n0 time time\n', /only 1/],
    ]) assert.throws(() => read(encode(source)), error => error.name === 'RawfileError' && message.test(error.message));
    assert.throws(() => read(ascii(['bad'])), /invalid numeric value/);
    assert.equal(read(ascii([1, 2], {count: 1000})).plots[0].npoints, 2);
    const partial_binary = fixture.subarray(0, fixture.length - 7);
    assert.equal(read(partial_binary).plots[0].npoints, 2);
    const footer = Buffer.concat([fixture, Buffer.from('\nEnd of simulator\nVersion: fixture\n')]);
    assert.equal(read(footer).plots.length, 1);
    const crlf = new TextDecoder().decode(ascii([1, 2])).replace('Variables:\n\t0', 'Variables: 0').replaceAll('\n', '\r\n');
    assert.equal(read(encode(crlf)).plots[0].vectors[1].data[1], 2);
});

test('run identity hashes exactly bytes plus resolved parameters; state is filename-based', async () => {
    const payload = await decimate(plan(read(fixture, {file_name: 'µ.raw'})));
    const expected_id = hash(Buffer.concat([fixture, Buffer.from('\0[2,8000,8,1000]')]));
    assert.equal(payload.run.id, expected_id.slice(0, 16));
    assert.equal(payload.run.state_key, hash('µ.raw').slice(0, 8));
    const same = await decimate(plan(read(fixture, {file_name: 'µ.raw'})), {min_bucket: 1000, base: 8, min_points: 8000});
    assert.equal(same.run.id, payload.run.id);
    const renamed = await decimate(plan(read(fixture, {file_name: 'new.raw'})));
    assert.equal(renamed.run.id, payload.run.id);
    assert.notEqual(renamed.run.state_key, payload.run.state_key);
    const no_name = await decimate(plan(read(fixture)));
    assert.equal(no_name.run.state_key, hash('<in-memory>').slice(0, 8));
    for (const options of [{base: 2}, {min_points: 0}, {min_bucket: 1}]) {
        const changed = await decimate(plan(read(fixture, {file_name: 'µ.raw'})), options);
        assert.notEqual(changed.run.id, payload.run.id);
        assert.equal(changed.run.state_key, payload.run.state_key);
    }
    const bytes = Buffer.from(fixture);
    bytes[8] ^= 1;
    const changed = await decimate(plan(read(bytes, {file_name: 'µ.raw'})));
    assert.notEqual(changed.run.id, payload.run.id);
    assert.equal(changed.run.state_key, payload.run.state_key);
});

test('invalid decimation settings fail before producing a payload', async () => {
    const pp = plan(read(fixture));
    for (const options of [{base: 0}, {base: 1}, {min_points: -1}, {min_bucket: 0},
        {base: 2.5}, {base: Infinity}, {min_points: NaN}, {min_bucket: '2'}]) {
        await assert.rejects(decimate(pp, options), RangeError);
    }
    await assert.rejects(decimate(pp, {signals: ['v(out)']}), /unknown decimation option/);
    assert.throws(() => plan({plots: []}), error => error.name === 'ViewError');
});

test('reference min/max emits first-index extrema in sample order, ties, gaps, and ragged tails', async () => {
    const input = ascii([3, 1, 2, 4, 'nan', 'nan', -2, -2, '-0']);
    const payload = await decimate(plan(read(input)), {min_points: 0, base: 2, min_bucket: 1});
    const levels = payload.plots[0].signals[0].levels;
    assert.deepEqual(levels.map(level => level.stride), [1, 2, 4, 8, 16, 32, 64, 128]);
    assert.deepEqual(levels[1].x, new Float64Array([0, 1, 2, 3, 4, 6, 6, 8, 8]));
    compare(levels[1].y, new Float32Array([3, 1, 2, 4, NaN, -2, -2, -0, -0]));
    assert.equal(payload.plots[0].signals[0].nonfinite, 2);
    // The pinned numpy reference treats NaNs as +/-Inf for each arg reduction.
    const infinities = await decimate(plan(read(ascii(['nan', 'inf', 'nan', '-inf']))),
        {min_points: 0, base: 2, min_bucket: 1});
    compare(infinities.plots[0].signals[0].levels[1].y, new Float32Array([NaN, Infinity, NaN, -Infinity]));
});

test('float32 range escape hatch is based on the full precision data', async () => {
    for (const [values, expected] of [
        [['-0', 0, 'nan', 'inf', '-inf'], 'f32'],
        [[3e38, 1e-37], 'f32'], [[3.000000000000001e38], 'f64'],
        [[0.9999999999999999e-37], 'f64'], [[1e-310], 'f64'],
    ]) {
        const result = await decimate(plan(read(ascii(values))));
        const level = result.plots[0].signals.find(s => s.name === 'v(out)').levels[0];
        assert.equal(level.dt, expected);
        assert.equal(level.y.constructor.name, expected === 'f32' ? 'Float32Array' : 'Float64Array');
    }
});

test('default stride ladder matches the documented million-point example', async () => {
    for (const [n, expected] of [[8000, [1]], [8001, [1, 8]], [16000, [1, 8]], [1_000_000, [1, 8, 64, 512]]]) {
        const header = Buffer.from(`Plotname: Transient\nFlags: real\nNo. Variables: 2\nNo. Points: ${n}\nVariables:\n0 time time\n1 v(out) voltage\nBinary:\n`);
        const payload = await decimate(plan(read(Buffer.concat([header, Buffer.alloc(n * 16)]))));
        assert.deepEqual(payload.plots[0].signals[0].levels.map(level => level.stride), expected);
    }
});

test('comparison gate distinguishes signed zero and dtype but equates NaN', () => {
    compare(new Float64Array([NaN, -0]), new Float64Array([NaN, -0]));
    assert.throws(() => compare(new Float64Array([0]), new Float64Array([-0])), /0 !== -0/);
    assert.throws(() => compare(new Float32Array([0]), new Float64Array([0])), /dtype/);
    assert.throws(() => compare(new Float64Array([1]), new Float64Array([2])), /1 !== 2/);
    const negative_zero = Buffer.alloc(8);
    negative_zero.writeDoubleLE(-0);
    compare(normalise({stride: 1, n: 1, dt: 'f64', y: negative_zero.toString('base64')}).y,
        new Float64Array([-0]));
});

test('comparison excludes exactly four named values and detects missing keys and other drift', async () => {
    const original = await decimate(plan(read(fixture)));
    for (const key of ['id', 'emitted_at', 'source']) {
        const changed = structuredClone(original);
        changed.run[key] = 'different';
        compare(changed, original);
        delete changed.run[key];
        assert.throws(() => compare(changed, original), /keys/);
    }
    const version = structuredClone(original);
    version.generator_version = 'different';
    compare(version, original);
    for (const key of ['state_key', 'title']) {
        const changed = structuredClone(original);
        changed.run[key] = 'different';
        assert.throws(() => compare(changed, original), new RegExp(key));
    }
    const nested = structuredClone(original);
    nested.plots[0].generator_version = 'different';
    assert.throws(() => compare(nested, original), /keys/);
    const selection = structuredClone(original);
    selection.plots[0].signals = [];
    assert.throws(() => compare(selection, original), /keys/);
});


test('per-array pyramid preserves extrema, gaps, dtype and nonfinite counts', async () => {
    const core = await import('../../silk/signals/view/assets/waveform-core.mjs');
    assert.equal(typeof core.build_pyramid, 'function');
    const x = Float64Array.from({length: 9}, (_, i) => i);
    const result = core.build_pyramid(x, new Float64Array([3, 1, 2, 4, NaN, NaN, -2, -2, -0]),
        {min_points: 0, base: 2, min_bucket: 1});
    assert.equal(result.nonfinite, 2);
    assert.deepEqual(result.levels.map(l => l.stride), [1, 2, 4, 8, 16, 32, 64, 128]);
    compare(result.levels[1].x, new Float64Array([0, 1, 2, 3, 4, 6, 6, 8, 8]));
    compare(result.levels[1].y, new Float32Array([3, 1, 2, 4, NaN, -2, -2, -0, -0]));
    assert.equal(result.levels[0].x, undefined);
    for (const values of [[1e40, NaN], [1e-40, Infinity]]) {
        const tiny = core.build_pyramid(new Float64Array([0, 1]), new Float64Array(values));
        assert.equal(tiny.levels[0].dt, 'f64');
        assert.equal(tiny.nonfinite, 1);
    }
    const subnormal = core.build_pyramid(new Float64Array([0]), new Float32Array([1e-40]));
    assert.equal(subnormal.levels[0].dt, 'f64');
    assert.ok(subnormal.levels[0].y instanceof Float64Array);
    assert.throws(() => core.build_pyramid(x, new Float64Array(2)), /length/);
    assert.throws(() => core.build_pyramid(x, x, {base: 1}), /base/);
});

test('format two identity retains label and address suffix distinctions', async () => {
    const bytes = ascii([2, 3]);
    const plain = await decimate(plan(read(bytes)));
    const empty = await decimate(plan(read(bytes)), {labels: {}});
    const designerPlain = await decimate(plan(read(bytes)), {labels: 'designer'});
    const expected = hash(Buffer.concat([Buffer.from(bytes), Buffer.from('\0[2,8000,8,1000]')])).slice(0,16);
    assert.equal(plain.run.id, expected);
    assert.equal(empty.run.id, expected);
    assert.equal(designerPlain.run.id, expected);
    const arrayBytes = encode(new TextDecoder().decode(bytes).replace('v(out)', 'v(xx1<0>.mid)'));
    const labelled = await decimate(plan(read(arrayBytes)), {labels: {'v(xx1<0>.mid)':'v(xx1[0].mid)'}});
    const addressed = await decimate(plan(read(arrayBytes)), {labels:'designer'});
    assert.equal(labelled.plots[0].signals[0].label, addressed.plots[0].signals[0].label);
    assert.equal(labelled.plots[0].signals[0].address, undefined);
    assert.equal(addressed.plots[0].signals[0].address, 'v(xx1[0].mid)');
    assert.notEqual(labelled.run.id, addressed.run.id);
    const triples = '[[0,"v(xx1<0>.mid)","v(xx1[0].mid)"]]';
    assert.equal(addressed.run.id, hash(Buffer.concat([Buffer.from(arrayBytes),
        Buffer.from('\0[2,8000,8,1000]\0labels'+triples+'\0addresses'+triples)])).slice(0,16));
    assert.equal(addressed.run.state_key, labelled.run.state_key);
});


test('one-point analyses keep their axis and vector-zero designer address use original rows', async () => {
    const dc = encode(new TextDecoder().decode(ascii([3], {plotname:'DC transfer characteristic'}))
        .replace('time\ttime', 'v(v-sweep)\tvoltage'));
    const p = await decimate(plan(read(dc)));
    // TASK-591: a one-point sweep keeps its real axis; only a true OP has none.
    assert.equal(p.plots[0].kind, 'dc');
    assert.equal(p.plots[0].x.name, 'v(v-sweep)');
    assert.equal(p.plots[0].signals[0].role, undefined);
    const op = encode(new TextDecoder().decode(ascii([3], {plotname:'Operating Point'}))
        .replace('time\ttime', 'v(xx1<0>.in)\tvoltage'));
    const q = await decimate(plan(read(op)), {labels:'designer'});
    assert.equal(q.plots[0].signals[0].address, 'v(xx1[0].in)');
    assert.equal(q.plots[0].signals[0].label, 'v(xx1[0].in)');
    assert.equal(q.plots[0].signals[0].role, undefined);
});
