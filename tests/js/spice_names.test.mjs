// SPDX-License-Identifier: MIT
// TASK-527: JS counterpart of silk.spice_names and the label identity behaviour.
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import test from 'node:test';
import {decimate, plan, read, to_designer_name, to_spice_name}
    from '../../silk/signals/view/assets/waveform-core.mjs';

const encode = text => new TextEncoder().encode(text);

function plot(names, plotname) {
    const n = names.length + 1;
    const vars = ['\t0\ttime\ttime', ...names.map((nm, i) => `\t${i + 1}\t${nm}\tvoltage`)].join('\n');
    const rows = [0, 1].map(p => `${p}\t${p}\n${names.map((_, i) => `\t${p + i}\n`).join('')}`).join('');
    return `Title: t\nPlotname: ${plotname}\nFlags: real\nNo. Variables: ${n}\nNo. Points: 2\nVariables:\n${vars}\nValues:\n${rows}`;
}
const file = (...plots) => encode(plots.join(''));
const sha = bytes => createHash('sha256').update(bytes).digest('hex').slice(0, 16);

async function run(bytes, labels) {
    const planned = plan(read(bytes, {file_name: 'x.raw'}));
    return decimate(planned, labels === undefined ? {} : {labels});
}
const shown = p => p.plots.map(pl => pl.signals.map(s => [s.name, s.label ?? null]));

test('codec: instance positions only, wrappers set role, leaves untouched', () => {
    assert.equal(to_spice_name('xx1[0].xy[1].g[0]'), 'xx1<0>.xy<1>.g[0]');
    assert.equal(to_designer_name('xx1<0>.xy<1>.g[0]'), 'xx1[0].xy[1].g[0]');
    assert.equal(to_designer_name('g<0>'), 'g<0>');
    assert.equal(to_spice_name('g[0]'), 'g[0]');
    assert.equal(to_spice_name('V(vvs[0])'), 'V(vvs[0])');
    assert.equal(to_spice_name('I(vvs[0])'), 'I(vvs<0>)');
    assert.equal(to_spice_name('vvs[0]', {kind: 'branch'}), 'vvs<0>');
    assert.equal(to_designer_name('@m.x<0>.m0[gm]'), '@m.x[0].m0[gm]');
    assert.equal(to_designer_name('N(XDUT:XX1<0>:XMM1:GM)'), 'N(XDUT:XX1[0]:XMM1:GM)');
    assert.equal(to_designer_name('v(a<0>)-v(b<1>)'), 'v(a<0>)-v(b<1>)');
    assert.throws(() => to_spice_name('a', {kind: 'net'}), RangeError);
    assert.throws(() => to_spice_name('a', {style: 'bogus'}), RangeError);
});

test('no labels: identity bytes are exactly today\'s', async () => {
    const bytes = file(plot(['v(a)', 'v(b)'], 'Transient Analysis'));
    const base = await run(bytes);
    const suffix = encode('\0' + JSON.stringify([2, 8000, 8, 1000]));
    const ident = new Uint8Array(bytes.length + suffix.length);
    ident.set(bytes); ident.set(suffix, bytes.length);
    assert.equal(base.run.id.length, 16);
    assert.equal((await run(bytes, {})).run.id, base.run.id);
    assert.equal((await run(bytes, undefined)).run.id, base.run.id);
    assert.ok(base.plots[0].signals.every(s => !('label' in s)));
    assert.equal(base.run.id, sha(ident), 'matches the documented identity (default decimation 8000/8/1000)');
});

test('labels are emitted, run.id changes, state_key and raw names unchanged', async () => {
    const bytes = file(plot(['v(x<0>.m)'], 'Transient Analysis'));
    const a = await run(bytes);
    const b = await run(bytes, {'v(x<0>.m)': 'v(x[0].m)'});
    const c = await run(bytes, {'v(x<0>.m)': 'v(x[0].M)'});
    assert.equal(b.plots[0].signals[0].label, 'v(x[0].m)');
    assert.equal(b.plots[0].signals[0].name, 'v(x<0>.m)');
    assert.equal(new Set([a.run.id, b.run.id, c.run.id]).size, 3);
    assert.equal(a.run.state_key, b.run.state_key);
});

test('collisions drop labels for all affected traces, per plot, until stable', async () => {
    const bytes = file(plot(['v(a)', 'v(b)'], 'P0'), plot(['v(a)', 'v(c)'], 'P1'));
    const p = await run(bytes, {'v(a)': 'L', 'v(b)': 'L'});
    assert.deepEqual(shown(p), [[['v(a)', null], ['v(b)', null]], [['v(a)', 'L'], ['v(c)', null]]]);
    const cascade = await run(file(plot(['v(a)', 'v(b)', 'v(c)'], 'P')), {'v(a)': 'v(b)', 'v(b)': 'v(c)'});
    assert.ok(cascade.plots[0].signals.every(s => !('label' in s)));
});

test('hash distinguishes the plot of the surviving label (A/B regression)', async () => {
    const bytes = file(plot(['v(a)', 'v(b)'], 'P0'), plot(['v(a)', 'v(c)'], 'P1'));
    const A = await run(bytes, {'v(a)': 'L', 'v(b)': 'L'});   // plot 1 v(a) shows L
    const B = await run(bytes, {'v(a)': 'L', 'v(c)': 'L'});   // plot 0 v(a) shows L
    assert.deepEqual(shown(A)[1][0], ['v(a)', 'L']);
    assert.deepEqual(shown(B)[0][0], ['v(a)', 'L']);
    assert.notEqual(A.run.id, B.run.id);
});

test('identical effective labels keep run.id; all-dropped equals no labels', async () => {
    const bytes = file(plot(['v(a)', 'v(b)'], 'P0'));
    const none = await run(bytes);
    const dropped = await run(bytes, {'v(a)': 'L', 'v(b)': 'L'});
    assert.equal(dropped.run.id, none.run.id);
    const one = await run(bytes, {'v(a)': 'Q'});
    const again = await run(bytes, {'v(a)': 'Q'});
    assert.equal(one.run.id, again.run.id);
});

test('label input is validated like Python: unknown keys and bad labels are refused', async () => {
    const bytes = file(plot(['v(in)'], 'Transient Analysis'));
    await assert.rejects(run(bytes, {'v(in)': 17}), TypeError);
    await assert.rejects(run(bytes, {'v(in)': ''}), TypeError);
    await assert.rejects(run(bytes, {'v(zzz)': 'x'}), error => error.name === 'ViewError' && /v\(in\)/.test(error.message));
    await assert.rejects(run(bytes, ['v(in)']), TypeError);
    assert.ok((await run(bytes, {})).plots[0].signals.every(s => !('label' in s)));
});

test('codec grammar guard: unsupported style grammar and unknown forms are unchanged', () => {
    assert.equal(to_designer_name('I(XA<0>:VB<1>)', {style: 'ngspice'}), 'I(XA<0>:VB<1>)');
    assert.equal(to_designer_name('v(xa<0>.mid)', {style: 'xyce'}), 'v(xa<0>.mid)');
    assert.equal(to_designer_name('@m.xa<0>.m0[gm]', {style: 'behavioral'}), '@m.xa<0>.m0[gm]');
    assert.equal(to_designer_name('xx1<0>.mid?xx2<0>.a:b'), 'xx1<0>.mid?xx2<0>.a:b');
    assert.equal(to_designer_name('xx1<0>.m1#body'), 'xx1<0>.m1#body');
    assert.equal(to_designer_name('xx1<0>.mid:xx2<0>.a'), 'xx1<0>.mid:xx2<0>.a');
    assert.equal(to_designer_name('XDUT:XX1<0>:MID'), 'XDUT:XX1[0]:MID');
    assert.equal(to_spice_name('x[-1].a'), 'x<-1>.a');
});

test('suffix triples use final names after complex expansion and dedup', async () => {
    const ac = encode(`Title: t\nPlotname: AC\nFlags: complex\nNo. Variables: 2\nNo. Points: 2\nVariables:\n\t0\tfrequency\tfrequency\n\t1\tv(a)\tvoltage\nValues:\n0\t1,0\n\t1,1\n1\t2,0\n\t2,2\n`);
    const p = await run(ac, {'v(a)': 'Q'});
    assert.deepEqual(shown(p)[0], [['v(a) (mag)', 'Q (mag)'], ['v(a) (phase)', 'Q (phase)']]);
    const dup = file(plot(['v(a)', 'v(a)'], 'P'));
    const d = await run(dup, {'v(a)': 'Q'});
    assert.deepEqual(shown(d)[0].map(([n]) => n), ['v(a)', 'v(a)#2']);
});

test('suffix order is plot then position', async () => {
    const bytes = file(plot(['v(a)', 'v(b)'], 'P0'), plot(['v(a)', 'v(b)'], 'P1'));
    const x = await run(bytes, {'v(a)': 'A', 'v(b)': 'B'});
    const y = await run(bytes, {'v(a)': 'B', 'v(b)': 'A'});
    assert.notEqual(x.run.id, y.run.id);
});

test('codec: unicode names convert exactly like Python', () => {
    for (const suffix of ['é', '中', '²', 'Ω']) {
        assert.equal(to_designer_name(`x${suffix}<0>.mid`), `x${suffix}[0].mid`);
        assert.equal(to_spice_name(`x${suffix}[0].mid`), `x${suffix}<0>.mid`);
        assert.equal(to_spice_name(`v${suffix}[0]`, {kind: 'branch'}), `v${suffix}<0>`);
    }
    assert.equal(to_designer_name('x\u00a0<0>.mid'), 'x\u00a0<0>.mid');
});

test('codec: ASCII marks outside the original set and every C1 control stay unrecognised', () => {
    for (const name of ['xx1<0>.mid&xx2<0>.mid', 'xx1<0>.mid%2', 'xx1<0>.mid^2', '!xx1<0>.mid',
        'xx1<0>.mid\u0080', 'xx1<0>.mid\u0085', 'xx1<0>.m\u008did', 'xx1<0>.mid\u009f']) {
        assert.equal(to_designer_name(name), name);
    }
    assert.equal(to_designer_name('a$b-c_d<0>.e9'), 'a$b-c_d[0].e9');
});

test('codec: line ends and trailing spaces leave a name byte for byte (no multiline matching)', () => {
    for (const name of ['xx1<0>.mid', 'v(xx1<0>.mid)', 'i(vvs<0>)', '@m.xx1<0>.m0[gm]']) {
        for (const ending of ['\n', '\r', '\r\n', '\u2003']) {
            assert.equal(to_designer_name(name + ending), name + ending);
        }
    }
});
