// SPDX-License-Identifier: MIT
// TASK-530 round 3: the JS twin of silk.vector_match and the explicit designer
// mode of the core. Shared fixed expectations: tests/fixtures/vector_match_cases.json.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';

import {collect_vector_candidates, decimate, designer_alias_map, first_row, plan, read, vector_address}
    from '../../silk/signals/view/assets/waveform-core.mjs';

const cases = JSON.parse(await readFile(new URL('../fixtures/vector_match_cases.json', import.meta.url), 'utf8'));
const stub = name => {
    const p = cases.plots[name];
    return {plotname: p.plotname, real: p.real, npoints: p.npoints, vectors: p.vectors.map(n => ({name: n}))};
};

test('first_row: 0 for operating points, 1 for axis plots', () => {
    for (const c of cases.first_row) assert.equal(first_row(stub(c.plot)), c.expect, c.plot);
});

test('candidates are the fixed row indices', () => {
    for (const c of cases.candidates) {
        const plot = stub(c.plot);
        assert.deepEqual(collect_vector_candidates(plot, c.name, c.type, first_row(plot)), c.expect, JSON.stringify(c));
    }
});

test('addresses are proven against the whole source plot', () => {
    for (const c of cases.address) assert.equal(vector_address(stub(c.plot), c.row), c.expect, JSON.stringify(c));
});

test('designer_alias_map keeps collision groups raw and ignores order', () => {
    const a = designer_alias_map(['xx1[0].mid', 'xx1<0>.mid', 'xx1<0>.o', 'd[0]']);
    const b = designer_alias_map(['d[0]', 'xx1<0>.o', 'xx1<0>.mid', 'xx1[0].mid']);
    assert.deepEqual([...a].sort(), [...b].sort());
    assert.equal(a.get('xx1<0>.mid'), 'xx1<0>.mid');
    assert.equal(a.get('xx1<0>.o'), 'xx1[0].o');
});

const bytesOf = async () => readFile(new URL('../fixtures/rawfiles/a_tran_ascii.raw', import.meta.url));

test('decimate(plan(x)) with no options carries no label, no address and the format-2 run.id', async () => {
    const payload = await decimate(plan(read(await bytesOf(), {file_name: 'a_tran_ascii.raw'})));
    // Format 2: SHA-256(raw bytes + NUL + '[2,8000,8,1000]'), first 16 hex digits.
    // Independently recomputed with Python hashlib (TASK-544).
    assert.equal(payload.run.id, '790140d1627e0cc6');
    for (const plot of payload.plots) for (const s of plot.signals) {
        assert.equal(s.label, undefined); assert.equal(s.address, undefined);
    }
});

function ascii(names) {
    const n = 3;
    const lines = ['Title: t', 'Date: d', 'Plotname: Transient Analysis', 'Flags: real',
        `No. Variables: ${names.length + 1}`, `No. Points: ${n}`, 'Variables:', '\t0\ttime\ttime'];
    names.forEach((name, i) => lines.push(`\t${i + 1}\t${name}\tvoltage`));
    lines.push('Values:');
    for (let p = 0; p < n; p++) {
        lines.push(`${p}\t${p}`);
        names.forEach((_, i) => lines.push(`\t${p + i}`));
    }
    return new TextEncoder().encode(lines.join('\n') + '\n');
}

test('designer mode labels and addresses; dicts and no options never do', async () => {
    const planned = () => plan(read(ascii(['v(xx1<0>.mid)', 'v(out)']), {file_name: 'x.raw'}));
    const auto = await decimate(planned(), {labels: 'designer'});
    const sigs = auto.plots[0].signals;
    assert.equal(sigs[0].label, 'v(xx1[0].mid)'); assert.equal(sigs[0].address, 'v(xx1[0].mid)');
    assert.equal(sigs[1].label, undefined); assert.equal(sigs[1].address, undefined);
    const dict = await decimate(planned(), {labels: {'v(xx1<0>.mid)': 'v(xx1[0].mid)'}});
    assert.equal(dict.plots[0].signals[0].label, 'v(xx1[0].mid)');
    assert.equal(dict.plots[0].signals[0].address, undefined);          // a dict never yields an address
    assert.notEqual(auto.run.id, dict.run.id);                          // an address exists only in auto mode
    const bare = await decimate(planned());
    assert.equal(bare.plots[0].signals[0].label, undefined);
});

test('auto equals an identical explicit dict when no address can be proved', async () => {
    const planned = () => plan(read(ascii(['v(xx1<0>.mid)', 'v(xx1[0]:mid)']), {file_name: 'x.raw'}));
    const auto = await decimate(planned(), {labels: 'designer'});
    const sigs = auto.plots[0].signals;
    assert.equal(sigs[0].label, 'v(xx1[0].mid)');            // the label is shown ...
    assert.equal(sigs[1].label, undefined);
    for (const s of sigs) assert.equal(s.address, undefined); // ... but no address is proved
    const explicit = await decimate(planned(), {labels: {'v(xx1<0>.mid)': 'v(xx1[0].mid)'}});
    assert.equal(auto.run.id, explicit.run.id);
    // trivial control: no labels on either side
    const t = () => plan(read(ascii(['v(xx1<0>.mid)', 'v(xx1[0].mid)']), {file_name: 'x.raw'}));
    assert.equal((await decimate(t(), {labels: 'designer'})).run.id, (await decimate(t(), {labels: {}})).run.id);
});
