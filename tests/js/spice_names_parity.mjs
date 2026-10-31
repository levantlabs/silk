#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Python/JS parity for the SPICE-name codec and effective labels (TASK-527).
// run.id is NOT compared: the two hosts hash differently on purpose.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {dirname, resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {decimate, plan, read, to_designer_name, to_spice_name}
    from '../../silk/signals/view/assets/waveform-core.mjs';

const {values} = parseArgs({options: {cases: {type: 'string'}}});
const cases = JSON.parse(await readFile(resolve(values.cases), 'utf8'));
for (const c of cases.codec) {
    assert.equal(to_spice_name(c.name, {kind: c.kind, style: c.style, allow_glob: c.allow_glob ?? false}), c.spice, `to_spice ${c.name} ${c.kind} ${c.style}`);
    assert.equal(to_designer_name(c.name, {kind: c.kind, style: c.style}), c.designer, `to_designer ${c.name} ${c.kind} ${c.style}`);
}
for (const c of cases.labels) {
    const bytes = await readFile(resolve(dirname(values.cases), c.raw));
    const payload = await decimate(plan(read(bytes, {file_name: c.raw})), {labels: c.labels});
    const shown = payload.plots.map(p => p.signals.map(s => [s.name, s.label ?? null]));
    assert.deepEqual(shown, c.shown, `labels ${c.raw}`);
}
for (const c of cases.designer) {
    const bytes = await readFile(resolve(dirname(values.cases), c.raw));
    const payload = await decimate(plan(read(bytes, {file_name: c.raw})), {labels: 'designer'});
    const shown = payload.plots.map(p => p.signals.map(s => [s.name, s.label ?? null, s.address ?? null]));
    assert.deepEqual(shown, c.shown, `designer ${c.raw}`);
}
console.log(`ok: ${cases.codec.length} codec cases, ${cases.labels.length} label cases, ${cases.designer.length} designer cases`);
