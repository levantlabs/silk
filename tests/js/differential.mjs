#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Standalone: only Node built-ins and the Python-generated oracle directory.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {parseArgs} from 'node:util';

const EXCLUDED = new Set(['run.id', 'run.emitted_at', 'run.source', 'generator_version']);

function decode_floats(base64, dt) {
    const bytes = Buffer.from(base64, 'base64');
    const width = dt === 'f32' ? 4 : 8;
    assert.equal(bytes.length % width, 0, 'unaligned oracle leaf');
    const out = dt === 'f32' ? new Float32Array(bytes.length / width) : new Float64Array(bytes.length / width);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < out.length; i++) {
        out[i] = dt === 'f32' ? view.getFloat32(i * width, true) : view.getFloat64(i * width, true);
    }
    return out;
}

export function normalise(value) {
    if (value === null || typeof value !== 'object' || ArrayBuffer.isView(value)) return value;
    if (Object.hasOwn(value, '$f64')) return decode_floats(value.$f64, 'f64');
    if (Array.isArray(value)) return value.map(normalise);
    const out = Object.create(null);
    for (const [key, child] of Object.entries(value)) {
        out[key] = Object.hasOwn(value, 'stride') && (key === 'x' || key === 'y') && typeof child === 'string'
            ? decode_floats(child, key === 'x' ? 'f64' : value.dt) : normalise(child);
    }
    return out;
}

export function compare(actual, expected, path = '', exclude = EXCLUDED) {
    if (exclude.has(path)) return;
    if (ArrayBuffer.isView(expected)) {
        assert.equal(actual?.constructor?.name, expected.constructor.name, `${path}: typed-array dtype`);
        assert.equal(actual.length, expected.length, `${path}: length`);
        for (let i = 0; i < expected.length; i++) {
            assert.ok(Object.is(actual[i], expected[i]),
                `${path}[${i}]: ${Object.is(actual[i], -0) ? '-0' : actual[i]} !== ${Object.is(expected[i], -0) ? '-0' : expected[i]}`);
        }
        return;
    }
    if (expected !== null && typeof expected === 'object') {
        assert.ok(actual !== null && typeof actual === 'object', `${path}: expected object`);
        assert.equal(Array.isArray(actual), Array.isArray(expected), `${path}: array/object`);
        // Even the excluded fields must be present; only their values differ.
        assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), `${path}: keys`);
        for (const key of Object.keys(expected)) {
            compare(actual[key], expected[key], path ? `${path}.${key}` : key, exclude);
        }
        return;
    }
    assert.ok(Object.is(actual, expected), `${path}: ${JSON.stringify(actual)} !== ${JSON.stringify(expected)}`);
}

async function main() {
    const {values} = parseArgs({options: {module: {type: 'string'}, expected: {type: 'string'}, help: {type: 'boolean'}}});
    if (values.help) {
        console.log('node differential.mjs --module /vendored/waveform-core.mjs --expected /tmp/silk-oracle');
        return;
    }
    assert.ok(values.module && values.expected, '--module and --expected are required (see --help)');
    const core = await import(pathToFileURL(resolve(values.module)).href);
    assert.deepEqual(Object.keys(core).sort(), ['build_pyramid', 'collect_vector_candidates', 'decimate', 'designer_alias_map', 'first_row', 'plan', 'read', 'to_designer_name', 'to_spice_name', 'vector_address']);
    const dir = resolve(values.expected);
    const manifest = JSON.parse(await readFile(resolve(dir, 'manifest.json'), 'utf8'));
    assert.equal(manifest.format, 1);
    assert.ok(manifest.corpus_count > 0 && manifest.entries.length >= manifest.corpus_count, 'empty corpus');
    let payloads = 0;
    for (const entry of manifest.entries) {
        try {
            const bytes = await readFile(resolve(dir, entry.raw));
            const expected = JSON.parse(await readFile(resolve(dir, entry.expected), 'utf8'));
            // A Buffer is a Uint8Array view, often with an unaligned byteOffset.
            const raw = core.read(bytes, {file_name: entry.file_name});
            const {bytes: retained, ...reader} = raw;
            assert.deepEqual(retained, new Uint8Array(bytes));
            compare(reader, normalise(expected.reader), '', new Set());
            const planned = core.plan(raw);
            for (const example of expected.cases) {
                const payload = await core.decimate(planned, example.options);
                compare(payload, normalise(example.payload));
                payloads++;
            }
        } catch (error) {
            throw new Error(`${entry.file_name}: ${error.message}`, {cause: error});
        }
    }
    console.log(`PASS: ${manifest.entries.length} rawfiles (${manifest.corpus_count} corpus), ${payloads} payloads; Object.is leaves, four named exclusions`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
    main().catch(error => { console.error(error); process.exitCode = 1; });
}
