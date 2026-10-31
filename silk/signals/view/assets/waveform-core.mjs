// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Levant Labs. Distributed under silk's MIT LICENSE.
/**
 * Standalone nutmeg reader and DATA_FORMAT_VERSION 2 waveform producer.
 * No imports, build step, DOM, viewer, or third-party runtime dependencies.
 *
 * const raw = read(bytes, {file_name: file.name});
 * const payload = await decimate(plan(raw), {min_points: 8000, base: 8,
 *                                          min_bucket: 1000});
 * read/plan are synchronous; decimate uses Web Crypto (Node >=20 or a secure
 * browser context). All numeric leaves are Float64Array/Float32Array, never
 * base64. Complex reader data is {re: Float64Array, im: Float64Array}.
 *
 * run.id = first 16 lowercase hex digits of SHA-256(raw file bytes || UTF-8
 * of "\0" + JSON.stringify([2, min_points, base, min_bucket])). Defaults are
 * resolved before hashing; the pyramid has at most eight levels. This is
 * deliberately different from Python's hash of the base64 payload.
 * run.state_key = first 8 hex digits of SHA-256(UTF-8(file_name)), falling
 * back to '<in-memory>' when file_name is empty. Consumers may replace the
 * state key with a stable project/module/testbench identity.
 *
 * plan emits all vectors, expands complex data, and chooses a default plot.
 * Selection, annotations, context, scopes, and viewer integration belong to
 * the host. read snapshots the input bytes, retained through plan for hashing;
 * don't mutate the intermediate objects or detach them before decimate ends.
 * Parsing mirrors silk/signals/rawfile.py; planning and encoding mirror
 * silk/signals/view/{emit,encode}.py and the reference min/max loop in tests.
 */

class RawfileError extends Error {
    constructor(message) { super(message); this.name = 'RawfileError'; }
}

class ViewError extends Error {
    constructor(message) { super(message); this.name = 'ViewError'; }
}

const UNIT_TABLE = new Map(Object.entries({
    voltage: 'V', current: 'A', power: 'W', charge: 'C', capacitance: 'F',
    notype: '', '': '', time: 's', frequency: 'Hz', 'temp-sweep': '°C',
    'res-sweep': 'Ω', impedance: 'Ω', admittance: 'S',
}));
const DEFAULTS = {min_points: 8000, base: 8, min_bucket: 1000};

// JS trim/split differ from Python str.strip/split at U+0085 and U+001C-1F.
const SPACE = '[\\x09-\\x0d\\x1c-\\x20\\x85\\xa0]';
const TRIM = new RegExp(`^${SPACE}+|${SPACE}+$`, 'g');
const SPLIT = new RegExp(`${SPACE}+`);
function strip(s) { return s.replace(TRIM, ''); }
function words(s) { const t = strip(s); return t ? t.split(SPLIT) : []; }

function read_line(bytes, pos) {
    let end = bytes.indexOf(10, pos);
    if (end < 0) end = bytes.length;
    const next = Math.min(end + 1, bytes.length);
    while (end > pos && bytes[end - 1] === 13) end--;
    let line = '';
    // TextDecoder('latin1') actually uses Windows-1252 in browsers. A direct
    // map preserves every byte, including 80-9F, just like Python latin-1.
    for (let i = pos; i < end; i += 4096) {
        line += String.fromCharCode(...bytes.subarray(i, Math.min(i + 4096, end)));
    }
    return [line, next];
}

function to_int(text, fallback = 0) {
    const token = words(text ?? '')[0] ?? '';
    return /^[+-]?\d+$/.test(token) ? Number(token) : fallback;
}

function parse_float(token) {
    if (/^[+-]?nan$/i.test(token)) return NaN;
    if (/^[+]?inf(?:inity)?$/i.test(token)) return Infinity;
    if (/^-inf(?:inity)?$/i.test(token)) return -Infinity;
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(token)) {
        throw new RawfileError(`malformed rawfile: invalid numeric value ${JSON.stringify(token)}`);
    }
    return Number(token);
}

function to_float(text, fallback = 0) {
    const token = words(text ?? '')[0];
    if (token === undefined) return fallback;
    try { return parse_float(token); }
    catch (error) { if (error instanceof RawfileError) return fallback; throw error; }
}

function parse_variable(line) {
    const fields = line.includes('\t') ? line.split('\t').filter(Boolean) : words(line);
    const tail = words(fields.slice(2).join(' '));
    return {name: fields[1] ?? '', type: tail[0] ?? '', attrs: tail.slice(1)};
}

function value_tokens(line) {
    const tokens = [];
    for (const token of words(line)) {
        if (tokens.length && (tokens.at(-1).endsWith(',') || token.startsWith(','))) {
            tokens[tokens.length - 1] += token;
        } else tokens.push(token);
    }
    return tokens;
}

function make_columns(nvars, npoints, real) {
    return Array.from({length: nvars}, () => real ? new Float64Array(npoints) :
        {re: new Float64Array(npoints), im: new Float64Array(npoints)});
}

function read_binary(bytes, pos, nvars, npoints, real) {
    const width = real ? 8 : 16;
    npoints = Math.min(npoints, Math.floor((bytes.length - pos) / width / nvars));
    const columns = make_columns(nvars, npoints, real);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let point = 0; point < npoints; point++) {
        for (const column of columns) {
            const re = view.getFloat64(pos, true);
            if (real) column[point] = re;
            else {
                column.re[point] = re;
                column.im[point] = view.getFloat64(pos + 8, true);
            }
            pos += width;
        }
    }
    return [columns, npoints, pos];
}

function read_ascii(bytes, pos, nvars, npoints, real) {
    // Grow only for actual rows: a truncated file can declare an arbitrarily
    // large count without containing the corresponding data.
    const rows = [];
    for (let point = 0; point < npoints; point++) {
        const row = [];
        let first = true;
        while (row.length < nvars && pos < bytes.length) {
            let line;
            [line, pos] = read_line(bytes, pos);
            if (!strip(line)) continue;
            const tokens = value_tokens(line);
            if (first) { tokens.shift(); first = false; }
            for (const token of tokens) row.push(token);
        }
        if (row.length < nvars) break;
        rows.push(row.slice(0, nvars).map(token => {
            if (real) return parse_float(token);
            const comma = token.indexOf(',');
            return comma < 0 ? [parse_float(token), 0] :
                [parse_float(token.slice(0, comma)), parse_float(token.slice(comma + 1) || '0')];
        }));
    }
    npoints = rows.length;
    const columns = make_columns(nvars, npoints, real);
    for (let point = 0; point < npoints; point++) {
        for (let v = 0; v < nvars; v++) {
            if (real) columns[v][point] = rows[point][v];
            else {
                columns[v].re[point] = rows[point][v][0];
                columns[v].im[point] = rows[point][v][1];
            }
        }
    }
    return [columns, npoints, pos];
}

function realify_scale(vectors, real) {
    if (real || !vectors.length) return;
    const scale = vectors[0].data;
    if (scale.re.length) {
        let max_re = 0, max_im = 0;
        for (let i = 0; i < scale.re.length; i++) {
            max_re = Math.max(max_re, Math.abs(scale.re[i]));
            max_im = Math.max(max_im, Math.abs(scale.im[i]));
        }
        // Number.MIN_VALUE is denormal; numpy's tiny is the smallest NORMAL.
        const tolerance = Math.max(1e-12 * max_re, 2.2250738585072014e-308);
        if (max_im > tolerance) return;
    }
    vectors[0].data = scale.re;
}

function finish_plot(header, vectors, bytes, pos, binary) {
    const flags = words((header.flags ?? '').toLowerCase());
    const real = !flags.includes('complex');
    let npoints = to_int(header['no. points']);
    let columns;
    if (!vectors.length || npoints <= 0) {
        npoints = 0;
        columns = make_columns(vectors.length, 0, real);
    } else {
        [columns, npoints, pos] = (binary ? read_binary : read_ascii)(
            bytes, pos, vectors.length, npoints, real);
    }
    vectors.forEach((vector, index) => {
        vector.data = columns[index];
        // ngspice marks a vector it could not fill `dims=0`: honour it as
        // zero-length, matching silk.signals.rawfile (TASK-440).
        if (vector.attrs.includes('dims=0')) {
            vector.data = vector.data instanceof Float64Array ? vector.data.subarray(0, 0)
                : {re: vector.data.re.subarray(0, 0), im: vector.data.im.subarray(0, 0)};
        }
    });
    realify_scale(vectors, real);
    const offset = to_float(header.offset);
    if (offset !== 0 && vectors.length && npoints) {
        const scale = vectors[0].data;
        const re = scale instanceof Float64Array ? scale : scale.re;
        for (let i = 0; i < re.length; i++) {
            re[i] += offset;
            // numpy complex addition adds a +0 imaginary component too.
            if (!(scale instanceof Float64Array)) scale.im[i] += 0;
        }
    }
    return [{header, vectors, npoints, nvars: vectors.length, real, flags,
        padded: flags.includes('padded') ? true : flags.includes('unpadded') ? false : null,
        title: header.title ?? '', date: header.date ?? '',
        name: header.plotname ?? '', plotname: header.plotname ?? '', offset}, pos];
}

function is_plot_header(header) {
    return Object.hasOwn(header, 'plotname') || Object.hasOwn(header, 'no. variables');
}

function parse_plot(bytes, pos) {
    let header = Object.create(null);
    const vectors = [];
    while (pos < bytes.length) {
        let line;
        [line, pos] = read_line(bytes, pos);
        const stripped = strip(line);
        if (!stripped) continue;
        const colon = stripped.indexOf(':');
        if (colon < 0) {
            if (is_plot_header(header)) {
                throw new RawfileError(`malformed rawfile: expected a Keyword: value header or data section, got ${JSON.stringify(line)}`);
            }
            header = Object.create(null);
            continue;
        }
        const key = strip(stripped.slice(0, colon)).toLowerCase();
        const value = strip(stripped.slice(colon + 1));
        if (key === 'binary' || key === 'values') {
            return finish_plot(header, vectors, bytes, pos, key === 'binary');
        }
        if (key === 'variables') {
            const nvars = to_int(header['no. variables'], -1);
            if (nvars < 0 || !Number.isSafeInteger(nvars)) {
                throw new RawfileError('malformed rawfile: Variables: reached before a valid No. Variables: count');
            }
            let pending = value;
            while (vectors.length < nvars) {
                let varline;
                if (pending) { varline = pending; pending = ''; }
                else {
                    if (pos >= bytes.length) {
                        throw new RawfileError(`malformed rawfile: header declares ${nvars} variables but only ${vectors.length} are present`);
                    }
                    [varline, pos] = read_line(bytes, pos);
                }
                if (strip(varline)) vectors.push(parse_variable(varline));
            }
        } else header[key] = value;
    }
    if (is_plot_header(header)) {
        throw new RawfileError('malformed rawfile: plot header is not followed by a Binary: or Values: data section');
    }
    return [null, pos];
}

/** Parse ArrayBuffer/Uint8Array bytes, keeping complete rows of truncated data. */
export function read(input, {file_name = ''} = {}) {
    if (!(input instanceof ArrayBuffer) && !(input instanceof Uint8Array)) {
        throw new TypeError('read expects an ArrayBuffer or Uint8Array');
    }
    if (typeof file_name !== 'string') throw new TypeError('file_name must be a string');
    const bytes = new Uint8Array(input instanceof ArrayBuffer ? new Uint8Array(input) : input);
    const plots = [];
    let pos = 0;
    while (pos < bytes.length) {
        let plot;
        [plot, pos] = parse_plot(bytes, pos);
        if (!plot) break;
        plots.push(plot);
    }
    if (!plots.length) {
        throw new RawfileError(`${JSON.stringify(file_name)} is not a nutmeg rawfile: no plot header found (expected Plotname:/No. Variables: followed by Binary: or Values:)`);
    }
    return {plots, bytes, file_name};
}

function infer_kind(plotname, x_name, x_type) {
    const pn = plotname.toLowerCase(), xn = x_name.toLowerCase(), xt = x_type.toLowerCase();
    if (pn.includes('constants')) return 'constants';
    if (pn.includes('noise')) return 'noise';
    if (pn.includes('ac analysis') || xn === 'frequency' || xt === 'frequency') return 'ac';
    if (pn.includes('transient') || xn === 'time' || xt === 'time') return 'tran';
    if (pn.includes('dc transfer') || pn.includes('dc analysis') ||
        ['sweep', 'v-sweep', 'temp-sweep'].includes(xn) || xn.startsWith('v(')) return 'dc';
    return 'other';
}

function unwrap_phase_degrees(z) {
    const out = new Float64Array(z.re.length);
    let previous, correction = 0;
    for (let i = 0; i < out.length; i++) {
        const angle = Math.atan2(z.im[i], z.re[i]);
        if (!Number.isFinite(angle)) { out[i] = NaN; continue; }
        if (previous !== undefined) {
            const delta = angle - previous;
            let remainder = (delta + Math.PI) % (2 * Math.PI);
            if (remainder < 0) remainder += 2 * Math.PI;
            let wrapped = remainder - Math.PI;
            if (wrapped === -Math.PI && delta > 0) wrapped = Math.PI;
            correction += Math.abs(delta) < Math.PI ? 0 : wrapped - delta;
        }
        // numpy unwrap leaves the first sample untouched (including -0).
        out[i] = (previous === undefined ? angle : angle + correction) * (180 / Math.PI);
        previous = angle;
    }
    return out;
}

function dedup_name(name, seen) {
    let count = (seen.get(name) ?? 0) + 1;
    if (count === 1) { seen.set(name, 1); return [name, false]; }
    let candidate = `${name}#${count}`;
    while (seen.has(candidate)) candidate = `${name}#${++count}`;
    seen.set(name, count);
    seen.set(candidate, 1);
    return [candidate, true];
}

function plan_plot(plot) {
    const first = plot.vectors[0];
    const constants = plot.plotname.toLowerCase().includes('constants');
    const op = !constants && plot.real && !!first &&
        plot.plotname.toLowerCase() === 'operating point';
    const start = op ? 0 : 1;
    const x_vec = op ? null : first;
    const x_arr = op ? Float64Array.from({length: plot.npoints}, (_, i) => i) : x_vec ? (x_vec.data instanceof Float64Array ? x_vec.data : x_vec.data.re) : new Float64Array(0);
    const x_name = op ? 'sample' : x_vec?.name ?? '', x_type = x_vec?.type ?? '';
    const kind = constants ? 'constants' : !plot.real ? 'ac' : op ? 'op' : infer_kind(plot.plotname, x_name, x_type);
    const signals = [], seen = new Map();
    for (const [index, vec] of plot.vectors.slice(start).entries()) {
        const parts = plot.real ? [[null, vec.data]] : [
            ['mag', Float64Array.from(vec.data.re, (re, i) => Math.hypot(re, vec.data.im[i]))],
            ['phase', unwrap_phase_degrees(vec.data)],
        ];
        for (const [part, y_arr] of parts) {
            const [name, renamed] = dedup_name(part ? `${vec.name} (${part})` : vec.name, seen);
            signals.push({name, type: vec.type, part, group: part ? vec.name : null, y_arr, renamed, raw: vec.name, row: index + start});
        }
    }
    return {plotname: plot.plotname, kind, complex: !plot.real, npoints: plot.npoints,
        source: {plotname: plot.plotname, real: plot.real, npoints: plot.npoints,
            vectors: plot.vectors.map(vec => ({name: vec.name}))},
        x_arr, x_name, x_type, signals, date: plot.date,
        x_log_default: kind === 'ac' && x_arr.length > 0 && x_arr.every(x => x > 0)};
}

function pick_default_plot(plots) {
    for (const predicate of [p => p.kind !== 'constants' && p.signals.length,
        p => p.signals.length, p => p.kind !== 'constants']) {
        for (let i = plots.length - 1; i >= 0; i--) if (predicate(plots[i])) return i;
    }
    return Math.max(plots.length - 1, 0);
}

/** Plan every vector from read() without selection or host-specific metadata. */
export function plan(raw) {
    if (!raw?.plots?.length) throw new ViewError('rawfile contains no plots');
    const plots = raw.plots.map(plan_plot);
    const default_plot = pick_default_plot(plots);
    return {plots, default_plot, bytes: raw.bytes, file_name: raw.file_name,
        title: raw.plots[default_plot].title || raw.file_name.split('/').at(-1) || 'rawfile'};
}

function decimation_strides(n, {min_points, base, min_bucket}) {
    const strides = [1];
    if (n > min_points) {
        let stride = 1;
        while (strides.length < 8) {
            stride *= base;
            if (Math.ceil(n / stride) < min_bucket) break;
            strides.push(stride);
        }
    }
    return strides;
}

function minmax_level(x, y, stride) {
    const xs = new Float64Array(2 * Math.ceil(y.length / stride));
    const ys = new Float64Array(xs.length);
    let used = 0;
    for (let lo = 0; lo < y.length; lo += stride) {
        const hi = Math.min(lo + stride, y.length);
        let imin = lo, imax = lo, min = Infinity, max = -Infinity, all_nan = true;
        for (let i = lo; i < hi; i++) {
            const value = y[i];
            if (!Number.isNaN(value)) all_nan = false;
            // numpy nanargmin/max replace NaN with +/-Inf. Strict comparisons
            // preserve first-index ties, even a leading NaN tied with an Inf.
            const vmin = Number.isNaN(value) ? Infinity : value;
            const vmax = Number.isNaN(value) ? -Infinity : value;
            if (vmin < min) { min = vmin; imin = i; }
            if (vmax > max) { max = vmax; imax = i; }
        }
        if (all_nan) { xs[used] = x[lo]; ys[used++] = NaN; continue; }
        for (const index of imin <= imax ? [imin, imax] : [imax, imin]) {
            xs[used] = x[index]; ys[used++] = y[index];
        }
    }
    return [xs.subarray(0, used), ys.subarray(0, used)];
}

function decide_y_dtype(y) {
    return y.some(v => Number.isFinite(v) &&
        (Math.abs(v) > 3e38 || (v !== 0 && Math.abs(v) < 1e-37))) ? 'f64' : 'f32';
}

function count_nonfinite(values) {
    let count = 0;
    for (const value of values) if (!Number.isFinite(value)) count++;
    return count;
}

function signal_meta(name, type, part, group, values) {
    return {name, type, unit: part === 'phase' ? 'deg' : UNIT_TABLE.get(type.toLowerCase()) ?? '',
        part, group, nonfinite: count_nonfinite(values)};
}

function build_x_signal(pp) {
    return {...signal_meta(pp.x_name, pp.x_type, null, null, pp.x_arr),
        levels: [{stride: 1, n: pp.x_arr.length, dt: 'f64', x: pp.x_arr}]};
}

/** Build a fresh min/max pyramid from aligned level-0 numeric arrays.
 * x is the shared plot x array; every decimated level has its own extrema x.
 * Returns {levels, nonfinite}; level 0 deliberately carries no x of its own.
 * Input arrays must remain unchanged while the result is in use.
 */
export function build_pyramid(x, y, options = {}) {
    if (!x || !y || !Number.isSafeInteger(y.length) || x.length !== y.length) {
        throw new RangeError('build_pyramid: x and y must have the same array length');
    }
    const params = decimation_options(options);
    const dt = decide_y_dtype(y);
    const cast = values => dt === 'f32' ? new Float32Array(values)
        : values instanceof Float64Array ? values : new Float64Array(values);
    const levels = [{stride: 1, n: y.length, dt, y: cast(y)}];
    for (const stride of decimation_strides(y.length, params).slice(1)) {
        const [xs, ys] = minmax_level(x, y, stride);
        levels.push({stride, n: xs.length, dt, x: xs, y: cast(ys)});
    }
    return {levels, nonfinite: count_nonfinite(y)};
}

function build_data_signal(sig, x, options) {
    // A zero-length vector (rawfile `dims=0`, TASK-440) has no points to align
    // with x; like the Python encoder it carries a single empty level 0.
    const pyramid = sig.y_arr.length === 0
        ? build_pyramid(x.subarray(0, 0), sig.y_arr, options)
        : build_pyramid(x, sig.y_arr, options);
    const signal = {...signal_meta(sig.name, sig.type, sig.part, sig.group, sig.y_arr),
        ...pyramid};
    if (sig.renamed) signal.renamed = true;
    return signal;
}

async function sha256(bytes) {
    if (!globalThis.crypto?.subtle) {
        throw new ViewError('SHA-256 requires Web Crypto: use Node >=20 or a secure browser context');
    }
    const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
}

function decimation_options(options) {
    for (const key of Object.keys(options)) {
        if (!Object.hasOwn(DEFAULTS, key)) throw new TypeError(`unknown decimation option: ${key}`);
    }
    const params = {...DEFAULTS, ...options};
    for (const [key, value] of Object.entries(params)) {
        const minimum = key === 'min_points' ? 0 : key === 'base' ? 2 : 1;
        if (!Number.isSafeInteger(value) || value < minimum) {
            throw new RangeError(`${key} must be a safe integer >= ${minimum}`);
        }
    }
    return params;
}

// ---- SPICE-name codec (TASK-527): twin of silk/spice_names.py -------------
const NAME_STYLES = ['auto', 'ngspice', 'xyce', 'behavioral'];
const NAME_KINDS = ['node', 'branch'];
const WRAPPER_ROLE = {v: 'node', i: 'branch', n: 'node', id: 'branch'};
const RAW_XYCE = ['n', 'id'];
const STYLE_WRAPPERS = {auto: ['v', 'i', 'n', 'id'], ngspice: ['v', 'i'],
    xyce: ['v', 'i', 'n', 'id'], behavioral: []};
const STYLE_SEPS = {auto: '.:', ngspice: '.', xyce: ':', behavioral: '.'};
const STYLE_AT = {auto: true, ngspice: true, xyce: false, behavioral: false};
// One path segment, spelled out (the same text as silk/spice_names.py) so that
// recognition never depends on a runtime's Unicode tables. A character is EITHER
// in the ASCII set exactly (A-Z a-z 0-9 _ $ - [ ] < >) OR non-ASCII and neither
// a C1 control (U+0080-U+009F) nor a listed Unicode space/BOM (U+00A0, U+1680,
// U+2000-U+200A, U+2028, U+2029, U+202F, U+205F, U+3000, U+FEFF). Recognition is
// of the WHOLE string (no multiline flag, so '$' is the absolute end).
const SEGMENT = /^(?:[A-Za-z0-9_$\-\[\]<>]|[^\x00-\x9f\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff])+$/;

// Selector recognition (TASK-530, allow_glob): the same text plus '*' and '?'.
const SEGMENT_GLOB = /^(?:[A-Za-z0-9_$\-\[\]<>*?]|[^\x00-\x9f\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff])+$/;

const spice_component = part => part.replaceAll('[', '<').replaceAll(']', '>');
const designer_component = part => part.replaceAll('<', '[').replaceAll('>', ']');

function convert_path(path, conv, style, last_is_instance, seps = STYLE_SEPS[style], glob = false) {
    if (!path) return null;
    const present = ['.', ':'].filter(c => path.includes(c));
    if (present.length > 1 || (present.length && !seps.includes(present[0]))) return null;
    const segments = present.length ? path.split(present[0]) : [path];
    const segment_re = glob ? SEGMENT_GLOB : SEGMENT;
    if (!segments.every(seg => segment_re.test(seg))) return null;
    const last = segments.length - 1;
    return segments.map((seg, i) => (i < last || last_is_instance) ? conv(seg) : seg)
        .join(present.length ? present[0] : '');
}

function convert_body(body, conv, style, role, wrapper = null, glob = false) {
    if (body.startsWith('@')) {
        if (!STYLE_AT[style]) return null;
        const tail = /\[[^\[\]]*\]$/.exec(body);
        const param = tail ? tail[0] : '';
        const done = convert_path(body.slice(1, body.length - param.length), conv, style, true, '.', glob);
        return done === null ? null : '@' + done + param;
    }
    const seps = RAW_XYCE.includes(wrapper) ? ':' : undefined;
    if (seps !== undefined && !STYLE_SEPS[style].includes(seps)) return null;
    return convert_path(body, conv, style, role === 'branch', seps, glob);
}

function convert_name(name, conv, {style = 'auto', kind = 'node', allow_glob = false} = {}) {
    if (!NAME_STYLES.includes(style)) throw new RangeError(`style must be one of ${NAME_STYLES.join(', ')}`);
    if (!NAME_KINDS.includes(kind)) throw new RangeError(`kind must be one of ${NAME_KINDS.join(', ')}`);
    if (typeof name !== 'string') throw new TypeError('name must be a string');
    const match = /^([A-Za-z]+)\(([\s\S]*)\)$/.exec(name);
    if (match) {
        const head = match[1].toLowerCase();
        if (!STYLE_WRAPPERS[style].includes(head)) return name;
        const done = convert_body(match[2], conv, style, WRAPPER_ROLE[head], head, allow_glob);
        return done === null ? name : `${match[1]}(${done})`;
    }
    const done = convert_body(name, conv, style, kind, null, allow_glob);
    return done === null ? name : done;
}

/** Bracket spelling to the simulator's: xx1[0].mid -> xx1<0>.mid. */
export const to_spice_name = (name, options) => convert_name(name, spice_component, options);
/** Simulator spelling to a bracketed SPICE alias (not designer case or a verified entity). */
export const to_designer_name = (name, options) => convert_name(name, designer_component, options);

// ---- the vector matcher (twin of silk/vector_match.py, TASK-530) ----------
// Shared fixed expectations: tests/fixtures/vector_match_cases.json.
const OP_PLOTNAME = 'operating point';

function request_forms(name, type) {
    const kind = String(type).toUpperCase() === 'I' ? 'branch' : 'node';
    const fold = text => text.toLowerCase().replaceAll(':', '.');
    return new Set([fold(name), fold(to_spice_name(name, {kind}))]);
}

/** Indices (row identity) of every vector of `plot` that `name` names. */
export function collect_vector_candidates(plot, name, type = 'V', first = 1) {
    const forms = request_forms(name, type), wanted = String(type).toLowerCase();
    const nvars = plot.nvars ?? plot.vectors.length, found = [];
    for (let i = first; i < nvars; i++) {
        const vector = plot.vectors[i].name;
        if (forms.has(vector.toLowerCase().replaceAll(':', '.'))) { found.push(i); continue; }
        const open = vector.indexOf('(');
        if (open < 0) continue;
        const head = vector.slice(0, open), rest = vector.slice(open + 1);
        if (rest.includes('(')) continue;
        if (forms.has(rest.replace(/\)+$/, '').toLowerCase().replaceAll(':', '.')) && head.toLowerCase() === wanted) found.push(i);
    }
    return found;
}

/** 0 for an operating point, 1 for every plot with an independent axis. */
export function first_row(plot) {
    if (plot.real === false) return 1;
    return (String(plot.plotname || '').toLowerCase() === OP_PLOTNAME || plot.npoints === 1) ? 0 : 1;
}

/** The proven bracket address of vector `row`, or null (see vector_match.py). */
export function vector_address(plot, row) {
    const name = plot.vectors[row].name, alias = to_designer_name(name);
    if (alias === name) return null;
    const match = /^([vi])\(([\s\S]*)\)$/i.exec(alias);
    if (!match) return null;
    const inner = match[2].trim();
    if (!inner || /[(),@#\s'"\\]/.test(inner)) return null;
    const found = collect_vector_candidates(plot, inner, match[1].toLowerCase() === 'i' ? 'I' : 'V', first_row(plot));
    return found.length === 1 && found[0] === row ? alias : null;
}

/** raw name -> collision-safe bracket alias over distinct `names`; `protected`
 * spellings (axes) occupy their raw name. Order independent. */
export function designer_alias_map(names, protected_names = []) {
    const protectedList = [...new Set(protected_names)];
    const own = new Set(protectedList);
    const both = [...protectedList, ...[...new Set(names)].filter(n => !own.has(n))];
    const alias = both.map((n, i) => i < protectedList.length ? n : to_designer_name(n));
    const counts = new Map();
    for (const a of alias) counts.set(a.toLowerCase(), (counts.get(a.toLowerCase()) ?? 0) + 1);
    return new Map(both.map((n, i) => [n, (i < protectedList.length || counts.get(alias[i].toLowerCase()) > 1) ? n : alias[i]]));
}

/** labels='designer': labels and addresses from the COMPLETE source plots. */
function designer_plan(planned) {
    const axes = planned.plots.filter(pp => pp.kind !== 'op').map(pp => pp.source.vectors[0]?.name).filter(n => n !== undefined);
    const every = planned.plots.flatMap(pp => pp.source.vectors.slice(pp.kind === 'op' ? 0 : 1).map(v => v.name));
    const aliases = designer_alias_map(every, axes);
    const labels = {};
    for (const pp of planned.plots) for (const sig of pp.signals) {
        const alias = aliases.get(sig.raw) ?? sig.raw;
        if (alias !== sig.raw) labels[sig.raw] = alias;
    }
    const addresses = planned.plots.map(pp => {
        const proven = new Map();
        return pp.signals.map(sig => {
            if (!proven.has(sig.row)) proven.set(sig.row, vector_address(pp.source, sig.row));
            return proven.get(sig.row);
        });
    });
    return {labels, addresses};
}

/** The checks of emit.py's _validate_labels: a plain object of raw name to
 * non-empty string label, every key a vector name of some plot. */
function check_labels(planned, labels) {
    if (labels === undefined || labels === null) return;
    if (typeof labels !== 'object' || Array.isArray(labels)) {
        throw new TypeError('labels must be an object of raw name -> label');
    }
    const valid = new Set(planned.plots.flatMap(pp => pp.signals.map(sig => sig.raw)));
    for (const [key, value] of Object.entries(labels)) {
        if (typeof value !== 'string' || !value) {
            throw new TypeError(`labels[${JSON.stringify(key)}] must be a non-empty string`);
        }
        if (!valid.has(key)) {
            throw new ViewError(`labels key ${JSON.stringify(key)} does not match any signal in this rawfile; `
                + `available signals include ${JSON.stringify([...valid].slice(0, 3))}`);
        }
    }
}

/** Effective labels per plot: [{index: label}] after the collision plan. */
function plan_labels(planned, labels) {
    const entries = labels ? Object.entries(labels) : [];
    const map = new Map(entries);
    return planned.plots.map(pp => {
        const proposed = new Map();
        pp.signals.forEach((sig, i) => {
            const base = map.get(sig.part ? sig.group : sig.raw ?? sig.name);
            if (base) proposed.set(i, sig.part ? `${base} (${sig.part})` : base);
        });
        for (;;) {
            const shown = new Map();
            pp.signals.forEach((sig, i) => {
                const key = proposed.has(i) ? proposed.get(i) : sig.name;
                if (!shown.has(key)) shown.set(key, []);
                shown.get(key).push(i);
            });
            const drop = [];
            for (const members of shown.values()) {
                if (members.length > 1) for (const i of members) if (proposed.has(i)) drop.push(i);
            }
            if (!drop.length) break;
            for (const i of drop) proposed.delete(i);
        }
        return proposed;
    });
}

/** Assemble a typed-array payload from plan(); resolves to DATA_FORMAT_VERSION 2. */
export async function decimate(planned, options = {}) {
    // `labels` (raw name -> label) is the only opt-in of the labelled path; the
    // bare API is unchanged. Labels enter the identity as the triples of the
    // labels actually shown: [plot_index, final_signal_name, label] in plot then
    // position order, after complex expansion, dedup and the collision plan.
    const {labels: requested, ...decimation} = options;
    const params = decimation_options(decimation);
    // labels: 'designer' is the explicit opt-in that also proves addresses; a
    // plain object (even one equal to the codec aliases) never yields one.
    if (requested !== undefined && requested !== null && typeof requested === 'string' && requested !== 'designer') {
        throw new TypeError("labels must be an object of raw name -> label or 'designer'");
    }
    const designer = requested === 'designer' ? designer_plan(planned) : null;
    const labels = designer ? designer.labels : requested;
    check_labels(planned, labels);
    const effective = plan_labels(planned, labels);
    const address_triples = [];
    if (designer) designer.addresses.forEach((per, p) => per.forEach((address, i) => {
        if (address) address_triples.push([p, planned.plots[p].signals[i].name, address]);
    }));
    const triples = [];
    effective.forEach((shown, p) => planned.plots[p].signals.forEach((sig, i) => {
        if (shown.has(i)) triples.push([p, sig.name, shown.get(i)]);
    }));
    const encoder = new TextEncoder();
    const suffix = encoder.encode('\0' + JSON.stringify([2, params.min_points, params.base, params.min_bucket])
        + (triples.length ? '\0labels' + JSON.stringify(triples) : '')
        + (address_triples.length ? '\0addresses' + JSON.stringify(address_triples) : ''));
    const identity = new Uint8Array(planned.bytes.length + suffix.length);
    identity.set(planned.bytes); identity.set(suffix, planned.bytes.length);
    const [id, state_key] = await Promise.all([
        sha256(identity), sha256(encoder.encode(planned.file_name || '<in-memory>')),
    ]);
    return {
        format: 2, endian: 'le', generator: 'silk.signals.view', generator_version: 'browser-core-2',
        run: {id: id.slice(0, 16), state_key: state_key.slice(0, 8),
            emitted_at: new Date().toISOString(), source: planned.file_name, title: planned.title},
        plots: planned.plots.map((pp, pi) => ({name: pp.plotname, kind: pp.kind, complex: pp.complex,
            npoints: pp.npoints, x: build_x_signal(pp),
            signals: pp.signals.map((sig, i) => {
                const signal = build_data_signal(sig, pp.x_arr, params);
                if (sig.role) signal.role = sig.role;
                if (effective[pi].has(i)) signal.label = effective[pi].get(i);
                if (designer && designer.addresses[pi][i]) signal.address = designer.addresses[pi][i];
                return signal;
            }),
            x_log_default: pp.x_log_default, date: pp.date})),
        default_plot: planned.default_plot, annotations: {markers: [], bands: [], notes: []},
    };
}
