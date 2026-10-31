# Browser waveform core

`silk/signals/view/assets/waveform-core.mjs` is a dependency-free ES module
shipped in the circuitsilk wheel. Copy it directly into a browser application
or import it from Node 20 or newer. It has no imports, build step, or dependency
on `view.js`. The file is MIT-licensed; retain its license header and silk's
MIT license when vendoring it.

```js
import {read, plan, decimate} from './waveform-core.mjs';

// file is a browser File, for example from an <input type="file">.
const raw = read(await file.arrayBuffer(), {file_name: file.name});
const planned = plan(raw);
const payload = await decimate(planned);
```

The host supplies the file and renders the result. The core does no network
I/O. The [standalone explorer](waveform-explorer.md) embeds this module and
accepts its typed-array payloads directly. Its
**Open .raw…** button and file-drop entry run this pipeline offline.

## Exports

| Export | Input and output |
|---|---|
| `read(bytes, {file_name = ''} = {})` | Synchronously parses an `ArrayBuffer` or `Uint8Array` (including a Node `Buffer`). Returns `{plots, bytes, file_name}`. The retained bytes are a snapshot of precisely the supplied view, used later for hashing. |
| `plan(raw)` | Synchronously expands every data vector, deduplicates names, infers plot kinds, and picks the default plot. Returns `{plots, default_plot, bytes, file_name, title}`. It applies no signal selection. |
| `await decimate(planned, options = {})` | Builds the min/max pyramid and resolves to a typed-array payload with `format: 2`. Uses Web Crypto for SHA-256: Node 20+, or a secure browser context such as HTTPS or localhost. |

For display labels, pass a dictionary from raw vector names to non-empty
strings: `decimate(planned, {labels: {'v(out)': 'Output'}})`. Labels that
would collide with another displayed signal name in the same plot are omitted.
For complex vectors the label gains a ` (mag)` or ` (phase)` suffix.
A label changes display text while the raw name remains the signal identity.

Pass `{labels: 'designer'}` to generate collision-safe bracket aliases and
attach an `address` where the source vector can be identified unambiguously.
An address is a bracket-form voltage or branch-current reference usable in a
probe. A dictionary of labels never adds addresses. Omitting `labels`, passing
`null`, or using an empty dictionary adds neither labels nor addresses.
Invalid label arguments or non-string/empty values raise `TypeError`;
a key that names no signal raises `ViewError`.

The module also exports `to_spice_name(name, {style, kind, allow_glob})` and
`to_designer_name(name, {style, kind})`, matching
[`silk.spice_names`](api-reference.md#silkspice_names--the-spice-name-codec).
The defaults are `style: 'auto'` and `kind: 'node'`. A bad `style` or
`kind` raises `RangeError`; a non-string name raises `TypeError`.
For selector conversion, `allow_glob: true` preserves `*` and `?` inside
path segments. The vector lookup helpers `collect_vector_candidates`,
`first_row`, and `vector_address` match `silk.vector_match`;
`designer_alias_map` matches `silk.spice_names.designer_alias_map`.
Python and JavaScript conversion and effective labels are compared by `tests/test_spice_names_js_parity.py`; their `run.id` values
use different hashing rules.

`build_pyramid(x, y, options = {})` is a synchronous export for
aligned level-0 arrays. It returns `{levels, nonfinite}` using the same dtype,
stride ladder, and min/max rules as `decimate`; options use the same names and
defaults. Its level 0 omits x, while each decimated level has its own extrema
positions. It does not parse, hash, or change a payload. Keep its input arrays
unchanged while the result is in use. The explorer uses this helper for
[derived signals](waveform-explorer.md#derived-signals).

Each reader plot includes `header`, `name`/`plotname`, `title`, `date`, `flags`,
`real`, `padded`, `offset`, `nvars`, `npoints`, and ordered `vectors`.
Each vector has `name`, `type`, `attrs`, and `data`. Real `data` is a
`Float64Array`; complex `data` is `{re: Float64Array, im: Float64Array}`.
For waveform plots, vector zero is the independent axis. A real plot whose
`Plotname:` is `Operating Point` (case-insensitive) has display kind `op`
and retains **all** vectors as signals, including vector zero. Its `x` is a
synthetic zero-based `sample` index, with empty type/unit, used only to align
arrays. Complex plots have kind `ac`; plots with `constants` in their name have
kind `constants`. A single sample alone does not give a plot kind `op`.
Negligible imaginary scale components
are discarded with the Python reader's relative tolerance and smallest-normal
float64 floor. `Offset:` is already applied to that axis. When the reader
retains a genuinely complex scale, `plan()` uses only its real component for x.

Headers use a lossless Latin-1 byte map, including bytes `80`–`9F` (browser
`TextDecoder('latin1')` maps those through Windows-1252 instead). Binary values
are little-endian float64 or interleaved complex128 in point-major order.
ASCII `Values:` and multiple plots are supported. Truncated data retains
complete points; a zero-point plot remains empty. A vector marked `dims=0`
remains present with empty data and one empty payload level, even when its
plot contains points. Unrecognizable files raise `RawfileError`; some invalid
header fields are tolerated: missing/invalid point counts and invalid offsets
default to zero. Bad API arguments throw `TypeError`/`RangeError`.
`read()` with no recognizable plot throws `RawfileError`; `plan()` with no
plots and unavailable Web Crypto during `decimate()` throw `ViewError`.

The plan's plots retain full-precision `x_arr` and `signals[].y_arr` arrays.
Complex data becomes magnitude and unwrapped phase in degrees; nonfinite
phase samples do not poison later samples. No selection, annotation input,
context, or scopes are handled by these functions. The final payload contains
the schema's empty `annotations: {markers: [], bands: [], notes: []}`.

## Payload and decimation

The core emits format 2. For an explicit operating-point plot, every row is
a signal; the synthetic sample x is only an alignment axis. Other plots use
vector zero as x, including one-point transient, DC and noise plots.
The core does not infer `role: "coordinate"` or restore those rows as signals.

The default plot is the last non-constants plot with signals. If none exists,
the planner tries the last plot with signals, then the last non-constants plot,
then the last plot.

Keep the copied core and explorer versions compatible. The explorer accepts
formats 1 and 2 on initial load, reload and scope changes, with base64 or
typed-array numeric leaves. Legacy payloads display the data they contain;
re-emit them to recover an omitted operating-point row.

```js
const payload = await decimate(planned, {
    min_points: 8000,
    base: 8,
    min_bucket: 1000,
});
```

All three settings must be safe integers: `min_points >= 0`, `base >= 2`, and
`min_bucket >= 1`. Unknown settings raise `TypeError`; invalid values raise
`RangeError`. The separate `labels` option is accepted only by `decimate`.
The pyramid has at most eight
levels, including the full-resolution level. The defaults match Python's
`emit_view_data`; its keywords have the additional `decimate_` prefix.

`plots[].x.levels[0].x` is a `Float64Array`. Signal level zero contains `y`
and shares the plot's X array. Higher levels carry their own `Float64Array`
`x` and typed `y`; each signal can select different X positions. Each signal
level's `dt` is `f32` unless a finite magnitude exceeds `3e38` or a nonzero finite
magnitude falls below `1e-37`, in which case every level uses `f64`.
Decimation chooses extrema from the original float64 values before converting
Y. All-NaN buckets emit one point; other buckets emit both extrema in original
sample order, including duplicate indices on ties.

The module follows Python's NaN/Inf tie behavior:
`nanargmin`/`nanargmax` replace NaNs with positive/negative infinity for their
respective comparisons. A leading NaN can therefore tie with an infinite
extremum. This behavior is included in the differential checks.

Payloads are suitable for `structuredClone`/Worker messages. Multiple leaves
may share buffers: deduplicate a transfer list by `ArrayBuffer` identity and
finish hashing before transferring. Treat reader/plan objects as immutable
until `decimate` resolves; release them afterward if only the payload is needed.
Returned x arrays and float64 level-zero y arrays can share storage with the
plan; keep those arrays unchanged while the payload is in use.
The core does not encode leaves as base64 or serialize them as JSON arrays.

## Run identity

`run.id` is the first **16 lowercase hexadecimal digits** of:

```text
SHA-256(raw file bytes || UTF-8("\0" + JSON.stringify([2, min_points, base, min_bucket])))
```

When labels are shown, the identity bytes continue with
`"\0labels" + JSON.stringify([[plot_index, final_signal_name, label], ...])`,
one triple per displayed label in plot then position order (after complex
expansion, name dedup and the collision rule); with none, nothing is appended.
When executable addresses are present, the next suffix is
`"\0addresses" + JSON.stringify([[plot_index, final_signal_name, address], ...])`.
Both suffixes use effective payload fields in plot/signal order and are omitted
when empty. Equal labels with different addresses therefore have different IDs;
requesting a label mode with no effective label or address does not change the ID.
The zero byte separates the file from the versioned parameter tuple. Default
and explicitly defaulted options hash identically, regardless of option-key
order. Changed bytes or parameters change the ID. The supplied filename and
generated `run.emitted_at` are excluded; timestamps inside the rawfile remain
part of the hash. This is intentionally different from Python's identity over
the base64 payload.

`run.state_key` is the first **8 lowercase hexadecimal digits** of
`SHA-256(UTF-8(file_name))`, with `'<in-memory>'` as the seed when the name is
empty. Supply `File.name` in browsers. A consumer that wants view state to
survive filename changes should replace the result with its stable
project/module/testbench key. `run.source` is the supplied filename;
`run.title` uses the default plot's title, then the final
slash-separated component of the filename, then `rawfile`.

## Differential gate and vendoring

Run these from the silk checkout using its Python environment and Node 20+:

```bash
.venv/bin/python tests/js/dump_waveform_expected.py --output /tmp/silk-oracle
node tests/js/differential.mjs --module silk/signals/view/assets/waveform-core.mjs --expected /tmp/silk-oracle
```

For a vendoring consumer, substitute the copied module's absolute path for
`--module`. The Node harness needs only Node built-ins, the module, and the
dump directory; it can run from any working directory. The dump includes
every `tests/fixtures/rawfiles/*.raw`, plus generated boundary cases, with
default and two forced-decimation parameter sets. No simulator is needed.
`--fixtures /path/to/rawfiles` selects another corpus for the Python dump.

The harness compares reader metadata and numeric data, then the entire
payload. It decodes Python's base64 leaves into typed arrays, checks dtypes,
and compares every element with `Object.is`: NaN equals NaN and negative zero
differs from positive zero. Only the values of `run.id`, `run.emitted_at`,
`run.source`, and `generator_version` are excluded; even those keys must
remain present. Identity tests separately check the JS hashing rules.

```bash
.venv/bin/python -m pytest -q tests/test_view_js_differential.py -m 'slow or not slow'
```

The pytest wrapper **fails when Node is absent**. Developers knowingly
working without Node may set `SILK_ALLOW_NO_NODE=1`; no other value opts out.
Merge gates and consumer CI should install Node and leave this opt-out unset.
The optional browser smoke test uses the `SILK_FIREFOX` or `SILK_CHROME`
setting, resolved from the environment or silk's settings file. Set one to a
headless-capable executable; otherwise the test skips. A browser on `PATH`
alone is not selected. See [settings](settings.md) for configuration and
`.venv/bin/python -m silk.doctor` to inspect the resolved paths.
The Node gate remains required independently.

The generated `f_nonascii_title.raw` contains both `C2 B5` and a lone `E9` in
its title, plus Latin-1 control-range bytes. Reproduce it with:

```bash
.venv/bin/python tests/fixtures/rawfiles/gen_browser.py
```


### Python-produced node metadata

Checked circuit metadata is produced only by Python and consumed by the
explorer. The browser preserves the incoming Python `run.id`; raw names and
state keys remain identities. Opening a local rawfile discards previous
`node_paths`, `source_row` and `node_probe` fields. The local-raw `decimate`
hash includes only the raw bytes, decimation options, effective labels and
addresses described above; the core does not produce circuit-graph metadata.
Hosts import the shared textual selector API from `selectors.mjs`
(see [integration](explorer-integration.md#checked-designer-node-metadata)).
