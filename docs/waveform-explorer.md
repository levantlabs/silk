# Waveform explorer

The standalone explorer displays nutmeg waveforms with a signal picker,
multiple panels, zoom, cursors, markers, and saved layouts. The
[CLI guide](cli.md#silksignalsview--waveform-explorer) covers emitting a page
from Python and selecting signals or scopes. To place the same complete
explorer inside another application, see the
[host-integration guide](explorer-integration.md).

## Default layout

Open an emitted `view.html` in your browser. A waveform run (AC, transient,
DC or noise) starts with one empty panel. Drag a signal from the picker onto
it, double-click a signal, or select signals and click **Add** to plot them.
Use **+ panel** to add another panel and **f** to fit all plotted signals.
Drag across a waveform to zoom; **0** resets X and **f** fits X and Y. Hover
over a waveform and press **a**/**b** for cursors or **m** for a marker; their
positions snap to samples. **Export** downloads plotted traces as CSV for the
visible, A–B or full range, or all panels as PNG; a panel's Export menu exports
that panel only.
An operating point starts with one filled panel per unit, as described below.

A saved layout from a previous session, a `.silkview.json` file, or a host's
`stateKey` restores its surviving signals. Ordinary waveform layouts drop
empty panels and fall back to the default layout when none survive. Only
operating-point layouts preserve empty panels and an empty panel list. These
defaults also apply when there is no saved layout. Hosts can populate panels
with the `addSignal` and `addPanel` commands.
Use **View → Save view** to download a `.silkview.json` layout, and
**View → Load view** to restore it; this file contains the view settings,
not the waveform samples.

## Operating points and one-point results

**An operating point** (a real, non-complex rawfile plot named `Operating Point`,
which has no x axis; a complex plot with that name is an AC plot) is drawn inside the ordinary panels as dot-plot strips, not as a chart. A
fresh one opens with one panel per unit (voltages, then currents, then any other
unit), each holding every signal of that unit; a saved layout is used as saved,
even an empty one. A panel draws one strip per unit it contains: each row shows
the name, a stem from 0, a dot and the value with its unit; hovering the dot
gives the stored value, unrounded. The voltage axis includes 0 V and all finite
voltages, with dashed reference lines at 0 V and at the highest net
when its value is nonzero (an all-zero range is padded); the current axis is symmetric about 0 and its heading
states the sign convention. Each strip's range comes from every signal of the
plot with that unit, so adding or removing a panel signal never rescales it.
Hover highlights a row in the strip and in the picker. Each picker section
counts the signals plotted in panels. Signal colours match the traces and
legend, including any colour override.

The panels work as for waveforms: click selects a signal, double-click, Add and
Plot visible add to the focused panel (dropping several signals on a panel tags every new row `added`), drag and drop targets a panel, Clear
panel empties it, `+ panel` adds one, `1 col` / `2 cols` lays them out. A newly
added row carries an `added` tag that fades after about 2.5 s, and rows slide
when the order changes. Each panel header has a **Sort** select: by value
(default), by name or in the order added; it is saved per panel as `sort`.
Strips have fixed axes, so a panel header has no Y-mode or autoscale and the
style popover offers colour only. `X: lin` and `Samples` are disabled (no x
axis, one point) and the A/B cursor keys do nothing.

**Markers** on an operating point are named, draggable reference levels on one
unit's strip axis. `+ marker` (or `m`) uses the strip last hovered; the level is
not snapped. Every strip of that unit draws the marker, and each of its rows
shows its difference from the unit's **reference** marker (a `Δ vs <name>`
label names it). The newest marker added of a unit is its reference; dragging
a marker or editing its level makes it the reference; "Use as reference" in the tab's
right-click menu or the markers table switches it; deleting the reference makes
the most recently added remaining marker of that unit the reference. The
markers table (it stays current while open, as panels change) lists every marker against every signal plotted in any panel,
showing `signal - level` where the units match and a dash where they do not.
Saved markers carry `unit` beside `x` (the level) and `ref`; operating-point
markers are kept in insertion order, waveform markers stay sorted by x. The
commands are `addMarker({x, unit, name?})`, `setMarkerX({name, x, unit?})` and
`setMarkerReference({name})`. `addMarker` requires `unit` on an operating point
and refuses it on a waveform; `setMarkerX` keeps the marker's unit and accepts
`unit` only if it matches.

**Export** composes the strips for the panel and all-panels PNG (sized from the strips, which are at least 520 px wide, so a narrow panel is not cropped). The CSV has one
row per signal (all panels: one row per distinct plotted signal) with the
columns `signal`, `value`, `unit` and one `delta:<marker> (<unit>)` column per
marker, empty where the units differ; `visible` and `full` are the same file and
`cursor` is refused ("cursor export needs an x axis; use visible or full").

**One-point DC, AC, transient and noise runs** keep their real x axis and are
ordinary charts: the single sample is always drawn as a point (whatever the
Samples toggle or style says) on a padded x range. A waveform whose finite
full-resolution x values are all exactly equal is a chart on a padded domain too.
A nonzero span, however small, remains a waveform; chart ranges are padded
when necessary to keep tick positions distinct.

Copy-as-code is available from the picker. A genuine voltage named like an axis
(for example `v(time)`) remains an ordinary, probe-able signal. The display kind
`op` does not override the analysis declared by the rawfile header.

Current payloads use `DATA_FORMAT_VERSION = 2`. The explorer also accepts
legacy format 1, but it cannot recover a source vector that an old producer
omitted. Re-emit the rawfile for that value.

## Signal list sections

The picker groups signals into one collapsible section per kind (keyed on
the payload's signal type): **Voltages**, then **Currents**, then any other
kind (shown in the hierarchy tree without a separate section), then
**Derived signals** last. Voltages opens and Currents
starts closed; a closed section's header shows its signal count and how
many of them are already plotted, e.g. `Currents (3 · 1 plotted)`. A single
click on a section's twisty opens or closes it -- collapsing only hides rows
in the list, it never affects what is plotted. Inside a section, signals
keep the ordinary hierarchy tree (a nested current still sits under its own
subcircuit group). Pass `groupSignalsByKind: false` to `createExplorer`, or
`emit_view(..., group_signals_by_kind=False)` for a generated page, to
restore one plain, unsectioned list; see
[host-integration guide](explorer-integration.md) for both.

## Bode plots and multiple Y axes

Overlay magnitude and phase in one panel to make a Bode plot. Each distinct
signal unit automatically gets its own Y axis and autoscaled range. Add the
magnitude first for volts on the left and degrees on the right; choose **dB**
in the magnitude axis row for a dB/phase plot. The shared **X** control sets
frequency to logarithmic or linear.

Groups follow the first appearance of each literal unit string in the panel's
signal order. Exactly two groups use left and right axes; three or more use
axes stacked on the left in that same group order. For example, `V`, `A`,
and `deg` produce three independent axes. An empty unit is a separate group:
its ticks have no unit suffix and its controls say **no unit**. Several
unknown types share that empty-unit group. A panel containing only one unit
keeps its existing appearance and controls.

With multiple axes, the rows below the chart identify each axis and provide
its range, **linear/log/dB** selector, and **Y↺** autoscale button. dB is offered
only when every signal in that group is a magnitude; a phase or real-valued
signal on another axis does not disable it. Multi-axis dB autoscaling and tick
precision preserve small magnitude changes near unity. Log and dB manual
ranges must be positive; switching to either mode clears an incompatible
manual range.

Type either range endpoint to edit just that axis. A Y drag, box zoom, or
Shift-wheel transforms every axis using its own coordinates. The header's
**Y↺** and **Shift-0** reset all Y axes in that panel; **f** fits all panels and
the shared X range. Each row's **Y↺** resets only that row's axis.

Cursor values, statistics, and range endpoints use each signal's original
unit, including when its axis ticks show dB. CSV exports contain the original
per-signal samples. Signal-specific Y markers and bands use that signal's
axis; a Y annotation without a signal uses the first axis. PNG exports include
all axes and annotations.

**Sample dots.** The **Samples** button (key **s**, off by default) draws a
dot at every simulated sample of each plotted trace in the visible X window,
so timestep density and large steps are visible. It is display-only, shared
by every panel, and survives zoom and pan. While the samples in view would be
closer than 4 px apart the dots are hidden and a note on the plot gives the
sample count; zoom in to see them. It is not saved with the view.

Saved views store an ordered `yAxes` array per panel, with each group's `unit`,
`signals`, `yMode`, and `yRange`. Reload preserves the groups' settings;
assignments are reconciled against the current signals and their units, so
missing groups are pruned and new groups autoscale. Older views without
`yAxes` still load: groups are derived from the saved signal order, the old
mode applies where eligible, and the old manual range belongs to the first
axis. The legacy `yMode` and `yRange` fields continue to describe the first
axis when saving.

The existing command bus accepts an optional literal `unit` in `setYMode`,
`setYRange`, and `autoscaleY`, including `unit: ''`. Without it, `setYMode`
and `setYRange` target the first axis, and `autoscaleY` resets all axes in the
panel. For example, with a mounted `explorer` instance and the named AC signals:

```javascript
await explorer.run([
  {cmd: 'clearPanel', args: {panel: 0}},
  {cmd: 'addSignal', args: {panel: 0, signals: ['v(out) (mag)', 'v(out) (phase)']}},
  {cmd: 'setYMode', args: {panel: 0, unit: 'V', mode: 'dB'}},
  {cmd: 'setYRange', args: {panel: 0, unit: 'deg', min: -180, max: 0}},
  {cmd: 'setXLog', args: {on: true}},
]);
```

## Open a local rawfile

Open an emitted `view.html`, then click **Open .raw…** in the toolbar or drop
one `.raw` file anywhere on the page. The page embeds the same
[`waveform-core.mjs`](browser-waveform-core.md) asset shipped for browser
consumers. It reads and displays the selected file locally.

This works offline, including when opening `view.html` directly with a
`file://` URL. No Python process, server, rawfile upload, or module download
is involved in opening the new file. Both sidecar and `--inline` pages carry
the core; a sidecar page still needs its emitted data file for its initial
load. Use a current browser with ES modules and Web Crypto. For served
pages, Web Crypto requires HTTPS or localhost.

Sidecar pages poll the viewer files, not the rawfile; re-emit them after
simulation, or use the CLI's `--serve` watcher. An inline page needs its HTML
regenerated and reloaded.

Opening a rawfile replaces the current run and shows all its signals. The
original page's signal selection, context, scopes, and embedded view preset
belong to that original run. Local files use their own saved layouts. The
page stops polling the original sidecars, clears its scope selection, and
shows **local file — reopen .raw to refresh**. Reopen the file after a
simulation changes it. Reloading the page returns to its original emitted
run; open the local file again to restore its saved layout.

The core's [identity rules](browser-waveform-core.md#run-identity) apply:

- `run.id` hashes the exact file bytes, format version and resolved decimation settings, plus effective labels and addresses when present.
  Reopening identical bytes under the same filename leaves the live charts
  and layout untouched.
- `run.state_key` hashes `File.name`. New contents under the same filename
  load the same saved layout, pruning signals that no longer exist. A
  different filename gets a separate layout, even for identical contents.
  Files from different directories with the same name share this key;
  browsers do not expose their full paths.

Only the latest file selection can replace the displayed run. Module errors,
such as a malformed or aborted file with no complete plot header, appear in
the explorer's fatal notice with their error name and message. Drop a valid
`.raw` file to recover, or reload the page. Truncated data retains complete
points, and a declared zero-point plot shows a no-data hint.

Dropping a `.silkview.json` file still loads a saved view. Internal signal
and panel drags continue to work. Reading, planning, and decimation currently
run on the browser's main thread; large files can briefly pause interaction
and require memory for the bytes and decoded arrays.

<a id="format-1-two-leaf-encodings"></a>

## Payload encodings

For hosts consuming payloads directly, both current format-2 and legacy
format-1 payloads support these numeric array encodings:

| Leaf | Python `emit_view` / `data.js` | Browser core |
|---|---|---|
| `plots[].x.levels[0].x` | Base64 little-endian float64 | `Float64Array` |
| `signals[].levels[].y`, `dt: 'f32'` | Base64 little-endian float32 | `Float32Array` |
| `signals[].levels[].y`, `dt: 'f64'` | Base64 little-endian float64 | `Float64Array` |
| Higher signal levels' `x` | Base64 little-endian float64 | `Float64Array` |

`explorer.mjs` accepts either form at `decodeF64` and `decodeY`, returning typed
arrays without copying an already decoded leaf. Every downstream consumer
uses the same decoded representation and existing nonfinite-gap handling.
Level zero shares the plot's X array; decimated signals retain independent
X positions, combined by the existing union-X logic when overlaid. The
browser payload is suitable for structured cloning, not JSON serialization.
Python's existing payload-based run ID intentionally differs from the core's
file-byte-based ID; identical numeric data does not imply identical run IDs.

The optional browser checks compare both producers and exercise opening local
files. Install the [browser dependencies](../tests/browser/README.md), then
run each file from the repository root:

```bash
.venv/bin/python -m pytest -q tests/test_view_e2e.py -m 'slow or not slow'
.venv/bin/python -m pytest -q tests/test_view_open_raw_e2e.py -m 'slow or not slow'
```

## Derived signals

Click a signal in the picker to select it, Ctrl/Cmd-click to add or remove one,
Shift-click for a range, Escape to deselect everything,
or Space on the focused row, then **right-click a selected row** to derive a
trace. With the picker focused, **Shift+F10** or the context-menu key opens the
same menu. Every entry names its actual operands:

| Selection | Menu entries |
|---|---|
| One signal, A | Abs \|A\|; derivative d(A)/d(x); integral ∫ A d(x); scale gain × A + offset |
| Exactly two signals, A and B | Sum A + B; difference A − B and B − A; product A × B; quotient A / B and B / A; arithmetic mean (A + B) / 2 (common mode) |
| More than two signals | Sum, product and arithmetic mean of the whole selection |

Sum, difference and mean are disabled when the selected signals have different
literal units. A visible explanation lists the signals and their units; product
and quotient remain available. **Scale** opens a small form in the menu with
`gain` (default 1) and `offset` (default 0); Tab moves between the fields,
Enter creates the trace, and a value that is not a finite number keeps the form
open with a message. Right-clicking an unselected signal selects it and opens
the explorer menu; empty list space and headers retain the browser menu.
Arrow keys move between enabled entries, Enter
chooses one, and Escape dismisses the menu and returns focus to the picker.
Clicking elsewhere inside the explorer also dismisses it. Right-clicks outside
this explorer are left to the host page.

Choosing an operation adds the result to the **focused panel**, using the same
placement rule as **Add selected**: fall back to the first panel, or create one
if no panels exist. The operand selection clears after creation. Select a trace
from **Derived signals** together with another signal to compose operations.
Choosing an existing definition reveals that same trace in the target panel;
it creates neither another definition nor a duplicate panel entry. Definitions
travel with [saved views](#saved-derived-definitions), even when they are not
plotted in a panel.

The following JavaScript examples run in the browser console of an emitted
page, where `silkView` is available. For a mounted explorer, use the instance
returned by `createExplorer` as `explorer` instead. In a real-valued plot containing
`v(out)` and `v(in)`:

```javascript
const result = await silkView.run({
  cmd: 'deriveSignal',
  args: {operation: 'difference', operands: ['v(out)', 'v(in)']},
});
if (result.ok) {
  const {name, unit, created} = result.results[0];
  await silkView.run({cmd: 'addSignal', args: {signals: [name]}});
}
```

A mounted explorer exposes the same command through `explorer.run(...)` and
requires a matching `waveformCore` option with `build_pyramid`. The optional
`plot` integer selects a plot by index; omitting it uses the active plot. All
operand names must exist in that plot. Creation returns `{name, unit, created}`;
it adds the signal to the **Derived signals** picker group.
The command alone does not add it to a panel. Derived names are displayed
literally, including dots in operand names, in tree, flat and filtered views.
Add them using the picker or ordinary `addSignal`/`addPanel` commands.

A host can supply `name: 'math1'` in the command's `args`. The legend, picker
and value table then show `math1`, with the canonical expression available
by hovering the name. Complex results show `math1 (mag)` and `math1 (phase)`;
`abs` still produces one real trace. Names are literal, case-sensitive strings:
empty/whitespace-only names, `null` and non-strings fail with
`deriveSignal name must be a non-empty string`. Valid names are not trimmed.
There is no trace rename control; change the name where the math is defined.

Repeating a name with the same **normalized definition** returns
`{name, unit, created: false}`. Complex part aliases, commutative operand
ordering and default scale parameters normalize exactly as in unnamed math.
The same expression under different names creates separate traces. A name
collision reports both names and the plot, for example:
`deriveSignal name "math1" conflicts with existing signal or trace "math1 (mag)" in plot 0`.
Explicit names never receive an automatic collision suffix. Omitting `name`
retains expression naming, suffix handling and unnamed deduplication.

Use `replace: true` to redefine an existing explicitly named derived trace:

```javascript
await silkView.run({cmd: 'deriveSignal', args: {
  name: 'math1', operation: 'scale', operands: ['v(out)'],
  params: {gain: 2}, replace: true,
}});
```

`replace` defaults to `false`; non-booleans fail with
`deriveSignal replace must be a boolean`. If the name is unused, this creates
it. An identical normalized definition remains an idempotent no-op; a changed
definition returns `created: true`. Raw signals, unnamed derived traces and
another trace's complex parts cannot be replaced. Without `name`, `replace`
has no target and the unnamed behavior remains unchanged.

Replacement recomputes every direct and transitive derived dependent in
dependency order, including unnamed dependents, preserving their definitions
and names. If any dependent becomes invalid, the command fails with that
descendant's name and reason; the entire replacement rolls back, including
samples, identities, panels and saved definitions. Unrelated traces are
untouched. Cyclic replacements are rejected.

Panels already showing the replaced trace or its dependents update their data.
A real-to-complex change replaces the entry with both magnitude and phase;
a complex-to-real change replaces either/both part entries with a single real
entry. Same-shape changes retain the displayed subset and order. These rules
also apply to saved panels on inactive plots, and axes regroup by result unit.

| Operation | Operand count | Unit rule |
|---|---|---|
| `sum` | Two or more | All literal units must match |
| `difference` | Exactly two, first minus second | Both literal units must match |
| `product` | Two or more | Multiply units; for example, V × A → W |
| `quotient` | Exactly two, first divided by second | Divide units; V / V is dimensionless, V / A → Ω |
| `mean` | Two or more | All literal units must match; two operands give common mode |
| `scale` | Exactly one; parameters `gain` (default 1.0) and `offset` (default 0.0) | `gain * a + offset`; `gain` is a plain number, so the unit is kept |
| `abs` | Exactly one | \|a\| (`Signal.mag`); unit kept |
| `derivative` | Exactly one, at least 2 samples | d(a)/dx (`Signal.derivative`); divides by the x unit, for example V/s |
| `integral` | Exactly one, at least 1 sample | Cumulative trapezoid from 0 (`Signal.integral`); multiplies by the x unit, for example A × s → C |

These names, operand counts, parameters and maths are silk's own `Signal`
operations in `silk/signals/base.py`, so a derived trace equals what a test
recomputes in Python: `derivative` uses central differences on the actual,
possibly non-uniform, sample grid with one-sided differences at the ends
(`numpy.gradient`), and `integral` starts at `y[0] = 0`. There is no dB
operation: use the dB Y-axis mode for AC magnitudes.

On complex AC plots, operands name **whole complex vectors**. `v(out)`,
`v(out) (mag)` and `v(out) (phase)` all resolve to the same operand. Every
operation above computes on real and imaginary components; `scale` adds its
real offset only to the real component. Derivative and integral use the
frequency axis. A complex result returns its group `name` in the unchanged
`{name, unit, created}` response and creates two drawable traces:
`name + ' (mag)'` (result unit) and `name + ' (phase)'` (unwrapped degrees).
Add those part names with `addSignal`/`addPanel`; the context menu adds both.
`abs` returns one real magnitude trace, with the returned `name` directly
addressable. Subsequent operations on that real trace stay real unless mixed
with a complex operand. Chained complex results accept their group or either
part name too. Unit checks use the vector's magnitude unit, even when an
operand is named by its phase part.

Complex samples are reconstructed from the existing level-0 magnitude and
phase; derived rectangular samples are retained internally for chaining, so
a singularity does not erase later valid complex values. This preserves the
payload's float32 rounding. Derivatives and nearly
cancelling expressions can amplify that rounding; use Python measurements
when the numerical precision matters.

Saved views store one definition per complex result and reconstruct both parts,
including chained definitions. Copy-as-code for a derived magnitude or phase
applies `.mag()` or `.phase_deg_unwrapped()` **after** the complex expression;
its operands remain complex probes. For example, a difference's magnitude is
`(tb.probe('out') - tb.probe('in')).mag()`.


Pass parameters as `params`, for example multiplying a current trace by 1e6:

```javascript
await silkView.run({cmd: 'deriveSignal',
  args: {operation: 'scale', operands: ['i(vdd)'], params: {gain: 1e6}}});
```

`params` is accepted only by `scale`; an unknown key, a value that is not a
finite number, or parameters on another operation is an error. Omitted
parameters take their defaults. `scale` keeps the operand's unit label, so a
gain of 1e6 on amps still displays A: the gain is a number, not a unit change.
An operation that cannot apply, such as the derivative of a one-sample trace,
is refused with an error naming the trace and its sample count; it never
creates an empty trace.

Product and quotient display an empty unit when the result has no unit in the
viewer vocabulary. Derivative and integral name a vocabulary unit when one
matches (A × s → C, 1/s → Hz); otherwise they show the operand and x units
composed, such as V/s, and an empty unit when either unit is unknown. Dimensionless products and quotients display an empty unit. That empty
string forms its own Y-axis group. Known dimensions are retained internally for
chained derivations, so `(V × V) / V` recovers V, and dimensionless gain times A
recovers A. An ordinary signal with an empty unit remains unknown. Sum,
difference and mean compare the literal unit strings, including empty strings;
matching empty labels do not establish a physical equivalence between unknown
quantities.

Definitions deduplicate by operation, operands and parameters: operand order is ignored for
sum, product and mean, and preserved for difference and quotient. Repeating an
operand counts (A + A differs from A + B). Recreating a definition returns its
existing name with `created: false`. Names are generated from the operation and
operand names, plus every `scale` parameter (`scale(i(vdd), gain=1000000, offset=0)`),
with a numeric suffix when another signal already has that name.

Arithmetic uses only the plot's aligned level-0 arrays. The result gets a fresh
min/max pyramid; no operand's decimated values enter the calculation. Dtype and
non-finite counts come from the result itself. Division by zero follows ordinary
floating-point arithmetic: NaN/Infinity remain non-finite, draw as gaps, and are
excluded from finite-value statistics. Invalid operations, counts, units, names
or plot indices return the command bus's actionable error and create no trace.

Derived traces work with the usual panels, cursor readouts, viewport statistics,
CSV data export and PNG image export. Their definitions travel with
[saved views](#saved-derived-definitions), including traces still only in the
picker. There is no rename command, expression parser,
cross-plot/cross-scope math or special phase wrapping in this version.

These browser values never enter a report, spec outcome, metric extraction or
stored database evidence. Cursor/statistics displays and exports are readouts.
When a derived quantity matters to a design decision, define its measurement in
Python so it can become evidence through the normal measurement workflow.

## Saved derived definitions

Saved layouts in `localStorage`, `getState`, and the `view` block of a
`.silkview.json` file share the same additive per-plot `derived` array:

```json
{
  "v": 1,
  "plots": {
    "0": {
      "derived": [
        {
          "name": "difference(v(out), v(in))",
          "operation": "difference",
          "operands": ["v(out)", "v(in)"]
        }
      ],
      "panels": [
        {"signals": ["difference(v(out), v(in))"], "yMode": "linear"}
      ]
    }
  }
}
```

Each definition stores its exact signal `name`, `operation`, and ordered
`operands` (signal names in the same plot). A `scale` definition also stores
`params` with both `gain` and `offset`; other definitions omit it. Commutative operands use the
engine's canonical order. Units, dimensions, samples, and decimation pyramids
are recomputed from the current payload; they are not stored in the view file.

Explicitly named definitions additionally store `given_name`, equal to their
base `name` (for example, both are `"math1"` for a complex pair). Unnamed
and legacy definitions omit this field. Restore passes it through the same
name validation and creation path as the command; it never infers a given
name from the legacy display-name field. A mismatch is an invalid definition.
Restored names work identically with idempotent repeats and `replace: true`,
including `reload()`, a new instance sharing the `stateKey`, `loadViewState`
and `.silkview.json` imports.

Definitions rebuild in dependency order **before panel names are resolved**.
The array need not be sorted: a derived operand can occur later in the list.
Saved names, including collision suffixes, remain stable. Definitions are
saved even when absent from panels or created on an inactive plot.

A missing operand, dependency cycle, invalid operation or unit combination,
conflicting name, missing plot, or host without the required waveform core
causes the affected definition to be dropped with a visible toast naming the
problem. Definitions depending on a dropped definition are also dropped;
valid independent definitions and panel signals still load. Existing panel
pruning reports stale panel references as usual.

Older views without `derived` load as before. The view and envelope versions
remain `1`; older viewers can ignore this additive field and apply their usual
pruning to unknown panel signals. Loading a view replaces its definitions,
and **Reset view to defaults** clears them. Embedded sidecars follow the
existing precedence and **use my saved view** restores the prior definitions.

Reloading a payload rebuilds retained definitions against the new samples.
Opening another local `.raw` file selects that filename's state, so definitions
from the previous file do not carry over. Reopening the original filename
restores its definitions when its operands still exist. For mounted explorers,
a stable host `stateKey` enables persistence across fresh loads; omitting it
keeps fresh loads unpersisted, while `reload` retains the current view in memory.

### Xyce hierarchy selectors

`emit_view`, `emit_view_data`, `tb.netlist.view(signals=..., exclude=...)` and
`python -m silk.signals.view` with `--signals` / `--exclude` accept the same
bare hierarchy selectors on ngspice and Xyce rawfiles. For example, `xx1[0].xxy[1].g[0]`,
`xx1[0]`, and `g[0]` match `V(XX1<0>:XXY<1>:G[0])`. Matching is
case-insensitive, and instance indices can use `[k]` or `<k>`. Node leaves
remain literal: `g[0]` does not select `g<0>`.

Python selection and the browser tree both split raw hierarchy on `.` or `:`
outside square brackets. Bracket contents and raw vector identities are
preserved. Wrapped selectors keep full-name matching, including the raw
separator: `v(xx1[0]:xxy[1]:g[0])` selects that Xyce vector; `v(g[0])` does
not select a nested vector. Block discovery and `max_depth` use the same
hierarchy on both simulators.

A colon always separates hierarchy levels on both simulators; it is not a
literal character in a silk net, port or instance name: silk rejects such
names at construction. External rawfiles with colons are interpreted as
hierarchy by the viewer.

### Bench-rooted designer paths

After a successful run, `tb.netlist.view(signals=['dut.x1[0].mid'])` (or
`mod.view(...)` for a directly simulated Module) selects and displays checked
paths such as `v(dut.x1[0].mid)`. `exclude=` accepts the same selectors. Wired
ports prefer the connected top-level name. Copy as silk code preserves the
checked authored case and punctuation and retains the analysis/complex-part
selection. Different physical targets with the same exact designer spelling
are refused; emitted matches keep precedence. The same node in transient and
AC plots is not a collision.

Independent-source current cards also get checked labels and copyable probes,
with array copies shown as `[k]`, for example `i(Vstim[0]__silk_dc)`.
In the hierarchy list, including inside the Currents section, a cascaded
source's `__silk_wave` (or `__silk_pwl`) and `__silk_dc` cards sit under one
plottable source row, collapsed by default. That row reads the source's logical
current, just like `tb.probe('i(stim[0])')`: a voltage source uses its series
branch and a current source adds its parallel card currents. Copy as silk code
and host trace-menu items receive that source probe. Expand the row to inspect
the cards, which keep their raw identities and checked card probes. Flat mode
lists the source current and every card separately.
Behavioral results only label currents actually present in the result; they
do not invent voltage-source branch currents.

Checked labels require the successful result's address record and unchanged
parsed file bytes. External files, unsupported result routes and stale graph
associations omit checked labels; unavailable retained run files require
restoring the outputs or rerunning. Raw signal identities never change. The
picker's text search remains substring search. Python hosts and browser hosts
can use the [checked metadata and shared selectors](explorer-integration.md#checked-designer-node-metadata)
for programmatic selection. Excluding waveform rows does not hide their names
in metadata. Metadata exceeding its 8 MiB cap or the available payload budget is normally
omitted with a warning, retaining the selected data and legacy display. If
that fallback would discard a selected logical source current, emission raises
`ViewError` instead. Reduce the exported metadata, or increase `max_bytes`
when the payload budget is the limiting factor.
