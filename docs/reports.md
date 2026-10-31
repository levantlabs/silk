# Circuit reports — `silk.circuit.report`

`generate_report(module)` returns a `CircuitReport` describing a module's
current topology, port annotations, lint findings, reconciled specs and loaded
simulation results. It runs lint checks but does not run simulations. Call it
again after changing the circuit or loading new results. Generating a report
also reconciles specs and may apply declared calibration bindings to behavioral
parameters.

Use `print_summary()` to inspect the report, `save()` for Markdown,
`save_html()` for HTML, and `save_json()` to preserve data for later comparison.
Pass completed test results explicitly through `test_results=` or
`spec_reports=` to include their metrics, warnings and run details.

## Minimal example

This example uses the packaged sky130 catalog. It needs neither installed
PDK models nor a simulator:

```python
from silk import Module, load_pdk, generate_report

pdk = load_pdk('sky130')

inv = Module('PDK_skywater130', 'out', 'inv.cir', pdk_details=pdk, module_name='inv')
inv.add(pdk.pmos1p8('mp',
    d='vout', g='vin', s='vdd', b='vdd', w=1.0, l=0.18))
inv.add(pdk.nmos1p8('mn',
    d='vout', g='vin', s='vss', b='vss', w=0.5, l=0.18))

report = generate_report(inv, auto_discover_config=False)
report.print_summary()
```

Output:

```
Circuit:     inv
  Instances: 0  |  Nets: 4
  Annotation coverage: No ports to annotate
  Interface: PASS
  Domains:    PASS
  Junctions:  STR-001 INCOMPLETE: 0/2 devices fully checked; 0 partial, 2 unchecked, 0 failed; mp: no_waveform; mn: no_waveform
  Simulation: not run
```

When there are no ports to annotate, HTML, Markdown and terminal reports say
"No ports to annotate", with no count, percentage or tick. The
`AnnotationCoverage` data and coverage gates retain their existing behavior.

`topology.instances` counts *subcircuit*
instances from `module_dict` (other `Module`s added via `add_module`), not
primitive PDK devices like the two MOSFETs above; wire a sub-`Module` in
via `add_module` to see a non-zero count (see
`silk/examples/hierarchical_example.py`). `auto_discover_config=False`
suppresses `generate_report()`'s default behavior of looking in the current
working directory for `silk_waivers.yaml`/`silk_lint_exclusions.yaml` —
worth passing explicitly in a script or doc example so a stray file in the
cwd can't change the output.

## What's inside a `CircuitReport`

The main fields on the returned dataclass are:

| Field | What it holds |
|---|---|
| `topology` | `TopologySummary` — subcircuit instances (from `module_dict`) and total net count |
| `annotation` | `AnnotationCoverage` — port-role annotation coverage, from `annotation_coverage(module)` |
| `violations` | Interface violations (`INT-*` rule IDs), unwaived and unexcluded |
| `domain_violations` | Voltage-domain violations (`DOM-*`) |
| `junction_violations` | Junction-overrange violations (`STR-*`) |
| `lint` | Full `LintReport` — every rule violation plus applied waivers plus no-connect markers, in one object |
| `coverage` | `CoverageReport` — which lint rules ran, were excluded, or were never reached, as a tree |
| `simulation` | `SimulationSummary` if the module has simulation data, else `None` — says which `analysis` ran (`transient` / `ac` / `noise` / `dc` / `op`), names its axis fields for that axis (`t_*`, `f_*`, or `sweep_name` + `sweep_*`; an `op` has no axis), and lists per-signal min/max/mean for a value-domain run, every vector's value for an operating point, or magnitude statistics (`magnitudes`) for AC / noise |
| `spec_reconciliation` | `SpecReconciliationReport` from `module.reconcile_specs()` — always attached (empty `outcomes` if no specs declared), never `None` |
| `layout_estimate` | Whatever `LayoutEstimate` was passed in via `generate_report(module, layout_estimate=...)`; `None` if omitted |
| `metadata` | `ReportMetadata` — silk version, PDK name/version, module name |

`violations`/`domain_violations`/`junction_violations` partition
`lint.violations` by rule-ID prefix. Use `lint.violations` for the complete list. See
[api-reference.md#generate_reportmodule--circuitreport](api-reference.md#generate_reportmodule--circuitreport)
for the field-by-field table (`TopologySummary`, `SimulationSummary`,
`SignalStats`, `MagnitudeStats`) and [lint_and_coverage_user_guide.md](lint_and_coverage_user_guide.md)
for `LintReport`/`CoverageReport` themselves.

## Reporting a PVT grid

Pass the whole result collection to `generate_report`. This is the 27-point
OTA grid from the agent guide (sky130 models and ngspice required):

```python
from pathlib import Path

from silk import generate_report
from silk.circuit.report import CircuitReport
from silk.examples import example_output_dir
from silk.examples.ota_5t_guide import build_5t_ota

ota = build_5t_ota()
ota.set_temperatures([-40, 27, 125])
ota.set_supply_voltages([1.62, 1.8, 1.98])
ac = ota.get_test('ac').set_corners(['tt', 'ss', 'ff'])
results = ac.run()
report = generate_report(ota, test_results={'ac': results},
                         auto_discover_config=False)
out = Path(example_output_dir('pvt_report'))
report.save(out / 'pvt_grid.md')
report.save_html(out / 'pvt_grid.html')
report.save_json(out / 'pvt_grid.json')
loaded = CircuitReport.load_json(out / 'pvt_grid.json')
loaded.print_summary()
```

The report starts with a compact PVT section, immediately before the module's
spec reconciliation. Each declared check has a row with its status, target,
worst value and operating point, margin, Min and Max. The margin is the
absolute distance from the target, as in spec reconciliation. A metric with
no declared check has Min and Max, with dashes for the spec fields. Numbers
use six significant digits.

The full per-run table has Corner, Temp (°C), Supply (V), Status and one column
per metric. It retains planned order; a run that produced no report shows
**NO REPORT**, with empty metric cells and its first error line below the table. HTML collapses this
table under **All 27 runs**; Markdown and the terminal summary show it in full.
The collection's `table()`, `to_csv()` and `summary()` retain their existing
prefixed columns and wording for programmatic use.

The following is the rendered Markdown PVT section from the sky130/ngspice-42
run above:

<!-- PVT sample start -->

Run details: silk 0.2.0, ngspice-42, sky130 (sky130.lib.spice 17c208a699)

## PVT Grid: ac — PASS

Summary: 27 runs (tt, ss, ff × -40, 27, 125 °C × 1.62, 1.8, 1.98 V): 27 PASS, 0 FAIL, 0 failed and produced no report

| Status | Spec | Op | Target | Worst | Worst at | Margin | Min | Max |
|---|---|---|---|---|---|---|---|---|
| PASS | DC gain &gt; 20 dB | &gt; | 20 | 30.349 | ff, -40 °C, 1.62 V | 10.349 | 30.349 | 37.533 |
| PASS | BW &gt; 10 MHz | &gt; | 1e+07 | 1.12202e+08 | ss, 27 °C, 1.98 V | 1.02202e+08 | 1.12202e+08 | 3.98107e+08 |

| Corner | Temp (°C) | Supply (V) | Status | gain_dB | bw_hz |
|---|---|---|---|---|---|
| tt | -40 | 1.62 | PASS | 31.5392 | 2.81838e+08 |
| tt | -40 | 1.8 | PASS | 35.265 | 1.99526e+08 |
| tt | -40 | 1.98 | PASS | 36.7161 | 1.77828e+08 |
| tt | 27 | 1.62 | PASS | 31.9872 | 2.51189e+08 |
| tt | 27 | 1.8 | PASS | 34.8507 | 1.77828e+08 |
| tt | 27 | 1.98 | PASS | 36.1249 | 1.58489e+08 |
| tt | 125 | 1.62 | PASS | 32.4953 | 1.99526e+08 |
| tt | 125 | 1.8 | PASS | 34.401 | 1.58489e+08 |
| tt | 125 | 1.98 | PASS | 35.3886 | 1.58489e+08 |
| ss | -40 | 1.62 | PASS | 32.63 | 1.99526e+08 |
| ss | -40 | 1.8 | PASS | 36.1391 | 1.41254e+08 |
| ss | -40 | 1.98 | PASS | 37.533 | 1.25893e+08 |
| ss | 27 | 1.62 | PASS | 32.9768 | 1.77828e+08 |
| ss | 27 | 1.8 | PASS | 35.6966 | 1.25893e+08 |
| ss | 27 | 1.98 | PASS | 36.9283 | 1.12202e+08 |
| ss | 125 | 1.62 | PASS | 33.3554 | 1.41254e+08 |
| ss | 125 | 1.8 | PASS | 35.1733 | 1.25893e+08 |
| ss | 125 | 1.98 | PASS | 36.1298 | 1.12202e+08 |
| ff | -40 | 1.62 | PASS | 30.349 | 3.98107e+08 |
| ff | -40 | 1.8 | PASS | 34.295 | 2.81838e+08 |
| ff | -40 | 1.98 | PASS | 35.798 | 2.51189e+08 |
| ff | 27 | 1.62 | PASS | 30.9027 | 3.54813e+08 |
| ff | 27 | 1.8 | PASS | 33.9053 | 2.51189e+08 |
| ff | 27 | 1.98 | PASS | 35.2146 | 2.23872e+08 |
| ff | 125 | 1.62 | PASS | 31.5347 | 2.81838e+08 |
| ff | 125 | 1.8 | PASS | 33.5111 | 2.23872e+08 |
| ff | 125 | 1.98 | PASS | 34.5187 | 1.99526e+08 |


## Specs not measured

These specs are declared on ota_5t, but no test checked them.

| Spec | Op | Target |
|---|---|---|
| settles &lt; 100 ns | &lt; | 1e-07 |
| input-referred noise &lt; 2 mV | &lt; | 2000 |

<!-- PVT sample end -->

Specs covered by the grid's recorded checks appear only in its spec table,
matched by stable spec key and owner path. Remaining PASS or FAIL outcomes
appear under **Specs measured in a single run**, beginning with “Specs the PVT
grid measures are in its table above.” Remaining UNMEASURED outcomes appear
under **Specs not measured**, with only Spec, Op and Target columns and an
explanation naming the module. Empty sections are omitted. Reports without a
grid retain **Spec Reconciliation**. The stored reconciliation and live spec
measurements remain unchanged.

Monte Carlo grids show the same spec extrema table, followed by a yield table
for each point and spec and an **All points (informational)** pooled row.
Yields and 95% Wilson intervals come from the recorded collection and use
three significant digits, retaining trailing zeros. **FAIL** counts samples that ran but did not meet that
spec; **No report** counts runs that produced no report. The per-sample table
shows those runs as **NO REPORT**, with empty metric cells. The informational
pooled row has no status verdict. The summary
states the required yield, for example **required yield 0.9 per point**; when
`min_yield` is unset, it says **every sample must pass**. The per-sample table
lists every planned trial and is collapsed in HTML. The grid verdict follows
the collection's per-point requirement; a pooled yield cannot override a
failing point.

JSON stores the collection evidence and summary inputs. Reloading renders the
same grid without the bench, rawfiles or calibration. Single-run and
corner-only grid presentation remains unchanged. Reports carrying producer
records show compact run details as described below.

## Saving and loading

```python
report.save('report.md')          # formatted Markdown
report.save_json('report.json')   # JSON, figures excluded
report.save_html('report.html')   # self-contained HTML, figures embedded as base64
```

Load saved data with `CircuitReport.load_json(path)`. The current writer uses
**CircuitReport JSON schema version 5**. It preserves topology, annotations,
violations, simulation statistics, metadata, spec reconciliation, layout
estimates and caller-supplied test evidence. `diff_reports()` can compare
reloaded reports, including spec transitions and layout deltas.

Readers accept unversioned files and versions 1–5. A missing version key means
v1; a newer unsupported version raises `ValueError`. Older silk readers that
support only v1–v4 reject v5 files. See [aggregate evidence](#aggregate-evidence-and-circuitreport-v4)
for the historical v4 measurement fields.

V5 stores all five violation location types (net, port, instance, terminal and
module) with a `__type` tag and their complete fields. Terminal pairs are JSON
lists restored as tuples. Valid legacy structured net records
(`net_name`, `port_names`) and terminal records (`instance_path`, `terminal`,
`terminal_pair`) remain readable. Older text-only (`describe`), empty or missing
locations cannot be restored: loading refuses with the filename and list/index
location path. Regenerate reports containing text-only locations; display text
cannot recover the original fields.

Report JSON uses strict JSON tags for non-finite numbers. At each numeric
value slot, including spec measurements and signal statistics, the saved values
are:

| Python value | JSON value |
| --- | --- |
| `float('nan')` | `{"value": null, "status": "unavailable"}` |
| `float('inf')` | `{"value": null, "status": "infinite", "sign": "+"}` |
| `-float('inf')` | `{"value": null, "status": "infinite", "sign": "-"}` |
| `0.0` | `0.0` |
| `"Error: missing signal"` | `"Error: missing signal"` |
| `None` | `null` |

For example, a measurement's `value` field becomes
`"value": {"value": null, "status": "unavailable"}`. The containing
measurement, its verdict, and its provenance keep their existing fields.
An infinity is a signed numerical limit, not an unavailable measurement;
metrics such as stopband attenuation and IMD3 retain their Python results.
`CircuitReport.load_json()` restores floats, so an unavailable measurement
still fails each supported `SpecReport.check` comparison. Numeric encoding preserves the metric value; it does not recalculate metrics
or create diagnostic warnings.

The shared encoder/decoder in `silk._json_codec` is used by report writers
and matching readers. Writers use `allow_nan=False` to reject unencoded
non-finite values. Finite-only input boundaries (for example, view annotations and
simulation conditions) retain their validation errors and unchanged finite
JSON shapes; these writers emit no tags or escapes and use ordinary JSON
readback. Finite scalar values
and ordinary strings retain their existing JSON spelling. Non-finite mapping
keys are rejected: JSON object keys cannot hold the numeric object tags.

Only exact tag shapes are reserved. A literal mapping with one of those
shapes is escaped as `{"value": <mapping>, "status": "literal"}` and restored
as a mapping; extra fields prevent a mapping from being mistaken for a tag.
This escape also applies recursively to literal escape-shaped mappings.

Legacy CircuitReport v1 files, including files without a version key and
Python's old bare `NaN`/`Infinity` tokens, still load. New report files are
strict JSON and parse with browser `JSON.parse`; old silk readers reject the
new report version. Consumers of raw JSON must support these tags before
adopting the new writer.

`lint` and `coverage` are listed in `omitted_sections` and return as `None`
after loading. Figures, waveform links, composition diagrams and rendered
operating-point tables also do not round-trip. Captured device data survives
inside supplied test evidence; use a restored `SpecReport.op_table()` to render
it again. Keep the in-memory report or regenerate it when you need its other
omitted sections.

## Complete standalone test records

Ordinary run/evaluate reports retain detached original per-spec observations.
The standalone record preserves their precise observation timestamps and attempt
attribution. A later evaluation or mutation of a live measurement does not change
the report, so you can store report A after running B, even when values differ.
Malformed or inconsistent captured observations are rejected.

Any report lacking this optional evidence remains readable and storable,
including manual reports, Monte Carlo samples, fault-campaign scoring reports
and older JSON. Storage never borrows live observation time, run ID, notes,
temperature, voltage, analysis type or tool. Report-owned operating conditions
and provenance remain intact, and supported explicit storage arguments retain
their effect subject to snapshot validation; existing configured storage defaults
also remain effective without a snapshot. Fields without that evidence stay
unknown: observation time/run ID and missing context use SQL NULL; missing notes
use empty text. Save time never becomes observation time. Unknown observation
times rank before known times and restore as `0.0`. Circuit/spec compatibility
checks still apply.

`SpecReport.to_dict()` returns a fresh strict-JSON wire record;
`save_json(path)` writes the same record. Both work immediately after `run()`
or `evaluate(rawfile=...)`, including evaluation after `prepare()` and an
external simulator run. Loading needs no circuit, database, simulator or rawfile:

```python
from silk.circuit.spec_report import SpecReport

# A manual report makes this example runnable without a simulator.
# With a prepared testbench and external results, use tb.evaluate(rawfile=...).
report = SpecReport('bias', 'amp', {'bias_v': 0.9})
report.check('bias', 'bias_v', '>=', 0.8)
report.save_json('bias.json')
wire = report.to_dict()                 # already encoded: do not encode again
restored = SpecReport.load_json('bias.json')
assert restored.to_dict() == wire       # wire tags make NaN equality ordinary
print(restored.summary())
# SpecReport.from_dict(wire) is the in-memory reader.
```

The standalone format is `"format": "silk.spec-report"`, `"schema_version": 1`.
Its version is independent of **CircuitReport JSON v5**; embedded records use
this same version-1 contract.
Every required key is emitted, including nulls and empty containers:

| Fields | Contents |
| --- | --- |
| `test_name`, `module_name`, `owner_path`, `report_key`, `parameters` | Snapshotted execution identity and declared parameters. |
| `characterization_context`, `operating_conditions`, `corner` | Complete condition snapshots, preserving unknown, none, fixed and swept facts, their basis, and the run's corner label. Context contains all six fields; its non-applicable scalars use `{"state":"none"}`. A corner named `"none"` remains a string. |
| `metrics`, `metric_types` | Every extracted output, even without a spec, and its catalog type (or null for custom/derived/unknown outputs). Expanded outputs retain their parent's type. |
| `metric_errors` | Failed outputs map to `message` (exact exception text), `exception_type`, and `available_signals` (loaded names, null if unavailable, or an empty list for a known empty inventory). Original `"Error: ..."` metric strings are retained. A successfully returned error-looking string is not classified as an exception. |
| `metric_warnings` | Ordered warning records keyed by the exact registered output, each with `metric`, `code`, `message`, `details`, `suggestion`, `output`. Two outputs of the same type retain separate warnings. Multi-output warnings stay on their registered parent. |
| `results`, `passed` | Ordered stored spec results (`name`, `metric_name`, `op`, `target`, `value`, `passed`, `spec_key`, `owner_path`) and their aggregate verdict. No checks are rerun on load; empty results means true, even if an extractor failed. |
| `rawfile_path`, `behavioral_elements`, `storage` | Informational original path, ordered approximation notes, and historical storage receipt. Loading does not open/rebase paths or write to a database. |
| `op_devices`, `op_unavailable_reason` | Captured device records (`path`, `name`, `model`, `polarity`, `w_um`, `l_um`, `m`, `declared_params`, `op_values`), or null with the unavailable diagnostic. An empty captured list stays distinct from null. `op_table()` uses the captured data offline. |
| `provenance` | Every RunProvenance field, including libraries/hashes, options, save list, seed, timestamp and scheme. No environment facts are refreshed. |

To join records to a hierarchy traversal, use the current selected handle:
`entry.test.evaluate(rawfile=...)` for an entry from `root.list_test_entries()`.
The report snapshots `entry.test.report_key`, equal to `entry.report_key`.
The key is `module_name.test_name` when unique across the selected entries,
otherwise `owner_path:module_name.test_name`. Failed tests still count toward
that identity. Never split names on dots/colons or recalculate keys from only
successful reports. A direct unselected bench uses its local key; a manually
constructed report defaults to null. The snapshot survives later selections
and release of the bench.

These records use the shared nonfinite tags described above and opt into
complex values: `{"status":"complex","re":1.0,"im":-2.0}`. Each component
may itself be a nonfinite tag. Signed zero is preserved. Literal dictionaries
matching reserved tag shapes are escaped with `status: literal`, including
nested collisions. The decoder restores Python numbers; the default shared
codec and legacy CircuitReport v2 readers do not opt into complex tags.
CircuitReport v3–v5 enable them once across the entire JSON tree.

Numpy numeric/bool scalars become Python scalars and tuples become lists.
Every mapping key must be a string. Arrays, arbitrary objects (including
successful dataclass metric result objects), circular containers and unsupported
keys are refused with field paths; an expanded scalar output does not silently
remove its unsupported parent result from the record. This is an export boundary,
not a change to metric extraction. Invalid exports are rejected before opening
the destination. Readers reject malformed types, missing required keys,
unsupported format/version, invalid operators and inconsistent verdicts with
`ValueError`; file readers also name the file. Filesystem errors propagate.
Unknown additive fields may be ignored. Loaded reports retain metrics, checks,
conditions, warnings, captured op data and provenance, but have no live
`sim_results` waveform reader.

The complete consumer examples are frozen in
[`spec_report_v1_wire.json`](../tests/fixtures/reports/spec_report_v1_wire.json)
and
[`spec_report_v1_no_op_wire.json`](../tests/fixtures/reports/spec_report_v1_no_op_wire.json).
For collections, use the adapters below. No export copies rawfiles, decks or
sources, reconstructs circuits, or writes to the database.


## Optional enrichments

- **`report.add_figure(fig, title)`** — attach a matplotlib `Figure` for
  HTML embedding (base64 PNG). In-memory only; excluded from `save_json()`;
  call repeatedly to attach more than one.
- **`report.add_waveform_view(source, *, out_path=None, label=None, annotations=None, context=None, title=None, inline=False)`**
  — emit an interactive waveform-explorer page from a rawfile or simulated
  module/testbench and link it from the report. Auto-populates the page's
  info popover with whatever provenance it can find (module, test,
  analysis, corner, PDK, this report's own spec pass/fail summary, silk
  version) — see [cli.md](cli.md#silksignalsview--waveform-explorer) and
  the agent guide's waveform-explorer section for the full auto-population
  rules.
- **`report.add_composition_diagram(module, depth=1)`** — render a Mermaid
  block diagram of `module`'s connectivity (via
  `silk.circuit.composition.module_graph`) into the report; `save()` emits
  a fenced ` ```mermaid ` block, `save_html()` a `<pre class="mermaid">`
  block. The saved HTML contains Mermaid text but no renderer; use a
  Mermaid-enabled viewer to display the diagram. Pure — reads existing
  connectivity, mutates nothing.
- **`generate_report(module, layout_estimate=est)`** — pass a
  `LayoutEstimate` (from `silk.layout.estimate_layout()` or
  `capture_layout()`) at call time to attach a Layout Estimate section
  (bounding box, per-net R/C, per-device area) to the Markdown/HTML/JSON
  output. See [api-reference.md#silklayout](api-reference.md#silklayout).

<a id="provenance"></a>

## Run details

"Why does this number differ from last month?" is answered by the
**provenance record** (`silk.simulation.provenance.RunProvenance`). Every
`SpecReport` a testbench builds (`run()`, `evaluate()`, `run_corners()`,
`run_all_tests()`) carries one as `report.provenance`; a
hand-built `SpecReport` has `None` unless you pass `provenance=`.
`SpecReport.summary()` includes a `Run details:` line identifying the silk and
simulator versions, PDK libraries and deck hash when those facts are available.

`generate_report(module, spec_reports=[...])` collects the distinct records
of the reports you pass (deduplicated by digest) into
`CircuitReport.provenance`. Successful records supplied through `test_results=`
feed the same section. Without supplied evidence (or when none carries a
record) the list is empty and no run details line is rendered; the
silk version and PDK name are still in `report.metadata`. HTML and Markdown
show one **Run details** line for each distinct combination of silk version,
simulator version and PDK library, for example:

Run details: silk 0.2.0, ngspice-42, sky130 (sky130.lib.spice 17c208a699)

Runs sharing those facts show the line once. Library corner sections and
deck hashes are omitted from this presentation; corner, temperature and supply
are in the grid table. `save_json()` retains every full collected record under
the top-level `provenance` list using the shared codec. Database records,
`report_diff`, `RunProvenance` and its `summary_line()` retain their existing
fields and behavior.
Report files written before this key existed load with an empty list.

### Field contract — `silk.provenance/v3`

`scheme` is the format version. The field names below are a **stable
contract** for that scheme: a later scheme may add fields, never rename or
re-type these. Every value is JSON-plain (string, number, boolean, `null`,
lists and objects of those).

| Field | Meaning |
|---|---|
| `scheme` | `'silk.provenance/v3'`; older records preserve their scheme. |
| `run_origin` | `'silk-launch'` when silk launched the simulator for these results; `'external-rawfile'` when the rawfile was handed to silk (`evaluate(rawfile=...)`, `load_file()` + `read_raw_file()`). |
| `silk_version`, `silk_git` | silk's `__version__`, and `git describe --always --dirty` of the silk source tree when it is a git checkout (2 s limit; `null` when git is missing, fails, or silk is an installed wheel). |
| `python_version`, `numpy_version` | Interpreter and numpy versions. |
| `simulator` | Backend name (`ngspice`, `xyce`, `behavioral`). On an external rawfile, whatever the rawfile itself names, or `null`. |
| `simulator_binary`, `simulator_binary_source`, `simulator_version` | The binary silk launched, where it came from (`'SILK_NGSPICE_BIN'`, `'SILK_XYCE_BIN'`, `'PATH'`, or `'in-process'` for behavioral) and its version string as `python -m silk.doctor` reports it (e.g. `ngspice-42`). All three are `null` on `'external-rawfile'`: silk did not launch anything, so it claims no binary, and it never probes a local binary's version on that path. |
| `pdk_name` | The PDK's name. |
| `pdk_libraries` | One entry per `.lib "<file>" <section>` line of the prepared deck: `file` (relative to `$PDK_ROOT` when under it), `section`, `file_sha256` — the bytes of **that one file** — and `section_text_sha256` — **only the text between `.lib <section>` and `.endl`** in that file. Neither hash covers the files that section includes: they say which library file and corner section were named (which is enough to tell two sky130 trees apart by their top-level library), not that the whole model tree is identical. Hashes are `null` when the file cannot be read. Empty for a PDK-free deck. |
| `corner`, `corner_kind`, `temperature_c`, `temperature_kind`, `supply_voltage_v`, `supply_kind` | Copied from `report.operating_conditions`; no second source of truth. The `*_kind` fields (`fixed`, `none`, `sweep`, `unknown`) say what a `null` value means. `corner_kind` and `supply_kind` are new in v2; a v1 record reads back with both `null` and keeps `scheme` `'silk.provenance/v1'`. |
| `simulator_options` | The prepared deck's resolved simulator options (after presets). |
| `save_list` | The `set_save_list()` settings (`signals`, `exclude`, `max_depth`), or `null` for `save all`. |
| `deck_sha256` | sha256 of the deck text silk **prepared** (`generate_netlist()`) for this bench. On a silk launch that is also the deck that ran; on an external rawfile it is the last deck silk prepared for this bench, or `null` when it prepared none. |
| `normalized_deck_sha256`, `producing_test`, `launch_corner_label` | v3 optional launch facts: versioned normalized deck identity, exact logical `owner_path`/`test_name`, and launch corner label (even when stored corner is none). Missing on v1/v2 or runs without a supported launch identity. Backend is the existing `simulator` field. The record does not store deck-lookup results. See [automatic checks](database.md#automatic-deck-checks-through-a-built-design). |
| `seed`, `seedinfo` | Monte Carlo records the sample seed in `seed`, and variation, section and sample index in `seedinfo`; ordinary runs use the `seed` simulator option when set, else `null`, and leave `seedinfo` null. |
| `timestamp_utc` | When the record was built (ISO 8601, `Z`). Excluded from `digest()` and from every diff. |

**Privacy.** Nothing in the record names a host or a user or holds an
absolute home path. PDK files are made relative to `$PDK_ROOT`; any other
path under the home directory becomes `~/...`. Inside `simulator_options`
and `save_list`, every string (keys and values, at any nesting depth) has
**every** occurrence of the home directory rewritten to `~`, mid-string
included (`'-I/home/u/x'` becomes `'-I~/x'`, `'lib=/home/u/a.osdi'` becomes
`'lib=~/a.osdi'`), at a path boundary only: the match must not be followed
by a filename character (letter, digit, `.`, `_`, `-`), so `/`, the end of
the string, whitespace, quotes and separators all end it (`'-I/home/user
-L...'` is caught), while `/home/userX` is untouched when the home directory
is `/home/user`. A binary or library outside both
keeps its absolute path only under a system prefix (`/usr`, `/opt`, `/bin`,
`/sbin`, `/nix`, `/snap`); otherwise only its file name is kept.

**Cost.** Nothing is added to a simulation. The silk `git describe` and the
Python/numpy versions are read once per process, a simulator's version once
per (binary, modification time), and a library file's hashes once per (path,
modification time, size).

`RunProvenance.diff(other)` lists `(field, a, b)` for every differing field;
`digest()` is the sha256 of the canonical JSON without the timestamp, the
key the database stores records under (see
[database.md](database.md#run-provenance-schema-v4)).

## Comparing runs — `diff_reports`

`diff_reports(a, b, label_a='a', label_b='b', sections=None)` (from
`silk.circuit.report_diff`) accepts either in-memory `CircuitReport`
objects or JSON file paths (loaded via `CircuitReport.load_json()`) and
returns a `ReportDiff`. `sections` picks which of `{'signals', 'topology',
'annotation', 'violations', 'provenance'}` to compare (default: all five).
The `violations` section compares interface findings only; use
`report.lint.diff(previous.lint)` for the full lint list.
`'provenance'` pairs the two reports' provenance records by position and
lists every differing field in `ReportDiff.provenance_diff.changed` as
`{'record', 'field', 'a', 'b'}` rows (`timestamp_utc` is never compared);
the summary, `str()` and Markdown output show them in a Provenance section,
which is omitted when neither report carries a record. Records can come from
`spec_reports=` or `test_results=`; supplying neither leaves provenance empty.
The layout diff and spec-transition list are always computed, independent of
`sections`. `layout_diff` is `None` unless both reports carry a
`layout_estimate`; that field survives `save_json()` / `load_json()`, so two
reloaded JSON reports can produce the same layout delta as their in-memory
sources. Spec transitions include rows present on only one side as additions
or removals, with `before_status` or `after_status` set to `None`, respectively.

Signals are compared by stored min/max/mean statistics, not waveform samples or
axis bounds; different waveforms can compare unchanged. AC/noise use the
corresponding magnitude statistics.
`signal_diff` is kind-aware: a transient/DC pair diffs
`SimulationSummary.signals` into a `SignalDiff`, while an
AC/noise pair diffs `SimulationSummary.magnitudes` (`MagnitudeStats` — mag_min/mag_max/mag_mean of `|x|`) into a
`MagnitudeDiff` with the same `changed`/`unchanged`/`added`/`dropped`
shape, since those runs keep their statistics in a different field and
leave `.signals` empty. If the two reports ran different analyses (one
transient, one AC), `signal_diff` is a `KindMismatch(analysis_a,
analysis_b)` instead — a structural difference reported explicitly, never
an empty diff. Both `print_summary()` and `save()` render all three shapes.

Continue the minimal example to compare two device sizings. This comparison
shows the same topology because device widths are not part of the topology
summary: it checks immediate instance names and total net count, not rewiring,
changed definitions or port counts.

```python
from silk import diff_reports

report_a = generate_report(inv, auto_discover_config=False)   # w=1.0/0.5 from above

inv2 = Module('PDK_skywater130', 'out', 'inv2.cir', pdk_details=pdk, module_name='inv2')
inv2.add(pdk.pmos1p8('mp',
    d='vout', g='vin', s='vdd', b='vdd', w=2.0, l=0.18))
inv2.add(pdk.nmos1p8('mn',
    d='vout', g='vin', s='vss', b='vss', w=1.0, l=0.18))
report_b = generate_report(inv2, auto_discover_config=False)

diff = diff_reports(report_a, report_b, label_a='w1', label_b='w2')
diff.print_summary()
```

Output:

```
Diff: w1 → w2
  Topology: unchanged
  Annotation: unchanged (100.0%)
  Violations: unchanged (0)
```

`signal_diff` is `None` here (`diff.signal_diff` — printed only when
non-`None`) because neither module has simulation data; `diff_reports`
degrades gracefully rather than raising. `signal_diff` is also `None` when only
one report contains simulation data; this is not an added/dropped signal result.
For a layout-only comparison that
populates `layout_diff` while leaving `signal_diff` as `None`, see
`silk/examples/layout_intelligence_example.py` section 9, which builds two
sized variants of a diff amp, attaches a `layout_estimate` to each report,
and diffs them — printing a real bounding-box delta and per-net ΔR/ΔC
table. `diff.save(path)` writes the same content as Markdown, mirroring
`CircuitReport.save()`.

## Where to see it in action

Start with the dedicated walkthrough,
**`silk/examples/circuit_report_example.py`** (`python -m
silk.examples.circuit_report_example`; packaged sky130 catalog only, no simulator): generate
→ `print_summary()` → `save_html`/`save_json` → `load_json()` with its omitted lint and coverage sections → `diff_reports` on two design iterations
(with a structural change the diff can actually see — a bare parameter
change is invisible to it without simulation data).

Two examples show reporting in a larger workflow:

- **`silk/examples/layout_intelligence_example.py`**, sections 8–9:
  section 8 calls `generate_report(amp, layout_estimate=est,
  auto_discover_config=False)` and prints the resulting Layout Estimate
  Markdown section, then saves it via `save_json()`; section 9 builds a
  second, scaled-up sizing of the same amp and estimates its layout. It captures
  only the nominal layout with `capture_layout()`, then passes the two estimates
  to `generate_report()` and compares them with `diff_reports()`. This is a
  layout-only comparison; neither report has simulation data.
- **`silk/examples/lint_coverage_example.py`** — the fullest walk of the
  lint/coverage side of a report: `rep.lint.summary()`,
  `rep.lint.compliance_matrix()`, waivers via `rep.lint.waivers_applied`,
  and saving `rep.save_html()` alongside the lint and coverage sub-reports'
  own `save_html()`/`save_json()`/`save_junit_xml()`.

`silk/examples/flash_adc_example.py` saves HTML reports alongside behavioral
and optional SPICE results. To include per-test operating-point tables and
metric evidence in your own report, pass completed `SpecReport` objects via
`spec_reports=` or runner results via `test_results=`. Set
`save_device_params = True` on the DUT or testbench before running to capture
operating-point tables; reports without captured data are skipped.

## Limitations

- **Reports describe the state when generated.** Later topology changes do
  not update the report. Generate a new report after changes. Optional objects
  such as a supplied layout estimate can remain shared; avoid mutating them
  when you need to preserve a report.
- **JSON does not preserve every rendered section.** See "Saving and loading"
  for the omitted sections. Metadata, reconciliation and layout do survive.
- **`CircuitReport` and the tracking database are separate systems.**
  `generate_report()` harvests a `Module`'s in-memory state and never touches
  a database connection; `CircuitReport` JSON is a shareable report artifact,
  separate from SQLite tracking records. Do not confuse it with the similarly
  named `Module.store_report()`: that method accepts the `SpecReport` returned
  by `ModuleTestbench.run()` and atomically stores its numeric metrics plus
  validated first-class spec measurements, including each output's
  `SpecReport.metric_warnings` (every `MetricWarning` raised while
  that output's extractor ran, attributed by output name; see [report
  capture and
  persistence](metrics-library.md#report-capture-and-persistence-of-metricwarning)
  and the [database guide](database.md)'s `silk_metric_warnings` section).
  Warnings accompany stored numeric outputs; use standalone report JSON to
  preserve warnings for skipped outputs too.
  Use `store_results()` for other scalar observations, and preserve the
  SQLite file for history. See the [database guide](database.md) for the
  exact persistence boundary.


## Collections and complete CircuitReport evidence

Continue the minimal example below. Since `inv` has no registered tests, its
result collection is empty. For a tested design, use that module in place of
`inv`. Calling `run_all_tests()` requires whatever backend its tests select.

```python
from silk.circuit.spec_report import (
    results_to_dict, results_from_dict, save_results_json, load_results_json,
)

results = inv.run_all_tests()  # the caller runs tests explicitly
save_results_json(results, "test-results.json")
restored = load_results_json("test-results.json")
report = generate_report(inv, test_results=results, auto_discover_config=False)
report.save_json("circuit-report.json")
```

`results_to_dict(results)` returns fresh strict wire data;
`results_from_dict(payload)` restores it offline. The file pair uses the same
contract, `format: silk.test-results`, integer `schema_version` 1 for legacy collections, 2 when
a `TestFailure` carries structured error details, or 3 for retained run
inventories, grid coordinates and Monte Carlo metadata:

| Input / `kind` | `entries` | Loaded object |
|---|---|---|
| `run_all_tests()` dict / `tests` | Ordered `{key, result}` objects | Dict with exact original string keys |
| `CornerResults` / `corners` | Successful reports in `reports` order, then failures in `failed` order | Detached `CornerResults`, including `failed` and boolean `skip_unavailable` policy |
| `MonteCarloResults` / `monte_carlo` | Tagged point/sample keys and full records or failures, plus campaign metadata | Detached `MonteCarloResults` |
| List of `SpecReport` / `records` | Ordered full per-test records, including duplicates | List of detached `SpecReport` objects |

A keyed entry's `result` is a full version-1 per-test record, a nested corners or Monte Carlo
node (inside tests), or a failure node: `{"kind":"failure","error":"original error"}`,
or in a version-2 tests collection also `error_type`, `error_class`, `details` and
`run` (the `RunRecord` fields, or `null`). Version-1 files load unchanged. A plain
`{"error": ...}` dict still writes the version-1 node. An enriched `TestFailure`
(with non-None `error_type`) loads back as a `TestFailure` with `exception=None`
(no class named in the file is ever imported) and `exception_is_original=False`.
A message-only `TestFailure` writes a version-1 node and restores as a plain
`{"error": "original error"}` dictionary, as do other unstructured failed tests;
failed corners populate `CornerResults.failed`. Keys are never parsed or
recalculated. Duplicate keys in a node are invalid. List records retain
repeated names/runs. Empty collections remain empty. Malformed nodes, unsupported
versions and invalid keys fail with field paths; file readers also name the
file. Monte Carlo collections retain settings, seeds, failures, recorded draws and
aggregate inputs; rerun hooks are not serialized. Custom containers fail explicitly.
Grid and Monte Carlo collections use test-results schema version 3; legacy
versions 1 and 2 remain readable. Versions belong to individual collection
nodes: a tests container can use version 1 or 2 while a nested corners or
Monte Carlo node uses version 3.
For a `CornerRunError`, include its errors explicitly in
`CornerResults(exc.results.reports, failed=exc.errors)` before export.

`generate_report(..., spec_reports=[...])` retains the list API and embeds a
`records` collection. Alternatively, `test_results=` accepts a result dictionary
or a corner or Monte Carlo collection. Passing both non-None inputs raises `ValueError`.
Neither supplied means `test_results: null`; an explicit empty input stays an
empty collection. Generation snapshots supplied evidence and derives operating
point tables, warning sections and provenance from its successful records, in
order. It never runs tests or discovers test evidence from live benches.
Reconciliation still comes from `module.reconcile_specs()` independently.

**Complete test evidence is supported in CircuitReport JSON v3–v5.**
Unversioned/v1/v2 files load with absent evidence represented as `None` and
retain their original numeric decoding. V3–v5 evidence can be read without a
circuit or rawfiles; resaving writes v5. The outer tree uses the shared
nonfinite/complex codec once, including literal tag escaping in nested metrics,
warnings and provenance. The per-test version remains 1; collection nodes
use versions 1, 2 or 3 as described above. These report formats are separate from `export_schema`
and database schemas.

Frozen consumer fixtures:
[`test_results_v1_wire.json`](../tests/fixtures/reports/test_results_v1_wire.json)
and [`circuit_report_v3_wire.json`](../tests/fixtures/reports/circuit_report_v3_wire.json).
They show complete records and their enclosing collections.


### Reconciliation placement identity

Displayed spec names have no placement suffix for the report’s own module or
unknown ownership. Placed blocks append a clear instance label, such as
`(instance u0)` or `(instance u0/u1)`, without the internal `$/` root. Stored
owner paths retain that root unchanged. One-line reconciliation summaries put the label immediately
after the name: `[PASS] gain (instance u0): 2 >1`.

Each reconciliation outcome carries `owner_path` and `report_key`, and its
serialized `spec` carries the immutable `key`. These are additive fields in
CircuitReport JSON v3. The owner comes from the design traversal, including
unmeasured rows; the report key comes from the test that produced the selected
measurement. Manual measurements and annotations have no originating test
report and use `report_key: null`. Reconciliation selects recorded attempts
independently of the test reports supplied to `generate_report`. A failed latest
attempt produces FAIL with a null measurement and margin, and no calibration.
Its reason and UTC completion time come first, followed by the explicitly
labelled last successful extraction's value, verdict and time. A successful
extraction can have a failing numeric verdict. If none exists, the row says
“no successful attempt.” A never-attempted placement remains UNMEASURED;
preflight and configuration refusals do not count as attempts.

CircuitReport JSON v3 adds `latest_attempt` and `last_successful_attempt` to
reconciliation outcomes. They hold only the selected observation snapshots,
not the live journal. An attempt includes `attempt_id`, `source`, `timestamp`,
`kind`, `reason`, `test_name`, `report_key`, `failure_stage`, `measurement`,
`requested_conditions` and `proved_conditions`. Kinds are `measurement`,
`extractor_error`, `unavailable`, `simulation_error` and `transport_error`.
Failures have no measurement; unknown time is 0 and unknown condition evidence
is null. Requested conditions never claim a proved simulator state. Observation
measurements retain producer value/verdict/time and physical provenance; storage
profile IDs, row IDs, overrides and private row associations are not serialized.
Missing attempt fields in older reports load as null without inventing history.

Loading a CircuitReport restores its selected snapshots and wording. It does
not rebuild a live journal. SQL persistence is unchanged: fresh SQL hydration
cannot reconstruct failed attempts or private observation-to-row associations
that SQL never stored. Saving an observation to several database profiles
preserves its original identity, provenance and order. Repeated reconciliation of retained current
annotations does not create a new attempt.

Warnings attach only when owner path, metric and spec key match. Multiple tests
or corners within that placement still contribute their warning notes. Report
keys are provenance, not a rule for choosing the latest attempt. HTML, Markdown
and terminal rows include placement labels; terminal rows also include warning
notes. Loading older JSON without row ownership produces `owner_path: null`
(unknown), which never acts as a wildcard for warning matching. Missing spec
keys in legacy records retain the old name-as-key behavior.

`diff_reports` joins rows by owner path and immutable spec key and labels each
transition with both. **Ambiguous inputs raise `ConfigurationError`**: two rows
with the same identity in either input cannot silently overwrite one another,
even if their values agree. The message identifies the input side and colliding
rows. Supply distinct identities or remove an unintended duplicate; regenerate
legacy reports with placement identity when it is missing. Unknown owners never
match known ones. A unique unknown-owner row can match a unique unknown row
with the same key and stays labeled `unknown placement`. Transitions sort unknown
owners first, then known paths, then spec keys, all lexically within each group.
Their records add `owner_path`, `spec_key` and `report_key`; the report key comes
from the after row if present, otherwise the before row.

Fault-runner `outcome.json` records likewise add `owner_path`, `spec_key` and
`report_key` alongside their existing name, status and value fields. Missing
legacy/mock identity is null. This list projection preserves duplicates and
order; it does not perform a diff join.

### Aggregate evidence and CircuitReport v4

Every newly saved CircuitReport uses JSON v5. V4 introduced the optional
`Measurement` aggregate record, which survives save/load, including evidence
restored from a database. Missing
aggregate fields in older reports load as None. A malformed known aggregate
payload, unsupported scheme, or metric/run identity contradiction refuses
with the report filename and outcome. Save validates attribution before opening
the destination, so a refusal leaves existing output untouched.
The aggregate codec and attribution validator are shared with database storage.

V3-only readers reject v4 before decoding. They otherwise drop unknown
Measurement fields, so a v3 reader cannot be claimed to preserve aggregate
evidence. New readers accept supported v1-v4 inputs; v3–v5 use the same complete
nonfinite/complex evidence decoder after the upper-version gate. Resaving an
old report writes v5 and preserves its other evidence. The standalone
SpecReport wire format remains v1 (with optional `run` and `launch_id` keys);
test-results is v1 for legacy collections, v2 when it holds an enriched
`TestFailure` with non-None `error_type`, or v3 for grid inventories and
Monte Carlo metadata. A message-only `TestFailure` stays v1 and restores as an error
dictionary. `export_schema` is unchanged.
Monte Carlo collection evidence is preserved, including recorded draws and
summary inputs; live rerun hooks and waveform files are not embedded.

The frozen [v4 fixture](../tests/fixtures/reports/circuit_report_v4_wire.json)
adds aggregate evidence; the existing frozen v3 fixture is unchanged.
