# Operating-point table (`SpecReport.op_table()`)

`op_table()` turns the raw operating-point capture `Module.save_device_params`
produces (see [api-reference.md § Device operating-point parameters](api-reference.md#device-operating-point-parameters))
into one row per MOS device: raw simulator readings, a handful of derived
small-signal figures, and an operating-region classification. The table
includes the device under test (DUT) and helper modules in its testbench.

The following snippets assume an existing DUT and registered testbench;
see [agent-guide.md](agent-guide.md) for creating them. Enable capture on
the DUT, or set `tb.save_device_params = True` for one test. The testbench
setting defaults to `None`, which inherits the DUT's setting. A behavioral
run has no electrical operating point; `op_table()` raises `NotSupportedError`
for that backend. On ngspice, `save_device_params=True` cannot be combined
with `save_device_currents=True`; use separate tests.

```python
# dut is a Module with a registered test named 'my_test'.
dut.save_device_params = True         # before running (Module default: False)
report = dut.run_test('my_test', corner='tt')

table = report.op_table()
print(table)                          # or table.to_markdown()

row = table[0]                        # positional -- netlist-walk order
row = table['dut.st1.MN']                 # by the designer's own instance name
row.name                              # 'dut.st1.MN' -- bench placement plus designer names
row.path                              # 'xdut.xst1.XMMN' -- bench-rooted emitted SPICE path
row.gm, row.vth, row.region           # raw + derived + classification
row.sources['gm']                   # 'sim' | 'derived' | 'derived_estimate' | None

cutoff = table.filter(region='cutoff')             # which devices are off?
weak = table.filter(saturation_weak=True)          # which are weakly saturated?
either = table.filter(lambda r: r.region == 'cutoff' or r.saturation_weak)
```

## Device-parameter probes

`op_table()` is an all-devices overview captured once, after a run,
via `Module.save_device_params`. On ngspice it comes from a separate
operating-point solve after the main analysis; use device probes for values
throughout a transient or sweep. When a caller wants ONE device's
parameter — used in arithmetic, fed to `add_metric`, checked with a
`Spec`, or read in a DC sweep / transient / AC where the value varies —
`tb.probe()` takes the DUT-relative designer instance name with
the parameter after a colon. Here `tb` is a registered `ModuleTestbench`
containing `st1.MN` inside its DUT and a voltage source emitted as `Vvin`:

```python
from silk.simulation import DC

gm  = tb.probe('st1.MN:gm')           # Signal, same as any node probe
id_ = tb.probe('st1.MN:id')
gm_id = gm / id_                      # ordinary Signal arithmetic — gm_id
                                       # is NOT a probe parameter itself

tb.add_sim_type(DC(source='Vvin', start=0, stop=1.8, step=0.05))
report = tb.run(corner='tt')
gm.y                                  # gm against the swept source
```

One spelling, every analysis: `Op()` gives a single value, a `DC` sweep
and `Transient` give the parameter against the swept source or time, and
`AC` gives the bias-point value held constant across frequency (ngspice
writes an `'@'` vector in an AC plot as a zero-imaginary complex number;
the returned Signal is real, same `xlabel` as any other AC probe). It is
the same `Signal` type every other probe spelling returns — no separate
accessor, no per-analysis shape to remember. Device-parameter probes are
refused in Noise-only tests; use a separate operating-point or other supported
analysis.

**Probe spelling.** The final colon separates the device name from a
case-insensitive parameter or terminal suffix. The terminal suffixes
`d`/`g`/`s`/`b` request terminal currents; the 17 raw parameter names below
request device parameters. An unknown suffix raises `ValueError` in
`probe()`, listing supported suffixes and a spelling suggestion.

Brackets do not select device parameters. Authored net, port and instance
names accept numeric trailing indices such as `bus[3]`, and refuse `:`
and `.`. In raw waveform addresses, `:` separates hierarchy levels.
A `v(...)` or `i(...)` wrapper selects a raw voltage or branch address
before device-parameter dispatch; it does not permit a colon in an
authored design name.

**Not limited to `op_params`.** `op_table()`'s declared per-device
`op_params` set is a *table* concern (it excludes parameters not verified for the model and backend). A probe is an explicit, single request and
may name any of `silk.simulation.backend.DEVICE_OP_PARAMS` the backend
can address for that device — validated before the run.

**A derived column is not a probe parameter.** `gm_id`, `gm_gds`, `ro`,
`vov`, `vds_margin`, `ft`, and `region` are `op_table()`-only arithmetic
over raw probes. `tb.probe('st1.MN:gm_id')` raises before the run,
naming the plain-Signal expression to use instead
(`probe('st1.MN:gm') / probe('st1.MN:id')`).

A device name must resolve to a MOS device inside the DUT. Unsupported
Xyce devices or parameters, and device-parameter probes combined with
`save_device_currents=True` and `terminal_current_method='ammeter_wrapper'`,
are refused before simulation. The latter combination cannot reliably
detect missing ngspice vectors.

For a recognised parameter outside a device's declared `op_params`,
ngspice first checks availability in a separate single-device simulation.
Results are cached by resolved binary path, model and parameter for the
process. An unavailable parameter is omitted from the main run's saves;
reading its probe raises with diagnostic details instead of returning a
silent zero. Declared parameters skip that check, but readback still checks
the run log for a warning about the requested vector. Create probes before
running: a device-parameter probe created after the last run requires a rerun.

Device probes add saves to the current save mode, including an explicit
`Module.set_save_list`; they do not replace the other saved vectors.
See [probes.md](probes.md) for probe selection and deferred reads.

## Where it lives, and why

`op_table()` is a method on `SpecReport` — the object `run()`/`run_test()`/
`evaluate()` return — not on the DUT `Module` and not on the testbench (`tb`).
Each `SpecReport` captures one run's operating-point inventory during evaluation.
`op_table()` uses that snapshot without accessing the DUT or rawfile.
`SpecReport.save_json()`/`load_json()` preserve the captured inventory.
`save_device_params = False` means `report.op_devices` is `None` (ngspice and
Xyce both capture the inventory when it is on; Xyce only with `Op()` as the
declared analysis, see [Xyce support](#xyce-support)); in that case `op_table()`
raises `SimulationError` naming `save_device_params` rather than silently
returning an empty table.

## Row naming and lookup

With more than a handful of devices, the emitted SPICE path (see below) is
a hand-translation exercise: `add_module(stage, 'st1')` then
`stage.add(pdk.nmos1p8('MN', d='out', g='in', s='vss', b='vss'))` shows up as `'xdut.xst1.XMMN'` — the `'x'`
per submodule level plus the PDK wrapper's own `'XM'` prefix, none of it
written by the designer. `OpRow.name` is the fix: the designer's **own**
hierarchical instance name, dot-joined straight from the strings passed to
`add_module`/`add_element`/`add` — `'dut.st1.MN'` for that same device — with
no emission prefix at any level.

`table['dut.st1.MN']` looks a row up by `name` — exact-match, case-sensitive,
while circuit construction rejects placements whose emitted SPICE names differ
only by case. Two
placements of the same submodule under different instance names, e.g.
`st1`/`st2` both wrapping one `stage` object, are disambiguated for free
by their own placement names, not by any identity of the underlying
`Module`. An unknown name raises `KeyError` listing the closest known
names; `table[0]` indexes depth-first testbench insertion order, including helpers.

An older `op_devices` inventory captured without designer names has `name is None` on every row (rendered as the `'—'`
placeholder, same as any other hole) — string lookup on such a table
raises explaining the report predates designer names, rather than a
confusing bare miss.

### Filtering

`OpTable.filter()` returns a new `OpTable` (same rows subset, still
iterable/renderable/name-indexable, so it composes) for the everyday
questions:

```python
table.filter(region='cutoff')                       # a single region
table.filter(region=('cutoff', 'subthreshold'))      # several
table.filter(saturation_weak=True)                   # weakly-saturated devices
table.filter(lambda r: r.region == 'cutoff' or r.saturation_weak)  # OR queries
```

`region=`/`saturation_weak=` are mutually exclusive with a raw
`predicate=` callable (raises `TypeError` if combined — an ambiguous
AND/ignore choice otherwise); `region=` validates against the known
region set, so a typo raises rather than silently returning zero rows.

## Row keying: names and paths start at the testbench

Every newly captured row includes its actual bench placement. MOS devices
placed directly in the testbench also receive rows, using their unprefixed
designer names. A DUT block
named 'bias1' and a helper also placed as 'bias1', each containing MN and
MP in that order, produce these rows when the DUT is placed first:

| Order | name and lookup key | path |
| --- | --- | --- |
| 0 | 'dut.bias1.MN' | 'xdut.xbias1.XMMN' |
| 1 | 'dut.bias1.MP' | 'xdut.xbias1.XMMP' |
| 2 | 'bias1.MN' | 'xbias1.XMMN' |
| 3 | 'bias1.MP' | 'xbias1.XMMP' |

Use table['dut.bias1.MN'] for the DUT and table['bias1.MN'] for the helper.
Nested and repeated helper placements retain their complete placement
hierarchy. Backend readback matches from the actual hierarchy root after
simulator decoration, allowing internal-device and wrapper suffixes, so a
missing helper reading cannot borrow a similarly named DUT reading.

Rows follow depth-first bench insertion order; a helper inserted before
the DUT can become `table[0]`. Stored reports retain their captured labels
and order. Device-parameter probes use DUT-relative names:
`tb.probe('bias1.MN:gm')`.

The existing save_device_params setting captures all MOS-shaped devices
in the DUT and helper modules. Undeclared devices keep unavailable rows
(nan readings, no declared parameters or region, absent source markers);
they do not cause guessed or additional saves. The same capability and
region policies apply to both scopes.

## Columns

| Column | Meaning |
| --- | --- |
| `name` | The designer's own hierarchical instance name, e.g. `'dut.st1.MN'` — `None` on a row from a legacy capture without designer names. |
| `path` | The bench-rooted emitted SPICE path, e.g. `'xdut.xst1.XMMN'`. |
| `model` | The PDK model name (e.g. `sky130_fd_pr__nfet_01v8`). |
| `polarity` | `'n'` / `'p'`, a per-device PDK declaration (see below) — `None` if undeclared. |
| `w_um`, `l_um`, `m` | Geometry from `BasePDK.device_geometry_um`: `w_um` and `l_um` are micrometres regardless of the PDK's native unit; `m` is the dimensionless instance multiplier. |
| `id vgs vds vbs vth vdsat gm gds gmbs cgg cgs cgd cgso cgdo cgbo capbd capbs` | Raw simulator readings — the fixed 17-parameter set `save_device_params` captures. |
| `vov` | `vgs - vth`. |
| `gm_id` | `gm / id`. |
| `gm_gds` | `gm / gds`. |
| `ro` | `1 / gds`. |
| `ft` | `gm / (2·π·cgg)` — **an estimate**, not a directly measured transit frequency (see below). |
| `vds_margin` | `vds - vdsat`. |
| `region` | `'cutoff'` / `'subthreshold'` / `'triode'` / `'saturation'`, or `None` when any of `vgs`/`vth`/`vds`/`vdsat`/`id` is unavailable. |

`row.saturation_weak` is `True` when the region is `'saturation'` and
`0 < vds - vdsat <` `weak_sat_margin_v` (see **Region rule** below); `None`
when `region` is `None`.

## Per-column source markers

`row.sources` maps each raw/derived column name to one of:

- **`'sim'`** — a real `device_op_vectors` reading: this device's model
  **declares** the parameter (see below).
- **`'derived'`** — computed here from `'sim'` inputs.
- **`'derived_estimate'`** — `ft` only. `gm/(2·π·cgg)` is a small-signal
  estimate of the transit frequency, not a direct measurement — `cgg` (not
  `cgg + cgs + cgd` or similar) is the correct denominator, but a real `fT`
  sweep is a separate characterization exercise. Every other derived column
  uses plain `'derived'`.
- **`None`** — unavailable: the model does not declare this raw parameter,
  or one of a derived column's own inputs was unavailable. The value is
  normally `nan`, **never** a silently-omitted key. Explicit probe saves can supply
  finite raw readings even with a `None` marker; derived columns ignore undeclared
  inputs. An undeclared device's name, path,
  model, geometry and declared polarity can still be present.

A column can be `'sim'` and still read `nan`: a declared parameter that
genuinely comes back unreadable at a given bias (a real simulator reading,
not a catalog gap) keeps its `'sim'` marker — the marker records
*declared-ness*, not *non-nan-ness*.

## Which devices are declared

`op_params`, `op_internal_name`, `polarity`, and `op_backends` are
**per-device-model PDK catalog declarations**
(`silk/pdk/data/*_device_details.yaml`), not global constants — the
self-describing-PDK pattern. A model with no declaration is silently
skipped: no `'@'`/`N()`/`ID()` columns are attempted for it, and its parameter columns are unavailable. Geometry and other identifying
fields may still be present. This is deliberate, not a bug: a name a
model doesn't actually support is a zero-length rawfile vector on ngspice
or an unresolvable `.print` symbol on Xyce, and referencing even one such
name aborts the *entire* rawfile write (ngspice) or the *entire run*
(Xyce) — declaring only verified sets is what keeps one bad name from
losing a run's data.

`op_backends` lists the backends supported by the device's declaration.
Table readings are limited to the intersection of that declaration and
the backend's parameter capability.

The catalog declares these MOS devices:

| PDK | Devices | Parameters | Polarity | Backends |
| --- | --- | --- | --- | --- |
| sky130 | `nmos1p8`, `pmos1p8` | 14 of 17 (`cgso`/`cgdo`/`cgbo` excluded — see below) | `n` / `p` | ngspice, xyce |
| sky130 | `nmos1p8_lvt`, `pmos1p8_lvt`, `pmos1p8_hvt`, `nmos3p3_native`, `nmos5v`, `nmos5v_native`, `pmos5v`, `nmos_esd`, `pmos_esd` | 14 of 17; Xyce reads the 8-column intersection below | `n` / `p` | ngspice, xyce |
| sky130 | `nmos20v`, `nmos20v_iso`, `pmos20v` | 14 of 17 | `n` / `p` | ngspice only |
| sky130 | `nmos20v_native`, `nmos20v_zvt` | 13 of 17 (`capbd` also excluded: only zero observed) | `n` | ngspice only |
| gf180mcu | `nmos3p3`, `pmos3p3` | 14 of 17 (`cgso`/`cgdo`/`cgbo` unavailable on this model) | `n` / `p` | ngspice only — see below |
| sg13g2 (PSP/OSDI) | none | — | — | — |

sky130 overlap-capacitance parameters `cgso`/`cgdo`/`cgbo` are excluded
because availability varies by ngspice version. A zero-length rawfile
vector is read as `nan`, rather than zero.

The five 20 V wrappers expose the internal MOS primitive `m1`, behind their
drift/series circuitry. Its voltages, small-signal parameters and region
describe that primitive, not the complete wrapper at its external terminals.
The native and zero-threshold wrappers also exclude `capbd`, whose
operating-point readings were only zero. The five wrappers are not declared
for Xyce; their table parameters and derived values remain unavailable there.

Other device types (bipolars, varactors, etc.) remain undeclared pending
their own verification. `polarity` cannot be inferred from the model name by any general rule
— sky130 spells it `nfet_01v8`/`pfet_01v8`, gf180mcu spells it
`nmos3p3`/`pmos3p3` — so it is declared the same way, never sniffed from a
substring.

Operating-point declarations are not included in `export_schema()`.

## Xyce support

Xyce exposes device operating-point values through `N(<path>:PARAM)` /
`ID(<path>)` in a `.print` line, not ngspice's interactive `'@'`-vector
save sequence (Xyce has no `.control` block equivalent). `op_table()`
works the same way regardless of backend — the difference is entirely in
what a Xyce row can populate:

| Column | Xyce | Notes |
| --- | --- | --- |
| `id`, `vgs`, `vds`, `vbs`, `vth`, `vdsat`, `gm`, `gds` | available | measured directly against real ngspice output on the same bias — agreement is checked with tolerances (`id` sign-corrected for PMOS, see **Sign conventions** below) |
| `gmbs`, `cgg`, `cgs`, `cgd`, `cgso`, `cgdo`, `cgbo`, `capbd`, `capbs` | **unavailable** | silk's fixed eight-parameter set was verified on Xyce 7.10; these other columns are unavailable through this backend |
| `vov`, `gm_id`, `gm_gds`, `ro`, `vds_margin` | derived, available | every input above is available |
| `ft` | **unavailable** | needs `cgg` |
| `region` / `saturation_weak` | available | computed identically to ngspice — needs only `vgs`/`vth`/`vds`/`vdsat`/`id` |

The 8-name Xyce-readable set is `silk.simulation.backend.XYCE_OP_CAPABILITY`;
`SimulatorBackend.op_param_capability(declared)` intersects a device's own
`op_params` with it (identity on every other backend), so a Xyce hole reads
`sources[col] is None`, never `'sim'`.

**Only `Op()` may pair with `save_device_params` on Xyce.** Xyce runs one
analysis per deck and has no post-run interactive step —
`Op()` itself renders as a single-point `.dc` sweep over a private
grounded 0 V source that silk adds (user sources are left untouched), and the `N()`/`ID()` terms are appended
*additively* to that same `.print` line. Any other declared analysis
(`Transient`, a swept `DC`, `AC`, `Noise`) combined with
`save_device_params = True` raises `NotSupportedError` at generation
time, naming the rule — appending device columns to a print line that
already records every solve point would capture them at every point too,
which silk's design forbids on every backend (`DEVICE_OP_PARAMS`'s
`save_device_params` is deliberately an operating-point-only capture, not
a per-timestep one).

GF180MCU operating-point capture is declared only for ngspice. GF180MCU
and SG13G2 remain preview integrations; catalog availability does not imply
release-validated transistor simulation. The GF180MCU ngspice model files are
incompatible with Xyce 7.10; silk selects the PDK's dedicated Xyce files.
Operating-point capture remains declared only for ngspice.
For sky130, Xyce 7.10 replaces the authored BSIM4 version 4.5 with 4.6.1,
so matching operating-point columns do not establish agreement beyond those
columns.

## Sign conventions (PMOS)

For the declared sky130 and GF180MCU models, ngspice reports normally
conducting PMOS `vgs`, `vds` and `id` as positive source-referenced values.
The table applies its formulas and region rule identically to both polarities.
Xyce's `ID()` uses the drain branch-current sign instead, so silk negates
`id` for devices declared with polarity `p`. It does not take an absolute
value: reversed current retains its sign. Xyce's `vgs` and `vds` need no
correction. A new model family needs its own sign verification.

## Region rule

Every threshold is a parameter, not a hardcoded constant
(`RegionThresholds`, `silk.circuit.op_table`):

```python
from silk.circuit.op_table import RegionThresholds

report.op_table(region_thresholds=RegionThresholds(
    subthreshold_margin_v=0.08, min_current_a=1e-9, weak_sat_margin_v=0.05))
```

### Project and customer threshold files

Defaults live in the shipped
[`region-thresholds.default.yaml`](../silk/circuit/data/region-thresholds.default.yaml),
with a separate entry for every registered PDK. Copy that file into your
project and edit the values for your customer. From a source checkout:

```bash
cp silk/circuit/data/region-thresholds.default.yaml customer-regions.yaml
```

For an installed package, this copies the same shipped file:

```python
from shutil import copyfile
from silk.circuit.region_thresholds import DEFAULT_THRESHOLDS_PATH
copyfile(DEFAULT_THRESHOLDS_PATH, "customer-regions.yaml")
```

For example, changing the sky130 weak-saturation margin from 50 mV to 80 mV
is a one-line edit: `weak_sat_margin_v: 0.08`. A complete small customer file
can contain just this entry:

```yaml
version: 1
pdks:
  sky130:
    min_current_a: 1.0e-9
    weak_sat_margin_v: 0.08
    subthreshold_margin_kt_q: 3.0
```

Select the file once in your [silk settings file](settings.md):

```dotenv
SILK_REGION_THRESHOLDS=/path/to/project/customer-regions.yaml
```

For project-local selection, put that line in a project `silk.env`, then set
`SILK_SETTINGS=/path/to/project/silk.env` in your shell. You can also set
`SILK_REGION_THRESHOLDS` directly as an environment variable; it overrides the
settings file. Leading `~` expands to your home directory; relative threshold
paths resolve against the working directory, so absolute paths are useful
when launching from different directories. `python -m silk.doctor` shows the
value and source and validates the selected file before simulation.

Both `report.op_table()` and `generate_report(dut, spec_reports=[report])`
use the selected policy without per-call code. Precedence is:

1. An explicit `RegionThresholds` object passed as `region_thresholds=`.
2. The user file's complete entry for the report's PDK.
3. The shipped file's entry for that PDK.

PDK identity comes from the report's captured `provenance.pdk_name`, with
registered aliases resolved to canonical names. Use canonical names in your
YAML. Historical or hand-built reports without PDK identity use the explicit
`default` entry; you may override `default` in the user file too. A named PDK
never falls back to `default`: a custom PDK needs its own user entry if no
shipped entry exists. Omitted PDK entries use shipped policy, but an entry
that is present must include all thresholds; fields are not merged.

`subthreshold_margin_kt_q` is a multiplier of kT/q at the report's captured
ambient temperature, with the existing 27 °C fallback for unknown/unscoped
temperature. A temperature explicitly recorded as `none` refuses this thermal
calculation. To specify fixed volts instead, replace the multiplier line with
`subthreshold_margin_v: 0.08`; supply exactly one of these two keys. The
shipped sets use (1 nA, 50 mV, 3 kT/q). The GF180MCU
and SG13G2 entries do not imply independent
validation or change those PDKs' preview status.

Values must be finite, nonnegative YAML numbers (no unit suffixes or booleans).
Unknown/duplicate keys, missing fields, invalid YAML, and unreadable files
raise `RegionThresholdsError` (`ValueError`), naming the file and offending
key (or `<root>` for file/syntax errors). The error is also importable from
`silk.circuit.op_table`. Doctor reports these as `FAIL` with a fix hint.

For a one-off HTML override:

```python
from silk.circuit.op_table import RegionThresholds
from silk.circuit.report import generate_report

thresholds = RegionThresholds(weak_sat_margin_v=0.08)
html_report = generate_report(dut, spec_reports=[report],
                              region_thresholds=thresholds)
```

An explicit object overrides policy for every included report and bypasses
user-file loading. Its omitted constructor fields use the shipped `default`
entry at nominal 27 °C. Without an override, each included report uses its
own PDK and temperature. Policy is resolved when the table/report is built;
changing the file affects future table builds, without rerunning simulation.

Classification, applied to every device (both polarities, see above):

1. **cutoff** if `vgs < vth - subthreshold_margin_v`, **or** (`vgs < vth`
   **and** `abs(id) < min_current_a`).
2. **subthreshold** if `vgs < vth` (and not cutoff by rule 1).
3. **triode** if `vgs >= vth` and `vds < vdsat`.
4. **saturation** otherwise, with `saturation_weak = True` when
   `0 < vds - vdsat < weak_sat_margin_v`.

A device with `vgs >= vth` is **never** cutoff on current alone: rule 1's
`abs(id) < min_current_a` branch only ever applies together with
`vgs < vth`.

`region` (and `saturation_weak`) is `None` — never a default bucket — when
any of `vgs`/`vth`/`vds`/`vdsat`/`id` is unavailable for that device, which
includes a reading that is `nan` (or infinite): a declared device whose
vectors could not be read (no op plot in the rawfile, or no/ambiguous match)
has region `None`, not `'saturation'`.

When the ngspice matcher finds **no** operating-point vector for a declared
device, or **more than one** for a parameter (another instance whose name
contains the device path), it emits a `MetricWarning`
(`code='op_device_not_found'` / `'op_device_ambiguous'`) naming the device
and, for the ambiguous case, the candidate vectors — the row is still `nan`.

A weakly-saturated device renders as `'saturation (weak)'` in both
`str(table)` and `table.to_markdown()` — `saturation_weak` is a real `bool`
field on `OpRow` for a caller to branch on, but the rendered `region` cell
folds it in so it's visible at a glance rather than only reachable
programmatically.

## Rendering

`OpTable` (the return value of `op_table()`) is iterable over its `OpRow`
rows (`len()`, indexing, `for row in table`), and renders as plain text
(`str(table)`) or Markdown (`table.to_markdown()`). Rows are frozen `OpRow`
dataclasses; pandas is not required.

A hole — a column this device's model/backend never declared
(`row.sources[col] is None`) — normally renders as `'—'` on every surface
(console, Markdown, and HTML alike). A column that *is* declared but
computes to a genuine `nan` (e.g. `ro` when a real `gds` reads exactly
`0.0`) renders as `'nan'` instead — a different fact from a hole, so it
gets a different placeholder. Both cases are `math.isnan(value)`; only
`row.sources` distinguishes them. A finite undeclared raw reading supplied by an
explicit probe save renders numerically despite its `None` marker.

## HTML report

`silk.circuit.report.generate_report()` takes an optional `spec_reports=`
kwarg — a list of already-evaluated `SpecReport`s to source Operating
Point Table sections from:

```python
from silk.circuit.report import generate_report

report = dut.run_test('opchk', corner='tt')   # dut.save_device_params = True
circuit_report = generate_report(dut, spec_reports=[report])
circuit_report.save_html('report.html')
```

`spec_reports=` supplies a list of evaluated `SpecReport` objects directly;
`test_results=` is the alternative route for runner results, from which
`generate_report()` extracts successful reports. These arguments are mutually
exclusive. Cell values come from the same `op_table()` rows; nothing is
re-derived from a rawfile.

One `<h2>Operating Point Table — {test_name}{ (corner)}</h2>` section is
rendered per captured `SpecReport`, **in the order passed**, right after the
Behavioral Simulation section. A report whose `op_devices` is `None`
contributes no operating-point section. Supplying neither evidence argument
also produces no operating-point section. A captured report whose table has
zero MOS devices still renders its heading, with the same
`'(no MOS devices)'` message `OpTable.__str__`/`to_markdown()` use.

Table cells reuse `op_table._cell_text` — the single renderer shared with
the console/Markdown table — so a hole and a real `nan` never diverge
between surfaces (see above). HTML additionally styles what only HTML
can:

- A hole cell (`row.sources[col] is None`) gets `class="op-hole"`.
- The region cell gets `class="op-region-<region>"` (e.g.
  `op-region-saturation`, `op-region-cutoff`).
- A row with `saturation_weak`, or whose `region` is `'cutoff'`, gets
  `class="op-flag"` on its `<tr>`.
- An `ft` cell sourced `'derived_estimate'` (`gm / (2π·cgg)` — a
  small-signal estimate, never a directly measured transit frequency) gets
  `class="op-estimate"` plus a tooltip, and its section gets a one-line
  note explaining the estimate.
