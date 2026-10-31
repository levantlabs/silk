# Known Issues and Limitations

This page describes current limits and practical workarounds. For open work, see
the [public issue tracker](https://github.com/levantlabs/silk/issues).

## Authored names

Authored net, port and instance names use ASCII letters, digits and `_`,
start with a letter or `_`, and may end in one numeric index such as `data[3]`.
The ground net `0` is the only leading-digit exception; instances cannot be `0`.
Names such as `a:b`, `a.b`, `v-bias`, `bus[en]`, `net[3]bias`, `d[1][2]`
and `gnd!` are refused at construction and rechecked at netlist emission.
`gnd!` is an ordinary node in SPICE, not ground. Use `0`; inside a block,
declare `Port.ground()` (conventionally `vss`) and wire it to `0` at the bench.
The `.` and `:` separators remain legal in hierarchy selectors and probe paths,
including `tb.probe('m1:gm')`; they are not part of an authored name.

## Declared names and dot shortcuts

Declared nets and buses take precedence over accessor fields in `module.nets`,
including names beginning with `_`: `module.nets._module` returns the net
called `_module`. Bracket access remains available. Behavioral ports and buses
use ordinary dot access: a declared `_state` port is read or published through
`ports._state.v` or `ports._state.i`. The `__silk_` namespace is reserved
for port-view bookkeeping, so valid port names cannot collide with its
internal fields or methods.
Port names starting with `__silk_` are refused at declaration.

Placement handles give declared nets, ports, buses, child blocks and devices
precedence over the seven public names `bus`, `child`, `device`, `instance_name`,
`net`, `path` and `subcircuit`. On a block with an item called `net`, `stage.net`
returns that item; the tool `stage.net('x')` is unavailable through that handle
and calling a noncallable item raises Python's normal `TypeError`. Tools
remain available where no item shadows them. Multiple matching item kinds
raise an `Ambiguous placement attribute` error. Placement names beginning with
`_` remain unavailable through the shortcut.

Python double-underscore attributes such as `__class__` retain normal lookup
on all three objects; declaring an item with that name does not make it
available through the dot shortcut.

## Multiple result plots of the same kind

Signal and deferred probe reads with `analysis='tran'`, `'ac'`, `'dc'`,
`'noise'` or `'op'` raise `SimulationError` when more than one plot matches.
Selection among multiple plots of the same kind is not currently supported
by silk and is a future extension. Save the intended result in a separate
rawfile. This affects custom or concatenated ngspice files and repeated Xyce
step plots. Spectral and integrated noise plots both count as noise; equal-valued plots are still
ambiguous. Operating-point tables also refuse duplicate operating-point plots.

Rawfile loading still preserves every plot, unique unrelated analysis kinds
remain readable, and the viewer retains indexed plots and its existing
default. Explicit low-level `NgspiceSignal.from_rawfile(..., plot=raw.plots[i])`
reads remain available. Ordinary native one-analysis-per-class registration
is unchanged. See [multiple-plot selection and workaround](multi-analysis-tests.md#multiple-plots-of-the-same-kind).

## Shared device and source wiring

Factory device pin edits (including passive terminal aliases) and built-in
source terminal edits update every module that directly holds the same object.
The complete edit must be accepted in every holder before warning or mutation;
source update batches include their value fields in this validation. A `Net`
argument names each holder's own local net; it does not merge nets across modules.
A changed
pin on an object with multiple direct holders emits one `UserWarning`. Values
have no sharing warning. A device inside a module placed several times still
has one direct holder, so an ordinary internal edit is silent and each
placement retains its own outside wiring.

Current-controlled sources still resolve their controlling source name through
`_module`, the last module that successfully added that source object. Sharing
one such source between modules with different controlling-source layouts does
not give it a separate control lookup for each holder. Build distinct sources
when those contexts must differ.

Before deck emission and lint, silk checks supported factory devices, built-in
sources and submodule placement wiring against both `Net.pins` and the reverse
index. Missing, stale or duplicate bindings raise `CircuitError`; coherently
unbound ports remain available for lint to diagnose. The check is read-only and
does not repair direct dictionary edits. Custom device/source emitters and
opaque custom lines are outside this check; supported siblings are still
checked. Legacy passive subclasses with their own terminal storage do not
participate in shared pin propagation.

## Instance identity requires the placement APIs

Direct edits to `elem_dict`, `passives_dict` or `sig_sources_dict` bypass instance
name checks and are unchecked. Use the placement APIs to register physical card
identity and lock names; create a new part instead of renaming an added part.
Custom legacy `PassiveBase` subclasses must implement `physical_instance_names()`
with a nonempty tuple of unique, value-independent card tokens and use that same
inventory in `to_spice()`. Subclasses implementing only `to_spice()` are refused
at placement. Changing a custom passive's declared card identity while placed
is unsupported.

## Analysis completion

silk checks completion only for ngspice/Xyce requests it can verify from
its generated deck. These are called **certified requests** and cover
transient, DC, AC and noise requests. Generation stores immutable request
records with the final deck hash in the prepared run; the backend checks that
same deck and associates each request with exactly one result plot (or Xyce's
noise CSV). Missing or truncated certified results raise `SimulationAborted`.
Xyce `Step Analysis:` sample prefixes are classified by the trailing analysis
name, including one-point DC samples. Integrated Noise totals are excluded.
Ambiguous repeated plots have no endpoint
verdict, rather than being compared against every request.

Transient endpoints use `1e-7` times the largest absolute requested start,
stop or span, with no absolute floor. Discrete sweeps also require their
expected point count and cap that tolerance at one quarter of the final
resolvable interval. A narrow sweep cannot pass with only its first point.
DC and fixed-ratio frequency sweeps compare their final reachable grid point
when the stop lies between steps. Sub-decade ngspice DEC AC requests use
version-aware completion grids from the simulator YAML: releases through 42
stretch the step to reach the stop; 43 and later use whole `10^(1/points)`
steps and can finish below it. An unknown version retains no endpoint verdict
for that request. Measured `dec 4 1k 2k` ends at 2000 on 42 and
1778.279410 on 47; `dec 10 1k 2k` ends at 2000 on 42 and 1995.262315 on 47.

`AC` and `Noise` reject DEC requests shorter than one whole step at
construction with `ValueError`, including equal start/stop frequencies.
For example, `dec 1 1k 9k` needs at least 2 points per decade; use more
points, a wider range, or `lin`. ngspice 42 AC runs away in memory for
positive sub-step spans; ngspice 47, Noise on both versions, and Xyce 7.10
AC/Noise return only the start point. The shared constructor rule refuses
these malformed requests on every backend. Zero-span DEC AC on
42 writes no data; 47 and Xyce write the start point. OCT requests did not
run away and retain their existing behavior. Custom/external decks bypass
this constructor validation and must not contain known runaway requests.
Xyce 7.10 DEC AC measurements show `dec 2 1 9` ending
at `sqrt(10)` and OCT noise `oct 2 1 3` ending at `sqrt(8)`.
ngspice 42 and 47 write only the start frequency for a two-point LIN AC
sweep (`.ac lin 2`), despite exiting 0. silk reports the missing requested
point; use three or more points or Xyce. This limitation does not apply to
two-point LIN noise or two-point DC sweeps.
Logs enrich the error reason; saved data decides completion. A rescued
operating-point failure followed by a complete transient is accepted.

**Unverified decks retain only the data-presence gate, with no endpoint
verdict.** This includes direct runner/backend decks, standalone rawfile
loads, modified decks, opaque custom content, caller control commands,
repeated analyses of one kind, `.step`, continuation cards, nonnumeric or
numerically unresolvable sweep requests, and nested DC. The current `DC`
API emits one sweep dimension; an external nested sweep is not certified.
There is no general control-language interpreter, and a card after `.end`
cannot create a request.

Opaque API inputs are `Module.add_custom_line()` (or direct `custom_lines`
entries) anywhere in the emitted hierarchy, caller-defined analysis subclasses
passed to `add_sim_type()`, replacement netlisters, and caller-defined
`to_spice()` emitters supplied through `add()`, `add_source()` or
`add_passive()`, including custom waveform emitters supplied to
`VoltageSource`/`CurrentSource`; custom `pdk_details.generate_netlist_entry()`
providers passed to `Module`; and `Module.generate_netlist()`/`generate_subckt()`
overrides on the top-level or children supplied through `add_module()` or
`add_submodule()`. Built-in source waveform data, including PWL knots, does not
remove certification. Operating-point-only runs use data-presence checks.

Custom text belongs to its owning circuit: a child's `add_custom_line()` text is
emitted verbatim inside that child's `.subckt` (after its elements, before
`.ends`), under the same validation as top-level text. It must be legal for the
simulator and for subcircuit scope (silk does not hoist, rewrite or parse it),
and emitting it does not make the deck certified or verified. Two same-named
children that differ only in custom text are refused unless `variants=True`,
which names each body separately.

A parsed rawfile with no analysis data rows raises `SimulationAborted`
(`data_rows=0`), including for unverified decks. Empty or malformed files
fail during parsing with `RawfileError`, a `SimulationError`.

## Arrayed SPICE instance spelling

Use the emitted instance spelling in raw SPICE references: an authored
submodule instance `X1[0]` emits as `xX1<0>`. This avoids ngspice dropping
device-parameter saves for bracketed instances. Authored port and net names
retain their spelling. Literal angle brackets in authored instance names
are refused.
ngspice 42 and Xyce 7.10 are covered by real arrayed operating-point/probe
regressions; Xyce already accepted bracketed instances, and uses the shared
angle spelling for consistent identity. Behavioral raw hierarchy uses it too.

Caller-authored `custom_lines` and B-source expressions are not rewritten.
Update raw instance references in them to the emitted spelling. PWL voltage
sources change their two physical instance names while retaining the private
intermediate net's original spelling. See [arrayed probe paths](probes.md#arrayed-instance-names-and-raw-paths).

## Junction stress coverage is limited by ratings and saved terminal data

An empty `check_junction_overrange()` list means no violations were found.
It can also mean no simulation data was available; it does not prove that
every device was checked. Use
`silk.circuit.junction_check.junction_stress_coverage(module)` to inspect each
device and the independent full/partial/none and failed counts. The STR-001 lint rule
and lint, coverage and circuit reports expose these counts and reasons. Only
PDK catalog device types are counted. Ideal R/L/C loads, controlled sources
and other non-catalog types are excluded; a catalog device lacking ratings
still reports `no_ratings`.

SPICE needs every terminal waveform for a device; omitting a supply with
`set_save_list` leaves that device unchecked (`no_waveform`). Save all required
terminal nodes before simulation. AC/noise data cannot establish absolute
junction stress (`small_signal`); absent rating keys (`no_ratings`), mismatched
waveform lengths (`length_mismatch`) and all-ground SPICE devices (`all_ground`)
also leave checks incomplete. Behavioral stress still scores top-level
terminal pairs only; nested devices are explicitly unavailable, not measured
by their modeled block's output. A device can fail a rated pair while other
pairs are unchecked, so failed and partial counts can overlap.

## Eye measurements do not detect population overlap

`eye_height(sig, period=ui, threshold=y)` measures the gap between the nearest
traces bounding the explicit threshold at the centred sampling instant. It does
not know the transmitted bits and cannot distinguish a genuine opening from a
small spacing inside overlapping populations. **Specs need a meaningful positive height margin, not merely `> 0`.**

No measurement window or automatic level discovery is used. `eye_width` measures
horizontal clearance at the same supplied threshold, not a BER-qualified eye.
Unavailable or sensitive centering still gives NaN measurements; a finite centre
without a warning does not certify correct alignment or an open eye. See the
[eye measurement definition](metrics-library.md#eye-centering-and-original-samples).

## Operating temperature

Managed fixed TEMP is verified at run time on ngspice and Xyce; every other
backend refuses a temperature request rather than labelling it applied. A Xyce
run that configures no TEMP reports temperature unknown, because silk neither
set nor observed it, and ngspice's `.temp` card is not portable to Xyce. Custom SPICE is conservatively opaque: a managed TEMP request with custom
lines fails, and an unmanaged custom deck reports unknown temperature.

Supply voltage is configured and confirmed with one bounded
claim: the check reads the deck that was launched — the line declaring the
designated source and the DC value on it — so a report proves what the
simulator was *given*, not what the rail settled at. No backend reports a rail
the way ngspice and Xyce report ambient TEMP. A supply silk cannot resolve (no
voltage source, several equally supply-shaped ones, a waveform source, or a
non-literal DC value) is recorded as unknown and refuses any setter or
`run(voltage=)` assertion rather than guessing; name the source to settle it.
`supply_voltage_v` describes one designated supply. When that supply is
swept, `supply_kind='sweep'` and `supply_sweep` records its DC bounds instead
of a scalar. A multi-rail operating point is not representable — the other rails are physically wired and belong in declared
test parameters. See
[operating conditions](api-reference.md#operating-conditions).

## Operating conditions: none versus unknown

Corner, temperature and supply are each a value, **none** (provably does not
apply), **unknown** or (temperature/supply) **sweep**; see
[the four states and the consumer contract](api-reference.md#operating-conditions-none-unknown-and-the-consumer-contract).
Verified limits:

- **None is a proof, so it is conservative.** Any `add_custom_line` text or any
  element silk does not positively recognise as ideal defeats `has_no_pdk_device`,
  and so also the supply-none rule: the run records corner and supply
  *unknown*, not none. A DC bias source (a non-zero or parameter-reference
  `dc`) also keeps supply unknown, because silk does not decide whether it is a
  supply; `set_supply_voltage(...)` makes it a value.
- **A schema-v4 database migrates NULL to unknown, never none.**
  `migrate_v4_to_v5` cannot re-prove a PDK-free run from a row that keeps no
  deck, so an old ideal-RC row stays unknown, and an old PDK-free row that
  recorded `'tt'` stays `fixed` / `'tt'` with `corner_basis='legacy'`. Select
  none evidence by re-running. `latest_only=True` shows a pre-upgrade row beside
  its re-run once.
- **External evidence is the caller's word.** `evaluate(rawfile=...)` records
  an undeclared corner as unknown and forms no profile; a declared none has
  basis `caller-declared`. silk cannot attest that a rawfile came from a deck it
  prepared.
- **`SpecReport.corner` is a label.** It names the corner requested, even when
  the evidence is corner none. Code that read it as "the corner that ran" must
  read `operating_conditions` instead.
- **A sweep never forms a profile**; neither does unknown.

## Waveform timing controls

The following limits apply to emitted source syntax on **ngspice 42** and
**Xyce 7.10.0**, separately for voltage and current sources:

| Control | ngspice voltage | ngspice current | Xyce voltage/current |
|---|---|---|---|
| Plain PWL points | Supported | Supported | Supported |
| PWL suffix `td=` | Supported | Parser rejects it | Accepted but produces zero in the measured fixture |
| PWL suffix `r=` without delay | Supported | Parser rejects it | Supported, but exact discontinuous joins differ from ngspice/NumPy |
| PWL `r=` with `td=` | Supported | Parser rejects it | Produces zero in the measured fixture |
| Pulse finite `np` | Supported; holds `v1` after exhaustion | Same | Count ignored; pulses continue |
| Pulse with omitted count | Supported | Supported | Supported |

**Current-source PWL timing requires ngspice 47 or newer.**
The [ngspice 47 release NEWS](https://raw.githubusercontent.com/imr/ngspice/ngspice-47/NEWS)
adds both `td` and `r` to the current-source PWL implementation. Thus **44.2
and 45.2 are below the required release**, including browser engines based
on 45.2. Browser engines running an exported deck themselves bypass
silk's selected-ngspice check; they must enforce their own compatibility.
Local ngspice 47 measurements of `PWL(0 0 1n 1m)` driving 1 kΩ give
0.5 V at 1.5 ns and 1.0 V at 3 ns for `td=1n`. For `r=0`, the
measured values are 0.5 V at 1.5 ns, 0.5 V at 2.5 ns, and 0 V at
3 ns. The last point is a discontinuous repeat join; its sampled endpoint
is sensitive to floating-point time placement. The 2.5 ns sample confirms
repetition away from that join.
The table above records ngspice 42; its current-source refusals do not apply
to ngspice 47 and newer.

Silk rejects current-source PWL `r`/`td` when the selected ngspice is older
than 47 or its version cannot be determined. The `NotSupportedError` names
the detected/unknown version, required release, and `SILK_NGSPICE_BIN` setting.
For an external ngspice release, use `generate_netlist(ngspice_version='47')`
or `tb.prepare(ngspice_version='47')`; silk refuses to run that deck locally
if the installed release differs or is unknown.
The probe uses the runner's binary selection and caches by canonical executable
path for the process lifetime; restart after replacing a binary in place.
Supported decks emit unchanged, and `export_source_schema()` continues listing
both parameters regardless of the local simulator. Voltage-source support and
NumPy delay/repeat semantics are unchanged.

Silk rejects unsupported combinations during netlist generation, including
nested sources and generic `waveform=` slots. Xyce
repeat-only PWL is permitted when repeat-start and final values are equal.
Discontinuous repeats are refused: Xyce retains the final value at every exact
join, whereas ngspice/NumPy select the repeat-start value after the first
traversal. Raw
`source.to_spice()` remains a backend-neutral formatter; its output alone is
not a support check. Explicit PWL points without `r`/`td` remain the portable
alternative. The NumPy model supports PWL delay/repeat and finite Pulse counts;
finite counts require explicit positive width/period, finite levels/times,
nonnegative delay and nonoverlapping pulses. It refuses analysis-dependent
width/period defaults and overlapping finite-count pulses instead of guessing.
PWL duplicate or negative point times and nonfinite coordinates are refused
by NumPy evaluation. With no finite Pulse count, NumPy uses omitted
`pw=0` and `per=0` literally: one non-repeating pulse with no high-level
hold interval. The SPICE card also writes zero, but the simulator interprets
it using analysis-dependent defaults. Set explicit `pw` and `per` when
matching behavioral and SPICE results.

## Slow sky130 library loading on ngspice 45 and 46

ngspice **45, 45.2 and 46** can take several times longer to load the
full sky130 library than 42, 44 or 47. The delay depends on the machine. Each included file appends its directory to
`sourcepath` without removing duplicates, then re-evaluates the growing list.
**ngspice 47** filters duplicate entries and avoids this slowdown.

Use **ngspice 44 or 47 or newer** with sky130 to avoid this slowdown.
silk emits `NgspiceVersionWarning` for the affected releases through its
version advice; simulations can still run.

## Simulator backends

**ngspice is the release-validated simulator. Xyce is a preview backend in
0.2.0.** Select it with `mod.simulator = 'xyce'`. Its optional
integration tests do not establish full parity with ngspice.

**Covered by Xyce integration tests:**

- Source-current probes, `tb.probe('i(<source>)')`.
- Managed fixed temperature, confirmed at run time.
- gf180mcu simulation on Xyce through the PDK-shipped Xyce model set
  (gf180mcu itself stays a preview PDK).
- Operating-point tables as *partial* tables with explicit holes,
  and device-parameter probes in the columns Xyce reports.
- Netlist emission, run files and simulator selection per testbench.

Single-analysis transient, AC, DC and noise runs have readback tests,
including source-current comparisons. Coverage remains specific to the tested
circuits and saved quantities.

**Xyce limits and portability concerns:**

| Item | Status |
|---|---|
| Op-table `gmbs`, `ft` and capacitance columns | **Xyce limit**: Xyce exposes only `id`, `vgs`, `vds`, `vbs`, `vth`, `vdsat`, `gm` and `gds` (`XYCE_OP_CAPABILITY`), so these cells are holes by design. Capture works on Xyce with `Op()` as the only declared analysis; see [operating-point.md](operating-point.md#xyce-support) |
| Transient noise | **Xyce limit**: there is no transient noise source |
| More than one analysis in one deck | **Xyce limit**: silk refuses at emission with `SimulationError`; split into separate tests |
| Monte Carlo / statistics | Xyce sample decks emit `.SAMPLING useExpr=true`; Xyce refuses `set_mismatch(False)` on a block. Random expressions return the mean unless `.SAMPLING useExpr=true` is present |
| Custom/imported SPICE text carrying ngspice-only cards | **Xyce limit**: silk passes `add_custom_line()` text through verbatim. Xyce rejects ngspice control blocks and `.save` cards. Use syntax supported by the selected simulator; silk does not translate custom text. |
| Relative `.include`/`.lib` paths in custom text | On ngspice, a relative `.include`/`.lib` resolves against the deck's folder or your working directory, so use an absolute path or one relative to where you run Python. **Xyce limit**: Xyce runs with the results directory as working directory, so `.include inc.sp` fails with `Could not find include file`. Use an absolute path. The deck argument and generated `.print file=` paths are absolute. |

A registered backend is not a support guarantee.

---

## Testbench report cache

`ModuleTestbench(cache=True)` is disabled in
0.2.0. Its fingerprint hashes device NAMES, specs
and the simulator only, so it cannot detect a changed device value, metric or
analysis — a cache hit could return a stale `SpecReport` for a circuit that
has since changed. Passing `cache=True` only emits a `UserWarning` naming
this at construction; the testbench then behaves exactly like `cache=False`,
so every `run()` simulates. The parameter and `clear_cache()` remain callable
so existing code does not break. A correct, fingerprint-aware report cache is not supported. See the `cache`
parameter in [the `ModuleTestbench` constructor](api-reference.md#moduletestbench).

## Code generation

Generated submodule entries with `parameters` still emit
`add_element(..., parameters=...)`, which Module children refuse at execution.
Use Python factories to configure distinct Module children before placing them.
See [the generator contract](codegen-contract.md#submodules).

## PDK integrations

| # | Issue | Notes |
|---|-------|-------|
| P1 | **GF180MCU and SG13G2 are preview/catalog integrations in 0.2.0** | Their manifests, device catalogs, corner/include assembly, and unit conversion remain available. They are not release-validated simulation targets: GF180MCU and SG13G2 have optional integration profiles. SG13G2 transistor simulation is limited to MOSFETs and needs compiled OSDI plugins and ngspice ≥ 44; ngspice 42 cannot load those plugins. Only sky130 is exercised by the release-gating SPICE profile. `silk.pdk.available()` means “registered catalog,” and a doctor's `[ ok ] pdk ...` means “installed files found”; neither is a broader support guarantee. |
| P2 | **sky130 MOS factories expose no fingers or junction geometry** | The accepted device parameters are `w`, `l` and `m`. Passing `nf`, `ad`, `as`, `pd`, `ps`, `nrd`, `nrs`, `sa`, `sb`, `sd` or `mult` raises `TypeError`. The ordinary 1.8 V NMOS and PMOS wrappers default `ad`, `as`, `pd` and `ps` to zero, underestimating junction capacitance and making timing optimistic relative to a circuit with its junction geometry included; gate-edge junction capacitance remains modelled. Extended-drain wrappers can supply internal junction geometry. The GF180MCU catalog declares `nf`; SG13G2 declares `ng`. |

## Database tracking

The complete current-schema workflow and compatibility contract are in
[`docs/database.md`](database.md).

| # | Issue | Notes |
|---|-------|-------|
| D1 | **Evidence tracking is deliberately scoped, not a run ledger** | The current database schema ([database.md](database.md)) stores circuit/spec identity, typed metric observations, first-class spec measurements, lineage, characterization profiles, and selected operating-current sets. It does not store raw runs, waveform samples, simulator logs, reconstructible Python, parasitic restore, or calibration restore — only evidence a caller explicitly chose to keep. |
| D2 | **JSON export is a lossy projection** | `export_to_json()` marks its format and omissions. It is for inspection/interchange, not a database backup or circuit archive; import recomputes identity and allocates new version/evidence UUIDs. An exact destination identity collision is rejected before evidence writes. Preserve the closed SQLite file for the canonical tracking record. Passives and sources are carried in their own lists, and import raises `DatabaseError` for a record it cannot use instead of skipping it (see [database.md](database.md)). |
| D3 | **Older-schema files require explicit copy migrations to the current schema** | Missing/zero-byte paths create a current-schema file and complete current-schema files open normally. Other versions, unversioned legacy, malformed, unrelated, and corrupt files are rejected without modification. Follow the [copy-migration chain](database.md#migrating-a-schema-v1-database) from your source version through the current schema, using a distinct, non-existing destination at each step and preserving originals. A v1 file needs every step in the chain, not only `migrate_v1_to_v2`; its evidence remains explicitly unscoped because v1 has no test-identity descriptor from which to derive a characterization profile. |

## Waveform source operating points

Every independent waveform source has two cards. `dc=` is an additive shift
for the entire signal: voltage sources connect the cards in series, current
sources in parallel. A zero shift still has its own card. `ac=` belongs only
to the DC card. A DC sweep replaces the shift, preserving the waveform's
initial value. Constant sources retain their original single card.

Xyce receives a derived DC clause on the waveform card equal to that
waveform's own `initial_value()`. ngspice uses its native waveform initial
value, without an authored DC override on that card. In particular, silk must
not put a derived DC clause on ngspice `TRRANDOM`: that changes its initial
value semantics.

| Waveform | Initial value | Xyce support |
|---|---|---|
| Pulse | `v1` | Supported |
| PWL / BitPattern / PRBS | First point, held before its first time | Supported subject to PWL timing limits |
| Sine | `voffset + vampl*sin(phase)` | Supported, including positive delay |
| Exponential | `v1` | Supported |
| SFFM | `vo + va*sin(phasec + mdi*sin(phases))` | Only resolved zero phases; nonzero or unresolved phases raise `NotSupportedError` |
| AM | Zero | Refused; use ngspice subject to its version limits |
| TransientNoise | Zero, including RTS | Refused |
| RandomSource | `param2`, without a random draw | Refused |

**ngspice AM/SFFM compatibility:** AM uses the selected release's
native card layout from the ngspice version table. Legacy releases before 43
use `VA VO MF FC TD PHASES`; releases from 43 use
`0 (va*vo) va mf fc td phases 0`, preserving silk's envelope and carrier.
The offset product resolves numeric/SI values and live Variables, or remains a
native card parameter expression. AM requires an identifiable release at or
above the table's minimum (42); unknown and older releases raise
`NotSupportedError` naming the version and source. Legacy AM phases must
resolve to zero because those releases ignore the phase token. On newer
releases `phases` is the modulator phase in degrees. Operating points, transient
starts, and delayed signals are verified on 42, 45.2, and 47 for voltage and
current sources, including generic `waveform=` slots.

SFFM remains restricted to the exact measured release 42. Newer voltage-source
cards start at zero even after correcting their parameter order; an explicit
DC clause fixes the operating point but not the transient start. Newer cards
also clamp modulation depth, and 45.2 reads current-source phases differently
from voltage-source phases. Silk therefore refuses these SFFM cards rather
than emitting a different signal. Select ngspice 42 with `SILK_NGSPICE_BIN`, or
use explicit PWL points. SG13G2 benches can use AM with a compatible newer
ngspice, but SFFM remains unavailable there. Xyce retains the restrictions in
the table above.

Phases are in degrees and SFFM modulation depth is in radians. Negative delays
and unresolved arithmetic needed to infer a waveform initial value raise
`NotSupportedError`; a nonzero `dc` does not bypass this validation.
Unphased level expressions and PWL first-value tokens remain SPICE tokens.
Multiple waveform slots are a `CircuitError`, regardless of backend.

**Current-source AC readback:** an ideal current source's AC current is its
`ac=` value; to measure it, place a 0 V `VoltageSource` in series (silk's
ammeter pattern). Silk raises `NotSupportedError` for logical or physical-card
current reads in AC, including probe, metric and annotation paths. Transient
and DC source currents are measured from saved simulator vectors; a waveform
current source sums both parallel operands. It never substitutes authored
values for missing data. A summed annotation has `current_vector=None`
because no single raw vector is its provenance.

Xyce `Op()` still uses a single-point sweep of a private grounded 0 V source.
Its collision loop checks opaque emitted instance names and private node
names; authored sources cannot use the reserved `__silk_op` name. No user source is swept. The private node and branch are
omitted from silk's explicit output list. Include-file contents remain outside
the collision check. Source private nets reserve the case-insensitive
`__silk_` prefix; designer instance names refuse `__silk_` anywhere, case-insensitively;
opaque emitted bodies still undergo collision checking,
while externally included content must honor that reservation itself.

## Behavioral Simulation

`simulator = 'behavioral'` supports the ordinary testbench workflow for
transient and DC: probes, metrics, specs, reports, rawfiles, plotting and the
viewer. `simulate_behavioral()` is a shortcut to the same backend, restoring
the prior simulator and analysis declarations. It includes the stop sample
and uses the shared reader's signal names and probe refusals. See
[one workflow, two engines](agent-guide.md#one-workflow-two-engines) for
runnable examples.

| # | Remaining limit or refusal | Behavior / remedy |
|---|---|---|
| B1 | **INOUT evaluation order** | `INOUT` warns that evaluation order is assumed; use explicit input/output directions. An unmodelled empty block raises `SimulationError`; an entered transistor without a model above it raises `NotSupportedError` naming its path. A model is a leaf, replacing its contents. |
| B2 | **Fixed transient step** | Step is `min(step, maxstep)` when capped. Evaluation starts at zero; samples before `start` are discarded afterwards. The grid includes `stop`, with a shorter last interval when needed. There is no adaptive stepping. |
| B3 | **Feedback across blocks** | Loops containing ports-form models run in placement order, with one complete call per vector model or one call per point per sample model. Unpublished feedback reads use reader-local initial values (zero by default); sample reads retain the producer's most recent publication. Undelayed transient loops warn once per run. `previous_value` explicitly delays a read by one transient point and can break the warning cycle; `initial_value` and delays never change the producer's waveform. Voltage, current and bus views share this interface. Mixed sample/whole-array loops raise `LoopError` whatever the port types: before calls when declared outputs show the loop, otherwise at the end of evaluation when actual INOUT reads and writes show it, before any result is returned or published. DC feedback raises `LoopError` at the end of the point, before publication; a model that only reads voltages can still run. Any DC `previous_value` read raises `SimulationError` immediately: branch on `silk.analysis()`. Setting `initial_value` is allowed and has no effect in DC. See [the behavioral workflow](agent-guide.md#one-workflow-two-engines). |
| B4 | **Transient waveforms without numpy models** | `Exponential`, `SFFM`, `AM`, `TransientNoise` and `RandomSource` raise `NotImplementedError`; use supported DC/pulse/sine/PWL/bitpattern/PRBS stimuli or SPICE. DC uses each waveform's initial value plus `dc`; a sweep replaces that source's `dc` shift. |
| B5 | **Calibration is not restored from storage** | Behavioral parameters remain an in-memory dictionary. Rebuild the model and its calibration bindings, restore the matching spec measurements, then call `reconcile_specs()`; storage does not reconstruct the model. |
| B6 | **Only defined quantities can be read** | Entered internal nets are readable; modelled-leaf internals, voltage-source branch currents, device-terminal currents/parameters and report `op_table()` raise `NotSupportedError`. Node current raises not-found `SimulationError`. Independent `CurrentSource` waveforms are available as current vectors in transient; DC writes only net arrays and the sweep axis, so source branch-current probes give not-found `SimulationError`; model output nets remain typed voltage, even if their author interprets the values as currents. |
| B7 | **Model-only blocks have no SPICE form** | SPICE deck generation refuses an instantiated block that has a model but no SPICE contents, with `NotSupportedError`. Add SPICE contents for a twin or select `simulator = 'behavioral'`. |
| B8 | **Faults and current annotation** | `FaultCampaign` refuses behavioral selection with `NotSupportedError`, including execution after a backend change. A campaign verifies only the bound testbench's `dut` placement (specs of other tests or placements are listed as not checked), reads each fault run's own rawfile, and keeps every rawfile of a batch until the batch is scored, so `archive_waveforms=False` does not bound disk use to the runs in flight; `reconcile_specs=False` runs report `simulated`, not a pass. Behavioural Monte Carlo supports declared `Normal`/`Uniform` model parameters; see the limits below. `annotate_currents()` retains `SimulationError`: device-terminal current annotation requires an electrical solve. |
| B9 | **DC evaluates points without settling** | Models receive single-sample inputs in sweep order; `t` is the increasing point index. Callable state can retain hysteresis; function locals reset each call. Unswept sources hold `dc + waveform.initial_value()`; swept sources hold the coordinate plus the waveform initial value. Voltage/current source sweeps support ascending, descending and single-point grids. A single-point real plot reads as `xlabel='op'`, `x=[0]`; the rawfile retains the sweep value. Missing source names raise `SimulationError`; temperature sweeps raise `NotSupportedError`. |
| B10 | **Ideal elements are approximations** | Outside modelled leaves, R/C loads to ground or declared supply rails are ignored; single-driver series R/L paths are wires; E evaluates gain times differential control voltage relative to its output reference. Dividers, ambiguous R/L paths, series C, G/F/H and B elements raise `NotSupportedError`. Rails are `gnd`/`0`, declared POWER/GROUND ports or net roles, or configured supplies; a DC stimulus alone is not a rail declaration. One warning per run and `SpecReport.summary()` list ignored/wired elements, also on rawfile reload. |
| B11 | **Analysis and configuration refusals** | AC, Noise, Op, ambient temperature, custom SPICE text, `save_device_currents` and `save_device_params` raise `NotSupportedError`. A refused declared analysis refuses the whole run. No declared analysis also refuses: add `Transient` or `DC`. Top-level `generate_netlist()` and `run_simulator()` refuse on the behavioral backend (no SPICE deck); subcircuit string emission remains available. Unknown simulator options raise `NotSupportedError`. |
| B12 | **Corners and worker counts do not add physical modeling** | Behavioral tests run in-process regardless of `n_workers`. Distinct corner labels warn once that runs are identical; evidence records corner/temperature none, not a simulated process corner. `evaluate()` preserves silk memory; `evaluate(rawfile=...)` records external evidence. Measurements and stored reports use the evidence source; a contradictory explicit storage source raises `EvidenceMismatchError`. Configured/driven supply evidence uses `configured-and-driven`; overriding the configured supply net with a different model value raises `SimulationError`. |

Ports models read voltages through `ports.VIN.v` and publish by assignment to
`ports.VOUT.v` or `ports.VOUT.i`. They return `None`. String-key access to the
ports object and returned values raise `SimulationError` with the supported
spelling. Explicit NumPy reads are detached copies; modifying a local copy
cannot alter electrical storage. Scalar assignments broadcast across the
selected grid; incompatible shapes fail when assigned. Unrelated model errors
retain their original type and message.

Actual voltage assignments are checked per net and sample. Two models writing
the same voltage raise `ConflictError` naming the net and both models, even
when the values agree. A repeated assignment by the same model raises
`RewriteError`; calculate a local value and assign it once. These errors and
`KCLError` inherit `SimulationError`. The internal current bookkeeping fills
one requested unknown pin current per net and sample with minus the sum of
other assigned currents, treating other unset currents as zero. Asking it to
fill a second unknown raises `KCLError`; this is not an electrical solver.
Independent current-source terminal contributions are `+I` at the positive
terminal and `-I` at the negative terminal. Wires pass voltage without
transferring current bookkeeping. Source orientation and passive eligibility
still follow declared directions before model execution.

Voltage sources enforce a difference in transient and DC: a 1 V reference
plus a 0.5 V source from `out` to that reference gives `out = 1.5 V`.
Keep the reference easy to evaluate. A source's negative terminal must be
`0`/`gnd`, or come from an independent voltage source, an E source, or a
modeled `OUT`, possibly through a single-driver series R/L wire path.
Reversed sources are supported when their positive terminal is the known
reference. Model producers must actually assign the required output at every
DC point as well as in transient. Resistor-derived references that require
a network solve, including a node grounded only through a resistor, refuse;
use SPICE or put the whole network in one Python behavioral model.

Unknown references, undriven rail defaults and two
already-owned voltage endpoints raise `NotSupportedError`. The diagnostic
names the source and its terminals. Even a numerically consistent extra
voltage constraint between two independently owned endpoints refuses: the
engine cannot validate electrical constraints. These are capability limits,
not declarations that the circuit is floating or electrically invalid.
Current sources never write net voltages or feed model inputs. Their waveforms
remain available in transient logical and physical branch-current records;
their signed terminal currents also contribute to internal pin-current bookkeeping. A net
connected only to current sources behaves as an ordinary undriven net, with
no additional current-source warning or refusal. A current source can share a
net with a voltage producer; the voltage stays with that producer. Ignored passive loads remain approximations and do not establish references.

<a id="signal-metrics-silkmetricsac--silkmetricsnoise"></a>

## Signal metrics

Import metrics from `silk.metrics`, including AC and integrated-noise metrics.

Low-frequency gain is the lowest-frequency sample, which may differ from DC
when the sweep starts too high. `dc_gain_dB` uses that sample regardless of
sweep order; `peak_gain_dB` returns the peak. `cmrr_dB` also uses each sweep's
lowest frequency by default, or a sample selected by `freq=`. It compares
magnitudes and ignores relative phase.

Stability margins describe the supplied frequency sweep. `phase_margin`
uses unwrapped phase and maps the result to [−180°, +180°). `gain_margin`
returns the interpolated negative-real-axis crossover margin closest to
0 dB, considering crossings in either direction. Use `gain_margins()` to
inspect every crossover: one number cannot describe both limits of a
conditionally stable loop. With no crossover, `gain_margin` returns +inf
if the highest-frequency magnitude is below one, otherwise NaN; neither
establishes behavior outside the sweep. See [the metric definitions](metrics-library.md).

<a id="bit-source-default-dc-asymmetry"></a>

## Bit-source DC values

`BitPattern` and `PRBS` follow the [waveform-source DC and AC rules above](#waveform-source-operating-points);
their initial value is the first bit's level. See the
[finite-source API](api-reference.md#finite-bit-pattern-and-prbs-sources).

## Netlist Emission

### Line breaks in caller strings

silk refuses a line break in any text it writes into a card it builds
itself. That means any character `str.splitlines()` splits on: `\n`, `\r`,
U+2028 and the rest. The refusal raises `silk.exceptions.DeckTextError`
(a `ConfigurationError` and a `ValueError`), naming where the text was going.

**The final text is checked, where it is written.** The check runs after
conversion: `os.fspath` for a `pathlib.Path`, `os.path.abspath` for Xyce's
output files, `str()` for any other object. Raw strings are also checked at
construction, before the SPICE number parser or `float()` strips a
trailing break, so `'1n\n'` is refused.

Covered:

- device, element and passive parameters, and the device-line prefix
- source `dc`/`ac`, waveform fields and PWL points
- controlled-source gains, `ctrl_vname` and B-source expressions
- `Variable` values, when rendered
- simulator option names and values, and analysis fields and point counts
- `module_name`
- the output directory and Xyce's raw, noise and temperature output paths
- PDK library, include, OSDI and Monte Carlo wrapper paths
- the parameter-preflight deck's model, parameter and internal address
- Monte Carlo switch names, and `showmod` selectors and record names
- saved vector and Xyce print terms, and corner names

A corner name also may not hold a control character, or be exactly `.` or
`..`. Any other name, including one with a separator, is percent-encoded
into one folder inside the test's own.

**Opaque content is written verbatim, line breaks included.** That is:

- `add_custom_line()` / `custom_lines`
- a source or waveform subclass with its own `to_spice()`, which may store
  multiline text in its attributes
- objects passed to `add()`, `add_source()` or `add_passive()` that carry
  their own `to_spice()`, including custom waveforms in a source's `waveform=` slot
- custom analysis subclasses, `generate_netlist_entry()` providers,
  replacement netlisters and `generate_netlist()` overrides

This content permits multiline SPICE and is not sandboxed; reserved-name and
collision checks do not make untrusted text safe. An ngspice
`.control` block in it runs under `ngspice -b`, and ngspice's `shell` command
runs a program. Never pass untrusted text there.

silk's own generated continuation lines (`+` lines in bit-pattern PWL) are
unaffected.

Not refused, because neither can start a new line:

- **Spaces in a corner name** (`'tt ff'`) write a malformed `.lib` card.
- **A `"` inside a library or include path** ends the quoted path early.

`run_corners()` records a refused corner as that corner's error, like an
unknown corner, and writes nothing for it.

## Output files and lifecycle

### Testbench directories follow the live placement

`run()` selects a deterministic instance/test directory using the full live owner path,
including testbenches built before composition. Unambiguous standalone
`test_<module>~<test>/` paths remain; escaping prevents name/delimiter collisions. Deck
and rawfile stay together with ngspice's `output.log`. See [API output
lifecycle](api-reference.md#testbench-output-directories-and-run-labels). Standalone
`Module.simulate()` exposes its actual rawfile path as `results_file`; explicit
overwrite mode writes `<dir_folder>/rawfile.raw`.

### Fresh run folders; reports hold live Modules

Default `set_output_options(mode='new')` retains a persistent unique child per prepared
deck, including standalone, corner and optimizer launches. Explicit `mode='overwrite'`
reuses the stable instance/test/label directory, losing prior same-location outputs on
an actual attempt even if it fails. Tests read their live owner at run time; recursive
`run_all_tests` uses its root mode for that invocation without changing child settings.
Labels select the stable base. Save successful `tb.results_file` paths for later
explicit readback; reports are not waveform snapshots. The report cache is disabled (see
"Testbench report cache" above), so every run allocates output as usual. Manual new-mode
launches require a newly generated deck each time. Fault campaigns still honor
`archive_waveforms=False`, deleting each rawfile after it is scored. No retention
scheduler, DB waveform archive or concurrent mutable-Module guarantee is provided. See
[API output lifecycle](api-reference.md#testbench-output-directories-and-run-labels).

### Example output needs manual cleanup

Examples write into `silk-output/` **under the directory you run them from** (Delete outputs you no longer need; archive reports and rawfiles you want to retain), and some examples also leave scratch
directories under the system temporary directory. The `SILK_EXAMPLES_OUT` setting
redirects example output; see [the example output guide](cli.md#a-note-on-silkexamples).

### Diagnostic logs

Launched ngspice and Xyce attempts publish their diagnostic log to `<dir>/output.log`
(or the configured public logfile path), including on timeout, non-zero exit, abort and
cancellation that unwinds the runner. Xyce stdout/stderr are captured there. Post-launch
`SimulationError` messages name that public path and `error.log_path` exposes it without
parsing; the optional attribute defaults to `None` for errors without a run log. The log
persists until the next attempt clears it or explicit cleanup removes it. Worker
failure-log coordinator calls have a one-second deadline; SIGTERM cleanup has a
five-second hard-exit watchdog. If the coordinator is unavailable or publication is
refused (for example, another writer creates the public log or an output target becomes
unsafe), the original failure or cancellation is preserved. Simulation errors name the
refusal and retained private log in `.silk-generation-*`, with `log_path` pointing
there. If no diagnostic log was written, the error says so and `log_path` is `None`.
Uncatchable termination cannot run publication cleanup.

### Failed launches invalidate current results

Immediately before each ngspice/Xyce launch, the runner clears the concrete expected
public result files (including applicable Xyce noise CSV) and diagnostic logs, privately
retaining artifacts still leased by old readers. Validation and preflight refusals leave
prior public results intact. A failed launched attempt publishes only its diagnostic
log, never rawfile, noise or temperature results. Failed attempts invalidate current
waveforms, implicit reload paths and current context; held generation readers, earlier
scalar reports, Spec measurement history and committed DB evidence remain. See [API
output lifecycle](api-reference.md#testbench-output-directories-and-run-labels).

### Lazy rawfile reads still need backing storage; default saves can be large

SPICE run readback and `ModuleTestbench.evaluate(rawfile=...)` load vectors on first use
and cache copied arrays per reader. Additional benches therefore retain the vectors they
use, rather than a full private rawfile copy each. Managed reruns preserve leased old
artifacts on disk so held result readers can finish reading their generation; retained
readers/reports can keep that disk space in use. Each failed attempt leaves a hidden
`.silk-<id>.cir` deck and `.silk-generation-*` directory containing any unpublished
artifacts. Its diagnostic log is published as described in "Diagnostic logs" above. They stay as the
diagnosis until a later attempt in the same output folder is accepted, which then
removes the earlier failed decks and directories (including a retained private log that
an earlier `error.log_path` named); a generation still held by a reader is never
removed, and a removal that fails is retried on the next success. Only failures known to
the running process are pruned: leftovers of a process that exited or was killed stay,
and automatic crash cleanup is not provided. Managed object views support one-shot HTML
(`serve=False`); `serve=True` refuses because the file watcher cannot preserve a
retained generation. Changed or missing external backing raises a typed error on an
unread-vector access. Reading every vector, full waveform exports and browser payloads
can still be expensive; the public `read_rawfile()` remains eager unless `lazy=True` is
requested. The default ngspice `save all` also writes device-internal body nodes
(`#body`, `#dbody`, `#sbody`); these can account for much of a rawfile. Use
`mod.set_save_list(...)` to reduce saved output, or, on ngspice 45+ with verified
content, set `dut.omit_device_internal_nodes = True` to keep circuit nodes while
omitting device body nodes; see [the strict admission
boundary](simulator-save-list.md#dropping-device-internal-nodes). The default remains
`save all`.

## Parasitic injection

Placed modules are live references. Injecting into a child before
or after placement affects the parent's next netlist and its testbenches'
next runs. Re-inject or rebuild explicitly when the layout estimate changes.

`inject_parasitics(mode='rc')` adds a lumped RC **loading branch**. All
original terminals remain on the original net. R connects that net to a new
midpoint; C/2 connects each of those nodes to `return_net` (default global
ground `'0'`). The return net itself gets no branch. Only recorded injected
objects are removed on re-injection; unrelated nets remain. Only the added R and
second capacitor touch the midpoint. It does not insert resistance between
original terminals or model DC wire drop, branch-specific RC, or routed
inter-terminal delay. The OTA bandwidth demonstration verifies added loading.

`c_only` adds one lumped shunt capacitor with unchanged original connections
and no resistor or midpoint. It is useful for first-order capacitive-loading
estimates; neither mode offers signoff or extracted-layout accuracy. See the
[exact injection network and units](api-reference.md#inject_parasiticsmodule-estimate-moderc---none).

Connectivity (`Net.pins` / `PortBinding`), signed per-pin current summaries
with vector provenance, conservative net current aggregates, and
`HalfCircuit`-matched estimates are available. Injection does not choose a physical branch topology, map terminals
to separate RC nodes, or distribute RC between terminals.

## Monte Carlo

Behavioural Monte Carlo requires deterministic model callables and at least
one active `Normal`/`Uniform` declaration. It draws once per run, holding that
value across DC points and analyses; it does not model time-varying noise.
Normal tails reach approximately ±8.2 sigma under draw scheme 1. Process corners
are labels only and repeat identical chips. Replay checks explicit model parameters,
declarations, sources and conditions; closure values, globals and other external
Python state are outside the deterministic-model guarantee.

When every sample of a declared Monte Carlo plan fails, or every corner of a
declared corner plan fails, the bench raises `SimulationError`. Its `failed`
attribute retains the ordered condition/sample failure map, including sample
seeds, through workers and the recursive runner's `TestFailure.exception`.
Explicit `run_corners` has a different partial-failure contract; see
[the run API](api-reference.md#moduletestbench).

See [Monte Carlo](monte-carlo.md) for the supported workflow. The following
limits apply; unsupported requests raise errors where described:

- **Several samples per process is ngspice-only.** By default one simulator
  process runs per sample; `samples_per_process` ([monte-carlo.md](monte-carlo.md#several-samples-in-one-ngspice-process))
  shares one ngspice process between consecutive samples. Xyce, behavioral and
  custom emitters or netlist content, `save_device_params`, OSDI PDKs
  and `save_device_currents` refuse it.
- **No timeout in Monte Carlo today, and none per sample in a shared
  process.** A Monte Carlo campaign passes no timeout to the simulator, so a
  stuck sample is not stopped in either mode. A timeout supplied at the
  low-level launch is a budget for the whole shared process (the per-sample
  value times the chunk's samples), not a per-sample watchdog: one hung
  sample can use the whole budget, and the samples after it in that process
  are then rerun one per process.
- **Preparation memory grows with `samples_per_process` times `n_workers`.**
  silk prepares every sample of a chunk before its process starts, and
  `n_workers` chunks are prepared at once. Size `samples_per_process` and
  `n_workers` together for the available memory.
- **No W/L check for the process section.** silk does not check that a
  device's width and length lie inside the range the PDK's process model is
  valid for; a `'process'` or `'both'` result outside it is not flagged.
- **Mismatch off per device is not supported.** `set_mismatch(False,
  devices=[...])` raises `NotSupportedError`; per subcircuit block works on
  ngspice.
- **Mismatch off per block on Xyce is not supported.** Xyce ignores the
  subcircuit-local switch, so the request raises `NotSupportedError`.
- **Samples are not stored in the database**, and `run_test(store=True)`
  raises for a test with Monte Carlo declared.
- **`correlate=` quantities are recorded but not ranked**:
  `MonteCarloResults.sensitivity` raises `NotSupportedError` for them.
- **Sample files remain after a campaign**, but another overwrite-mode
  campaign replaces them; choose `output_mode='new'` to preserve campaigns.
- **Two samples can share a simulator seed (not detected).** The simulators
  take a positive 31-bit seed, so each sample's seed (NumPy `SeedSequence`
  keyed by the base seed and the sample index, `derive_sample_seed`) is folded
  into that range. Two samples of one campaign then land on the same seed with
  probability about n²/2³²: roughly 0.001 % at 200 samples, 0.02 % at 1,000
  and 2 % at 10,000. Measured example: base seed 0, samples 50,699 and 55,170
  both get seed 1479113579. Such a pair simulates the same chip twice.

## Future Features

Co-simulation (time-step-coupled SPICE and Python), frequency-domain
behavioral simulation, and fitting behavioral models from characterization
are not supported.

## Source ground connectivity

Use `silk.simulation.validate_simulation_setup(module)` to check the assembled
circuit before running it. A clean return does not guarantee that the
simulator will converge.

Generated ngspice/Xyce runs and behavioral execution check source wiring
groups before launch and before replacing previous results. A source group
without actual ground raises `SimulationError`, naming the source paths and
resolved nets and suggesting a ground connection. This check is always on.
The public entry point is `silk.simulation.validate_simulation_setup(module)`
(or `ModuleTestbench.validate_simulation_setup()`), which raises the first
covered static refusal with the same type and message as a run, or returns
`None`, without a simulator, installed PDK models, emitted files or evaluated
source values. It does not check installed models, compiled plugins, deck
freshness or runtime behavior; see the
[API reference](api-reference.md#validate_simulation_setupmodule). The
inspection that backs it is internal and not a stable entry point.

Every element joins all its physical pins in this permissive wiring graph:
R/L/C, independent sources, controlled sources including voltage-sense pins,
diodes, transistors, switches, and subcircuit/modelled-leaf ports. A capacitor
path to ground counts regardless of the analysis. F/H current-control source
names are branch references, not additional node pins. B-source expression
references are not parsed into wiring. Placement-local ports and bus bits
are resolved before joining; unrelated internal nets and repeated placements
remain distinct. Behavioral modeled leaves contribute their ports instead
of their replaced physical contents; SPICE visits its actual contents.

Node `0` is global at every depth. ngspice also folds `gnd` case variants;
behavioral execution recognizes its `0`/`gnd` globals. Xyce retains its
existing refusal of `gnd` spelling rather than treating it as ground. A
`vss` name, supply/ground role or negative-terminal designation alone never
grounds a group. Ground elsewhere in the circuit does not ground an island.
Formal block ports named `0` or `gnd` (case-insensitive) are refused; name the
port `vss` and wire it to parent ground. See [Authored names](#authored-names).

Passing means only that this wiring graph finds ground. It does not establish
zero reference voltage, a conducting DC path or solvability. Grounded
contradictory voltage-source loops, same-terminal sources and current-only
groups pass this check; the backend can still fail. Behavioral reference
computation and unsupported solving/feedback have separate diagnostics.

Opaque custom lines, additional external includes, replacement netlisters,
custom device/source/waveform/analysis/Module emitters and multiline element/option
content create no claimed graph edges. Inspection never executes those
emitters. Their possible touch set exempts affected groups from refusal;
overridden source selection/layout/terminal hooks and cached replacement
netlisters at every SPICE hierarchy depth have the same exemption.
Unbounded SPICE text can touch global nets and therefore exempts the whole
request. Ordinary shipped PDK devices and their manifest includes do not
exempt a separate floating source. External or fault-altered decks without silk's prepared-run metadata
are not checked against the live Module's wiring.
There is no topology opt-out.

## Placed-test simulation sharing

Hierarchy execution shares proven identical SPICE simulations within one
invocation. Behavioral callables, Monte Carlo samples, random sources and
opaque/custom deck directives are conservatively separate. No cross-invocation
cache is implied. Selected handles retain loaded run data; several distinct
runs therefore retain several waveforms in memory. Their filenames can be
reused in overwrite mode, so archive files explicitly when disk history matters.
Choose a placement with `top.get_test(name, instance=path)` for numeric reads
when a child has several distinct results. Shared calibration parameters cannot
accept conflicting placement measurements; reconciliation raises before writing
any parameter.
