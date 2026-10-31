# silk internal architecture

**For maintainers and coding agents extending silk's internals.** This page says what
each subpackage owns, how data moves through the design loop, where important rules are
enforced, and where to add new functionality. It describes what **is**; what silk will
grow into is in [capability-roadmap.md](capability-roadmap.md).

It is not the place to learn how to *use* silk. For that, start at
[agent-guide.md](agent-guide.md) (workflow, decision guides, worked spine,
example index) and [api-reference.md](api-reference.md) (signatures). The
repo-level contract for agents is [../AGENTS.md](../AGENTS.md); the
golden-test rule and contribution mechanics are in
[../CONTRIBUTING.md](../CONTRIBUTING.md). This page links to those rather than
restating them.

**Maintenance rule.** This document is updated **in the same PR** as any change
that adds or removes a subpackage, moves a responsibility between packages,
changes an identity (structure hash, spec key, profile, current set), or adds,
removes or relocates an invariant's enforcement point. Two guards keep it
honest: `tests/test_architecture_doc.py` fails when a `silk/*` package has no
row in §1, when a cited repo path no longer exists, or when the cross-links to
the agent guide break; `tests/test_unit_authority.py` enforces the unit
invariant in §3.

Sections: [1. Package map](#1-package-map) · [2. Data flow](#2-data-flow) ·
[3. Invariants](#3-invariants) · [4. Seams](#4-seams) ·
[5. Internal gotchas](#5-internal-gotchas) · [6. Where does X go](#6-where-does-x-go)

---

## 1. Package map

### Layering

The central object is `Module` in `silk/circuit/`. Devices and sources describe its
contents; backends run it; signals, metrics and specs interpret the results. Database
tracking and layout estimation consume the same design and results.

These are responsibility boundaries, not a strict import hierarchy. Several packages
import one another. Keep imports of circuit objects inside functions where needed to
avoid cycles; inspect the existing import path before adding a new dependency. In
particular, `silk.signals` and `silk.metrics` depend on each other, and `silk.specs`
uses operating-condition types from `silk.simulation`.

Two boundaries are easy to miss: **simulator-specific emission lives in
`silk/circuit/`** (`spice_netlister.py`, `xyce_netlister.py`), not in
`silk/simulation/`; and **the structure hash that `silk/circuit/` depends on is computed
in `silk/db/`** (`identity.py`, `structure_tracker.py`), imported lazily at the circuit
call site to avoid a circular dependency.

The package map below has one entry per top-level subpackage and per top-level
module. Each names its responsibility, the entry points a maintainer starts
from, and what it must **not** do.

### `silk/circuit/` — the design object and everything that reads or writes it

**Responsibility.** `Module` (the circuit), its element graph, nets, ports and
roles, subcircuit instances, testbench registration and running, SPICE/Xyce
netlist emission, current annotation, reports and diffs, and the composition
graphic. It is the hub: almost every other package is either below it or a
consumer of it.

**Entry points.** `Module` in `silk/circuit/modules.py` (the class that inherits eight
part-mixins, one per file: `modules_structure.py`, `modules_faults.py`, `modules_db.py`,
`modules_testbench.py`, `modules_netlist.py`, `modules_results.py`, `modules_sim.py`,
`modules_spec.py`; `SubcircuitInstance` and the active-module stack live in
`silk/circuit/_module_base.py`, while `Variable` and its expressions live in
`silk/circuit/variable.py`). `ModuleTestbench` in `silk/circuit/module_test.py`.
`CircuitGraph` in `silk/circuit/circuit_elements.py` is the sole owner of the element
graph; the `.subckt` registration function `register_module_definition` lives there too.
`Net` and `PortBinding` in `silk/circuit/net.py`; `Port`/`PortDir`/`PortRole` in
`silk/circuit/ports.py`. `PySpiceNetlister` in `silk/circuit/spice_netlister.py` and
`XyceNetlister` in `silk/circuit/xyce_netlister.py`. `emitted_instance_name` in
`silk/circuit/emitted_names.py` is the single source of truth for the instance name silk
emits per element. `CircuitReport`/`generate_report` in `silk/circuit/report.py`,
`diff_reports` in `silk/circuit/report_diff.py`. `RoleRuleset`/`infer_port_roles` in
`silk/circuit/port_roles.py` (matching engine only; the shipped opt-in patterns are data
in `silk/circuit/data/port-roles.default.yaml`). `module_graph`/`graph_to_mermaid`/
`save_graph_png` in `silk/circuit/composition.py`.

`PlacementDeviceView` in `silk/circuit/device_view.py` retains the existing
`SubcircuitInstance` owner and the original leaf key, entry and device. Reads reuse the
owner's `_placement_chain()` validation, then check leaf identity; the view keeps no
separate ancestry inventory. Its live read-only mappings validate through that same
owner. The geometry adapter in `silk/pdk/basepdk.py` lazily imports the circuit
inspection helper so geometry reads accept placement views. This preserves the lazy
PDK-to-circuit dependency.

**Must not.** Part-mixins must not gain runtime state or class-body annotations, and
helpers must not become `Module` methods casually (the frozen `dir(Module)` golden, §3).
`_module_protocol.py` is never imported at runtime. `_module_base.py` holds shared
construction helpers without importing `Module`. `emitted_names.py` must not re-derive
naming rules the netlister/factory own. `interface_check.py` and `junction_check.py`
hold no detection logic of their own (they dispatch into `silk/lint/`). `composition.py`
adds no data and mutates nothing. The element graph is never aliased onto `Module`; all
mutation goes through `CircuitGraph`. `characterization_context` does not affect emitted
circuit text.

### `silk/elements/` — ideal elements and the device-class factory

**Responsibility.** Turns YAML device specs into Python classes whose
constructor arguments match pin and parameter names (`make_device_class` in
`silk/elements/factory.py`); the ideal R/L/C passives
(`silk/elements/passives.py`); `ElementLibrary` over
`silk/elements/analog_devices.yaml`; and the machine-readable **device**
schema export (`export_schema`, `module_schema`, `export_schema_json` in
`silk/elements/schema.py`) that inspection tools and code generators consume. Elements
auto-register with the active `Module` inside `with mod:` via a metaclass hook.

**Must not.** Import `silk.circuit` at import time — every such import is lazy
inside a method, which is what keeps `elements` below `circuit`. Interpret
units: `DeviceParamConstraint` stores the `pdk_unit` tag `BasePDK` handed it and
applies scales `BasePDK` computed (§3, unit authority). Source catalog entries remain available through schema export; construct sources with `silk.sources`.

### `silk/sources/` — sources, waveforms, and the source catalog

**Responsibility.** `VoltageSource`, `CurrentSource`, the four controlled
sources and the behavioral sources (`silk/sources/classes.py`); waveform
shapes `PWL`, `Pulse`, `Sine`, … (`silk/sources/waveforms.py`); the two
declarative surfaces `sources.voltage.pulse(...)` and `sources.vpulse(...)`
built by `build_source_surfaces` in `silk/sources/library.py` (same function
objects, aliases not copies); and the **source** schema export
(`export_source_schema` in `silk/sources/schema.py`). Params and defaults are
reflected from `__init__` signatures; authored metadata (symbol, pins, waveform
kind → class, unit hints) lives in `silk/sources/source_metadata.yaml`. Adding a
waveform kind is one class plus one YAML line; the package `__init__`
auto-exports whatever the YAML names.

**Must not.** Emit a zero-edge pulse: `Pulse` takes `tr`/`tf` as mandatory keyword-only
arguments because a zero edge is resolved differently by ngspice (ramps over `TSTEP`)
and by silk's behavioral twin (steps). Silently reduce an unmodelled waveform to DC in
behavioral simulation (`_reject_unmodeled_waveform` raises). Import circuit construction helpers at
module load time; those imports are local. Source-name validation imports
`silk.circuit.emitted_names` at module load.

### `silk/pdk/` — process design kits and the single unit authority

**Responsibility.** Discover shipped PDKs (`silk/pdk/registry.py` scans `silk/pdk/data/`
for `*.pdk.yaml` manifests), resolve model libraries against `PDK_ROOT` (`load_pdk` in
`silk/pdk/_loader.py`; order: explicit argument, the `PDK_ROOT` setting: environment
variable, settings file, then `~/pdk`), parse the device catalog into typed `PDKDevice`/
`ParameterConstraint` objects and mint factory classes (`BasePDK` in
`silk/pdk/basepdk.py`), convert between the author's unit and the model's unit
(`BasePDK._scale_factor`, `_param_scales`), and expose the one geometry accessor other
layers use (`device_geometry_um`) plus the deck-wide length unit a simulator may need
told (`length_unit_in_metres`, consumed by the Xyce netlister's `.options parser
scale=`). `load_pdk` defaults author units to the manifest's `user_units`; all shipped
manifests use `u` (micrometres for W/L). It resolves model paths without requiring the
model files to be installed. `silk/pdk/validate.py` is the static pin-arity/prefix
vetting CLI with an opt-in per-device smoke sim. Three PDKs ship: sky130, gf180mcu and sg13g2, each using
`BasePDK` with a manifest and device catalog; sg13g2 declares OSDI plugins. sky130 is the release-validated transistor
PDK; GF180MCU and SG13G2 are preview integrations.

**Must not.** Guess a length's unit: an untagged or unknown `pdk_unit` on W/L
raises rather than defaulting to µm. Compile OSDI plugins (that is environment
setup, like installing ngspice); silk loads them and fails loud at run time if
declared but missing. Accept duplicate device names in a catalog
(`_StrictSafeLoader` raises where PyYAML would last-wins). Scale catalog
defaults (they are already in the PDK's unit).

### `silk/simulation/` — analysis objects, backends, runners, gates

**Responsibility.** Everything about *asking a simulator for results*: the typed
analysis objects `Transient`/`AC`/`DC`/`Noise`/`Op` (`silk/simulation/analysis.py`); the
name-keyed `SimulatorBackend` registry that is the single dispatch point for simulator
choice, the shared `SpiceBackend` render/load verbs, and the
`NgspiceBackend`/`XyceBackend` classes (`silk/simulation/backend.py`); the subprocess
runners (`silk/simulation/runner.py`, `silk/simulation/xyce_runner.py`); the
data-presence gate `_require_data` and the certified-analysis completion check in
`silk/simulation/completion.py` (logs enrich failures); the `Corner` enum; and the numpy
behavioral engine (`silk/simulation/behavioral.py`). Behavioral scheduling,
mixed-contract loop admission and feedback-read diagnostics are enforced in
`silk/simulation/_behavioral_feedback.py`; INOUT admission checks do not add scheduler
dependencies. A backend owns the whole produce-a-Signal pipeline for one simulator:
`render_analysis`, `make_netlister`, `_launch`, `load_signal`, plus the optional
`device_current_vectors` and `has_data` seams. `SimulatorBackend.run` is a template
method: it launches the backend, checks for data, then checks completion for certified
silk-generated SPICE requests. A subclass that defines `run` is refused at
class-definition time. Opaque or external decks retain the data-presence check without
an endpoint verdict; see [analysis completion](known_issues.md#analysis-completion).
`silk/simulation/run_files.py` describes prepared artifact paths;
`silk/simulation/generation.py` manages retained run artifacts, and
`silk/simulation/executor.py` runs serial or parallel jobs. Behavioral modules use
`run_module` and `_launch_module` for their in-process launch.

**Must not.** Trust an exit code as proof of usable, complete data. Logs provide
diagnostics; the data and certified request determine completion. Own netlist text (that
is `silk/circuit/`) or waveform parsing (that is `silk/signals/rawfile.py`). Guess
device currents: `SpiceBackend.device_current_vectors` raises `NotSupportedError`.

The simulation package also owns result provenance and behavioral run context:

- `silk/simulation/provenance.py` — `RunProvenance`, the per-result record of what
  produced it: silk/Python/numpy versions, `run_origin`, the
  launched simulator binary and version, the prepared deck's PDK `.lib`
  file/section hashes, options, save list, deck sha256, seed. Built once per
  evaluation from facts `generate_netlist()` and `run_simulator()` leave on
  the module; environment facts are cached per process and every builder
  swallows its own errors. Carried by `SpecReport.provenance`, rendered by
  `silk.circuit.report`, diffed by `report_diff`, stored in `silk.db`'s
  `run_provenance` table. Field contract in
  [reports](reports.md#provenance).
- `silk/simulation/variation.py` — `Normal` and `Uniform` statistical declarations for
  behavioural model parameters, nominal resolution outside Monte Carlo, and
  the keyed draw scheme.
- `silk/simulation/_analysis_context.py` — the context variable naming the analysis the
  behavioral engine is dispatching. `analysis()` is re-exported from `silk`;
  `analysis_scope` is used by `silk/simulation/behavioral.py` and
  `silk/simulation/behavioral_backend.py`. Imports only `contextlib`,
  `contextvars` and `silk.exceptions`. Must not save the value as model state
  or behavioral identity.

### `silk/simulators/` — per-simulator configuration data

**Responsibility.** Not to be confused with `silk/simulation/`. This is a configuration
package: `ngspice.yaml`, `xyce.yaml` and `behavioral.yaml` describe options syntax,
presets (`default`/`tight`/`fast`), the known-options allowlist, the device-current save
directive and the terminal-current method for each simulator, including version policy.
`load_simulator_config` and `SimulatorConfig` in `silk/simulators/_loader.py` are the
only code. Netlisters, backends and condition handling consume these settings.

**Must not.** Launch a simulator. The loader uses shared deck-text validation and typed
errors; simulator execution belongs in `silk/simulation/`. Adding a simulator's *dialect
knobs* means a YAML here; adding a simulator's *behaviour* means a backend in
`silk/simulation/`.

### `silk/signals/` — signals, probes, rawfiles, metric schemas and the viewer

**Responsibility.** `Signal` in `silk/signals/base.py` is the simulator-agnostic return
type the whole analysis contract is written against; `NgspiceSignal` in
`silk/signals/spice.py` loads one from a rawfile, keying the x-axis off **vector 0
positionally, never by name** for swept analyses. Operating-point plots instead use a
synthetic `x=[0.0]` axis and `xlabel='op'`. `silk/signals/rawfile.py` is the MIT
clean-room nutmeg reader (`read_rawfile` → `RawFile`/`RawPlot`/`RawVector`; provenance
in [mit-nutmeg-reader-provenance.md](mit-nutmeg-reader-provenance.md)).
`silk/signals/schema.py` owns the `@metric` decorator and the registry it feeds; the
metrics it tags live in `silk/metrics/`. `ModuleTestbench.add_metric` and
testbench code generation use this registry. `silk/signals/kinds.py` defines accepted
signal kinds; `silk/signals/operations.py` describes the operations available on
deferred probes. `silk/signals/view/` is the self-contained waveform explorer
(`emit_view`, `emit_views_by_block`, `list_blocks`, the `--serve` watcher, the uPlot
assets); `Module.view` and `silk.view` are thin on-ramps that import it lazily.

**Must not.** Import additional silk packages into the viewer. Its allowed imports are
stdlib, numpy, `silk.signals.rawfile`, `silk._json_codec`, `silk.spice_names` and
`silk.vector_match` (I12). The reader must not text-decode the `Binary:` payload,
case-fold vector names, or reorder vectors. `silk.signals` must not grow a second metric
library beside `silk.metrics`.

### `silk/metrics/` — metrics grouped by purpose

**Responsibility.** A function library organised by what each measurement does:
`waveform.py` (any-axis tools), `statistics.py`, `timing.py`, `jitter.py`, `eye.py`,
`power.py`, `spectral.py`, `rf.py`, `converter.py`, `fom.py`, `ac.py` and `noise.py`.
`diagnostics.py` supplies measurement warnings. Every metric keeps its schema domain;
its input signal kinds describe the data it accepts. Waveform functions take
`Signal` inputs; closed-form formulas take scalars. `crossings` in `waveform.py`
provides shared edge detection; `spectral.py` supplies clock-sampled spectra.
Not re-exported from the top-level `silk` package.

Import shared input guards directly from the leaf `silk/metrics/_guards.py` in
`ac.py` and `noise.py`; keep both modules independent of the other metric
implementation files (waveform, statistics, timing, jitter, eye, power, spectral,
rf and converter). `silk.signals.schema.get_metric()` and
`export_metric_schema()` load the `silk.metrics` facade locally when called.
The leaf guards preserve this lazy dependency relationship.

**Must not.** Grow a second metric package outside `silk.metrics` or assume a
coherent-FFT measurement has removed spectral leakage. Windowed and Welch
estimators have their own sampling contracts; follow each function's documentation.

`@metric` from `silk/signals/schema.py` registers functions and declares units;
signatures supply parameter types and accepted signal kinds. Scalar-returning functions
form the exported catalog. Result-object functions can register with an explicit
`outputs={field: unit}` mapping: evaluation publishes dotted scalar fields, but schema
export omits those functions. Undeclared non-scalar returns and missing return
annotations raise `NonScalarMetricError`. See
[metrics-library.md](metrics-library.md#schema-export--which-metrics-are-in-the-machine-readable-catalog).

Malformed requests raise `ValueError` with a diagnostic. Degenerate simulation data
returns the function's documented result, such as `nan` or a finite failing floor, so
other corners can still be measured. See each function's failure contract for details.

### `silk/specs/` — the design-stage bridge

**Responsibility.** Spec declarations, measurements, evaluation attempts and
reconciliation. `Spec` (with explicit `key=` identity, `parent` hierarchy and an
operator/target), `Measurement`, and `CalibrationBinding` in `silk/specs/spec.py`;
`reconcile` in `silk/specs/reconciler.py` walks a spec list and returns a
`SpecReconciliationReport` (`silk/specs/report.py`); `resolve_metric` in
`silk/specs/resolver.py` resolves dotted `net.<name>.<attr>` / `device.<name>.<attr>`
metrics against a Module. `Module.reconcile_specs` in `silk/circuit/modules_spec.py` is
the caller. Measurement slots are keyed so a profile-scoped observation can never
collide with the legacy `source@corner` string key (`_measurement_slot_key`).
`silk/specs/attempt.py` records evaluation attempts, and `silk/specs/aggregate.py`
defines aggregate evidence. `silk/specs/evidence.py` selects measurements for a
placement without copying the shared spec declaration.

**Must not.** Add eager imports of `Module` to the declaration layer. Auto-create a
behavioral parameter during calibration (a missing key is a declaration error). Feed
behavioral measurements back into behavioral models (the default `source_filter`
excludes them).

### Operating conditions: four states

`silk/simulation/conditions.py` owns the one definition. `ConditionState` (a plain
`Enum`: fixed, none, unknown, sweep), `CONDITION_VOCABULARY` (states per fact, which may
scope a profile, basis words and their categories), `encode_condition` /
`decode_condition` (the one wire encoding) and `export_conditions_schema()`.
`OperatingConditions` is the immutable snapshot: kinds (`corner_kind`,
`temperature_kind`, `supply_kind`), bases, and `state(fact)` / `scalar(fact)`; its value
fields hold `None` for both none and unknown. None is recorded only on proof, by two
public functions that silk's launch path and external callers share: `has_no_pdk_device`
(corner none, over the whole hierarchy) and `has_no_supply` (the supply rule). The
behavioural engine records corner and temperature none. Downstream condition values
carry the scalar from the snapshot: `CharacterizationContext` (value or none forms a
profile; unknown and sweep never do), `Measurement`, `store_report`, database rows,
report JSON and `silk.provenance/v3`. Provenance separately retains a launch corner
label for [automatic deck
checks](database.md#automatic-deck-checks-through-a-built-design); that label does not
replace the condition scalar. The consumer contract and migration are in
[api-reference.md](api-reference.md#operating-conditions-none-unknown-and-the-consumer-contract).

### `silk/db/` — opt-in SQLite design history with derived identity

**Responsibility.** Enabled by `Module(db_path=...)`. `Database` in
`silk/db/circuit_db.py` owns the database operations; `CircuitTracker` in
`silk/db/tracking.py` is the delegate `Module` creates lazily (its methods map
one-to-one onto the `Module` wrappers in `silk/circuit/modules_db.py`).
`silk/db/identity.py` canonicalises and fingerprints structure (`silk.structure/v4`),
spec sets (`silk.spec/v1`) and characterization profiles (`silk.profile/v1`);
`silk/db/structure_tracker.py` extracts structure from a Module and computes the
structure hash the goldens and net-current persistence share. `silk/db/context.py` holds
`CharacterizationContext` (all six fields or none). `silk/db/schema.py` is read-only
file-format policy (the current schema version, see [database.md](database.md); tables,
foreign keys, the two partial-unique indexes that enforce one current set per selection
key); `silk/db/migration.py` holds the explicit copy migrations (never in place).
The successive copy migrations are `migrate_v4_to_v5`, `migrate_v5_to_v6` and
`migrate_v6_to_v7`; see [database.md](database.md) for formats and commands.
A corner, temperature or supply is a value, none or unknown; temperature and supply may
also be sweeps. Historical NULL values migrate to unknown, never none.
`silk/db/circuit_import.py` is the JSON/CSV/SPICE importer and exporter.

**Must not.** Accept a display label or precomputed digest for a profile (the
API takes only a context; identity is derived). Mutate schema in `schema.py`
(table creation is an explicit new-destination operation in `circuit_db.py`).
Return plausible evidence when applicability cannot be proved (typed
`Evidence*` errors instead). Touch netlist generation (`context.py` is Python
metadata only). Default a corner silently.

### `silk/layout/` — statistical layout estimation

**Responsibility.** `estimate_layout` in `silk/layout/estimator.py` computes
per-device footprint (`silk/layout/area_model.py`), a module bounding box,
Rent's-rule wire lengths (`silk/layout/wirelength_model.py`), EM-driven wire
width from peak current and `RoutingModel.j_max`
(`silk/layout/routing_model.py`), lumped R/C, and matched-half-circuit
equivalence classes (`silk/layout/matching.py`); `inject_parasitics` in
`silk/layout/parasitics.py` writes those parasitics back into the module as
real R/C elements so the next simulation feels them; `capture_layout` snapshots
the layout estimates stored on the module, nets and devices. Footprint numbers are declared heuristics (`packing_factor=2.5`,
`routing_overhead=2.0`), overridable per call and recorded on the
`LayoutEstimate`.

**Must not.** Interpret units. Geometry is read through an adapter over
`silk.pdk.basepdk.device_geometry_um`; unit conversion belongs to the PDK layer. Place
or route: there is no placer or router in silk; that work lives in layout consumers
(§4). Put `LayoutWarning`s on `Net` objects (they are run-scoped and live on the
returned estimate).

### `silk/constraints/` — device-relational layout intent

**Responsibility.** Typed, pydantic-v2 constraint declarations
(`SymmetricBlocks`, `SameTemplate`, `GroupBlocks`, `ProximityConstraint` in
`silk/constraints/types.py`), each inheriting the audit fields of `ConstraintEnvelope`
(`silk/constraints/envelope.py`): id, source, timestamp and
confidence. Each JSON object is flat. The constraint types follow the schema of the open-source ALIGN
analog layout project (BSD-3-Clause licence), so constraint files can be
exchanged with it. silk's implementation is its own. Shared device-relational
entries use the same type names and field shapes; silk's JSON also carries
`PortRoles`, `NetIntent` and `ProximityConstraint` entries that require
adaptation for consumers accepting only that schema
(`silk/constraints/serialization.py`). Importing the package monkey-patches
`Module.add_constraint`/`Module.list_constraints` onto `Module`
(`silk/constraints/_module_patch.py`) so this package is the single source of
that API surface. `silk/constraints/checkers.py` is an *open* dispatch table
(`register`/`unregister`) shipped with `NotImplementedError` stubs so an
external tool can supply checker logic without editing silk.

**Must not.** Own port semantics (those are `PortRole` on `silk/circuit/ports.py`)
or per-net budgets (fields on `silk/circuit/net.py`); both round-trip through
the same serialisation bridge. Pre-build a consumer-specific export filter before a
consumer needs one.

### `silk/lint/` — the analog design-rule framework

**Responsibility.** `Rule` base and the default and opt-in catalogs (`silk/lint/rules/`:
interface, domain, stress, structural, bus families), `RuleRegistry` in
`silk/lint/registry.py` dispatching rules over a module hierarchy and recording a
per-visit outcome (`PASS`/`VIOLATION`/`EXCLUDED`/`SKIPPED_NO_SIM`), YAML waivers and
scope exclusions (`silk/lint/waivers.py`, `silk/lint/exclusions.py`), and versioned
`LintReport`/`CoverageReport` objects with JSON/HTML/XML output and diffing
(`silk/lint/reports/`). Reports carry a `context_hash` and refuse to diff across
different PDKs.

**Must not.** Hide a waiver (every waived violation is still reported under
`waivers_applied`). Duplicate detection logic in circuit wrappers or reports.
`interface_check.py` and `junction_check.py` dispatch rules; `silk/circuit/report.py`
consumes lint reports and coverage.

### `silk/lint_markers/` — intentional-exception markers

**Responsibility.** `NoConnect` and `NoConnectPort`, attached via
`Module.add_no_connect`/`add_no_connect_port`, declare a floating net or an unwired pin
intentional (suppressing `STRUCT-003`/`STRUCT-001`) while still surfacing in the report.
They live outside `silk/lint/` because markers are *circuit data* that `silk/circuit/`
imports, while `silk/lint/` imports `silk/circuit/`; putting them in `lint` would close
a cycle. The marker types do not import circuit objects.

**Must not.** Import anything from silk. Accept a blank `reason`.

### `silk/faults/` — Monte-Carlo fault-injection campaigns

**Responsibility.** `Fault` (declarative data), `AppliedFault` (what was emitted
post-RNG) and `FaultManifest` (`silk/faults/types.py`); `FaultSet` with `sprinkle`
(`silk/faults/set.py`); the sugar `stuck_open`, `short_ds`, `vth_shift`, …
(`silk/faults/sugar.py`); `FaultCampaign`/`CampaignResult` (`silk/faults/campaign.py`)
running jobs through the shared executor and `run_one_for_executor` / `run_one`
(`silk/faults/runner.py`); `CampaignReport` (`silk/faults/report.py`). The realization
`_realize_fault` in `silk/faults/application.py` turns one fault into device changes —
instances to emit (none = removed) with PDK-unit parameter overrides and pin rewires,
plus added-element specs — and is called once per retained complete target during fault
preparation (`modules_faults.py::_silk_apply_fault_markers`), before deck text is
written. Preparation records each applied fault and stores its realization on a
temporary device copy; repeated emissions reuse that realization without draws or
records. Campaigns use shared execution, JSON and exception helpers; fault realization
itself reads the supplied device and PDK objects. The netlister
(`spice_netlister.py::_emit_faulted_device`) builds the silk objects and emits each
through the element's own writer.

**Must not.** Carry shared state in `run_one` (it must be pool-safe). Reuse a seed
across runs (`_derive_run_seed`). `silk/faults/pdk_metadata.py` reads `fault_params:`
bounds off `PDKDevice.fault_params` — parsed by `BasePDK`, never opened as a catalog
file here. The PDK it resolves against comes from the owning placement module through
its netlister (`modules_faults.py::_silk_apply_fault_markers`,
`owner.netlister.pdk_details`) as `_realize_fault(..., pdk=...)`. It reads the PDK
object duck-typed (`pdk._devices[name].fault_params`), never importing `silk.pdk`.

### `silk/optimize/` — parameter space and evaluation harness

**Responsibility.** `VariableSpace` in `silk/optimize/parameter_space.py` samples a
joint space over `Variable`s (random, LHS, Halton, grid); `Optimizer` in
`silk/optimize/optimizer.py` defines the evaluation protocol (`add_objective`,
`add_constraint`, `evaluate`, `sweep`) and logs every call under a six-tag failure
taxonomy. No optimiser algorithm lives here; an external driver or agent calls
`evaluate`.

**What a variable may sit on.** A `Variable` — and any expression over one, `vampl * 2`
— is accepted on **any device parameter, on a source's `dc=` or `ac=`, and on any
electrical waveform parameter** (`Pulse`, `Sine`, `PWL`, `Exponential`, `SFFM`, `AM`,
`TransientNoise`, `RandomSource`; `PWL` per point, on the time or the value).
Bit sequences, `RandomSource.type`, `BitPattern.repeat` and PRBS
`order`/`length`/`seed` require literals. It is
retained and resolved to its current value when the deck is emitted, so
`space.apply(point)` followed by `generate_netlist()` writes that point's number — which
is what makes the standalone sweep path (sample a grid, apply, simulate, measure) work
for a *stimulus* amplitude and not only for a device width. `_resolve_value` in
`silk/utils/_utils.py` reads their current values; device parameters use `_render_param`
and `_fmt`, while waveform classes preserve their string representation. The deprecated
passive constructors are an exception: `Resistor('r1', 'a', 'b', var)` raises
`TypeError`; use factory-generated `elements.resistor` for variable parameters.

**Must not.** Grow search logic or infer success from a metric's numeric value.
`Optimizer` consumes `SpecReport.metrics` directly; custom test objects can also return
`{'metrics': {...}}` from `run(corner=...)`.

### `silk/codegen/` — declarative spec → Python source

**Responsibility.** `Generator` in `silk/codegen/generator.py` turns a
`CircuitSpec` (dict/JSON from any tool; contract in
[codegen-contract.md](codegen-contract.md)) into a `Module` script,
choosing top-level deck or subcircuit by whether ports are declared;
`generate_testbenches` in `silk/codegen/testbench_generator.py` turns a
`TestbenchSpec` into testbench Python, failing loud with `TestbenchSpecError`
on any structural defect. Both validate every caller-supplied token before
emission so hostile names stay data.

**Must not.** Emit uncompilable Python. Raise a bare `KeyError` on a malformed
spec. Let `_ANALYSIS_KINDS` in the testbench generator drift from `_sim_code` in
the circuit generator (hand-synced, by comment).

### `silk/examples/` — shipped, runnable worked examples

**Responsibility.** A real installed package (wheel auto-discovery includes it), worked
examples run as `python -m silk.examples.<name>`; shared helper modules need not provide
a command-line entry point. Several are reusable building blocks other examples import
(`stdcells.py`, `cml_cells.py`, `strongarm_tb.py`). The human map is
[examples.md](examples.md); the API-level index is [agent-guide.md
§7](agent-guide.md#7-curated-example-index); the flagship is
`silk/examples/design_loop_walkthrough.py` (§2). Output goes to the directory returned
by `silk.examples.example_output_dir()`: the `SILK_EXAMPLES_OUT` setting (environment
variable, then the user's settings file), or the gitignored `silk-output/` under the
current working directory. See [settings.md](settings.md).

**Must not.** Import a sibling by bare module name, touch `sys.path`, run simulations
on import, or hard-code a `.lib` path (`tests/test_examples_importable.py`,
`tests/test_examples_no_raw_lib.py`). Import-time work must not require installed
model files (`tests/test_suite_runs_without_a_pdk.py` checks collection without them).
Packaged PDK catalogs may be loaded at import time.

### `silk/utils/` — shared formatting and validation

**Responsibility.** `_fmt` (SPICE value formatter with a plain-decimal band so
`l=0.5` never emits as `500m`) and `_parse_si` (its inverse, and silk's *one*
SPICE-number parser — I16; its docstring is the parse contract) in
`silk/utils/_utils.py`; `trapezoid` in `silk/utils/_compat.py`, the one copy
of the NumPy 2.0 rename shim. `format_spice_value` and `parse_spice_value`
are the public aliases (`silk.utils.__all__`); both delegate to the same
`_fmt`/`_parse_si` pair, so a second parse table is never justified.

**Must not.** Depend on circuit construction or simulation execution.
`silk/utils/deck_text.py` uses the shared `DeckTextError` exception to refuse line
breaks in values written into a SPICE card.

### Top-level modules

- `silk/__init__.py` — the façade. Public re-exports plus
  `__version__` and one function, `silk.view`, which imports the viewer lazily.
  Not re-exported here: `silk.metrics` (import it directly).
- `silk/exceptions.py` — the exception hierarchy rooted at `CircuitError`; imports
  nothing from silk. `ConfigurationError` marks malformed requests that
  multi-test runners must propagate. `SimulationAborted` carries
  `simulator`, `reason`, `data_rows` and `log_path`; other exceptions also
  carry context, such as `ReportStorageError.report`.
- `silk/doctor.py` — `python -m silk.doctor` environment preflight. Stdlib-only
  at module level and every check catches its own `ImportError`, because it must
  run *inside* a broken environment. A registered-but-uninstalled PDK is `warn`,
  not `FAIL`; the `PDK_ROOT` directory check uses the resolved setting,
  including the `~/pdk` default.
  Running as `python -m silk.doctor` imports `silk` first; if numpy is
  missing, use `python silk/doctor.py` to reach the report.
- `silk/settings.py` — the one reader of tool and PDK locations:
  `PDK_ROOT`, `SILK_NGSPICE_BIN`, `SILK_XYCE_BIN`, `SILK_EXAMPLES_OUT`,
  `SILK_FIREFOX`, `SILK_CHROME`, and the optional `SILK_REGION_THRESHOLDS`
  policy file, resolved as environment variable > the user's
  settings file > built-in default. Stdlib-only apart from
  `silk.exceptions.SettingsError` (with a fallback for `python silk/doctor.py`),
  because the doctor imports it. Every read site (`load_pdk`, the runners,
  `pdk.validate`, provenance, examples output, the doctor, `tests/_env.py`)
  goes through it. User contract in [settings.md](settings.md).
- `silk/_json_codec.py` — the stdlib-only `encode_json` / `decode_json` pair
  shared by JSON writers and persistence readers. Encodes NaN and signed
  infinity without changing Python metric results; every dump also passes
  `allow_nan=False`. The wire contract is in [reports](reports.md#saving-and-loading)
  and the package-wide writer guard is `tests/test_json_codec.py`. Must not
  import metrics, reports or database code, or infer measurement diagnostics.
- `silk/spice_names.py` — the stdlib-only SPICE-name codec:
  `to_spice_name` / `to_designer_name` convert instance components between
  `[k]` (silk) and `<k>` (simulator) in recognised vector syntax only.
  `silk/circuit/emitted_names.py` uses its component conversion; the waveform
  viewer may import it (see invariant I12). Keep name conversion independent of circuit objects.
- `silk/vector_match.py` — the stdlib-only vector matcher:
  `collect_vector_candidates` (also re-exported by `silk/signals/spice.py`), `first_row` and `vector_address` decide which rows of a
  plot a request names and whether a bracket address resolves to exactly one
  row. The reader and the waveform viewer share it (invariant I12); its JS twin
  is in `waveform-core.mjs`. Must not import anything but `re` and
  `silk.spice_names`.

---

## 2. Data flow

### The loop as layers

```
 Python design intent                Module + ModuleTestbench            silk/circuit
        │  create_circuit_version → structure hash + spec-set hash        silk/db/identity.py
        ▼
 backend selection                   Module.simulator → backend registry
        ├───────────────────────────────────────┐
        ▼                                       ▼
 SPICE                                   behavioral: Python/numpy
 netlist emission                        models + supported stimuli/analyses
 PySpiceNetlister / XyceNetlister          BehavioralBackend
 deck bytes (checked by goldens)              no simulator or installed PDK models
 analysis + circuit contents                    │
        ▼                                       ▼
 simulator run                           nutmeg rawfile + RawResult
 ngspice / Xyce subprocess                       │
 backend launch → data + completion checks                 │
        │         (else SimulationAborted)      │
        ▼                                       │
 RawResult                                      │
 rawfile reader                                 │
        └──────────────────┬────────────────────┘
                           ▼
 shared results                      RawFile → Signal / deferred probes
        │  load_signal, tb.probe, plotting and rawfile viewer
        ▼
 metrics                             silk.metrics (waveform/timing/converter/...)      add_metric extractors
        ▼
 specs / measurements / reports      Spec.record_measurement, SpecReport
        │  spec key (explicit key=), measurement slot key
        ▼
 evidence (opt-in)                   store_report, store_net_currents     silk/db/tracking.py
        │  profile id (derived from owner/test/params), current-set id
        ▼
 layout estimation                   estimate_layout(module, RoutingModel) silk/layout/estimator.py
        │  EM width = I_peak / j_max, lumped R/C                            (units via BasePDK only)
        ▼
 parasitics                          inject_parasitics → real R/C         silk/layout/parasitics.py
        ▼
 re-sim                              same Module, same tests               back to backend selection
```

### Identities: where minted, where consumed

| Identity | Minted by | Consumed by | Changes when |
|---|---|---|---|
| Instance and test **names**, net **names**, spec **keys** | the author (`add_module`, `add_net`, `create_test`, `declare_spec(key=)`) | emission (`emitted_names.py`), rawfile lookup, DB owner paths | duplicate instance/test names and spec keys raise `DuplicateNameError`; repeating a scalar `add_net` reuses the canonical net. A spec display name is the default key only when `key=` is omitted. |
| `.subckt` name | `Module.module_name` via `register_module_definition` | the whole deck (one flat SPICE namespace) | a different structure claims a registered name → `CircuitError` by default; `variants=True` permits hash-suffixed definitions |
| **Structure hash** `silk.structure/v4` | `silk/db/identity.py` `_fingerprint_structure` through `silk/db/structure_tracker.py`; wrapped for circuit code by `_module_structure_hash` in `silk/circuit/circuit_elements.py` | `create_circuit_version` (seals a version), `net_current_sets` rows, the refactor goldens | device, passive, source, net, parameter or hierarchy changes represented in the canonical structure, including child port bindings |
| **Spec-set hash** `silk.spec/v1` | `silk/db/identity.py` over every `Spec.key` at its owner path | `create_circuit_version`; `restore_spec_measurements` | a spec key, owner path, metric, operator, target, units or parent changes |
| **Profile id** `silk.profile/v1` | derived in `_canonicalize_characterization_profile` from owner path + test name + declared parameters, **never a label** | `store_report`, `store_net_currents`, `get_characterization_profile`, `restore_characterization_evidence` | the test's declared parameters or name change; PVT does *not* enter it (PVT selects beneath it) |
| **Current-set id** | `Database.replace_net_current_set` per (version, profile, corner, T, V) | `get_net_current_set`, `restore_characterization_evidence` | a new selection for the same key mints a new row and marks the old one superseded |
| Run id (viewer) | `_compute_run_id` in `silk/signals/view/emit.py` | the emitted view's provenance | the emitted payload changes (data, metadata or view configuration) |

### One artifact traced end to end: the flagship walkthrough

`silk/examples/design_loop_walkthrough.py` (`python -m
silk.examples.design_loop_walkthrough`; pinned by
`tests/test_design_loop_walkthrough.py`) is the canonical trace. Each step
below names the function in the example and the silk code it lands in.

1. **Intent → sizing.** `size_from_intent` is pure arithmetic on the `INTENT`
   dict; no silk involved. This is the "Python is the source of intent" rule in
   its simplest form.
2. **Build.** `build_sampling_buffer` constructs a `Module`
   (`silk/circuit/modules.py`), adds devices via the PDK factory classes minted
   by `BasePDK` (`silk/pdk/basepdk.py`) inside `with mod:` (auto-registration
   through `_active_module` in `silk/circuit/_module_base.py`), declares typed
   ports with `define_module_ports`, and registers four uniquely named tests
   with `mod.create_test` (`silk/circuit/modules_testbench.py` →
   `ModuleTestbench` in `silk/circuit/module_test.py`, which raises
   `DuplicateNameError` on a repeated name). Each test gets metrics via
   `add_metric` and first-class specs via `add_spec` (`silk/specs/spec.py`).
   `test.parameters` becomes part of characterization identity and is never
   read by netlisting.
3. **Characterize.** `run_test` toggles `mod.save_device_currents` per run (a
   transient-only opt-in) and calls `ModuleTestbench.run`, which builds the
   testbench netlist (`_create_netlist`), emits the deck through
   `Module.generate_netlist` (`silk/circuit/modules_netlist.py` →
   `PySpiceNetlister.generate_complete_netlist` in
   `silk/circuit/spice_netlister.py`, options line from
   `silk/simulators/ngspice.yaml`), runs it through
   `Module.run_simulator` (`silk/circuit/modules_sim.py` →
   `SimulatorBackend.run` → `NgspiceBackend._launch` in
   `silk/simulation/backend.py` → `NgspiceRunner` in
   `silk/simulation/runner.py`), passes the result through `_require_data`,
   and loads signals via `NgspiceSignal.from_rawfile`
   (`silk/signals/spice.py` over `read_rawfile` in `silk/signals/rawfile.py`).
   Metric extractors call `settling_time` (`silk/metrics/timing.py`) and
   `dc_gain_dB`/`bandwidth_3dB` (`silk/metrics/ac.py`); `SpecReport`
   (`silk/circuit/spec_report.py`) is returned.
4. **Report + persist.** `characterize_and_store` first seals the design with
   `mod.create_circuit_version` (`silk/circuit/modules_db.py` →
   `CircuitTracker` in `silk/db/tracking.py`; structure and spec-set hashes
   from `silk/db/identity.py`). Then per context: `mod.store_report` writes
   metrics and spec measurements scoped to the derived profile
   (`Database.store_characterization_profile` in `silk/db/circuit_db.py`);
   `annotate_currents` (`silk/circuit/modules_results.py`) reads the
   per-terminal current vectors through the backend's
   `device_current_vectors` seam and fills a signed `avg` plus nonnegative
   `rms` and absolute `peak` on each `PortBinding`; `store_net_currents` persists them as a current set under
   the same profile and PVT, with the structure hash on the row. `reconcile_specs`
   (`silk/circuit/modules_spec.py` → `reconcile` in
   `silk/specs/reconciler.py`) and `generate_report` (`silk/circuit/report.py`)
   produce the pre-layout JSON/HTML.
5. **Reopen.** `reopen_and_query` opens a bare `Database` with no live circuit
   and asks `list_characterization_profiles`, `get_characterization_profile`,
   `get_net_current_set` and `get_simulation_results` — the stored characterization evidence ([database.md](database.md)).
6. **Rebuild + restore.** `rebuild_and_restore` rebuilds from the same factory,
   calls `create_circuit_version` again and asserts the id is unchanged (the
   structure and spec-set identity are content-derived, so an identical rebuild
   at the same stage resolves to the sealed version), then `restore_characterization_evidence` puts one exact
   context's spec measurements and currents onto the live objects without a
   simulator. `restore_spec_measurements` attaches the rest.
7. **Layout feedback.** `layout_feedback` calls `estimate_layout(fresh, MET1)`
   (`silk/layout/estimator.py`): per-device area through
   `device_geometry_um` (`silk/pdk/basepdk.py`, the only unit interpreter),
   EM width from the *restored* `Net.peak_current` and `RoutingModel.j_max`,
   lumped R/C per net. `inject_parasitics` (`silk/layout/parasitics.py`) adds
   real R/C elements through `silk.elements.elements`, a
   post-layout AC test runs the same path as step 3, and `diff_reports`
   (`silk/circuit/report_diff.py`) shows the bandwidth cost.
8. **Fail loud.** `demonstrate_mismatch` shows a rebuilt design with one knob
   changed, and the post-layout design itself, both refused with
   `EvidenceMismatchError` (`silk/exceptions.py`) when they ask for pre-layout
   evidence — and asserts nothing was half-applied.
9. **Netlist.** `Module.generate_netlist` from the same object, parasitics
   included.

---

## 3. Invariants

Each rule names its implementation and relevant tests. Tests cover specific cases;
conventions marked as inspection rules still need a source check.

| # | Invariant | What it means in code | Enforcement point |
|---|---|---|---|
| I1 | **Deck bytes are sacred.** | Netlist whitespace, value formatting and line order are observable output. A cosmetic change to emission is a behaviour change. | `tests/test_golden_netlist.py` (control block, every analysis card, `tests/golden/ota_5t.subckt`, `tests/golden/top_netlist_ac.cir`); `tests/refactor/test_module_golden.py` replaying `tests/refactor/battery.py` against `tests/refactor/goldens/` (`.netlist`, `.schema.json`, `.hash`, `.faults`). Regeneration only via `scripts/capture_module_goldens.py`, in its own commit, per the rule in [../CONTRIBUTING.md](../CONTRIBUTING.md). |
| I2 | **Metadata-only propagation.** | Attaching a `CharacterizationContext` or declaring test `parameters` changes zero deck bytes. Context rides on testbenches and modules as plain Python attributes. | `tests/test_characterization_context_propagation.py`. |
| I3 | **The `Module` API surface is frozen.** | `dir(Module)` and every callable's signature are goldens. Part-mixins inherit `object` at runtime (the `_Base` trick, §5); helpers that need not be methods are module-level functions. | `test_module_api_names_frozen` / `test_module_api_signatures_frozen` in `tests/refactor/test_module_golden.py` against `tests/refactor/goldens/module_api.json`. |
| I4 | **Names identify objects.** | One definition per emitted `.subckt` name deck-wide; a different structure claiming a registered name raises by default; `variants=True` emits hash-suffixed definitions. Identical rebuilds deduplicate. Instance, passive, source and test names are unique within their owner. Repeating a scalar `add_net` reuses the canonical net. Specs reject duplicate keys, not duplicate display names; the display name supplies the default key when `key=` is omitted. | `register_module_definition` in `silk/circuit/circuit_elements.py`; tests in `tests/test_spice_netlister.py` (collision raises, reuse dedupes, identical rebuild allowed); `DuplicateNameError` raise sites in `silk/circuit/circuit_elements.py`, `silk/circuit/modules_structure.py`, `silk/circuit/module_test.py`, `silk/circuit/modules_spec.py`, tested in `tests/test_module_test.py`, `tests/test_add_method.py`, `tests/specs/test_module_declare_spec.py`. |
| I5 | **No profile labels.** | A characterization profile's identity is derived from owner path, test name and declared parameters. The store/get APIs take a context, never a label or a raw digest; PVT selects beneath the profile and never enters it. | `test_public_apis_reject_free_standing_label_and_raw_digest` and the digest-recompute-on-read tests in `tests/db/test_profile_identity.py`; `_canonicalize_characterization_profile` in `silk/db/identity.py`; reserved PVT keys in `silk/db/context.py`. |
| I6 | **`Signal` is the simulator-agnostic contract; backend selection is the only dispatch.** | A backend registered by name drives a `Module` through render → emit → run → load with no edit to `silk/circuit/`. Downstream of `load_signal` nothing knows which simulator ran. Analysis-specific metrics validate signal kind and axis; wrong-kind errors are `SignalKindError`, a `ValueError` subclass. Testbench evaluation promotes these to `MetricBindingError`. There is no per-simulator `Signal` registry. | `tests/test_backend_seam_smoke.py` (a fake backend closes the four seams with no simulator); `tests/test_backend_runner_selection.py`, `tests/test_backend_netlister_selection.py`; `tests/test_xyce_netlister.py` (real second backend); axis guards in `tests/metrics/test_ac.py`, `tests/metrics/test_transient.py`, `tests/metrics/test_dc.py`. **Honest limit:** a non-`Signal` argument fails with `AttributeError`, not a typed error; there is no `isinstance` guard. |
| I7 | **silk never trusts a simulator's exit code, and a backend cannot opt out of the check.** | A run that produces no data rows, a constants-only plot, or no rawfile raises `SimulationAborted` with structured fields, even on exit 0. The gate is applied by `SimulatorBackend.run` (a template method around each backend's `_launch`), so a backend carries no gate code and cannot skip it; defining `run` on a backend subclass raises `TypeError` at class-definition time. Certified silk-generated transient/DC/AC/noise requests also pass `require_complete` in `silk/simulation/completion.py`; unprovable or external requests get no endpoint verdict. A non-nutmeg native answers data-presence through `silk_has_data()` (`DATA_PRESENCE_HOOK`) or a `has_data()` override, and one that answers through neither raises rather than being guessed at. | `_require_data` / `_has_data_rows` / `__init_subclass__` in `silk/simulation/backend.py`; `tests/test_backend_data_gate.py`, `tests/test_backend_seam_smoke.py` (no-data fake backend, override-`run` rejection), `tests/test_exceptions_sim_aborted.py`, `tests/test_xyce_abort.py`. |
| I9 | **Evidence is append-only with typed loud mismatches.** | `simulation_results` and `characterization_profiles` rows are never rewritten; `net_current_sets` supersede explicitly and keep history; sealed versions are immutable; every applicability failure is one of `EvidenceNotFoundError` / `EvidenceMismatchError` / `EvidenceAmbiguityError` / `LockedVersionError` / `CharacterizationContextError`, and a refused restore mutates nothing. | `tests/db/test_current_set_lifecycle.py`, `tests/db/test_profile_scoped_evidence.py`, `tests/db/test_spec_history.py`, `tests/test_db_transaction.py`, `tests/test_circuit_db.py`; the two partial-unique indexes and foreign keys validated by `silk/db/schema.py` (`tests/db/test_schema_v2.py`); migration byte-fidelity in `tests/db/test_migration_v1_to_v2.py`. |
| I10 | **Python is the single source of design intent.** | The authored `Module` is the source for generated decks. The database importer can read foreign netlists, but does not replace that authored design. Examples may not hard-code `.lib` paths or bypass `load_pdk`. | `tests/test_examples_no_raw_lib.py`; generated-Python safety and byte-identical emission for well-formed specs in `tests/test_codegen_emission_safety.py`. |
| I11 | **`BasePDK` is the sole reader and interpreter of the device catalog and of units.** | Only `silk/pdk/basepdk.py` turns a `pdk_unit:` tag into a number (`_SI_TO_PDK_SCALE`, `_scale_factor`, `_pdk_unit_to_um`, `device_geometry_um`). Everything else stores what `BasePDK` decided or calls the accessor; an unknown tag or device shape raises rather than assuming µm. `silk/faults/pdk_metadata.py` reads `PDKDevice.fault_params`, a `BasePDK`-parsed, unit-documented (PDK-native, unconverted) attribute, off the PDK object the calling module carries; no catalog-read exceptions outside `silk/pdk/` are currently allowlisted. | `tests/test_unit_authority.py` (no unit-interpretation identifier outside `silk/pdk/`; catalog-file reads outside `silk/pdk/` limited to the recorded allowlist); cross-PDK, cross-construction-style footprint equivalence in `tests/layout/test_area_model.py`. Adding a consumer means extending the allowlist *and* the audit in the same PR. |
| I12 | **The viewer depends only on the rawfile reader and shared leaf helpers.** | `silk/signals/view/` imports stdlib, numpy, `silk.signals.rawfile` and the stdlib-only leaves `silk._json_codec`, `silk.spice_names` and `silk.vector_match`; `ViewError` is a plain `Exception`. | `tests/test_view_boundary.py` (AST walk of every import at any scope, including the stdlib-only guards for all three leaves). |
| I13 | **Port-role patterns are data, not code.** | `silk/circuit/port_roles.py` contains the matching engine and zero patterns; nothing infers unless a ruleset is passed or loaded. | Tested in `tests/test_port_role.py`; shipped default is `silk/circuit/data/port-roles.default.yaml`. |
| I14 | **Examples ship and import clean.** | Examples import without running simulations or requiring installed model files, and without sibling bare imports or `sys.path` edits. Packaged PDK catalogs may load at import time. | `tests/test_examples_importable.py`, `tests/test_suite_runs_without_a_pdk.py`. |
| I15 | **This document matches the tree.** | Every `silk/*` subpackage and top-level module has a §1 row; every cited path exists; the front doors cross-link. | `tests/test_architecture_doc.py`. |
| I16 | **One SPICE-number parser.** | `silk/utils/_utils.py` `_parse_si` is the shared Python parser that turns `'2.56us'` into a float; its docstring is the contract (ngspice's suffix table incl. `mil`/`a`, `M` = milli, letters after the suffix ignored, Unicode micro rejected, `ValueError` the one failure mode). `circuit_import` (foreign netlists; the one lenient caller — `None` only with a `UserWarning`), `analysis` and `waveforms` all call it. Display tables (`modules_results._PREFIXES`, the explorer's `Fmt`) write SI `M` for mega and are display-only; the explorer's `parseSI` refuses an uppercase `M`/`P` so a copied readout is never silently 1e9 small. | `tests/test_spice_number_contract.py`: the contract cases, entry-point agreement, a per-file guard that no other Python file binds a mega token to `1e6`, and the explorer's `Fmt` block run under node. |

Additional conventions and implementation details to check when extending these layers:

- Circuit construction imports in `silk/elements/` and `silk/sources/` are local;
  source-name validation imports `silk.circuit.emitted_names` at module load.
  There is no dedicated import-order guard. Adding eager circuit imports can
  create an `ImportError` cycle at `import silk`.
- Avoid eager circuit imports in lower-level declarations and configuration
  helpers; these packages are not strict import leaves.
- The `_ANALYSIS_KINDS` set in `silk/codegen/testbench_generator.py` is
  hand-synced with `_sim_code` in `silk/codegen/generator.py`.
- The SPICE instance prefix of a PDK device is catalog data (`spice_prefix:`,
  default `XM`), carried as `_spice_prefix` on the factory class and as
  `PDKDevice.spice_prefix` on the legacy `add_element` path; both are read from
  the same key with the same default (`DEFAULT_PDK_SPICE_PREFIX` in
  `silk/pdk/basepdk.py`). Emission and naming read the field — `generate_netlist_entry`,
  `silk/circuit/emitted_names.py` and fault realization in
  `silk/faults/application.py` — and `silk/pdk/validate.py` judges
  `wrong_emission` on it. The two paths agreeing is pinned by
  `tests/test_pdk.py` (`TestSpicePrefixLegacyPath`), and a GF180MCU diode
  emitting a `D` line on the legacy path by `tests/test_gf180mcu_passives.py`.
  The ammeter wrapper (`silk/circuit/spice_netlister.py`) is prefix-agnostic
  (it re-uses whatever token the line starts with). The importer
  (`silk/db/circuit_import.py`) defaults to M/XM detection; pass `device_prefixes=` for a PDK's other prefixes.
- `silk/circuit/xyce_netlister.py` derives the `.options parser scale=`
  factor from `BasePDK.length_unit_in_metres`, derived
  from the catalog's length `pdk_unit:` tags (1e-6 for sky130, no line for the
  metre-native GF180MCU/SG13G2).

---

## 4. Seams

File exchange uses JSON, YAML, decks and rawfiles. In-process extensions, such as
constraint checkers, can also receive Python objects through their documented APIs.

### Schema export and code generation

Catalog exports are JSON; code generation takes structured specifications and returns
Python source:

- **Device catalog** — `export_schema(pdk_or_library)` / `module_schema(mod)` /
  `export_schema_json` in `silk/elements/schema.py`; CLI
  `python -m silk.elements.schema --elements` for ideal elements, or add
  `--pdk <name>` instead. Contract:
  [schema-export.md](schema-export.md). The `unit` key reflects the catalog's
  `pdk_unit` verbatim and interprets nothing.
- **Source catalog** — `export_source_schema` in `silk/sources/schema.py`
  (params reflected from `__init__`, symbol/pins/waveforms from
  `silk/sources/source_metadata.yaml`); folded into device-catalog shape with
  per-kind `code_template`s by `_source_catalog_fields` in
  `silk/elements/schema.py`.
- **Metric catalog** — `export_metric_schema` in `silk/signals/schema.py`, fed
  by the `@metric` decorator on the `silk.metrics` functions it publishes.
- **Codegen** — a caller sends a `CircuitSpec` dict
  ([codegen-contract.md](codegen-contract.md)) and gets a Python
  module back from `Generator` in `silk/codegen/generator.py`; it sends a
  `TestbenchSpec` and gets testbench Python from `generate_testbenches` in
  `silk/codegen/testbench_generator.py`. silk owns every emitted API line;
  the caller owns the spec; a malformed testbench spec raises `TestbenchSpecError`, while malformed
  circuit specs raise `ValueError` or a more specific validation error.

These schema and codegen boundaries do not pass `Module` objects, netlist
text or rawfiles.

### Layout consumers (placement, routing, GDS)

silk has no placer, router or geometry engine. Its layout interfaces carry data:

- `silk.constraints` → silk `*.const.json`
  (`silk/constraints/serialization.py`) carrying device-relational intent plus
  a `PortRoles` overlay per module and a `NetIntent` overlay per net.
- Per-net currents (`Net.avg_current`, `Net.peak_current`, filled by
  `annotate_currents` or restored from the database) are EM
  inputs to silk's minimum-width calculations.
- `silk/constraints/checkers.py` is an open dispatch table so layout consumers can
  register concrete checkers without editing silk.

silk's `estimate_layout` numbers are statistical estimates; nothing in `silk/layout/`
reads a GDS back. Planned extraction work is described in
[capability-roadmap.md](capability-roadmap.md).

### PDK installs

silk ships manifests and catalogs, never model files. `load_pdk(name)` resolves the
model library against `pdk_root=` or the `PDK_ROOT` setting (environment variable →
settings file → `~/pdk`). `library_location=` overrides the main library path only
(`silk/pdk/_loader.py`); OSDI plugins are declared in the manifest and loaded via
`pre_osdi` lines but never compiled by silk (`_require_osdi_compiled` in
`silk/circuit/modules_sim.py` fails a *run*, not deck generation, with instructions).
Installation is the user's or pdksetup's job
([installing-the-pdk.md](installing-the-pdk.md)); adding a PDK is a manifest plus a
catalog ([adding-a-pdk.md](adding-a-pdk.md)) and is vetted by `python -m
silk.pdk.validate` ([pdk-validation.md](pdk-validation.md)).

---

## 5. Internal gotchas

Things that bite maintainers specifically. User-facing gotchas (lambda
capture, units, `save_device_currents`, …) are in
[agent-guide.md §8](agent-guide.md#8-gotchas) and are not repeated here.

- **The `_Base` trick in the part-mixins — do not tidy it.** Every
  `silk/circuit/modules_*.py` file does `if TYPE_CHECKING: _Base = _ModuleAttrs
  else: _Base = object`. Collapsing that into class-body annotations leaks
  `__annotations__` into `dir(Module)` and stringifies signatures under PEP 563;
  both break the API golden (I3). The reason is written in
  `silk/circuit/_module_protocol.py`; read it before touching a part. The same
  concern is why `_resolve_terminal_current_method`, `_require_osdi_compiled`
  and `_resolve_db_path` are module-level functions rather than methods.
- **`M` means two things, and which one depends on direction.** Everything
  that *parses* (`_parse_si`, and so the deck, the importer, the analysis
  fields) reads `M`/`m` as milli and needs `Meg` for mega — SPICE rules.
  Everything that *displays* (`modules_results._PREFIXES` axis labels, the
  explorer's `Fmt.num` readouts) writes SI: `M` = mega, `µ` = micro. Never
  route a label or readout string back into `_parse_si`; the explorer's own
  `parseSI` refuses an uppercase `M`/`P` for exactly this reason (I16).
- **`add_module` places a live reference to the child.** `Module.add_module`
  (`silk/circuit/modules_structure.py`) stores the child object in the
  element graph, and `register_module_definition` indexes the same object in
  `module_dict` under its emitted definition name (placement registers references only;
  `validate=True` compares distinct objects claiming one name at emission).
  Consequences: the object you built is the placed object, so a later edit to
  it (a spec, a role, a net) is visible through the placement; two placements
  of one `Module` share authored definitions, specs and live parameters.
  Captured callable state is copied per placement and run; module-level globals
  remain shared and are not reset, including for a plain function placed once
  (see [BehavioralEngine](api-reference.md#behavioralengine)). Build one
  `Module` per placement when their authored definitions or specs must differ.
  `Module.__deepcopy__` exists for explicit copies (it shares an attribute that fails to copy rather than raising), but
  placement does not call it.
- **The viewer imports only the rawfile reader and the three shared leaf helpers
  listed in I12 from outside its package.** Adding a convenience import from `silk.exceptions`
  or `matplotlib` into `silk/signals/view/` fails `tests/test_view_boundary.py`
  by design; the shared codec is separately guarded to remain stdlib-only.
- **The structure hash lives in `silk/db/`, imported lazily from
  `silk/circuit/`.** A module-level import of `silk.db` from
  `silk/circuit/circuit_elements.py` would create a circular dependency; keep it inside `_module_structure_hash`.
- **`import silk.constraints` is a side-effect import** that patches `Module`.
  The refactor battery imports it at module load so `add_constraint` is always
  in the API golden; a test that imports `Module` without it sees a different
  `dir(Module)`.
- **`_require_data` must leave `native=None` alone.** That is a legitimate
  Xyce noise-only run validated separately as CSV, not an abort.
- **Backends implement `_launch`, never `run`.** `run` is the base-class
  template method that applies the data gate; `SimulatorBackend.__init_subclass__`
  raises `TypeError` on a subclass that defines `run`. `load_raw` stays outside
  the template method — re-loading a saved rawfile is not a fresh-run failure —
  but applies the same gate itself.
- **`save_device_currents` and `set_save_list` are mutually exclusive** and
  the check lives in `silk/circuit/spice_netlister.py`; `save_device_currents`
  refuses any deck without exactly one `Transient`.
- **PDK manifests define valid corners.** `Corner` in
  `silk/simulation/corner.py` is a string enum for five standard process corners; `generate_netlist(corner=None)` resolves the PDK's
  declared `default_corner` (GF180MCU calls its typical corner `typical`, not
  `tt`).
- **Goldens are regenerated deliberately, never to make a test pass.** The
  procedure and the own-commit rule are in
  [../CONTRIBUTING.md](../CONTRIBUTING.md); the script is
  `scripts/capture_module_goldens.py`.
- **Python-version drift in the API golden.** `module_api_snapshot` in
  `tests/refactor/battery.py` excludes `__getstate__` (3.11+) and
  `__firstlineno__`/`__static_attributes__` (3.13+) and normalises two implicit
  classmethod signatures. A new dunder that CPython adds to `object` will
  surface as a golden failure on one interpreter only; extend the exclusion
  list there, do not regenerate the golden per version.

---

## 6. Where does X go

| Adding… | File to edit | Pattern to follow | Tests / goldens that move |
|---|---|---|---|
| a **waveform / ADC / timing metric** | the `silk/metrics/` submodule matching its purpose (`waveform.py`, `statistics.py`, `timing.py`, `jitter.py`, `eye.py`, `power.py`, `spectral.py`, `rf.py`, `converter.py`, `fom.py`, `ac.py`, `noise.py`); export in `silk/metrics/__init__.py` (`__all__` too) | call the submodule's `_check_*` first; spectral metrics take `(tclk, tstart, n_samples)` and sample via `_sample_at_clock`; document in [metrics-library.md](metrics-library.md) | covering test files in `tests/metrics/` |
| an **AC metric to export** | `silk/metrics/ac.py` with the `@metric(domain=…, unit=…)` decorator from `silk/signals/schema.py` | existing `bandwidth_3dB` | `tests/` signal-metric tests; the exported metric schema |
| a **PDK** | `silk/pdk/data/<name>.pdk.yaml` + `silk/pdk/data/<name>_device_details.yaml`; no Python unless the PDK needs behaviour sky130 does not | [adding-a-pdk.md](adding-a-pdk.md); every parameter tagged with `pdk_unit:`; run `python -m silk.pdk.validate --pdk <name>` | `tests/test_load_pdk.py`, `tests/layout/test_area_model.py` cross-PDK equivalence; sky130 decks must stay byte-identical (I1) |
| a **device to an existing PDK** | that PDK's `*_device_details.yaml` | pins + `paramList` with `pdk_unit`; `fault_params:` if fault campaigns should bound it | `python -m silk.pdk.validate`; `export_schema` goldens in `tests/refactor/goldens/` if the battery uses it |
| an **ideal element** | `silk/elements/analog_devices.yaml` (factory-generated); `silk/elements/passives.py` holds deprecated compatibility wrappers | an existing ideal element specification | `tests/test_circuit_elements.py`; emission goldens |
| a **source / waveform kind** | `silk/sources/waveforms.py` + one line under `waveforms:` in `silk/sources/source_metadata.yaml` | `Pulse` (mandatory keyword-only edges, SI strings preserved verbatim) | `tests/test_pwl.py`-style emission tests; the source schema drift-guard |
| a **simulator** | a backend class in `silk/simulation/` registered with `@SimulatorBackend.register('<name>')` implementing `render_analysis`, `make_netlister`, `_launch` and `load_signal` (+ optional current/probe capabilities); inherit `run`. For an in-process backend, set `runs_module=True`, implement `_launch_module`, and inherit `run_module`; a dialect YAML in `silk/simulators/` | `XyceBackend` for a subprocess simulator, `BehavioralBackend` for in-process; adapt results *into* a `RawFile` rather than teaching downstream a new shape | copy `tests/test_backend_seam_smoke.py`; ngspice emission must stay byte-identical (I1) |
| a **netlist emission site** (new line kind in the deck) | `silk/circuit/spice_netlister.py` (`generate_complete_netlist` and the `generate_*` it calls); Xyce divergences in `silk/circuit/xyce_netlister.py` | thread through `EmitContext.fmt`; do not reformat neighbours | `tests/test_golden_netlist.py`; regenerate `tests/refactor/goldens/` deliberately, own commit |
| a **`Module` method** | the part-mixin whose subsystem it belongs to (`silk/circuit/modules_*.py`); a helper that need not be a method is a module-level function | keep the `_Base` trick; add the attribute to `_ModuleAttrs` in `silk/circuit/_module_protocol.py` if parts need its type | `tests/refactor/goldens/module_api.json` will change: regenerate deliberately and say why |
| a **spec / measurement behaviour** | `silk/specs/spec.py` (declarative) or `silk/specs/reconciler.py` (walk) | stage reconciliation and calibration before applying changes | `tests/specs/` |
| an **evidence type** (new stored thing) | table + columns in `silk/db/schema.py` (`REQUIRED_COLUMNS`, FKs, unique sets), creation in `silk/db/circuit_db.py`, tracker method in `silk/db/tracking.py`, `Module` wrapper in `silk/circuit/modules_db.py`, record dataclass in `silk/db/models.py`; if it must be profile-scoped, carry `profile_id` and follow `net_current_sets`' supersession | define the evidence contract first; scheme-versioned identity; typed `Evidence*` failures; a schema bump means a new explicit migration in `silk/db/migration.py` | `tests/db/test_schema_v2.py`, lifecycle tests in `tests/db/`, `tests/test_characterization_context_propagation.py` (deck bytes unchanged) |
| a **lint rule** | `silk/lint/rules/<family>.py`, registered in `DEFAULT_CATALOG` in `silk/lint/rules/__init__.py` | `FloatingNetRule`; respect `NoConnect` markers; give it a stable ID | `tests/lint/rules/`; report goldens under `tests/lint/reports/` |
| a **layout constraint type** | `silk/constraints/types.py` + `ALL_CONSTRAINT_TYPES` in `silk/constraints/__init__.py`; JSON mapping in `silk/constraints/serialization.py`; a stub in `silk/constraints/checkers.py` | `SymmetricBlocks` | `tests/constraints/` round-trip tests |
| a **layout estimate quantity** | `silk/layout/estimator.py` (+ a field on `LayoutEstimate`/`NetLayout`) | read geometry only through `device_geometry_um`; new heuristics are keyword-only, validated, recorded on the estimate | `tests/layout/`; report JSON round-trip in `tests/test_circuit_report.py` |
| a **fault kind** | `silk/faults/application.py` (transform) + `silk/faults/sugar.py` | read the device without mutation; return a prepared realization using the supplied seeded RNG; the netlister emits it | `tests/faults/`; `.faults` goldens in `tests/refactor/goldens/` |
| a **codegen field** | `silk/codegen/generator.py` (`CircuitSpec.from_dict` + the emitter); document in [codegen-contract.md](codegen-contract.md) | validate the identifier before emission; rewrite templates, never interpolate raw | `tests/test_codegen_*.py`, `tests/test_codegen_emission_safety.py` |
| an **example** | `silk/examples/<name>.py` with a `__main__` guard, package-relative sibling imports, `load_pdk('sky130')`, output via `example_output_dir()` under the configured `SILK_EXAMPLES_OUT` root (default: `./silk-output`) | [../CONTRIBUTING.md](../CONTRIBUTING.md) "Adding an example"; copy the index entry pattern in [agent-guide.md §7](agent-guide.md#7-curated-example-index) and add a row to [examples.md](examples.md) | `tests/test_examples_importable.py` imports modules under a fake simulator to catch simulator work on import; separate example/integration checks execute workflows; a slow test if it is a release proof |
| a **CLI tool** | `python -m silk.<pkg>.<module>` with a `main(argv)`; if the module is also imported for exports, add a tiny runner module like `silk/circuit/rolecheck.py` to avoid the `runpy` double-import warning | `silk/pdk/validate.py`; document in [cli.md](cli.md) | `tests/test_doctor.py`-style subprocess test |
| a **subpackage** | `silk/<name>/__init__.py`; re-export from `silk/__init__.py` only if it is public surface | keep it below `circuit` unless it must walk a `Module`; lazy-import upward | a row in §1 of this file, or `tests/test_architecture_doc.py` fails |
