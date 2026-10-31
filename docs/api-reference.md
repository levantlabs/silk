# circuitsilk API Reference

Package name: `circuitsilk` — import as `silk`.

The [generated API reference](reference/index.md) documents the public API
from its docstrings. To build it from the repository root, install the documentation dependencies with `.venv/bin/python -m pip install -e '.[docs]'`,
then run:

```bash
.venv/bin/python -m sphinx -E -n -W --keep-going -q -b html -d silk-output/previews/api-reference/doctrees docs/reference silk-output/previews/api-reference/html
```

The build needs network access to download the Python, NumPy and matplotlib
cross-reference inventories. Unavailable inventories fail the strict build.

Open `silk-output/previews/api-reference/html/index.html` to browse the HTML.
This page provides prose and signature tables alongside the generated reference.
Most examples are fragments: supply the circuit, PDK, nets and metrics named in
the surrounding text before running them. Ellipses mark code you must provide.

---

## Top-level imports (`silk`)

```python
from silk import (
    Module, ModuleTestbench, Port,
    Variable, CornerResults,
    export_schema, module_schema, export_schema_json,
    view,                       # silk.view(source) — waveform-explorer shortcut
    RoutingModel, HalfCircuit, LayoutWarning, NetLayout, DeviceLayout,
    LayoutEstimate, LayoutInferenceError,
    estimate_layout, capture_layout, inject_parasitics,
    BasePDK, UNIT_PDK, UNIT_SI, load_pdk,
    Transient, AC, DC, Noise, Corner,
    Signal, NgspiceSignal,
    VariableSpace, Optimizer,
    Database, DesignStage,
    Generator,
    PortDir, PortRole, VoltageDomain,
    Net, PortBinding,
    check_interfaces, check_voltage_domains, InterfaceViolation,
    check_junction_overrange, JunctionViolation,
    require_annotation_coverage, AnnotationCoverageError,
    list_unannotated_ports, annotation_coverage, UnannotatedPort, AnnotationCoverage,
    RoleRuleset, infer_port_roles,
    CircuitReport, generate_report,
    ReportDiff, diff_reports,
    # Design-stage bridge (silk.specs)
    Spec, Measurement, CalibrationBinding,
    SpecOutcome, SpecReconciliationReport,
    CircuitError, DuplicateNameError, UndeclaredNetError,
    SimulationError, PDKError,
    sources,
    # Analog lint framework
    Severity,
    Rule, RuleRegistry, VisitEvent, VisitOutcome, RegistryRunResult,
    Waiver, WaiverFile, WaiverMatcher, apply_waivers,
    ScopeExclusion, ExclusionFile,
    ReportMetadata, SCHEMA_VERSION,
    LintReport, LintDiff,
    CoverageReport, RuleCoverageReport, RuleCoverageNode,
    help_rule, list_rules,
    NoConnect,
)
```

---

## silk.circuit

### `Module`

Primary circuit-design object. Manages devices, sources, passives, and simulation.

```
Module(pdk_name, dir_folder, netlistname, pdk_details,
       debug=False, strict_nets=False, module_name=None,
       db_path=None, version='1.0', stage=None, created_by='user',
       variants=False)
```

| Parameter | Type | Description |
|---|---|---|
| `pdk_name` | str | PDK identifier string (e.g. `'PDK_skywater130'`) |
| `dir_folder` | str | Output directory for netlists and raw files |
| `netlistname` | str | Deck file name, stored unchanged; put the folder in `dir_folder`. A `/` in the name is refused at deck preparation. |
| `pdk_details` | BasePDK | Loaded PDK object |
| `debug` | bool | Print verbose output |
| `strict_nets` | bool | Raise `UndeclaredNetError` for undeclared nets |
| `module_name` | str | Subcircuit name (None = top-level netlist). Must match `^[A-Za-z_][A-Za-z0-9_]*$` (a valid SPICE identifier) or the constructor raises `ValueError` naming the value; validated again by `define_module_name(...)`. **Case-insensitive identity:** SPICE `.subckt` names are case-insensitive (ngspice folds to lower, Xyce to upper), so a `module_name` that differs only by case from an already-registered one (e.g. `'Bias'` then `'bias'`) resolves onto that same definition instead of creating a second, colliding `.subckt` — a structurally identical body dedups there (under the first-registered spelling, the only one ever emitted); a genuinely different body raises the same "structurally different circuits" `CircuitError` a same-spelling clash raises, naming both spellings. |
| `db_path` | str | Path to SQLite database (enables tracking when set) |
| `version` | str | Version string for database tracking |
| `stage` | DesignStage | Design stage (default: DEV) |
| `created_by` | str | Author name for database records |
| `variants` | bool | Opt-in, per-Module, default `False`. With `True`, this `module_name` is emitted under an ALWAYS-suffixed, deterministic name — `<module_name>_<8 hex chars of the definition hash>` — instead of raising when a structurally different body claims the same `module_name`; identical bodies still dedup to one definition, and the suffix is order/count-independent (a module built and placed exactly once is still suffixed). Also settable as a plain attribute after construction (`mod.variants = True`) for builders that name the module via `define_module_name(...)` instead. The caller's own `Module` object always keeps its bare `module_name`; only the emitted `.subckt` definition (and everything keyed off it — the `x` instance line, `run_all_tests()` result keys, current-annotation grouping, fault targeting, the circuit report) carries the suffix. Without `variants=True`, a same-name/different-body clash raises `CircuitError` at netlist generation. See the example below. |

Two bias blocks from one builder, at different currents, placed side by side:

```python
def build_bias(pdk, i_bias_a):
    mod = Module("PDK_skywater130", "/tmp", "bias.cir", pdk_details=pdk,
                 module_name="bias", variants=True)
    mod.define_module_ports({"A": Port.analog_input(), "B": Port.analog_output(),
                             "VSS": Port.ground()})
    with mod:
        sources.CurrentSource('bias', positive='A', negative='B', dc=i_bias_a)
    return mod

top.add_module(build_bias(pdk, 10e-6), "xb1", A="a1", B="b1", VSS="0")
top.add_module(build_bias(pdk, 20e-6), "xb2", A="a2", B="b2", VSS="0")
# -> two .subckt definitions ("bias_<hash1>", "bias_<hash2>"), two x lines;
#    without variants=True this raises CircuitError('bias' is claimed by
#    two structurally different circuits).
```

Same `module_name` means the same circuit **and behavioral model**. Rebuilt
Python functions dedup when their code and explicit `params` match;
`functools.partial` models also compare bound positional and keyword arguments,
including nested partials. Source filenames and line locations are excluded, so
moving a model within a file does not change its identity. Different code or
parameters (including model present versus absent) raises
`CircuitError` saying the behavioral model differs. With `variants=True`, the
naming digest includes model code and parameters. Descendant models also
participate, so wrappers cannot merge different behavioral definitions.
Hierarchies without models or custom lines keep their existing structure-hash suffixes.
Behavioral models are excluded directly from database structure hashes, but a
`variants=True` child's structural name includes its model identity, so changing
that model changes the parent's database structure hash.

**Identity limits:** captured closure values, globals, function defaults and
callable-instance state are not compared; put identity-bearing values in
`params`. Supported identity values are numeric scalars, strings, bytes, NumPy
arrays and built-in containers of these values, plus `None`, `Variable`
objects, variable expressions, and `Normal` / `Uniform` declarations. A callable without a Python
code object or an unsupported parameter value raises an actionable `CircuitError`.
With the default `variants=False`, this refusal applies only when two same-named
definitions meet; attaching such a model or placing its first definition does
not require comparison. With `variants=True`, constructing the deterministic
suffix also requires a supported identity, even for the first definition.
Suffixes are stable across processes using the same Python version and model code; they are not a
portable code fingerprint across Python versions. Parameters are compared at
netlist generation time; calibration still uses the live parameter dict during runs.

**Live placement.** `add_module` and
`ModuleTestbench.add_module_under_test` keep the supplied Module by reference.
The parent emits its current body; a bench runs its current DUT. Wiring is
independent for each placement. Place, edit, then place the same object again
and both instances use the final body. Call a factory twice for independent
circuits. PDK device templates retain per-placement copies. Definition
collisions are checked against the whole deck at netlist generation, and
`variants=True` names follow the current definition hash.

Rewire one placement with `parent.connect_module('stage', out='new_net')`.
The dict and keyword forms, Net-to-name conversion, whole buses, declared bit
keys, input/inout scalar tie-offs, width and overlapping-key checks are the
same as `add_module`. Nets resolve in the receiving parent's namespace,
including when a Net object came from another module. Strict-net checks use
that parent's declarations. Other placements keep their wiring. Accepted
rewires also synchronize live definition port metadata; refused calls leave
all placement declarations, bindings and net maps unchanged.

Placement handles and numbered placement arrays reject all public attribute
assignments with `AttributeError`, including unknown names and method names.
Use `connect_module` on the module holding the placement to rewire its ports.
Reads, slices and copied or pickled handles continue to work. A held unbound
reference (`stage.out`, `stage.net('out')`, or `stage.bus('data')`) follows the
current wiring when bound; a probe already bound before rewiring keeps naming
its original net. Device objects returned by a handle lookup retain their
existing behavior.

**Hierarchy and test inspection:**

| Method | Result |
|---|---|
| `iter_instances(*, include_self=False)` | Iterator of frozen `ModuleInstance` records, in depth-first insertion order. Each has `path` (e.g. `amp.bias`), `spice_path` (e.g. `xamp.xbias`, for rawfile node naming), `module` (live Module), `owner_path` (e.g. `$/amp/bias`), and `module_name` (resolved definition name, including a variant suffix). Repeated placements yield separate records holding the same module. `include_self=True` starts with the root: `path` and `spice_path` are `''`, owner path `'$'`. Legacy `add_submodule` registrations are also walked so their tests remain reachable. `placed` is `True` for emitted instances and `False` for the root and organizational registrations. |
| `list_test_entries(*, recursive=True)` | Tuple of frozen `TestEntry` records in exactly the order `run_all_tests` uses. Each has `path` (user instance path), `spice_path` (rawfile instance path), `module`, `owner_path`, `test_name`, `test` (original root bench or selected child handle), `report_key`, `module_name`, `inherited_temperature`, `inherited_supply`, and `output_mode`. Unique keys retain `module_name.test_name`; repeated keys become `owner_path:module_name.test_name`. `recursive=False` lists only the root's tests. Legacy `add_submodule` test registrations are included, as in the runner. |

`top.get_test(name, instance='amp.bias')` returns a stable selected handle for
that test and owner. `entry.test` is that same handle. `get_test(name)` and
`get_test(name, instance='')` return the original local bench, also used by root
entries. Unknown or ambiguous selections raise `ConfigurationError` and list
the available selections. Looking up a test never runs it.

Selected `prepare()`, `run()` and `evaluate(rawfile=...)` use the owner's
current inherited conditions. Metrics, probes and results refer to that owner's
retained evidence. All selections share the authored bench circuit. Setter calls
through a selected handle configure that same authored bench, so the edits are
visible through other selections, including selections under another root.
Public writable properties use the authored setter too: for example,
`selected.corners = ['tt', 'ss']` configures the shared corner list and
`selected.corners = None` clears it. Read-only properties reject assignment
with `AttributeError`; use `selected.set_corner('ss')` to configure the corner.
Explicit test settings override inherited conditions. Fluent methods that
return the bench retain the selected handle: for example,
`selected.set_corner('tt').prepare()` prepares that placement. Setter calls
leave retained reports, metrics and probe results intact; the next execution
uses the updated configuration. Independent circuit definitions require separate
factory calls.

Within one `run_all_tests()` invocation, placements of the same definition and
named test with proven identical SPICE inputs share a simulation, including
equal factory-built blocks. Reports, measurements and
metric extraction remain separate for every owner. Equivalence uses emitted
inputs, inherited conditions and capture settings. Behavioral callables,
Monte Carlo samples, random sources and opaque/custom directives conservatively
run separately. Different named tests never share simulations; root tests
always run their own netlists in their own output folders. There is no
persistent cache; direct `run()` always launches.

Unqualified child readback uses its latest invocation when it contains one
distinct simulation. Several distinct runs require `get_test(..., instance=...)`
for numeric reads (`UnboundSignalError` lists the choices); plots overlay the
distinct retained runs and label their placement aliases and corners/samples.
Selected handles retain loaded data across another owner's overwrite-mode run.
A failed run clears that selection's current readback. Reconciliation and DB
publication select the root and canonical owner before selecting evidence.
Conflicting calibration writes to one shared parameter refuse atomically.

Listing does not run or prepare tests. It refreshes derived definition-name
indexes (including variant names). Returned records are immutable; the
referenced Module and bench remain configurable. For example:

```python
for instance in top.iter_instances(include_self=True):
    instance.module.simulator = 'behavioral'
entries = top.list_test_entries()
for entry in entries:
    print(entry.test.owner_path, entry.test.report_key)
reports = top.run_all_tests()
```

Runtime simulator inheritance still applies: a bench's explicit simulator
selection overrides its owning module. Per-instance current annotations are
available from `annotation_report.instance_modules[path]`; the shared Module
holds the worst-case definition annotation.

**Circuit methods:**

| Method | Description |
|---|---|
| `add(*devices)` | Add one or more device objects. Idempotent: re-adding the same instance is a no-op; a different instance with the same name raises `DuplicateNameError`. |
| `add_module(module, name, nets=None, **ports)` | Place an instance of a `Module` (subcircuit) — the hierarchy-placement verb shared with `ModuleTestbench.add_module`. Port wiring via port-name kwargs (recommended) or `nets={port: net}` dict; values may be `Net` objects (from `tb.add_net()` / `tb.nets.<name>`) or net-name strings. Returns a `SubcircuitInstance` handle giving attribute-access to internal nets and nested children after sim. A non-`Module` raises `TypeError` pointing to `add()`. See the "Typed wiring" section below. |
| `connect_module(name, nets=None, **ports)` | Atomically rewire selected ports of an existing module placement in this parent; omitted ports keep their wiring. Uses `add_module` dict/keyword, Net and bus rules. Unknown placements or ports raise `CircuitError`; devices use pin assignment. A refused call leaves connections and declaration metadata unchanged. |
| `add_element(elemDevice, elemName, nets=None, parameters=None, **port_kwargs)` | **Deprecated** — accepts device classes, device instances, legacy PDK devices and `Module`s. Warns with the concrete `add(pdk.device(name, **pins, **params))` replacement for a device factory target or `add_module()` for a `Module` target. Use `add()` or `add_module()` directly in new code. |
| `add_net(netname, parameters=None, classification=None, *, bits=None, naming='bracketed')` | Declare a named net. Returns the new (or pre-existing) `Net` object, so callers can capture it directly: `avdd = tb.add_net('avdd')`. When `bits` is set, returns a bus handle and creates per-bit nets using `naming` (`'bracketed'` → `name[0]`, `name[1]`, …; `'underscore'` → `name_0`, `name_1`, …). |
| `add_source(source)` | Add a voltage or current source. Same idempotency semantics as `add()`. |
| `add_passive(passive)` | Add R, C, or L elements. Same idempotency semantics as `add()`. **Deprecated** — prefer `add(elements.resistor/capacitor/inductor(...))`. |
| `__enter__` / `__exit__` | A Module is a context manager — inside a `with mod:` block, every device/source/passive constructor auto-registers with `mod`. See the "Context-manager auto-registration" section below. |
| `add_no_connect(net_name, reason)` | Register a `NoConnect` lint marker on `net_name`; suppresses STRUCT-003 FLOATING_NET (zero connections) and STRUCT-011 SINGLE_CONNECTION_NET (exactly one) for that net. `reason` is required and non-empty. Returns the `NoConnect` instance. A repeat call on the same net with an identical `reason` is a no-op returning the original marker; a different `reason` raises `DuplicateNameError` naming both. |
| `add_sim_type(simType)` | Register an analysis object (`Transient`, `AC`, `DC`, `Noise` or `Op`), not a string. Subcircuits raise `ValueError`; put analyses on the top-level testbench. |
| `set_supply_voltage(volts, *, source=None)` | Configure the primary supply on the executing top-level Module: sets one voltage source's DC value before emission, and confirms it against the launched deck. Finite real volts, excluding booleans; `None` clears the setting without restoring an overwritten value and deliberately opts out of inheritance; it does not resume inheritance. Returns self. A subcircuit that owns **no registered tests** raises `ValueError`; configure its testbench. A block that owns tests accepts it as their default. Omit `source` and the supply is resolved from an explicit name, a declared `POWER` role, or a sole DC rail — never from a name; ambiguity raises. Inherits to registered tests exactly as `set_temperature` does. See [Operating conditions](#operating-conditions). |
| `set_temperature(temperature_c)` | Configure ambient TEMP on the executing top-level Module; finite real Celsius above -273.15, excluding booleans; `None` clears TEMP without changing TNOM and deliberately opts out of inheritance; it does not resume inheritance. Returns self. A subcircuit that owns **no registered tests** raises `ValueError`; configure its testbench. A block that owns tests accepts it as their default. Shares one setting with native `temp`, last setter call wins. The setting applies to this Module's own `simulate()` **and** to every registered test that has none of its own, resolved at each run; a nearer setting always wins. Supported on ngspice and Xyce, which each confirm the applied value at run time; other backends raise `NotSupportedError`. See [Operating conditions](#operating-conditions). |
| `set_mismatch(enabled, *, devices=None)` | Monte Carlo: `False` switches statistical mismatch off for this subcircuit block and everything below it (`.param <switch>=0` first in its `.subckt`, mismatch/both sample decks only). Per block only: `devices=` raises `NotSupportedError`; top-level raises `ValueError`; non-bool `TypeError`. Set before generating Monte Carlo sample decks; all placements of a shared block use its current setting. `mismatch_enabled` reads it back. On Xyce a Monte Carlo mismatch/both deck with any block set off raises `NotSupportedError` (Xyce ignores a subcircuit-local `.param`). |
| `set_simulator_options(**options)` | Configure simulator-specific options (tolerances, integration method, …); pass `preset=` as a keyword option. Options rendered via the YAML config for `self.simulator`. See [Simulator options](#simulator-options). |
| `set_save_list(signals=None, *, exclude=None, max_depth=None)` | **Opt-in**: restrict which node voltages the simulator writes to the rawfile (the escape hatch for runs whose `save all` rawfile is too big to store). Bare full-name, glob and hierarchy selectors, also accepting designer paths rooted at this Module (e.g. `tb.netlist.set_save_list(['dut.mid'])`); connected port paths select parent nets. Emitted matches win with a warning when designer matches differ. No `v(...)` wrappers. Validated against the design's net inventory at netlist time — a selector matching nothing raises with near-miss suggestions. `set_save_list(None)` restores the default `save all`. Incompatible with `save_device_currents`; ignored on Noise decks. See [simulator-save-list.md](simulator-save-list.md). |
| `set_output_options(mode='new', *, layout='flat')` | Configure subsequent runs: `'overwrite'` reuses deterministic paths; `'new'` (default) prepares a persistent unique child per deck. `layout` is keyword-only; `'flat'` is the only supported value (`'nested'` raises `NotSupportedError`, anything else `ValueError`). Invalid values raise without changing policy or artifacts. See [output lifecycle](#testbench-output-directories-and-run-labels). |
| `simulate(corner=None, auto_track_structure=False, timeout=None)` | Run full pipeline: generate → simulator → load. `corner=None` resolves the PDK manifest's `default_corner` (sky130: `tt`; GF180MCU: `typical`); an unknown corner raises with the available names listed. File-producing launches use [exact-file overwrite cleanup and failure invalidation](#testbench-output-directories-and-run-labels). |
| `simulate_behavioral(t_step, t_stop)` | Run Python/numpy behavioral simulation; returns `self` |
| `plot(*signal_names, overlay=False, show=True)` | Quick multi-panel plot of signals after simulation (`show=False` skips `plt.show()` so the figure can be customized first) |
| `view(*, out_path=None, open=True, signals=None, exclude=None, max_depth=None, title=None, inline=False, serve=False, port=0, context=None)` | Open the interactive waveform explorer on this run — `tb.simulate(); tb.view()` emits `view.html` beside the rawfile and opens it, no output path or `Report` needed. Delegates to `emit_view` using retained results when available; accepts its selector/flag set. `open=False` emits only and returns the path. Managed results refuse `serve=True`; to watch the public file, use `python -m silk.signals.view <rawfile> --serve` (blocks). Raises `ViewError` if unsimulated. The package-level `silk.view(source)` is the same shortcut for a rawfile path, a `RawFile`, or a simulated `Module` (not a `ModuleTestbench` — pass `tb.results_file`). For a report-linked page with auto-discovered provenance, use `report.add_waveform_view(tb)`. |
| `generate_netlist(corner=None, subckt=None, fault_set=None, manifest=None, seed=0, *, ngspice_version=None)` | Emit the SPICE netlist (`corner=None` → the PDK's `default_corner`; the `fault_set`/`manifest`/`seed` arguments are the fault-injection hooks used by `silk.faults`). ``ngspice_version`` accepts a release string such as `'45.2'`, `'42'` or `'47+'` and targets every compatibility decision and completion certificate in that call, including subcircuits. It requires ngspice as the effective backend; invalid values raise `ValueError`. `None` uses the installed release. Explicit targeting needs no local binary; `run_simulator()` raises `SimulationConfigurationError` before consuming the preparation when the installed release differs or is unknown. **Dual-mode:** if `module_name` is set (subcircuit mode), returns a `.subckt` string without writing to disk. If `module_name` is `None` (top-level mode), writes the full simulation netlist and returns its concrete file path (`str`): `dir_folder/netlistname` in overwrite mode, or the deck inside a fresh `run-*` child in new mode. The `subckt` kwarg overrides auto-detection; `subckt=False` with a named module raises `ValueError`. **Raises `CircuitError` when the module (or any submodule reached in the same deck) declares a port named `gnd`, any case** — ngspice folds that port name onto the global SPICE node `0` while Xyce does not, so the parent's wiring to it is silently discarded on one backend and honoured on the others. Backend-independent: the guard runs before the netlister is consulted. Rename the port (e.g. `vss`, verified not folded). The same condition is `STRUCT-012` at lint time. |
| `load_signal(signalName, signalType='V', *, analysis=None)` | Return a named `NgspiceSignal` (or behavioral signal). Use `load_signal('i(vdd)')` for a voltage source authored as `'vdd'`; silk resolves the emitted cards. Independent current-source readback needs saved current operands and has analysis-specific limits (see [element naming](#silkcircuit--element-naming)). The low-level `signalType='I'` form also accepts emitted card names such as `'vvdd'`. `analysis` (`'tran'`/`'ac'`/`'dc'`/`'noise'`/`'op'`) picks the plot in a rawfile written by a test that declared several analyses — see [multi-analysis-tests.md](multi-analysis-tests.md); a single-plot file needs no argument. Several plots require a selector; when device capture adds an implicit operating-point plot to one declared non-op analysis, that declared analysis remains the default. Raises `SimulationError` when the name is not in the rawfile. |
| `get_signal_names(*, raw=False)` | Return the loaded results' vector names, independent axis first. Array instances are shown as `[k]` (`i(vbias[0])`) only where collision-safe: names whose bracket spelling would coincide (a literal net `xx1[0].mid` beside the copy's `xx1<0>.mid`) all keep their raw spelling, so no new alias collision is introduced (duplicate rows already in the rawfile stay duplicates); order and count are preserved. `raw=True` returns the rawfile's exact names. A returned name does not always read back uniquely (a collision group is ambiguous by design; `load_signal` refuses it). |
| `run_simulator(auto_track_structure=False, timeout=None)` / `load_file(dataFile)` / `read_raw_file()` | `simulate()` chains `generate_netlist()` → `run_simulator()` → `read_raw_file()`, callable individually. Separately, `load_file(dataFile)` selects external results. Relative paths are used as given if they exist; otherwise silk tries them beneath `dir_folder`. |
| `results_file` | Result path maintained by silk: path of the rawfile the last run produced (`None` before any simulation); the source `emit_view`/`silk.view` read. On a `ModuleTestbench` the same-named read-only property forwards to its netlist module. |
| `annotate_currents(tstart=None, recursive=True, force=False)` | After a transient run made with `save_device_currents=True`, populate signed per-pin currents on each net's `PortBinding` (+ = device→net) and conservative `½·Σ|·|` aggregates on each `Net`, for EM wire sizing. Returns a `CurrentAnnotationReport`. See [Current annotation](#current-annotation). |
| `print_elements()` | Print element dictionary |
| `print_nets()` | Print net dictionary |
| `define_module_name(name)` | Update subcircuit name after construction. `name` is validated the same way as the constructor's `module_name` (must match `^[A-Za-z_][A-Za-z0-9_]*$`); an invalid name raises `ValueError` and leaves `module_name` unchanged. |
| `define_module_ports(ports)` | Declare module ports. Each value can be a `Port` (recommended — use factories `Port.analog_input`, `Port.supply`, `Port.ground`, …), a bare `PortDir`, a direction string (`'in'`/`'input'`, `'out'`/`'output'`, `'inout'`, `'power'`, `'ground'`, any case — normalized to `PortDir`; an unknown spelling raises `ValueError`), or a `(PortDir, PortRole)` tuple. A supply direction with no stated role implies the role: bare `PortDir.POWER` / `PortDir.GROUND` (or `'power'` / `'ground'`, or a role-less `Port(dir=...)`) is stored with `PortRole.POWER` / `PortRole.GROUND`, so it is equivalent to `Port.supply()` / `Port.ground()` for every role-keyed consumer; an explicit role is never overridden and IN/OUT/INOUT imply nothing. Two port names that differ only by case are ONE SPICE node (simulators fold node-name case), so declaring both raises `DuplicateNameError` naming both spellings — unlike **net** names, which merge onto the first spelling with a warning; a subcircuit's ports cannot be merged without changing its signature. Redeclaring a port with a different direction, role, or domain also raises `DuplicateNameError`; an identical redeclaration is a no-op. |
| `define_module_nets(pin_dict)` | Declare module pins (backward-compat; all default to `INOUT`) |
| `set_behavioral_model(fn, params=None, *, execution='vector')` | Attach an electrical-port model; `execution='vector'` is the default, with `'sample'` available for one call per point. The model is called as `fn(ports, t, params)`, reads `ports.VIN.v`, assigns `ports.VOUT.v` / `ports.VOUT.i`, and returns `None`. The parameter view reads the attached live dictionary through attributes or string keys; calibration writes are visible on later runs. Statistical declarations resolve per run. The execution contract participates in model identity and replay. Other explicit execution values raise `ValueError` without changing the attachment. |
| `create_test(name='default_test', parameters=None, *, strict_nets=None, save_device_params=None)` | Construct and register a `ModuleTestbench` linked to this module. Test names are unique within one owning module; a duplicate raises `DuplicateNameError` without replacing the existing test. `parameters` is an optional declared test-parameter map (e.g. `{'cload_f': 2e-12}`) that becomes part of the test's **characterization identity** in the tracking database (see [database.md#characterization-profiles](database.md#characterization-profiles)); it is never read by netlisting, and `None` means `{}`. `strict_nets` overrides the testbench's inherited `strict_nets` (default `None` inherits it from this module — see `ModuleTestbench`). `save_device_params=None` inherits DUT operating-point capture; a boolean overrides it for this test. Report caching is disabled; `ModuleTestbench(cache=True)` warns and every run simulates; `create_test` has no `cache` keyword. |
| `run_test(name='default_test', *, store=False, store_options=None, **config)` | Run one registered testbench by name (`config` is forwarded to `ModuleTestbench.run`, e.g. `corner='ff'`, `run_label='baseline'`; configure conditions with the test's `set_temperature()` and `set_supply_voltage()` setters — `temperature=`/`voltage=` are deprecated assertions). Returns the `SpecReport`. Raises `ValueError` naming the available tests when `name` is unknown. `store=True` (keyword-only, opt-in) then persists that report through `store_report()` against the already-bound circuit version and puts the receipt on `report.storage` (`metrics_stored`/`specs_stored`/`skipped` plus `cached`); nothing is created implicitly — no `db_path`, no bound version, a locked/unsealed version or a drifted live design is rejected **before** the run (`RuntimeError`/`LockedVersionError`/`DatabaseError`/`EvidenceMismatchError`). `store_options` forwards `store_report`'s `tool_name`/`units`/`sim_type`/`source`/`conditions`; `corner`/`temperature`/`voltage` there raise `ValueError` (the run is the only source of provenance). A save that fails after the run raises `ReportStorageError` carrying `.report`. See [database.md#run-and-store-in-one-call-run_teststoretrue](database.md#run-and-store-in-one-call-run_teststoretrue). |
| `get_test(name, *, instance=None)` | Omit `instance` (or pass `''`) for the original local `ModuleTestbench`. Pass a listed silk instance path for a stable selected handle with `metrics`, `report`, probes and execution methods. Use it for probes, metrics and rawfile readback after `run_all_tests()`. An unknown name raises `silk.exceptions.ConfigurationError`, naming the module and its registered tests (or saying none are registered). |
| `list_tests()` / `has_tests()` | Names of the testbenches registered on this module / whether there are any. |
| `run_all_tests(recursive=True, *, n_workers=None, **config)` | Run all testbenches registered on this module (and, when `recursive=True`, on every live nested submodule instance). Returns a `dict` mapping `"<module_name>.<test_name>"` → `SpecReport` (or a `TestFailure` — a `dict` equal to `{'error': str}` — for any test whose **circuit or run** misbehaved — `SimulationAborted`, a convergence failure, a backend exiting nonzero, a malformed or missing rawfile). One bad circuit never abandons the sweep. An error that means **the call itself is mis-configured** propagates instead and no dict is returned: every exception inheriting `silk.exceptions.ConfigurationError` — `CharacterizationContextError` (including a `temperature=` assertion that does not match configured TEMP), `SimulationConfigurationError` (a missing, consumed or stale prepared deck; a duplicate or unreadable temperature declaration), `NotSupportedError`, `RunArgumentError` (a keyword the tests' `run()` does not accept — the arguments are bound against each test's signature before it is called, so nothing runs) and `OutputPathError` (an unusable `run_label`, or an output directory that would leave the DUT output root). These configuration errors stop the whole call. If repeated instances would produce the same key, every colliding key is qualified as `"<owner_path>:<module_name>.<test_name>"`. **`n_workers`:** `None`/`1` (default) is the serial walk above, unchanged. Any other value — an int (capped at `os.cpu_count()`) or `'all'` (every CPU), resolved through `silk.simulation.executor.resolve_n_workers` — resolves every test's inherited conditions in this process first, dispatches simulation to a shared worker pool, and evaluates each result back here in the same order the serial walk would have used, so a clean tree's reports/summary/database records are identical either way; `ConfigurationError` still stops the whole call (even one raised inside a worker's simulate step) and an ordinary failure still becomes just that test's `TestFailure` entry. A test whose `run` is overridden by a subclass, or whose netlist resolved to the `behavioral` backend (models may be closures), always runs in-process regardless of `n_workers`. Each extra worker is a full OS process with its own parsed PDK, not a shared one — size `n_workers` for memory, not only CPU count. |
| `TestFailure` | What `run_all_tests` stores for a test that raised: a `dict` equal to `{'error': str(error)}` (so `r['error']`, `set(r) == {'error'}`, `r == {'error': ...}`, pickling and `json.dumps` behave as before; an exact `type(r) is dict` check sees a subclass). It also carries failure evidence captured when the test failed: `error_type` and `error_class` (the original exception's class name and dotted path), `details` (`simulator`, `reason`, `data_rows`, `log_path` for a `SimulationAborted`, empty otherwise), `exception` (the exception; `None` after a JSON round trip or if it could not cross a worker process), `exception_is_original` (`False` when `exception` is a stand-in for a class that could not cross a worker process), and `run` (a `RunRecord`, or `None` when nothing was launched or the launch is unknown). Serial and `n_workers>1` runs give the same object. Mutating `['error']` does not change the evidence. Importable from `silk.circuit`. |
| `RunRecord` | Frozen files record of one simulator launch: `launch_id`, `deck_path` (the prepared deck at its public path, not the private staged copy), `log_path` (the public log; for a failed run whose log could not be published, the retained private log, equal to `SimulationError.log_path`), `rawfile_path`, `noise_path`, `directory`; all absolute, `None` where the backend writes no such file. Paths only: silk promises no content reader, and ordinary runs keep public files in fresh folders by default; explicit `set_output_options(mode='overwrite')` replaces the files on the next run into the same folder (private artifacts can remain while retained result readers need them). On `SpecReport.run` and `TestFailure.run`. `SpecReport.launch_id` is `run.launch_id`, equal for every report that shares one launch (placements deduplicated by `run_all_tests`) and different for separate launches. An external rawfile or a prelaunch refusal has no record. `SpecReport.to_dict()` carries it as the optional `run` and `launch_id` keys (absent without a record; wire version stays 1; older files load with `run=None`). |
| `add_constraint(constraint)` | Attach a layout `ConstraintEnvelope` to this module. **Requires `import silk.constraints` first** — that import monkey-patches this method onto `Module`; it does not exist on bare `Module`. Raises `TypeError` if the arg is not a `ConstraintEnvelope` subclass. Returns the constraint. |
| `list_constraints()` | Return a copy of the module's constraint list. Requires `import silk.constraints`. |
| `assign_net_domain(net_name, domain)` | Assign a `VoltageDomain` to the named net in `net_dict`. |
| `add_no_connect_port(instance, port, reason)` | A whole-bus port name returns one marker per bit as a list; scalar and individual-bit calls return one marker. Register a per-port no-connect marker, suppressing `STRUCT-001` for that specific port on that instance. `reason` is required. A repeat call on the same `(instance, port)` — including a bit already covered by a prior whole-bus call — with an identical `reason` is a no-op returning the original marker; a different `reason` raises `DuplicateNameError` naming both. |
| `declare_spec(name, metric, op, target, units='', description='', parent=None, calibration=None, *, key=None)` | Create a first-class `Spec` owned by this module. `key=` is the stable persistence identity; when omitted, `name` is the key. Appends to `self._specs` and returns the `Spec`. See the `silk.specs` section below. |
| `reconcile_specs(prefer_source=None)` | Walk this module and every nested submodule, evaluate every declared `Spec` against its latest recorded attempt, apply any calibration side effects, return a `SpecReconciliationReport`. |
| `calibrate_behavioral(param_name, value)` | Directly overwrite `self._behavioral_params[param_name]`. Raises `RuntimeError` if the model is not parameterized and `KeyError` if the key was not declared. Manual path — normally calibration fires as a side effect of `reconcile_specs()` via a `CalibrationBinding`. |

Formal scalar and bus port names use the authored ASCII identifier grammar
(letters, digits and `_`, starting with a letter or `_`, with an optional
numeric index `[k]`). `define_module_ports` rejects formal
`'0'` with `ValueError` before registering that port; integer `0`/`1` remain
`TypeError`, and strings `'1'`/`'01'` remain invalid. Rename the formal to
`vss` and wire it to the parent's ground net. Actual ground net, device-pin,
passive and source-terminal values `'0'` remain valid. Formal `gnd` (any case)
retains its declaration-time acceptance and lint/emission-time refusal;
static setup and behavioral execution also refuse it, including `GND`.
Restored or mutated interfaces are rechecked before use with the same
block/port migration diagnostic (`CircuitError`). Actual/internal `gnd`
handling remains specific to the selected backend.

**State attributes:**

`elem_dict`, `net_dict` and `module_dict` are read-only `MappingProxyType`
views: entries cannot be assigned or deleted through them. Their nested records
and objects can still be mutable. `sig_sources_dict` and `passives_dict` are
ordinary mutable dictionaries.

| Attribute | Type | Description |
|---|---|---|
| `elem_dict` | Mapping | Maps instance name → element record |
| `net_dict` | Mapping[str, Net] | Maps net name → `Net` object with connectivity, parameters and an optional behavioral voltage array. See the `Net` section below. |
| `module_dict` | Mapping | Maps resolved subcircuit definition name → definition entry; placements are in `elem_dict` |
| `sig_sources_dict` | dict | Maps source name → source object |
| `passives_dict` | dict | Maps passive name → passive object |

**Tracking methods** (require `db_path`; snapshot lifecycle and failure
semantics are in [database.md](database.md)):

| Method | Description |
|---|---|
| `create_circuit_version(description='', netlist_path=None, *, parent_version_id=None)` | Atomically create or resolve an immutable sealed snapshot containing structure and spec declarations; return its UUID. |
| `store_results(results_dict, corner=None, temperature=None, voltage=None, sim_type=None, tool_name=None, source='user', conditions=None)` | Atomically store numeric/text metric observations (`{param: (value, unit)}`) with context. |
| `store_report(report, corner=None, temperature=None, voltage=None, sim_type=None, tool_name=None, units=None, source=None, conditions=None)` | Atomically store numeric report metrics, compatibility spec-status rows, and validated first-class spec measurements. Non-numeric metrics are returned in `skipped`. `source=None` takes the source from the report's recorded backend (`'behavioral'` or `'spice'`; `'spice'` when the report records none); an explicit `source` that contradicts a recorded backend raises `EvidenceMismatchError`. |
| `store_result(parameter, value, unit, ..., source='user', conditions=None)` | Store one numeric or text metric observation. |
| `store_pass_fail(spec_name, status)` | Store a compatibility text row; not canonical/restorable spec evidence. |
| `query_results(*, parameter=None, source=None, sim_type=None, corner=None, temperature=None, voltage=None, tool_name=None, conditions=None, profile_id=None, latest_only=False, include_ancestors=True)` | Return immutable flagged `SimulationResultRecord` history (checked/stale/not checked); selected stale latest rows raise. Each has `origin_version_id` and (nullable) `profile_id`. `profile_id=` is a 3-state filter: omit for no filter, `silk.db.UNSCOPED_PROFILE` for unscoped-only, or an exact profile ID. |
| `get_stored_specs(*, include_measurements=True, include_ancestors=True, profile_id=None)` | Return validated immutable `SpecDeclarationRecord` history; same 3-state `profile_id` filter as `query_results`. |
| `restore_spec_measurements(*, source=None, corner=None, temperature=None, voltage=None, tool_name=None, profile_id=None, include_ancestors=True)` | Validate the live declaration set, then atomically attach observations after rejecting any selected stale deck; unknowns warn and remain usable. Same 3-state `profile_id` filter. |
| `list_characterization_profiles(*, include_ancestors=True)` | List every `CharacterizationProfileRecord` compatible with this structure, nearest lineage first. |
| `resolve_characterization_profile(*, profile_id=None, context=None, include_ancestors=True)` | Resolve exactly one profile by ID or by a complete `CharacterizationContext` (never a partial one). |
| `select_characterization_evidence(*, profile_id=None, context=None, corner=None, temperature=None, voltage=None, include_ancestors=True)` | Return one exact `CharacterizationEvidenceRecord` (metrics + spec measurements + selected current set) for a profile at an exact PVT, without mutating the live module. |
| `restore_characterization_evidence(*, profile_id=None, context=None, corner=None, temperature=None, voltage=None, include_ancestors=True, force=False)` | Select, then atomically apply the same evidence onto the live module; validates before any mutation. Returns a `CharacterizationRestoreResult`. |
| `store_net_currents(report, corner, db_path=None, *, temperature=None, voltage=None, tool_name=None)` | Persist a new current set (unscoped, or profile-scoped when the testbench declared a full PVT context) for each characterized definition at the required corner. The prior selected set for that key is superseded, not overwritten — its `current_set_id` and rows remain readable. Call on the testbench netlist that ran `annotate_currents()`. |
| `load_net_currents(corner, *, db_path=None, force=False)` | Validate and load the currently-selected compatible unscoped current set. Missing/stale/pin-drift cases fail before live mutation; DB-owned rows absent from a replacement are cleared. |
| `set_corner(corner, temperature=None, voltage=None)` | Set corner for subsequent stores |
| `view_summary()` | Return a freshly rendered simulation summary, including unavailable observations; reads do not use or write stored summary cache columns. |
| `promote(to_stage, new_version=None, lock=True, description='')` | Create/reuse a sealed child UUID and rebind. Evidence (including characterization profiles) stays on its origin version; compatible reads search lineage. Locks the child by default, so later writes raise `LockedVersionError`. |
| `export_to_json(output_file)` | Export a marked lossy `silk.circuit-projection/v2`; not a database backup or circuit archive. |
| `get_corner_summary()` | Get all corners simulated |
| `get_corners_for_parameter(parameter, latest_only=False)` | Get results across corners — every stored observation by default (the db is append-only); `latest_only=True` selects the newest per (parameter, unit, corner, temperature, voltage), with parameter fixed by the argument. Its `value` is the current number, text, or `None` for unavailable numeric evidence; a newer text error supersedes an older number. |
| `close()` | Close database connection |

**`store_net_currents` / `load_net_currents` usage** — annotate, store, then
read currents back on a fresh build with zero re-simulation:

```python
# Here tb is the testbench's netlist Module, with current capture enabled.
tb.simulate(corner='tt')
report = tb.annotate_currents(tstart=1e-6)
tb.store_net_currents(report, corner='tt', db_path='chip.db')

fresh = build_dut(pdk)                          # unsimulated rebuild, e.g. next session
fresh.load_net_currents('tt', db_path='chip.db')
fresh.net_dict['out'].peak_current              # populated, no SPICE required
```

Two things to know: a **partial net** (KCL-exempt, e.g. MOS source/gate — see
the MOS drain-only limitation below, which applies to the plain `savecurrents`
flow and is resolved by `terminal_current_method='ammeter_wrapper'`, reading
all four MOS terminals) has no aggregate row and round-trips its
`Net.peak_current`/`avg_current`/`rms_current` as `None`, never `0.0`; and
`load_net_currents` will not clobber `'user'`/`'spice'` values it finds
already set — an in-session `annotate → store → load` round on the *same*
module needs `force=True` to overwrite the just-measured 'spice' values with
the just-stored 'db' ones.

**Versioning & staleness.** An exact live snapshot resolves by structure and
spec-set identity. Current applicability is structural: a spec-only successor
may reuse a compatible ancestor set, while a structural successor may not.
Every row carries a fingerprint scheme/digest and shared operating context;
unknown schemes, stale hashes, inconsistent context, and pin drift fail before
live mutation. One current set is selected per `(version, corner)` when
unscoped, or per `(version, profile, corner, temperature, voltage)` when
profile-scoped (see "Tracking methods" above and
[database.md](database.md)): re-storing at the same selection key mints a
new `current_set_id` and marks the old one superseded rather than
overwriting it — the old set's rows remain retrievable by that ID — while
other selection keys coexist untouched.

**Precedence on load** (per net aggregate and per pin, independently):

| target's `_current_source` | behavior |
|---|---|
| `'user'` or `'spice'` | preserved — skipped unless `force=True` |
| `'db'` or `None` | overwritten; DB-owned values absent from the replacement set are cleared |

Everything loaded is marked `_current_source='db'`. A later
`annotate_currents()` freely overwrites `'db'` values (fresh simulation is the
fresher truth), and `store_net_currents` never re-stores `'db'`-sourced values
(loaded evidence is not republished as a new measurement). Downstream consumers
(`layout.estimator`, `layout.matching`, spec resolution of
`'net.<name>.peak_current'`) read the populated fields directly — they do not
distinguish `'db'` from `'spice'`.

**Errors are loud and distinct:** unknown schemes raise
`UnsupportedFingerprintSchemeError`; known incompatible evidence raises
`EvidenceMismatchError`; missing version/corner, a stale row, or pin drift
raises `RuntimeError`; conflicting `db_path=` raises `ValueError`. Validation
precedes mutation. The complete exception matrix and provenance semantics are
in [database.md](database.md#failure-modes).

The four built-design reads `query_results`, `select_characterization_evidence`,
`restore_spec_measurements` and `restore_characterization_evidence` automatically
compare each row's producing test at its retained launch corner with the rebuilt
design's current normalized deck and backend. `silk.db.DeckCheck(state, reason)`
is exposed as `.deck_check` on SimulationResultRecord, SpecMeasurementRecord,
NetCurrentSetRecord and CharacterizationEvidenceRecord. It is an ephemeral
lookup snapshot, excluded from equality/hash behavior; raw Database reads leave
it `None` and emit no new warning. Silk exports/provenance/report schemas do not
persist verdicts, though caller asdict/pickle can include them.

States are `checked`, `stale` and `not checked`. Default metric history and
read-only aggregate selection return flags rather than refuse stale history.
Selected stale latest rows and restores raise EvidenceMismatchError; restores
check the complete selected set before any mutation. Not-checked values remain
usable with one combined UserWarning per public call. Aggregate selection and
successful aggregate restore always warn because current sets are not frozen
at launch; the aggregate is never wholly checked. Latest grouping is unchanged,
with no older-checked fallback. See [the deck-rule contract](database.md#automatic-deck-checks-through-a-built-design)
for coverage exclusions and the exact guarantee.

#### Typed wiring — port kwargs and `Module.nets` accessor

`add_module()` accepts port names as keyword arguments and their targets as
`Net` objects, so typos surface at the call site rather than as dangling-net
errors at simulate time:

```python
avdd = tb.add_net('avdd')
vinp = tb.add_net('vinp')
vout = tb.add_net('vout')
tb.add_net('0')

fcamp_inst = tb.add_module(fcamp, 'fcamp',
    vinp=vinp, vout=vout,
    avdd_1p8=avdd, vss=tb.nets['0'])   # bracket form for numeric-leading names
```

- **`tb.add_net(name)`** returns the new `Net` object — capture it once,
  reuse the variable wherever it wires.
- **`tb.nets.<name>`** is a live attribute view of `tb.net_dict`. A missing
  net raises `AttributeError` immediately; `dir(tb.nets)` lists every net
  for IDE autocomplete.
- **Kwarg values** may be a `Net` object *or* a net-name string — mixing is
  fine.
- **Kwargs override** entries in a `nets={...}` dict when both are given,
  which lets you write a default dict plus per-call tweaks.
- **Python reserved words** (e.g. `in`) cannot be raw keyword arguments.
  Use the `**{...}` escape hatch. Port names still need to start with a
  letter or underscore:
  ```python
  tb.add_module(sub, 'u1', **{'in': tb.nets.a, 'vdd': avdd})
  ```
- The `nets={port: net_name_str}` dict form also accepts declared bus names.

For a multi-bit child port, a parent bus name or bus `Net` handle expands
bit-wise, in declared bit order; a width mismatch raises a typed
`CircuitError` containing `BUS_WIDTH_MISMATCH`. Declaring a bus port creates
its handle automatically: after `define_module_ports({'A': Port(dir=PortDir.IN,
bits=4)})`, `mod.nets.A` can feed a child bus port. Both `bracketed` and
`underscore` naming work, and a later `add_net('A', bits=4)` returns the same
handle. Unused port bits do not add nets to the circuit graph.

A declared **scalar net or scalar port** broadcasts to every bit of an
**input or inout** bus, whether passed as a string or as a scalar `Net`. For
example, `parent.add_module(child, 'x1', nets={'P': 'vss'})` and
`parent.add_module(child, 'x1', P=parent.add_net('vss'))` both tie every port
bit of `P` to `vss`; for four bits, the instance line contains
`xx1 vss vss vss vss`. Declare `vss` as a scalar net or port before using its
string name. Scalar connections to **output** buses continue to raise
`BUS_WIDTH_MISMATCH`, because tying output bits together would short them.

An undeclared string for a multi-bit port raises `UndeclaredNetError`.
A bus-port name used as a scalar device connection also raises `CircuitError`
naming the port and its bit names; select an individual bit instead.

#### `SubcircuitInstance` — attribute-access to internal nets

`add_module()` places a `Module` (subcircuit) and returns a
`SubcircuitInstance` handle. The handle exposes every internal net and
every nested child subcircuit as an attribute, so reading signals
never requires building the `xfcamp.vbiasn` hierarchical string by hand:

```python
fcamp_inst = tb.add_module(fcamp, 'fcamp', nets={...})
tb.simulate(corner='tt')

fcamp_inst.vbiasn             # deferred net reference
fcamp_inst.vbiasn.y[-1]       # final voltage from the loaded results
fcamp_inst.vout               # port → resolves to top-level 'vout'

# Hierarchy: if fcamp is instantiated inside a bias wrapper:
bias_inst = tb.add_module(bias, 'bias', nets={...})
bias_inst.fcamp.vbiasn        # auto-composes 'xbias.xfcamp.vbiasn'
```

**Path resolution:**
- Internal nets → hierarchical path, e.g. `xfcamp.vbiasn`.
- Ports → the top-level net name they were mapped to in the `nets={}`
  dict at `add_module()` time, walking up through nested parents (ngspice
  aliases ports in the rawfile, so the hierarchical path wouldn't exist).
- Missing attributes raise `AttributeError`; use `dir(handle)` to discover available names. When a name identifies multiple kinds of item, attribute lookup raises `CircuitError`; use `handle.net`, `bus`, `child` or `device` explicitly.

`dir(fcamp_inst)` returns the net and child-instance names — IDEs get
autocomplete. Useful methods on the handle:

| Method / Attribute | Description |
|---|---|
| `handle.<net>` | Return a deferred net reference; numeric access reads the loaded results |
| `handle.<child>` | Return a nested `SubcircuitInstance` |
| `handle.path(name, *, raw=False)` | Resolved path string (no signal load); `raw=True` retains simulator array spelling |
| `handle.instance_name` | SPICE name with leading `x` (e.g. `'xfcamp'`) |
| `handle.subcircuit` | The underlying `Module` definition |

---

#### Context-manager auto-registration (`with mod:`)

A `Module` is a context manager. Every call to a PDK device factory,
a source constructor, or a passive constructor inside a `with mod:`
block auto-registers the new instance with `mod`. No explicit
`add()` / `add_source()` / `add_passive()` call is needed.

**Usage:**
```python
fcamp = Module('PDK_skywater130', out_dir, 'fc_amp.cir',
               pdk_details=pdk, module_name='fc_amp')

with fcamp:
    pdk.nmos1p8('mn1', d='vo', g='vi', s='0', b='0', w=1, l=0.18)
    pdk.pmos1p8('mp1', d='vo', g='vi', s='vdd', b='vdd', w=2, l=0.18)
    sources.CurrentSource('ibias', positive='vdd', negative='vb', dc=1e-4)
    sources.VoltageSource('vdd_src', positive='vdd', negative='0', dc=1.8)
    elements.capacitor('cl', p='vo', n='0', value=1e-12)

fcamp.define_module_ports({'vi': Port.analog_input(), 'vo': Port.analog_output(),
                           'vb': Port.analog_input(), 'vdd': Port.supply()})
```

**What auto-registers inside the block:**

- Any `pdk.<device>(...)` factory call (MOSFETs, resistors, capacitors,
  inductors, diodes, BJTs — any `AnalogElement` subclass) → `mod.add(self)`
- Any `elements.resistor/capacitor/inductor(...)` call (ideal R/C/L via the
  `ElementLibrary`) → `mod.add(self)`
- Any `sources.*` constructor (`VoltageSource`, `CurrentSource`, VCVS /
  VCCS / CCVS / CCCS, `BehavioralVoltage`, `BehavioralCurrent`) →
  `mod.add_source(self)`
- `passives.*` (`Resistor`, `Capacitor`, `Inductor`) also still auto-registers,
  via `mod.add_passive(self)`, but is **deprecated** — prefer `elements.*` above.

**Explicit registration** works both outside and inside the block. Outside a
`with` block, call `add()` / `add_source()` / `add_passive()` yourself:

```python
# Explicit registration
mn1 = pdk.nmos1p8('mn1', d='vo', g='vi', s='0', b='0', w=1, l=0.18)
ibias = sources.CurrentSource('ibias', positive='vdd', negative='vb', dc=1e-4)
cl = elements.capacitor('cl', p='vo', n='0', value=1e-12)

mod.add(mn1)
mod.add_source(ibias)
mod.add(cl)
```

**Mixing styles is safe.** Calling `mod.add(x)` inside the block on
something that was already auto-registered is a no-op (idempotency
is based on instance identity — a different instance sharing the
same name still raises `DuplicateNameError`).

**Declarative source library.** Sources can also be instantiated from a
YAML-declared library the way PDK devices are (`pmos1p8`-style), with the
waveform as the primary noun. Both forms are generated from
`silk/sources/source_metadata.yaml`:

```python
# Quantity namespace:
sources.voltage.dc('vdd', 'vdd', '0', 1.8)
sources.voltage.pulse('clk', 'clk', '0', v1=0, v2=1.8, td='1n', tr='1n',
                      tf='1n', pw='5n', per='10n')
sources.current.dc('ib', 'a', '0', 1e-3)

# Flat aliases (the same function objects):
sources.vdc('vdd', 'vdd', '0', 1.8)
sources.vpulse('clk', 'clk', '0', v1=0, v2=1.8, td='1n', tr='1n',
               tf='1n', pw='5n', per='10n')
sources.idc('ib', 'a', '0', 1e-3)
```

Terminals accept `p`/`n` or `positive`/`negative`; `dc=`/`ac=` are optional
offsets that coexist with a waveform (`vpulse(..., dc=0.9)`). Each builds a real
`VoltageSource`/`CurrentSource`, so emission, `add_source`, and the
optimization-variable API are unchanged; `ckt.add(v)` accepts a source too. A new
waveform kind is a YAML entry + an emitter class (with a `to_spice()`) — no source-class
edit. The hand-written `VoltageSource(...)`/`add_source(...)` form is fully
supported and unchanged.

**Nesting** — nested `with` blocks push on a thread-local stack; the
innermost active module receives new elements. On block exit (normal
or exception), the stack pops cleanly.

```python
with buffer:
    with inverter:
        pdk.nmos1p8('mn', d='vo', g='vi', s='0', b='0', w=1, l=0.18)
        # → registered to `inverter` (innermost)
    buffer.add_module(inverter, 'inv1', nets={...})
```

---

### `ModuleTestbench`

Test harness for a `Module`. Wraps a module under test, wires sources and
passives around it, defines simulation type, declares metrics and specs, and
runs simulations to produce pass/fail results. `Module.create_test()` and
direct `ModuleTestbench(module, name)` construction both register the test on
the owning module, making it visible to `list_tests()` and `run_all_tests()`.
Names are unique within one module (the same name may be used on another
module); a duplicate raises `DuplicateNameError` and preserves the first test.

```
ModuleTestbench(module, name='default_test', cache=False, parameters=None, *, strict_nets=None, save_device_params=None)
```

| Parameter | Type | Description |
|---|---|---|
| `module` | `Module` | The module under test |
| `name` | `str` | Test name (used in `SpecReport` and `CornerResults`); unique within the owning module |
| `cache` | `bool` | Caching is disabled. `cache=True` warns; every run simulates and extracts fresh results. `clear_cache()` remains callable. |
| `parameters` | `dict \| None` | Declared test-parameter map (`None` → `{}`). Part of the test's characterization identity for the tracking database; stored as a frozen deep copy (`tb.parameters`), never read by netlisting. |
| `strict_nets` | `bool \| None` | Keyword-only. Whether the testbench netlist rejects undeclared nets (an `UndeclaredNetError` naming the net, same as a strict standalone `Module`) instead of auto-creating them. `None` (the default) **inherits** `module.strict_nets` — a strict DUT gets a strict testbench, so a misspelled net in `add_module_under_test`'s `port_mapping` raises rather than silently wiring a fresh net. Pass `True`/`False` to override the inherited value. `Module.create_test()` exposes the same keyword. |
| `save_device_params` | `bool \| None` | Keyword-only. `None` inherits DUT operating-point capture; `True`/`False` overrides capture for this test. |

#### Operating conditions

Use `tb.set_temperature(85); tb.set_supply_voltage(1.8); report = tb.run()` on
an already wired test. The test setters forward to `tb.netlist`; a plain Module
uses the same setters followed by `simulate()`. Native
`set_simulator_options(temp=85)` shares the temperature setting.
`run(temperature=85)` and `run(voltage=1.8)` are both deprecated
**assertions**, requiring matching setup before launch; neither configures
anything. All wrappers forward that contract.

**Which source is the supply.** `set_supply_voltage(volts)` sets one voltage
source's DC value before the deck is emitted, so the supply a report names is
the supply the simulator was given. Silk decides which source that is only from
evidence you supplied, in this order:

1. `set_supply_voltage(volts, source='vdd')` — the name of a source you
   created. Always wins, and always available.
2. A declared `POWER` port role. Among the purely-DC voltage sources referenced
   to ground (SPICE node `0`, or a net whose role is `GROUND`), the one whose
   positive net is classified `POWER` by `define_module_ports`, `Port(role=)`
   or opt-in `infer_port_roles`.
3. A single DC rail. When nothing is role-classified and the bench has exactly
   one purely-DC voltage source to ground, that is the supply.

Nothing else resolves. Zero or several equally supply-shaped sources leave the
supply **unresolved**: silent while nobody asks (the report records unknown),
and a `SimulationConfigurationError` naming the candidates when preparation
or execution resolves a configured supply. No port name, net name or source name
decides anything — those rules live in your configuration. A time-varying source is never a supply: no scalar setpoint
is true of a moving rail, so record its waveform knobs as declared test
parameters instead. Several rails are never collapsed into one scalar; name the
primary one.

`set_supply_voltage(None)` clears the **setting**, the way
`set_temperature(None)` does. It does not restore a value an earlier call
overwrote: the source keeps the DC value it was last given, which is the one
the deck carries.

The supply inherits exactly as temperature does (the table below applies to
both), and the named source is re-resolved against each executing testbench, so
one setting on a circuit reaches every test that has a rail of its own.

**Where the setting has to be applied.** A run applies the temperature of the
Module that executes it, which for a registered test is `tb.netlist` and not
the circuit under test. Three levels reach it, nearest first:

| Set it on | Applies to |
|---|---|
| The test (`tb.set_temperature(85)` / `tb.set_supply_voltage(1.8)`) | That test's runs, always. It wins over any inherited value in both directions, and `None` opts the test out of its owner's setting rather than resuming inheritance. |
| A block that owns registered tests (`block.set_temperature(85)`) | Every test of that block with no setting of its own, and the block's own `simulate()` if it is run directly. This is the one-call default for an embedded testbench: a block with five tests is made hot once, without a wrapper circuit. A test's own setting still wins. |
| A top-level circuit (`chip.set_temperature(85)` / `chip.set_supply_voltage(1.8)`) | That circuit's own `simulate()`, plus every test with no setting of its own that runs from it — its own registered tests, and under `run_all_tests(recursive=True)` the tests of the blocks it contains, unless a nearer block or test set one. |

The value is re-read at **every** run, so changing or clearing it on the
circuit changes the next run of every test that inherits it; an inherited value
never becomes the test's own. A propagated setting is in force for that
invocation only and changes nothing stored on a child, so running a child
block's test on its own asks that block for its own setting. The setter
raises `ValueError` only where nothing would apply it: a subcircuit that owns
**no registered tests**, since an instantiated `.subckt` carries no simulator
options and holds no rail of its own. Register a block's tests before setting
its default — the refusal reads the Module as it is at the call.

A successful Module and its `SpecReport` expose an immutable
`silk.simulation.OperatingConditions` snapshot:

| Field | Meaning |
|---|---|
| `origin`, `backend`, `corner` | Silk execution, behavioural run or external declaration; selected backend (external: loader hint); the process corner **only when `corner_kind == 'fixed'`**, otherwise `None` |
| `corner_kind`, `corner_basis` | `fixed`, `none`, or `unknown`; and how it was established (see [the four states](#operating-conditions-none-unknown-and-the-consumer-contract)) |
| `temperature_kind`, `temperature_c` | `fixed` with a scalar Celsius value, `none` or `unknown` with `None`, or `sweep` with `None` |
| `temperature_basis` | `configured-and-observed`, `native-and-observed`, `raw-and-observed`, `backend-observed`, `caller-declared`, `proven-behavioral` (none), `configured` (silk DC-temperature sweep) or `unknown` |
| `temperature_sweep` | Immutable `ConditionSweep(start, stop, step)` for a recognized DC temperature axis; otherwise `None` |
| `supply_kind`, `supply_voltage_v` | `fixed` with the primary-supply setpoint of the deck that was launched, `none` (the stated rule below) or `unknown`; or `sweep` for a designated-supply DC sweep; the value is `None` for none, unknown and sweep, and is never invented |
| `supply_basis` | `configured-and-emitted`, `configured-and-driven` (a `'behavioral'` backend run drove the configured source at that value), `declared-and-emitted`, `caller-declared` (external evidence with a declared voltage or none), `proven-no-supply` (none), `configured` (supply sweep), or `unknown` |
| `supply_sweep` | Immutable `ConditionSweep(start, stop, step)` in volts for a recognized DC sweep of the designated primary supply; otherwise `None`. `ConditionSweep` uses degrees Celsius in `temperature_sweep`. |
| `supply_source` | Name of the source the supply was read from, or `None` |


A DC analysis that targets the designated primary supply records
`supply_kind='sweep'`, `supply_voltage_v=None`, the resolved `supply_source`,
and configured `supply_sweep` bounds in volts. This label applies to the
whole launch, including mixed analyses. Sweeping another source or temperature
does not change a fixed supply condition. `run(voltage=...)` refuses a supply
sweep because one scalar cannot confirm the sweep, even at the nominal card
voltage. Behavioral runs verify the rail against the DC plot's actual axis.
External/unprovable evidence does not infer a supply identity from an axis.

Condition dictionaries omit `supply_sweep`
when absent, and strict report readers accept historical records without it.
Supply-sweep evidence can be stored unscoped; it cannot create or match a
fixed-voltage characterization profile. Current databases use schema v7:
follow the earlier copy migrations with `silk.db.migrate_v6_to_v7(source, dest)`.


#### Operating conditions: none, unknown and the consumer contract

Each of corner, temperature and supply is in exactly one of four states
(`silk.simulation.ConditionState`, a plain `Enum`; on the wire the strings
`'fixed'`, `'none'`, `'unknown'`, `'sweep'`):

| State | Meaning | Value | Facts |
|---|---|---|---|
| value (`'fixed'`) | One scalar is known, and a basis says how | yes | all three |
| none | The condition provably does not apply to this evidence (an ideal RC has no PDK corner and no supply) | no | all three |
| unknown | It applies, but silk cannot establish it | no | all three |
| sweep | The evidence spans a range of the condition | bounds only | temperature or supply |

None is recorded **only on proof**; anything silk cannot prove is unknown.
A scoped `CharacterizationContext` (a profile) needs every fact to be a value
or none; unknown and sweep never form a profile.

**The consumer contract.** These hold for every consumer of a live snapshot:

- `.corner is None` means none or unknown; `.supply_voltage_v is None` can
  also mean sweep. Read the kind (`corner_kind`, `supply_kind`, `state(fact)`)
  to distinguish them. `scalar(fact)` returns a value, `ConditionState.NONE`,
  `ConditionState.SWEEP`, or `None` for unknown. The value fields never hold
  a `ConditionState` member.
- `SpecReport.corner` is the run's **label** (the corner that was requested),
  not evidence of what ran. A PDK-free deck run with `corner='ss'` keeps the
  label `'ss'` while its snapshot says corner none.
- A consumer reads the kind and **never fills a `None` value from the label**.

**Corner none** is proven when no PDK device exists anywhere in the hierarchy
(`has_no_pdk_device`, below): the emitted deck carries no PDK `.lib` or
`.include`, so no corner reached the simulator (`corner_basis='proven-no-pdk-device'`).
A deck that keeps its PDK includes records `configured-and-emitted`. The
behavioural engine records corner and temperature none (`proven-behavioral`).
A `'behavioral'` backend run (`simulator = 'behavioral'`, `simulate()`)
records `origin='silk'`, `backend='behavioral'`. `simulate_behavioral()`
uses that same backend and records the same origin and backend, then restores
the caller's simulator selection and analysis declarations.
`run_corners` over two or more distinct corners of a PDK-free deck emits one
`UserWarning` (the runs are identical) and records each as corner none; result
keys are unchanged.

**The supply rule.** Supply is **none** (`proven-no-supply`) only when all hold
over the whole hierarchy: no supply is configured or declared; the corner is
proven none; no net is classified `PortRole.POWER` (declared ports,
`Port(role=)` or opt-in role inference, never a name list); and every
independent source carries a waveform or has a literal-zero `dc` (a
parameter-reference `dc` fails; controlled sources are ignored). When any check
fails the supply stays unknown, and `set_supply_voltage(...)` makes it a value.

**The public proofs** `has_no_pdk_device(module)` and `has_no_supply(module)`
([below](#operating-condition-proofs)) answer "PDK-free / supply-free" before or
without a launch, and silk's own launch path calls the same functions.
A launch with a custom DC analysis records the supply as unknown even when
this inventory proof holds.

**External evidence.** `evaluate(rawfile=...)` cannot prove anything about a
deck it did not launch. With no `corner=` and no `set_corner()` the corner is
**unknown** (not the PDK default) and no profile forms; declare each fact with a
value or `ConditionState.NONE` (basis `caller-declared`). Other
`ConditionState` members raise `CharacterizationContextError`.

**Basis categories.** Every basis word carries a machine-readable `category`
(`observed`, `configured`, `declared`, `proven`, `default`, `legacy`,
`unknown`) so a consumer can group a basis it does not know.
`silk.simulation.export_conditions_schema()` (`format_version` `1.3`) returns
the states per fact, which may scope a profile, every basis word and its
category; read it instead of hard-coding the strings. `CONDITION_VOCABULARY`
is the in-process definition. Labels for display are the consumer's.

**JSON and storage.** A condition on any JSON record is the value (a string or
number), the tagged object `{"state": "none"}` / `{"state": "sweep"}`
(`encode_condition` / `decode_condition`), or `null` for unknown. A reader that
treats every non-null corner as a name or every non-null temperature as a
number fails on the tagged object instead of mislabelling. `SpecReport.summary()`
prints each fact as a value, `none (<reason>)`, `sweep` or `unknown`. Run
provenance is scheme `silk.provenance/v3`: it retains v2's `corner_kind` and
`supply_kind`, adding optional producing deck/test/launch-label facts. v1/v2
records still load with absent new facts and preserve their original scheme.

**Condition states in the database.** `simulation_results`, `spec_measurements` and
`net_current_sets` carry `corner_state`, `temperature_state` and
`voltage_state` beside the value columns. A v4 file does not open:
`UnsupportedSchemaVersionError` names `silk.db.migrate_v4_to_v5(source, dest)`,
a copy migration (the source is never modified). It maps a non-NULL value to
`fixed`, NULL to **unknown, never none** (none is a proof made at launch and a
v4 row keeps no deck to re-prove it from), and a NULL temperature recorded as a
sweep to `sweep`. Every profile keeps its hash and its evidence selection. The current schema is
v7: follow with `silk.db.migrate_v5_to_v6(source, dest)` (nullable aggregate
evidence) and `silk.db.migrate_v6_to_v7(source, dest)` (supply-sweep states)
before opening the final copy with `Database`. A
`corner=`/`temperature=`/`voltage=` selector matches state and value, so
`ConditionState.NONE` never selects unknown rows. The JSON projection is
`silk.circuit-projection/v2`. See [known_issues.md](known_issues.md#operating-conditions-none-versus-unknown).

**What the supply claim proves.** The configured or declared value is confirmed
against the deck that is actually launched: the one line declaring that source,
and the DC value on it. A value that never reached the deck, or custom SPICE
that redeclares the source, raises `SimulationError` before the launch and
publishes no report. That is a claim about what the simulator was *given*, not
a measurement of the rail the circuit settled at — read the node itself if you
need that. A supply nobody established stays `None`, and no nominal is assumed.

ngspice and Xyce both confirm ambient TEMP from this launch before metric
extraction, to the precision each reports (ngspice six decimals from its run
log, Xyce eight from the `{TEMP}` probe printed beside its rawfile); confirmed
configured values retain their original numeric precision.
Configured TEMP with missing/disagreeing confirmation raises `SimulationError`
and clears live evidence. With no explicit setting, a unique observation may
establish TEMP; otherwise it is unknown. No 27 C default is invented and no
default directive is emitted. TNOM remains the model reference temperature.
DC `source='temp'` (including mixed analyses) makes the report swept/unscoped;
a fixed initialization temperature cannot label sweep-wide metrics.

On ngspice, a sole literal `.temp 85` or `.options temp=85` added with
`add_custom_line()` is recognized and independently confirmed by the launch log.
Recognition is case-insensitive, covers multiline strings and `+` continuation
lines (including separate custom entries), and preserves the emitted bytes.
A literal TEMP embedded in native option text, such as
`set_simulator_options(reltol="1e-5 temp=85")`, is recognized too. These settings
report `raw-and-observed`; ordinary `set_simulator_options(temp="85")` retains
`native-and-observed`. Literal values use the existing SPICE-number parser and
must be finite and above -273.15 C. Invalid raw literals raise `SimulationError`
before launch. Two or more explicit declarations, including equal duplicates
and raw/typed/native combinations, raise `SimulationError` naming each input.
Repeated setter calls still replace the shared setting. Prefer the typed setter
for new code; a matching literal raw setting also satisfies the deprecated
`run(temperature=...)` assertion.

Inspection is limited to emitted top-level caller content and native option
text. It does not evaluate expressions or quoted values, control blocks,
conditional logic, raw temperature lists, or caller-supplied subcircuit bodies,
and never expands `.include`/`.inc`/`.lib` files. These inputs yield unknown
temperature even if a startup log contains a scalar. Combining opaque inputs
with managed TEMP fails preflight with `NotSupportedError`; a scalar assertion
also cannot establish their temperature. Unrelated custom lines and comments
leave backend observations available. TNOM options do not establish ambient TEMP.
Registered PDK model libraries remain the trusted boundary and are not re-read.
ngspice and Xyce both accept managed TEMP, fixed assertions and DC temperature
sweeps; every other backend refuses them rather than labelling a request as
applied, and its unmanaged runs retain unknown temperature. Behavioral
simulation also rejects configured TEMP. On Xyce, TEMP is emitted into the
DEVICE options package and confirmed at runtime; a run configuring none reports
unknown rather than Xyce's 27 C default, and `.temp` is an ngspice card Xyce does
not recognize at all — see
[Xyce support and limitations](known_issues.md).

Fixed known temperature plus an established supply produces the existing complete
`CharacterizationContext`; temperature-only, unknown or swept runs are valid
and unscoped. Storage preserves their partial PVT, including NULLs; neither
tracker defaults nor storage arguments can fill unknown values. See
[database.md](database.md#characterization-profiles).

A changed prepared temperature or edited deck requires `generate_netlist()`
before `run_simulator()` in either output mode. Direct externally prepared decks
without retained preparation produce unknown temperature. The report cache
(`cache=True`) is disabled — see the `cache` parameter above. Every `run()`
simulates. Reports keep their scalar observations; their `sim_results` reader
retains the selected run when placement evidence is available.

Explicit `evaluate(rawfile=...)` always rereads and gives external origin, even
for a saved local result. External temperature and voltage may be declared
independently; omitted conditions remain unknown. Omitted-file evaluation reuses the snapshot: Silk-origin arguments
can only confirm it, while external memory allows redeclaration. Recognized
loaded temperature axes reject scalar PVT declarations. A mixed-analysis file
may contain only its final plot; only axes actually present can be recognized.

#### Testbench authoring loop

Build and run a testbench in seven steps:

```python
tb = amp.create_test('ac_test')

# 1. Wire the module under test into the testbench netlist
tb.add_module_under_test({'vp': 'inp', 'vm': 'inn', 'vout': 'out', 'vdd': 'vdd', 'vss': '0'})

# 2. Add surrounding sources
tb.add_source(sources.VoltageSource('vdd_src', positive='vdd', negative='0', dc=1.8))
tb.add_source(sources.VoltageSource('vin', positive='inp', negative='0', dc=0.9, ac=1))
tb.add_source(sources.VoltageSource('vref', positive='inn', negative='0', dc=0.9))

# 3. Add surrounding passives (optional)
tb.add(elements.capacitor('cl', p='out', n='0', value=1e-12))

# 4. Set the simulation type
tb.add_sim_type(AC('dec', 20, 1e3, 1e9))

# 5. Define metrics: name the signal, hand it to a registered metric
out = tb.probe('out')                      # unbound Signal; no data until step 7
tb.add_metric('gain_dc', dc_gain_dB, sig=out)
# (the callable form, `lambda m: dc_gain_dB(m.load_signal('out'))`, still works
#  and is the answer when the extractor does more than load and call)

# 6. Declare specs against metrics
tb.add_spec('gain_spec', 'gain_dc', '>=', 60.0)

# 7. Run
report = tb.run(corner='tt')               # SpecReport
results = tb.run_corners()                 # CornerResults over all 5 PVT corners
```

**`ModuleTestbench` methods:**

| Method | Description |
|---|---|
| `add_module_under_test(port_mapping)` | Wire the module into the testbench using `port_mapping` (dict of module-port → testbench-net-name). Must be called exactly once; raises if called a second time. The DUT is placed as instance `dut` (`xdut` in the netlist); designer node paths into it start with `dut.`, e.g. `tb.probe('dut.mid')`. Its **internal** nodes remain `xdut.<net>` in the rawfile. Its **ports** alias to the mapped bench nets: `tb.probe('dut.out')` follows that wiring (`xdut.<port>` has no separate vector). |
| `add(*items)` | Unified front door parallel to `Module.add()`: dispatches source objects to `add_source` and forwards every other item (PDK devices, ideal R/C/L, other elements) to the testbench's top-level circuit via `Module.add()` with the same validation and errors. This is the form generated testbench code uses (`tb.add(sources.vdc(...))`). Returns `self`. |
| `add_module(module, name, nets=None, **ports)` | Place a named helper subcircuit beside the DUT; lazily creates the executing netlist and returns the `SubcircuitInstance` from `Module.add_module`. Ports are wired via `nets={'pin': net, ...}` and/or `**ports` keyword form — the positional third argument keeps accepting a plain dict for existing callers. Helpers may be added before or after the DUT, but `add_module_under_test` is still required exactly once before running. Existing port, duplicate-name and strict-net checks apply; missing required ports fail at netlist generation. A non-Module raises `TypeError`; a Module without a name raises `NotASubcircuitError`. See the [helper contract and example](codegen-contract.md#testbench-helper-submodules). |
| `connect_module(name, nets=None, **ports)` | Forward partial rewiring to the existing bench netlist for a helper or the DUT (name `dut`). Uses `Module.connect_module` rules. Before a netlist exists, raises `CircuitError` naming `add_module_under_test` and `add_module`; does not create a netlist. |
| `add_source(source)` | Add a source to the surrounding testbench netlist. Returns `self` for chaining. |
| `add_passive(passive)` | Add a passive (R/C/L) to the surrounding testbench netlist. Returns `self` for chaining. |
| `add_sim_type(simType)` | Set the simulation analysis. Requires a typed analysis object (`AC`, `Transient`, `DC`, `Noise`, `Op`); strings raise `TypeError` (the legacy string form was removed). Returns `self` for chaining. |
| `probe(name, type='V', analysis=None)` | Name a signal of this testbench and return it as an **unbound** `Signal` — no data until the bench has run. Spellings: a plain net (`'out'`/`'v(out)'`, validated against this testbench's nets at the call); a DUT-internal path (`'xdut.mid'`, not validated — a rawfile-presence fact); a source branch current (`'i(vs)'`, the emitted-instance mapping applied for you); a device terminal current (`'xdut.m1:d'`, through the 0 V ammeter wrappers — needs `save_device_currents=True` on the DUT, `terminal_current_method='ammeter_wrapper'`, and a `Transient` analysis, or raises naming what's missing). `type='V'/'I'` is the low-level loader selector, not the taught form. **Which analysis it reads:** `analysis=` (`'tran'`/`'ac'`/`'dc'`/`'noise'`/`'op'`) names the plot in a test that declares more than one analysis; an unknown word raises `ValueError` at the call, and whether the test declares that analysis is settled at run time. Omit it when the test declares one analysis, or when only one of the declared analyses can produce the input kind of the metric this probe feeds (`dc_gain_dB` in a tran+ac test reads the AC plot with nothing said); it is **required** when more than one could (`average`, whose input kind is `any`), and its absence raises `MetricBindingError` from `run()` naming the candidates. A probe that names an analysis reads that plot from `.x`/`.y`, `snapshot()` and `tb.plot()` as well. See [multi-analysis-tests.md](multi-analysis-tests.md). Arithmetic (`+ - * /  **`, unary `-`) and every Signal-to-Signal transform (`mag`, `derivative`, `window`, `fft`, ...) on an unbound Signal **defer**: `vcm = (tb.probe('vp') + tb.probe('vm')) / 2` builds a recipe, not data. Reading `.x`/`.y`/`.xlabel`/`.ylabel` (or `snapshot()`) resolves against the bench's *current* results on access and stores nothing back; **`Signal.snapshot(module=None, *, analysis=None, corner=None)`** returns a plain Signal frozen at the current data (keeps an earlier run across a re-run, or outlives the bench); `resolve()` is its alias with identical behaviour. Reading, metrics and plots never need it. Before the bench has run, reading raises `UnboundSignalError` (never `AttributeError`), naming the probe and the bench. `probe.bound` answers the same "has this run" question without raising. **Across corners:** after `run_corners([...])` retained more than one corner, `.x`/`.y` **raise** `UnboundSignalError` naming the corners instead of quietly showing the last one, and `bound` is `False`; `probe.corner('ff')` returns a bound `Signal` carrying that corner's samples (each leaf of a derived expression resolved from that corner's file, the live netlist untouched), and `probe.plot()` overlays every retained corner with a legend. After a plain `run()`, the probe reads that run directly. Corner sweeps retain separate folders in either output mode. |
| `add_metric(name, extractor_func=None, *, type=None, signal=None, signals=None, analysis=None, **params)` | Bind the registered function's named Signal inputs to probes or expressions. <br>**Function:** `vcm = (tb.probe('vp') + tb.probe('vm')) / 2; tb.add_metric('vcm_avg', average, sig=vcm)`. <br>**Declarative:** `tb.add_metric('gain', type='dc_gain_dB', signal='vout')`. <br>**Callable:** `tb.add_metric('gain', lambda m: dc_gain_dB(m.load_signal('out')))`. <br>**Derived:** `tb.add_metric('ratio', numerator / denominator)`. <br>**Binding and analysis rules:** A slot value may also be a plain signal-path string, resolved through `Module.load_signal` exactly as the declarative form below. Positional signal arguments raise `TypeError` naming the slots — `hysteresis`'s four inputs cannot be swapped silently. Codegen also supports metrics over derived Signals via function-form emission; see [derived signals](codegen-contract.md#derived-signals-waveform-math). Read signals directly with `tb.netlist.load_signal(name)` or `report.sim_results.load_signal(name)`. The declarative form resolves `type` against the registered metric catalog (`silk.signals.schema.get_metric`; `export_metric_schema()` lists 80 scalar entries; six result-object metrics also register declared outputs for Python binding; `p1db` resolves by `type=`, but its input declares `kind='power_sweep'`, which no analysis produces, so a consumer that evaluates the schema's refusal rule will not offer it; E.g. `bandwidth_3dB`, `dc_gain_dB`, `phase_margin`, `rise_time`, `settling_time`, `enob`, `thd_percent`; extra `**params` such as `freq=` are forwarded); `signal` is a rawfile signal name — a testbench net (`'vout'`) for anything on a DUT port, `'xdut.<net>'` for a DUT-internal node. **N signal inputs:** a metric that takes more than one `Signal` — `propagation_delay`, `average_power`, `dynamic_energy_per_cycle`, `error_norm`, `relative_error_norm`, `hysteresis` (four), `cmrr_dB` — is declared with `signals={slot: path}` instead, keyed by the metric's own signal-input parameter names as published in `export_metric_schema()[type]['signal_inputs']` (in schema `format_version` `'2.0'` each entry is a `{name, kind}` object — read `entry['name']`): `tb.add_metric('cmrr', type='cmrr_dB', signals={'vdiff': 'voutd', 'vcm': 'voutc'}, freq=1e3)`. By name and never by position, because a swapped pair is a plausible wrong number rather than an error. `signal=` and `signals=` are mutually exclusive; `signals={'sig': path}` on a one-input metric is the explicit spelling of `signal=`. In the callable form, `extractor_func` receives the post-simulation testbench `Module` and returns a float. The binding is validated **when the metric is declared**, not after a run: a missing slot, an unknown key, or `signal=` on a multi-input metric raises `TypeError` naming the slots involved. **Which analysis it reads:** in a test declaring more than one analysis, the declarative form takes `analysis=` (`'tran'`/`'ac'`/`'dc'`/`'noise'`/`'op'`), the function form names it on the probe (`tb.probe('vout', analysis='tran')`), and the callable form passes the keyword to `ckt.load_signal` — `analysis=` here is refused for the latter two. Omit it when the test declares one analysis, or when only one declared analysis can produce this metric's input kinds (`dc_gain_dB` in a tran+ac test); it is **required** when more than one could (`average`, whose input kind is `any`), and its absence is refused before extraction naming the candidates — `ValueError` from the declarative form, `MetricBindingError` from the function form. `domain` is never consulted for the choice. See [multi-analysis-tests.md](multi-analysis-tests.md). Returns a `MetricHandle`; public testbench methods forward for chaining. The derived form consumes earlier reported metric values; read `handle.value` after evaluation. Arithmetic supports `+ - * / ** abs` and numeric operands. Boolean conversion raises; equality/hash retain identity. See [metric handles](probes.md#derive-metrics-from-reported-values). |
| `add_spec(name, metric_name, op, target, *, key=None)` | Declare a pass/fail spec. `metric_name` is a name previously registered with `add_metric`, or the `MetricHandle` it returned (resolved to its name; a handle from another testbench raises `ValueError`, an unregistered expression raises naming `add_metric`, any other non-`str` raises `TypeError`). Also declares the spec on the owning module (`Module.declare_spec`), so it is a first-class design `Spec`; `key=` is its stable persistence identity (defaults to `name`). Creates a `Spec` and returns it (the `Spec` itself; not `self`). |
| `set_supply_voltage(volts, *, source=None)` | Forward to the executing netlist setter and return the testbench. This test's own setting, so it wins over any supply it would otherwise inherit from its owner, in both directions; `None` here opts the test out rather than resuming inheritance. |
| `set_temperature(temperature_c)` | Forward to the executing netlist setter and return the testbench. This test's own setting, so it wins over any temperature it would otherwise inherit from its owner, in both directions; `None` here opts the test out rather than resuming inheritance. |
| `set_corner(corner)` | Set the process corner this test runs at when `run()` / `evaluate(rawfile=)` is called without one; returns the testbench. Precedence, highest first: a corner passed to the call (`tb.run(corner='ss')`, or `run_all_tests(corner=...)`, which forwards it), then this setting, then the PDK's `default_corner`. `run_corners()` is an explicit sweep and ignores it. The name is checked at the call against the PDK's declared corners: an unknown or empty name raises `silk.exceptions.UnknownCornerError` (a `ConfigurationError` and a `ValueError`) listing the valid corners, never a fallback to the default; a PDK that declares no corners accepts any non-empty name. Before that check, a name holding a line break or other control character, or exactly `.` or `..`, raises `silk.exceptions.DeckTextError`. A non-string raises `TypeError`; a `Corner` member is stored as its name; `None` clears. Read it back with the `corner` property (`None` when unset). |
| `set_corners(corners)` | Declare a list of corners this test runs at; returns the testbench. With a list declared, `run()` and `run_all_tests` (serial **and** `n_workers>1`: the expansion is a planning step both walks call) run the test once per corner, each in its own `<testbench dir>/<corner>/` folder, and return one `CornerResults` keyed by corner in the declared order instead of a `SpecReport`. A corner passed to the call (`tb.run(corner='ss')`, `run_all_tests(corner='ss')`) runs only that corner of the plan, still as a `CornerResults`. **A failed corner** is kept out of `reports`, listed in `CornerResults.failed` (corner → error), makes `passed` `False`, is counted in `summary()`, and `stat`/`worst` skip it with one counted `MetricWarning`; only when every corner fails does the run raise `SimulationError` carrying the first error (which `run_all_tests` records as that test's `{'error': ...}`). A `ConfigurationError` still stops the run. Each name is checked like `set_corner`'s (`UnknownCornerError`); an empty list raises `UnknownCornerError`; repeated ungrouped coordinates and repeated complete runs are refused during planning before deck emission; a bare string or a non-string entry raises `TypeError`; `None` clears. Read it back with the `corners` property (a tuple, or `None`); assigning to `corners` is `set_corners`. A subclass that overrides `run()` (the `PVTTestbench` example pattern) is never expanded: its override decides what a run is. Declared on the test only — not inherited from the owner. `run_corners()` ignores it and is unchanged. |
| `set_monte_carlo(n, *, variation='mismatch', seed=0, record_draws=False, correlate=None, min_yield=None, samples_per_process=1, output_mode='overwrite')` | Declare a Monte Carlo run for this test; returns the testbench. Arguments are validated by `MonteCarloSettings` (`MonteCarloError` at the call; nothing stored); `n=None` clears; read back with the `monte_carlo` property. With it declared, `run()` and `run_all_tests` (serial and `n_workers>1`) run `n` samples at every corner of `set_corners` (else at the test's corner; an explicit `corner=` restricts to that one) and return one `MonteCarloResults` keyed `(corner, sample)` instead of a `SpecReport`. For SPICE campaigns, each sample's deck loads the PDK manifest's statistical section (`variation='mismatch'`: the corner's `*_mm`-style section; `'process'`: the process section, only at the PDK's process corners; `'both'`: silk's wrapper `<testbench dir>/<corner>/mc_both.lib.spice`) and carries `.options seed=<s>` with `s = derive_sample_seed(seed, index)` on ngspice. On Xyce the deck carries `.SAMPLING useExpr=true` and `.options SAMPLES numsamples=1 sample_type=MC` instead, the same seed goes on the command line as `-randseed <s>` (`XyceRunner(..., extra_args=())`, `SimulatorBackend.sample_args(sample)`), and with `record_draws=True` each sample's normalised draws are read from its `.res` file via `parse_xyce_res` under designer names (`report._mc_draws`). Sample *k* runs in `<testbench dir>/<corner>/mc_<kkkk>/` by default, even when the module writes new folders; `output_mode='new'` inserts a fresh `run-.../` per corner per campaign and keeps its samples, draws and wrapper together; other values raise `ValueError` naming `'overwrite'` and `'new'`; sample artifacts remain after a run; overwrite mode replaces matching sample files on subsequent attempts. Before the first sample, and before any deck is written, it prints `Monte Carlo: C corners × N samples = T simulations` (full point counts for a grid) and, for SPICE, refuses: a PDK with no `statistical:` block (`NotSupportedError`, naming the PDK); a deck with no PDK device, two specs of one name, `'process'`/`'both'` off the PDK's process corners, a corner with no mismatch section (`MonteCarloError`); a model library missing the needed section (`MissingLibrarySectionError`). A failed sample is kept out of `reports` and listed in `failed` (`SampleFailure(error, seed)`); the run raises `SimulationError` only when every sample fails. `run()` simulates one sample at a time; `run_all_tests(n_workers=N)` prepares, simulates and evaluates samples in batches of N (one batch of netlist copies in memory). Each sample's `SpecReport.corner` is the declared corner; its `provenance.seed` is the sample seed and `provenance.seedinfo` reads `monte-carlo variation=<v> section=<section> sample=<k>`. Samples record **no** design-stage `Spec` measurement. Declared on the test only. A subclass that overrides `run()` is not expanded. `record_draws=True` records each sample's drawn model parameters for every PDK transistor whose device type has a manifest `statistical.mismatch.record` entry, under designer names (`'dut.m1'`): on ngspice one `showmod <device> : <params>` line per such device goes into the sample's control block after the analysis and its rawfile `write`, and silk reads the actual values back from the sample's log in the order it requested them; on Xyce silk reads the normalised draws from the `.res` file. They are kept as `MonteCarloResults.draws` (`raw=True` on ngspice, `False` on Xyce; a run holding both raises `MonteCarloError`) and written as `<testbench dir>/<corner>/draws.csv` per corner (`corner, sample, seed, <device>:<param>...`, one row per sample that produced a report). It raises `MonteCarloError` before any deck when no device in the deck has a `record` entry, or on ngspice when combined with `save_device_currents` (batch-write mode has no control block; the netlister refuses the same combination). An ngspice sample whose log lacks its `showmod` blocks fails with `SimulationError`. |
| `run_monte_carlo(n, *, n_workers=None, variation='mismatch', seed=0, record_draws=False, correlate=None, min_yield=None, samples_per_process=1, output_mode='overwrite')` | Run the test as a Monte Carlo campaign once with these arguments and return its `MonteCarloResults`; the declared `set_monte_carlo` setting is left unchanged. Corners and refusals as for `set_monte_carlo`. `samples_per_process > 1` groups consecutive samples in one ngspice process; other backends and unsupported capture/deck combinations refuse batching (see [batching](monte-carlo.md#several-samples-in-one-ngspice-process)). `n_workers` sets the batch size and pool as for `run_corners`. |
| `run(corner=None, *, temperature=None, voltage=None, run_label=None)` | Run the testbench for one corner (`None` → this test's `set_corner` corner, else the PDK's default corner). `temperature` and `voltage` are both deprecated assertions about independently configured TEMP and supply; neither sets anything (see operating conditions above). Returns a `SpecReport` (a `CornerResults` when `set_corners` declared a list; see above). `run_label` selects an optional output-directory label (see below); invalid labels raise `ValueError` before files are written. Raises `ValueError` if `add_module_under_test` was not called first or the corner is unknown. Implemented as `netlist.simulate(corner)` followed by the same extraction tail `evaluate()` runs — one copy of the metric/spec logic. |
| `prepare(corner=None, *, temperature=None, voltage=None, run_label=None, ngspice_version=None)` | Write the single SPICE deck `run()` would write and return its concrete path (`str`); `ngspice_version` forwards a temporary target release to `generate_netlist()`, including through a selected test, and requires the effective ngspice backend, with live inherited simulator, temperature, supply, options and device capture settings. Arguments have `run()` semantics; temperature/voltage assert existing settings. Simulates nothing and leaves metrics, reports, cache and database unchanged. Raises `NotSupportedError` for behavioral benches, Monte Carlo and multi-deck runs (including ngspice probes of undeclared device parameters that need a preflight simulation); select one declared corner with `corner=...`. Retains corner subdirectories and honors output mode (`new` returns a fresh child path per call). Existing rawfiles are preserved. Use `evaluate(rawfile=...)` after external simulation. |
| `validate_simulation_setup()` | Check, without running or changing anything, that this bench's setup passes silk's static admission checks, resolving its effective simulator, TEMP, supply and options as `run()` would. Returns `None` or raises the first refusal the run would raise. Prepares nothing, writes no deck or folder, begins no attempt, consumes no prepared run and leaves placement evidence alone. An unwired bench raises `SimulationConfigurationError`. See [`silk.simulation.validate_simulation_setup`](#validate_simulation_setupmodule). |
| `evaluate(corner=None, *, rawfile=None, temperature=None, voltage=None, log=None, reason=None)` | Extract metrics and check specs without launching. Explicit files always read afresh and carry externally declared provenance. Each of `corner`, `temperature` and `voltage` is a value, `ConditionState.NONE` (declared not to apply, basis `caller-declared`) or omitted (unknown); `UNKNOWN`, `FIXED` and `SWEEP` raise `CharacterizationContextError`. A profile is scoped only when all three are fixed-or-none; anything else is recorded as given and stays unscoped. Memory preserves Silk conditions and rejects relabeling. Omitted corner uses the snapshot for memory; for files it is the `set_corner()` corner, else **unknown** (never the PDK default) and the evidence forms no profile. Failed loads clear live evidence before metrics. See operating conditions above and *Simulate elsewhere, evaluate here* below. |
| `run_corners(corners=None, *, temperature=None, voltage=None, n_workers=None)` | Run the testbench across multiple PVT corners (`None` → the PDK's process corners). `temperature`/`voltage` are forwarded to every corner's run. Every corner is prepared and evaluated in this process, in the order given; `n_workers` only changes how the simulator launches themselves are scheduled — `None`/`1` (the default) is serial, an int runs up to that many at once (capped at the CPU count), `'all'` uses every CPU, and a `behavioral`-backend testbench is always forced to `1` — so serial and parallel sweeps preserve request order and use the same preparation and evaluation logic; launch IDs differ and timestamps are recorded per launch. Every corner gets its own `<testbench dir>/<corner>/` output folder, serial or parallel, regardless of output mode. A corner whose prepare, simulate or evaluate raises does not stop the sweep; once every corner has been attempted, `silk.circuit.spec_report.CornerRunError` is raised naming which corner(s) failed (`.errors`), with the successful corners' reports still on `.results`. Returns a `CornerResults` object (re-exported from `silk`) when every corner succeeded; its `failed` is always `{}`. |
| `sweep_corners` | Property: a copy of this bench's **latest sweep** as `{corner name: rawfile path}`, in the order the corners ran. `run()` and `evaluate()` leave exactly one entry; a completed `run_corners()` installs every corner it ran, each in its own folder and so always distinct (this no longer depends on output mode); a bench that has produced no results has an empty table. These paths are what `Signal.corner()` reads and what each `SpecReport.rawfile_path` records. |
| `set_temperatures(temperatures)` | On Module and ModuleTestbench: declare a nonempty numeric temperature list (finite, strictly above -273.15 C, repeated values are allowed only within a group declared on the testbench; duplicate checks run when planning), or explicitly clear with None. Inherits live through the selected owner hierarchy. Scalar setters override lists; an explicit list clear suppresses inherited lists and falls back to scalar settings. Native simulator `temp` options are equivalent scalar choices. Returns self. |
| `set_supply_voltages(voltages, *, source=None)` | On Module and ModuleTestbench: declare finite supply points with the same source-selection rules as `set_supply_voltage`, or clear with None and no source. No guessed rail names. Same live inheritance and scalar/list precedence as temperatures. Returns self. |
| `set_variable_values(var, values)` | On ModuleTestbench only: declare a nonempty ordered list/tuple for one `Variable`; `None` clears that list. Replacing preserves its axis position. Assignment uses existing bounds and resolution; coordinates use the effective snapped values. Numeric values must be finite real scalars, excluding booleans; categorical values must be declared finite JSON scalar categories (strings, booleans, integers or floats). Unset, complex and container values refuse. Two different Variable objects cannot share a swept name or collide with a `run.*` column. Authored values and source bindings are restored after preparation, execution and evaluation, including failures. A swept Variable used by an overridden supply source, including an expression, is refused. |
| `set_groups(groups)` | On ModuleTestbench only: step lists together within each group; cross groups and ungrouped lists. Members are Variables or `'corner'`, `'temperature'`, `'supply'`. Each list belongs to at most one group, has been declared, and has the same length as its group peers. `None` restores plain crossing. Relationships, ungrouped repeats and repeated complete runs are checked on the whole plan before emission. Grouped `corner=...` selects every complete matching row and refuses an absent corner; ungrouped overrides keep their usual behavior. See [Variable lists and grouped run plans](monte-carlo.md#variable-lists-and-grouped-run-plans). |
| `CornerResults.table()` | Ordered namespaced per-run coordinates, captured condition values/states/bases/sweep bounds, metrics and stable-spec-key checks. Failed runs remain with lowercase `run.status='failed'`, blank observations and UNAVAILABLE checks. Completed reports use PASS/FAIL. Grid keys are `(corner, temperature, supply)`; Variable plans append a fourth component `((name, effective_value), ...)` and add one `run.<name>` column to tables, CSV and report grids. Full-key equality follows Python, so `True`, `1` and `1.0` cannot identify different otherwise identical runs. Group and Variable metadata use test-results JSON v4, including nested/detached grouped PVT-only plans; older readers refuse v4. Legacy plans retain their existing encoding; `worst(...).corner` carries that full key. MC adds sample and seed; see [operating grids](monte-carlo.md#operating-grids). |
| `CornerResults.to_csv(path)` | Serialize the same table, including failed runs, with missing values empty. Namespaces `run.`, `observed.`, `metric.`, `check.` apply to corner-only Monte Carlo tables too (breaking change). |
| `MonteCarloResults.point(key)` | Select one full point, retaining ordered plan, conditions, seeds, failures, draws, declarations and replay identity. `corner(name)` selects the physical process corner and may contain multiple points; measurement/sensitivity require exactly one full point. Unknown points raise MonteCarloError with available choices. |
| `MonteCarloResults.yield_by_point()` | Ordered point-to-spec yield/interval mapping. `passed` requires every point to meet `min_yield`, including corner-only MC. Existing `yield_fraction()` is informational pooled point/sample evaluation yield, not same-device all-condition yield; pooled Wilson bounds do not establish all-PVT manufacturing confidence. |
| `Signal.point(key, *, analysis=None)` | Resolve a deferred probe/expression at one deterministic grid point. All leaves use that exact retained key. Bound signals refuse. MC uses `sample((point, index))`; `corner(name)` refuses new grids even if only one point matches. |
| `CornerResults.stat(metric, how)` | Aggregate one scalar metric (`metric` is a name or a registered `MetricHandle`; so for `worst` and `SpecReport.metrics[...]`; `MetricHandle.name` is the registered name) across the runs `run_corners()` returned. `how` is `'mean'`, `'rms'`, `'std'` (sample, `ddof=1`, like `period_jitter`), `'min'` or `'max'`; always returns a float. A run whose value is a failed-metric string, missing or `nan` makes the result `nan` with one `MetricWarning` (`code='stat_unavailable_corners'`, `details['corners']`); `'std'` with fewer than 2 runs is `nan` with the same warning. An unknown metric or `how` raises `ValueError` listing the valid names. Scalar metrics only. **Results of a declared corner list** (`set_corners`) instead skip runs in `failed` and unavailable values, with one `MetricWarning` (`code='stat_skipped_runs'`, `details['failed']`, `details['unavailable']`, `details['runs']`) giving both counts and the corners, e.g. `1 of 3 runs failed to simulate ('ss')`; `nan` only when no usable value is left. |
| `CornerResults.worst(metric)` | `WorstCase(value, corner)`: the run closest to failing (or furthest past) a spec on that metric, ranked by signed margin to the target (`>`/`>=`: value − target; `<`/`<=`: target − value; `==`: −\\|value − target\\|), smallest margin over all specs on the metric. Raises `ValueError` for an unknown metric or one no spec checks; a failed/missing/`nan` run gives `WorstCase(nan, None)` plus the same `MetricWarning`. Results of a declared corner list skip those runs with the counted warning instead, as `stat` does. |
| `CornerResults(reports, *, failed=None, skip_unavailable=False)` | The constructor. `failed` maps run key → error for runs that produced no report (not keys of `reports`); `passed` is `False` whenever it is non-empty, and `summary()` adds an `N of M runs failed` line and one `[FAILED]` line per run. `skip_unavailable=True` selects the skipping `stat`/`worst` above; `run_corners()` builds the default. |
| `plot(*signals, overlay=False, show=True, corners=None)` | Plot probes and derived Signals of this bench after it has run, delegating to `Module.plot`'s auto-scaled multi-panel/overlay drawing. Each `Signal` argument is resolved exactly once. With no arguments, plots every probe `probe()` has created on this bench; raises `ValueError` if none exist. Raises `UnboundSignalError` before drawing anything if the bench has not run. **Signals from different analyses** do not share an x-axis: each panel is scaled and labelled from its own, and `sharex` is used only when every x really is the same array, so a single-analysis figure is unchanged; `overlay=True` refuses a mixed set naming both axes, since one axis cannot be seconds and hertz at once. A probe (and so `plot()`) shows the bench's last simulated run. **Corners:** after `run_corners([...])`, each signal's panel carries one labelled trace per retained corner with a legend — two probes across three corners is two panels of three traces. `corners=['tt','ss']` selects a subset; an unknown name raises `UnboundSignalError` listing what `sweep_corners` retained. Each corner is read through `Signal.corner()`, so the bench's live results are untouched. After a plain `run()` the drawing is exactly what it was. |
| `results_file` | Property: path of the most recent run or successful `evaluate(rawfile=...)` import; `None` before results are available or after an explicit file load fails. The bridge to the waveform explorer — `emit_view(tb.results_file, ...)`, `silk.view(tb.results_file)`, `report.add_waveform_view(tb)`. |
| `clear_cache()` | Clear the (currently always-empty) report cache. Callable while the cache is disabled; see the `cache` parameter above. |

#### `MonteCarloResults`

Behavioral campaigns draw `Normal` / `Uniform` model parameters without a
simulator or PDK models; at least one active declaration is required.

Workflow and current limits: [monte-carlo.md](monte-carlo.md); worked example `silk/examples/mc_diff_pair_offset.py`.

`silk.circuit.monte_carlo`; re-exported as `silk.MonteCarloResults`. A `CornerResults` subclass keyed `(corner, sample index)` without new axes, or `((corner, temperature, supply), sample index)` for grids, built as `MonteCarloResults(reports, settings=, seeds=, failed=None, structure_hash=None, rerun_hook=None, draws=None, metric_names=None, spec_declarations=None, campaign_source=None, draw_scheme=None)`. `ModuleTestbench.set_monte_carlo` / `run_monte_carlo` produce it, with `structure_hash` (the device under test and testbench) and a `rerun_hook` set.

| Member | Behaviour |
|---|---|
| `MonteCarloSettings(n, variation='mismatch', seed=0, record_draws=False, correlate=(), min_yield=None, samples_per_process=1, output_mode='overwrite')` | Frozen; `output_mode` accepts `'overwrite'` (default) or `'new'`, else raises `ValueError`. Raises `MonteCarloError` for `n` not a positive int, `variation` not in `mismatch`/`process`/`both`, non-int `seed`, non-bool `record_draws`, `min_yield` outside (0, 1]. |
| `derive_sample_seed(base, index)` | `1 + _derive_run_seed(base, index) % (2**31 - 1)`: positive, 31-bit, reproducible. |
| `wilson_interval(k, n, z=1.959963984540054)` | Wilson score interval; `(nan, nan)` for `n == 0`; `MonteCarloError` for `k` outside `0..n`. |
| `corner(name)` | The samples of one physical process corner (possibly several grid points), retaining the campaign metric/spec inventory even when every selected sample failed. |
| `corners` | Property: the corners that ran, as a list in run order (the declared `set_corners` order), including a corner whose every sample failed. Keys without a recorded seed fall back to sorted order. |
| `stat(metric, how)`, `percentile(metric, q)` | Skip failed samples and `nan`/missing values with one `MetricWarning` (`code='sample_skipped'`, `details` = failed/unavailable/total/used counts). `nan` when nothing is usable (including zero observations) or `std` has fewer than 2 samples, with exactly one warning. Unknown names and invalid requests raise `ValueError`. |
| `worst(metric)` | `SampleWorstCase(value, corner, sample)`, ranked as `CornerResults.worst`. No usable observations: `(nan, None, None)` with one `MetricWarning`. A metric without a declared spec raises `ValueError`. |
| `yield_fraction()` | `{spec name: (fraction, (lo, hi))}` for every declared spec; failed samples count as failing. An all-failed corner has zero yield; zero observations gives a `nan` fraction and interval. |
| `passed` | Every sample passes (no failures); with `min_yield`, every spec's yield at every full point must meet it, per corner when no new axes are declared. Pooled yield is informational. |
| `table()`, `to_csv(path)` | One namespaced row per planned sample including failed ones (`run.error` column), in retained plan order. Metric columns use `metric.` and checks use `check.<stable spec key>`, including corner-only campaigns. Even an empty CSV retains declared columns. Captured observations distinguish fixed/unknown/none/sweep states. |
| `plot_histogram(metric, ax=None)` | Histogram with each spec limit as a vertical line. |
| `summary()` | Corner-only run/failed/used lines plus per-corner yields and informational aggregate yield. Grids give per-point verdicts, planned runs, completed PASS/FAIL reports, and runs that failed and produced no report and pooled scalar ranges/statistics with a full worst-point key. |
| `draws` | The run's drawn model parameters, a `silk.circuit.monte_carlo_draws.DrawTable` (`columns` = `(designer device name, param)` pairs such as `('dut.m1', 'vth0')`; `values` = sample key -> array aligned to `columns`; `raw=True` for ngspice's actual values, `False` for Xyce's normalised draws; `skipped_types` = device types the manifest records nothing for), or `None` when the run did not set `record_draws=True`. `corner(name)` keeps that corner's rows. |
| `sensitivity(metric, *, by='parameter')` | Rank *metric* against the drawn parameters through `rank_sensitivity`: each column is standardised across the usable samples (failed and `nan` samples skipped with the counted `MetricWarning`), the metric is fitted by least squares, and each row is a `SensitivityRow(name, effect_per_sigma, share, correlation)` sorted by share, with the fit's `r_squared`; `by='device'` sums each device's parameters. More varying parameters than samples falls back to correlation only (no shares, one `MetricWarning`). `ValueError` naming `record_draws=True` when the run recorded no draws, and `ValueError` requiring `mc.point(point).sensitivity(...)` when more than one planned full point is selected (use `mc.corner(name)` for corner-only runs) (pooling operating points confounds condition shifts with sample variation); `NotSupportedError` when the run declared `correlate=` (ranking those quantities is not implemented). |
| `rerun(key)` | Calls `rerun_hook(key, seed)`; `MonteCarloError` when there is no hook. From a run the hook re-prepares that sample's deck with its recorded seed in its own `mc_NNNN/` folder (overwriting it), simulates and returns its `SpecReport`; it raises `MonteCarloError` naming the change when the circuit's structure hash differs from the one recorded at run time. Replay also checks sources, analyses, conditions, simulator settings and statistical declarations; it evaluates the currently registered metrics/specs. See [replay inputs](monte-carlo.md#replay-inputs). |
| `Signal.sample(key, *, analysis=None)` | A probe read from one sample of the bench's latest Monte Carlo run, the way `Signal.corner()` reads one corner: `vout.sample(('ss', 17)).y`. `UnboundSignalError` when the latest run was not Monte Carlo or did not retain *key* (a failed sample retains no file); `TypeError` for a key that is not a `(corner, index)` pair. After a Monte Carlo run `vout.corner('ss')` raises `UnboundSignalError` naming `.sample(...)`, and `sweep_corners` holds one `'<corner>/mc_NNNN'` entry per retained corner-only sample, or an exact `(point, index)` key for a grid sample. |

The runner snapshots registered metric names (including structured output fields)
and spec definitions. Hand-built results can supply `metric_names` and
`spec_declarations` explicitly; the latter is an iterable of mappings with
`name`, `metric`, `op` and `target`. Omitted inventories are inferred once from
the root's supplied reports, so a hand-built root with no successful reports
needs explicit declarations. Conflicting definitions for one spec name raise
`ValueError` naming both definitions. See [failed corners](monte-carlo.md#failed-corners-and-declared-metrics).

**`silk.circuit.monte_carlo_draws.Sensitivity` methods:** the object `sensitivity()` returns.

| Method | Description |
|---|---|
| `summary(top=3, *, scale=1.0, unit='')` | The returned `Sensitivity`'s first `top` rows as text. `method='least_squares'`: one line per row with effect per sigma and share, then the fit's R squared; `'correlation'` (fallback): one line per row with its correlation, then a line saying shares need more samples than recorded parameters. `scale`/`unit` restate the effect in other units (`scale=1e3, unit=' mV'` for volts); a row with no fitted effect or share prints `n/a`. `ValueError` when `top` is not a positive int. |

`measurement(metric, how='std', *, corner=None)` returns explicit aggregate
evidence for one selected campaign corner. It uses the existing statistical
reduction and warnings, raises `MonteCarloError` for ambiguous corners or
missing captured source, and never records or saves evidence implicitly.
Hand-built campaigns may provide `campaign_source` and captured `draw_scheme`;
unknown metadata stays unknown.

#### Testbench output directories and run labels

A root test, a singly placed child, and aliases sharing one simulation retain
the authored `test_<module>~<test>/` directory. When one hierarchy invocation
retains distinct simulations competing for that path, subsequent simulations
get a distinct test directory whose name includes the encoded owner. Every deck stays beside its rawfile and log.
Separate roots can reuse the same disk spelling in overwrite mode: selected
readback retains loaded run data, so an overwrite cannot change another owner's
waveform. Use new mode or copy artifacts when files must survive independently.


Each test's folder is `test_<module>~<test>/`, for example
`test_comparator_sa~decision_time_od0005mV/`; underscores stay literal. Names
use percent escaping for unsafe bytes, `%` and `~`, so the first `~` after
`test_<module>` is always the separator and `(a_b, c)` and `(a, b_c)` cannot
alias (`test_a_b~c` vs `test_a~b_c`). The deck is `test_<module>~<test>.cir`. An unnamed module uses `%N` to distinguish it from a module named `None`. Run labels use the reserved `~run~` delimiter. Long encoded components are
bounded with a digest of the full name. Distinct owner folders are an execution
detail; obtain concrete paths from `prepare()`, `results_file` or the report.


Pass `tb.run(run_label='baseline')`, `mod.run_test('level',
run_label='baseline')`, or `parent.run_all_tests(run_label='baseline')` to
retain selected runs separately. Labels contain 1–64 ASCII characters:
start with a letter or digit, then use letters, digits, `_`, `.` or `-`.
Invalid labels raise `ValueError` before simulation or output replacement.
Labels never change test names, recursive report keys, structure fingerprints,
characterization profiles, declared parameters, spec targets or database
storage policy.

Choose artifact retention once on the circuit:

```python
circuit.set_output_options(mode='overwrite')  # explicitly replace at stable paths
circuit.set_output_options(mode='new')        # default: retain successive attempts
```

The setter only validates and configures subsequent runs. It does not generate
a deck, allocate a directory, launch or delete results. Invalid values leave
the previous setting and artifacts intact. There is no global policy. Ordinary run methods use this module setting;
Monte Carlo has its own `output_mode` option, defaulting to overwrite.

In **overwrite** mode, the same instance/test/label reuses its deterministic
directory and basenames. In **new** mode, preparation exclusively allocates a
persistent `run-<unique>/` child below that same base **before deck emission**.
The deck, results, applicable companions and log use that one location. These
children survive success, failure and process exit. Their generated names are
not an API; save actual successful paths instead of reconstructing them.

Each call selects its own label: omitting it returns to the unlabeled base.
Repeated new-mode runs are siblings. Switching `overwrite → new → new →
overwrite` returns to the original deterministic location; switching labels
does not compound suffixes. `dir_folder` and `netlistname` remain the configured
folder and deck file name, separate from the prepared run's concrete paths.

Embedded testbenches consult their **live owning circuit** at run time, even
when constructed or copied before the setting changed. `run_all_tests` uses
the invoking root's mode throughout that invocation, including recursive live
instances and legacy registered submodules, without changing child settings.
A later independent child run uses its own mode. This applies to direct
`ModuleTestbench.run`, `run_test` (with or without `store=True`),
`run_all_tests`, each actual `run_corners` launch, and ordinary silk testbenches
used by `Optimizer.evaluate`/`sweep`. Custom external test objects have their
own policies. Standalone `Module.simulate` uses that Module's setting.

Fault workers use the serialized circuit setting and prepare once before
emitting the faulted deck. Campaign run IDs, seeds, corners, control files,
resume and error behavior are unchanged. `manifest.json` and `outcome.json`
stay in the campaign run directory; a new-mode deck and simulator artifacts
live in its unique child. `archive_waveforms=False` still discards that
attempt's waveform, including in new mode.

For the manual `generate_netlist()` → `run_simulator()` flow, generation
selects the location and launch consumes that exact deck. In new mode, call
`generate_netlist()` again before **each** launch, including after a failed
attempt. Reusing a consumed preparation, launching without preparation, or
changing its configured paths, simulator or mode raises `SimulationError`
with regeneration instructions before launch or cleanup. For tooling that
needs the emitted path, use the return value of `module.generate_netlist()`;
`module.netlistname` remains the configured file name. Relative folders produce
relative returned paths and absolute folders produce absolute paths.
Direct backend/runner callers supplying explicit paths retain explicit-path
semantics; they do not inherit a circuit setting.

**Changing deck-affecting configuration between the two steps also raises**
(in both output modes), naming the first changed deck line and instructing you
to regenerate. Deck-affecting means anything the netlister renders: sources,
elements, passives, submodules, analyses, custom lines, the simulator and its
options, the save list, `save_device_currents`, the terminal-current method and
the corner passed to `generate_netlist()`. Launch re-emits the deck from the
live configuration and compares it with what was prepared, so the set is
complete by construction rather than enumerated; configuration that never
reaches the deck (`debug`, `timeout`) never refuses. Overwrite mode retains the
prepared deck after a filename change; new mode requires regeneration when
configured paths change. A
`generate_netlist()` call that raises also refuses the next launch rather than
letting the earlier deck run. Decks with fault markers applied are not
reproducible from module state and are covered by the on-disk deck check alone;
an externally supplied deck has no prepared record and is never second-guessed.

Every ngspice/Xyce launch uses **exact-file prelaunch cleanup**, including
manual `generate_netlist()` → `run_simulator()` flows, direct backend/runner
calls and fault workers. After validation and deck generation succeed, the
runner removes only the concrete prior result files that this attempt will
write/check/read: the selected rawfile, applicable Xyce noise CSV and known
generated diagnostic logs (ngspice's selected log). It preserves the generated
deck, user files, DB files, sibling-run outputs and unrelated rawfiles/logs.
Cleanup unlinks **exact regular files only, never following a link** at an
output target. Symlinked parent directories are allowed. Unsafe cleanup targets
(directories, symlinks, special files or aliases of the input deck) and removal
failures raise an actionable `SimulationError` before launch. There is no
globbing or directory clearing.

**An overwrite attempt deliberately loses the preceding same-location result,
even when the replacement fails.** Exit 0 without fresh usable output raises
`SimulationAborted`; timeout, launch, exit and malformed-result failures remain
typed simulation errors. Failed actual attempts clear current parsed waveforms,
loaded-result flags, `results_file`/`dataFile` reload fallbacks and current
characterization/corner evidence. `load_signal()`, `read_raw_file()` and ordinary
`evaluate()` cannot silently return the preceding waveform. Newly generated
partial files and diagnostics may remain for inspection, but are not accepted
automatically as a successful result.

A failed test attempt stops before metric extraction, new Spec measurements
or `run_test(store=True)` persistence. Previously returned scalar reports,
existing measurement history and committed DB rows remain unchanged.

Invalid labels, corners, output/context configuration and failed deck generation
do not trigger cleanup. Pure path resolution, explicit file evaluation/readback
and behavioral simulation also perform no launch cleanup (the report cache is
disabled in 0.2.0 — so a cache hit is no longer one of these cases).
Subcircuit text generation and mere path reads also allocate no run directory.
Behavioral runs write a rawfile, then hand readers the in-memory result
without rereading that file. Unique directories isolate files; they do not make
concurrent execution on one mutable Module safe. File retention has no automatic
cleanup schedule or database waveform archive.

Direct reports follow the bench's current waveform; selected-placement reports
retain their run reader, including root entries executed by `run_all_tests()`.
Save `report.rawfile_path` for later disk access. Scalar
metrics remain the values measured for each report. For a configured testbench
with a DC source named `supply`, an `out` node and a final-voltage metric named
`voltage` (the regression uses a 1 kΩ resistive load):

```python
tb.module.set_output_options(mode='new')
baseline = tb.run()                          # supply initially 1 V
baseline_path = tb.results_file

tb.netlist.sig_sources_dict['supply'].dc = 2.0
higher_bias = tb.run()                       # another persistent child
higher_bias_path = tb.results_file

# Explicit evaluation reads the first saved file from disk again.
first_again = tb.evaluate(rawfile=baseline_path)
assert abs(first_again.metrics['voltage'] - 1.0) < 1e-9
assert abs(higher_bias.metrics['voltage'] - 2.0) < 1e-9
```

Here `evaluate(rawfile=...)` also makes the first file current on `tb`;
`higher_bias_path` still names the retained second file. Calling `evaluate()`
without a file measures the currently loaded samples without rereading disk.

For a complete runnable walkthrough, see
[`silk/examples/output_modes_example.py`](../silk/examples/output_modes_example.py)
and the [curated example index](agent-guide.md#7-curated-example-index).
Run from the repository root:

```bash
MPLBACKEND=Agg .venv/bin/python -m silk.examples.output_modes_example
```

Its ideal resistive load owns a registered voltage test. The explicitly chosen overwrite
runs at 1 V and 2 V reuse one actual result path; a fresh disk read through the
first saved path returns **2 V**. Setting the owner to `mode='new'` after test
construction retains **3 V** and **4 V** in separate directories, both reread
after both launches. Returning to overwrite puts **5 V** at the original base
while the two retained rawfiles remain readable and byte-identical. The script
prints each actual result path and the files beside it (deck, rawfile and
ngspice log), without guessing unique directory names.

The example uses `read_rawfile(path)` and
`NgspiceSignal.from_rawfile(raw, 'out')` for fresh disk reads independent of
live reports. It needs ngspice (on `PATH` or `SILK_NGSPICE_BIN`) and silk's
Python dependencies. The sky130 catalog ships with silk; this ideal-only deck
omits PDK model includes, so **no installed PDK is required**. Default outputs
are rooted under `silk-output/output_modes/` in the directory you run it from
(or under `$SILK_EXAMPLES_OUT`); `run_example(output_root=...)` selects an
isolated location. Repeating it keeps
earlier new-mode runs. Its docstring explains label grouping and the
exact-file overwrite-on-attempt/no-data failure contract above.

The report cache (`cache=True`) is disabled: passing it warns, and every
`run()` simulates using the configured output mode. See the constructor's
`cache` parameter above.

#### Simulate elsewhere, evaluate here

`run()` is two steps — simulate, then evaluate — and the second is public. When
the simulator lives on another machine (a remote SPICE farm, a licensed
simulator, a CI runner), generate the deck here, run it there, and bring the
rawfile back. For Xyce noise metrics, copy `noise.csv` beside the imported
rawfile as well:

```python
deck_path = tb.prepare(corner='tt')  # includes live inherited run settings
# ... copy the deck out, run `ngspice -b deck.cir -r rawfile.raw` (or Xyce) elsewhere ...
report = tb.evaluate('tt', rawfile='/path/to/rawfile.raw')   # no simulator launched
```

If the deck has a `.control` block, ngspice writes the rawfile to the path in
its `write` line, and `-r` does not replace that path. Edit the line in the
copied deck before running it; see [known issues](known_issues.md#running-a-prepared-ngspice-deck-in-another-directory).

`report` has the same metrics, verdicts and test identity as evaluation after
`run()`, but its conditions have external origin and caller-declared (or unknown)
PVT. Usable real metrics record `Measurement(source='spice', corner='tt')`
observations on attached specs; failed or unavailable extraction records
unsuccessful attempts for `reconcile_specs()` / `store_report()`.
`tb.results_file` now points at the rawfile. Create
`circuit_report = generate_report(tb.module, spec_reports=[report])`, then call
`circuit_report.add_waveform_view(tb)` to attach a waveform view. The corner you pass is metadata — nothing in a rawfile says which
corner it was run at, so pass the one you actually simulated.

An explicit `evaluate(rawfile=path)` always reads the current file from disk,
even when an external simulator overwrites the same filename between calls.
No cache-clearing step or new flag is needed. Omitting `rawfile` evaluates the
already-loaded in-memory results without a disk read; `run()` and `run_test()`
use this path after simulation, so they do not load the output twice.
An explicit file uses this call's characterization arguments (unscoped when
temperature and voltage are both omitted). With no file and no new PVT arguments, the loaded context
is retained.

If a file load fails, the exception propagates before metric extraction or
new measurements: missing/unreadable files raise `OSError` subclasses,
malformed files raise `RawfileError` (a `SimulationError`), and no-data output
raises `SimulationAborted`. The netlist's parsed data, `results_file`, and
characterization context and operating conditions are cleared; `dataFile`
keeps the attempted path.
`evaluate()` without a file then raises until results are successfully loaded.
Prior report values, spec measurements, and stored database evidence remain
unchanged. Unselected direct bench reports reference the live netlist. Reports
from placement selections retain a run reader, including root entries executed
by `run_all_tests()`. Invalid characterization arguments are
rejected before loading and leave existing results intact.

An aborted run can leave partial data or no analysis data. Certified silk-generated ngspice/Xyce runs check
each transient/DC/AC/noise endpoint against its generation-bound request (see
[analysis completion](known_issues.md)). Sub-decade ngspice DEC AC sweeps use the
version-specific grid from the simulator YAML (stretched through 42, plain
steps from 43); only unknown versions retain no endpoint verdict. Other
certified analyses in the same run remain checked. Unverified external/custom decks and standalone loads
can only check data presence. ngspice's write-with-no-analysis fallback leaves the synthetic
`constants` plot, which reports `npoints = 1`. `evaluate()` raises
`SimulationAborted` (`data_rows=0`) on such a file instead of measuring it, and
no `Measurement` is recorded on any `Spec`. The gate is in the loader, so
`Module.load_file()` + `Module.read_raw_file()` raise the same way. Pass
`log=<the remote run's log>` to have the abort reason scraped from it the way
`run()` scrapes ngspice's `output.log`, or `reason=<already scraped>` directly;
`load_file(path, simulatorType)` names the simulator on the exception.

**`SpecReport`** (`silk.circuit.spec_report`) — returned by `run()` and
`evaluate()`, or loaded as detached evidence with `from_dict()` / `load_json()`:

Ordinary evaluations retain detached original spec observations through JSON.
`store_report()` can persist an earlier report after later evaluations, using
its original precise observation time and attempt attribution. Historical/manual
reports without observations retain unknown time/correlation metadata (SQL NULL
for the spec observation timestamp), even when a live measurement has equal
values. Existing declaration, provenance and report consistency checks apply.

| Attribute / method | Description |
|---|---|
| `passed` | `True` when every checked spec passed (or no specs were declared). |
| `results` | `list[SpecResult]` — one per spec: `name`, `metric_name`, `op`, `target`, `value` (the measured value, possibly complex or unavailable, or extraction-error text; complex, missing and error values fail the check), `passed`, `spec_key`, `owner_path`. |
| `metrics` | `dict` of every extracted metric value, including ones no spec checks. |
| `to_dict()` | Fresh strict-JSON wire record (`silk.spec-report`, version 1) of complete evidence, including unspecced metrics. Nonfinite/complex values are tagged once. See [standalone records](reports.md#complete-standalone-test-records). |
| `save_json(path)` | Save the standalone wire record; validate before opening the destination. CircuitReport uses a separate wire schema. |
| `from_dict(payload)` | Classmethod restoring a detached report from wire data; validates fields and stored verdicts without rerunning checks. |
| `load_json(path)` | Classmethod reading standalone evidence offline. Malformed data raises `ValueError` naming file and field; filesystem errors propagate. |
| `report_key` | Snapshotted runner key, matching the selected `TestEntry.report_key`; null on manual reports without execution identity. |
| `metric_types` | Every metric output maps to its catalog type or null for custom/derived/unknown outputs; expanded fields retain the parent type. |
| `metric_errors` | Structured extraction exceptions by output: exact `message`, `exception_type`, and `available_signals` (list or null). No error-string parsing. |
| `metric_warnings` | Ordered warning records by exact registered output, preserving separate attribution for two metrics of the same type. |
| `summary()` | Multi-line human-readable pass/fail listing. |
| `sim_results` | Live reports only (not restored by standalone loading). A testbench `Module` or a retained result reader for a selected placement — `report.sim_results.load_signal('vout')` reads any signal of the run (testbench nets by name, DUT-internal nodes as `xdut.<net>`). |
| `test_name`, `module_name`, `owner_path`, `parameters`, `characterization_context` | Identity of the run: which test on which module, its declared parameters, and the PVT context (`None` when unscoped). |
| `rawfile_path` | `str` or `None` — the rawfile these metrics were extracted from (`Module.results_file` at extraction time). Behavioral runs also write a rawfile; `None` means no rawfile path was recorded. After `run_corners()` this is how a script finds the waveforms behind a per-corner number, use this saved path rather than assuming the bench's current results still select that corner. |
| `corner` | `str` — the run's **label**: the corner that was requested (the PDK's `default_corner` when none was passed). It is **not evidence** of what ran: a PDK-free deck simulates no corner, and external evidence with no declared corner has `None`. Read `operating_conditions.corner_kind` / `.scalar('corner')` for the evidence, and never fill a `None` from this label ([consumer contract](#operating-conditions-none-unknown-and-the-consumer-contract)). `None` on a hand-built `SpecReport` and on undeclared external evidence. `store_report()` files an unscoped report under the evidence scalar from the snapshot (see [database.md](database.md)). |
| `storage` | `dict` or `None` — the receipt of the most recent `Module.run_test(..., store=True)` that persisted this report: `store_report()`'s `metrics_stored`/`specs_stored`/`skipped` plus `cached` (always `False` in 0.2.0 — the report cache is disabled). `None` otherwise — including after a `ReportStorageError` and on a report stored only through an explicit `store_report(report)` call. |

---

### `Variable`

A design variable with a current value and optional bounds. Assign through
`var.value`; out-of-bounds numeric assignments raise `ValueError` without changing
it. Use `var.clip(value)` for explicit clipping.

```
Variable(name, value=None, *, bounds=None, var_type='continuous',
         scale='linear', resolution=None,
         enforce_resolution=True, categories=None)
```

Defined in `silk.circuit.variable`; also `silk.Variable`.

`var_type` accepts exactly `'continuous'`, `'integer'`, or `'categorical'`;
`scale` accepts exactly `'linear'` or `'log'`. Both validate on construction
and assignment; misspellings raise `ValueError` listing the choices.

Sampling takes a keyword-only `seed=0`: the same integer seed repeats the
points. Use another seed or `seed=None` (fresh OS entropy) for fresh draws.
Booleans and non-integers raise `TypeError`; negative integers raise `ValueError`.
The generator is local to the call, leaving global random states unchanged.
Categorical `random` and `lhs` draws use this seed. Categorical `grid` and
`halton` are independent of the seed. For positive `n`, categorical `grid`
enumerates every category once in declared order, even when `n` is smaller
or larger than the category count; `n=0` returns no samples. Categorical
`halton` maps its base-2 sequence to indices with `floor(u * category_count)`.
It has no finite-prefix coverage guarantee: five draws from five categories
visit indices `[2, 1, 3, 0, 3]`. Use `grid` for exhaustive category coverage.

`sample(n, method='random', *, seed=0)` accepts `'random'`, `'lhs'`, `'halton'`, and `'grid'`.
Every method honours `scale`, including resolution-constrained sampling.
Log draws are made in log coordinates, then rounded to integers or projected
onto the resolution grid. Log sampling requires finite `0 < lo <= hi` bounds;
incomplete PDK bounds may be constructed but must be completed before sampling.
Log scale cannot be combined with categorical variables.

Resolution grids anchor at the lower bound. For bounds `(1, 10)` and
`resolution=2`, the realizable values are `[1, 3, 5, 7, 9]`. Linear grid
sampling selects at most `n` distinct resolution values; log grid sampling
keeps `n` draws, so projection can produce duplicates. Setting
`enforce_resolution=False` bypasses projection. `VariableSpace.apply`,
`to_bounds`, `to_list`, and `from_list` always use physical units.

Integer variables require a whole-number `resolution` when one is set and a
whole-number lower bound when both are set. Construction and assignment reject
incompatible grids with `ValueError`, naming the variable, resolution, bounds
and remedy. Use an integer resolution and lower bound, or
`var_type='continuous'`. This rule applies to both scales, even with
`enforce_resolution=False`. An open lower bound is allowed until sampling,
when it must be supplied. Integral floats such as `2.0` are accepted; sampled
integer values are Python integers.

`resolution` sets the assignment grid; `sample(n, 'grid')` chooses a count
across the full domain. Sampling methods belong in `sample`, while the
variable's `scale` controls linear or logarithmic spacing.

Linear resolution and integer grids return at most `n` distinct domain values,
preserving the lower anchor and selecting across the full range. `n=0` returns
none, `n=1` selects the first value. Resolution grids omit genuinely off-grid upper bounds. Only
one terminal index may normalize to the exact upper bound within four ULPs
of the arithmetic operands, preserving distinct neighbours and the lower
anchor; integer endpoints remain integral. Continuous count grids retain the
requested count, including repeats for singleton bounds, and pin both endpoints
exactly for `n>1`. Numeric samples and their applied values stay within bounds, including decimal cleanup and fractional integer bounds.
Finite integer bounds containing no integer raise `ValueError` at construction
or property assignment. Open bounds remain allowed at construction.

Shortened linear resolution and integer grids emit one ordinary `UserWarning`
per variable per call, pointing to the public caller and respecting standard
filters, including errors.
Warnings appear in terminals, editors or notebooks; reports do not persist them.
Random, LHS and Halton sampling return `n` draws.

#### Arithmetic — derived expressions

`Variable` overloads `+`, `-`, `*`, `/`, `//`, and unary `-` to
return lazy expression nodes. Expressions re-evaluate each time their
`.value` is read, so updating a leaf variable's `.value` propagates
through every expression that references it on the next
`generate_netlist()`:

```python
m_diode  = Variable('m_diode', value=4, bounds=(2, 16),
                           var_type='integer')

with fcamp:
    pdk.nmos1p8('mn_diode', d='bias', g='bias', s='0', b='0', w=1, l=0.15, m=m_diode)
    pdk.nmos1p8('mn_mirr', d='out', g='bias', s='0', b='0', w=1, l=0.15, m=2 * m_diode)   # always 2× the diode
    pdk.nmos1p8('mn_half', d='half', g='bias', s='0', b='0', w=1, l=0.15, m=m_diode // 2)  # integer half
```

Expressions are **derived**, not independent — they cannot be added to
a `VariableSpace`. Put the *leaf* `Variable` in the space; every
expression built from it follows automatically. This lets you encode
matching ratios (mirror ratios, diff-pair equality, cascode sizing)
as invariants that any optimizer / sweeper respects by construction.

Supported: `var * x`, `x * var`, `var + x`, `var - x`, `x - var`,
`var / x`, `x / var`, `var // x`, `-var`, and arbitrary chaining
(`(a + b) * 2 - c / 3`). Operand `x` may be a number, another
`Variable`, or another expression.

---


---

### `PortDir`

Port **direction** enum — describes signal flow. Inherits from `str` so values compare equal to plain strings.

```python
from silk import PortDir

PortDir.IN      # 'input'  — driven externally; read by this block
PortDir.OUT     # 'output' — driven by this block
PortDir.INOUT   # 'inout'  — bidirectional (warns during behavioral sim)
PortDir.POWER   # 'power'  — supply rail shorthand (backward compat)
PortDir.GROUND  # 'ground' — reference rail shorthand (backward compat)
```

---

### `PortRole`

Port **semantic role** enum — describes what kind of signal the port carries. Optional; annotate only where it adds meaning. Roles enable semantic checks; name-based inference requires an explicit `infer_port_roles(module, ruleset)` call.

```python
from silk import PortRole

PortRole.ANALOG   # 'analog'  — continuous-valued signal
PortRole.DIGITAL  # 'digital' — logic-level signal (enable, data, flag)
PortRole.CLOCK    # 'clock'   — timing reference
PortRole.POWER    # 'power'   — supply rail
PortRole.GROUND   # 'ground'  — ground reference
```

Prefer the convenience factories on `Port` — they cover the common cases, read naturally, and let user code skip the `PortDir` / `PortRole` imports entirely:

```python
comp.define_module_ports({
    'vp':   Port.analog_input(),             # domain optional
    'vm':   Port.analog_input(),
    'clk':  Port.clock_input(),
    'en':   Port.digital_input(),
    'vdd':  Port.supply(avdd),               # direction defaults to IN
    'vss':  Port.ground(),                   # never name a ground port 'gnd'
    'vout': Port.digital_output(),
})
```

**Available factories:**

| Factory | Equivalent |
|---|---|
| `Port.analog_input(domain=None)`    | `Port(dir=PortDir.IN,  role=PortRole.ANALOG,  domain=...)`  |
| `Port.analog_output(domain=None)`   | `Port(dir=PortDir.OUT, role=PortRole.ANALOG,  domain=...)`  |
| `Port.digital_input(domain=None)`   | `Port(dir=PortDir.IN,  role=PortRole.DIGITAL, domain=...)`  |
| `Port.digital_output(domain=None)`  | `Port(dir=PortDir.OUT, role=PortRole.DIGITAL, domain=...)`  |
| `Port.clock_input(domain=None)`     | `Port(dir=PortDir.IN,  role=PortRole.CLOCK,   domain=...)`  |
| `Port.clock_output(domain=None)`    | `Port(dir=PortDir.OUT, role=PortRole.CLOCK,   domain=...)`  |
| `Port.supply(domain=None, direction=PortDir.IN)` | `Port(dir=<direction>, role=PortRole.POWER,  domain=...)` |
| `Port.ground(domain=None, direction=PortDir.IN)` | `Port(dir=<direction>, role=PortRole.GROUND, domain=...)` |

The six signal factories also accept keyword-only `bits=None, naming='bracketed'`;
supply and ground ports require scalar rails.

A ground port must not be *named* `gnd` (any case): ngspice folds that port name onto the global SPICE node `0` and silently discards whatever the parent wired to it, while Xyce honours it — lint reports `STRUCT-012`, and netlist generation and static setup refuse this interface, so name it `vss` (verified not folded). This guard applies to formal port names. Internal net handling depends on the backend; use `'0'` for the simulator ground node.

`supply()` and `ground()` default to `IN` because rails are almost always inputs; override only for bias/reference generators that emit a rail. The `direction` kwarg accepts either a `PortDir` enum or a short string (`'in'`, `'out'`, `'inout'`) or the enum value (`'input'`, `'output'`) so override sites don't need to import `PortDir`.

For ports that don't match one of the shortcuts (e.g. bidirectional analog), fall back to the canonical constructor — all three forms (`Port(...)`, bare `PortDir`, `(PortDir, PortRole)` tuple) remain supported:

```python
comp.define_module_ports({
    'bias_out': Port(dir=PortDir.OUT, role=PortRole.POWER, domain=avdd),
})
```

---

### Port-role inference

Fill port roles **from their names** instead of hand-annotating each one, using a
user-defined, replaceable ruleset — so a large block is cheap to bring up to the
`require_annotation_coverage` gate. silk source holds no patterns; they live in a
ruleset (an opt-in shipped default, or a company's own). Full guide:
[authoring-port-role-rules.md](authoring-port-role-rules.md).

```python
from silk import RoleRuleset, infer_port_roles

rs = RoleRuleset.default()                    # silk's shipped opt-in default
report = infer_port_roles(module, rs)         # fills unset roles on module.pin_list
```

**`RoleRuleset`** — an ordered set of name→role rules. Matching is glob by default
(`*vdd*`), case-insensitive, most-specific match wins; a per-rule `regex:` escape
hatch is available in YAML.

| Member | Returns | Notes |
|---|---|---|
| `RoleRuleset.default()` | `RoleRuleset` | Load silk's shipped default YAML (opt-in). Fresh instance per call. |
| `RoleRuleset.from_yaml(path)` | `RoleRuleset` | Load a ruleset file; raises `RoleRulesetError` (a `ValueError`) on any malformed rule. |
| `RoleRuleset().add(pattern, role)` | `RoleRuleset` | Add a glob rule; chainable. |
| `rs.classify(name)` | `Optional[PortRole]` | Most-specific matching role, or `None`. |
| `len(rs)` | `int` | Rule count. |

**`infer_port_roles(module, ruleset)`** → `InferReport`. Fills each *unset* role on
`module.pin_list` (the module's **own** ports) as a `PortRole` enum; **never**
overwrites an explicitly-set role; unmatched ports stay `None` so
`require_annotation_coverage` still flags them. `InferReport` has
`.filled: list[tuple[str, PortRole]]` and `.unmatched: list[str]`.

Call after `define_module_ports`. Inference works before or after placement:
parents refresh port metadata and `Net.pins` from the live definition. The
`role_inferred` provenance remains attached to each refreshed binding.

Every role this call fills is marked `pin.role_inferred = True` (declared,
factory and direction-implied roles are `False`), and the mark is snapshotted
into `PortBinding.role_inferred` at wiring. An inferred role is a guess:
INT-004 NO_DRIVER does not let an inferred POWER role silence an undriven
input (see the rule table), while a declared one keeps its skip.

Validate/dry-run a ruleset from the shell:
`python -m silk.circuit.rolecheck --file rules.yaml [--names vdd clk …]` (see
[cli.md](cli.md)).

---

### `Net` / `PortBinding`

First-class representation of electrical nets. Every entry in
`module.net_dict` is a `Net` object (not a raw parameter list). Nets are
created automatically by `add_module`, `add_passive`, `add_source`, and
`add_net`, and carry a live record of which pins are connected to them.

```python
from silk import Net, PortBinding

vout = ckt.net_dict['vout']          # a Net instance
assert isinstance(vout, Net)
```

**`Net` attributes:**

| Attribute | Type | Description |
|---|---|---|
| `name` | str | Net name (same key as in `net_dict`) |
| `module` | Module | Back-reference to the parent `Module` |
| `parameters` | list | Legacy per-net parameter list from `add_net(..., parameters=...)`. Rarely used; preserved for backward compatibility. |
| `pins` | list[PortBinding] | Every pin currently bound to this net. Maintained automatically whenever elements are added, connected, or rewired. |
| `fanout` | int (property) | Convenience alias for `len(pins)`. |
| `voltage` | np.ndarray \| None | Stored directly on the `Net` object by `simulate_behavioral()`. Returns `None` before any behavioral simulation has run. |

**`PortBinding` fields** (dataclass):

| Field | Type | Description |
|---|---|---|
| `net` | str | Name of the net this binding belongs to |
| `instance` | str | Name of the owner (device, passive, source, or submodule instance) |
| `port` | str | Pin/terminal name on the owner (e.g. `'d'`, `'pos'`, `'positive'`) |
| `direction` | str \| None | `PortDir` value as string (submodule ports and directed source terminals) |
| `role` | str \| None | `PortRole` value as string (submodule ports only) |
| `role_inferred` | bool | `True` when `role` was filled by `infer_port_roles` (a name-pattern guess) rather than declared; snapshotted from `ModulePin.role_inferred` at wiring. `False` for every other binding |
| `domain` | VoltageDomain \| None | Voltage domain attached to the pin (submodule ports only) |
| `kind` | str \| None | `'submodule'`, `'device'`, `'passive'`, `'source'`, or `'lint_marker'`. Electrical interface checks exclude marker bindings. |

**Iterating connectivity:**

```python
for net_name, net in ckt.net_dict.items():
    print(f"{net_name}: fanout={net.fanout}")
    for b in net.pins:
        print(f"  [{b.kind}] {b.instance}.{b.port}")
```

**Rewiring a device pin (attribute assignment):**

```python
mn1 = pdk.nmos1p8('mn1', d='vout', g='vin', s='0', b='0',
                  w=1.0, l=0.15)
ckt.add(mn1)

mn1.d = 'vout_new'   # rewire drain

# Net.pins is updated automatically when the pin changes:
len(ckt.net_dict['vout'].pins)     # old net: binding removed
len(ckt.net_dict['vout_new'].pins) # new net: binding added
mn1.nets['d']                       # owner-side also updated
```

For a device added via `ckt.add(dev)`, assigning a pin removes its old
`Net.pins` binding, adds the new binding, and updates the device's `nets` map.

See `silk/examples/net_object_example.py` for a runnable walkthrough.

---

### `check_interfaces(module)` / `check_voltage_domains(module)` / `check_junction_overrange(module)`

Per-checker shortcuts that live in `silk.circuit` and return flat lists of lint
violations from `silk.lint.violations`. Interface and domain checks inspect this
module's connections and directly connected child ports; junction checks traverse
nested devices. Interface and domain rules inspect connected subcircuit ports; individual rules use their required directions,
roles or voltage domains. Junction checks inspect PDK device terminal voltages
and ratings, independently of port-role annotation.

```python
from silk import (
    check_interfaces, check_voltage_domains, check_junction_overrange,
    Severity,
)

iface_viol  = check_interfaces(top)        # list[InterfaceViolation]
domain_viol = check_voltage_domains(top)   # list[InterfaceViolation]
junc_viol   = check_junction_overrange(top)  # list[JunctionViolation]

for v in iface_viol:
    # v.severity is a Severity IntEnum; compare with Severity.ERROR / .WARNING / .INFO
    if v.severity >= Severity.ERROR:
        print(f"[{v.severity.to_string()}] {v.rule_id}: {v.message}")
```

**Interface rules** (all `category='interface'`):

| Rule ID | Default severity | What it flags |
|---|---|---|
| `INT-001` (DriverConflict) | ERROR | Multiple output bindings on the same net, including source outputs |
| `INT-002` (PowerGroundShort) | ERROR | A `POWER`-role port and a `GROUND`-role port sharing a net, or a `POWER`-role port on the simulator ground node `'0'`. Inside a hierarchical block the block's own POWER / GROUND port takes part: a child's GROUND port wired onto the block's own POWER port (or POWER onto GROUND) is reported with the block's port named |
| `INT-003` (RoleMismatch) | ERROR | A `DIGITAL`-role port connected to an `ANALOG`-role port — child against child; the block's own port is not a peer of its children and is never compared |
| `INT-004` (NoDriver) | ERROR | A signal net (never the ground node `'0'`) has at least two relevant bindings but no output binding, including nets with multiple INOUT ports. Lone connections are checked by STRUCT-011. Supply nets are skipped — a GROUND role declared or inferred, a POWER role only when **declared**: an inferred POWER role (`infer_port_roles`, e.g. `*vdd*` on `vdd_sense`) is a guess and does not silence the rule. A same-level device or passive pin, or the module's own IN/POWER/GROUND/INOUT port on a pass-through net, is the other party on the net and silences the rule (presence only, never a driver for INT-001); the module's own OUT port is not |
| `INT-005` (ClockOnAnalog) | ERROR | A `CLOCK`-role port shares a net with an `ANALOG`-role port — child against child; the block's own port is never compared |

`check_voltage_domains` runs the three `DOM-*` rules
(`RailMismatch`, `DomainCrossing`, `SupplyDisabledStress`) and returns
the same `InterfaceViolation` type.

`check_junction_overrange` runs `STR-001 JunctionOverrange` against terminal
voltages from either data path — behavioral or SPICE — via the shared
`module_has_simulation_data` predicate. AC and noise results are declined (see
`Rule.simulation_data_usable`), since their values are not node voltages.

### `junction_stress_coverage(module)` → `StressCoverage`

```python
from silk.circuit.junction_check import junction_stress_coverage
coverage = junction_stress_coverage(top)
print(coverage.summary())  # total, full, partial, none, failed
for device in coverage.devices:
    print(device.instance_path, device.coverage, device.failed, device.reasons)
```

`StressCoverage` and `DeviceStressCoverage` are also exported from
`silk.circuit.junction_check`. Each device records `instance_path`, `type_name`,
`failed`, `coverage`, `reasons` (a tuple), and `pairs` (a dictionary).
Coverage and STR-001 findings use the same terminal data and ratings.
Each call evaluates current results without launching a simulation.

- `full`: every recognized terminal pair was rated and checked.
- `partial`: at least one rated pair was checked, but others were not.
- `none`: no rated pair was checked, including devices with no ratings.
- `failed`: at least one rated pair failed. **This is independent of coverage.**
  A device with a failing rated pair and an unrated pair is both failed and
  partial. `full + partial + none == total`; do not add `failed` to that sum.

Pair values are `pass`, `fail`, `unrated`, or `unavailable`. Reasons are
`no_ratings` (including missing pair rating keys), `no_waveform` (including
save-list exclusion and no simulation), `small_signal`, `length_mismatch`, and
`all_ground`. An unrated pair stays `unrated` even if its waveform is absent.
SPICE preserves the whole-device availability gate: if any terminal is
unreadable, all rated pairs are unavailable. Behavioral data retains the
existing per-pair check on top-level devices; nested devices are recorded as
unavailable because the behavioral stress path has no terminal data for them.
All-ground SPICE devices are unchecked (no reference waveform); behavioral
all-ground devices can be checked using the behavioral time grid and zeros.
Coverage includes only types in the module's PDK device catalog. Ideal
R/L/C elements, controlled sources and all other non-catalog types are out
of scope: they are neither counted nor listed. A catalog device with no
ratings remains in scope with `no_ratings`.

Retained `report.sim_results` readers preserve the selected circuit and
waveforms for these checks. `copy.copy()` and `copy.deepcopy()` return result
readers with the same data. Pickling a retained reader raises `TypeError`
naming `EvidenceResults`: its live circuit and process-local evidence owner
cannot be serialized.

**Additive report data:** STR-001 `VisitEvent.context` contains
`stress_coverage` with a `summary` dictionary (integer `total`, `full`,
`partial`, `none`, `failed` counts) and a `devices` list. An evaluated rule
with unchecked pairs and no violations has `VisitOutcome.INCOMPLETE`; existing
`VIOLATION` and `SKIPPED_NO_SIM` outcomes remain, carrying the same coverage
context. No-simulation and small-signal visits have coverage without loading
waveforms. STR-001's coverage-tree counts count **PDK catalog devices**: checked means
full, unchecked means partial + none; other rules still count scope visits.

`LintReport`, `CoverageReport`, and `CircuitReport` add an optional
`stress_coverage` JSON key with this same object; older files load with `{}`.
CircuitReport persists it independently of the intentionally omitted `lint`
and `coverage` sections. Schema version numbers and violation fields are
unchanged. HTML, CircuitReport terminal/Markdown and lint JUnit outputs show
counts and per-device reasons; incomplete stress with no active error is
marked skipped in JUnit. Coverage describes evaluation before waivers and
scope-exclusion filtering of findings, so its failed count can include a
suppressed finding.

**`InterfaceViolation` fields** (`silk.lint.violations.InterfaceViolation`):

| Field | Type | Description |
|---|---|---|
| `rule_id` | str | Stable rule identifier, e.g. `'INT-001'` |
| `severity` | `Severity` | IntEnum (`INFO=10`, `WARNING=20`, `ERROR=30`) |
| `message` | str | Human-readable description |
| `location` | `NetLocation` | Tagged-union location (`net_name`, `port_names`) |
| `context` | dict | Optional rule-specific extras |
| `net` | str (property) | Alias for `location.net_name` |
| `ports` | list[str] (property) | Alias for `location.port_names` (formatted `"instance.port"` strings) |
| `kind` | str (property) | Backward-compat alias for `rule_id` |

**`JunctionViolation` fields** (`silk.lint.violations.JunctionViolation`):

| Field | Type | Description |
|---|---|---|
| `rule_id` | str | `'STR-001'` |
| `severity` | `Severity` | IntEnum |
| `message` | str | Human-readable description |
| `location` | `TerminalLocation` | `instance_path`, `terminal`, `terminal_pair` |
| `device_type` | str | e.g. `'nmos1p8'` |
| `measured_v` | float | Worst-case junction voltage: `\|V\|` for a magnitude-rated pair, the signed value for a signed-range pair |
| `rated_max` | float | The rated bound crossed: the magnitude, or for a signed range the `max` (message `exceeds <pair>_max`) or the negative `min` (message `below <pair>_min`) |
| `worst_time` | float | Axis value at the worst sample: seconds for transient, swept value for DC |
| `instance` | str (property) | Alias for `location.instance_path` |
| `kind` | str (property) | Backward-compat alias for `rule_id` |

**Example — catching a driver conflict:**

```python
amp1.define_module_ports({'vout': (PortDir.OUT, PortRole.ANALOG)})
amp2.define_module_ports({'vout': (PortDir.OUT, PortRole.ANALOG)})

top.add_module(amp1, 'amp1', nets={'vout': 'output'})
top.add_module(amp2, 'amp2', nets={'vout': 'output'})

violations = check_interfaces(top)
# → [ERROR] INT-001: Net 'output' has multiple drivers: amp1.vout, amp2.vout
```

Interface and domain rules read connected subcircuit `Net.pins` bindings.
Junction checks instead walk the PDK devices and their terminal waveforms.

For full-registry runs across all rule categories, use
`RuleRegistry.default()` — see the **Lint and coverage** section below.

See `silk/examples/port_role_example.py` for a runnable demonstration.

---

### `list_unannotated_ports(module)` / `annotation_coverage(module)`

Audit the ports of this module's direct child instances for a `PortRole`.
These functions do not recurse into grandchildren or count the module's own
ports. Missing roles reduce the checks that depend on roles; direction-based
interface checks can still find conflicts.

```python
from silk import list_unannotated_ports, annotation_coverage

# List every port with no role declared
unannotated = list_unannotated_ports(top)
for u in unannotated:
    print(f"  {u.instance}.{u.port}  direction={u.direction}  block={u.module_name}")

# Get structured coverage summary
cov = annotation_coverage(top)
print(cov)                      # "Annotation coverage: 12/15 ports (80.0%)"
cov.total                       # 15
cov.annotated                   # 12
cov.unannotated                 # 3
cov.coverage_pct                # 80.0
cov.unannotated_ports           # list[UnannotatedPort] — same as list_unannotated_ports()
```

**`UnannotatedPort` fields:**

| Field | Type | Description |
|---|---|---|
| `instance` | str | Instance name in `module_dict` (e.g. `'comp1'`) |
| `module_name` | str | Subcircuit definition name (e.g. `'comp'`) |
| `port` | str | Port name (e.g. `'vp'`) |
| `direction` | str | PortDir value (`'input'`, `'output'`, etc.) or None |

**`AnnotationCoverage` fields/properties:**

| Name | Type | Description |
|---|---|---|
| `total` | int | Total ports across direct child instances |
| `annotated` | int | Ports with a non-None role, including inferred roles and roles implied by POWER/GROUND directions |
| `unannotated` | int | Property: `len(unannotated_ports)` |
| `coverage_pct` | float | Property: `100.0 * annotated / total` (100.0 if total is 0) |
| `unannotated_ports` | list | `UnannotatedPort` objects for each unannotated port |

---

### `generate_report(module)` / `CircuitReport`

> Task-oriented guide with verified examples: [reports.md](reports.md).

Assemble a full circuit summary report from a Module. Covers topology,
port annotation coverage, interface check results and simulation statistics
when behavioral or SPICE results are available. The report object can be printed to the
terminal or saved as a Markdown, JSON, or HTML file. `generate_report(mod)` does
not discover bench results; pass `test_results=results` or `spec_reports=[report]`
to include completed test evidence.

```python
from silk import generate_report

report = generate_report(adc)
report.print_summary()              # condensed terminal output
report.save('flash_adc_report.md')  # formatted Markdown file
report.save_json('flash_adc_report.json')   # JSON (figures excluded)
report.save_html('flash_adc_report.html')   # self-contained HTML with embedded figures
```

For example, an empty, unsimulated circuit needs no simulator or PDK:

```python
from silk import Module, generate_report

empty = Module('example', 'silk-output/api-reference', 'empty.cir', pdk_details=None)
report = generate_report(empty, auto_discover_config=False)
report.print_summary()
```

This prints:

```text
Circuit:     unnamed
  Instances: 0  |  Nets: 0
  Annotation coverage: No ports to annotate
  Interface: PASS
  Domains:    PASS
  Junctions:  STR-001 NO DEVICES: 0/0 devices fully checked; 0 partial, 0 unchecked, 0 failed
  Simulation: not run
```

When there are no ports to annotate, HTML, Markdown and terminal reports say
"No ports to annotate", with no count, percentage or tick. The
`AnnotationCoverage` data and coverage gates retain their existing behavior.

The **Operating point** and **Metrics** sections auto-appear when the underlying data
is present (simulation has been run; specs have been declared). For sim runs, the
table filters to nets declared on the top-level module — hierarchical subcircuit
signals like `v(xfcamp.vbiasn)` stay out of the terminal view for reports built
from a live Module; reloaded reports display all retained rows. Access the full
list via `report.simulation.signals` for value-domain results,
`report.simulation.magnitudes` for AC/noise, or `save_html()`.

**`CircuitReport` fields:**

| Field | Type | Description |
|---|---|---|
| `module_name` | str | Name of the top-level module |
| `generated_at` | str | ISO timestamp of report generation |
| `topology` | `TopologySummary` | Subcircuit instances and net count |
| `annotation` | `AnnotationCoverage` | Port annotation coverage (from `annotation_coverage()`) |
| `violations` | list | Unsuppressed INT-* findings from the selected registry, after waivers and scope exclusions |
| `simulation` | `SimulationSummary` or None | Signal stats if a simulation (behavioral or SPICE) was run; `None` otherwise |
| `metadata` | `ReportMetadata` or None | Schema version, silk/PDK version, context hash. Populated by `generate_report()`. |
| `lint` | `LintReport` or None | Full lint framework report (violations + waivers + metadata). Populated by `generate_report()`. See **Lint and coverage** below. |
| `coverage` | `CoverageReport` or None | Rule-coverage tree (checked / excluded / unchecked per rule). Populated by `generate_report()`. |
| `spec_reconciliation` | `SpecReconciliationReport` or None | Reconciled specs (pass/fail/unmeasured + calibrations applied). Populated by `generate_report()` via `module.reconcile_specs()` — always attached, empty outcomes when no specs are declared, never `None` in practice. See the **silk.specs** section below. |
| `interface_check_passed` | bool | Property: read-only. True when no ERROR-severity violations |
| `figures` | `List[Tuple[Figure, str]]` | In-memory figures attached via `add_figure()`. Excluded from JSON. Embedded as base64 PNG in HTML output. |

**`TopologySummary` fields:**

| Field | Type | Description |
|---|---|---|
| `module_name` | str | Top-level module name |
| `instances` | list[`InstanceSummary`] | One entry per subcircuit instance |
| `total_nets` | int | Number of nets in `net_dict` |
| `total_instances` | int | Property: `len(instances)` |

**`SimulationSummary` fields:**

The summary says which analysis produced it and names its axis fields for
the axis actually present. Legacy JSON without analysis fields loads as a transient summary.

| Field | Type | Description |
|---|---|---|
| `analysis` | str | `'transient'`, `'ac'`, `'noise'`, `'dc'` or `'op'` — from `silk.signals.analysis_kind.rawfile_analysis_kind`. `'op'` is a real plot whose plotname is `Operating Point` (ngspice) or that holds exactly one point (Xyce writes an `.op` under the DC plotname with a placeholder `sweep` vector; a one-point `.dc` classifies `'op'` by the same rule) |
| `sweep_name` | str | The axis vector's own name, case-folded like `Signal.xlabel`: `'time'`, `'frequency'` (both backends), `'v(v-sweep)'` (ngspice DC sweep), `'sweep'` (Xyce DC sweep); `''` for an operating point, which has no axis |
| `sweep_start` / `sweep_stop` | float or None | Axis bounds in the axis's own unit — populated whenever there is an axis; `None` for an operating point |
| `t_start` / `t_stop` | float or None | Start / stop time in seconds; `None` unless the axis is time |
| `f_start` / `f_stop` | float or None | Start / stop frequency in hertz; `None` unless the axis is frequency (AC, noise) |
| `n_points` | int | Number of axis points (1 for an operating point) |
| `t_step` | float or None | Property: MEAN time spacing `(t_stop - t_start) / (n_points - 1)` — SPICE transient grids may be adaptive, while behavioral transient is fixed-step; `None` when the axis is not time |
| `sweep_step` | float or None | Property: mean axis spacing for transient / DC; `None` for all AC / noise grids, including linear ones |
| `signals` | list[`SignalStats`] | Per-signal min/max/mean for a value-domain run (transient, DC); every vector of the plot with `min == max == mean` = its value for an operating point (vector 0 included — it is a node on ngspice, Xyce's placeholder `sweep` on Xyce); empty for AC / noise |
| `magnitudes` | list[`MagnitudeStats`] | Per-signal magnitude statistics for a small-signal run (AC, noise); empty otherwise |
| `n_signals` | int | Property: `len(signals) + len(magnitudes)` |

**`SignalStats` fields** (value-domain runs):

| Field | Type | Description |
|---|---|---|
| `name` | str | Signal name |
| `min` | float | Minimum value across all axis points |
| `max` | float | Maximum value across all axis points |
| `mean` | float | Mean value across all axis points |

**`MagnitudeStats` fields** (AC / noise runs — the vector's magnitude `|x|`;
an AC vector is a complex small-signal value, while a noise vector can be a
density in V/√Hz or A/√Hz. The units depend on the vector; these records carry
no signed `min`/`max`/`mean`):

| Field | Type | Description |
|---|---|---|
| `name` | str | Signal name |
| `mag_min` | float | Minimum of `|x|` across all frequency points |
| `mag_max` | float | Maximum of `|x|` across all frequency points |
| `mag_mean` | float | Mean of `|x|` across all frequency points |

`test_results` holds detached caller-supplied evidence (tests dict, corners, or
record list), or `None` when omitted. `spec_reports=` embeds an ordered list;
`test_results=` accepts runner results. Supplying both raises `ValueError`.
CircuitReport JSON uses schema version 4; older readers refuse it. New readers retain legacy
decoding and restore absent evidence as `None`.

**`CircuitReport` methods:**

| Method | Description |
|---|---|
| `print_summary()` | Print a condensed summary to the terminal |
| `save(path)` | Write a formatted Markdown report to a file |
| `save_json(path)` | Write the report to a JSON file. Figures, lint/coverage, waveform links, composition diagrams and rendered operating-point tables do not round-trip; captured test evidence is separate. See [persistence limits](reports.md). Use with `load_json` for cross-session persistence. |
| `load_json(path)` | **Class method.** Reconstruct a `CircuitReport` from a JSON file saved by `save_json`. |
| `add_figure(fig, title)` | Attach a matplotlib `Figure` for HTML embedding. Figures are in-memory only — excluded from JSON. Can be called multiple times. |
| `save_html(path)` | Write a self-contained HTML report with inline CSS, color-coded pass/fail status, and base64-embedded figures. CSS and figures are embedded; linked waveform pages and their sidecars must be retained separately. |
| `add_waveform_view(source, *, out_path=None, label=None, annotations=None, context=None, title=None, inline=False)` | Emit a waveform-explorer page for `source` (a simulated `Module`/`ModuleTestbench`, or a rawfile path) and link it from the HTML report. Auto-populates the page's provenance popover (module, test, analysis, corner, PDK, spec pass/fail, silk version — each best-effort) and merges a caller-supplied `context=` on top. For an in-memory `RawFile`, use `silk.view` / `emit_view` directly. See [cli.md](cli.md#silksignalsview--waveform-explorer). |
| `add_monte_carlo(mc, *, metrics=None, top=3)` | Attach a Monte Carlo section built from a `MonteCarloResults`: yield table per spec and corner (95% Wilson interval; samples run/failed/used), one embedded histogram per metric (default: metrics some spec checks) with its limits, and the failed samples with seeds. With `mc.draws` recorded (`record_draws=True`) it adds the sensitivity table per metric and corner in share order plus scatter plots of the metric against its `top` parameters; without draws there is no sensitivity section. `save_json` writes the same numbers under `'monte_carlo'` (no images, key absent when none added); `load_json` restores them. Raises `ValueError` for `top` not a positive int and `MonteCarloError` for an unknown metric. |
| `add_composition_diagram(module, depth=1)` | Attach the module's block diagram (see `silk.circuit.composition`; `silk/examples/composition_graph_demo.py`). Markdown can render it in a Mermaid-enabled viewer; saved HTML displays Mermaid source text without a Mermaid runtime. |

`generate_report(module, registry=None, auto_discover_config=True,
layout_estimate=None, spec_reports=None, test_results=None, *, region_thresholds=None)` — `registry` overrides the lint `RuleRegistry`
(default: `RuleRegistry.default()`), `auto_discover_config` looks for
`silk_waivers.yaml` and `silk_lint_exclusions.yaml` in the current working
directory only when no registry was supplied, and `layout_estimate` attaches a
`LayoutEstimate` (see the layout additions at the end of this document).
`region_thresholds` overrides operating-region policy for all included reports;
when omitted each report uses its PDK and the `SILK_REGION_THRESHOLDS` setting
(see [operating-point.md](operating-point.md#project-and-customer-threshold-files)).

The Markdown report includes topology, port annotation coverage and interface
checks. Depending on the available data, it also includes simulation, domain
and junction findings, stress coverage, specs, layout, composition diagrams and
attached test evidence. The Spec Reconciliation section is emitted only when the
module has at least one declared spec — empty-spec reports skip the header to
avoid noise.
See `tests/test_report_diff.py` for comprehensive usage examples and
`docs/design_stage_bridge_user_guide.md` for the full walkthrough of the
reconciliation section.

---

### `diff_reports(a, b, ...)` / `ReportDiff`

Compare two `CircuitReport` objects or JSON files for cross-session diffing.
Typical use: compare schematic vs. extracted simulation results, or before/after
an optimization pass.

```python
from silk import diff_reports

# Cross-session: load from JSON files saved in earlier runs
diff = diff_reports('sim_schematic.json', 'sim_extracted.json',
                    label_a='schematic', label_b='extracted')
diff.print_summary()
diff.save('schematic_vs_extracted.md')

# In-session: compare two in-memory reports
diff = diff_reports(report_a, report_b, sections={'signals', 'violations'})
```

For the empty `report` above:

```python
diff_reports(report, report, label_a='schematic', label_b='extracted').print_summary()
```

This prints:

```text
Diff: schematic → extracted
  Topology: unchanged
  Annotation: unchanged (100.0%)
  Violations: unchanged (0)
```

**`diff_reports()` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `a`, `b` | `CircuitReport` or `str` | — | Reports to compare; `str` treated as JSON path |
| `label_a`, `label_b` | str | `'a'`, `'b'` | Human-readable names for each report |
| `sections` | set of str | all | Sections to compare: `'signals'`, `'topology'`, `'annotation'`, `'violations'` (interface findings only; excludes voltage-domain and junction findings), `'provenance'`. Layout and spec transitions are compared regardless of this selection |

**`ReportDiff` fields:**

| Field | Type | Description |
|---|---|---|
| `label_a`, `label_b` | str | Report labels |
| `signal_diff` | `SignalDiff`, `MagnitudeDiff`, `KindMismatch` or None | `SignalDiff`, `MagnitudeDiff` or `KindMismatch` for value-domain, small-signal or differing analyses; `None` if simulation is absent on either side |
| `topology_diff` | `TopologyDiff` or None | Instance and net changes |
| `annotation_diff` | `AnnotationDiff` or None | Coverage percentage delta |
| `violation_diff` | `ViolationDiff` or None | Interface findings introduced, resolved, unchanged; excludes voltage-domain and junction findings |
| `spec_transitions` | `list[dict]` | Per-spec status or value transitions — see the `silk.specs` section for the dict shape |

**`SignalDelta` fields:**

| Field | Type | Description |
|---|---|---|
| `name` | str | Signal name |
| `mean_a`, `mean_b` | float | Mean value in each report |
| `delta_mean` | float | Property: `mean_b − mean_a` |
| `max_a`, `max_b` | float | Max value in each report |
| `delta_max` | float | Property: `max_b − max_a` |

Violations are matched by `(rule_id, location.describe())`, ignoring message, severity and context changes; repeated findings with the same rule and location produce one entry.
See `tests/test_report_diff.py` for comprehensive usage examples.

---

## silk.circuit — Element naming

Public naming helpers for saved source entries and built elements:

```python
from silk.circuit import ElementNames, element_names, source_element_names

# One saved sources[] entry; no elements are constructed.
entry = {"type": "VoltageSource", "instance": "Stim",
         "positive": "in", "negative": "0", "dc": 0.2,
         "pwl": [0, 0, 1e-9, 1], "array": 2}
records = source_element_names(entry)
# (ElementNames('Stim[0]', 'i(Stim[0])', ('VStim<0>__silk_pwl', 'VStim<0>__silk_dc')),
#  ElementNames('Stim[1]', 'i(Stim[1])', ('VStim<1>__silk_pwl', 'VStim<1>__silk_dc')))

# At results time, use the actual source or device from the built netlist.
# record = element_names(tb.netlist.sig_sources_dict['Stim[0]'])
# signal = tb.probe(record.current_probe, analysis='tran')
```

`ElementNames(designer_name: str, current_probe: str | None,
card_names: tuple[str, ...])` is a frozen dataclass. `designer_name` preserves
its authored spelling; `current_probe` is a supported current request or
`None`; `card_names` lists **every emitted instance token**, in card order,
not whole SPICE lines. Card tokens preserve emission case. **Ngspice
lowercases rawfile vector names**, so match raw names against lowercase card
tokens, retaining their array angle brackets. Array brackets transform only
in instance components: `Stim[0]` emits as `VStim<0>`, while an underscore
array emits as `VStim_0`.
For a top-level independent current card `IStim`, a saved ngspice operand is
`i(@istim[current])`; `card_names` contains `IStim`, without that raw-vector
wrapper.

`source_element_names(entry: dict) -> tuple[ElementNames, ...]` accepts one
`sources[]` entry in the testbench-codegen format. Scalars yield one record;
arrays yield one for each created element, omitting the bare list variable.
Integer counts start at zero; `[start, stop]` ranges exclude `stop`.
`naming='bracketed'` is the default; `naming='underscore'` is also accepted.
Invalid entry shapes, unknown types, malformed/conflicting waveforms,
invalid naming, empty arrays and testbench variable references raise
`TestbenchSpecError`. This call does not validate whole-spec duplicate names
or controller references, evaluate expressions/values, or construct devices
or waveforms. Waveform constructors and simulation still validate electrical
values. The entry helper and record are also exported from `silk.codegen`.

`element_names(element) -> ElementNames` reads a built `SourceBase` or
`AnalogElement`, including PDK devices and ideal passives, without rendering
the element or resolving its values. Sources report their physical layout;
other devices report their emitted card. Pass the device object from an
`elem_dict` entry, not the surrounding dict or a catalog device description.
Custom legacy `PassiveBase` instances are supported when their subclass
implements `physical_instance_names()`: a nonempty tuple of unique,
value-independent card tokens shared with `to_spice()`. Missing declarations
raise `NotSupportedError`, and malformed declarations raise `CircuitError`.
The helper reports no current probe for a custom passive. Custom identity must
remain stable after placement; silk does not infer it from rendered SPICE.
Unsupported object types raise `TypeError`. These are local names: a caller
walking a hierarchy must supply the placement path when matching raw vectors.

On ngspice, independent voltage/current sources, VCVS (`E`), CCVS (`H`), and
`BehavioralVoltage` (`B V=`) return a logical `i(<designer name>)` probe.
VCCS (`G`), CCCS (`F`) and `BehavioralCurrent` (`B I=`) return `None`.
Literal colons in authored source names are refused with `CircuitError`:
colons separate hierarchy levels and cannot be part of a local name.
Resistors, capacitors and PDK transistor/device cards also return `None`:
this API reports branch-current probes, not device-terminal selectors.
Built inductors and ideal voltage-defined device cards (`V`, `E`, `H`) return
the existing raw probe `i(<emitted card>)`, for example `i(Lcoil<2>)` for
an inductor named `coil[2]`; these devices are outside the logical source-name
lookup. A non-`None` probe can therefore use a different spelling from
`designer_name`; use the returned string directly.

Constant independent sources write one card. All independent waveforms
write a waveform card followed by a DC card, even with a zero offset:
dedicated PWL (including a `waveform` of type `PWL`) uses `__silk_pwl` / `__silk_dc`;
other waveform kinds use `__silk_wave` / `__silk_dc`. A voltage source's logical current
reads the waveform branch; a current source's logical current sums both
parallel cards. `element_names` also handles directly built generic waveforms,
whose suffix is `__silk_wave`, including a generic `waveform=PWL(...)`.

A non-`None` probe describes ngspice readback capability, not support in
every backend/analysis. Independent current-source probes require saved
operands in transient, DC and operating-point analyses, and refuse AC with the
existing 0 V voltage-source ammeter advice. Behavioral simulation and other analyses retain their own
probe limits. Declare `tb.probe` before running to acquire its operands.
The legacy `emitted_instance_name(element, name)` and `_entry_names(entry,
where)` remain available with their existing behavior; the former still
returns a waveform source's logical base token, not its physical card list.

## silk.simulation.variation — Behavioural parameter declarations

```python
from silk import Normal, Uniform
from silk.exceptions import VariationError
```

| Constructor | Meaning |
| --- | --- |
| `Normal(nominal, sigma, *, variation='mismatch', name=None)` | Normal distribution; finite nominal and nonnegative finite sigma. Zero sigma is constant. |
| `Uniform(low, high, *, variation='mismatch', name=None)` | Finite ordered limits with a finite span; nominal is the midpoint. Equal limits are constant. |

These frozen declarations are supported as top-level values in a behavioural
model's `params` dict. Fields accept real numbers (including NumPy scalars,
excluding booleans), `Variable` objects and their arithmetic expressions.
Variable fields are read and validated when the run starts. The model receives
plain nominal floats, held for all analyses and DC points of that run, in a
fresh parameter dict per executed instance. Without declarations, it still
receives its live parameter dict.

`variation` is exactly `'mismatch'` or `'process'`. Process declarations require
a nonempty string `name`; mismatch declarations cannot have a name. Reusing a
process name with different distribution kinds or resolved fields raises
`VariationError`, naming both blocks, even in nominal runs.

`VariationError` inherits both `ConfigurationError` and `ValueError` and
propagates through `run_all_tests`. Unsupported field types raise `TypeError`.
Nested declarations and declarations used as SPICE values raise
`NotSupportedError`; a declaration cannot be a `Variable.value` (`TypeError`).
Calibration without a field refuses: reconciliation raises `VariationError`,
and `calibrate_behavioral` directs callers to `CalibrationBinding(field=...)`;
field targets preserve and validate the statistical declaration.

Behavioural Monte Carlo campaigns draw these declarations once per sample via
`set_monte_carlo` / `run_monte_carlo`; at least one active declaration is
required. See
[declaring variation](monte-carlo.md#behavioural-models-declaring-variation).

## silk.simulation — Simulator backends (ngspice, Xyce)

silk is **simulator-agnostic above the `Signal` line**: metrics, specs, and
reconciliation consume `Signal` objects and never care which simulator ran. The
*produce-a-Signal* pipeline lives behind a single selector, `module.simulator`
(a string, default `'ngspice'`), which resolves to a `SimulatorBackend`.

### Selecting a simulator

```python
mod.simulator = 'xyce'      # validated at assignment — unknown names raise ValueError
```

Built-in backends are `'ngspice'` (default), `'xyce'`, and `'behavioral'`,
silk's numpy engine selected through the same setting. A
backend whose `runs_module` is `True` (only `'behavioral'`) has no SPICE deck:
`Module.simulate()` runs the Module itself through
`SimulatorBackend.run_module`, which applies the same data-presence gate as
`run` (an empty result raises `SimulationAborted`); `generate_netlist()` for a
top-level deck and `run_simulator()` raise `NotSupportedError` naming the
backend; and `set_simulator_options(...)` raises `NotSupportedError` for every
key, since it takes no options (the portable presets are accepted as empty
no-ops). It evaluates declared `Transient` analyses and DC source sweeps, sharing model
state across analyses in one run. Other analysis types raise `NotSupportedError`. A standalone
`Module.simulate()` uses that Module's selection for emission, execution and
loading; its netlister rebuilds lazily when the selection changes.

For an embedded test, **`tb.netlist` is the Module that executes**, and `tb.module`
is its live owning circuit. At each `tb.run()`, the executing Module inherits
its owner's current `simulator` before deck preparation. This
works when the owner selects before or after test construction, changes after
a successful run, or is a copied live child instance. `run_test()` (including
`store=True`), `run_all_tests()`, `run_corners()` and ordinary optimizer tests
all use that shared entry point.

An explicit `tb.netlist.simulator = 'ngspice'` or `'xyce'` takes precedence and
survives later owner changes and copying. Assigning `'ngspice'` explicitly is an
override even though it is the constructor default; automatic inheritance never
becomes an override. Until a run resolves inheritance, `tb.netlist.simulator`
shows its initialized or last-used selection. Calling `tb.netlist.simulate()`
directly uses ordinary standalone Module semantics.

Recursive `run_all_tests()` resolves **each descendant's own live owner**;
selecting the root does not retarget children. Mixed-backend tests are supported.
Before each run, a testbench inherits its live owner's simulator, ambient
temperature and supply settings unless it explicitly overrides the corresponding
setting. Numerical options inherit per key: explicit testbench values win,
inherited values refresh on each run, and clearing an owner option removes its
inherited value. Backend selection does not translate simulator-specific option
names; use keys supported by the selected backend. Set test-specific numerical
options with `tb.netlist.set_simulator_options()`. `$SILK_XYCE_BIN` / `$SILK_NGSPICE_BIN` locate executables;
they do not select a backend. A missing selected executable raises
`SimulationError`, with no fallback. Xyce needs its CLI on `PATH` or
`$SILK_XYCE_BIN`; see [Xyce support and limitations](known_issues.md).

Selection alone does not launch, prepare or delete files. Actual runs retain
[the output lifecycle](#testbench-output-directories-and-run-labels), including
exact-file freshness and failure invalidation. The report cache is disabled
in 0.2.0: `cache=True` warns, and every `run()` simulates. Explicit
`evaluate(rawfile=...)` continues to read disk afresh without running or
preparing a simulation.

Runnable embedded-test example (requires ngspice and Xyce; the ideal load needs
no installed PDK). The existing factory registers `level` before returning:

```python
from tempfile import TemporaryDirectory
from silk.examples.output_modes_example import build_load

with TemporaryDirectory() as output:
    dut = build_load(output_root=output)
    tb = dut.get_test('level')
    dut.simulator = 'xyce'             # selected after test construction
    xyce_report = dut.run_test('level')
    assert xyce_report.passed
    assert abs(xyce_report.metrics['voltage'] - 1.0) < 1e-9

    dut.simulator = 'ngspice'          # the same test follows the live owner
    assert dut.run_all_tests()['load.level'].passed

    tb.netlist.simulator = 'xyce'      # deliberate test-specific override
    assert tb.run().passed            # Xyce despite the ngspice owner
```

### `SimulatorBackend` — the six-seam contract

The backend interface covers the whole pipeline through six methods (`silk/simulation/backend.py`).
Only `render_analysis`, `make_netlister` and `_launch` are `@abstractmethod` on
`SimulatorBackend` itself; `has_data` carries a working default there, and
`load_signal` / `device_current_vectors` are supplied by `SpiceBackend` for the
SPICE family, so a non-SPICE backend must provide those two itself:

| Method | Returns | Role |
|---|---|---|
| `render_analysis(analysis, ctx)` | `str` | one analysis → command line(s) |
| `make_netlister(pdk_details)` | netlister | builds the deck (body + wrapper) |
| `_launch(netlist_path, results_dir, *, debug, timeout)` | `RawResult`, or `(RawResult, log_path, reason)` | run the simulator and load its native output |
| `load_signal(raw, name, *, type, analysis)` | `Signal` | extract one waveform from a `RawResult` |
| `device_current_vectors(raw, instance_path, leaf_name)` | `dict[str, str \| None]` | map a device's terminal-current suffixes → rawfile vector names (for current annotation); absent key: no match; `None`: ambiguous match |
| `has_data(native)` | `bool` | did this run produce data rows? Default answers the nutmeg shape and the `silk_has_data()` protocol; override only for a non-nutmeg native |

`run(netlist_path, results_dir, *, debug, timeout) -> RawResult` is **concrete on
the base class and must not be overridden**: it is a template method that calls
this backend's `_launch` and then applies silk's data-presence gate
(`_require_data`), so no backend can produce a false green by forgetting the
check. `SimulatorBackend.__init_subclass__` raises `TypeError` on a subclass that
defines `run`. See
[simulation-error-handling.md](simulation-error-handling.md).

`SpiceBackend` (abstract) implements `render_analysis`, `load_signal`, and
`load_raw` for the SPICE family, and a **concrete** `device_current_vectors` that
raises `NotSupportedError` (so backends that can't expose device currents — Xyce
today — degrade gracefully rather than failing to instantiate); `NgspiceBackend`
and `XyceBackend` subclass it and supply `make_netlister` + `_launch`, and
`NgspiceBackend` overrides `device_current_vectors`. A non-SPICE simulator (e.g.
a simulator with its own netlist dialect) would subclass `SimulatorBackend` directly.

`RawResult(native, source_path)` is the run→load handoff: `native` is the
backend's native payload (for SPICE, a parsed `RawFile` from `silk.signals.rawfile`), `source_path`
the file it was read from. Its shape is backend-private; only `load_signal`
crosses backends.

### Adding a backend

```python
from silk.simulation.backend import SimulatorBackend, SpiceBackend, RawResult

@SimulatorBackend.register('mysim')
class MySimBackend(SpiceBackend):          # SPICE family → inherits render/load
    def make_netlister(self, pdk_details):
        return MySimNetlister(pdk_details)  # usually a PySpiceNetlister subclass
    def _launch(self, netlist_path, results_dir, *, debug=False, timeout=None):
        # launch the simulator; read its output into a native object.
        # No abort bookkeeping here: SimulatorBackend.run() called this and
        # gates the result. Return `(raw, log_path, reason)` instead if this
        # simulator has a log to scrape or a diagnostic already in hand.
        return RawResult(native=..., source_path=results_dir + '/rawfile.raw')
```

No edits to `Module` are needed — selection is by name. See
`tests/test_backend_seam_smoke.py` for a complete fake backend driving the core
seams, and `silk/circuit/xyce_netlister.py` for a real netlister subclass.
`device_current_vectors` is optional: leaving `SpiceBackend`'s base (which raises
`NotSupportedError`) simply means `Module.annotate_currents()` reports the backend
as unsupported. `has_data` is optional too — override it (or give the native a
`silk_has_data()` method) only if the native result is not a nutmeg rawfile.

---

## Current annotation

`Module.annotate_currents(tstart=None, recursive=True, force=False)` turns a transient SPICE run into per-net current
data for EM wire sizing. It populates `PortBinding` pins with a signed `avg_current` (`+` = current
flowing from the device **into** the net) and nonnegative `rms_current` and
`peak_current` magnitudes plus conservative `Net`-level
aggregates (partial nets and subcircuit port nets receive none; inspect
`report.partial_nets` and per-pin readings). For complete nets, `½·Σ|·|` is an upper bound on any routed segment's current, valid for
every routing tree; exact for a 2-pin net. A pin records its rawfile vector
in `current_vector` when there is one. Composite source currents sum multiple
terms and have `current_vector=None`.

**Enable the save, then annotate** (two flows):

```python
# Flow A — module used directly as the top-level netlist
tb.save_device_currents = True        # BEFORE netlist generation (default False)
tb.simulate(corner='tt')
report = tb.annotate_currents(tstart=1e-6)   # tstart=None → full trace

# Flow B — through the test API (save flag is propagated to the tb by run())
mod.save_device_currents = True
spec_report = mod.run_test('tran')
report = spec_report.sim_results.annotate_currents()   # sim_results IS the tb Module
```

Setting `save_device_currents = True` switches the ngspice deck to batch-write
mode (`.options savecurrents` + deck-level `.save all`, no `.control` block) and
requires exactly one `Transient` analysis. With the flag off, ordinary emission
is unchanged. Windowing is time-weighted (trapezoidal) over `t ≥ tstart`.
User overrides are skipped unless `force=True`; `recursive=False` limits
annotation to this module. A window starting at/past the final sample raises `ValueError`.

**Read paths (hierarchy).** Each placement has a separate annotation snapshot
in `report.instance_modules['x<inst>']`, containing its exact currents.
The shared live module, available through `iter_instances()` or the original
reference, receives each pin's readings from the placement with the largest absolute average
current; RMS and peak are not independently maximized.

**Report** (`CurrentAnnotationReport`): `nets_annotated`, `pins_annotated`,
`skips` (`(net, pin, reason)` with reason `no-vector`/`ambiguous-match`/
`unsupported-kind`/`user-override`), `worst_kcl_residual` per fully-annotated net
(the self-check: `Σ avg ≈ 0`), `partial_nets`, `instance_modules`,
`definition_reduced`, and provenance (`results_file`, `analysis`, `window`).

> **Limitation (MOS drain-only) — applies to `savecurrents` mode only.**
> ngspice `savecurrents` populates only the **drain** current for sky130
> subckt-wrapped MOS; source/gate/bulk read 0, so drain-only annotation leaves
> source/gate-fed nets partial (KCL-exempt). Use
> `Module.terminal_current_method = 'ammeter_wrapper'` (0 V series ammeters on
> every terminal — all four MOS terminal currents, uniform sign), which
> unblocks EM sizing of those nets.

### Device operating-point parameters

The DUT's capture settings are inherited by its testbenches before each run:

| Attribute | Default | Effect |
|---|---|---|
| `save_device_currents` | `False` | Capture device terminal currents; see the current-capture restrictions above. |
| `save_device_params` | `False` | Capture operating-point parameters for MOS devices in the DUT and all testbench helper placements as described below. On a testbench `tb.save_device_params` / `create_test(..., save_device_params=)` is `None` (inherit the device under test's value, the default) or a `bool` that overrides it for that test alone; emission and the report's `op_table()` use the same effective value, serial and `n_workers>1`. Anything else raises `TypeError`. |
| `omit_device_internal_nodes` | `False` | `True` emits ngspice `save nointernals` (or batch `.save nointernals`). Requires the version-table capability (45+), built-in ideal content or the exact audited sky130 primitive-model certificate. Refuses other backends, custom/opaque content, unverified model bundles and any `set_save_list` selection/exclusion/depth limit. See [the admission boundary](simulator-save-list.md#dropping-device-internal-nodes). |

`Module.save_device_params = True` (ngspice and Xyce, default `False`)
captures small-signal operating-point parameters (`id`, `vgs`, `vds`, `vbs`,
`vth`, `vdsat`, `gm`, `gds`, `gmbs`, `cgg`, `cgs`, `cgd`, `cgso`, `cgdo`,
`cgbo`, `capbd`, `capbs`) on a dedicated op-point solve. On ngspice this
runs alongside whatever analysis the test itself declares; on Xyce
`Op()` must be the deck's *only* declared analysis (Xyce has no post-run
interactive step — it implements `Op()` as a one-point DC sweep with
operating-point columns in its print output), and only 8 of the 17
names are ever readable there (`gmbs`/the capacitances are confirmed
unavailable on Xyce — see `docs/operating-point.md`'s column matrix):

```python
tb.save_device_params = True          # BEFORE netlist generation (default False)
tb.simulate(corner='tt')

from silk.simulation.backend import SimulatorBackend
backend = SimulatorBackend.get('ngspice')
params = backend.device_op_vectors(tb._raw, instance_path='', leaf_name='XMM1')
# Parameter-name keys map to floats; unavailable values are NaN.
```

**Only a device whose PDK model DECLARES a verified op-parameter set has
parameter saves emitted** — a per-device `op_params:`/`op_internal_name:`/`op_backends:`
triple in the PDK catalog YAML (`silk/pdk/data/*_device_details.yaml`),
absent by default. `op_backends:` lists which simulator
backends that device's addressing has actually been verified on by a real
run — Xyce aborts its *whole* run on one unresolvable `.print` symbol, so
"works on ngspice" is never assumed to also work on Xyce. An undeclared
device (any model nobody has verified with a real run, or verified on a
different backend than the one selected) has no parameter saves emitted:
its table row remains present with unavailable values. When the backend's operating-point plot is absent
(`op` on ngspice, `dc` on Xyce), `device_op_vectors` returns `{}`; the report fills the row with NaNs. A design with no declared
device at all warns once, naming the PDK, without raising — the main
analysis and its own data are never affected either way. See the
[current device/backend table](operating-point.md#which-devices-are-declared)
for the declared families. The overlap-capacitance columns
`cgso`/`cgdo`/`cgbo` are excluded from the catalog's capture set.

When the switch is off, it adds no parameter saves. On ngspice, captured
parameters land in a separate `op1` plot alongside the declared analysis —
`tb.probe()`/`tb.load_signal()` calls written before the switch was ever
turned on keep resolving exactly as they did with it off (an omitted
`analysis=` still picks a single-analysis test's one declared analysis, never
`op1` itself). Incompatible with `save_device_currents` (any
`terminal_current_method`) — that combination raises at generation time,
since `save_device_currents` forces ngspice's batch-write mode, which emits
no `.control` block at all. On Xyce, use `Op()` as the only analysis, as
described above.

**`SpecReport.op_table()`** turns this raw capture into a
testbench-wide table — one row per MOS device in the DUT and all helper
placements, raw + derived columns, an
operating-region classification, and console/Markdown rendering. See
`docs/operating-point.md` for the full column list, the region rule, and
the sign conventions. Rows use bench-rooted names and emitted paths:
`table['dut.bias1.MN']` has path `xdut.xbias1.XMMN`, while a helper
placed as `bias1` uses `table['bias1.MN']` and `xbias1.XMMN`. Rows follow depth-first bench
insertion order, so a helper placed first can become table[0]. Historical
stored reports retain their labels/order. Undeclared devices keep unavailable
rows; only verified parameters produce save requests. Capture is an
evaluate-time snapshot and needs no live bench/rawfile for later lookup.

---

## silk.faults — Fault injection and Monte Carlo campaigns

Inject transistor-level faults into a silk `Module`, run fault realizations
through ngspice serially by default or in a worker pool, and
aggregate pass / fail statistics and fault-correlations across the
population. Faults are applied *at netlist-emission time* via a
non-mutating transform — the Module object is never mutated, so a
single Module instance can drive arbitrarily many campaigns.

### Quick start

```python
from silk.faults import (
    FaultSet, FaultCampaign,
    stuck_open, vth_shift, w_variation,
)

# Supply a top-level ngspice testbench with an analysis and matching metrics.
# make_flash_adc() stands for your own circuit factory.
adc = make_flash_adc()

# Declare faults. Pass a placed device handle to target that instance.
fs = FaultSet([
    stuck_open(adc.comp1.mn1),
    vth_shift(adc.comp5.mn2, delta=0.05),
])
fs.sprinkle(scope=r'xcomp\d+\.mn5',
            fault=w_variation(sigma=0.02),
            rate=1.0)

# Run a Monte Carlo campaign (each run gets a deterministic per-run seed).
campaign = FaultCampaign(adc, fs, n_runs=20, seed=42, n_workers=4,
                         out_dir='fault_out/')
result = campaign.run()

# Per-spec accessors and a markdown / HTML / JSON report.
print(result.pass_fraction('adc_sndr'))
result.generate_report().save('fault_out/summary.md')
```

See `silk/examples/fault_injection_example.py` for the end-to-end
runnable script.

### Fault types

Fault helpers accept a device handle or path; omitting the target creates a
factory for `sprinkle()`. Fault parameters are keyword-only.

| Name            | Netlist effect                                           | Params                         |
|-----------------|----------------------------------------------------------|--------------------------------|
| `stuck_open`    | delete device (m=1) or decrement `m` by one; `per_finger=False` removes the whole device              | `per_finger: bool = True`      |
| `short_ds`      | shunt 1 µΩ resistor drain↔source                         | `per_finger: bool = True`      |
| `short_gs`      | shunt 1 µΩ resistor gate↔source                          | `per_finger: bool = True`      |
| `short_gd`      | shunt 1 µΩ resistor gate↔drain                           | `per_finger: bool = True`      |
| `w_variation`   | rewrite `w` via `nominal × (1 + σ·N(0,1))` (schema-dependent)  | `sigma: float`                 |
| `l_variation`   | rewrite `l` via `nominal × (1 + σ·N(0,1))` (schema-dependent)  | `sigma: float`                 |
| `vth_shift`     | rename gate net; a unity-gain VCVS from the gate, sitting on a DC source of `delta + jitter`, drives it | `delta: float, sigma: float=0` |

W/L perturbations are clamped, quantized and snapped using the declared fault
schema; without one, silk warns and applies the perturbation without clamping.

A fault realizes to device changes — overrides on the target, its removal,
added silk elements — and every one is emitted through the element's own
writer, exactly as in an unfaulted deck: unit conversion, the shared
parameter renderer and (in ammeter mode) the terminal ammeters all apply.
Overrides are applied in the PDK's own unit, the unit `fault_params` bounds
are declared in. The manifest's realized `w_nominal`/`w_new`
(`l_nominal`/`l_new`) are in the user's units, the units the circuit was
written in.

### `FaultCampaign`

```
FaultCampaign(
    module,                # silk.circuit.modules.Module
    fault_set,             # silk.faults.FaultSet
    n_runs,                # int — number of Monte Carlo runs
    seed: int = 0,
    out_dir: str = 'campaign_out/',
    n_workers: int | str | None = None,   # None/1 → serial; 'all' → every CPU
    archive_waveforms: bool = False,
    reconcile_specs: bool = True,
    progress: bool = True,
)
```

`n_workers` is resolved by `silk.simulation.executor.resolve_n_workers`:
`None`/`1` runs serially in-process (the default), an int `N` runs a pool of `N`
workers capped at `os.cpu_count()`, and `'all'` uses every CPU. Each run
still gets its own `runs/run_NNNN/` folder, SeedSequence-derived seed and
spec evaluation regardless of `n_workers`. Evaluation happens in the parent, on each
run's own rawfile, never in the worker.

Worker pools use the `'fork'` multiprocessing context so workers inherit
registered PDK classes. On a platform without `'fork'`, execution falls back
to serial scheduling with a `RuntimeWarning`.

`campaign.run()` returns a `CampaignResult` with `runs: list[RunResult]`
and accessors `pass_fraction(spec)`, `measurement_distribution(spec)`,
`runs_failing(spec)`, `correlate_faults_with_failure(spec)`, plus
`generate_report()` which returns a `CampaignReport` with `summary()`,
`save(md_path)`, `save_html(path)` and `save_json(path)`.
With no usable outcomes, `pass_fraction()` returns NaN,
`measurement_distribution()` an empty array, `runs_failing()` an empty list
and `correlate_faults_with_failure()` an empty mapping.

**Verified versus execution-only.** With `reconcile_specs=True`
the campaign binds the one testbench whose `netlist` is the campaign module
(`FaultCampaign(tb.netlist, ...)`) and evaluates that bench's metrics and
specs on each run's faulted output through the ordinary
`evaluate(rawfile=...)` path (no extra launch, nothing written to the
caller's specs or results). A bare top module binds no bench, so its specs
are `UNMEASURED` and the run is `unverified`; old recorded measurements are
never read. A run's `status` is `passed` only when at least one spec was
checked, every checked spec passed on fresh data, and none is unbound;
`failed` when any checked spec failed (an extractor error or NaN is a failure
with no value); `unverified` otherwise; `simulated` when
`reconcile_specs=False` (executed, nothing scored); `errored` on a simulator,
configuration, scoring-infrastructure or cleanup error encountered during an
individual run (an extractor error is a failed spec, not an errored run).
Campaign preflight failures propagate before individual runs launch; for example,
an ambiguous bound metric can raise `MetricBindingError` from `campaign.run()`.
Only the bench's
`dut` placement is checked; specs of sibling tests or other placements are
listed in `RunResult.excluded` with the reason. `RunResult.evaluation` is
`'fresh-v1'`, `'execution-only'` or `None` (older). `archive_waveforms=False`
removes each rawfile after it is scored, so all rawfiles of a batch coexist
until scoring finishes. Statistics are keyed by `(owner_path, spec_key)`; a
display name that matches several specs raises `ValueError`, and only checked
pass/fail runs are counted.

`campaign.resume(prior_result)` re-runs the `errored` / missing `run_id`s and
any prior run that was not scored by this pipeline in the current mode (every
legacy result), or has a different mode, campaign ID/seed, fault set or test name;
it writes `summary.json` from the merged result. Resume assumes the module,
metrics and specs remain unchanged; it does not compare those inputs.

### Fault limitations

* **Placement-local faults.** A fault on one complete target path (for example
  `xdut.xleft.mn1`) affects only that placement, including when siblings share
  one `Module` object or equal definitions deduplicate. Faulted decks emit
  separate definitions for the affected placement and every ancestor on its
  path, including a once-placed `xdut`. Generated names use
  `__silk_fault_<healthy definition name>_<hex ancestry>` with collision-safe
  suffixes; instance names and wires stay the same, so saved/probed instance
  paths remain valid. Healthy definitions remain shared. Authored names,
  devices and registrations stay unchanged, and later clean emission restores
  the original deck.
* **One fault per complete target per run.** Multiple realized faults for the
  same complete target path are deduped — only the first is applied. Different
  placement paths receive their own faults even when they share a Python
  device.
* **SPICE-only.** Campaigns refuse the behavioral backend with
  `NotSupportedError`; faults have no behavioral-model representation.

---

## silk.specs — Design-stage bridge

First-class hierarchical `Spec` objects with multi-source measurement,
bidirectional reconciliation, and optional calibration write-back onto
parameterized behavioral models. This is the data model behind the
design-stage bridge that connects architecture-level specs to
design-level measurements and back.

Core classes (`Spec`, `Measurement`, `CalibrationBinding`, `SpecOutcome`, `SpecReconciliationReport`) are re-exported from `silk`. The three type aliases `Status`, `ComparisonOp`, and `MeasurementSource` are **only** available from `silk.specs` directly — they are not in `silk.__all__`. For the end-to-end user workflow see
`docs/design_stage_bridge_user_guide.md`.

### `Spec`

A declared target owned by the `Module` on which it was declared.
Create instances via `Module.declare_spec(...)` — the constructor takes
a `module` back-reference that is most cleanly set by the helper.

```python
s = module.declare_spec(
    name='lna_NF',
    metric='noise_figure',
    op='<=',
    target=2.0,
    units='dB',
    description='LNA noise figure at 1 GHz',
    parent=None,
    calibration=None,
)
```

**Declaration changes:** after successful construction, exactly `target`, `op`,
`metric`, and `key` are read-only. Direct assignment, `setattr`, and attribute
deletion raise `AttributeError` before changing the field, including on copied
Specs. For example, `s.target = 3.0` raises:

```text
Spec.target is read-only after construction; edit the declaration and rebuild the circuit/testbench.
```

To change a target, edit the `target=` argument in the Python
`declare_spec(...)` or `add_spec(...)` declaration, rebuild the circuit/testbench,
then run it to obtain a fresh report. The same rule applies to `Spec(...)`
constructed directly. Assignments to these fields fail immediately.

Runtime state remains mutable: measurements can be recorded or replaced,
`Measurement.passes` is updated during evaluation/reconciliation, `parent` and
`calibration` can be attached later, and `children` can grow. Fields outside
those four retain their existing behavior; the entire object and its nested
collections are not frozen. Testbench ownership and `add_spec` chaining continue
to work.

There is no public spec update, replace, or remove API. Calling `add_spec` or
`declare_spec` again on the same owner does not overwrite an existing spec:
a duplicate key raises `DuplicateNameError` without registering another entry.
The default key is the exact `name`; identical display names coexist only when
they have distinct keys. Supply `key=` to distinguish additional declarations. Private-list edits and bypassing the attribute
guard are not supported spec-editing workflows. Rebuilding creates new live
declarations and reports; it does not rewrite historical reports or evidence.

**Constructor parameters (via `declare_spec`):**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | str | — | Human-readable label; also the default key. Repeated display names require distinct keys on the same owner (use `key=` for additional declarations). |
| `metric` | str | — | Metric name or dotted path (`'net.<name>.<attr>'`, `'device.<name>.<attr>'`). Resolution is deferred to reconcile time. |
| `op` | str | — | One of `'<'`, `'<='`, `'>'`, `'>='`, `'=='`. `KeyError` on any other value when `evaluate()` is called. |
| `target` | float | — | The declared real numeric target value; bool and non-real values raise `TypeError`. Read-only after construction. |
| `key` | str | `None` | Keyword-only stable local identity, defaulting to the exact `name`. Non-string keys raise `TypeError`; empty or whitespace-only keys raise `ValueError`. Read-only after construction. |
| `units` | str | `''` | Free-form string; shown in rendered reports. |
| `description` | str | `''` | Free-form description. |
| `parent` | `Spec` or None | None | Explicit parent. On construction, `self` is appended to `parent.children` via `__post_init__`. |
| `calibration` | `CalibrationBinding` or None | None | Optional write-back binding. See `CalibrationBinding` below. |

**Instance attributes:**

| Attribute | Type | Description |
|---|---|---|
| `name` | str | Declared name |
| `module` | `Module` | Back-reference to the owning module |
| `metric`, `op`, `target`, `key` | — | Read-only after successful construction |
| `units`, `description` | str | Mutable declaration metadata |
| `parent` | `Spec` or None | Explicit parent |
| `children` | `list[Spec]` | Appended to by child specs via `__post_init__` |
| `measurements` | mapping of source/profile keys to `Measurement` | Public source/profile view; runtime root/owner slot keys remain private. Unqualified collisions show the latest observation per public slot. |
| `calibration` | `CalibrationBinding` or None | Optional write-back binding |
| `declared_at` | str | Best-effort `'file.py:lineno'` of the `declare_spec` call site |

**Methods:**

| Method | Description |
|---|---|
| `record_measurement(m)` | Replace the observation in its source/condition/profile slot and record an attempt with the computed verdict. NaN records an unavailable attempt. Does not trigger reconciliation. |
| `record_attempt(attempt)` | Record an immutable `silk.specs.SpecAttempt` without inventing a numeric observation. Repeated IDs deduplicate; conflicting observation facts or a measurement verdict inconsistent with the owning spec raise `ValueError`. |
| `attempt_history(source=None)` | Tuple of immutable snapshots for the selected owner in publication order, optionally filtered by source. |
| `latest_attempt(source=None)` | Latest recorded attempt by original time, with publication order breaking ties; `None` when no attempt exists. |
| `latest(source=None)` | Return the most recent measurement by timestamp, optionally filtered by source. Returns `None` if there are no matches. |
| `evaluate(m)` | Pure function: compare `m.value` against `self.target` using `self.op`. Returns `bool`. Raises `KeyError` on unknown `op`. |
| `add_spec(name, metric_name, op, target, *, key=None)` | Chain forwarder — delegates to the `ModuleTestbench` that created this spec (if any). Raises `RuntimeError` when the spec was not created via `ModuleTestbench.add_spec`. |

### `Measurement`

One observation of a spec's metric value, tagged by source.

```python
m = Measurement(
    source='spice',     # 'behavioral' | 'spice' | 'extraction' | 'user'
    value=2.3,
    passes=None,        # filled by reconcile_specs()
    corner='tt',        # optional
    timestamp=0.0,      # unix seconds; 0 means unknown
    sim_run_id=None,    # optional run correlation ID; campaign ID for aggregates
    notes='',
)
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `source` | `'behavioral' \| 'spice' \| 'extraction' \| 'user'` | — | The source of the observation. Used to key `Spec.measurements` and to gate `CalibrationBinding` writes. |
| `value` | float | — | The measured numeric value. |
| `passes` | `Optional[bool]` | None | Cached result of `op(value, target)`. Filled by `reconcile_specs()` as a side effect so other consumers can see the pass/fail status without re-evaluating. |
| `corner` | str, `ConditionState` or None | None | Process corner, `ConditionState.NONE` when none applies, or unknown. Unscoped string corners use the public `source@corner` slot. |
| `timestamp` | float | 0.0 | Unix seconds; 0 means unknown. `record_measurement()` does not stamp it. Evaluated testbenches supply evaluation time; `latest()` picks the maximum timestamp. |
| `sim_run_id` | `Optional[str]` | None | Optional run correlation ID; equals `aggregate.campaign_id` for aggregate measurements. |
| `notes` | str | `''` | Free-form notes. |
| `temperature` | float, `ConditionState` or None | None | Ambient degrees Celsius, sweep/none, or unknown. |
| `voltage` | float, `ConditionState` or None | None | Supply volts, none, or unknown. |
| `sim_type`, `tool_name`, `origin_version_id` | str or None | None | Analysis, producing tool, and stored-version provenance. |
| `profile_id` | str or None | None | Scopes the live slot by profile, source, corner, temperature and voltage. |

`aggregate=None` holds optional `silk.specs.AggregateMeasurement` attribution.
The frozen record and its nested tuples retain metric/statistic, variation,
selected campaign label, counts, used indices, sample seeds, campaign ID,
full base seed, requested sample count, structure hash and draw-scheme version.
See [aggregate persistence](database.md#aggregate-measurements-schema-v6).

### `CalibrationBinding`

Names a target parameter on a behavioral model. When a reconciliation
run produces a measurement from an allowed source, silk writes
`measurement.value` into
`binding.target_module._behavioral_params[binding.param_name]`.

```python
CalibrationBinding(
    target_module=lna,
    param_name='noise_figure',
    field=None,  # nominal/sigma for Normal, low/high for Uniform
    source_filter=frozenset({'spice', 'extraction'}),   # default
)
```

**Fields:**

| Field | Type | Default | Description |
|---|---|---|---|
| `target_module` | `Module` | — | The module whose `_behavioral_params` will be written to. Does not have to be the same module the spec belongs to. |
| `param_name` | str | — | The key inside `target_module._behavioral_params` to overwrite. **Must already exist** — a missing key raises `KeyError` (declaration error) when the binding has an eligible measurement. |
| `field` | str or None | None | Declaration field target; numeric fields use a copy, Variable fields update the Variable, expressions refuse. Combined updates are validated before any write. |
| `source_filter` | `frozenset[str]` | `frozenset({'spice', 'extraction'})` | Only measurements whose `source` is in this set trigger a write. Default excludes `'behavioral'` to avoid feedback loops. |

**Silent skip conditions** (none raise): no `calibration` on the spec; no eligible measurement (no measurement yet, a failed latest attempt, or a legacy NaN measurement); and `measurement.source not in binding.source_filter`. A usable numeric measurement can calibrate even when its spec verdict is FAIL.

**Refusal conditions** (checked for every binding that has an eligible measurement, before any value is written, so a refused `reconcile_specs()` changes no parameter):

- `binding.target_module._behavioral_params is None` (model not parameterized): `RuntimeError`, naming the parameter, the module and that it declares no parameters.
- `binding.param_name not in binding.target_module._behavioral_params` (key not declared): `KeyError`, naming the module, the parameter and the declared keys.
- Conflicting writes or repeated Variable field bindings: `ConfigurationError`.
- Invalid distribution fields: `VariationError`; expression-valued statistical fields: `NotSupportedError`.

### `SpecAttempt`

Import from `silk.specs`. An immutable snapshot of one completed observation:

```python
from silk.specs import SpecAttempt
attempt = SpecAttempt(source='spice', timestamp=200.0,
                      kind='extractor_error', reason='gain signal unavailable')
spec.record_attempt(attempt)
```

The constructor accepts `source`, `timestamp=0.0`, `kind='measurement'`,
`reason=''`, `attempt_id=None`, `measurement=None`, `test_name=None`,
`report_key=None`, `failure_stage=None`, `requested_conditions=None`,
`proved_conditions=None`, and `aggregate=None` as keyword-only arguments.
Aggregate attribution is allowed only for an unavailable attempt. A measurement attempt
requires a real non-NaN value and boolean verdict; its source and timestamp
must match the snapshot, with no failure reason. Failure kinds require a
reason and no measurement. Explicit attempt IDs must be nonempty strings. Signed infinity remains a usable numeric value.
The default ID is fresh; zero timestamp means unknown. Kinds are `measurement`,
`extractor_error`, `unavailable`, `simulation_error` and `transport_error`.
Source must be nonempty and timestamp finite/nonnegative; invalid combinations raise `ValueError`.

`measurement` returns a defensive copy, so changing it cannot alter history.
Value, verdict, original time and producer provenance are snapshotted on
publication. Storage projections never overwrite these facts. `as_dict()` and
`from_dict(data)` carry selected observation evidence for report codecs.
See [reports](reports.md#reconciliation-placement-identity) for wire fields,
restoration limits and the difference between requested and proved conditions.

### `SpecOutcome`

Produced by `reconcile_specs()` — one per spec placement.

```python
@dataclass
class SpecOutcome:
    spec: Spec
    measurement: Optional[Measurement]
    status: Literal['PASS', 'FAIL', 'UNMEASURED']
    margin: Optional[float] = None
    source_used: Optional[str] = None
    calibrated: bool = False
    notes: str = ''
    owner_path: Optional[str] = '$'
    report_key: Optional[str] = None
    latest_attempt: Optional[SpecAttempt] = None
    last_successful_attempt: Optional[SpecAttempt] = None
```

| Field | Type | Description |
|---|---|---|
| `spec` | `Spec` | The reconciled spec |
| `measurement` | `Measurement` or None | The evaluated observation snapshot; `None` for a failed latest attempt or `UNMEASURED` |
| `status` | `'PASS' \| 'FAIL' \| 'UNMEASURED'` | Reconciliation outcome |
| `margin` | float or None | `abs(measurement.value - spec.target)` when measured, otherwise `None` |
| `source_used` | str or None | Source of the selected attempt or measurement; `None` when unmeasured |
| `calibrated` | bool | `True` iff a calibration write-back fired for this spec |
| `notes` | str | Free-form notes |
| `owner_path` | str or None | Traversal placement, including `$` for root. Old JSON rows without this field load as `None` (unknown); manually constructed outcomes default to root. |
| `report_key` | str or None | Originating test report key captured with the selected measurement; `None` for manual, annotation, unmeasured or legacy evidence without a known report. |
| `latest_attempt` | `SpecAttempt` or None | Selected immutable observation; failed attempts carry a reason and no measurement. |
| `last_successful_attempt` | `SpecAttempt` or None | Latest usable extraction snapshot, including a numeric FAIL verdict; absent when none exists. |
| `display_label` | str | Single-line spec name; placed blocks append their instance path, e.g. `gain (instance u0/u1)`. |
| `placement_label` | str | Single-line suffix such as ` (instance u0/u1)`, omitting the internal `$/` root; empty for root or unknown ownership. |

**How a status is decided:**

- `UNMEASURED`: no attempt or legacy measurement is recorded for the placement.
- A failed latest attempt (including unavailable/NaN) is `FAIL`, with `measurement=None`, `margin=None`, and `calibrated=False`, even on its first attempt. Its historical value is in `last_successful_attempt`, never presented as the current result.
- Legacy numeric evidence without an attempt retains numeric comparison behavior, including NaN failing every operator. SQL and the mutable `Measurement` API still support NaN; this does not make it a successful extraction.
- A recorded **±inf** is compared like any number: `inf` fails `< x` and passes `> x`, and `-inf` does the opposite.

Also has an `as_dict()` method that returns a JSON-friendly nested dict
with `spec`, `measurement`, `status`, `margin`, `source_used`,
`calibrated`, `notes`, `owner_path`, `report_key`, `latest_attempt`, and
`last_successful_attempt`; the nested `spec` includes
its immutable `key`. Used internally by `CircuitReport.save_json()`.

### `SpecReconciliationReport`

Aggregated result of reconciling every spec on a module tree.

```python
@dataclass
class SpecReconciliationReport:
    module: 'Module'
    outcomes: list[SpecOutcome] = field(default_factory=list)
    calibrations_applied: list[str] = field(default_factory=list)
    timestamp: float = 0.0
```

| Field | Type | Description |
|---|---|---|
| `module` | `Module` | The root module that `reconcile_specs()` was called on |
| `outcomes` | `list[SpecOutcome]` | One per spec in the walked tree. Pre-order: root module first, then submodule instances in `_element_mgr.elem_dict` order. |
| `calibrations_applied` | `list[str]` | Human-readable log of calibration side effects, e.g. `'lna.noise_figure = 2.3 (from lna_NF, source=spice)'` |
| `timestamp` | float | Unix seconds of the reconcile run |

**Methods and properties:**

| Member | Description |
|---|---|
| `passed` (property) | True when no outcome is FAIL or UNMEASURED; an empty report passes |
| `pass_count` (property) | Count of outcomes with `status == 'PASS'` |
| `fail_count` (property) | Count of outcomes with `status == 'FAIL'` |
| `unmeasured` (property) | List of outcomes with `status == 'UNMEASURED'` |
| `summary()` | Multi-line human-readable overview |
| `as_dict()` | JSON-friendly dict for `CircuitReport.save_json()` |

### `ComparisonOp` / `MeasurementSource` / `Status`

Type aliases for static-check convenience.

```python
ComparisonOp      = Literal['<', '<=', '>', '>=', '==']
MeasurementSource = Literal['behavioral', 'spice', 'extraction', 'user']
Status            = Literal['PASS', 'FAIL', 'UNMEASURED']
```

### `silk.specs.resolver.resolve_metric(module, metric, *, reader=None)`

Translates a dotted-path metric string into a value from the module's
`net_dict` or `elem_dict`. An optional `reader(metric)` can supply a
`ResolveResult` first; returning `None` falls back to the module lookup. Not re-exported from `silk` —
import from `silk.specs.resolver` directly if you need it.

```python
from silk.specs.resolver import resolve_metric, ResolveResult

result = resolve_metric(my_module, 'net.vout.peak_current')
if result.resolved:
    print(result.value)
else:
    print(f'unresolved: {result.reason}')
```

Supported prefixes: `net.<name>.<attr>` (reads `module.net_dict[name].<attr>`)
and `device.<name>.<attr>` (reads the device object's attribute, unwrapping
`module.elem_dict[name]['device']` when the record is a dictionary).
Returns `ResolveResult(resolved=True, value=float(...))` on success or
`ResolveResult(resolved=False, reason='...')` on failure, where `reason`
is one of `'not_a_dotted_path'`, `'malformed_path'`, `'unknown_prefix'`,
`'net_not_found'`, `'device_not_found'`, `'attr_not_found'`,
`'value_is_none'`, `'value_not_numeric'`.

### `silk.specs.reconciler.reconcile(module, specs, prefer_source=None)`

Evaluates a list of `Spec` objects, selects evidence for each, updates legacy
measurement verdicts, applies eligible calibration writes, and returns a
`SpecReconciliationReport`. Not re-exported from `silk` — the public
entry point is `Module.reconcile_specs()`, which wraps this function
with the tree walk.

```python
from silk.specs.reconciler import reconcile

r = reconcile(module=top, specs=[spec_a, spec_b, spec_c])
r = reconcile(module=top, specs=all_specs, prefer_source='behavioral')
```

### Spec transitions (`ReportDiff.spec_transitions`)

Surfaced by `diff_reports(a, b)`. Each entry is a dict:

| Key | Type | Description |
|---|---|---|
| `spec_name` | str | Display name from the after row, or before row for a removal |
| `owner_path` | str or None | Placement identity; `None` explicitly means unknown |
| `spec_key` | str | Immutable spec key; matched together with `owner_path` |
| `report_key` | str or None | Originating report from the after row if present, otherwise the before row |
| `before_status` | str or None | `PASS` / `FAIL` / `UNMEASURED`, or `None` if the spec did not exist before |
| `after_status` | str or None | Same semantics; `None` if the spec was removed |
| `before_value` | float or None | The evaluated measurement's value in the before report |
| `after_value` | float or None | Same in the after report |
| `source_used` | str or None | Which measurement source was used in the more recent of the two states |

Rows join by `(owner_path, spec_key)`. Duplicate identities within either
input raise `ConfigurationError`, naming both colliding rows and how to resolve
the ambiguity. This includes identical duplicates and duplicate legacy rows.
A unique unknown owner can match only another unknown owner with the same key;
an unknown/known pair produces removal and addition. Ordering is unknown owners
first, then known paths lexically, then spec keys lexically.

A transition is listed when status, value, source, or the complete
`latest_attempt` or `last_successful_attempt` evidence differs. A new attempt ID,
timestamp or attempt provenance can therefore produce a transition even when
status and value are unchanged. Top-level display-name or report-key changes
alone do not create a transition when that compared evidence is unchanged.
Changing only a target, operator, metric, units or description also creates no
transition. Rendered by `str(diff)` as:

```
Spec transitions:
  lna_NF [$; key=lna_NF]: PASS (2.0) -> FAIL (2.3)
```

### Database persistence (`silk.db.circuit_db`)

The database persists sealed structure/spec identity, declarations, metric
observations, first-class spec measurements, current sets, characterization
profiles, and promotion lineage. `Module.create_circuit_version()` stores declarations;
`store_report()` writes validated measurements; `get_stored_specs()` queries
history; and `restore_spec_measurements()` attaches compatible observations to
an equivalent live rebuild.

`net_currents` stores one row per aggregate (`pin_id=''`) or pin
(`pin_id='inst.port'`), grouped into **current sets** that each carry a stable
`current_set_id`. A central writer inserts each complete set atomically and
enforces locks; re-storing at the same selection key — `(version, corner)`
unscoped, or `(version, profile, corner, temperature, voltage)` when
profile-scoped — marks the previous set superseded rather than overwriting it.
Current rows hold values, source, vector and timestamp; their parent current set
holds shared analysis, window, conditions, tool, results path and structure
provenance. Use the Module methods rather than direct SQL.

See [database.md](database.md) for the file-version matrix, full lifecycle,
record types, characterization profiles, restore failures, the schema-v1
migration path (`silk.db.migrate_v1_to_v2`), and JSON projection scope.

---

## silk.lint — Analog lint and coverage framework

Rule-based static and post-simulation checker for analog/mixed-signal
designs. Common entry points are re-exported from `silk` (`from
silk import Severity, RuleRegistry, ...`). Import the base `Violation` from
`silk.lint.violations` and location classes from `silk.lint.locations`. The default catalog has 22
rules across four categories: interface (5), domain (3), stress (1),
structural (13, including BUS-002). One additional structural rule is opt-in.

For the end-to-end user workflow see `docs/lint_and_coverage_user_guide.md`.

### `Severity`

```python
from silk import Severity

Severity.INFO     # 10
Severity.WARNING  # 20
Severity.ERROR    # 30  — fails CI gates by default
```

`Severity` is an `IntEnum`, so comparisons (`sev >= Severity.ERROR`) and
sorting work as expected. `Severity.from_string('ERROR')` and
`sev.to_string()` round-trip through reports.

---

### `Rule` / `RuleRegistry`

`Rule` is the abstract base class every lint rule subclasses. The
`RuleRegistry` holds a list of rule instances, dispatches them against a
scope, and produces a `RegistryRunResult` with violations plus a
per-(rule × scope) visit log.

```python
from silk import RuleRegistry, Severity

registry = RuleRegistry.default()         # built-in 22-rule catalog
registry = RuleRegistry.default(include_optional=True)   # + STRUCT-009 (opt-in)
result   = registry.run(top_module)       # returns RegistryRunResult
errors   = [v for v in result.violations if v.severity >= Severity.ERROR]

# Filter to one category
result_struct = registry.run_category('structural', top_module)
```

`Rule(config=None, severity_override=None)` accepts per-instance configuration
and severity overrides.

**Rule class attributes:** subclasses must set `rule_id`, `category`,
`applicable_to` and `rationale`. The other attributes have inherited defaults
and may be overridden:

| Attribute | Type | Description |
|---|---|---|
| `rule_id` | str | Stable identifier, e.g. `'INT-001'`, `'STRUCT-003'` |
| `category` | str | One of `'interface'`, `'domain'`, `'stress'`, `'structural'`, `'isolation'`, `'solver'` |
| `default_severity` | `Severity` | Defaults to `Severity.WARNING`; can be overridden per-instance |
| `applicable_to` | tuple[type, ...] | Scope types the rule visits (e.g. `(Module,)`) |
| `rationale` | str | Prose explaining why the rule exists and how to fix violations |
| `requires_simulation` | bool | Defaults to `False`; if True, rule is skipped when no simulation data is attached |
| `config_schema` | dict[str, type] | Optional rule-config field schema; defaults to `{}` |

**Rule instance method**:

```python
def check(self, scope) -> Iterator[Violation]: ...
```

**`RuleRegistry` methods:**

| Method | Description |
|---|---|
| `RuleRegistry(rules=None, exclusions=None, waivers=None)` | Construct with an explicit rule list |
| `RuleRegistry.default()` | Classmethod: return a registry loaded with the built-in catalog |
| `add_rule(rule)` | Append a rule instance |
| `load_exclusions(path)` | Load a YAML exclusion file; appends to registry |
| `load_waivers(path)` | Load a YAML waiver file; appends to registry |
| `is_excluded(instance_path)` | Return the matching `ScopeExclusion` if any, else None |
| `rules` | Property: copy of the rule list |
| `run(scope, profile='default', categories=None, has_simulation_data=False)` | Dispatch rules; return `RegistryRunResult` before waiver/exclusion filtering. Only `profile='default'` is supported; use `generate_report(module, registry=registry)` to apply loaded filters |
| `run_category(category, scope, **kwargs)` | Convenience wrapper for a single category |

**`Rule.simulation_data_usable(scope)`** — override on a rule that sets
`requires_simulation` to decline a *kind* of result it cannot read (returning
`False` records the visit as `SKIPPED_NO_SIM` instead of dispatching `check`).
`has_simulation_data` answers "was this simulated at all?"; this answers "and
is it the kind of result I read?". STR-001 uses it to decline AC and noise
plots. Defaults to `True`.

**`module_has_simulation_data(module)`** (module-level function) — the shared
predicate for the `has_simulation_data=` argument: `True` when the module
carries behavioral results (`_behavioral_mode`) **or** SPICE results
(`sim_results_loaded`). `generate_report()` and `check_junction_overrange()`
both call it, so a simulation-requiring rule such as STR-001 runs on either
data path rather than being recorded as `SKIPPED_NO_SIM`.

**`VisitEvent` fields** (one per dispatched rule × scope):

| Field | Type | Description |
|---|---|---|
| `rule_id` | str | Rule that was dispatched |
| `location` | `ViolationLocation` | Scope location (typically `ModuleLocation`) |
| `outcome` | `VisitOutcome` | `PASS`, `VIOLATION`, `INCOMPLETE`, `EXCLUDED`, or `SKIPPED_NO_SIM` |
| `timestamp` | float | Unix time when the rule ran |
| `simulator` | str \| None | Simulator name if relevant |
| `context` | dict | Rule-specific data; STR-001 carries `stress_coverage` |
| `metric_results` | dict | Functional coverage is not implemented |

**`VisitOutcome`** enum members: `PASS`, `VIOLATION`, `INCOMPLETE`, `EXCLUDED`, `SKIPPED_NO_SIM`.

**`RegistryRunResult`** dataclass: `violations: list[Violation]` and
`visit_log: list[VisitEvent]`.

> **Coverage note**: `run_category()` only logs visits for rules in the
> filtered category. To build an accurate `CoverageReport`, drive
> `CoverageReport.from_visit_log(...)` from an **unfiltered**
> `registry.run(...)` result, or it will under-count checked rules.

---

### `Violation` / `InterfaceViolation` / `JunctionViolation`

Base and subclasses that lint rules emit. All three are plain
dataclasses, JSON-serializable through `LintReport.save_json()`.

**`Violation` fields** (`silk.lint.violations.Violation`):

| Field | Type | Description |
|---|---|---|
| `rule_id` | str | Emitting rule |
| `severity` | `Severity` | IntEnum severity |
| `message` | str | Human-readable description |
| `location` | `ViolationLocation` | Tagged-union location |
| `context` | dict | Rule-specific extras (optional) |

`InterfaceViolation` and `JunctionViolation` are documented in the
`check_interfaces` / `check_voltage_domains` / `check_junction_overrange`
section above; they subclass `Violation` and expose convenience properties
derived from `location` and `rule_id`.

---

### Violation locations (tagged union)

All lint violations carry a `location` field, an instance of a
`ViolationLocation` subclass. Every subclass implements `describe()`
returning a short string fingerprint used for diffing.

| Class | Fields | `describe()` example |
|---|---|---|
| `NetLocation` | `net_name: str`, `port_names: list[str]` | `'net:vdd'` |
| `PortLocation` | `port_name: str`, `module_name: str` | `'port:amp.vout'` |
| `InstanceLocation` | `instance_path: str` | `'instance:top.ldo.mn1'` |
| `TerminalLocation` | `instance_path: str`, `terminal: str`, `terminal_pair: tuple[str,...]` | `'terminal:top.ldo.mn1.d'` |
| `ModuleLocation` | `module_name: str` | `'module:ldo_nmos'` |

---

### `Waiver` / `WaiverFile` / `WaiverMatcher` / `apply_waivers`

File-based waiver system. Waivers suppress specific violations with a
required justification; they are first-class audit artifacts — every
waived violation is still reported in `LintReport.waivers_applied`.

```python
from silk import Waiver, WaiverMatcher, WaiverFile, apply_waivers

w = Waiver(
    rule_id='INT-001',
    matcher=WaiverMatcher(location_type='net', fields={'net_name': 'bias_out'}),
    reason='Dotted-OR of two matched op-amp outputs',
    author='mec',
)
unsuppressed, suppressed = apply_waivers(violations, [w])
```

**`WaiverMatcher`** has `location_type` (`'net'`, `'port'`, `'instance'`,
`'terminal'`, or `'module'`) and a `fields: dict` of expected
attribute values on the matched location object. All fields must match.

**`Waiver`** has `rule_id`, `matcher`, `reason` (non-empty, raises
`ValueError` otherwise), `author`, and optional `expires: date`. Expired
waivers stop matching.

**`WaiverFile.load(path)`** parses a YAML file of the form:

```yaml
waivers:
  - rule_id: INT-001
    matcher:
      location_type: net
      fields: {net_name: bias_out}
    reason: 'Dotted-OR of matched outputs'
    author: mec
    expires: 2026-12-31
```

**`apply_waivers(violations, waivers)`** returns
`(unsuppressed, suppressed)` — two parallel lists of `Violation`.

---

### `ScopeExclusion` / `ExclusionFile`

Exclusions filter reported violations for matching instance paths and their
descendants. `generate_report()` runs the rules first, then filters matching
violations; it derives coverage from the original visit log. Exclusions do not
prevent rule execution or change those visits to `EXCLUDED`. Use them to omit
findings for blocks maintained by another team. Instance exclusions match
`instance_path`, otherwise `module_name`; net locations provide neither
and are not filtered by instance exclusions.

```python
from silk import ScopeExclusion

excl = ScopeExclusion(
    target_type='instance',
    target_path='top.sram_macro',
    reason='Macro findings are maintained by another team',
    owner='vendorX',
)
excl.matches_instance_path('top.sram_macro.bit_cell')  # True
```

`ExclusionFile.load(path)` parses YAML of the form:

```yaml
exclusions:
  - target: {type: instance, path: top.sram_macro}
    reason: 'Macro findings are maintained by another team'
    owner: vendorX
```

`RuleRegistry.load_exclusions(path)` appends loaded exclusions to the
registry.

---

### `NoConnect` and `Module.add_no_connect`

The schematic "no-connection" flag: declares a net as
intentionally unconnected, suppressing both `STRUCT-003 FLOATING_NET`
(a net with zero connections) and `STRUCT-011 SINGLE_CONNECTION_NET`
(a net with exactly one) for that net, and surfacing in lint reports.

```python
from silk import Module, NoConnect

m = Module(...)
m.add_no_connect('probe_out', reason='differential test tap')
# Equivalent effect: registers a PortBinding with kind='lint_marker'
```

`NoConnect(reason)` requires a non-empty reason string; raises
`TypeError` or `ValueError` otherwise. The class exposes
`is_lint_marker = True` and the `reason` attribute.

**`NoConnectPort(instance, port, reason)`** (`silk.lint_markers`) is the
per-pin counterpart, registered via `Module.add_no_connect_port`; it
suppresses `STRUCT-001` for that one pin and is listed in
`LintReport.no_connect_ports`. A whole-bus `add_no_connect_port()` call returns
a list of per-bit markers; scalar and per-bit calls return one marker.

Both registration methods dedupe by target: a second
`add_no_connect`/`add_no_connect_port` call on the same net or
`(instance, port)` — including a bus bit already covered by an earlier
whole-bus call — with the identical `reason` is a no-op that returns the
original marker (`Module._no_connects` / `_no_connect_ports` is not
extended); a *different* `reason` raises `DuplicateNameError` naming
both instead of silently registering a second marker on the same
target. `find_stale_markers` produces one row per registered marker found stale.

**`StaleMarker` / `find_stale_markers(module)`** (`silk.lint_markers`):
one frozen record per marker whose target is wired after all — a
`NoConnect` on a net with two or more real bindings (zero is STRUCT-003,
one is STRUCT-011: both are the marker's job), or a `NoConnectPort` on a
pin that carries a net. Fields `kind` (`'net'`/`'port'`), `reason`,
`net_name`, `instance`, `port`, `wired_to`; `target()` renders
`net probe_a` / `port u1.en`, `describe()` a one-line diagnosis.
`generate_report` runs this pass and fills `LintReport.stale_markers`.

---

### `LintReport` / `LintDiff`

Top-level report container: metadata, violations, and applied waivers.
Supports JSON, HTML, and JUnit XML output. Two reports generated against
the same `(silk_version, pdk_name, pdk_version, module_name)` can be
diffed; mismatched `context_hash` values raise `ValueError`.

```python
from silk import LintReport, RuleRegistry, ReportMetadata

result = RuleRegistry.default().run(top_module)
report = LintReport(
    metadata=ReportMetadata(
        silk_version='0.2.0',
        pdk_name='sky130',
        pdk_version='2024.3',
        module_name='ldo_top',
    ),
    violations=result.violations,
)
report.save_json('lint.json')
report.save_html('lint.html')
report.save_junit_xml('lint.xml')

counts = report.summary()
# {'errors': 1, 'warnings': 3, 'infos': 0, 'waived': 0, 'stale_markers': 0}
```

**`LintReport` fields:**

| Field | Type | Description |
|---|---|---|
| `metadata` | `ReportMetadata` | Schema version, silk/PDK version, context hash |
| `violations` | list[Violation] | All unsuppressed violations |
| `waivers_applied` | list | What `generate_report` suppressed: the waived `Violation` objects (hand-built reports may hold `Waiver` objects). JSON: `waivers_applied` lists the waivers, `waived_violations` the suppressed violations. |
| `waivers_matched` | list[Waiver] | Waivers that matched at least one violation |
| `no_connects` | list | `(instance_name, net_name, NoConnect)` tuples from `Module.add_no_connect`. Surfaced in HTML and JSON output. |
| `no_connect_ports` | list[NoConnectPort] | Per-pin markers from `Module.add_no_connect_port`. JSON lists them with the nets under `no_connects` (`kind: port`); the HTML "Intentionally Unconnected" table has a Port column. |
| `stale_markers` | list[StaleMarker] | Markers whose target is wired after all (see `find_stale_markers`). JSON key `stale_markers`; HTML "Stale Markers" table with WARNING rows. |
| `stress_coverage` | dict | Serialized device stress coverage; default `{}` |

**`LintReport` methods:**

| Method | Description |
|---|---|
| `summary()` | Return `{errors, warnings, infos, waived, stale_markers}` counts |
| `save_json(path)` / `load_json(path)` | Round-trip through JSON (classmethod load) |
| `to_dict()` / `from_dict(data)` | Pure-dict serialization |
| `save_html(path)` | Write a self-contained HTML report |
| `save_junit_xml(path)` | Write a JUnit-XML report: one `<testcase>` per rule with findings, plus STR-001 coverage when present; ERRORs fail and incomplete stress coverage can skip |
| `diff(other)` | Return `LintDiff(added, removed)`; raises on context_hash mismatch |
| `compliance_matrix()` | Return `{rule_id: {violations: int, waivers: int}}` |

**`LintDiff`** dataclass: `added: list[Violation]`, `removed:
list[Violation]`. Violations are matched by
`(rule_id, location.describe(), message)`.

---

### `ReportMetadata`

```python
from silk import ReportMetadata, SCHEMA_VERSION

meta = ReportMetadata(
    silk_version='0.2.0',
    pdk_name='sky130',
    pdk_version='2024.3',
    module_name='ldo_top',
)
# Auto-populated fields:
meta.generated_at     # ISO-8601 UTC string
meta.schema_version   # == SCHEMA_VERSION ('1.0')
meta.context_hash     # SHA-256 prefix of (silk_version | pdk_name | pdk_version | module_name)
```

`SCHEMA_VERSION` is the module-level constant `'1.0'`. Reports with
mismatched `context_hash` refuse to diff.

---

### `CoverageReport` / `RuleCoverageReport` / `RuleCoverageNode`

Hierarchical per-rule coverage derived from a `RegistryRunResult.visit_log`.
Functional coverage is not implemented. `from_visit_log()` builds one root
node with no children; trees can also be constructed with child nodes.

```python
from silk import RuleRegistry, CoverageReport, ReportMetadata

result = RuleRegistry.default().run(top_module)
cov = CoverageReport.from_visit_log(
    metadata=ReportMetadata(
        silk_version='0.2.0', pdk_name='sky130',
        pdk_version='2024.3', module_name='ldo_top',
    ),
    visit_log=result.visit_log,
)
cov.save_json('coverage.json')

# Per-rule coverage percentage (checked / (checked + unchecked))
pct = cov.rule_coverage.root.percentage('INT-001')
```

**`RuleCoverageNode` fields:**

| Field | Type | Description |
|---|---|---|
| `module_name` | str | Module this node represents |
| `local` | dict[str, tuple[int,int,int]] | `rule_id → (checked, excluded, unchecked)` |
| `children` | list[RuleCoverageNode] | Child module nodes; empty in trees built by `from_visit_log()` |

**`RuleCoverageNode` methods:**

| Method | Description |
|---|---|
| `aggregate` (property) | Recursive sum of `local` + all children's aggregates |
| `percentage(rule_id)` | `100 * checked / (checked + unchecked)` for one rule in this node's local counts only; use `aggregate` for subtree counts. 100% if the rule is present and the denominator is 0; 0% if the rule is absent |
| `to_dict()` / `from_dict(d)` | JSON-safe serialization |

**`RuleCoverageReport`** wraps a single `root: RuleCoverageNode`.

**`CoverageReport` fields:**

| Field | Type | Description |
|---|---|---|
| `metadata` | `ReportMetadata` | Versioning / context hash |
| `rule_coverage` | `RuleCoverageReport` | Hierarchical rule coverage tree |
| `stress_coverage` | dict | Serialized device stress coverage; default `{}` |
| `functional_coverage` | dict or None | Reserved for functional coverage; currently None |

**`CoverageReport` methods:**

| Method | Description |
|---|---|
| `CoverageReport.from_visit_log(metadata, visit_log, module_name='')` | Classmethod: build from a `RegistryRunResult.visit_log` |
| `save_json(path)` / `load_json(path)` | JSON round-trip (classmethod load) |
| `to_dict()` / `from_dict(data)` | Pure-dict serialization |
| `save_html(path)` | Write a styled HTML scorecard with per-rule coverage breakdown |
| `diff(other)` | Return `CoverageDiff`; raises `ValueError` on `context_hash` mismatch. `CoverageDiff` is importable from `silk.lint.reports` (not re-exported from `silk`). |

Visit outcomes map to the tuple columns as:
`PASS | VIOLATION → checked`, `EXCLUDED → excluded`, `SKIPPED_NO_SIM | INCOMPLETE → unchecked`. STR-001 visits with `stress_coverage` context count PDK catalog devices instead: full → checked, partial + none → unchecked.

---

### `help_rule(rule_id)` / `list_rules()`

Interactive exploration of the shipped rule catalog.

```python
from silk import help_rule, list_rules

list_rules()
# ['INT-001', 'INT-002', 'INT-003', 'INT-004', 'INT-005',
#  'DOM-001', 'DOM-002', 'DOM-003', 'STR-001',
#  'STRUCT-003', 'STRUCT-011', 'STRUCT-001', 'STRUCT-002', 'STRUCT-004',
#  'STRUCT-005', 'STRUCT-006', 'STRUCT-007', 'STRUCT-008',
#  'BUS-002', 'STRUCT-012', 'STRUCT-013', 'STRUCT-010']   # registry order (22 rules); STRUCT-003 is registered first
list_rules(include_optional=True)
# ... + ['STRUCT-009']   # opt-in rules (OPTIONAL_CATALOG)

print(help_rule('INT-001'))
# INT-001 — DriverConflictRule
# Category: interface
# Default severity: error
# Requires simulation: False
# In default catalog: yes
#
# A net with more than one OUT-direction port is almost always a
# wiring mistake. ...
```

`help_rule` resolves every shipped rule, including the opt-in ones
(`In default catalog: no (opt-in: RuleRegistry.default(include_optional=True))`),
and raises `KeyError` for IDs that do not exist at all.
`silk.lint.rules` exposes the three views: `DEFAULT_CATALOG()`,
`OPTIONAL_CATALOG()` and `FULL_CATALOG()`.

---

### `export_schema(source)` / `module_schema(module)` / `export_schema_json(source)`

Generate machine-readable device/module specifications.
`export_schema(source, include_metadata=True)` includes `_meta` and, when corners are declared, `_library`
metadata alongside device entries; pass `include_metadata=False` to omit them.
`export_schema_json(source, indent=2, **kwargs)` forwards export options.
See [schema-export.md](schema-export.md) for the output contract.

```python
from silk import export_schema, module_schema, export_schema_json
schema = export_schema(pdk)           # dict of all PDK devices
schema = export_schema(elements)      # dict of all ideal elements
mod    = module_schema(my_module)     # interface of a single Module
json_str = export_schema_json(pdk)    # same as export_schema but JSON string
```

Schema excerpt per device (ellipses stand for additional entries):

```text
{
  "nmos1p8": {
    "name": "nmos1p8",
    "is_pdk_device": true,
    "pdk_model": "sky130_fd_pr__nfet_01v8",
    "spice_prefix": "XM",
    "category": "transistor",
    "pins": [{"name": "d", "description": "drain"}, ...],
    "parameters": [{"name": "w", "default": 1.0, "unit": "u", "min": "0.42", "max": "-1"}, ...]
  }
}
```

---

## silk.utils — Python identifier validation

### `require_python_identifier(value, context)`

```python
from silk.utils import require_python_identifier
require_python_identifier('nmos1p8', 'device type')  # returns 'nmos1p8'
```

Returns the unchanged string when `str.isidentifier()` is true and
`keyword.iskeyword()` is false; otherwise raises `ValueError` naming the
context and value. Non-strings fail too. Unicode identifiers and soft
keywords are accepted. Code generation uses the same identifier rules.

File-backed catalogs apply the same rule at load: PDK device, pin and
parameter keys (including aliases) raise `PDKError`; `ElementLibrary`
and source metadata raise `CircuitError`. Errors name the file, device or
source where applicable, and field/key. Ideal `paramConstraints` names and
source namespaces, aliases, waveform kinds/classes, pins, scalar parameters,
references, expressions and parameter metadata keys are checked too.
Composed flat factory names (alias + scalar or waveform kind) must also
avoid Python keywords.
Ideal-element `source_class` references are checked as Python class names.
SPICE model names and descriptive labels are not Python names.

## silk.pdk

### `load_pdk(name, *, pdk_root=None, library_location=None, corner=None, user_units=None)`

```python
from silk import load_pdk

pdk = load_pdk('sky130')                        # resolves catalog/lib/corners from the manifest
pdk = load_pdk('sky130', pdk_root='/opt/pdk')   # override the PDK_ROOT setting
pdk = load_pdk('sky130', user_units='si')       # override manifest user_units
```

Loads a PDK from its manifest (`silk/pdk/data/<name>.pdk.yaml`) via the
registry. `name` accepts any registered PDK name or alias (sky130 aliases:
`'skywater130'`, `'skywater_130'`). All keyword args are optional overrides;
`library_location=` overrides the main model-library path; extra and
per-corner includes still resolve against the PDK root. The root defaults to
the `PDK_ROOT` setting: environment variable, then settings file, then `~/pdk`.
Shipped manifests use micrometres for W/L (`w=4` means 4 µm);
`user_units='si'` accepts metres. Bare `BasePDK` construction defaults to
`UNIT_PDK` (native PDK quantities), whereas `load_pdk` uses the manifest default.
Loading a catalog and generating a deck do not require installed models. A run
that needs missing models raises `SimulationError` with the library path and
installation advice. An unknown name raises `PDKRegistryError` (a `ValueError`).

Discovery: `silk.pdk.available()` → registered PDK names; `silk.pdk.manifest(name)`
→ the parsed manifest. See `docs/adding-a-pdk.md` to add a new PDK.

---

### sky130 device factories

```python
from silk.pdk import load_pdk

pdk = load_pdk('sky130')
```

**Device factory methods** (each returns an `AnalogElement`). Only `name` may
be passed positionally — pins and parameters are **keyword-only**, e.g.
`pdk.nmos1p8('mn', d='out', g='in', s='0', b='0', w=4.0, l=0.15)`:

| Method | Description |
|---|---|
| `pdk.nmos1p8(name, *, d, g, s, b, w=1.0, l=0.15, m=1, ...)` | sky130 1.8 V NMOS |
| `pdk.pmos1p8(name, *, d, g, s, b, w=0.42, l=0.15, m=1, ...)` | sky130 1.8 V PMOS |
| `pdk.nmos5v(name, ...)` | sky130 5 V NMOS |
| `pdk.pmos5v(name, ...)` | sky130 5 V PMOS |
| `pdk.npn_std(name, ...)` | sky130 standard NPN BJT (also: `pdk.npn_11v` for 11 V variant) |
| `pdk.pnp_large(name, ...)` | sky130 large PNP BJT (also: `pdk.pnp_small`) |
| `pdk.res_hpo(name, ...)` | sky130 high-poly-resistance resistor (also: `pdk.res_xhpo`, `pdk.res_generic_nd`, `pdk.res_generic_pd`, `pdk.res_iso_pw`) |
| `pdk.cap_mim1(name, ...)` | sky130 MIM capacitor (also: `pdk.cap_mim2`, `pdk.cap_vpp_11p5x11p7`) |

**Parameter descriptors** — for creating `Variable` objects:

```python
w_var = pdk.nmos1p8.w.variable(name='w_n', value=1.0, bounds=(0.42, 20.0))
l_var = pdk.nmos1p8.l.variable(name='l',   value=0.15, bounds=(0.15, 1.0))
```

---

### `BasePDK`

Catalog-backed PDK class. Most additional PDKs need a manifest and device
catalog; subclass it only when specialized behavior is required.

`BasePDK` loads the device catalog and converts caller-supplied quantities
from `user_units` to PDK units during emission.

#### `device_geometry_um(device, pdk=None) -> (W_um, L_um, M) | None`

```python
from silk.pdk import device_geometry_um
W_um, L_um, M = device_geometry_um(mod.elem_dict['M1']['device'], pdk)
# or, bound to a PDK:
W_um, L_um, M = pdk.device_geometry_um(device)
```

The one accessor for device channel geometry in canonical micrometres;
`silk.layout` reads every W/L through it. Understands both device
representations — a factory-built `AnalogElement` (`pdk.nmos1p8(...)`, which
carries its own scales, so `pdk` is optional) and an `add_element`-style
`PDKDevice` (for which `pdk` is required, because the author→PDK factor lives
on the PDK instance). Caller-supplied numeric quantities are converted from the author's
unit; numeric strings and catalog defaults are interpreted in PDK units; then the
parameter's `pdk_unit:` takes the value to µm. Returns `None` when the device
declares no W/L (passives, ideal elements), or a W/L is an unresolved SPICE string
or plain unassigned Variable. The same applies to a declared multiplier; expression evaluation errors propagate.
`M=1` is used only when no multiplier parameter exists. Raises `ValidationError` for a W/L with no recognised
`pdk_unit:` and `TypeError` for an unknown device shape — silk never guesses
the unit of a length.

#### `PDKDevice.junction_ratings` / `.junction_ranges` / `.junction_ratings_provenance`

`pdk._devices[name].junction_ranges` is the `{'<pair>_max': (lo, hi)}` dict
of signed operating ranges (volts, `Vxy = V(x) - V(y)`) STR-001 scores the
signed pair voltage against (empty for an unrated device): a catalog
magnitude `m` appears as `(-m, m)`, a catalog `{min, max}` mapping as
written (sky130 `nmos1p8` `Vbs_max` is `(-1.95, 0.3)`). `.junction_ratings`
is the sign-blind `{'<pair>_max': volts}` view of the same ratings
(`max(|lo|, |hi|)`) kept for consumers that score `|V|`;
`silk.pdk.basepdk.parse_junction_rating(key, raw)` is the one place either
catalog form is normalised. `.junction_ratings_provenance`
is the parsed `JunctionRatingsProvenance(source, heuristic)` recording where
each number comes from, or `None` when the catalog declares none.
`silk.pdk.validate.check_junction_provenance(pdk)` — also part of
`validate_pdk`'s static pass — returns the `unsourced_rating` /
`bad_provenance` / `rating_convention` issues for a catalog that breaks the
convention in `docs/adding-a-pdk.md` ("Junction ratings").

---

### `UNIT_PDK` / `UNIT_SI`

User-units constants for the `user_units=` constructor argument (see docs/adding-a-pdk.md §Units).

```python
from silk.pdk import UNIT_PDK, UNIT_SI
```

---

## silk.simulation

### `validate_simulation_setup(module)`

`silk.simulation.validate_simulation_setup(module)` checks, without running or
changing anything, that a simulation setup passes silk's static admission
checks. It returns `None` when no static refusal is found; otherwise it raises
the **first** refusal, with the same exception class and message the run raises
for the same input. A clean return does not guarantee convergence or successful
execution.

*Input:* the assembled top-level module with its effective simulator and
settings. For a testbench use `ModuleTestbench.validate_simulation_setup()`,
which resolves the bench's effective simulator, ambient TEMP, supply and
options as `run()` would (inheriting from the live owner) on a private view,
then calls this function. The check runs on a private copy of the graph structures it reads, because
reading them refreshes placement pins and definition records in place; the live
module and its hierarchy are never refreshed or synchronized by the call, and a
selected nested bench (`get_test(..., instance=...)`) is checked without
entering its placement scope. The bench method has adapter-only parts outside
the shared function, in this order: a DC or Noise analysis whose source the
bench lacks raises `SweepSourceError`, as `run()` does first; effective ambient
TEMP is then resolved as a run does (`TypeError` or `ValueError` for an invalid
value, `SimulationConfigurationError` for duplicate TEMP declarations or an
invalid literal temperature in custom text, and `NotSupportedError` for managed
TEMP or a temperature sweep on an unsupported backend or beside custom text that
may set a temperature); and the owner's device-capture switches
(`save_device_currents`, `save_device_params`, `omit_device_internal_nodes`) are
taken onto the view. These precede the list below. An unwired bench (no `add_module_under_test`) raises
`SimulationConfigurationError`; that is an input error of the bench method and
is outside the same-error-as-the-run guarantee, since `run()` raises a
`ValueError` there.

*Checks covered, in the order the run applies them:*

1. behavioral only: ambient temperature (`set_temperature`) refuses with
   `NotSupportedError`; for every simulator, a configured supply voltage must
   resolve to a source (`SimulationConfigurationError`);
2. the existing topology walk refuses formal port names `0` and
   case-insensitive `gnd` with `CircuitError`, including entered children and
   behavioral modeled-leaf interfaces. Replaced contents of modeled leaves
   are not entered. Rename the formal to `vss` and wire it to the parent's
   ground net; actual ground connections retain their current meaning.
   The walk then checks source ground connectivity (`SimulationError`): a
   source whose wiring group has no connection to ground, as for generated SPICE decks (see
   [known issues](known_issues.md#source-ground-connectivity));
3. behavioral only, the engine's read-only preparation (`NotSupportedError` or
   `SimulationError`): custom SPICE text and device-capture switches,
   analyses other than transient and DC (or none declared), transistors with
   no behavioral model above them, empty unmodelled blocks, unsupported
   sources (G/F/H/B), series capacitors, resistive dividers and ambiguous R/L
   paths, static source-reference refusals, and a DC sweep of ambient
   temperature or of a source that does not exist.

*Not covered:* installed PDK models, compiled plugins, deck freshness or
prepared-deck state, refusals decided during a run (they depend on what models
return), electrical solvability, and feedback between blocks. The engine's
mixed sample/whole-array loop refusals are
decided when it schedules blocks; those checks are statically decidable but
are not part of this check today. DC feedback is diagnosed after the point's
model calls. SPICE decks supplied externally or altered by fault injection are not
judged from the live module by the run; the public call judges the live module.

*Stable promise:* the name, the input, raise-first-or-`None`, no side effects
(source and `Variable` values, results, notes, provenance, request inventories,
run state and output files are unchanged; no model, waveform or selector
callable is called; nothing scales with the transient length), the listed
checks, and the error classes. Messages are silk's and may be reworded;
structured attributes on the errors (such as `error.topology`) are not
promised.

### Operating-condition proofs

```python
from silk.simulation import has_no_pdk_device, has_no_supply

has_no_pdk_device(module)  # -> bool
has_no_supply(module)      # -> bool
```

Each takes a `Module` and returns a `bool`. `True` is a proof; `False` means
"not proven" (the condition is unknown), never "proven the opposite". Both walk
the whole hierarchy. silk's own launch path uses these same functions, and a
consumer that runs a silk-generated deck itself can use them to declare a
condition *none*.

- `has_no_pdk_device(module)`: no PDK device anywhere, so the deck carries no
  PDK `.lib`/`.include` and no process corner reached the simulator (corner
  *none*; no PDK files are needed). Any element not recognised as ideal, and any
  `add_custom_line` text, makes it `False`. It is the inverse of the netlister's
  `module_uses_pdk_devices`.
- `has_no_supply(module)`: supply *none*. `True` only when all hold: no
  `set_supply_voltage` configured (own or inherited); `has_no_pdk_device` is
  `True`; no net is classified `PortRole.POWER`; every independent voltage or
  current source carries a waveform or has a literal-zero `dc` (a
  parameter-reference `dc` fails; controlled sources are ignored).

### Analysis types

```python
from silk.simulation import Transient, AC, DC, Noise, Op

Transient(step, stop, start=0, maxstep=None, uic=False)
AC(variation, points, fstart, fstop)
DC(source, start, stop, step)
Noise(output, ref, source, variation, points, fstart, fstop)
Op()
```

`DC.source` names an independent source, or `'temp'` for a SPICE temperature
sweep (behavioral simulation refuses ambient temperature sweeps).
`Noise.output`/`ref` are the differential output nodes; `source` is the
input source used for input-referred noise.

`AC`/`Noise` use `variation='dec'|'oct'|'lin'`: `points` means points per
decade/octave for logarithmic sweeps, total points for linear sweeps.

Analysis constructors validate the sweep before netlisting and raise an
actionable `ValueError` for invalid geometry:

- `Transient` requires finite `step > 0`, `stop > start >= 0`, and,
  when supplied, finite `maxstep > 0`.
- `AC` and `Noise` require `points` to resolve to a positive whole number,
  finite positive frequencies, and `fstop >= fstart`. DEC additionally
  requires at least one whole logarithmic step, including after SPICE value
  formatting: `points * log10(fstop / fstart) >= 1`. A sub-step request
  reports the minimum points per decade; increase the points, widen the
  range, or use LIN. Equality permits a single-frequency analysis for LIN
  and OCT, but is refused for DEC.
- `DC` requires finite values and a non-zero `step` whose sign moves from
  `start` toward `stop`. Descending sweeps use a negative step; equal
  `start` and `stop` permits a single-point sweep.

Point counts accept integers, whole-valued floats, and SPICE-formatted strings:
`10`, `10.0`, `"10"`, and `"1k"` are valid. Silk normalizes the value to
a built-in integer before emission, so `"1k"` emits as `1000`; fractional
counts such as `10.5` or `"10.5"` are rejected.

Time, sweep-value, and frequency fields likewise accept Python numbers or
SPICE-formatted strings such as `"1n"`; booleans are rejected as numeric
values. These checks enforce mathematical validity only. Silk does not cap a
valid analysis's duration, point count, or sweep size.

`Transient.maxstep` (optional) caps the simulator's time step. Useful for
regenerative circuits (StrongArm latches, SR flip-flops, ring oscillators)
where the default adaptive stepping can skip over a fast transition. Emitted
as the 4th field of the SPICE `.tran` line.

`Transient.uic` (optional, default `False`) appends `uic` (Use Initial
Conditions) to the `.tran` line. This skips the DC operating-point analysis
and starts transient from device initial values — required for circuits
whose DC op is ambiguous or non-convergent (StrongArm latches, cross-coupled
inverter pairs, oscillators on startup).

`Op()` takes no parameters and requests a standalone DC
operating-point solution.
On ngspice it emits a literal `.op` card; on Xyce, which has no `.print op`
form, `XyceBackend.render_analysis` synthesizes an equivalent single-point
`.dc` sweep over a private, grounded zero-volt source, leaving every user
source at its emitted DC value. Select `analysis='op'` on ngspice and
`analysis='dc'` on Xyce (`ModuleTestbench.probe`,
`select_plot`); node voltages/source currents load as ordinary
single-point Signals. **`'op' is not part of the published metric-schema
vocabulary**: it is deliberately absent from `SignalKind.produced_by` and
from every `@metric(domain=...)` value (the metric analysis vocabulary is
`tran`/`ac`/`dc`/`noise`), so a plain, undecorated metric function
can read an Op()-run signal, but a `@metric`-registered one cannot declare
`domain='op'`.

---

### Simulator options

Tolerances, integration method, and other simulator-level knobs are set via
`Module.set_simulator_options()`. The actual option keys and render syntax are
defined per simulator in `silk/simulators/<name>.yaml`, so silk stays portable
when new simulators are added (Xyce, …).

**User API:**

```python
tb = Module(...)
tb.simulator = 'ngspice'                           # already the default

# Apply a named preset — portable across simulators.
tb.set_simulator_options(preset='tight')

# Override or add single options (raw, simulator-specific keys).
tb.set_simulator_options(reltol=1e-6)

# Preset + per-call override.
tb.set_simulator_options(preset='fast', method='trap')
```

Calling repeatedly accumulates options; later calls override earlier keys. For SPICE, raw
keys outside the simulator's known-options allowlist produce a `UserWarning`
but are still passed through. Behavioral simulation refuses all options.

**Presets** (portable names — each simulator's YAML maps them to its own keys):

- `'default'` — adds no options; it does not clear options already configured.
- `'tight'` — aggressive tolerances for regenerative circuits. On ngspice:
  `reltol=1e-5, abstol=1e-12, vntol=1e-8, chgtol=1e-16, trtol=1, method=gear,
  maxord=2`.
- `'fast'` — looser tolerances for rough sweeps. On ngspice: `reltol=1e-3,
  abstol=1e-12, method=trap`.

**Simulator YAML schema** (see `silk/simulators/ngspice.yaml` for a complete
example):

```yaml
name: <simulator-short-name>

emit:
  line_format: ".options {options}"    # how one options line is built
  kv_format:   "{key}={value}"         # how one key=value is rendered
  separator:   " "                     # joins consecutive kv pairs

presets:
  default: {}
  tight: { reltol: 1e-5, method: gear, ... }
  fast:  { reltol: 1e-3, method: trap }

known_options:                         # allowlist for typo warnings
  - reltol
  - abstol
  - ...
```

A simulator backend also needs a registered implementation; its options
configuration lives in `silk/simulators/<name>.yaml`. Silk loads
the file on demand (`silk.simulators.load_simulator_config(name)`), renders
the options with `emit_options_line()`, and warns when a raw kwarg isn't in
`known_options`. A missing YAML falls back to a permissive
`.options key=value …` line for a registered backend without a configuration file. Unknown backend
names still raise `ValueError` when assigned to `Module.simulator`.

**Deprecation:** `Module.set_spice_options()` is a back-compat alias that
emits a `DeprecationWarning` and forwards to `set_simulator_options()`.

### `Corner`

String-subclass enum. Plain strings are also accepted wherever a corner is expected.

```python
from silk.simulation import Corner

Corner.TT    # 'tt'
Corner.FF    # 'ff'
Corner.SS    # 'ss'
Corner.SF    # 'sf'
Corner.FS    # 'fs'

Corner.standard()   # (Corner.TT, Corner.FF, Corner.SS, Corner.SF, Corner.FS)

Corner.TT == 'tt'   # True
```

### `BehavioralEngine`

`BehavioralEngine(*, scope=None, sample=None)` evaluates a `Module` circuit
entirely in Python using numpy — no ngspice required.
`run_dc(module, analysis) -> (signals, sweep, axis_type)` returns an axis type
of voltage or current.
Modeled blocks declare port directions via `define_module_ports()` and attach a model
via `set_behavioral_model()`; nonempty unmodeled blocks are traversed. Sources are converted to numpy arrays automatically.

Each independent `run()` or `run_dc()` call starts a fresh logical run with the
current model attachments and parameters, fresh per-placement callable state,
and resolution using the same constructor `scope` and `sample`. Reusing an
engine does not continue its preceding call. Analyses within one backend launch
share the attachment snapshot, callable state and resolved variation; declare
analyses in one testbench run when they should share state. User-owned Python
globals are not reset. Every placement receives a fresh callable copy on every
run, including a plain function placed once. Captured Python objects are copied;
model writes affect silk's copy. To observe logging outside the run, use a
module-level logging function or inspect saved results.

```python
from silk.simulation.behavioral import BehavioralEngine

engine = BehavioralEngine()
results, t = engine.run(top_module, t_step=1e-9, t_stop=100e-9)
# results: dict[net_name → np.ndarray]
# t:       np.ndarray of time points
```

`Module.simulate_behavioral()` wraps this and stores results so `load_signal()` /
`get_signal_names()` work transparently:

```python
top.simulate_behavioral(t_step=1e-9, t_stop=100e-9)
sig = top.load_signal('vout')   # returns NgspiceSignal with x=time, y=values
```

**Behavioral model callable interface:**

Models use the ports interface; `execution` selects vector or sample calls:

```python
def follower(ports, t, params):
    ports.vout.v = params.gain * ports.vin.v

block.set_behavioral_model(follower, params={'gain': 2.0}, execution='vector')
```

`ports.X.v` reads the connected net voltage; `ports.X.i` reads current into
this placement's pin. Assign to publish values; return `None`. Assignments copy
values, and each port sample may be assigned once. Calculate and revise local
values before assigning them. Competing net-voltage writers raise
`ConflictError`; repeated assignments raise `RewriteError`. A current read
fills one unknown pin current by KCL; a second unknown raises `KCLError`.
Unset stored voltage samples remain zero. Plain sample reads use the current
publication, otherwise the most recent earlier publication or the reader's
initial value. Returned arrays are copies; modifying them does not publish values.

`params.gain` and `params['gain']` read the same live entry, including parameter
names such as `items` or `get`. Attributes expose parameters only, with no
dictionary methods. Statistical parameters use the run's per-placement values.

| Contract | Transient | DC |
|---|---|---|
| `execution='vector'` | One call with the complete time array | One call per sweep point, with a one-element array containing the point index |
| `execution='sample'` | One call per point with scalar time and electrical values | One call per sweep point with scalar point index and electrical values |

Sample models run in declared dependency order at each point. Vector calls
finish over the whole grid; mixed feedforward chains preserve dependencies.
The ports-form attachment and execution contract are fixed for the run, including all analyses dispatched by that run.
Declared IN/OUT directions order drivers before readers; INOUT retains placement order
and warns. Loops containing ports-form models run members in placement order,
with each call completing once. Sample loops advance in that order at each
point; vector loops execute complete arrays once, using reader-local initial
values where the producer has not run. A transient loop mixing sample and
whole-array models raises `LoopError` whatever the port types; give the loop
models the same `execution=` or put the loop in one function. Mixed-contract
loops visible through declared outputs refuse before model calls. Loops discovered only
from actual INOUT reads and writes refuse at the end of evaluation, before
any result is returned or published. INOUT ordering and notices stay the
same.

Undelayed transient feedback warns once per run, including equal-valued
feedback. `ports.FB.v.previous_value` in sample mode and
`ports.FB.v.previous_value[i]` in vector mode read exactly one transient point
back, using `ports.FB.v.initial_value` (zero by default) before the first
point. Delay on a delayed view adds another point. Initial values are plain
reader-local values and may be changed; neither initial values nor delays
change the producer's waveform. Currents and buses expose the same interface.
An explicit delay breaking the feedback cycle removes its warning.

DC feedback raises `LoopError` (a `SimulationError` subclass) after the point's
calls finish and before publishing results. The error names the point,
models and reads, suggesting transient or one combined model. Pure voltage
watchers run. A DC `previous_value` read raises `SimulationError` immediately,
even without feedback: branch on `silk.analysis()` and use the delay only in
transient. Setting a numeric `initial_value` is allowed and has no effect in DC.
Sample scalar voltage/current reads return `np.float64`, including implicit
zeros and real numeric initials. Assigning `initial_value` accepts real numeric
scalars (including bool) and real numeric arrays; text, bytes, complex values,
`None` and object arrays raise `TypeError` immediately without changing the
previous initial. Bus initials retain their per-bit broadcasting rules.

Bus values have shape `(width, T)` in vector mode and `(width,)` in sample
mode, including width one. Rows follow `expanded_names()` order.
`ports.code[8]` selects declared bit 8, including buses starting at nonzero
indices. Assigning a bus sets its bits; wrong shapes raise `ValueError`.

Each ports-form placement has independent model state, retained across DC points.
Electrical values are evaluated separately at each DC point. A modeled block
remains a leaf replacing all its contents. Probes and reports read completed
results; failed execution produces no successful new result. Ordinary Python
side effects are not rolled back.

The calling contract defaults to `execution='vector'`. Indexing the ports object
with a string raises `SimulationError` showing `ports.VIN.v`; returning a value
raises `SimulationError` explaining assignment and `return None`.

**Which analysis is running: `silk.analysis()`**

During a behavioral run, `silk.analysis()` (also `from silk import analysis`)
returns the analysis being dispatched, `"transient"` or `"dc"`. Synchronous
helpers see the same value. The value is restored on return or failure.
Outside a run it raises `SimulationError`.

```python
from silk import analysis

def follower(ports, t, params):
    if analysis() == 'dc':
        ports.vout.v = ports.vin.v
    else:
        ports.vout.v = params.gain * ports.vin.v
```

**Errors and warnings:**

- `SimulationError` for an empty unmodeled block; nonempty unmodeled blocks
  are traversed
- `NotSupportedError` for a transistor outside a modeled leaf (select a
  SPICE backend to simulate it)
- `LoopError` on DC feedback or mixed sample/vector transient feedback
- `UserWarning` on transient feedback without an explicit point delay
- `UserWarning` for `INOUT` ports (evaluation order assumed)
- `UserWarning` for `POWER` ports with no connected source (defaults to 0 V)

---

### `NgspiceRunner`

Low-level ngspice subprocess wrapper. `Module.simulate()` uses this internally.

```python
from silk.simulation import NgspiceRunner

runner = NgspiceRunner(netlist_path='out/ckt.cir', results_dir='out')
runner.run()       # invokes ngspice; raises SimulationError on a non-zero exit,
                   # or a convergence error / SimulationAborted if no rawfile was written
results = runner.load()
```

> A successful exit code is not sufficient: ngspice can abort an analysis and
> still exit 0. Both backend execution (`backend.run` → `_require_data`) and
> direct `NgspiceRunner.load()` reject results with no data by raising
> `SimulationAborted`. See
> [`simulation-error-handling.md`](simulation-error-handling.md).

---

## silk.spice_names — the SPICE-name codec

`from silk.spice_names import to_spice_name, to_designer_name, spice_component, designer_component, designer_signal_names`

A stdlib-only leaf. Instance components use `[k]` in silk and `<k>` in the
simulator (ngspice drops `@` saves whose path has `[`); this is the
one conversion between the two spellings of a vector name.

| Function | Meaning |
|---|---|
| `to_spice_name(name, style='auto', *, kind='node', allow_glob=False)` | Bracket spelling to the simulator's: `xx1[0].xy[1].g[0]` -> `xx1<0>.xy<1>.g[0]`. `allow_glob=True` (selectors) also recognises `*` and `?` in a segment (`xx1[*].g[0]` -> `xx1<*>.g[0]`) and changes nothing else; default off. |
| `to_designer_name(name, style='auto', *, kind='node')` | Simulator spelling to a **bracketed SPICE alias** of the vector name. It is not designer case, not designer prefixes (`x`, `XM`, `V`) and not a verified entity. |
| `spice_component(part)` / `designer_component(part)` | ONE instance component, either direction. |
| `designer_signal_names(names, *, skip=1)` | Bracket aliases of a rawfile's vector names, only where collision-safe: names whose aliases coincide (case-insensitively) all keep their raw spelling; the first `skip` entries (the independent axis) are untouched; order and count are preserved. |

Only instance positions of recognised vector syntax change: dotted (ngspice,
behavioral) and colon (Xyce) paths, `v(...)`/`i(...)` wrappers, `@` addresses
(the final `[param]` is kept), and Xyce `N(...:PARAM)` / `ID(...)`. Every
segment but the last is an instance; the last is a node leaf and is left alone
(so `g<0>` stays distinct from `g[0]`), unless the name is a branch: a
`i(...)`/`ID(...)` wrapper or `kind='branch'` makes the last segment an
instance (`vvs<0>` <-> `vvs[0]`). A wrapper in the string overrides `kind`; a
bare name defaults to `'node'`. `style` (`'auto'`, `'ngspice'`, `'xyce'`,
`'behavioral'`) only restricts the recognised grammar; an invalid `style` or
`kind` raises `ValueError`. Case and separators are preserved, the engine is
never inferred, and expressions, parameter selectors and unknown internals come
back unchanged. The grammar is checked before converting: a path uses ONE separator
kind (`.` or `:`), each segment is a non-empty run of characters that are EITHER in the ASCII set
`A-Z a-z 0-9 _ $ - [ ] < >` exactly OR non-ASCII and neither a C1 control
(U+0080-U+009F) nor a listed Unicode space/BOM (U+00A0, U+1680,
U+2000-U+200A, U+2028, U+2029, U+202F, U+205F, U+3000, U+FEFF); so letters of
any script and combining marks pass, while every other ASCII character
(`& % ^ ! ~ #`, quotes, whitespace, controls...) leaves the name unchanged.
Recognition is of the whole string: a trailing newline or space is never
trimmed and leaves the name byte-for-byte unchanged, `style='ngspice'` recognises only dotted
bare paths and `v(...)`/`i(...)`/`@` forms, `style='xyce'` only colon bare paths
and `V`/`I`/`N`/`ID` forms, `style='behavioral'` only dotted unwrapped paths; anything else (mixed
separators, `#body`/`#branch` internals, `?`, `|`, expressions) is returned
unchanged.

## silk.signals

### `Signal`

Base waveform object. Supports arithmetic operators and plotting.

`ModuleTestbench.probe()` returns an initially unbound `Signal`; its arithmetic
and transforms resolve when results are read. See the [practical probe guide](probes.md)
for a runnable metric/plot workflow, current naming, analysis selection, corner
retention and the distinction between a live probe and a bound Signal.

```python
sig.name       # signal name string
sig.x          # numpy array (time / frequency)
sig.y          # numpy array (voltage / current / etc.)
sig.type       # 'voltage' | 'current' | ...
sig.xlabel     # axis label string — 'time' / 'frequency' from ngspice AND Xyce
               # (from_rawfile case-folds the rawfile's axis name; Xyce writes
               # 'TIME'/'FREQUENCY'); a DC sweep keeps the simulator's sweep
               # vector name ('v(v-sweep)' ngspice, 'sweep' Xyce)
sig.ylabel     # axis label string
sig.plot()     # matplotlib plot
```

Arithmetic (`+`, `-`, `*`, `/`, unary `-`, `**`):

```python
gain  = vout / vin       # Signal / Signal: same xlabel/x required; type
                          # composes to 'a/b' ('*' composes to 'a*b')
delta = vout - voutn      # Signal +/- Signal: same type required (raises
                          # ValueError on a mismatch); type is unchanged

half  = vout * 0.5        # Signal OP real scalar, either side (+, -, *, /):
offby1 = 1 - vout          # x and .type are unchanged; .name records the
scaled = vout / 2          # constant, e.g. 'vout*0.5', '1-vout'
inv   = 2 / vout

flipped = -vout            # unary negation: same type, name '-vout'
squared = vout ** 2        # real exponent only (TypeError for a Signal
                          # exponent); .type becomes 'voltage^2'
```

A complex Signal (AC-style samples) passes through every operator
unchanged — no magnitude is taken here. Generic scalar metrics that require
real samples need an explicit `.real()`, `.mag()`, or `.imag()` projection.
Signal-with-Signal operations require matching x coordinates and axis labels;
addition and subtraction also require matching types. Different `ylabel`s
(for example, different net names) are allowed without a warning.

### Default waveform plot (`pick_default_plot`)

`silk.signals.view.pick_default_plot(plots) -> int` returns the viewer's
zero-based default plot index without changing plot order. Pass a `RawFile`
for its unfiltered default (including all operating-point vectors), or an
ordered sequence of mappings with `kind` and `signals` fields, such as
`emit_view_data(raw, signals=["v(out)"])["plots"]`, for a filtered default.
It chooses the last non-constants plot with signals, then the last plot with
signals, then the last non-constants plot, then the last plot. Empty input
returns `0`; `emit_view_data` still rejects rawfiles containing no plots.
The emitter and this public selector share one selection implementation.

### Waveform view labels (`emit_view` / `emit_view_data`)

`labels=` (raw vector name -> display label, built by the caller, e.g. with
`silk.spice_names.to_designer_name`) adds a per-signal `label` field to the
payload; the viewer shows the label as the primary text of a trace (signal rows,
group rows, chips, the monitor table) with the raw spelling secondary (tooltip
or second span), and its filter box searches both. `labels='designer'` builds the map from the SELECTED signals on every emit: each
vector's `[k]` bracket alias where it differs from the raw name, so a selection
without arrays carries no label and its payload and `run.id` are unchanged.
`tb.view()`, `silk.view` and `report.add_waveform_view()` pass it by default;
`emit_view` / `emit_view_data` default to `None` (no labels). Labels are display
text for every caller: Copy-as-code never reads one. Designer mode supplies an
`address` only when the reader uniquely resolves the bracketed name in the
complete source plot. Copy-as-code uses it; otherwise it uses the raw name.
Explicit dicts, `labels=None` and old payloads carry no `address`.
`address` takes part in `run.id`. The raw name stays
`name`, `group` and every annotation, saved-state and measurement key. Labels
take part in `run.id`; `state_key` is unchanged; with no labels the payload and
hash are byte-identical. Within a plot, labels that collide with each other or
with an unlabelled name are all dropped (repeatedly, until stable). A key that
names no selected output vector raises `ViewError`. Automatic designer-label
collisions are checked across all rawfile plots and axes, including unselected
vectors; explicit label dictionaries must name selected output vectors.

### `AmbiguousSignalError`

`SimulationError` subclass raised by `load_signal`/`NgspiceSignal.from_rawfile`
when a request matches more than one distinct vector of the selected plot.

### `NgspiceSignal`

Subclass of `Signal` loaded from ngspice raw files. Returned by `Module.load_signal()`.

### Waveform & AC metrics (`silk.metrics.ac`, `silk.metrics.noise`)

Import every metric from `silk.metrics`; imports from its submodules are
equally supported, as shown in the teaching block below.

```python
from silk.metrics import (
    dc_gain_dB, peak_gain_dB, gain_at_freq,
    bandwidth_3dB, unity_gain_freq,
    phase_at_freq, phase_margin, gain_margin, gain_margins,
    cmrr_dB, integrated_noise,
)
```

These ten (all but `gain_margins`) are `@metric`-decorated, so they are ten of
the 80 entries in the catalog `export_metric_schema()` publishes — see
`docs/schema-export.md` Part F.

Pass complex AC data to `phase_at_freq`, `phase_margin`, `gain_margin` and
`gain_margins`; real-valued input emits one `real_response_no_phase`
`MetricWarning` per call and uses real-axis phase only (0° or 180°).

**Function signatures and return values:**

| Function | Signature | Returns | Notes |
|---|---|---|---|
| `dc_gain_dB` | `(sig: Signal) -> float` | Gain in dB at the sweep's lowest frequency | The DC gain, to the extent the sweep reaches down to it; use `peak_gain_dB` for the peak |
| `peak_gain_dB` | `(sig: Signal) -> float` | Peak magnitude in dB | `20·log10(max(|y|))` anywhere in the sweep |
| `gain_at_freq` | `(sig: Signal, freq: float, *, interpolate=True) -> float` | Linear magnitude `|y|` at `freq`, log-log interpolated between neighbouring samples; if either magnitude is zero, magnitude is interpolated linearly using the same frequency fraction (`interpolate=False`: nearest sample) | |
| `bandwidth_3dB` | `(sig: Signal, *, reference='dc') -> float` | −3 dB bandwidth in Hz | Returns the last sampled frequency at or above the threshold without cutoff interpolation; for band-pass responses this is the upper boundary, not upper minus lower cutoff. `reference='dc'` (default) is DC-relative, the same convention `dc_gain_dB`/`gain_bw_product` use; `reference='peak'` is peak-relative, for a band-pass/high-pass response with no DC gain to reference. Returns `0` if nothing reaches the threshold; `nan` (with a warning) if the upper roll-off was not observed, or if `reference='dc'` and the DC gain is exactly zero |
| `unity_gain_freq` | `(sig: Signal) -> float` | Unity-gain frequency in Hz | Returns `nan` with a `MetricWarning` if no unity crossing is observed; with multiple crossings, returns the lowest frequency and warns |
| `phase_at_freq` | `(sig: Signal, freq: float, *, interpolate=True) -> float` | Phase in degrees at `freq`, interpolated along the short arc (`interpolate=False`: nearest sample) | |
| `phase_margin` | `(sig: Signal) -> float` | Phase margin in degrees | Returns `NaN` if no unity-gain frequency exists |
| `gain_margin` | `(sig: Signal) -> float` | Gain margin in dB | Chooses the phase-crossover margin closest to 0 dB (lowest frequency breaks ties). With no crossover, returns `+inf` if the highest-frequency magnitude is below 1, otherwise `nan` |
| `gain_margins` | `(sig: Signal) -> list[(freq, gm_dB)]` | Every phase-crossover margin | All −180°+k·360° crossings, both directions, interpolated; a conditionally stable loop shows both a positive (raise-gain) and negative (drop-gain) limit |
| `cmrr_dB` | `(vdiff: Signal, vcm: Signal, freq=None) -> float` | CMRR in dB | Takes **two** signals: differential and common-mode; measured at `freq` (nearest sample) or each sweep's lowest frequency |
| `integrated_noise` | `(sig: Signal, f1: float, f2: float) -> float` | Integrated noise over `[f1, f2]` Hz | Raises `ValueError` if fewer than 2 samples lie in the band; requires real noise-density samples on a frequency axis |

---

## silk.metrics — ADC / spectral / timing / noise metrics

`silk.metrics` is silk's one metric library, organised by measurement purpose. Each metric keeps its schema domain;
its input signal kinds describe the data it accepts. Beyond the AC/waveform metrics above it
covers ADC performance (INL, DNL, SNDR, SFDR, ENOB), spectral analysis
(FFT, PSD, THD), timing / jitter, power, noise figures, and figures of
merit. Waveform functions take `Signal` objects; closed-form formulas take
scalars.

Files: `waveform.py`, `statistics.py`, `timing.py`, `jitter.py`, `eye.py`, `power.py`, `spectral.py`, `rf.py`, `converter.py`, `fom.py`, `ac.py`, `noise.py`.
Warnings live in `diagnostics.py`.

Full reference: [metrics-library.md](metrics-library.md).

Metric axes must be finite and one-dimensional, with matching samples.
Generic point queries, coordinate-window aggregates, resampling, error norms
and frequency measurements accept strictly rising or strictly falling axes.
They refuse repeated coordinates and axes that change direction with a
`ValueError` describing what was found and what the metric needs.
`value_at` interpolates by coordinate; `final_value` reads the last sample
in acquisition order, including the low endpoint of a descending DC sweep.
Aggregate `start`/`end` bounds are increasing coordinate bounds, regardless
of acquisition order; omitted bounds select the coordinate minimum/maximum.
`integral` integrates in that increasing direction, and tied extrema select
the lowest coordinate. `derivative` and `sample` preserve the requested
output order. Complex interpolation conventions remain unchanged.

`crossing` and `crossings` accept either monotonic direction on any analysis
axis. Events retain acquisition order: `edge='rising'` means y increases
as the simulator advances through samples, and occurrence `n` counts in
that order. `start` and `end` still bound a coordinate interval. Time-only
measurements require strictly increasing time; `moving_average`, `convolve`
and `cross_correlation` likewise require increasing coordinates.

`window` is a coordinate mask and `running_integral` is a signed path
integral starting at the original first sample. Both preserve acquisition
order and accept finite fold-back paths. `p1db` continues to accept repeated
coordinates in its nondecreasing power table. These exceptions do not permit
interpolation on a non-monotonic axis.

EVM and clocked spectrum sampling use a shared endpoint tolerance of
`1e-9 * period`, matching the available-sample count over the common I/Q
span for EVM. An endpoint within that rounding tolerance is sampled at the trace endpoint; a request beyond
it raises. EVM retains linear I/Q interpolation. Clocked spectra use linear
interpolation at the requested `tstart + k*tclk`, preserving exact matches.
Each off-grid input emits an `interpolated_samples` MetricWarning with signal
name, total/interpolated counts, maximum absolute nearest-recorded-time
distance in seconds, and a separate endpoint-clamped count. Distances refer
to the original requests, including endpoint roundoff. Significant distances
may warrant finer simulation time resolution; roundoff-sized distances
generally need no action. Keep the intended clock period and phase unchanged.
This distance is not a voltage-error bound; normal warning filters and existing
SpecReport export/persistence apply.

`average` and `integral` preserve complex samples, using trapezoidal
integration weighted by the x-axis (including nonuniform frequency grids).
`value_at` and `final_value` also return complex values for complex input.
Point and window-edge interpolation treats real and imaginary parts separately,
linearly in x, even on a logarithmically spaced frequency grid. Real inputs
still return floats. Out-of-range point queries return `nan`; empty inputs
and invalid aggregate windows still raise `ValueError`.

`variance` returns the real x-weighted mean of `abs(y - mu)**2`, where `mu`
is the complex x-weighted mean. `stddev` returns its square root. These
measure spread about the complex mean, not spread of `abs(y)`; varying phase
can produce nonzero spread even when magnitude is constant. RMS retains its
existing magnitude interpolation at complex window edges; the usual
variance/RMS identity therefore need not hold when such edges are inserted.

`peak`, `peak_to_peak`, `minimum`, `maximum`, `argmin_x`, `argmax_x`, and
`crossing` require a chosen real projection. Their `ValueError` names
`.real()`, `.mag()`, and `.imag()` choices (phase transforms also work).
For example, `average(h.real())` averages the real part and `average(h.mag())`
averages magnitude. Deferred expressions such as `average((a / b).real())`
work too.

`SpecReport.metrics` and `SpecResult.value` preserve complex values, including
zero imaginary parts and nonfinite components. Fresh and cached metric handles
retain those same values. Spec limits require a real value: a spec bound to a
complex metric fails, and the report summary names `result.real` or `abs(result)`
as the required projection. No real-valued `Measurement` is recorded. You can
also project the input Signal, or register `abs(metric_handle)` as a derived
real metric before running. Existing `metric_warnings` and their output
attribution are unchanged; the comparison refusal adds no metric warning.

Real-only corner/Monte Carlo reductions and optimizer comparisons explicitly
refuse complex values with projection guidance. Numeric database storage skips
complex metrics with an explicit reason, while preserving and validating the
failed spec evidence in memory. Monte Carlo tables retain complex values and
CSV writes their text representation.

`SpecReport` JSON and `CircuitReport` v4 JSON preserve complex results using
the existing results-export tags (`status: complex`, with `re` and `im` components).
Nonfinite components also round-trip through explicit tags. A raw complex-valued
metrics dictionary is not directly JSON-serializable; use the report export API.
`@metric` accepts `float | complex` return
annotations for the four complex-preserving metrics; complex-only helpers
remain outside the scalar catalog.

`rms` retains complex inputs: `sqrt(mean(abs(y)**2))`, with the mean weighted
by the x-axis, is the definition of complex RMS and returns a real scalar.
`error_norm` and `relative_error_norm` also retain complex inputs: they
interpolate the complex reference, subtract it, then measure `abs(error)`.
This measures the size of a complex difference rather than silently
projecting either input. Complex-preserving transforms retain their contracts.

```python
import silk.metrics

# Spectral (FFT-based)
from silk.metrics.spectral import sndr, sfdr, thd, fft, psd
from silk.metrics.converter import enob

# Data-converter linearity and RF compression
from silk.metrics.converter import inl, dnl, is_monotonic, missing_codes
from silk.metrics.rf import p1db

# Timing, jitter and power
from silk.metrics.timing import (
    rise_time,
    fall_time,
    rise_times,
    fall_times,
    slew_rates,
    settling_time,
    propagation_delay,
    propagation_delays,
    average_duty,
)
from silk.metrics.jitter import (
    period_jitter,
    cycle_to_cycle_jitter,
    long_term_jitter,
    crossing_jitter_rms,
    period_jitter_pp,
    cycle_to_cycle_jitter_pp,
    period_deviation,
    cycle_to_cycle_difference,
)
from silk.metrics.power import (
    average_power,
)

# Noise
from silk.metrics.noise import (
    noise_figure,
    input_referred_noise,
    sum_uncorrelated,
    kT_over_C,
    cascade_noise_figure,
    noise_figure_at_freq,
    band_noise_figure,
    noise_figure_vs_freq,
)
from silk.metrics.converter import (
    aperture_jitter_snr_db,
)

# AC filter metrics
from silk.metrics.ac import (
    passband_ripple_db, stopband_attenuation_db,
    group_delay, gain_bw_product, psrr_db,
)

# Figures of merit
from silk.metrics.fom import walden_fom, schreier_fom, iip3_extrapolation, pae
```

For the full function table (139 functions, 7 result classes and the 12
submodule namespaces — 158 public names in `silk.metrics.__all__` — with
signatures and return types) see `docs/metrics-library.md`.

---

## Public designer name checks

```python
from silk.circuit import validate_instance_name, validate_net_name

validate_instance_name('vs[0]')  # returns None
validate_net_name('out[0]')      # returns None
```

### `validate_instance_name(name: str) -> None`

Uses the same callable as silk's device, source and placement construction
and add paths. A case-insensitive `__silk_` occurrence anywhere in an instance
name raises `CircuitError` with this message:

```text
Instance name 'in__silk_dc' contains the reserved __silk_ prefix. The __silk_ prefix is reserved for silk-generated names; rename this instance.
```

Authored net, port and instance names use ASCII letters, digits and `_`,
start with a letter or `_`, and may end in one numeric index such as `data[3]`.
The ground net `0` is the only leading-digit exception; instances cannot be `0`.
Names such as `a:b`, `a.b`, `v-bias`, `bus[en]`, `net[3]bias`, `d[1][2]`
and `gnd!` are refused at construction and rechecked at netlist emission.
`gnd!` is an ordinary node in SPICE, not ground. Use `0`; inside a block,
declare `Port.ground()` (conventionally `vss`) and wire it to `0` at the bench.
The `.` and `:` separators remain legal in hierarchy selectors and probe paths,
including `tb.probe('m1:gm')`; they are not part of an authored name.

Invalid instance names raise `CircuitError`. This check does not validate nets.

### `validate_net_name(name: str) -> None`

Delegates to silk's existing net-token validator using the `add_net` context,
so exception types and messages match `Module.add_net`. Empty or
whitespace-containing names, names outside the character rule above, and names
starting with `__silk_` (case-insensitively) raise `ValueError`. Invalid value types
raise `TypeError`; Net-like objects with a string name retain the underlying
validator's normalization. Valid names return `None`. For example,
`xdut.__silk_y` is refused, while `in__silk_y` is a valid net name. Pin paths
use the same rule with their existing pin-specific error context.

Neither function has a `kind` argument or exports a reserved-prefix constant;
clients can call the checks without reproducing silk's rules.

## silk.sources

All source classes live under `silk.sources` (importable as `from silk import sources`).

`source.physical_instance_names()` returns every emitted card name without
resolving values or calling a waveform. A constant source keeps one card.
Every waveform source has two: `<instance>__silk_pwl` for the dedicated `pwl=` slot
(or `<instance>__silk_wave` for any other slot), followed by `<instance>__silk_dc`.

Generated waveform card suffixes follow an emitted array index:
`Vvs<0>__silk_dc` for `vs[0]`, or `Vvs_0__silk_dc` for `vs_0`.
Logical `i(<designer name>)` probes combine the generated cards.
Designer instance names containing `__silk_` anywhere, case-insensitively,
raise `CircuitError`, so an authored `in_dc` can coexist with the generated
`Vin__silk_dc` card of source `in`.

### Voltage / current sources

```
sources.VoltageSource(name, positive, negative, dc=0, ac=0,
                      pwl=None, pwl_r=None, pwl_td=None,
                      pulse=None, sine=None, exponential=None, sffm=None, am=None,
                      trnoise=None, trrandom=None, waveform=None)
sources.CurrentSource(name, positive, negative, dc=0, ac=0, ...same waveform kwargs...)
```

`dc` adds to the whole waveform, including its
initial value, for voltage and current sources. Voltage cards are in series;
current cards are in parallel. Both cards exist even when `dc=0`. For example,
`dc=.2, sine=Sine(.5, .1, 1e3)` starts at .7 and varies from .6 to .8.
Use `dc=0` when the waveform already specifies absolute levels.
`ac` appears only on the DC card; DC sweeps change that card, so the measured
quiescent value is the sweep coordinate plus the waveform's initial value.
Noise input selection also names the DC card. NumPy and behavioral DC use
the same addition.

Only one waveform slot may be supplied. Conflicts raise `CircuitError` naming
the source and slots at construction, update and emission. A placed source
cannot gain or lose a waveform, change its physical card names, or be renamed;
value updates and atomic replacements preserving the physical layout are
allowed. Failed layout updates leave its previous state intact.

Voltage sources use a private node beginning `__silk_src_V`, followed by the
designer source name with `[k]` retained. `_`, `.`, and `:` escape to `_e`,
`_d`, and `_c`. The case-insensitive `__silk_` net prefix is reserved across
hierarchy components and checked again at emission. Other generated nets keep
their names. Authored opaque lines may not use reserved net tokens; external
include contents are outside validation and must honor this namespace.

`Sine(voffset=0, vampl=required, freq=required, td=0, theta=0, phase=0)`
keeps the original positional order. Amplitude and frequency remain required;
the schema and generated factories report that contract. Before a positive
delay, a phased sine holds `voffset + vampl*sin(phase)` (phase in degrees).

`i(name)` reads the series branch for voltage sources and the signed sum of
both measured cards for current sources. Physical card names read individual
operands. Current-source AC readback is refused; measure it with a 0 V
`VoltageSource` in series. See [probes](probes.md) for saving and annotations,
and [backend limits](known_issues.md#waveform-source-operating-points).

### Finite bit-pattern and PRBS sources

```python
from silk import sources

wave = sources.BitPattern('11010', v1=0, v2=1.8, ui='1n',
                          tr='20p', tf='40p', td='500p', repeat=2)
src = sources.VoltageSource('data', positive='din', negative='0', waveform=wave)
# Equivalent generated factory; current levels are in amperes:
src = sources.voltage.bitpattern('data', p='din', n='0', bits='11010',
                                 v1=0, v2=1.8, ui='1n', tr='20p', tf='40p', td='500p', repeat=2)
random_wave = sources.PRBS(9, 220, seed=0, v1=0, v2=1.8,
                           ui='1n', tr='20p', tf='40p')
reference = sources.prbs_bits(9, 220, seed=0)
assert random_wave.stream == reference
parameters = random_wave.as_parameters()
```

Signatures: `BitPattern(bits, *, v1, v2, ui, tr, tf, td=0, repeat=1)`,
`PRBS(order, length, *, seed=0, v1, v2, ui, tr, tf, td=0)` and
`prbs_bits(order, length, *, seed=0)`. Factories are
`sources.voltage.bitpattern/prbs` and `sources.current.bitpattern/prbs`,
with aliases `vbitpattern`, `vprbs`, `ibitpattern`, `iprbs`. Both `p/n` and
`positive/negative` pin names work, including active-module registration.

`bits` accepts a nonempty binary string or a list/tuple of integer 0/1 values;
it is copied and normalized to the base string. Booleans, floating bits and
arbitrary iterables are rejected. `repeat` is a positive integer total play
count. `.stream` is an immutable tuple including every repetition. PRBS
requires an explicit positive integer `length`; it continues through natural
period boundaries without restarting a shorter prefix. Orders are 7, 9, 15,
23 and 31, with tap pairs (7,6), (9,5), (15,14), (23,18), (31,28).
The Fibonacci XNOR convention maps seed bit 0 to Q_1, inserts feedback into Q_1,
and outputs Qn **after** shifting. Seed defaults to zero; negative,
out-of-range and all-ones lock-up seeds are rejected. Against XOR/all-ones
pre-shift output, the stream is complemented and advanced one sample
(a left circular rotation over a full period). Structural fields must be
literal integers, never booleans, Variables or expressions.

`export_source_schema()` marks `bitpattern.bits` with `kind: bits` and
`prbs.order`, `prbs.length`, `prbs.seed`, and `bitpattern.repeat` with
`kind: integer` on both `VoltageSource` and `CurrentSource`. Integer fields
reject floats (even whole-valued floats) and numeric strings with `TypeError`;
consumers must send integers. Other numeric inputs, including dimensionless
`mdi`, omit `kind`. These hints come from the name-keyed `param_kinds` mapping
in `silk/sources/source_metadata.yaml`.

Bit k starts at `td + k*ui`. The source holds its first bit's level before
`td` and its last level after completion. A transition starts at the bit
boundary and reaches its target after `tr` (logical 0→1) or `tf` (1→0).
Repeat joins are ordinary adjacent bits. Equal adjacent bits and constant
patterns are valid. Swapping `v1/v2` preserves complementary timing even
with unequal edges; complementing the bits exchanges rise/fall directions.

All electrical fields accept numbers, SI strings, Variables and expressions,
resolved afresh for SPICE emission and NumPy evaluation. Levels must be finite,
`ui > 0`, `td >= 0`, and the mandatory edges satisfy `0 < tr, tf <= ui`.
Both waveforms use the same explicit finite PWL points for NumPy and SPICE;
no native repeat syntax is involved. Generation and deck size grow with
requested length (and pattern repeat count); requesting a full PRBS31 period
would require over two billion bits. Choose a bounded measurement window.

`as_parameters()` returns a fresh compact declaration snapshot with authored
Variable identities. BitPattern records base bits and repeat; PRBS records
kind, `convention='xnor-post-shift'`, order, seed and length, without expanding
the stream. Supply it as a named test parameter. Changing a literal after
registration requires rebuilding that declaration; Variables remain shared.
The PRBS seed is independent of the simulator RNG seed.

To replay a saved stimulus from a report or a scoped DB profile, use the
recorded constructor fields (not the expanded stream):

```python
record = dict(report.parameters['stimulus'])
kind = record.pop('kind')
record.pop('convention', None)  # descriptive PRBS metadata, not a constructor kwarg
wave_class = {'bitpattern': sources.BitPattern, 'prbs': sources.PRBS}[kind]
replayed = wave_class(**record)
```

Register `parameters={'stimulus': wave.as_parameters()}` when creating the
test. Reports freeze resolved electrical values for each run; later Variable
assignments do not rewrite earlier records. A DB profile exposes the same
record at `profile.parameters['stimulus']`. Scoped storage still requires
complete operating conditions: configure the simulator temperature explicitly
when the backend cannot establish it from its output. A bit waveform, even
with a nonzero `dc`, cannot serve as a fixed supply declaration.

`dc` adds to every transient bit level and to the waveform operating point,
as for other waveforms. `ac` supplies the small-signal amplitude on the
separate DC card. `initial_value()` returns the resolved first-bit level token,
also held before `td`. See [waveform operating points](#waveform-voltage-source-operating-points)
for backend rules.

### Controlled sources

```python
sources.VoltageControlledVoltageSource(name, out_pos, out_neg, ctrl_pos, ctrl_neg, gain)
sources.VoltageControlledCurrentSource(name, out_pos, out_neg, ctrl_pos, ctrl_neg, gm)
sources.CurrentControlledVoltageSource(name, out_pos, out_neg, transresistance, ctrl_vname)
sources.CurrentControlledCurrentSource(name, out_pos, out_neg, gain, ctrl_vname)

# Short aliases
sources.VCVS   # = VoltageControlledVoltageSource
sources.VCCS   # = VoltageControlledCurrentSource
sources.CCVS   # = CurrentControlledVoltageSource
sources.CCCS   # = CurrentControlledCurrentSource
```

### Behavioral sources

```python
sources.BehavioralVoltage(name, positive, negative, expression)
sources.BehavioralCurrent(name, positive, negative, expression)
```

### Waveform shapes

```
sources.PWL(points=None, r=None, td=None)    # piecewise-linear
sources.Pulse(v1, v2, td=0, *, tr, tf, pw=0, per=0, np=None)    # np limits repeat count
# tr/tf are mandatory, keyword-only, and must be > 0 — see the note below
sources.Sine(voffset=0, vampl=required, freq=required, td=0, theta=0, phase=0)
sources.Exponential(v1, v2, td1=0, tau1=0, td2=0, tau2=0)
sources.SFFM(vo, va, fc, mdi, fs, phasec=0, phases=0)   # also: sources.SingleFrequencyFM
sources.AM(va, vo, mf, fc, td=0, phases=0)               # also: sources.AmplitudeModulated
sources.TransientNoise(na, nt, nalpha=0, namp=0, rtsam=0, rtscapt=0, rtsemt=0)
# na = white-noise RMS amplitude, nt = time-step
# nalpha / namp for 1/f noise; rts* params for random-telegraph-signal noise
sources.RandomSource(type, ts, td=0, param1=1, param2=0)
# type: 1=Uniform, 2=Gaussian, 3=Exponential, 4=Poisson
```

**`Pulse` rise/fall times are mandatory and must be nonzero.** They are
keyword-only, so an edge can never arrive by accident of position:

```python
sources.Pulse(v1=0, v2=1.8, td=0, tr=100e-12, tf=100e-12, pw='25n', per='50n')
```

Supply explicit positive `tr` and `tf` so SPICE and behavioral evaluation
use the same ramp durations. A zero edge would let simulator defaults choose
the ramp duration from the transient step, making the stimulus depend on
analysis settings. silk rejects zero edges rather than relying on that default.

Pick the edge from the circuit's own time scale — a small fraction of its clock
period or pulse width, at least one transient time step so the ramp is
resolved. The shipped examples use 50 ps – 1 ns.

Scalar timing parameters accept numbers or SI-suffixed strings (`100e-12` or `'100p'`).
PWL point times use seconds as numbers, plain numeric strings or Variables.

`PWL` accepts a list of `(time, value)` tuples, a flat `[t0, v0, t1, v1, ...]`
list, or can be built incrementally with `.add(time, value)`:

```python
pwl = sources.PWL().add(0, 0).add(1e-9, 0).add(2e-9, 1.8)
vs = sources.VoltageSource('vin', positive='in', negative='0', pwl=pwl)
```

`PWL.to_numpy()` applies `td` once, holding the first value before the delayed
first point and the last value after a nonrepeated sequence. With `r`, the
first traversal includes the last point, then the segment from point time `r`
to the last point repeats without repeating the delay. Subsequent exact joins
select the repeat-start value. Set equal repeat-start and final values for a
continuous join. `r` must equal a point time strictly before the last point;
NumPy evaluation requires finite values, distinct nonnegative point times,
and finite nonnegative delay. Point sorting and valid untimed interpolation are
unchanged. Timing Variables are resolved on every evaluation.

`Pulse.to_numpy()` honors a nonnegative integral `np`: `None` and `0` mean
unlimited pulses; a positive count returns to `v1` after its final pulse.
Finite-count NumPy evaluation requires explicit `pw > 0`, `per > 0`, finite
levels/times, `td >= 0`, and `tr + pw + tf <= per`. Simulator-default widths
and periods depend on the transient analysis and cannot be inferred from an
arbitrary sample array. Counts and edge Variables are rechecked on evaluation.

These timing controls have backend limits. ngspice supports PWL `r`/`td` on
voltage sources and Pulse `np` on voltage and current sources. Current-source
PWL `r`/`td` require **ngspice 47 or newer**. Silk checks the selected
`SILK_NGSPICE_BIN` executable and raises `NotSupportedError` for older or
unknown versions, naming the detected version and required release. Version
probing is cached per canonical executable path for the process lifetime;
restart after replacing a binary in place. Supported versions emit the same
deck text. `export_source_schema()` retains `r` and `td` for current-source
PWL regardless of the selected version. Xyce supports `r` without `td` for both source
quantities when repeat-start and final values are equal. Its exact later join
retains the final value, whereas NumPy follows ngspice's repeat-start value;
silk rejects discontinuous Xyce repeat joins to avoid this mismatch. Xyce does
not honor silk's emitted PWL `td` suffix or Pulse `np` token. Silk raises
`NotSupportedError` during netlist generation for these unsupported combinations
(including explicitly supplied zero delay/count options). Use explicit
delayed/repeated PWL points with neither `r` nor `td`, or omit `np` for unlimited
pulses on Xyce. See
[the measured support matrix](known_issues.md#waveform-timing-controls).

#### Waveform voltage-source operating points

Every waveform source emits a waveform card plus a separate `dc` offset
card, even at `dc=0`: voltage cards are in series, current cards in parallel.
For voltage sources, the waveform card contributes its initial value to the
operating point; changing `dc` adds an offset rather than overriding that value. Xyce
waveform cards receive an explicit DC initial-value clause; ngspice infers
it from the waveform. A PWL holds its first value before its first point,
and its resolved first-value token is reused, including parameter expressions.
Current-waveform operating points follow the selected backend; see
[the operating-point rules](known_issues.md#waveform-source-operating-points).

A generic `waveform=` object supplies `initial_value()` alongside
`to_spice()`. The former returns a number or SPICE value token and must
resolve mutable parameters on each call. If an Xyce waveform element needs
this default but the method is missing, silk raises `NotSupportedError`
instead of silently using zero. The separate offset card does not replace
this requirement.

Built-in `initial_value()` definitions are: Pulse/Exponential `v1`, PWL's
first value, Sine `voffset + vampl*sin(phase)`, SFFM
`vo + va*sin(phasec + mdi*sin(phases))`, AM/TransientNoise zero, and
RandomSource `param2` (all distributions, without a draw). Phases are in
degrees; SFFM modulation depth is in radians. Negative delays and unresolved
numeric delay checks or phase calculations raise `NotSupportedError` during
Xyce inference.

Xyce matches the OP/DC/AC defaults for Pulse, PWL, Sine, Exponential, and
SFFM with zero `phasec` and `phases`. Nonzero or unresolved SFFM phases
raise `NotSupportedError` on Xyce. ngspice SFFM emission is restricted to
the verified release 42; other detected versions are refused. For AM,
ngspice 42 requires `phases=0`; releases 43 and newer use the translated
AM card with phase support. Silk raises `NotSupportedError` during Xyce netlist generation for
`AM`, `TransientNoise`, and `RandomSource` on **both voltage and current
sources**, including explicit DC and generic `waveform=` use. The error names
the kind and source and directs callers to ngspice. This prevents Xyce from
silently replacing an unsupported waveform with constant DC. See the
[measured operating-point rules and limits](known_issues.md#waveform-source-operating-points).

**`to_numpy(t)` — behavioral simulation support**

`BitPattern`, `PRBS`, `PWL`, `Pulse`, and `Sine` each implement `to_numpy(t: np.ndarray) -> np.ndarray`,
which evaluates the waveform at arbitrary time points. `VoltageSource` and
`CurrentSource` delegate to their waveform object automatically, or return a
constant DC array if no waveform is attached. This is used internally by
`BehavioralEngine` — no user action required. Other waveform kinds raise
`NotImplementedError` in `to_numpy`; use a supported waveform or SPICE.

---

## silk.elements

### `ElementLibrary` (`elements`)

Process-independent SPICE primitives. Instances are compatible with `Module.add()`.

Only `name` may be passed positionally — pins and parameters are
**keyword-only** on all generated factories:

```python
from silk.elements import elements

r = elements.resistor('r1',  p='out', n='0', value=10e3)
c = elements.capacitor('c1', p='out', n='0', value=100e-15)
l = elements.inductor('l1',  p='a',   n='b',   value=1e-9)
# Use sources.vdc / sources.idc for voltage and current sources.
```

The library holds exactly these three factories (`resistor`, `capacitor`,
`inductor`). Source entries remain available through schema export. Controlled sources live in
`silk.sources` (`VCVS`, `VCCS`, `CCVS`, `CCCS`), not on `elements`.

### `passives` (deprecated)

`silk.elements.passives.Resistor`/`Capacitor`/`Inductor` and `Module.add_passive()`
are deprecated — use `elements.resistor`/`capacitor`/`inductor`
with `Module.add()` above instead.

---

## silk.optimize

> Practical guide with runnable examples: [optimizer.md](optimizer.md).

### `VariableSpace`

Design parameter space. Wraps a list of `Variable` objects. Give numeric
variables finite lower and upper bounds before sampling; a PDK-derived
variable may need explicit bounds.

```python
from silk.optimize import VariableSpace

space = VariableSpace([w_n, w_p, l])
space.names              # list of variable names
space.to_bounds()        # list of (lo, hi) tuples
space.variables()        # new {space key: Variable} dict, same objects, names order

# Sampling
space.sample(n, method='random', seed=0)   # reproducible random draws
space.sample(n, method='lhs')      # Latin hypercube
space.sample(n, method='halton')   # Halton quasi-random
space.sample(n, method='grid')     # product of actual per-variable grids
space.sample(n, method='lhs', seed=None)  # fresh OS entropy

# Apply a point dict
space.apply({'w_n': 2.0, 'w_p': 4.0, 'l': 0.5})
```

`VariableSpace.sample(n, method='random', *, seed=0)` creates one NumPy
Generator shared by all variables and groups. Integer seeds reproduce the
joint points; `None` uses fresh OS entropy. Booleans and non-integers raise
`TypeError`; negative integers raise `ValueError`. Global random states are
unchanged. Halton and grid are independent of the seed. Categorical grid
uses the same declared-order enumeration as `Variable.sample`: every category
for positive `n`, no points for `n=0`, combined into the full factorial product.
Categorical Halton maps the variable group's prime-base sequence by
`floor(u * category_count)`, just like discrete numeric Halton. This is
deterministic but does not guarantee every category in a finite prefix;
use grid for exhaustive coverage. Categorical random/LHS retain seeded draws.


For linear numeric grids, `n` selects up to that many distinct values across
an enforced resolution domain or the integer domain `ceil(lo)..floor(hi)`.
The space delegates this policy to `Variable`; its grid size is the product
of the actual column lengths, including all categorical choices. A shortened
column emits one `UserWarning` naming the variable, request and actual count,
at the public sampling or sweep caller. Standard warning filters apply;
warnings are visible in terminals, editors and notebooks and are not saved
in reports. Log grids retain `n` draws and may repeat projected values.

Variables are created directly (`Variable(...)`, see above) or via `pdk.<device>.<param>.variable(...)`.

### `Optimizer`

Design optimizer. Takes a `ModuleTestbench`
and a `VariableSpace`, applies parameter points via `VariableSpace.apply()`, runs the testbench,
and collects metrics into an evaluation history. Standard testbenches work
directly: the optimizer reads `.metrics` from their returned `SpecReport`.
Custom tests returning `{'metrics': {...}}` from `run(corner=...)` remain
supported. Testbench specs constrain feasibility by default, preserving their
comparison operators. Explicit metric constraints replace all spec limits
on that metric. Per-spec margins tighten numeric targets without changing
testbench declarations. `SpecReport.passed` itself is not consulted. The space must contain the same
Variable objects used by the circuit; matching names alone do not bind
parameters. `sweep()` evaluates a fixed sample set without adapting later
points from earlier results.

`add_objective` and `add_constraint` also accept a `Variable` from the
VariableSpace; values come from recorded sample parameters, not a synthetic metric.
`add_margin(spec_name, amount)` takes a finite nonnegative amount in the
spec target's numeric unit, increasing lower limits and decreasing upper
limits. Unknown/ambiguous names and `==` specs raise `ValueError`. Repeated
calls replace the margin; explicit overrides always take precedence.
See [optimizer.md](optimizer.md#variables-default-specs-and-margins) for
examples and provenance fields.

Python and NumPy scalar NaN measurements remain unchanged in history with
`status='success'` when extraction completes. NaN fails a constraint on that
metric. NaN in any declared objective excludes the record from `best`.
An objective-only NaN
does not change constraint feasibility; unrelated NaN diagnostics affect
neither feasibility nor selection. Infinity keeps its existing comparisons
and scoring behavior.

`best` also excludes records with an absent, `None`, or string value for any
declared objective, without changing recorded measurements, status or
constraint feasibility. Variable objectives use recorded `applied_params`
(legacy `params` fallback), allowing empty metrics for variable-only objectives.
`summary()['best']` is the same selection, or `None` when
no candidate is eligible.

`best` checks successful records against the current testbench specs, margins
and explicit metric/Variable constraints each time it is read. Changing only
limits can immediately exclude an old winner or admit a previously infeasible
record. It compares recorded measurements and applied parameters again without
rerunning simulations or applying parameters; legacy records use `params`.
It returns the original record, preserving historical `feasible` verdicts and
`constraints` snapshots. A newly admitted winner can therefore still have
`feasible=False`. `summary()['best']` follows current selection, while
`summary()['feasible_count']` counts historical feasible verdicts.
This rechecks acceptance limits; it does not establish that old measurements
remain valid after circuit, backend, stimulus, model or extractor changes.

`Optimizer.evaluate(point, corner=...)` applies a design point and calls
`test.run(corner=...)` once. Optimization requires one report per point: clear
declared corners, Monte Carlo and temperature/supply lists with `None` first.
`ModuleTestbench.evaluate(rawfile=...)` measures
existing simulation data without launching a simulator.

```python
from silk.optimize import Optimizer

opt = Optimizer(test, space)         # test is a ModuleTestbench; space is a VariableSpace

# Configuration (all optional; chain before sweeping)
opt.add_objective('gain_dc', maximize=True)
opt.add_constraint('gain_dc', min=60.0)   # feasibility gate: must be ≥ 60

# Evaluate a single design point
record = opt.evaluate({'w_n': 2.0, 'l': 0.18}, corner='tt')

# Sweep n points sampled from the space
records = opt.sweep(n=50, method='lhs', corner='tt', max_failures=None)

# Results
opt.history       # list[dict] — all records
opt.successes     # records with status == 'success'
opt.failures      # records with any non-success status
opt.best          # best eligible feasible record, or None
opt.summary()     # dict: total_evaluations, by_status, feasible_count, best, objectives, constraints
```

> **Disambiguation — two different `add_constraint` methods:**
> - `Optimizer.add_constraint(name, min=None, max=None)` — numeric **feasibility gate** on a simulation metric; filters evaluation records.
> - `Module.add_constraint(constraint)` — attaches a **layout constraint** (`ConstraintEnvelope`) to a Module; requires `import silk.constraints` first.
> These are completely independent; never conflate them.

**`Optimizer` methods:**

| Method | Description |
|---|---|
| `add_objective(name, minimize=None, maximize=None)` | Declare a metric name or VariableSpace variable as an optimization objective; defaults to minimization. Multiple objectives have equal weights and no unit normalization: register appropriately scaled quantities. Returns `self`. |
| `add_constraint(name, min=None, max=None)` | Add an inclusive numeric constraint on a metric name or VariableSpace variable; overrides specs on that metric. Returns `self`. |
| `add_margin(spec_name, amount)` | Tighten one uniquely named non-equality spec in its target's numeric unit. Returns `self`. |
| `evaluate(point, corner='tt')` | Run the testbench at one `point` dict. Returns a record dict with keys `id`, `params`, `corner`, `status`, `metrics`, `feasible`, `error`, `error_type`, `timestamp`, `applied_params` (complete sample after snapping), `constraints` (effective-limit provenance snapshot). |
| `sweep(n, method='lhs', corner='tt', max_failures=None, n_workers=None, *, seed=0)` | Random/LHS/Halton evaluate `n` joint points; grid requests up to `n` values per numeric variable and evaluates the product of the actual grids, including every category. Forwards `seed` to the space. Returns `list[dict]`. |
| `feasible(metrics, params=None)` | Check effective limits; supply sample parameters for variable constraints. Missing values and NaN fail. |
| `history` | Property: all evaluation records (list). |
| `successes` | Property: records with `status == 'success'`. |
| `failures` | Property: records with any non-success status. |
| `best` | Property: best record eligible under current limits by weighted linear combination of objectives, excluding records with absent, `None`, string or NaN values in any declared objective; `None` if none are eligible or no objectives are declared. Preserves historical verdicts. |
| `summary()` | Return `dict` with `total_evaluations`, `by_status`, `feasible_count` (historical verdicts), `best` (current limits), `objectives` (list in declaration order of `{name, goal, value_source}`), `constraints` (current effective-limit provenance). |


---

## silk.db

### `Database`

```python
from silk.db import Database

# Context-manager exit closes the connection.
with Database('designs.db') as db:
    records = db.get_simulation_results(version_id, parameter='gain_dB')
    specs = db.get_stored_specs(version_id)
```

Opening a missing/empty path creates schema v7. Existing files are validated
read-only first; unsupported or malformed files are preserved and rejected.
Older schemas require explicit migration to a new destination, one version
at a time, using the exported `silk.db.migrate_v1_to_v2` through
`silk.db.migrate_v6_to_v7` functions. See [database.md](database.md).

### Database tracking via `Module`

Pass `db_path` to `Module` to enable tracking. Structure and stable spec
declarations are stored and sealed by `create_circuit_version()`.

```python
from silk.circuit.modules import Module

ckt = Module('PDK_skywater130', 'out', 'ldo.cir', pdk_details=pdk,
             module_name='ldo_nmos', db_path='designs.db')
ckt.add(mn1, mp1)
ckt.declare_spec('DC gain', 'gain_dB', '>=', 40, key='gain')
version_id = ckt.create_circuit_version('Initial design')
ckt.store_results(
    results, source='spice', sim_type='ac', corner='tt',
    temperature=27.0, voltage=1.8, tool_name='ngspice-44',
)
ckt.promote('stable')    # creates and binds a new version, locked by default; lock=False permits writes
ckt.close()
```

### `DesignStage`

```python
DesignStage.DEV
DesignStage.STABLE
DesignStage.POST_LAYOUT
DesignStage.TAPEOUT
```

### `CircuitImporter` / `CircuitExporter`

Batch import/export utilities for the circuit database.

```python
from silk.db.circuit_import import CircuitImporter, CircuitExporter

db = Database('designs.db')
importer = CircuitImporter(db)
importer.import_from_json('circuits.json')
importer.import_from_spice('amplifier.cir')

exporter = CircuitExporter(db)
exporter.export_to_json('ldo_nmos', 'output.json')
exporter.export_results_to_csv('ldo_nmos', 'results.csv')
```

JSON carries the `silk.circuit-projection/v2` marker and full metric context
(each condition as a value, a `{"state": ...}` tag or `null`, plus its
`*_state`; import also accepts `v1` and refuses other markers),
but omits circuit-version/evidence-row UUIDs, specs, currents, parasitics,
complete module connectivity, raw data, logs, and calibration. Import
recomputes identity from parsed content and creates new version/evidence UUIDs;
an exact destination identity collision is rejected before evidence writes. It
is not backup or an archival round-trip.

---

## silk.constraints — Layout intent constraints

`silk.constraints` is a side-effect import: the package's `__init__.py`
monkey-patches `Module.add_constraint` and `Module.list_constraints` onto
`Module` the moment it is imported. These methods do **not** exist on a bare
`Module` — you must `import silk.constraints` first.

For numeric feasibility limits, see `Optimizer.add_constraint`.

```python
import silk.constraints
from silk.constraints import (
    SymmetricBlocks, SameTemplate, GroupBlocks, ProximityConstraint,
    ConstraintEnvelope, ALL_CONSTRAINT_TYPES,
)

# Attach constraints to a Module
ckt.add_constraint(SymmetricBlocks(pairs=[['mn1','mn2']], direction='H'))
ckt.list_constraints()   # returns a copy of the list
```

### `ConstraintEnvelope` (base)

All constraint types inherit from `ConstraintEnvelope` and accept these
metadata fields as keyword arguments. Omitted fields use defaults:

| Field | Type | Description |
|---|---|---|
| `id` | str | Auto-generated UUID |
| `source` | `'motif'\|'llm'\|'human'\|'auto'` | Origin of this constraint; default `'human'` |
| `confidence` | float | Confidence score (0–1); default `1.0` |
| `severity` | `'hard'\|'soft'` | Hard = must satisfy; soft = best-effort |
| `created_at` | datetime | Creation timestamp |
| `rationale` | str \| None | Free-form justification; default `None` |
| `prompt_hash` | str \| None | Hash of the model prompt (when source='llm'); default `None` |

### Constraint types

| Class | Key fields | Notes |
|---|---|---|
| `SymmetricBlocks` | `pairs: List[List[str]]`, `direction: Literal['H','V']`, `severity='hard'` | Symmetric placement of device pairs; each inner list is 1 or 2 names |
| `SameTemplate` | `instances: List[str]` (min 2), `severity='soft'` | All listed instances use the same cell template |
| `GroupBlocks` | `name: str`, `instances: List[str]` (min 1), `generator: Optional[Dict]=None`, `severity='hard'` | Group instances into a floorplan block; `generator` stores metadata for a layout consumer |
| `ProximityConstraint` | `instance_a: str`, `instance_b: str`, `max_distance_um: float` (>0), `severity='soft'` | Maximum separation hint for layout backends |

`ALL_CONSTRAINT_TYPES` — tuple of the four concrete types; used by serialization for discriminated-union construction.

### Serialization

```python
from silk.constraints import serialization

json_payload = serialization.to_json(mod)         # pretty-prints silk-canonical JSON
obj_payload  = serialization.to_obj(mod)          # Python list of dicts
constraints  = serialization.from_json(mod, json_payload)  # attaches constraints; returns List[ConstraintEnvelope]
```

`from_json` applies `PortRoles` and `NetIntent` overlay types **in-place**; those types are not returned in the list.
Declare target ports and nets before importing overlays; unknown targets warn and are ignored.

A `PortRoles` entry's `roles` dict may be accompanied by an additive, sparse
`inferred` dict (`{port_name: True}`) naming every port whose role is
a guess (`ModulePin.role_inferred`) rather than a user statement.
`to_json`/`to_obj` emit it only when at least one exported role is inferred;
`from_json` restores `pin.role_inferred` from it, defaulting a port absent
from the map — including every port in a file with no `inferred` key at all —
to declared (`False`).

### Checkers

```python
from silk.constraints import checkers

# Register a checker before checking this type.
# my_check_fn accepts (constraint, layout) and returns the backend's result.
token = checkers.register(SymmetricBlocks, my_check_fn)
result = checkers.check(constraint, layout)
results = checkers.check_all(mod, layout)          # List[Tuple[ConstraintEnvelope, Any]]
checkers.unregister(SymmetricBlocks, token)        # restore the previous checker
```

All four built-in types have placeholder checkers that raise `NotImplementedError`. Register a real checker to connect a layout backend.

---

## silk.codegen

### `Generator`

Generates silk Python source from a circuit-description dict.

```python
from silk.codegen import Generator

gen  = Generator(indent=4)               # the emitter uses fixed formatting; indent is currently unused
code = gen.generate(payload_dict)        # returns Python source as a string
gen.generate_to_file(payload_dict, 'my_circuit.py')
```

**Emission contract.** Two spec fields are
interpolated into generated code as *identities*, so `generate` validates them
before producing any output and raises `ValueError` naming the field and the
accepted form, for dict and `CircuitSpec` input alike:

| Field | Rule | Why |
|---|---|---|
| `circuit_name` | a valid Python identifier that is not a keyword (`my_amp`, not `"my amp"` / `"diff-amp"` / `"class"`) | It is the SPICE `.subckt` name, the tracking-DB structure identity, the `module_name=` literal and the `<name>.cir` filename. |
| `corner` | matches `[A-Za-z0-9_-]+` (`tt`, `ff_hh_mm`, `typical`) | Emitted into `ckt.simulate(corner=...)`; only the **syntax** is checked at generation time — membership in the PDK manifest is enforced by `Module.simulate`, so codegen works without a resolvable PDK install. |

Use identifiers for circuit names. Literal and header emission escapes these
fields for their Python string or docstring context.

Output mode depends on the payload:

- **Omitted or `None`/JSON `null` `ports`** → standalone simulation script (`Module(...)`, `add(...)`, `simulate(...)`, and `add_sim_type(...)` when an analysis is supplied).
- **A `ports` mapping, including `{}`** → subcircuit / reusable module definition; `{}` declares a zero-port module (`Module(...)`, `define_module_ports({...})`, optional `voltage_domains` and `bus_nets` blocks).

### `CircuitSpec`

The parsed-and-validated form of the payload. `Generator.generate` accepts a dict or a `CircuitSpec` instance.

```python
from silk.codegen import CircuitSpec
spec = CircuitSpec.from_dict(payload_dict)   # or CircuitSpec.from_json(...) / from_json_file(...)
```

Key fields:

| Field | Type | Notes |
|---|---|---|
| `circuit_name` | `str` | Defaults to `'my_circuit'` when parsing a dict; direct `CircuitSpec(...)` construction requires it. Must be a valid non-keyword Python identifier |
| `pdk` | `str` | Default `'sky130'` |
| `output_dir` | `str` | Default `'./sim'`; used by `Module(...)` |
| `corner` | `str` | Default `'tt'`; syntax `[A-Za-z0-9_-]+` (see the emission contract above); standalone mode only |
| `voltage_domains` | `dict[str, VoltageDomain]` | Parsed into real `silk.circuit.voltage_domain.VoltageDomain` instances |
| `ports` | `dict[str, Port] \| None` | Default `None` selects standalone mode; a mapping selects module mode. Entries are parsed into real `silk.circuit.circuit_elements.Port` instances; rich fields (`dir`, `role`, `domain`, `bits`, `naming`) supported |
| `bus_nets` | `list[BusNetSpec]` | Internal bus net declarations; emit `mod.add_net(..., bits=N)` lines |
| `submodules` | `list[SubmoduleSpec]` | Hierarchical child instances; defaults to `[]` — see [codegen-contract.md](codegen-contract.md#submodules) |
| `variables` | `list[VariableSpec]` | Default `[]`; shared parameter declarations referenced as `{'variable': name}` with optional `scale` — see [shared variables](codegen-contract.md#variables--module-variables-shared-by-parameters) |
| `devices` | `list[DeviceSpec]` | PDK transistors, resistors, caps, diodes, BJTs |
| `ideal_elements` | `list[DeviceSpec]` | Process-independent R/C/L; every source belongs in `sources[]` |
| `sources` | `list[SourceSpec]` | Independent, controlled and behavioral sources; see `SourceSpec` below |
| `simulation` | `dict` | Optional, default `{}`; omission adds no analysis declaration, although the standalone script still calls `simulate()` |

`Port` and `VoltageDomain` are reused from silk core; codegen-local dataclasses include `BusNetSpec`, `DeviceSpec`, `SubmoduleSpec`, and `SourceSpec`.

The full JSON shape — including the `voltage_domains` block, the dict-form `ports` (with `dir`/`role`/`domain`/`bits`/`naming`), and the `bus_nets` list — is documented in **[`docs/codegen-contract.md`](codegen-contract.md)**.

### `SubmoduleSpec`

Codegen-local dataclass describing one child Module instantiation.

| Field | Type | Notes |
|---|---|---|
| `instance_name` | `str` | Local instance identifier (becomes `add_module`'s `name`). |
| `import_path` | `str` | Emitted verbatim in the `from … import …` line. |
| `module_var` | `str` | Variable name inside the imported file. |
| `nets` | `dict[str, str] \| list[dict[str, str]]` | Port-to-net wiring map, or one map per array copy; see [per-copy nets lists](codegen-contract.md#per-copy-nets-lists). |
| `parameters` | `dict \| None` | Per-instance parameter overrides. |
| `array` | `int \| tuple[int, int] \| None` | If set, this entry is an array; see contract's `Index templates` section. |
| `naming` | `str` | `"bracketed"` (default) or `"underscore"`. |

Full contract: [docs/codegen-contract.md#submodules](codegen-contract.md#submodules).

### `DeviceSpec`

Codegen-local dataclass describing one PDK device or ideal-element instantiation. Shared by `CircuitSpec.devices` and `CircuitSpec.ideal_elements`.

| Field | Type | Notes |
|---|---|---|
| `type` | `str` | Device/element type key. |
| `instance` | `str` | Authored instance name passed to the factory; the base must satisfy codegen's identifier rules. The factory determines its emitted SPICE spelling. |
| `nets` | `dict[str, str] \| list[dict[str, str]]` | Pin-to-net wiring map, or one map per array copy; see [per-copy nets lists](codegen-contract.md#per-copy-nets-lists). |
| `params` | `dict[str, Any]` | Per-instance parameter overrides (always a dict, never `None`). |
| `array` | `int \| tuple[int, int] \| None` | If set, this entry is an array; see contract's `Index templates` section. |
| `naming` | `str` | `"bracketed"` (default) or `"underscore"`. |

Full contract: [docs/codegen-contract.md#devices--pdk-devices](codegen-contract.md#devices--pdk-devices).

### `SourceSpec`

Codegen-local dataclass describing one independent, controlled or behavioral
source instantiation. Independent sources use `positive`, `negative`, `dc`,
`ac`, `waveform` and `pwl`. Controlled sources use `out_pos`/`out_neg`,
voltage-control terminals or `ctrl_vname`, and `gain`, `gm` or
`transresistance`. Behavioral sources use `positive`, `negative` and
`expression`. See the [source-family field contract](codegen-contract.md#sources--the-single-home-for-every-source)
for required fields and array restrictions.

| Field | Type | Notes |
|---|---|---|
| `type` | `str` | `VoltageSource`, `CurrentSource`, `VoltageControlledVoltageSource`, `VoltageControlledCurrentSource`, `CurrentControlledVoltageSource`, `CurrentControlledCurrentSource` (or `VCVS`/`VCCS`/`CCVS`/`CCCS`), `BehavioralVoltage` or `BehavioralCurrent`. |
| `instance` | `str` | Authored instance name passed to the factory; the base must satisfy codegen's identifier rules. The factory determines its emitted SPICE spelling. |
| `positive` | `str \| None` | Independent/behavioral source positive terminal; `None` for controlled sources. |
| `negative` | `str \| None` | Independent/behavioral source negative terminal; `None` for controlled sources. |
| `dc` | `float` | Independent-source DC value (V or A); default `0`. |
| `ac` | `float` | Independent-source AC small-signal magnitude; default `0`. |
| `waveform` | `dict \| None` | Independent-source transient waveform spec; default `None`. |
| `pwl` | `list \| None` | Independent-source piecewise-linear flat list; default `None`. |
| `array` | `int \| tuple[int, int] \| None` | If set, this entry is an array; see contract's `Index templates` section. |
| `naming` | `str` | `"bracketed"` (default) or `"underscore"`. |

Full contract: [docs/codegen-contract.md#sources--the-single-home-for-every-source](codegen-contract.md#sources--the-single-home-for-every-source).

### `generate_testbenches` / `TestbenchSpec` — declarative testbench codegen

The testbench counterpart of `generate`: a JSON-able `TestbenchSpec` (rigs,
tests, sources, analyses, declarative metrics, specs) becomes generated Python
that uses only public silk API — `mod.create_test(name)`,
`tb.add_module_under_test(...)`, `tb.add(sources.v<kind>(...))`, `tb.add_sim_type(
analysis.<Kind>(...))`, the declarative `tb.add_metric(name, type=..., signal=...)`
and `tb.add_spec(...)`. silk owns every emitted line; the caller owns the spec.
Runnable, PDK-free demo: `python -m silk.examples.testbench_codegen_example`.

```python
from silk.codegen import Generator
from silk.codegen.testbench_generator import (
    TestbenchSpec, TestbenchSpecError, generate_testbenches,
)

code = Generator().generate_testbenches(spec)          # spec: dict | TestbenchSpec
Generator().generate_testbenches_to_file(spec, 'ac_gain_tests.py')
code = generate_testbenches(spec)                      # module-level equivalent
```

| Field | Type | Notes |
|---|---|---|
| `circuit_name` | `str` | Required. When `circuit_builder` is set it is emitted as `def build_<circuit_name>_with_tests(pdk)`, so it must then be a valid, non-keyword Python identifier. |
| `circuit_builder` | `str \| None` | Name of the caller's circuit-builder function. When given, silk emits the assembler `build_<circuit_name>_with_tests(pdk)` = `mod = <circuit_builder>(pdk)` + one `_attach_<test>(mod)` per test. `None`/`''` omits the assembler. |
| `rigs` | `list[dict]` | Shared source groups — `{"name", "sources": [SourceSpec...]}` — emitted as `def _apply_<name>_rig(tb)`. |
| `tests` | `list[dict]` | `{"name", "dut_portmap", "applied_rigs", "sources", "analyses", "metrics", "specs"}` — emitted as `def _attach_<name>(mod)`. Source entries are the same `SourceSpec` shape as `CircuitSpec.sources`; `analyses[].kind` is `transient`/`ac`/`dc`/`noise`/`op`. |

Metric entries are declarative — `{"name", "type", "signal" | "signals",
"params"}` — where `type` is any registered metric (enumerate the live catalog
with `silk.signals.export_metric_schema()`) and a signal path is a rawfile
signal name: a testbench net from `dut_portmap` (`"vout"`) for anything on a
DUT port, or `"xdut.<net>"` for a DUT-internal node — `add_module_under_test`
always instances the DUT as `dut`, and SPICE aliases its ports to the testbench
nets, so `xdut.<port>` is not in the rawfile. A one-input metric uses
`"signal": "<path>"`; a metric with N inputs uses `"signals": {"<slot>":
"<path>", ...}`, keyed by the metric's own signal-input names as published in
`signal_inputs[].name`.

**Emission contract.** `circuit_builder`
is emitted **bare** as a call target (`mod = <circuit_builder>(pdk)`), so it
must be a bare, non-keyword Python identifier — the plain function name, e.g.
`_build_ac_gain`. Dotted paths (`pkg.mod.build`), call expressions
(`_build_c()`) and statement-shaped text (`evil(); x`, quotes, newlines,
comments) raise `TestbenchSpecError` naming the field and the accepted form
before any code is produced.

**Error contract.** `generate_testbenches` raises `TestbenchSpecError` (a
`ValueError` subclass) at generate time — never uncompilable Python, never a bare
`KeyError` — for: a missing `circuit_name`, a rig/test/metric/spec missing its
`name` or a test missing `dut_portmap`, a `spec.metric` naming no metric in the
same test, an `applied_rigs` entry naming no rig, a metric with no binding, an
unknown metric `type`, an unbound or unknown `signals` slot, a missing required
metric param (`gain_at_freq` without `freq`), a source entry missing required
fields, or an unknown analysis `kind`. It also refuses requests that cannot
be satisfied: an input whose kind has an empty `produced_by` (nothing silk
simulates produces one — `p1db`, `error_norm`), a param
outside its declared `choices`, and a signal input whose kind's `produced_by`
has no analysis in common with the test's declared analyses. A metric's
`domain` groups it in the catalog; it does not restrict compatible analyses.
For example, `average` accepts AC, DC and noise inputs despite its `tran` domain. Most messages identify the owning
entry; unknown metric-type errors name the requested type and known types. See `docs/codegen-contract.md` for the full list.

The generated file needs a PDK only when it *runs*
(`build_<circuit>_with_tests(pdk)`); generating it needs none. The full JSON
shape and the design rationale are in
[codegen-contract.md#testbench-codegen](codegen-contract.md#testbench-codegen).

---

## silk.exceptions

```python
from silk.exceptions import (
    CircuitError,        # base exception
    DuplicateNameError,  # test, element, net, source, or passive name conflict
    UndeclaredNetError,  # strict_nets mode: net not declared before use
    SimulationError,     # ngspice failure or signal not found
    SimulationAborted,   # run completed but produced no usable result (subclass of SimulationError)
    PDKError,            # bad YAML, duplicate device definition
    NotSupportedError,   # a quarantined/unsupported path was called (e.g. Xyce device currents)
    CharacterizationContextError,  # mismatched TEMP assertion, partial external PVT, or relabeling
)
```

Tracking-database errors (`LockedVersionError`, `ReportStorageError`, `EvidenceMismatchError`,
`EvidenceNotFoundError`, `EvidenceAmbiguityError`, `UnsupportedSchemaVersionError`,
`UnsupportedFingerprintSchemeError`, `DatabaseCompatibilityError`) are defined
here too and re-exported from `silk.db`; their triggers are tabulated in
[database.md#failure-modes](database.md#failure-modes). Codegen's
`TestbenchSpecError` (a `ValueError`) lives in
`silk.codegen.testbench_generator` — see [silk.codegen](#silkcodegen).

### `SimulationAborted`

Raised when a simulator run finishes at the process level but produced **no
usable result** — an aborted analysis, a rawfile with zero data rows, no
rawfile, or a certified analysis that stops before its requested final grid
point — even when the simulator's exit code claims success. It is a
**subclass of `SimulationError`**, so existing `except SimulationError`
handlers keep catching it, and it carries structured fields so a UI can render
the failure reason instead of a misleading green:

| Field | Meaning |
| --- | --- |
| `simulator` | `'ngspice'` / `'xyce'` — which backend produced it |
| `reason` | human string (e.g. `'Timestep too small; time = …'`), or `None` |
| `data_rows` | Observed row count (including positive counts for incomplete analyses); `None` when no rawfile was written |
| `log_path` | public diagnostic log path for a launched ngspice/Xyce attempt, or `None` |

`SimulationError(*args, log_path=None, failed=None)` exposes the optional `log_path` attribute
on the base error too. For ngspice and Xyce post-launch failures (timeout,
non-zero exit, convergence failure or abort), it names the public log, normally
`<dir>/output.log`; the message names the same path. Xyce stdout/stderr are
captured in that file. Serial and worker attempts publish only this diagnostic
on failure, leaving no public rawfile, noise or temperature result. Cancellation
that unwinds the runner's cleanup also publishes the log before re-raising;
non-silk exceptions need no `log_path` attribute. The log remains until the next
attempt clears it or explicit cleanup removes it. Pre-launch refusals preserve
prior public artifacts. Read `err.log_path`; it may name a retained private
diagnostic if publication fails, or be `None` if no log was written.
Uncatchable termination cannot run cleanup.

```python
from silk.exceptions import SimulationAborted

try:
    module.simulate(...)
except SimulationAborted as err:
    print(f"simulation aborted: {err.reason or 'no data produced'}")
    if err.log_path:
        print(f"  see {err.log_path}")
```

The **decision** to fail is simulator-agnostic — it keys off the produced
rawfile's data presence and, for certified requests, its final analysis grid
point, rather than the exit code — so it stays correct across ngspice,
Xyce, and future simulators. See
[`simulation-error-handling.md`](simulation-error-handling.md) for the full
model.

---

## pyproject.toml essentials

```toml
[project]
name = "circuitsilk"
version = "0.2.0"

[tool.setuptools.packages.find]
include = ["silk*"]
exclude = ["silk.signals.view.assets.samples*"]

[tool.setuptools.package-data]
"silk.pdk" = ["data/*.yaml"]
"silk.circuit" = ["data/*.yaml"]
"silk.elements" = ["*.yaml"]
"silk.simulators" = ["*.yaml"]
"silk.sources" = ["*.yaml"]
"silk.signals" = ["view/assets/*"]
```

---

## silk.layout

Statistical wire-length estimation, current-aware sizing,
matched-group declarations, parasitic injection, and layout diffing.

```python
from silk.layout import (
    RoutingModel,
    HalfCircuit,
    LayoutWarning,
    NetLayout,
    DeviceLayout,
    LayoutEstimate,
    estimate_layout,
    capture_layout,
    inject_parasitics,
)
```

---

### `RoutingModel`

Frozen dataclass describing the electrical properties of a single metal layer.
Field assignment is disabled; equality compares field values. The default stack rule uses the
first layer whose minimum-width wire carries the current, falling back to the
last layer; missing current selects the first layer.

```
RoutingModel(
    sheet_resistance,   # Ω/sq
    cap_per_um2,        # fF/μm²
    j_max,              # A/μm  (maximum current density)
    min_width,          # μm
    max_width,          # μm
    thickness=0.36,     # μm  (default: 0.36 μm)
)
```

| Field | Type | Description |
|---|---|---|
| `sheet_resistance` | float | Sheet resistance in Ω/sq |
| `cap_per_um2` | float | Capacitance per unit area in fF/μm² |
| `j_max` | float | Maximum current density in A/μm |
| `min_width` | float | Minimum wire width in μm |
| `max_width` | float | Maximum wire width in μm |
| `thickness` | float | Metal thickness in μm (default 0.36); stored but not used in estimation |

**`id` property** — pipe-delimited string encoding all constants with four
significant digits, used as `LayoutEstimate.routing_model_id`. Values that
differ beyond that precision can share an ID.

```python
# Manifest-derived layer or stack (pdk may also be a registered PDK name):
m1 = RoutingModel.from_pdk(pdk, layer=None)
stack = RoutingModel.stack_from_pdk(pdk)

sky130_m1 = RoutingModel(
    sheet_resistance=0.14,
    cap_per_um2=0.034,
    j_max=1.0,
    min_width=0.14,
    max_width=10.0,
)
print(sky130_m1.id)
# 0.14|0.034|1|0.14|10|0.36
```

---

### `HalfCircuit`

Frozen dataclass representing a multi-device matched group, used for
differential or symmetric layout constraints.

```
HalfCircuit(
    name,        # str  — group label
    left,        # tuple[str, ...]  — device names on the left half
    right,       # tuple[str, ...]  — device names on the right half
    strategy=None, # str | None — stored hint; not interpreted by the estimator
)
```

| Field | Type | Description |
|---|---|---|
| `name` | str | Identifier for this matched group |
| `left` | tuple[str, ...] | Device instance names forming the left half |
| `right` | tuple[str, ...] | Device instance names forming the right half |
| `strategy` | str \| None | Optional layout strategy hint (e.g. `'interleave'`) |

Raises `ValueError` on construction if `left` and `right` have different
lengths, either side is empty, or any device name appears in both sides.

Registered via `Module.declare_half_circuit()` or the two-device shorthand
`Module.declare_match_pair()`.

```python
ckt.declare_half_circuit('diff_pair', left=('mn1',), right=('mn2',))
ckt.declare_match_pair('load_pair', 'mp1', 'mp2')  # equivalent shorthand
```

---

### `LayoutWarning`

Frozen dataclass for a non-fatal layout issue produced during
`estimate_layout()`. Warnings are run-scoped: they live on
`LayoutEstimate.warnings` and are never attached to `Net` objects.

```
LayoutWarning(
    net_name,   # str | None
    reason,     # str
    severity,   # str
    detail,     # str
)
```

| Field | Type | Description |
|---|---|---|
| `net_name` | str \| None | Net associated with the warning, or `None` for module-level warnings |
| `reason` | str | Machine-readable reason code (see table below) |
| `severity` | str | `'info'`, `'warn'`, or `'error'` |
| `detail` | str | Human-readable explanation |

**Reason codes:**

| `reason` | Meaning |
|---|---|
| `'no_current_data'` | `net.peak_current` is `None`; estimator fell back to minimum width |
| `'child_footprint_missing'` | A direct child has no stored footprint in reuse mode; uses a 1 µm² placeholder |
| `'child_has_no_geometry'` | A child estimated recursively has no device geometry |
| `'matching_conflict_large_class'` | An equivalence class used for matching contains more than two nets, which may compromise symmetry |
| `'required_width_exceeds_max'` | The selected layer's current-based required width (`I / j_max`) exceeds that layer's `max_width`; the returned single-wire width is capped. This is a limit of the single-wire estimate, not a claim that the circuit is unroutable or that any specific parallel-wire or alternate-layer arrangement resolves it — see detail |

---

### `NetLayout`

Frozen dataclass holding the estimated layout parameters for a single net.
Returned inside `LayoutEstimate.nets`.

```
NetLayout(
    length,             # μm
    width,              # μm
    r,                  # Ω
    c,                  # fF
    equivalence_class,  # str | None
)
```

| Field | Type | Description |
|---|---|---|
| `length` | float | Estimated wire length in μm |
| `width` | float | Single-wire width in μm, clamped to `[min_width, max_width]` of the selected layer. When the current-based requirement exceeds `max_width`, this is the capped value — not a width guaranteed sufficient for the current — and the run carries a `'required_width_exceeds_max'` warning |
| `r` | float | Estimated series resistance in Ω |
| `c` | float | Estimated total capacitance in fF |
| `equivalence_class` | str \| None | Label of the matched equivalence class this net belongs to, or `None` |

---

### `DeviceLayout`

Frozen dataclass holding the estimated layout footprint for a single device.
Returned inside `LayoutEstimate.devices`.

```
DeviceLayout(area)   # area in μm²
```

| Field | Type | Description |
|---|---|---|
| `area` | float | Estimated device area in μm² |

---

### `LayoutEstimate`

Frozen dataclass — the snapshot returned by `estimate_layout()` and
`capture_layout()`. Field assignment is disabled, but the contained
dictionaries remain mutable.
Estimation writes `Net` fields and `Module.bounding_box` before returning it.

```
LayoutEstimate(
    computation_mode,    # str
    routing_model_id,    # str
    timestamp,           # float
    nets,                # dict[str, NetLayout]
    devices,             # dict[str, DeviceLayout]
    bounding_box,        # tuple[float, float]
    equivalence_classes, # dict[str, tuple[str, ...]]
    warnings,            # tuple[LayoutWarning, ...]
    packing_factor=None,   # float | None
    routing_overhead=None, # float | None
    child_footprints={},  # dict[str, tuple[float, str]] (fresh dict by default)
)
```

| Field | Type | Description |
|---|---|---|
| `computation_mode` | str | `'statistical'` (from `estimate_layout()`) or `'captured'` (from `capture_layout()`) |
| `routing_model_id` | str | Single-layer `RoutingModel.id`, or stack layer IDs joined by `;` in escalation order; `''` for capture |
| `timestamp` | float | `time.time()` at the moment of creation |
| `nets` | dict[str, NetLayout] | Per-net layout snapshot |
| `devices` | dict[str, DeviceLayout] | Per-device layout snapshot |
| `bounding_box` | tuple[float, float] | `(width_μm, height_μm)` of the estimated die area |
| `equivalence_classes` | dict[str, tuple[str, ...]] | Mapping from class label to the net names in that class |
| `warnings` | tuple[LayoutWarning, ...] | Non-fatal issues raised during estimation |
| `packing_factor` | float \| None | The per-device area heuristic this run used (default 2.5); `None` for a `capture_layout()` snapshot. Round-trips through `save_json`/`load_json` |
| `routing_overhead` | float \| None | The module routing multiplier this run used (default 2.0); `None` when captured |

---

### `estimate_layout(module, routing_model, pdk=None, *, packing_factor=2.5, routing_overhead=2.0, escalation_rule=default_escalation_rule, children='reuse') -> LayoutEstimate`

Main entry point. Estimates wire lengths, sizes wires for current density,
computes R/C, and derives a bounding box.

**Units — the contract.** Layout lengths are µm and areas µm². Geometry follows
each device's declared units; catalog defaults already use PDK units. Unknown
W/L units raise `ValidationError`; unsupported device representations raise
`TypeError`.
See [adding-a-pdk.md §Units](adding-a-pdk.md#units).

**Area heuristics.** Devices without usable W/L geometry contribute a 1 µm²
placeholder without `packing_factor` scaling. Otherwise device area is
`W·L·M × packing_factor` and the module
box is `sqrt(Σ area × routing_overhead)`. Both multipliers are **adjustable
heuristics, not physical layout measurements** — the 2.5 default stands in for
source/drain diffusion, contacts and well spacing until a PDK geometry
section exists. Override them per call; the values actually used are
recorded on the returned estimate (`packing_factor`, `routing_overhead`)
and in the report's Layout Estimate section.

**Child footprints.** Each child-module instance contributes its existing
`bounding_box` width × height, already in micrometres, with no additional
device packing factor or PDK unit conversion. A missing or `None` box retains
the 1 µm² placeholder. A supplied box must contain exactly two finite,
strictly positive real numeric dimensions and have finite positive area;
malformed boxes raise `ValueError`.

The default `children='reuse'` preserves these numbers and emits a
`child_footprint_missing` warning for each direct child using the placeholder.
Use `children='estimate'` to compute every descendant footprint bottom-up with
its own `pdk_details` and this call's packing/routing multipliers. Each module
object is computed once; shared modules are reused, while cycles raise
`CircuitError` naming the instance path. Children without device geometry
(including empty or behavioral-only blocks) emit `child_has_no_geometry`.
Both warnings have `net_name=None` and name the child in their detail.
Child nets retain all their existing `estimated_*` fields: only the root's
nets are sized. Other `children` values raise `ValueError`.

Placed children are live references. Traversal estimates those same Module
objects; footprint changes before or after placement reach subsequent parent
estimates. Re-estimate the parent after changing a child footprint.

`LayoutEstimate.child_footprints` maps each direct child instance name to
`(area_um2, source)`, where source is `estimated`, `fixed`, `reused`, or
`placeholder`. It defaults to an empty dictionary (including captured or
loaded estimates). `child_footprints` is available in memory but is not
rendered or saved.

**Fixed footprints.** `silk.layout.set_footprint(module, (width_um, height_um))`
validates and stores a fixed box. It accepts the same finite positive real
numeric dimensions and finite positive area as child boxes above.
`set_footprint(module, None)` clears both the box and its fixed mark. The mark
stays on the live placed module. Estimation preserves a fixed box at every level;
a fixed child is an opaque leaf in `children='estimate'`. Estimating a fixed
root still estimates its own entries and nets against the fixed box.

Both estimation modes validate all footprints, entry areas, and net sizing
before writing state. An error leaves the module, descendants, devices, and
net estimates untouched.


Parent `routing_overhead` applies once to the sum of local device footprints
and child-block footprints, in addition to the overhead already baked into
each child. A sole 4000 µm² child therefore yields an 8000 µm² parent with
overhead 2, or 4000 µm² with overhead 1, regardless of the parent's packing
factor. The parent still uses a square approximation and a 1 µm² area floor;
these numbers remain heuristic estimates of physical layout.

**Side effects** (on the live objects):

- Sets `net.estimated_length`, `net.estimated_width`, `net.estimated_r`,
  `net.estimated_c` on each `Net` in `module.net_dict`.
- Sets `estimated_area` on each device or child-module instance in the module.
- Sets `module.bounding_box` to `(width_μm, height_μm)`.

Returns a `LayoutEstimate` snapshot of all computed values.

| Parameter | Type | Description |
|---|---|---|
| `module` | Module | Circuit to estimate |
| `routing_model` | RoutingModel or ordered stack | One layer, a dict of layer names to models, or a sequence of models, lowest layer first |
| `escalation_rule` | callable, keyword-only | `(peak_current, stack) -> RoutingModel`; used only for a stack. Defaults to `default_escalation_rule` |
| `pdk` | BasePDK \| None | PDK whose unit declarations resolve device geometry; defaults to `module.pdk_details` |
| `packing_factor` | float, keyword-only | Local device footprint multiplier (`W·L·M × packing_factor`); existing child footprints are not rescaled. Default 2.5, a heuristic. Must be finite and positive |
| `routing_overhead` | float, keyword-only | Module-level multiplier over summed local device and child-block footprints, additional to child overhead. Default 2.0, a heuristic; same validation |
| `children` | str, keyword-only | `'reuse'` (default) keeps stored child boxes; `'estimate'` computes descendant footprints using each child's PDK, leaving child nets untouched |

```python
from silk.layout import estimate_layout, RoutingModel

m1 = RoutingModel(0.14, 0.034, 1.0, 0.14, 10.0)
est = estimate_layout(ckt, m1)

print(est.bounding_box)          # (width_μm, height_μm)
print(est.nets['vout'].r)        # Ω
for w in est.warnings:
    print(w.reason, w.detail)

# A denser device style: half the local channel-based device footprint.
tight = estimate_layout(ckt, m1, packing_factor=1.25)
print(tight.packing_factor)      # 1.25 — recorded provenance
```

---

### `capture_layout(module) -> LayoutEstimate`

Reads the current `Net` and device state without recomputing anything.
Useful for snapshotting layout fields that were set by a previous
`estimate_layout()` call or injected manually.

Returns a `LayoutEstimate` with `computation_mode='captured'` and an empty
`warnings` tuple. The `routing_model_id` is always `''` — it is not
read from the nets at capture time. Only nets with `estimated_length` and
devices with `estimated_area` are captured; missing net width/R/C become zero.
Matching classes are not reconstructed; keep the original estimate for that metadata.

```python
# After a prior estimate_layout() call:
snap = capture_layout(ckt)
assert snap.computation_mode == 'captured'
assert len(snap.warnings) == 0
```

---

<a id="inject_parasiticsmodule-estimate-moderc---none"></a>

### `inject_parasitics(module, estimate, mode='rc', *, return_net='0') -> None`

Injects lumped parasitic elements into the module netlist based on a
`LayoutEstimate`. The call is idempotent: re-running with the same estimate
replaces previously injected parasitics rather than doubling them.

| Parameter | Type | Description |
|---|---|---|
| `module` | Module | Target circuit (mutated in place) |
| `estimate` | LayoutEstimate | Source of R/C values |
| `mode` | str | `'rc'` (default) or `'c_only'` (see below) |
| `return_net` | str | Keyword-only capacitor return: global ground `'0'` by default, or a declared net/port such as `'vss'` |

The estimated net equal to `return_net` gets no branch in either mode.
An undeclared return (other than `'0'`) or estimated net raises
`UndeclaredNetError`. Mode and all net names are checked before any previous
injection is removed; a rejected call preserves the circuit. Generated-name
collisions with unrecorded passives or midpoint nets raise `DuplicateNameError`.

Each injection records exactly the passives and nets it adds, including `'0'`
if newly created. Re-injection removes only those recorded objects and their
pin bindings. Unrecorded nets and passives are never deleted by name.

Placed submodules are live references: injection after placement reaches the
parent deck on its next generation. With
`return_net='vss'`, capacitor returns follow the parent's binding of that
port; the default `'0'` always means global ground.

**`mode='rc'` (lumped-π loading branch)**

Every original device terminal stays connected to the original net, for
both two-pin and multi-pin nets. For each estimated net, the injector adds
`__par_R_<net>` from that net to `__par_<net>_mid`, `__par_C1_<net>` from
the original net to `return_net`, and `__par_C2_<net>` from the midpoint to `return_net`:

```text
all original terminals remain on <net>
                       |
                     <net> ---- R ---- __par_<net>_mid
                       |                    |
                      C/2                  C/2
                       |                    |
                  return_net           return_net
```

The midpoint connects only the newly added R and C2 terminals. Only C2 is
reached through the injected resistor; no original load pin is moved behind
it. `R = estimate.nets[net].r` in ohms. Total capacitance is
`estimate.nets[net].c` in fF, converted to farads by `c_farads` and split
equally between C1 and C2.

This is a useful approximation of added RC loading. It does **not** place
series resistance between original circuit terminals, model DC wire drop,
choose branch-specific RC, or represent routed inter-terminal delay. An AC
bandwidth change demonstrates loading, not those unsupported effects.
The injector selects no driver and emits no multi-pin approximation warning.

**`mode='c_only'`**  
A single lumped shunt capacitor `__par_C_<net>` (full `C`, converted from
fF to F) connects the original net to `return_net`. All original connections stay
unchanged; no resistor or midpoint is created. This is suitable for
first-order capacitive-loading estimates. Neither mode provides signoff
or extracted-layout accuracy. Switching from `rc` to `c_only` removes the
previously injected RC elements and midpoint nets.

Silk already records connectivity through `Net.pins` / `PortBinding`, signed
per-pin current summaries and current-vector provenance, conservative net
current aggregates, and `HalfCircuit`-matched net estimates. Those data do
not choose a physical branch topology or map terminals to RC nodes. Physical
branch rewiring, distributed RC and driver selection are unsupported.

```python
from silk.layout import estimate_layout, inject_parasitics, RoutingModel

m1 = RoutingModel(0.14, 0.034, 1.0, 0.14, 10.0)
est = estimate_layout(ckt, m1)
inject_parasitics(ckt, est, mode='rc')
# ckt now contains __par_<net>_mid nets and parasitic R/C elements
```

---

### The routing-parasitic feedback loop

Estimate → inject → simulate → observe.

`estimate_layout` and `inject_parasitics`, followed by a SPICE
`module.simulate()` or testbench `tb.run()`, turn statistical estimates into
added loading that can change the simulated response. Shunt capacitance on
a **high-impedance node** can move its dominant pole down and reduce
bandwidth; the effect depends on the circuit and node impedance. This
demonstrates loading sensitivity, not branch-aware routing or extracted
post-layout accuracy. The behavioral engine does not solve the injected RC
network electrically.

```python
from silk.layout import estimate_layout, inject_parasitics, RoutingModel
from silk.metrics import bandwidth_3dB

# 1. Baseline — before any routing parasitics exist.
report_before = tb.run(corner='tt')
bw_before = bandwidth_3dB(report_before.sim_results.load_signal('vout'))

# 2. Estimate per-net R/C from the statistical wirelength model, then inject
#    a loading branch (__par_R_*, __par_C1_*/__par_C2_*) onto the DUT.
est = estimate_layout(dut_module, RoutingModel(0.08, 1.0, 1e-3, 0.14, 10.0))
inject_parasitics(dut_module, est, mode='rc')

# 3. Re-simulate the same bench and observe the live DUT changes.
report_after = tb.run(corner='tt')
bw_after = bandwidth_3dB(report_after.sim_results.load_signal('vout'))
assert bw_after < bw_before   # shunt C on a high-Z node lowers bandwidth
```

`tests/layout/test_parasitics_effect.py` runs this exact loop against the 5T
OTA's high-impedance `VOUT` node (current-mirror load, no explicit load cap)
and checks that a single `inject_parasitics(mode='rc')` call reduces
bandwidth by more than 10%. This is evidence
of added loading, not inter-terminal resistive voltage drop or wire
propagation delay. `ModuleTestbench.add_module_under_test()` keeps a live DUT
reference: a bench built before `inject_parasitics()` sees the added elements
on its next run. Rebuilding the bench is optional.


---

### `Module` additions (layout)

These attributes and methods are available on any `Module` instance after the
`silk.layout` package is imported.

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `module.bounding_box` | tuple[float, float] \| None | `(width_μm, height_μm)` set by `estimate_layout()`; `None` before first estimate |
| `module.half_circuits` | dict[str, HalfCircuit] | All declared matched groups, keyed by group name |

**Methods:**

#### `module.declare_half_circuit(name, left, right, strategy=None)`

Declare a multi-device matched group. Constructs a `HalfCircuit` and stores
it in `module.half_circuits`.

| Parameter | Type | Description |
|---|---|---|
| `name` | str | Group identifier |
| `left` | tuple[str, ...] | Device names on the left half |
| `right` | tuple[str, ...] | Device names on the right half |
| `strategy` | str \| None | Optional layout strategy hint |

#### `module.declare_match_pair(name, a, b)`

Shorthand for declaring a two-device matched pair. Equivalent to
`declare_half_circuit(name, left=(a,), right=(b,))`.

| Parameter | Type | Description |
|---|---|---|
| `name` | str | Group identifier |
| `a` | str | First device instance name |
| `b` | str | Second device instance name |

```python
ckt.declare_half_circuit(
    'diff_pair',
    left=('mn1', 'mn3'),
    right=('mn2', 'mn4'),
    strategy='interleave',
)
ckt.declare_match_pair('load', 'mp1', 'mp2')
```

---

### `Net` additions (layout fields)

These fields are set to `None` on `Net` construction and populated by
`estimate_layout()`. They can also be set manually before calling
`capture_layout()`.

| Attribute | Type | Description |
|---|---|---|
| `net.estimated_length` | float \| None | Estimated wire length in μm |
| `net.estimated_width` | float \| None | Estimated wire width in μm |
| `net.estimated_r` | float \| None | Estimated series resistance in Ω |
| `net.estimated_c` | float \| None | Estimated total capacitance in fF |
| `net.peak_current` | float \| None | Peak current in A; read by `estimate_layout()` for current-aware wire sizing. Set this before calling `estimate_layout()` to enable density-driven sizing |
| `net.rms_current` | float \| None | RMS current in A (stored, not yet used by the estimator) |
| `net.avg_current` | float \| None | Average current in A (stored, not yet used by the estimator) |

```python
ckt.net_dict['vout'].peak_current = 0.5e-3   # 0.5 mA
est = estimate_layout(ckt, m1)
print(ckt.net_dict['vout'].estimated_width)   # sized for 0.5 mA
```

---

### `CircuitReport` additions (layout)

`generate_report()` accepts an optional `layout_estimate` keyword argument.

```python
from silk import generate_report

report = generate_report(ckt, layout_estimate=est)
```

**`report.layout_estimate`** — `Optional[LayoutEstimate]`  
Attached at report construction time; `None` if not provided.

**`_to_markdown()` rendering**  
When `layout_estimate` is set, the Markdown output includes a
`## Layout Estimate` section containing:

- Bounding box dimensions (width × height in μm).
- A table of nets with `length`, `width`, `R`, `C`, and `equivalence_class`
  columns.

**`save_json()` serialization**  
When `layout_estimate` is set, the JSON output includes a top-level
`'layout'` key containing the layout report fields. The top-level
`equivalence_classes` map and `child_footprints` are omitted and empty after reload.

---

### `ReportDiff` additions (layout)

When both reports passed to `diff_reports()` carry a `layout_estimate`, a
`LayoutDiff` is computed automatically and attached to the diff object.

#### `LayoutNetDelta`

Per-net delta dataclass. All delta properties are signed (`new − old`).

| Property | Type | Description |
|---|---|---|
| `delta_r` | float | Change in series resistance (Ω) |
| `delta_c` | float | Change in total capacitance (fF) |
| `delta_length` | float | Change in wire length (μm) |
| `delta_width` | float | Change in wire width (μm) |

#### `LayoutDiff`

| Field / Property | Type | Description |
|---|---|---|
| `changed` | list[LayoutNetDelta] | Nets whose layout parameters changed, sorted by name |
| `unchanged` | list[str] | Nets whose length, width, R and C each differ by less than `1e-12`, sorted by name |
| `nets_added` | list[str] | Nets present in the new report but not the old, sorted by name |
| `nets_dropped` | list[str] | Nets present in the old report but not the new, sorted by name |
| `bbox_a` | tuple[float, float] | Bounding box of the baseline report |
| `bbox_b` | tuple[float, float] | Bounding box of the new report |
| `bbox_area_delta` | float (property) | `area(bbox_b) − area(bbox_a)` in μm² |

**`diff.layout_diff`** — `Optional[LayoutDiff]`  
`None` when either report lacks a `layout_estimate`.

```python
from silk import diff_reports

d = diff_reports(report_a, report_b)
if d.layout_diff:
    ld = d.layout_diff
    print(f"Bounding box area delta: {ld.bbox_area_delta:.1f} μm²")
    for delta in ld.changed:
        print(f"  {delta.name}: ΔR={delta.delta_r:+.3f} Ω  ΔC={delta.delta_c:+.2f} fF")
```

### Test-result collection JSON adapters

Import from `silk.circuit.spec_report`:

| Function | Contract |
|---|---|
| `results_to_dict(results)` | Fresh `silk.test-results` wire data for a tests dict, `CornerResults`, `MonteCarloResults`, or `SpecReport` list. Versions 1/2 retain legacy collections; version 3 retains grid coordinates, planned inventory and Monte Carlo metadata. Preserves keys, order, failures and collection policy. |
| `save_results_json(results, path)` | Validate and encode before opening a strict JSON destination. |
| `results_from_dict(payload)` | Restore wire data into detached dict/corners/list objects; no circuit, simulator or rawfile needed. |
| `load_results_json(path)` | File reader with file/field diagnostics. |

Unsupported Monte Carlo/custom collections are rejected explicitly. See
[collection and v3 evidence](reports.md#collections-and-complete-circuitreport-evidence)
for node shapes and the v3 compatibility boundary.


### `PlacementReference` and placement lookups

`silk.circuit.placement_reference.PlacementReference` is the structural target
returned by a placement's net, port or bus attribute. Bind it with
`tb.probe(stage.vbias)` to receive a deferred Signal; it does not acquire data
at construction. Direct reads and arithmetic require one eligible live result
context. Net/port references mean voltage, including literal selector-looking
names. See [probes](probes.md#probe-a-placement-by-object) for lifetime and
selection rules.

Declared design names take precedence over helpers: if `stage.net` is hidden,
call `SubcircuitInstance.net(stage, name)` instead; the same class-qualified
form works for the other helpers.

`SubcircuitInstance.net(name)`, `.bus(name)`, `.child(name)` and `.device(name)`
select a kind explicitly and reach reserved or nonidentifier design names.
Cross-kind attribute ambiguity refuses. `PlacementArray` groups actual module
or device keys `name[k]`; integer subscripts select declared numbers and slices
retain those numbers. Net, Signal and reference buses use the same indexing
rule, including width-one aggregates.

Device lookups (`stage.m1`, `stage.device('m1')`, and members of a device
array) return read-only `silk.circuit.device_view.PlacementDeviceView` objects.
Each view is a distinct object: `stage1.m1 is stage2.m1` and `stage1.m1 is m1` are false, and its
`type()` and `isinstance()` results may differ from the original device.
Public attribute assignment and deletion raise `AttributeError`, naming the
holding module and asking you to edit the device there. The `params`, `nets`
and `layout` mappings are read-only; their values remain shared, including
live `Variable` values. This protection does not recursively freeze nested
objects. Legitimate changes to the original device and holding-module edits
remain visible through a held view.

Views support the existing `element_names`, `emitted_instance_name` and
`device_geometry_um` inspection helpers for the representations those helpers
already accept, including SI geometry. Placement and device insertion APIs
refuse views; pass the original device when adding it to a module. Whole-device
object probes remain unsupported: use net/port/bus references or string device
parameter and terminal probes.

A view retains its existing placement owner handle and original leaf identity.
The owner supplies ancestry and validates ancestor placements on reads. Reading it or
creating a fault from it raises `CircuitError` after any ancestor placement or
the leaf device is removed or replaced; obtain a new view. Copy, deepcopy and
pickle preserve the view and its graph references (shallow copies retain shared
children). Fault helpers keep the view's own target path, using authored keys
such as `xstage[8].bank[5]`, regardless of later lookups through other placements.
Fault application is [placement-local](#silkfaults--fault-injection-and-monte-carlo-campaigns):
the retained complete target selects only that placement, even when siblings
share a subcircuit definition. Duplicate requests to the same complete target
still apply only the first fault.

### Bus Signal maths and bench bus references

A bus `Signal` (from `tb.probe(stage.bus('d'))`) supports `+ - * / **`, unary
minus and the Signal-to-Signal transforms element-wise, like a numpy array.
For `+ - * /`, two buses of equal length combine by position, and a number or
a non-bus Signal applies to every bit on either side; buses of different length
raise `silk.exceptions.BusPairingError` (`ConfigurationError` and `ValueError`),
naming both widths. `**` takes a real-number exponent and applies it to every
bit. A maths result made from one bus keeps that bus's
declared bit numbers, including scalar arithmetic, powers, unary minus and
per-bit transforms. For `d` declared from 8, `(d * 2)[8]` and `d.mag()[8]`
use `d[8]`; `[0]` raises. Chains such as `(d * 2).mag()` keep those numbers.
A result combining two buses is numbered from 0 by position (`e = d + c`
gives `e[0] = d[8] + c[0]`), even when their declared numbers match.
`as_int()`, `len`, iteration and declared-number indexing are unchanged.
See [probes](probes.md#bus-maths-element-wise-like-numpy).

`ModuleTestbench.bus(name)` returns a
`silk.circuit.placement_reference.BenchBusReference` for a bus declared on the
bench's own netlist. It subscripts by declared number like a placement bus
reference and binds with `tb.probe(tb.bus(name))`. Unknown names, a reference
from another bench and a current (`type='I'`) probe raise `CircuitError`.

`ModuleTestbench.probe_bus(names, type='V', analysis=None)` returns a bus
`Signal` over `tb.probe(name, type=type, analysis=analysis)` for each string in
`names` (a list or tuple, at least one entry). The result is a new ordered
aggregate numbered 0..N-1 whose leaves keep their probe paths; it does not
rename or redeclare any circuit bus. A bare `str` and a non-list/tuple raise
`TypeError`, as does a non-string element (the message names its index);
`[]` and `()` raise `silk.exceptions.ConfigurationError`; errors from `probe`
propagate unchanged. Repeats are allowed here. See
[probes](probes.md#bus-maths-element-wise-like-numpy).

### Checked waveform node context

`silk.circuit.view_node_context(owner)` accepts a `Module`, `ModuleTestbench`,
or its selected-test proxy and returns `silk.signals.view.ViewNodeContext` or
`None`. It follows the owner's result route and checks the association between
results and circuit nodes against a fresh disk read. Unsupported, stale or missing associations
return `None`; incorrect owner types raise `TypeError`, and reader errors
propagate. Call it as a package function, not as a Module method.

`ViewNodeContext.source` is a borrowed read-only `RawFile`. Obtain a new context
for each emission; do not mutate its plots, vectors or samples. The context is
opaque and cannot be supplied as an arbitrary dictionary.

`emit_view(..., node_context=None)` and `emit_view_data(..., node_context=None)`
accept that checked context. The context links the owner's results to circuit
nodes and enables designer selectors and copyable probe expressions.
`labels='designer'` additionally enables graph-backed display labels. Omit the
context for rawfile-only output. See [the payload and size contract](explorer-integration.md#checked-designer-node-metadata).
