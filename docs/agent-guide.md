# silk Agent Guide

**Front door for AI agents building analog circuit examples with silk.**

Read this before writing code. The workflow, mental model, and decision
guides are here. Signatures and field tables are in the references — this
guide links to them by anchor and does not restate them.

Command-line tools (PDK validation, schema export, role-rule checking,
codegen) are collected in [docs/cli.md](cli.md).

Extending silk itself (a new metric, PDK, backend, evidence type, emission
site) is a different job from using it: the package map, data-flow trace,
implementation rules and the code that enforces them live in
[docs/architecture.md](architecture.md). This guide does not restate them.

Task-oriented pages (reports, operating point, Monte Carlo, lint, the
design-stage bridge, error handling) are triaged in
[docs/README.md](README.md); look there when this guide has no row for a job.

---

## 1. What silk is + the build → characterize → report → netlist loop

Silk is a Python library for analog/mixed-signal circuit design automation.
A designer describes a circuit as a `Module` — a live Python object holding
devices, nets, sources, and passives — then runs simulations (behavioral or
SPICE), evaluates metrics against specs, generates circuit reports, and
emits SPICE netlists, all from the same object.

The canonical loop is:

```
build → characterize → report → netlist
```

The whole loop — including the persist → restore and layout-feedback steps
that make it a *design* loop rather than a script — runs end to end in
`silk/examples/design_loop_walkthrough.py` (`python -m
silk.examples.design_loop_walkthrough`; §7 below). Read that after §4.

1. **Build** — construct a `Module`, add transistors via the PDK, declare
   ports, and optionally attach a behavioral model.
2. **Characterize** — create a `ModuleTestbench` via `mod.create_test()`,
   wire it, add sources, declare metrics and specs, then call `tb.run()`
   or `tb.run_corners()`.
3. **Report** — call `generate_report(mod)` to get a `CircuitReport` with
   topology, annotation coverage, interface violations, and spec outcomes.
   Save as Markdown, JSON, or self-contained HTML. See
   [reports.md](reports.md) for the full model (one-call harvest, not
   incremental) plus `diff_reports()` for comparing two runs.

   **After simulation, open the waveform explorer explicitly:**

   ```python
   mod.simulate()
   mod.view()                     # emit view.html beside the rawfile and open it
   mod.view(open=False)           # emit only (headless); returns the html path
   mod.view(signals=['xdut'])     # same selector grammar as the CLI / emit_view
   ```

   `mod.view()` is a thin wrapper over `emit_view`, defaulting the output path
   (beside `mod.results_file`) and opening a browser. (`view()` is a `Module`
   method; for a `ModuleTestbench` use `tb.netlist.view()`.) `silk.view("sim.raw")`
   — or `silk.view(mod)` for a simulated `Module` — is the same shortcut at
   package level; it does not accept a `ModuleTestbench` object, so from a
   testbench pass `tb.results_file` (or use `report.add_waveform_view(tb)`,
   which does accept one). For explicit control, call `emit_view` directly:

   ```python
   from silk.signals.view import emit_view
   emit_view(tb.results_file, 'view.html')
   ```

   For one shareable HTML file, pass `inline=True`; otherwise keep the
   `.data.js` and `.stamp.js` sidecars beside the HTML.

   To link the viewer from a report, call `report.add_waveform_view(tb)`. See
   [the viewer CLI](cli.md#silksignalsview--waveform-explorer) for selectors,
   split pages, provenance and startup commands. `emit_view(signals=...)`
   filters the page; `tb.netlist.set_save_list([...])` restricts what the
   simulator writes, producing a partial rawfile. See
   [simulator-save-list.md](simulator-save-list.md).

4. **Netlist** — call `mod.generate_netlist()` to emit SPICE. Dual-mode:
   returns a `.subckt` string when `module_name` is set, writes a top-level
   file otherwise. A top-level SPICE deck requires a SPICE backend; a
   behavioral-selected subcircuit can still emit its `.subckt`.

**Objects to know:**

| Object | Role |
|---|---|
| `Module` | The circuit. Holds devices, nets, sources, passives, tests, and specs. |
| `ModuleTestbench` | Test harness returned by `Module.create_test()`. Wires the DUT, runs sim, evaluates specs. |
| `BasePDK` / `load_pdk` / `UNIT_PDK` | PDK layer. W/L in µm (manifest `user_units: u`). |
| `silk.sources` | Voltage / current / controlled / behavioral sources and waveform shapes. |
| `Signal` / `NgspiceSignal` | 1-D waveform with `.x`, `.y`, and arithmetic operators. `.plot(ax=...)` is AC-aware: real signals plot as x vs y; complex AC sweeps default to magnitude-in-dB on a log-frequency axis (override with `mode=`/`logx=`). |
| `Transient` / `AC` / `DC` / `Noise` | Analysis type objects passed to `add_sim_type()`. |
| `silk.metrics` | **The** metric library, organised by what each metric measures: AC/waveform (`dc_gain_dB`, `bandwidth_3dB`, `phase_margin`), ADC / spectral / timing / noise. |
| `silk.specs` | `Spec`, `Measurement`, `CalibrationBinding` — first-class hierarchical spec objects. |


### One workflow, two engines

Select `dut.simulator = 'behavioral'` to evaluate Python/numpy models without
an external simulator or installed PDK models. Tests inherit the DUT's setting
at each run; `tb.netlist.simulator = 'behavioral'` overrides it for one test.
Probes, metric bindings, specs, `tb.run()`, `dut.run_all_tests()` and
`tb.run_corners()` use the same workflow as SPICE. A twin block has both SPICE
contents and a behavioral model: changing the setting chooses which runs.

This complete example uses the packaged catalog only. Run with
`MPLBACKEND=Agg`; files go under `out/two-engines/`:

```python
from silk import Module, PortDir, load_pdk, sources
from silk.metrics import final_value
from silk.simulation import Transient


def build_gain():
    dut = Module('PDK_skywater130', 'out/two-engines', 'gain.cir',
                 pdk_details=load_pdk('sky130'), module_name='gain')
    dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
    dut.add_source(sources.VoltageControlledVoltageSource(
        'gain', out_pos='vout', out_neg='0',
        ctrl_pos='vin', ctrl_neg='0', gain=2))
    def model(ports, t, params):
        ports.vout.v = 2 * ports.vin.v

    dut.set_behavioral_model(model, execution='vector')
    dut.simulator = 'behavioral'
    tb = dut.create_test('transfer')
    tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
    tb.add_source(sources.VoltageSource(
        'drive', positive='vin', negative='0', dc=0.5))
    tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
    tb.add_metric('final_V', final_value, sig=tb.probe('vout'))
    tb.add_spec('Output', 'final_V', '>=', 0.9)
    return dut


dut = build_gain()
tb = dut.get_test('transfer')
vout = tb.probe('vout')
report = tb.run()
assert report.passed and report.metrics['final_V'] == 1.0
assert len(vout.y) == 11                  # includes the stop sample
fig = tb.plot(vout, show=False)
fig.savefig('out/two-engines/transfer.png')
viewer_path = tb.netlist.view(open=False) # same rawfile-backed viewer as SPICE
```

With ngspice installed, continue with the **same test object**. No PDK models
are needed for this ideal gain source:

```python
dut.simulator = 'ngspice'
spice_report = tb.run()
assert spice_report.metrics.keys() == report.metrics.keys()
assert [r.name for r in spice_report.results] == [r.name for r in report.results]
assert abs(spice_report.metrics['final_V'] - 1.0) < 1e-9
```

**Hierarchy:** a model is a leaf, replacing everything inside that block,
including transistors. A block without a model is entered; its internal nets
are readable as `xdut.<net>`. An empty unmodelled block raises
`SimulationError`; an unmodelled transistor raises `NotSupportedError` with
its instance path. Model-only blocks cannot run as SPICE: add SPICE contents
or select the behavioral backend. Declare port directions on modelled blocks.

**Ports-form models:** pass `execution='vector'` or `execution='sample'`
to `set_behavioral_model` to choose the calling contract for `fn(ports, t, params)`;
vector is the default. Read and
assign `ports.X.v` (net voltage) and `ports.X.i` (current into the pin), and
return `None`. `params.gain` and `params['gain']` read the same live parameter;
attributes expose parameters only. Bus values keep their bus axis:
`(width, T)` for vector calls, `(width,)` for sample calls. `ports.code[k]`
uses the declared bit number. Assign once per sample; revise a local value
before publishing. Reads of arrays cannot modify electrical storage. One
unknown current per net can be filled by KCL; a second unknown refuses.
See the [complete model interface](api-reference.md#behavioralengine).

**Transient:** vector models receive arrays over a fixed grid from zero through
`stop`. `maxstep` caps `step`; a final shorter interval includes an off-grid
stop. `start` discards earlier samples after evaluation, so stateful models
still see the lead-in. A loop containing a ports-form model runs its members
in placement order, each call to completion. Sample loops advance in that
order at each point: a read uses the producer's current published value,
otherwise its most recent earlier value or the reader's initial value. Vector
loops run each complete array call once, reading initial values where the
producer has not yet run. A loop mixing sample and whole-array calls raises
`LoopError` whatever the port types; choose the same `execution=` or put the
loop in one function. Mixed-contract loops visible through declared outputs refuse before
model calls. Loops found only through actual INOUT reads and writes refuse
at the end of evaluation, before any result is returned or published.
INOUT ordering and notices stay the same.

An undelayed transient feedback loop warns once per run, even if its values
are equal. Use `ports.FB.v.previous_value` in a sample model, or
`ports.FB.v.previous_value[i]` in a vector model, for one point of explicit
delay. A delay on a delayed view adds another point. At the first point, the
view reads `ports.FB.v.initial_value`, a plain value defaulting to zero.
Initial values and delays affect only this model's reads; they never reset
or fill the producer's waveform. Pin currents (`ports.X.i`) and buses expose
the same interface. An explicit delay breaking the feedback cycle removes
the feedback warning.

**DC:** a feedback loop raises `LoopError`, a `SimulationError` subclass,
after the point's calls finish and before results are published. A pure
voltage watcher runs. The error names the point, models and reads. Reading
`previous_value` raises `SimulationError` immediately, including on a
feedforward path: a DC point has no previous time point. Branch on
`silk.analysis()` to use delays only in transient. Setting `initial_value`
is allowed and has no effect in DC.

Continue from `build_gain()` above to register a source sweep:

```python
from silk.simulation import DC

dut.simulator = 'behavioral'
dc_tb = dut.create_test('dc_transfer')
dc_tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
dc_tb.add_source(sources.VoltageSource(
    'drive', positive='vin', negative='0', dc=0))
dc_tb.add_sim_type(DC('drive', 0, 1, 0.25))
dc_out = dc_tb.probe('vout', analysis='dc')
dc_tb.add_metric('top_V', lambda sig: float(sig.y[-1]), sig=dc_out)
dc_report = dc_tb.run()
assert dc_out.x.tolist() == [0, 0.25, 0.5, 0.75, 1]
assert dc_out.y.tolist() == [0, 0.5, 1, 1.5, 2]
assert dc_report.metrics['top_V'] == 2
```

DC calls each model once per point, in sweep order, with single-sample arrays
for vector models, or scalars for sample models.
The model's `t` is the increasing point index, not time or a settling interval.
Each ports-form placement has independent callable state. Keep hysteresis
state on the callable or in its closure; local variables reset
on every call. Unswept sources use `dc + waveform.initial_value()`; a DC
sweep replaces only the source's `dc` shift. Transient and DC can
coexist; select `analysis='tran'` or `'dc'` on probes when needed. AC, noise,
operating-point and ambient-temperature requests are refused.

Sample models run point by point in declared model order; each vector call
finishes over its complete grid. Drivers precede readers using declared
IN/OUT dependencies; INOUT warns and retains placement order. During each
dispatched analysis, `silk.analysis()` returns `'dc'` or `'transient'`, also
in synchronous helpers. Outside a run it raises `SimulationError`.

**Results and evidence:** behavioral runs write a nutmeg rawfile and set
`results_file`; `load_signal`, signal arithmetic, plots and `emit_view` read it
through the shared reader. `evaluate()` retains in-memory silk evidence:
`origin='silk'`, `backend='behavioral'`, corner and temperature none
(`proven-behavioral`). `evaluate(rawfile=...)` instead records external
evidence. Measurements and `store_report()` derive their source from that
evidence; explicitly storing a contradictory source raises
`EvidenceMismatchError`. Behavioral tests stay in the parent process whatever
`n_workers` requests. Distinct corner labels warn once that behavioral runs
are identical; a label is not process-corner evidence.

**Transient shortcut:** `simulate_behavioral(t_step, t_stop)` runs
an implicit transient through this same backend, then restores the caller's
analysis declarations and selected simulator. Results include `t_stop`,
expose rawfile names such as `v(vout)`, and support `results_file` and the
viewer. Node-current requests raise instead of returning voltage arrays;
analysis selection and probe refusals follow the loaded evidence even after
the prior simulator setting is restored. `Net.voltage` remains populated.

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
Current sources contribute to `ports.X.i` through KCL; they do not drive net
voltages. Transient results also retain their branch-current waveforms. A net
connected only to current sources behaves as an ordinary undriven net, with
no additional current-source warning or refusal. A current source can share a
net with a voltage producer; the voltage stays with that producer. See [known issues](known_issues.md#behavioral-simulation).
Ignored passive loads remain approximations and do not establish references.

See the [probe comparison](probes.md#behavioral-and-spice-probes) and
[behavioral limitations](known_issues.md#behavioral-simulation) for element
approximations and every remaining refusal. The verification commands there
cover these contracts. The three snippets above are executed verbatim by
`.venv/bin/python -m pytest -q tests/test_docs_snippets.py`; the ngspice
continuation skips with a reason when the simulator is absent.

---

## 2. Feature map

Every row links to the reference anchor where signatures and field tables live.

| Capability | Module | Entry symbols | Reference anchor | Best example |
|---|---|---|---|---|
| Build / hierarchy | `silk.circuit` | `Module`, `with mod:` + `pdk.<device>(...)`, `add_module` (hierarchical), `define_module_ports` | [api-reference.md#module](api-reference.md#module) | `silk/examples/ota_5t_guide.py` |
| Sources | `silk.sources` | `VoltageSource`, `CurrentSource`, `Pulse`, `PWL`, `Sine`, `BitPattern`, `PRBS`, `prbs_bits` | [api-reference.md#silksources](api-reference.md#silksources) | `silk/examples/source_examples.py` |
| SPICE simulation | `silk.simulation` | `Transient`, `AC`, `DC`, `Noise`, `Module.simulate` | [api-reference.md#silksimulation](api-reference.md#silksimulation) | `silk/examples/ota_5t_guide.py` |
| Behavioral simulation | `silk.simulation.behavioral` | `Module.simulate_behavioral`, `set_behavioral_model` | [api-reference.md#behavioralengine](api-reference.md#behavioralengine) | `silk/examples/behavioral_example.py` |
| Testbench authoring | `silk.circuit` | `ModuleTestbench`, `create_test(name, parameters=)`, `probe`, `add_metric(name, fn, sig=...)`, `set_temperature`, `set_supply_voltage`, `run(corner)`, `run_corners`; callable and declarative metrics also supported | [api-reference.md#moduletestbench](api-reference.md#moduletestbench) · [probes.md](probes.md) | `silk/examples/strongarm_tb.py` |
| Operating point / device parameters | `silk.simulation` | `Op`, `Module.save_device_params`, `SpecReport.op_table()` | [operating-point.md](operating-point.md) | `silk/examples/gm_id_lookup.py` |
| Re-measure a saved rawfile in a new process | `silk.circuit` | rebuild the factory, then `tb.evaluate(rawfile=path)` → `SpecReport`; `SpecReport.load_json` restores values only | [reports.md#complete-standalone-test-records](reports.md#complete-standalone-test-records) · [probes.md](probes.md) | — |
| Waveform explorer | `silk.signals.view` | `mod.view()` / `tb.netlist.view()`, `silk.view(source)`, `emit_view`, `emit_views_by_block`, `list_blocks`, `report.add_waveform_view` | [cli.md#silksignalsview--waveform-explorer](cli.md#silksignalsview--waveform-explorer) · [api-reference.md#module](api-reference.md#module) (`view` row) | — |
| AC/waveform metrics | `silk.metrics` | `dc_gain_dB`, `bandwidth_3dB`, `phase_margin` | [api-reference.md#waveform--ac-metrics-silkmetricsac-silkmetricsnoise](api-reference.md#waveform--ac-metrics-silkmetricsac-silkmetricsnoise) | `silk/examples/ota_5t_guide.py` |
| Window statistics | `silk.metrics` | `average`, `peak_to_peak` (`start`/`end`), `settling_time`, `final_value`, `value_at`, `window` | [metrics-library.md](metrics-library.md) | `silk/examples/current_mirror_example.py` |
| ADC / spectral metrics | `silk.metrics` | `sndr`, `enob`, `sfdr`, `inl`, `dnl` | [docs/metrics-library.md](metrics-library.md) | `silk/examples/flash_adc_example.py` |
| Timing / jitter metrics | `silk.metrics` | `average_period`, `average_duty`, `period_jitter`, `cycle_to_cycle_jitter` | [docs/metrics-library.md](metrics-library.md) | `silk/examples/ring_oscillator_example.py` |
| Specs / reconciliation | `silk.specs` | `Spec`, `Measurement`, `CalibrationBinding`; `Module.reconcile_specs()` (a Module method, not a `silk.specs` export). Keep the `Spec` that `declare_spec` / `add_spec` returns; `reconcile_specs()` reports on the rest | [design_stage_bridge_user_guide.md](design_stage_bridge_user_guide.md) · [api-reference.md#silkspecs--design-stage-bridge](api-reference.md#silkspecs--design-stage-bridge) | `silk/examples/flash_adc_example.py` |
| Layout constraints | `silk.constraints` | `SymmetricBlocks`, `SameTemplate`, `GroupBlocks`, `Module.add_constraint` | [api-reference.md#silkconstraints--layout-intent-constraints](api-reference.md#silkconstraints--layout-intent-constraints) | `silk/examples/constraints_5t_ota.py` |
| Fault injection | `silk.faults` | `FaultSet`, `FaultCampaign`, `stuck_open`, `vth_shift` | [api-reference.md#silkfaults--fault-injection-and-monte-carlo-campaigns](api-reference.md#silkfaults--fault-injection-and-monte-carlo-campaigns) | `silk/examples/fault_injection_example.py` |
| Layout intelligence | `silk.layout` | `estimate_layout`, `RoutingModel`, `capture_layout`, `inject_parasitics` | [api-reference.md#silklayout](api-reference.md#silklayout) | `silk/examples/layout_intelligence_example.py` |
| Lint / coverage | `silk.lint`, `silk.lint_markers` | `RuleRegistry` (`check_interfaces`/`check_voltage_domains` live in `silk.circuit.interface_check`, re-exported from `silk`); waivers (`silk_waivers.yaml`, `apply_waivers`) and the `NoConnect` / `NoConnectPort` markers for intentional exceptions. | [lint_and_coverage_user_guide.md](lint_and_coverage_user_guide.md) (waivers: [#waiving-false-positives](lint_and_coverage_user_guide.md#waiving-false-positives)) · [api-reference.md#silklint--analog-lint-and-coverage-framework](api-reference.md#silklint--analog-lint-and-coverage-framework) | `silk/examples/lint_coverage_example.py` |
| Port-role inference | `silk.circuit` | `RoleRuleset`, `infer_port_roles`, `RoleRuleset.default` | [api-reference.md#port-role-inference](api-reference.md#port-role-inference) | `silk/examples/port_role_inference_example.py` |
| Optimize / sweep | `silk.optimize` | `VariableSpace`, `Optimizer`, `Variable` (`Variable` is the `silk`-level export) | [api-reference.md#silkoptimize](api-reference.md#silkoptimize) · [optimizer.md](optimizer.md) | `silk/examples/parameter_sweep_example.py` |
| Signals | `silk.signals` | `Signal`, `NgspiceSignal`, deferred probe arithmetic/transforms, `resolve`, `corner`, `plot` | [api-reference.md#silksignals](api-reference.md#silksignals) · [probes.md](probes.md) | `silk/examples/ota_5t_guide.py` |
| Code generation | `silk.codegen` | `Generator.generate` (circuit spec → Module code; `circuit_name`/`corner` validated at emission), `Generator.generate_testbenches` / `TestbenchSpec` (import `TestbenchSpec` from `silk.codegen.testbench_generator`; declarative testbench spec → testbench code; `circuit_builder` must be a bare identifier) | [api-reference.md#silkcodegen](api-reference.md#silkcodegen) · [codegen-contract.md](codegen-contract.md) | `silk/examples/testbench_codegen_example.py` |
| Schema export | `silk` | `export_schema`, `module_schema`, `export_schema_json` | [api-reference.md#export_schemasource--module_schemamodule--export_schema_jsonsource](api-reference.md#export_schemasource--module_schemamodule--export_schema_jsonsource) | `docs/quickstart.md#11-schema-export` |
| Database / tracking | `silk.db` | `Database`, `DesignStage`, `SimulationResultRecord`; `Module.create_circuit_version`, `store_report`, `query_results`, `restore_spec_measurements`; profile-scoped `list_characterization_profiles` / `resolve_characterization_profile` / `select_characterization_evidence` / `restore_characterization_evidence`; [copy migrations to the current schema](database.md#migrating-a-schema-v1-database) | [database.md](database.md) · [api-reference.md#silkdb](api-reference.md#silkdb) | `silk/examples/circuit_tracking_example.py` (unscoped) · `silk/examples/net_current_persistence_example.py` (profile-scoped) |

Independent sources add `dc=` to every waveform in transient and at the
operating point. For absolute PWL levels or a pulse already containing its
bias, omit `dc` (default zero). For a shifted sine, use
`sources.Sine(vampl=..., freq=...)` with its default zero offset and put the
bias in the source's `dc`. Supply exactly one waveform slot. Every waveform
emits two cards; logical `i(name)` hides that layout. To measure current in AC,
insert a 0 V `VoltageSource` ammeter in series instead of probing an ideal
current source directly. See [source contracts](api-reference.md#silksources).

`inject_parasitics(module, estimate)` turns an `estimate_layout()` result
into lumped RC loading elements for a subsequent SPICE
`tb.run()`/`module.simulate()`. Original terminals remain on their net; only
an added capacitor is reached through R. The resulting bandwidth change
shows loading sensitivity, not inter-terminal wire resistance or routed delay.
See [the feedback loop](api-reference.md#the-routing-parasitic-feedback-loop)
for the full estimate → inject → simulate → observe workflow.

---

## 3. Decision guides

Behavioural Monte Carlo uses `Normal`/`Uniform` model parameter declarations
with `tb.set_monte_carlo(...)` or `tb.run_monte_carlo(...)`. Each sample holds
its draws across DC points and analyses; process sources share draws by name.
See [the behavioural workflow](monte-carlo.md#behavioural-models-declaring-variation)
for instance identity, mismatch switches and deterministic-model limits.


### Behavioral vs. SPICE

| Situation | Use |
|---|---|
| Rapid architecture exploration, no installed PDK, unit tests of logic | **Behavioral** — `simulate_behavioral(t_step, t_stop)` |
| Accurate transistor-level performance, PVT corners, parasitics | **SPICE** — `simulate(corner='tt')` |
| You want both: calibrate a behavioral model from a SPICE result | **Both** — use `CalibrationBinding` on a `Spec` (see [silk.specs](api-reference.md#silkspecs--design-stage-bridge)) |

See [One workflow, two engines](#one-workflow-two-engines) for model requirements and supported elements.

### Which source to use

Two equivalent styles — pick either. **Declarative library** (waveform-first,
`pmos1p8`-style, `ckt.add(...)`-friendly) or the explicit **class** form:

| Need | Declarative (`sources.*`) | Class form |
|---|---|---|
| Fixed DC bias or supply | `sources.vdc('vdd','vdd','0', 1.8)` | `VoltageSource(..., dc=value)` |
| AC small-signal analysis | `sources.vac('vin','in','0', 1.0)` | `VoltageSource(..., ac=1.0)` |
| Clock / digital switching | `sources.vpulse('clk','clk','0', v1=0, v2=1.8, tr=..., tf=..., ...)` | `VoltageSource(..., pulse=Pulse(...))` |
| Sine input for THD/SNDR | `sources.vsine('vin','in','0', voffset=.9, vampl=.1, freq=1e6)` | `VoltageSource(..., sine=Sine(...))` |
| Piece-wise ramp for INL/DNL | `sources.vpwl('vin','in','0', points)` | `VoltageSource(..., pwl=PWL(...))` |
| Constant current bias | `sources.idc('ib','a','0', value)` | `CurrentSource(..., dc=value)` |
| Bit pattern / PRBS | `sources.BitPattern(...)`, `sources.PRBS(...)`, `sources.prbs_bits(...)` | see [api-reference.md#silksources](api-reference.md#silksources) |
| Noise or random stimulus | `sources.vtrnoise(...)` / `sources.vtrrandom(...)` | `VoltageSource(..., trnoise=...)` |

The declarative form has a namespaced twin (`sources.voltage.pulse(...)` ==
`sources.vpulse(...)`), takes `p`/`n` or `positive`/`negative`, and allows
`dc=`/`ac=` offsets alongside a waveform. Both forms build the same
`VoltageSource`/`CurrentSource` and work with `add_source(...)` or `ckt.add(...)`.

`Pulse` requires positive, keyword-only `tr` and `tf`. Choose explicit edge
times, such as `CLK_PER / 500`, and a transient step small enough to resolve them.

See [api-reference.md#silksources](api-reference.md#silksources) for all shapes.

### Plain testbench vs. PVT/sweep pattern

**Plain testbench** — one corner, one result:
```python
tb = mod.create_test('ac')
# … wire, add_metric, add_spec …
report = tb.run(corner='tt')   # → SpecReport
```

**Process-corner sweep** — the PDK's process corners (five for sky130):
```python
results = tb.run_corners()    # → CornerResults over tt/ff/ss/sf/fs
```

**Temperature and supply:** apply the operating temperature with
`tb.set_temperature(85)` on the wired testbench, and the supply with
`tb.set_supply_voltage(1.8)`, which sets the rail's DC value before the deck is
emitted and confirms it against the deck that runs. Pass `source='vdd'` when the
bench has several candidate rails and none carries a declared `POWER` role.
`run(temperature=...)` and `run(voltage=...)` do not configure the circuit —
both are deprecated assertions that what you already configured is what will be
applied. Native `tb.netlist.set_simulator_options(temp=85)` is the same single
setting; the last call wins, and `tnom` sets the model reference temperature
and is not a substitute. Do not combine either with custom `.temp` lines.

A test with no temperature of its own takes the one set on its live owning
circuit, re-read at every run, so `chip.set_temperature(85)` before
`chip.run_all_tests()` runs the whole tree hot. A test that sets its own keeps
it, in both directions, and `tb.set_temperature(None)` opts that test out. A
block that owns registered tests takes the setting too, which is the one-call
default for the embedded-testbench pattern: `block.set_temperature(85)` before
`block.run_all_tests()` runs all of that block's tests hot, and a test that set
its own still wins. Only a subcircuit that owns no tests refuses the setter,
because there nothing would apply it. The same holds for
`set_supply_voltage`. Test result caching (`cache=True`) is disabled —
every `run()` simulates regardless.

**Declared operating grid:** keep conditions in the ordinary run plan,
using plural setters on a block or its test:

```python
from silk.examples.ota_5t_guide import build_5t_ota

ota = build_5t_ota()
ota.set_temperatures([-40, 27, 125])
ota.set_supply_voltages([1.62, 1.8, 1.98])
ac = ota.get_test('ac').set_corners(['tt', 'ss', 'ff'])
vout = ac.probe('vout')
res = ac.run()                       # 27 points in one CornerResults
print(res.summary())
res.to_csv('ac_pvt.csv')
point = ('tt', 125.0, 1.62)
report = res[point]
waveform = vout.point(point)
```

Expansion order is corner, temperature, supply, then sample. A list of length
one still requests a collection. With no list, `run()` still returns one
`SpecReport`; `run_corners()` keeps its existing behavior. Lists inherit live
through the selected owner's hierarchy, including different placements of one
definition. A test's scalar setter overrides the owner's list; its list
setter overrides the owner's scalar. Calling a scalar setter after a list
replaces that list. Explicit `set_temperatures(None)` suppresses inherited
lists and uses the test's scalar, else the owner's scalar; never declaring the
list permits inheritance. Supply follows the same rules, including source
selection. Native `set_simulator_options(temp=...)` is the equivalent scalar
choice; unrelated options do not clear a list. Scalar `None` retains its
existing explicit opt-out meaning.

Lists must be nonempty, finite numeric sequences without duplicate points;
temperature must exceed -273.15 C. They are separate declarations, never arrays
in simulator options or source DC values. Each point gets readable folders such
as `tt/temperature-125.0/supply-1.62/`, with `unset` for an unspecified axis.
Requested coordinates and captured operating conditions are distinct table
columns. Grid summaries count planned runs, completed PASS/FAIL reports, and
runs that failed and produced no report, then give scalar metric ranges and
the worst full point. Failed points remain visible; an entirely failed run
plan raises `SimulationError` with its failure map.

`Module.simulate()`, the Optimizer and `run_test(store=True)` refuse effective
collection declarations. `prepare()` allows exactly one deterministic point,
refusing multiple points and Monte Carlo. Behavioral temperature lists refuse,
as a single ambient-temperature setting does. A temperature list conflicts
with a DC temperature sweep; a supply list conflicts with a DC sweep of its
selected supply source. Unrelated DC sweeps are allowed. Use
[Monte Carlo operating grids](monte-carlo.md#operating-grids) for per-point
yield, aggregate statistics, sample probes and CSV details. Circuit/spec report
grid rendering and collection persistence are separate capabilities; do not
assume this result collection can be stored or rendered by those entry points.

**Parameterized sweep with varying stimulus** — the StrongArm example
`silk/examples/strongarm_tb.py` is the canonical pattern. It registers
multiple named testbenches on one module via repeated `comp.create_test(name)`
calls and runs them all with `comp.run_all_tests()`. Each testbench can use
a different VM bias or overdrive, which is the correct way to sweep stimulus
parameters without mutating a shared testbench.

### Constraints vs. specs vs. metrics — which to reach for

| You want to… | Use |
|---|---|
| Assert a performance target (gain > 20 dB) and get pass/fail | `tb.add_spec(...)` on a `ModuleTestbench` |
| Declare a first-class hierarchical spec with calibration write-back | `mod.declare_spec(...)` + `mod.reconcile_specs()` |
| Compute a scalar from a simulation Signal (no pass/fail) | `silk.metrics` functions |
| Encode layout placement intent (symmetry, proximity) | `silk.constraints` + `mod.add_constraint(...)` |

### Two `add_constraint` methods — DO NOT confuse

Silk has two completely unrelated `add_constraint` methods:

| Method | Owner | What it does | Import required |
|---|---|---|---|
| `Module.add_constraint(constraint)` | `Module` | Attaches a **layout** `ConstraintEnvelope` (symmetry, proximity, grouping) | **`import silk.constraints`** — this import monkey-patches the method onto `Module`; it does not exist without it |
| `Optimizer.add_constraint(name, min=None, max=None)` | `Optimizer` | Sets metric/variable feasibility limits; records retain their verdict, and `best` selects eligible results | No special import |

See [api-reference.md#silkconstraints--layout-intent-constraints](api-reference.md#silkconstraints--layout-intent-constraints)
and [api-reference.md#optimizer](api-reference.md#optimizer) for signatures.

### When to go hierarchical

Go hierarchical when:
- A block (comparator, bias generator, filter) is reused in multiple contexts.
- You want to run testbenches on sub-blocks independently.
- The design has distinct domains (AVDD / DVDD) that need `check_voltage_domains`.

Instantiate sub-blocks with `top.add_module(sub_mod, 'inst_name', nets={...})`.
The returned `SubcircuitInstance` handle gives attribute access to internal nets
after simulation (`inst.vbiasn`, `inst.child_block.vout`).

See [Module hierarchy API](api-reference.md#module) and
`silk/examples/hierarchical_example.py`.

---

<a id="4-worked-spine--silkexamplesota_5t_guidepy"></a>

## 4. Worked example — `silk/examples/ota_5t_guide.py`

`ota_5t_guide.py` walks the full loop from build to netlist. Here is the step-by-step reading.

**Step 1: Load the PDK**

```python
from silk import load_pdk

pdk = load_pdk('sky130')
```

`load_pdk('sky130')` resolves everything from the PDK manifest
(`silk/pdk/data/sky130.pdk.yaml`): the device catalog, the sky130 lib (via
`$PDK_ROOT`), the process corners, and any extra includes — **no yaml or lib
path in your code**. Set `$PDK_ROOT` to your PDK install (the directory
containing `sky130A/`). Loading the packaged catalog works without installed
models; simulation raises `SimulationError` with installation advice if the
deck needs a missing model library. Ideal-only SPICE needs no installed PDK.
Every shipped manifest sets `user_units: u`, so you write W/L
in microns on every PDK; pass `load_pdk('sky130', user_units='si')` for SI
metres. `silk.pdk.available()` lists the registered catalogs, not installed
models or validated simulation support. sky130 is the release-validated
transistor-simulation PDK; GF180MCU and SG13G2 are preview integrations.

(Low-level escape hatch, rarely needed: `from silk.pdk import BasePDK;
BasePDK(yaml_path, library_location=lib_path)` still works — e.g. for
build/netlist-only work with a placeholder lib. Prefer `load_pdk`.)

**Step 2: Construct the Module**

```python
from silk import Module
from silk.examples import example_output_dir

mod = Module("PDK_skywater130", example_output_dir("ota_5t_guide"), "ota5t.cir",
             pdk_details=pdk, module_name="ota_5t")
```

`example_output_dir("ota_5t_guide")` (from `silk.examples`) puts the netlist
and rawfiles in `ota_5t_guide/` under `./silk-output` (relative to the
directory you run from), or under the directory the `SILK_EXAMPLES_OUT`
setting names.

`module_name="ota_5t"` puts the module in subcircuit mode: `generate_netlist()`
returns a `.subckt` string instead of writing a file. If you want a runnable
top-level netlist, omit `module_name`.

**Step 3: Declare ports**

```python
from silk import Port          # Port is re-exported from silk.__init__

mod.define_module_ports({
    "VDD": Port.supply(), "VSS": Port.ground(),
    "VINP": Port.analog_input(), "VINM": Port.analog_input(),
    "VOUT": Port.analog_output(), "IBIAS": Port.analog_input(),
})
```

`Port.*` factories encode direction and role. These annotations give
`check_interfaces()` and `check_voltage_domains()` the information to check
connections. You can also construct a `Port` with explicit annotations.

**Step 4: Add devices inside a `with mod:` block**

Each PDK device is a class. Construct it inside `with mod:` and it is
auto-registered — devices, sources and passives alike, so one block covers the
whole module. This fragment illustrates registration; the complete five-device
OTA is in `_add_devices` in the example file. The capacitor below illustrates
a passive load and is not part of that OTA:

```python
from silk.elements import elements

with mod:
    pdk.nmos1p8("M1", d="vint", g="VINP", s="tail", b="VSS", w=4.0, l=0.5)
    pdk.pmos1p8("M3", d="vint", g="vint", s="VDD", b="VDD", w=8.0, l=0.5)
    elements.capacitor("CL", p="VOUT", n="VSS", value=1e-12)
```

Pins and parameters are plain keyword arguments, so a mis-typed pin fails at
construction rather than surviving inside a dict. W=4.0 µm, L=0.5 µm (manifest
`user_units: u`); `m=` defaults to 1.

Outside a `with` block, `mod.add(dev1, dev2, ...)` does the same thing
explicitly: `mod.add(pdk.nmos1p8("M1", d="vint", g="VINP", s="tail",
b="VSS", w=4.0, l=0.5))`. **Use `add_module` for hierarchical
instantiation** (`mod.add_module(sub_mod, 'inst', nets={...})`). The older
`add_element` alias is deprecated; its warning shows the constructed-device
replacement, including the supplied name, pins and parameters.

**Step 5: Embed the testbench in the build — the module owns its characterization**

This is the **testbench-as-part-of-module** pattern (`strongarm_tb.py` is the
canonical multi-testbench version). The factory does not return a bare netlist;
it registers the characterization testbench *on* the module before returning, so
the module knows how to test itself. This outline uses `...` for the port
declarations and calls helpers defined in the example file:

```python
PAIR_W, PAIR_L = 4.0, 0.5
MIRROR_W, MIRROR_L = 8.0, 0.5
TAIL_W, TAIL_L = 6.0, 1.0
TAIL_BIAS = 0.9


def build_5t_ota(*, pair_w=PAIR_W, pair_l=PAIR_L,
                 mirror_w=MIRROR_W, mirror_l=MIRROR_L,
                 tail_w=TAIL_W, tail_l=TAIL_L, tail_bias=TAIL_BIAS) -> Module:
    mod = Module("PDK_skywater130", example_output_dir("ota_5t_guide"), "ota5t.cir",
                 pdk_details=pdk, module_name="ota_5t")
    mod.define_module_ports({...})
    _add_devices(mod, pair_w=pair_w, pair_l=pair_l,
                 mirror_w=mirror_w, mirror_l=mirror_l,
                 tail_w=tail_w, tail_l=tail_l)
    _attach_ac_test(mod, tail_bias=tail_bias)        # mod.get_test('ac')    + gain/BW design specs
    _attach_transient_test(mod, tail_bias=tail_bias) # mod.get_test('tran')  + settling design spec
    _attach_noise_test(mod, tail_bias=tail_bias)     # mod.get_test('noise') + input-noise design spec
    return mod

mod = build_5t_ota()   # build explicitly when needed
```

The keyword-only design knobs default to module-level constants: pair,
mirror and tail W/L are in µm; `tail_bias` is the M5 gate voltage in V,
passed separately to every testbench's `vbias` source. It does not change
the input common mode or step stimulus. Devices are sized before the
benches register their live DUT reference.

The returned module carries its devices, **three** testbenches — AC
(`mod.get_test('ac')`), transient (`mod.get_test('tran')`), and noise
(`mod.get_test('noise')`) — and the
first-class design `Spec`s they declare through `tb.add_spec` (each call returns the
`Spec`; `mod.reconcile_specs()` checks them all). There is no external
test object to thread through your code, and `run_all_tests()` drives the whole
suite. Import `build_5t_ota` and call it to get a fresh circuit. The example
also supports `from silk.examples.ota_5t_guide import mod` through a lazy
module attribute; each lookup builds a fresh circuit.

**Step 6: The embedded AC testbench**

```python
from silk import sources
from silk.simulation import AC
from silk.metrics import dc_gain_dB, bandwidth_3dB

def _attach_ac_test(mod, *, tail_bias):
    tb = mod.create_test('ac')          # owned by mod (mod.get_test('ac'))
    tb.add_module_under_test({
        'VDD': 'VDD', 'VSS': '0', 'VINP': 'vinp',
        'VINM': 'vinm', 'VOUT': 'vout', 'IBIAS': 'ibias',
    })
    tb.add_source(sources.VoltageSource('vdd', positive='VDD', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('vbias', positive='ibias', negative='0', dc=tail_bias))
    tb.add_source(sources.VoltageSource('vinp', positive='vinp', negative='0', dc=0.9, ac=1.0))
    tb.add_source(sources.VoltageSource('vinm', positive='vinm', negative='0', dc=0.9))
    tb.add_sim_type(AC(variation='dec', points=20, fstart=1, fstop=1e9))
    vout = tb.probe('vout')
    tb.add_metric('gain_dB', dc_gain_dB, sig=vout)
    tb.add_metric('bw_hz', bandwidth_3dB, sig=vout)
    tb.add_spec('DC gain > 20 dB', 'gain_dB', '>', 20.0)
    tb.add_spec('BW > 10 MHz', 'bw_hz', '>', 10e6)
    return tb
```

`create_test(name)` registers the testbench on the module; retrieve it with
`mod.get_test(name)`. Lookup is local to that module and returns the original
testbench. An unknown name raises `silk.exceptions.ConfigurationError` with the
module and available test names.
`add_module_under_test` maps module ports → testbench net names and must be
called exactly once. One AC run feeds both specs.

For a placed block, bind its structural reference directly:
`vout = tb.probe(stage.vbias)` (or `stage.net('vbias[0]')` for an explicit lookup).
The handle reads no data before the run; connected ports read their parent
nets. Choose the bench explicitly when a definition serves several benches or
owner paths. After an unambiguous run, `stage.vbias.y` and arithmetic still
work. Bare definition-level Nets cannot choose a placement. Declared names
come first: a child called `child` is reached as `stage.child`. An item named
`bus`, `child`, `device`, `instance_name`, `net`, `path` or `subcircuit` makes
the handle's tool with that name unavailable through that handle. See
[declared names and dot shortcuts](known_issues.md#declared-names-and-dot-shortcuts).
Bus/array subscripts name declared
numbers: for `range(8, 12)`, use `stage.d[8]`, and a slice `[8:10]` keeps bits
8 and 9. A width-one bus remains a bus. See the
[object probe and indexing guide](probes.md#probe-a-placement-by-object).

`tb.probe('vout')` names a signal **before** the run and hands back a
`Signal` that carries no data yet; `add_metric(name, fn, sig=...)` binds that
Signal to a `@metric`-registered function (`dc_gain_dB` and `bandwidth_3dB`
are from `silk.metrics`), which is resolved once the bench has results. That
is the taught program — probe → `add_metric(name, fn, sig=...)` → `run` → plot
([practical probe guide](probes.md)). Arithmetic on a probe defers too, so
`sig=(vp + vm) / 2` measures a derived signal with no lambda anywhere. The
same probe is what `plot_transfer_function` draws at the bottom of the file,
so the curve and the numbers cannot come from different signals.

The older **callable form**, `add_metric('gain_dB', lambda ckt:
dc_gain_dB(ckt.load_signal('vout')))`, still works and is still the answer
when the extractor does more than load a signal and call a metric — scaling
the result, taking a reciprocal, or raising its own diagnosis. The extractor
receives the post-simulation testbench `Module` and returns a float.

A probe follows the bench's current results; it is not a saved run. Reading
it before simulation raises `UnboundSignalError`. For a multi-analysis bench,
put `analysis='tran'` or `analysis='ac'` on probes when the selection would be
ambiguous. For corner waveforms, call `tb.run_corners(...)` (every corner
always gets its own output folder and is retained, regardless of output
mode), then select `vout.corner('tt')` or use `vout.plot()` to
overlay retained corners. The [probe guide](probes.md) explains current
probes, metric binding, result lifetime and the disabled report cache in detail.

**Step 6b: A second testbench — transient step response**

A module can own as many testbenches as it has characterizations; each is a
fresh `create_test(name)`. The transient one wires the OTA as a **unity-gain
follower** — `VINM` and `VOUT` map to the *same* net, closing negative feedback
— so a small input step produces a settling step at the output instead of
hitting a supply rail under the open-loop gain:

```python
from silk.simulation import Transient
from silk.metrics import settling_time

def _attach_transient_test(mod, *, tail_bias):
    tb = mod.create_test('tran')
    tb.add_module_under_test({
        'VDD': 'VDD', 'VSS': '0', 'VINP': 'vinp',
        'VINM': 'vout', 'VOUT': 'vout', 'IBIAS': 'ibias',   # VINM↔VOUT feedback
    })
    tb.add_source(sources.VoltageSource('vdd', positive='VDD', negative='0', dc=1.8))
    tb.add_source(sources.VoltageSource('vbias', positive='ibias', negative='0', dc=tail_bias))
    tb.add_source(sources.VoltageSource(
        'vinp', positive='vinp', negative='0',
        pulse=sources.Pulse(v1=0.9, v2=0.95, td=1e-6, tr=1e-9, tf=1e-9,
                            pw=5e-6, per=10e-6)))
    tb.add_sim_type(Transient(step=1e-9, stop=2e-6))
    tb.add_metric('t_settle_1pct', settling_time, sig=tb.probe('vout'),
                  initial=0.9, tol=0.01, start=1e-6)
    tb.add_spec('settles < 100 ns', 't_settle_1pct', '<', 100e-9)
    return tb
```

This checks settling around the final output sample, using 1% of its change
from 0.9 V; it does not check final tracking error against the 0.95 V input.
This snippet binds the public settling metric directly; the example file
uses a circuit extractor with the same arguments. Closing feedback is a
net-mapping choice in
`add_module_under_test` (no extra component); and transient settling/slew
metrics live in **`silk.metrics.timing`** (`settling_time`, `slew_rate`,
`overshoot`), alongside the AC metrics in `silk.metrics.ac`.

**Step 7: `tb.add_spec` is also a module design spec**

`tb.add_spec(...)` returns a module-owned `Spec`; running the bench records
its measurement automatically for `mod.reconcile_specs()`.

**Step 8: Drive the embedded loop — `run_all_tests()` + `reconcile_specs()`**

```python
def run_characterization(mod, corner='tt'):
    results = mod.run_all_tests(corner=corner)   # runs every owned testbench
    return results, mod.reconcile_specs()        # checks the module's spec tree
```

`mod.run_all_tests()` drives every testbench the module owns (`'ac'`, `'tran'`,
and `'noise'`) and returns a dict whose unique keys are
`f'{module_name}.{test_name}'` (e.g.
`'ota_5t.ac'`, `'ota_5t.tran'`, `'ota_5t.noise'`). Because running each
testbench recorded its metrics onto their design `Spec`s,
`mod.reconcile_specs()` returns a `SpecReconciliationReport` that already
knows the verdicts — `recon.summary()` prints `4 PASS, 0 FAIL, 0 UNMEASURED`,
with lines like `[PASS] DC gain > 20 dB: 34.8507 >20`,
`[PASS] BW > 10 MHz: 1.77828e+08 >1e+07`, and `[PASS] settles < 100 ns: 1.24e-09 <1e-07`.
These AC results use no explicit output load capacitor.
These illustrative values use the default sizing, sky130 models and ngspice 42;
model and simulator versions can change them.
This is the same `declare_spec` / `reconcile_specs` machinery
`flash_adc_example.py` uses for hierarchical specs and calibration.

**Step 9: Plot the transfer function**

`plot_transfer_function(tb, report, png_path)` saves a Bode magnitude plot. It
leans on the AC-aware `Signal.plot()`: the `vout` AC signal is complex, so
`sig.plot(ax=...)` renders magnitude-in-dB on a log-frequency axis with no
manual `20*log10`/`semilogx`, and the example just annotates DC gain and the
3 dB bandwidth on top.

```python
sig = tb.probe('vout').snapshot()   # tb is the just-run AC testbench
sig.plot(ax=ax, label='|vout/vin|')   # complex → dB vs log-frequency
```

**Step 10: Emit a SPICE netlist**

```python
if __name__ == "__main__":
    m = build_5t_ota()
    print(m.generate_netlist())
```

Because `module_name="ota_5t"` is set, `generate_netlist()` returns the
`.subckt` block as a string (no file written). See §5 for dual-mode details.

### Account for tests, conditions, and failures

Before running, enumerate the registered tests with `list_test_entries()`.
Each entry's `report_key` is the exact result key: repeated module/test names
are disambiguated with an owner path. The default includes nested placements;
`recursive=False` lists only local tests. For an already built `mod`, run:

```python
from silk import CornerResults
from silk.circuit import TestFailure

entries = mod.list_test_entries()
for entry in entries:
    print(entry.report_key, entry.path, entry.test_name)
results = mod.run_all_tests()
for entry in entries:
    outcome = results[entry.report_key]
    print(entry.report_key)
    if isinstance(outcome, TestFailure):
        print("test error:", outcome.error_type, outcome['error'], outcome.details)
        print("attempt files:", outcome.run)
        continue
    # MonteCarloResults is also a CornerResults; its keys are (corner, sample).
    reports = outcome.reports if isinstance(outcome, CornerResults) else {None: outcome}
    if isinstance(outcome, CornerResults):
        for condition, failure in outcome.failed.items():
            print("condition/sample failed:", condition, failure)
    for condition, report in reports.items():
        print("condition/sample:", condition, "passed:", report.passed)
        print(report.summary())
        print("metric extraction errors:", report.metric_errors)
        print("launch:", report.launch_id, "files:", report.run)
```

A `SpecReport` can exist even when extraction failed: `metric_errors` records
extractor failures, while `metrics` and `summary()` show measured values and
spec verdicts. A measured value outside its limit is an unmet spec, not an
extraction error. `passed` covers checked specs only (and is true with no
checks); inspect the checks and metric errors as well. An unavailable/NaN
value also fails a spec; inspect metric warnings and the summary rather than treating it as a valid measurement
(see [metric failure conventions](metrics-library.md)). Configuration errors in the
call can propagate instead of returning a results dictionary; fix the request.

For declared `tb.set_corners([...])`, inspect both `reports` and `failed`;
Monte Carlo likewise retains failed samples in `failed`, with their error and
seed. Compare these keys with the corners and sample indices you declared.
The all-failed limitation remains: when every declared corner or every Monte
Carlo sample fails, the bench raises `SimulationError` instead of returning
that aggregate; `run_all_tests()` records a test failure, losing the individual
condition/sample map; see [known issues](known_issues.md#monte-carlo). Explicit
`run_corners()` has a different partial-failure contract: catch
`silk.circuit.spec_report.CornerRunError` and inspect its `results` and `errors`.

Registered tests, declared conditions/samples, and simulator launches are
different counts. Identical SPICE placements can share a launch (equal
`launch_id`); Monte Carlo expands a test into samples. `report.run` and
`TestFailure.run` name deck/log/rawfile paths when available, not archived
copies: preserve files before a later run replaces them. For device-parameter
readback, set `tb.save_device_params = True` before running, then use the
report's `op_table()`; `None` inherits the DUT's capture setting.

Passing every declared condition checks that declared set. It does not prove
an operating envelope complete: untested temperatures, supplies, loads, and
other conditions remain unverified. Likewise, `CircuitTracker.promote()`
advances a stored design stage and does not require passing specs; promotion
is not a readiness check. See [promotion and evidence origin](database.md#promotion-and-evidence-origin).

### Revise, rebuild, rerun

`add_module` and `add_module_under_test` keep live references. Editing a
placed child or calibrating its behavioral model affects the parent's next
netlist and an existing bench's next run. Wiring belongs to each placement:
placing one module object, changing it, then placing it again gives two
instances of the final circuit. Call the factory twice when you need two
independently configurable circuits. PDK device templates still get a copy
per placement, so transistor widths and pins remain independent. For
different definitions sharing a `module_name`, use distinct names or set
`variants=True` in the builder. Identical independent builds may share a name
without variants (see [Same builder, several sizings](#same-builder-several-sizings-variantstrue)).

`run_all_tests()` shares a simulation when resolved SPICE inputs are identical,
while evaluating and reporting every placement separately. Different inherited
conditions produce distinct runs. Use `top.get_test(name, instance='a.bias')`
for one placement's readback or execution; `list_test_entries()` supplies the
paths. Local `dut.get_test(name)` stays the original bench. Numeric readback
requires a placement when its latest invocation produced several distinct runs;
plots overlay them. Bench edits still reach the next run, and direct `run()`
always launches afresh. Behavioral callables and uncertain executable inputs
are conservatively run separately.

For factory construction inputs, call the factory again with the revised
values. The OTA factory sizes devices before creating its three embedded
benches, so a fresh call gives the revised circuit fresh tests.

Save this as `revise_ota.py` and run `MPLBACKEND=Agg python revise_ota.py`
(using `.venv/bin/python` in a checkout). It needs ngspice and installed sky130
models; check `python -m silk.doctor` first. This bounded comparison runs only
AC at `tt`; transient and noise specs remain **unmeasured**, not passed.

```python
from pathlib import Path
from silk import generate_report, diff_reports
from silk.examples import example_output_dir
from silk.examples.ota_5t_guide import build_5t_ota, plot_transfer_function

out = Path(example_output_dir("revise_ota"))
reports = {}
measurements = {}
designs = {}
for label, pair_w in (("baseline", 4.0), ("revised", 8.0)):
    dut = build_5t_ota(pair_w=pair_w)  # W in µm; all other knobs unchanged
    designs[label] = dut             # keep each owning bench alive
    dut.set_output_options(mode="new")  # retain a unique rawfile for each run
    tb = dut.get_test("ac")
    measured = tb.run(corner="tt")
    measurements[label] = dict(measured.metrics)
    print(label, measured.summary())
    print(label, measurements[label])
    report = generate_report(dut, spec_reports=[measured],
                             auto_discover_config=False)
    reports[label] = report
    report.save_json(str(out / f"{label}.json"))
    report.save_html(str(out / f"{label}.html"))
    # This DUT is a subcircuit: generate_netlist returns text, not a file path.
    (out / f"{label}.cir").write_text(dut.generate_netlist())
    plot_transfer_function(tb, measured, str(out / f"{label}-ac.png"))
    print(label, "retained rawfile:", tb.results_file)

diff_reports(reports["baseline"], reports["revised"],
             label_a="baseline", label_b="revised").print_summary()
print("gain change (dB):",
      measurements["revised"]["gain_dB"] - measurements["baseline"]["gain_dB"])
print("artifacts:", out)
```

The report, AC plot and `.cir` for each label come from that same freshly
built DUT and its just-run bench. Compare the numeric metrics as well as the
report diff: a gain change need not change a pass/fail verdict. In the netlists,
M1/M2 change from `w=4` to `w=8`, with `l=0.5` unchanged in this
sky130 subcircuit (PDK-native µm values). Larger input
pair width changes transconductance and capacitance; for this bias and load,
expect a small gain increase, not a doubling of gain. Inspect the AC curve and
bandwidth too; the sampled −3 dB bandwidth may stay in the same frequency bin.
A missing/non-finite metric is not evidence of improvement. For full
characterization, run the factory's embedded AC, transient and noise suite
with `run_characterization(dut)` as above and inspect every result.

Live circuit definitions and result lifetime are separate. A `tb.probe(...)`
follows **that bench's current results**, not the original DUT or an earlier
run. Keep its bench alive; use `.snapshot()` for an in-memory waveform copy.
A saved report preserves reported values but is not a durable waveform
archive, and a `SpecReport` can still reference a live simulation module.
For selected evidence, save the report/netlist/plot as here; retain the printed
rawfile paths (or copy those files to your evidence directory) for later
waveform readback. See [probes](probes.md) and [report persistence](reports.md#saving-and-loading).

Use this recipe for **edit → remeasure → report → netlist**. Record the
revision, environment, commands and measured changes so another user can
reproduce the comparison.

---

## 5. Emitting a SPICE netlist — `generate_netlist()` dual mode

`generate_netlist()` behaves differently based on whether `module_name` is set:

| `module_name` | Returns | Side effect |
|---|---|---|
| Set (e.g. `"ota_5t"`) | `.subckt` block as a string | No file written |
| `None` (top-level mode) | Deck file path (`str`) | Writes the full netlist under `mod.dir_folder` (a fresh child directory in `mode="new"`) |

The default output mode is `new`, which retains each run in a fresh child
directory. Use `mod.set_output_options(mode="overwrite")` for a fixed deck
path that later runs replace.

**Subcircuit mode** — "netlists for another project":

```python
mod.define_module_name('my_amp')
subckt_str = mod.generate_netlist(corner='tt')
# subckt_str starts with ".subckt my_amp VDD VSS VINP ..."
# Paste into a larger netlist, or pass to a tool expecting a .subckt block.
```

**Top-level mode** — deck file (add a circuit and analysis before simulating):

```python
top = Module('PDK_skywater130', 'out', 'top.cir', pdk_details=pdk)
# No module_name set; choose a fixed output path for this example.
top.set_output_options(mode='overwrite')
deck_path = top.generate_netlist(corner='tt')   # writes and returns out/top.cir
```

The `corner` kwarg selects the process corner include (e.g. `'tt'`, `'ff'`).
The `subckt` kwarg overrides auto-detection: passing `subckt=False` on a
named module raises `ValueError`.

See [api-reference.md#module](api-reference.md#module) (`generate_netlist` row).

### Choosing a simulator — ngspice (default) or Xyce

For a standalone top-level Module, `mod.simulator = 'xyce'` selects the
backend for its next `mod.simulate()` (generate → run → load). For the
embedded-testbench pattern, the factory returns an owning circuit and each
registered test runs a separate `tb.netlist` Module. At every `tb.run()`, that
executing Module inherits its **live owner's current selection**, before preparing the run. Selecting after construction or switching after a run
works through `run_test()`, `run_all_tests()`, corners and ordinary optimizer
tests too. Unknown backend names raise immediately.

This runnable example uses the existing ideal-load factory (ngspice and Xyce
required; no installed PDK). Its test is already registered:

```python
from tempfile import TemporaryDirectory
from silk.examples.output_modes_example import build_load

with TemporaryDirectory() as output:
    dut = build_load(output_root=output)
    dut.simulator = 'xyce'
    report = dut.run_test('level')
    assert report.passed
    assert abs(report.metrics['voltage'] - 1.0) < 1e-9
    dut.simulator = 'ngspice'
    assert dut.run_all_tests()['load.level'].passed
```

Explicitly assigning `tb.netlist.simulator = 'ngspice'` or `'xyce'` overrides
inheritance for that test and survives owner changes and copying. An explicit
`'ngspice'` assignment counts even though ngspice is the default. Recursive
runs resolve each descendant's own owner; changing the root does not retarget
children. This lets one invocation run tests on different backends. The
executing Module shows its initial or last-used selection until `run()` resolves
inheritance. Direct `tb.netlist.simulate()` retains standalone Module behavior.

`set_simulator_options()` configures numerical presets/tolerances for the
selected backend; it does not select it. At every run, numerical options
inherit from the live owner key by key. An explicit test option set with
`tb.set_simulator_options(...)` (or on `tb.netlist`) wins for that key;
other keys follow later owner changes, including removal. Options are not
translated between backends. Ambient temperature also inherits from the live
owner, with an explicit test setting taking precedence — see
**Temperature and supply** above.
`$SILK_XYCE_BIN` and `$SILK_NGSPICE_BIN` locate binaries without selecting a
backend. Xyce needs its CLI on `PATH` or `$SILK_XYCE_BIN`; a missing selected
executable raises an actionable error without fallback.

Selection itself leaves files untouched. Runs retain the configured output
mode and freshness checks. The report cache is disabled, so every run simulates against the currently selected backend.
Explicit `tb.evaluate(rawfile=...)`
continues to read disk without running. See the [API reference](api-reference.md#selecting-a-simulator)
for the override example and full contract. Metrics, specs and
`reconcile_specs` consume the same `Signal` interface across backends.

For a `Noise` analysis, load the spectral density by **canonical name** —
`load_signal('noise_out')` (output-referred) or `load_signal('noise_in')`
(input-referred), both in V/√Hz — and they resolve identically on ngspice and
Xyce. (The ngspice-native `onoise_spectrum`/`inoise_spectrum` still work.) Adding a *new* backend
(another simulator, etc.) is a registration, not a core edit — see
[api-reference.md#silksimulation--simulator-backends-ngspice-xyce](api-reference.md#silksimulation--simulator-backends-ngspice-xyce)
and `tests/test_backend_seam_smoke.py`.

---

## 6. Composition patterns

The snippets in this section illustrate construction patterns. Names such as
`pdk`, `top`, and `sub_mod` refer to circuits built by the caller; `...` marks
omitted circuit-specific code. They are not standalone scripts.

### Factory function → Module

Wrap each circuit block in a `build_<circuit>()` or `make_<circuit>()` factory.
Return the `Module`. Callers import the factory, not the internal variables:

```python
# my_comparator.py
def make_comparator() -> Module:
    comp = Module(..., module_name='comp')
    with comp:
        pdk.nmos1p8('m1', ...)
        ...
    comp.define_module_ports({...})
    return comp
```

### Expose design knobs as keyword arguments (don't bury them in globals)

**A factory's design knobs — device W/L, bias voltages, load/compensation
caps, reference currents — must be keyword arguments of the factory,
defaulting to module-level constants.** Helper functions receive them as
arguments rather than reading the module globals directly. This is what makes
a block *reusable and sweepable*: an importer can build a variant without
editing the file (`build_source_follower(c_load=2e-12, w_follower=20)`).
Different definitions sharing a `module_name` need distinct names or
`variants=True` in the builder; otherwise netlist generation raises. Identical
independent builds may share a name without variants.
`make_abo_th` names its module `abo_th` and takes no such flag, so
`make_abo_th(pdk, out, w_sw=2.0, c_hold=250e-15)` and
`make_abo_th(pdk, out, w_sw=4.0, c_hold=250e-15)` placed in one top-level
circuit raise `CircuitError`. The differential T/H
(`differential_th_example.py`) builds one `make_abo_th` and places that same
object twice, so both halves share one sizing.

```python
_CLOAD = 0.5e-12          # module constant = the DEFAULT

def _add_devices(mod, c_load):           # helper takes the knob as an argument
    with mod:
        elements.capacitor('CL', p='VOUT', n='VSS', value=c_load)

def build_source_follower(*, c_load=_CLOAD, w_follower=10.0) -> Module:
    mod = Module(..., module_name='source_follower')
    _add_devices(mod, c_load)
    ...
    return mod
```

When a metric extractor needs a knob (e.g. `i_ref` for
a current-copy error), bind it into the lambda with a default argument:
`tb.add_metric(name, lambda ckt, iref=i_ref: _error(ckt, iref))`. In the
function form a scalar knob needs no closure at all — extra keywords are
forwarded to the metric: `tb.add_metric(name, settling_time, sig=probe,
tol=0.01)`.

`source_follower_example.py`, `current_mirror_example.py`,
`two_stage_miller_ota_example.py`, and `abo_th_example.py` all follow this.

### Same builder, several sizings (`variants=True`)

To place one block several times in a circuit, with different sizes in some
placements, call the builder once per placement and construct the Module with
`variants=True`:

```python
def build_cs_amp(pdk, *, w_in=4.0, c_load=0.1e-12) -> Module:
    mod = Module('PDK_skywater130', out_dir, 'cs_amp.cir', pdk_details=pdk,
                 module_name='cs_amp', variants=True)
    with mod:
        pdk.nmos1p8('MN', d='VOUT', g='VIN', s='VSS', b='VSS', w=w_in, l=0.5, m=1)
        pdk.pmos1p8('MP', d='VOUT', g='VBP', s='VDD', b='VDD', w=8.0, l=0.5, m=1)
        elements.capacitor('CL', p='VOUT', n='VSS', value=c_load)
    mod.define_module_ports({...})
    return mod

top.add_module(build_cs_amp(pdk, w_in=4.0), 'xa', VDD='vdd', VSS='0', VIN='in0', VBP='vbp', VOUT='out0')
top.add_module(build_cs_amp(pdk, w_in=8.0, c_load=0.3e-12), 'xb', ...)
top.add_module(build_cs_amp(pdk, w_in=4.0), 'xc', ...)   # independent build with an identical body to xa
```

`amplifier_variants_example.py` exposes this flag to demonstrate the collision;
enable it when placing differently configured bodies under one module name.

With `variants=True` the deck gets one `.subckt` per distinct body, named
`<module_name>_<8 hex of the definition hash>`: `xa` and `xc` share
`cs_amp_…`, and `xb` gets its own. The suffix is there even for a single
placement, so the emitted name is never the bare `cs_amp`. Do not hard-code
it, because the hash changes when the body changes. Identical builds share one
definition. Without `variants=True`, the same three placements raise
`CircuitError` at netlist generation, naming the two instances that clash.
Use the flag when two different bodies must share a `module_name`. Other
examples avoid the clash by assigning distinct module names explicitly
(`bandgap_example.py`, `stdcells.py`); both approaches work. The worked
version is `silk/examples/amplifier_variants_example.py`; the option's full
contract is in [api-reference.md](api-reference.md) under `Module`.

### Shared test rig (don't re-type the supplies in every testbench)

Each `mod.create_test(name)` builds its **own** netlist (`ModuleTestbench`
holds a private `self.netlist`), so the same `VoltageSource('vdd', …)` written
in two tests are **independent objects in independent simulations** — never
shared mutable state. But re-typing the common supplies/bias in every test is
needless duplication. Factor it into a **rig helper** that each testbench
calls, exactly like `add_strongarm_rig(tb, …)` in `strongarm_tb.py`:

```python
def _add_supply_and_bias(tb, i_ref):
    tb.add_source(sources.VoltageSource('vdd', positive='VDD', negative='0', dc=1.8))
    tb.add_source(sources.CurrentSource('iref', positive='VDD', negative='ibias', dc=i_ref))

def _attach_ac_test(mod, v_in, i_ref):
    tb = mod.create_test('ac')
    tb.add_module_under_test({...})
    _add_supply_and_bias(tb, i_ref)        # shared rig; called once per test
    tb.add_source(...)                     # test-specific stimulus
```

### Bias with a diode-connected device + reference current, not an ideal voltage

A current-source/sink transistor's gate should be biased by a **current
mirror**, not an ideal `VoltageSource`. Add a diode-connected device on the
bias node and force a reference *current* into it from the testbench; the
sink/tail devices mirror it. This is what `current_mirror_example.py`
demonstrates as its whole point, and `source_follower_example.py` /
`two_stage_miller_ota_example.py` use for their `IBIAS` ports:

```python
# inside the module
with mod:
    pdk.nmos1p8('MBIAS', d='IBIAS', g='IBIAS', s='VSS', b='VSS', w=6.0, l=1.0)
# in the testbench rig
tb.add_source(sources.CurrentSource('iref', positive='VDD', negative='ibias', dc=i_ref))
```

The module's `IBIAS` port is then a **reference-current input** (the on-chip
diode turns it into the gate bias), so the bias current is set by `i_ref` and
the device/diode size ratio — physical, and corner-aware in SPICE — rather than
a hand-picked gate voltage.

### Module-level export (`mod = build_x()`)

For an intentionally shared definition, you may expose a module-scope instance;
import the factory instead for independent circuits:

```python
mod = make_comparator()
```

Downstream scripts can then do:

```python
from my_comparator import mod as comp_mod
top.add_module(comp_mod, 'comp1', nets={...})
```

`flash_adc_example.py` instead imports comparator factories from
`strongarm_tb.py` and calls them when building its hierarchy. Prefer a factory
when callers need independent circuits or different design settings.

### Hierarchical `add_module`

```python
handle = top.add_module(sub_mod, 'inst1', nets={'vin': 'vinp', 'vout': 'vo'})
# Port kwargs form (preferred for typo safety):
handle = top.add_module(sub_mod, 'inst2', vin=vinp_net, vout=vo_net)
```

`add_module` returns a `SubcircuitInstance` handle. After simulation, the handle gives attribute access to internal nets:

```python
top.simulate(corner='tt')
handle.vbiasn          # placed-net reference with a hierarchical SPICE path
handle.vbiasn.y[-1]    # final DC value
```

### Behavioral placements have independent runtime state

`add_module` stores a live reference to the child, so placements share the
same authored circuit definition, specs and live parameters. The behavioral
engine gives each placement independent callable runtime state: every
independent run starts each placement from a fresh copy of its model,
including a plain function placed once. Callable closure cells and captured
Python objects are copied; ordinary Python globals and external resources
are not promised isolation or resetting. Analyses within one backend launch
share their callable state and attachment snapshot. See the
[BehavioralEngine lifetime explanation](api-reference.md#behavioralengine).

### Context-manager auto-registration (`with mod:`)

Inside a `with mod:` block, every device/source/passive constructor
auto-registers with `mod` — no explicit `add()` calls needed:

```python
from silk.elements import elements   # resistor, capacitor, inductor

with comp:
    pdk.nmos1p8('mn1', d='vo', g='vi', s='0', b='0', w=1, l=0.18)
    pdk.pmos1p8('mp1', d='vo', g='vi', s='vdd', b='vdd', w=2, l=0.18)
    sources.VoltageSource('vdd_src', positive='vdd', negative='0', dc=1.8)
    elements.capacitor('cl', p='vo', n='0', value=1e-12)
```

Nested `with` blocks push on a thread-local stack; the innermost active module
receives new elements. See [api-reference.md#context-manager-auto-registration-with-mod](api-reference.md#context-manager-auto-registration-with-mod).

---

## 7. Curated example index

Human-facing index: [examples.md](examples.md) — organized as
a learning path and by-topic tables for a newcomer browsing rather than an
agent building.

| Pattern | File | Notes |
|---|---|---|
| **Full loop** (build + multiple testbenches + PVT + run_all_tests) | `silk/examples/strongarm_tb.py` | **Canonical** full-loop structural model. Multiple named tests, including a declared process-corner sweep; `run_all_tests()` drives everything. |
| **Hierarchy + specs + FFT + SNDR** | `silk/examples/flash_adc_example.py` | Imports StrongArm from strongarm_tb; composes 4-bit flash ADC; uses `silk.metrics.sndr`; `declare_spec` + `reconcile_specs`. |
| **Bootstrapped track-and-hold** (reusable subckt; settling/droop) | `silk/examples/abo_th_example.py` | Single-NMOS bootstrapped sampler as a reusable subcircuit (`make_abo_th`); settling and droop. |
| **Differential track-and-hold** (composition; SFDR ≥60 dB) | `silk/examples/differential_th_example.py` | Composition: two `abo_th` subcircuits on complementary inputs, SFDR on the differential output. |
| **Build only + layout constraints** | `silk/examples/constraints_5t_ota.py` | Layout intent only — NOT a full-loop template. No testbench or SPICE sim. |
| **Build only + layout constraints (mirror)** | `silk/examples/constraints_current_mirror.py` | Same: layout intent only. |
| **Build only + layout constraints (diff pair)** | `silk/examples/constraints_diff_pair.py` | Same: layout intent only. |
| **Constraint schema — ingest external JSON** | `silk/examples/constraints_external_load.py` | Loads a hand-authored constraint JSON with `silk.constraints.serialization.from_json`; companion to `constraints_current_mirror.py`. |
| **Flagship design loop** (intent → build → characterize under two contexts → report + persist → reopen → rebuild + restore → EM sizing + parasitics → compare → netlist) | `silk/examples/design_loop_walkthrough.py` | The flagship: intent → build → characterize under two contexts → report and persist → reopen from the database. |
| **Flagship design loop on GF180MCU** (same code, metre-native PDK) | `silk/examples/design_loop_walkthrough_gf180mcu.py` | The flagship's steps unchanged on GF180MCU; only the `Technology` object differs (metre-native PDK). |
| **Flagship design loop on SG13G2** (same shape, one documented boundary) | `silk/examples/design_loop_walkthrough_sg13g2.py` | The flagship on SG13G2 (compiled OSDI models), with the one documented boundary. |
| **Monte Carlo** (mismatch yield, sigma per corner, parameter ranking, HTML section) | `silk/examples/mc_diff_pair_offset.py` | Matched `nmos1p8` pair offset: mismatch yield, sigma per corner, parameter ranking, HTML section. |
| **Worked example** (embedded AC + transient + noise testbenches + design specs + reconcile + netlist) | `silk/examples/ota_5t_guide.py` | The §4 walk-through: embedded AC, transient and noise tests on the module; `run_all_tests()` + `reconcile_specs()`. |
| **Factory API basics** (direct device instantiation) | `silk/examples/factory_api_example.py` | `pdk.<device>('name', pin=net, ...)` direct instantiation, `ckt.add(...)` for devices and sources, attribute assignment. |
| **PWL waveform construction** | `silk/examples/pwl_example.py` | Seven ways to build a `sources.PWL` (incremental, fluent, from lists, repeat/delay, current source, from a file). |
| **Source follower** (buffer; gain/BW + output resistance) | `silk/examples/source_follower_example.py` | Full-loop testbench-as-part-of-module for an NMOS common-drain buffer; gain, bandwidth, output resistance. |
| **Calibration bridge** (behavioral ← SPICE) | `silk/examples/source_follower_calibrate.py` | First-time SPICE calibration via `CalibrationBinding`, then [restore observations and reapply the authored recipe in a later Python process](database.md#source-follower-calibration-across-python-processes); exact characterize/restore commands, no Python/model reconstruction. |
| **Optimize / minimize power** | `silk/examples/source_follower_optimize.py` | `silk.optimize` `Optimizer` over `Variable`s: minimize bias current subject to gain and bandwidth specs. |
| **Current mirror** (copy accuracy + output resistance) | `silk/examples/current_mirror_example.py` | Electrical characterization of a current mirror: copy accuracy and output resistance. |
| **gm/Id lookup curves** (sky130 NMOS/PMOS; device-parameter probes + `Signal.versus`) | `silk/examples/gm_id_lookup.py` | Device-parameter probes (`tb.probe('M1:gm')`) and `Signal.versus` for gm/Id lookup curves; see [operating-point.md](operating-point.md). |
| **Bandgap reference** (hierarchical Banba topology; TC-defining spec) | `silk/examples/bandgap_example.py` | Hierarchical Banba bandgap composed from OTA and follower blocks, with a `DC(source='temp', ...)` temperature sweep. |
| **Ring oscillator** (f_osc + duty + *real* jitter) | `silk/examples/ring_oscillator_example.py` | 5-stage CMOS ring from `build_inverter`/`build_nand2`; frequency, duty and jitter. |
| **Parameter sweep via the embedded testbench** | `silk/examples/ring_oscillator_stage_sweep.py` | Sweeps a design knob as `build → run_characterization → read metrics` using the embedded testbench. |
| **LDO regulator** (NMOS follower pass device) | `silk/examples/ldo_regulator_example.py` | Composable NMOS-follower LDO, built to become the ring oscillator's supply. |
| **Parameterized LDO** (`LDODesign` class; db-tracked) | `silk/examples/ldo_design.py` | An NMOS-output LDO as a reusable `LDODesign` class over a `DEFAULT_PARAMS` dict, with database tracking. |
| **Behavioral twin, calibrated** (LDO) | `silk/examples/ldo_calibrate.py` | The LDO's 2nd-order behavioral model written from measured SPICE specs via `CalibrationBinding`. |
| **Behavioral twin, calibrated** (ring VCO) | `silk/examples/ring_calibrate.py` | Calibrates a ring VCO's behavioral twin: a phase integrator, not a filter. |
| **Supply-regulated VCO** (LDO + ring composed) | `silk/examples/supply_regulated_vco_example.py` | `build_ldo()` → regulated supply → `build_ring_oscillator()` composed on one shared net. |
| **Composition graphic** (draw a circuit's block diagram from the live Module) | `silk/examples/composition_graph_demo.py` | `module_graph`, `graph_to_mermaid` and `save_graph_png` draw a block diagram from the live `Module`. |
| **PLL** (full loop over the calibrated twins) | `silk/examples/pll_example.py` | Composes sample models of the controller, LDO and ring; feedback uses `previous_value`. See the [PLL limitations](examples.md#pll_examplepy) for time-step sensitivity. |
| **PFD + DFF + divider** (the PLL's digital third) | `silk/examples/pfd_example.py` | Gate cells composed upward: NAND3 → edge-triggered DFF → PFD with NAND reset → /8 ripple divider. |
| **Charge pump** (the PLL's analog third) | `silk/examples/charge_pump_example.py` | A reference mirrored into a PMOS source and NMOS sink with series switches (the PLL's analog third). |
| **Calibration bridge** (PLL non-idealities ← SPICE) | `silk/examples/pll_calibrate.py` | Opt-in refresh of the PLL's imported nominal constants from the PFD/CP testbenches, with a drift table. |
| **PLL closed in SPICE** (capstone verification) | `silk/examples/pll_spice_example.py` | The PLL closed in SPICE, every transistor in one loop (capstone verification). Tests a near-lock release, not cold-start acquisition. |
| **Twin vs transistors overlay** (one axis, same experiment) | `silk/examples/pll_twin_vs_spice.py` | Runs the same near-lock release on the SPICE and behavioral loops and overlays them on one axis. |
| **Two-stage Miller OTA** (gain / phase margin / slew) | `silk/examples/two_stage_miller_ota_example.py` | Diff pair plus common-source stage with Cc/Rz Miller compensation: gain, phase margin, slew. |
| **Folded-cascode OTA** (Variable-linked matched sizing) | `silk/examples/folded_cascode.py` | `with module:` auto-registration; matched sizing enforced by `Variable` arithmetic. |
| **Standard-cell library** (shared, characterized, behavioral twins) | `silk/examples/stdcells.py` | The shared, characterized standard-cell library (inverter, buffer, NAND/NOR, mux, DFF, ...) with behavioral twins. |
| **PRBS generator family** (generalized LFSR; verify scales with the period) | `silk/examples/prbs_generator_example.py` | `build_prbs(order=7, form=...)`: a generalized LFSR; verification that scales with the period. |
| **SerDes TX driver + 1-tap FFE** (pre-emphasis; eye open vs closed) | `silk/examples/serdes_tx_driver_example.py` | Differential segmented CMOS TX driver with a 1-tap FFE; eye open versus closed. |
| **CML cell library** (differential current-mode logic) | `silk/examples/cml_cells.py` | The CML (current-mode logic) cell library: differential current-steering cells and an FFE driver. |
| **CML TX eye — SPICE vs calibrated twin** | `silk/examples/cml_ffe_eye.py` | Twin-vs-SPICE eye check for the CML FFE driver: measure realized taps, calibrate the twin. |
| **Full SerDes TX** (PRBS→serializer→**4-tap FFE**→driver→channel→RX, CMOS vs CML) | `silk/examples/serdes_tx_example.py` | The full TX chain: PRBS → serializer → 4-tap FFE → driver → channel → RX, CMOS or CML. |
| **CMOS TX eye — SPICE vs calibrated twin** | `silk/examples/cmos_ffe_eye.py` | The CMOS counterpart of `cml_ffe_eye.py`: measured taps, calibrated twin, eye comparison. |
| **10:1 serializer** (CMOS vs CML — the speed/power trade) | `silk/examples/serializer_example.py` | `build_serializer(variant="cmos"|"cml", ratio=10)`: one load-shift architecture in two families. |
| **SAR ADC behavioral twin** (N-bit differential charge-redistribution) | `silk/examples/sar_adc_example.py` | N-bit differential charge-redistribution SAR ADC as a behavioral twin. |
| **SAR ADC — real transistors + self-clocking FSM** (cap-DAC + StrongARM + stdcell controller) | `silk/examples/sar_adc_spice.py` | SAR ADC with real transistors (cap-DAC, StrongARM comparator) and a self-clocking FSM; accuracy notes below. |
| **SAR twin, calibrated** (what is — and isn't — a calibration target) | `silk/examples/sar_calibrate.py` | Measures DAC gain/nonlinearity and compares transistor-loop codes with the behavioral twin. |
| **SAR ADC resolution sweep** (where extra bits stop buying accuracy) | `silk/examples/sar_resolution_sweep.py` | Sweeps SAR resolution to find the ENOB ceiling and attribute it. |
| **Simulation output modes + saved-file readback** | [silk/examples/output_modes_example.py](../silk/examples/output_modes_example.py) | Overwrite versus `new` output modes with retained rawfiles and saved-file readback. |
| **Behavioral simulation** | `silk/examples/behavioral_example.py` | Comparator + RC filter composed in a top; `simulate_behavioral`; `load_signal`; `plot`. |
| **Declarative testbench codegen** | `silk/examples/testbench_codegen_example.py` | A `TestbenchSpec` dict through `Generator().generate_testbenches` into generated Python. |
| **One parameterised block, several sizings** (`variants=True`) | `silk/examples/amplifier_variants_example.py` | One `build_cs_amp` factory with `variants=True`, placed three times at two settings. |
| **Composition / hierarchy** | `silk/examples/hierarchical_example.py` | `add_module` on two Module sub-blocks; SubcircuitInstance handle; internal net access. |
| **Optimize / sweep** | `silk/examples/parameter_sweep_example.py` | `Variable` + `VariableSpace` sampling (`lhs`/`random`/`halton`/`grid`) + `apply → generate_netlist` sweep loop. Uses the packaged sky130 catalog; no installed models or simulator. For `Optimizer`/`add_objective`/`opt.best`, see `source_follower_optimize.py`. |
| **Database / tracking** | `silk/examples/circuit_tracking_example.py` | The unscoped `db_path=` workflow end to end: sealed identity, metrics and spec measurements, reopen and query. Guide: [database.md](database.md). |
| **Reports** | `silk/examples/circuit_report_example.py` | The report flow: `generate_report` → `save_html`/`save_json` → `load_json` round-trip → `diff_reports`. Guide: [reports.md](reports.md). |
| **Reuse simulated currents, metrics, and spec measurements without re-simulating (profile-scoped)** | `silk/examples/net_current_persistence_example.py` | Profile-scoped evidence: two characterizations of one DUT stored and restored without re-simulating. |
| **Fault injection** | `silk/examples/fault_injection_example.py` | `FaultSet`, `stuck_open`, `vth_shift`; `FaultCampaign.run()`; `CampaignReport`. |
| **Layout estimation** | `silk/examples/layout_intelligence_example.py` | `estimate_layout`; statistical area and wire-length estimates. |
| **Lint / coverage** | `silk/examples/lint_coverage_example.py` | `RuleRegistry.default().run(top)`, waivers, `CoverageReport`. |
| **Bus ports** | `silk/examples/bus_ports_example.py` | `add_net(..., bits=N)` for wide buses; `BUS-002` lint rule. |
| **Port roles** | `silk/examples/port_role_example.py` | Port directions/roles and `check_interfaces`. |
| **Port-role inference** | `silk/examples/port_role_inference_example.py` | `RoleRuleset` + `infer_port_roles`; default vs. house rules; YAML; inference follows the live module definition (§8). |
| **Net object** | `silk/examples/net_object_example.py` | `Net.pins`, `Net.fanout`, rewiring via attribute assignment. |

Design results, including SAR SPICE-accuracy measurements, are in [examples.md](examples.md#design-notes-and-results).

### Utility / non-teaching files

These files are smoke or comparison utilities, rather than circuit-building templates:

* `silk/examples/diff_pair_construction_example.py` — construction-only factory
  and structure inspection with a 1.8 V supply. Builds two ground-referenced
  NMOS devices with undriven gates and a PMOS mirror load; it does not simulate.
  For a broader factory walkthrough, read `factory_api_example.py`.
* `silk/examples/compare_outputs.py` — its own docstring says it: "retained
  for regression testing only... For new circuits use Module directly." Runs
  the same differential-pair circuit through two identically-implemented
  functions and diffs the output signals as a numerical backward-compat check,
  not a pattern demonstration.

---

## 8. Gotchas

### Late-binding lambda captures

When registering metrics in a loop, capture the loop variable explicitly:

```python
# WRONG — all lambdas capture the same 'vm' from the enclosing scope
for vm in (0.85, 1.00, 1.15):
    tb.add_metric(f'gain_{vm}', lambda ckt: measure(ckt, vm))

# CORRECT — bind vm at definition time
for vm in (0.85, 1.00, 1.15):
    tb.add_metric(f'gain_{vm}', lambda ckt, vm=vm: measure(ckt, vm))
```

The function form has no such trap: a probe is a **value**, not a closure over
the enclosing scope, so `tb.add_metric(f'gain_{vm}', dc_gain_dB,
sig=tb.probe(f'vout_{vm}'))` in the same loop binds a different Signal each
time with nothing to remember.

### PDK units — µm W/L and SI suffixes

Every shipped manifest sets `user_units: u` — circuit code writes microns on
every PDK. Sky130 W/L are in **µm** — `w=4.0` means 4 µm, not 4 m. The
generated netlist emits values in each parameter's own `pdk_unit`: W/L appear
as plain numbers in that unit (sky130: `w=4 l=0.5`), while other values such
as times and capacitances carry SI suffixes (`40n`, `10f`). These are correct;
do not convert to SI.

### Behavioral simulation requires port direction

Every modelled subcircuit used in a behavioral run must have port directions
declared via `define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT, ...})`.
Declare IN/OUT directions for dependency ordering. A declared port with
direction `None` defaults to INOUT, which triggers a `UserWarning` because
evaluation order is assumed.

### `import silk.constraints` is a side-effect import

`Module.add_constraint` and `Module.list_constraints` are **not** on the bare
`Module` class. They are monkey-patched onto `Module` by `silk.constraints`'s
`__init__.py`. Always do:

```python
import silk.constraints          # side-effect: patches Module
from silk.constraints import SymmetricBlocks, GroupBlocks
```

before calling `mod.add_constraint(...)`. Calling without the import raises
`AttributeError`.

### `silk.metrics` — one library, organised by purpose

| Submodule | What it covers | Import |
|---|---|---|
| `silk.metrics.ac` | AC/waveform analysis on `Signal` objects: gain, bandwidth, phase margin, CMRR, group delay, PSRR | `from silk.metrics import dc_gain_dB, bandwidth_3dB, phase_margin` |
| `silk.metrics.{waveform,statistics,timing,jitter,eye,power,spectral,rf,converter,noise,fom}` | ADC (INL/DNL/SNDR/SFDR/ENOB), spectral (FFT/PSD/THD), timing/jitter, noise figures, FoMs | `from silk.metrics import sndr, enob, inl, period_jitter, walden_fom` |

Everything is re-exported from the `silk.metrics` top level, so one import
line covers AC characterization and ADC dynamic metrics together.

For settled measurements, use `average(sig, start=..., end=...)` (an
x-weighted mean) alongside `peak_to_peak(sig, start=..., end=...)` to check
steadiness. `final_value(sig)` only reads the last sample; it does not prove
settling. `value_at(sig, x=...)` reads/interpolates a point. For a step,
`settling_time(sig, initial=..., target=..., tol=..., start=...)` finds the
last-out-of-band settling time; it has no `end` argument, so use
`window(sig, start, end)` first to restrict the trace. `window` returns a
Signal and preserves real or complex samples. See the
[metric library](metrics-library.md) for signatures and failure conventions.

For an NMOS output sink held by a voltage source named `vout`, signed sink
current is `-tb.probe('vvout', 'I')`: a reversed current must fail the copy
spec. The current-mirror example averages 0.5–1 us and checks peak-to-peak
ripple against 0.1% of its reference current. With DC sources, this checks
steadiness around the initial operating point, not power-on startup.

For low-frequency output resistance, declare an AC sweep from 0.1 Hz with
samples through 10 Hz, then use deferred probe arithmetic:

```python
from silk.metrics import value_at, peak_to_peak

gout = (-tb.probe('vvout', 'I') / tb.probe('out')).real()
rout_signal = 1.0 / gout
rout = tb.add_metric('rout_ohm', value_at, sig=rout_signal, x=0.1)
spread = tb.add_metric('rout_low_band_pp', peak_to_peak,
                       sig=rout_signal, start=0.1, end=10.0)
tb.add_metric('rout_flatness_pct', 100 * spread / abs(rout))
tb.add_spec('Rout flatness < 1%', 'rout_flatness_pct', '<', 1.0)
tb.add_spec('Rout > 50 kOhm', 'rout_ohm', '>', 50e3)
```

Divide the complex current by the **actual voltage phasor**, then call
`.real()` explicitly before real-valued metrics. This measures `1/Re(Y)`,
not `Re(1/Y)` or impedance magnitude. Read the low-frequency endpoint and
check flatness across the band; do not use `average` over a log AC sweep as
Rout, because its x-weighting gives the top decade about 91% of the weight
between 0.1 and 10 Hz. The positive resistance spec also rejects wrong signs.

Import AC/noise metrics from `silk.metrics`.

### `Optimizer` consumes testbench metrics directly

Pass a `ModuleTestbench` from `mod.create_test()` directly to
`silk.optimize.Optimizer`; it reads `.metrics` from the returned `SpecReport`.
Custom tests whose `run(corner=...)` returns `{'metrics': {...}}` remain
supported.

```python
opt = Optimizer(mod.get_test('ac'), VariableSpace([i_ref, w_follower]))
```

`Optimizer.evaluate(point, corner=...)` applies the variables and calls
`test.run(corner=...)` once. `ModuleTestbench.evaluate(rawfile=...)` measures
existing simulation data without launching a simulator; these are separate
operations. Variables linked to the live testbench circuit change the
simulated metrics when their values are updated.

The testbench's specs constrain feasibility by default. Use
`opt.add_margin('BW > 10 MHz', 90e6)` to tighten one spec in its target's
numeric unit, or `opt.add_constraint('bw_hz', min=100e6)` to override the
spec-derived limit with an inclusive floor. Results and `opt.summary()`
record each limit's source. Design knobs need no metric:
`opt.add_objective(i_ref, minimize=True)` reads `i_ref` from each recorded
sample; measured objectives still use their registered metric names.

### Derive a metric from other metric results

Keep the handles returned by `add_metric`, then register their arithmetic as
another metric. The inputs are evaluated once; the derived metric uses their
reported numbers and can have its own spec. For example, after registering
`i_up` and `i_dn` with catalog `average` on windowed current probes:

```python
mismatch = tb.add_metric('mismatch_pct',
                         100 * abs(i_up - i_dn) / ((i_up + i_dn) / 2))
mismatch.add_spec('matching', 'mismatch_pct', '<', 10)
tb.run()
print(mismatch.value)
```

Numbers mix with handles using `+ - * / ** abs`. Handles forward public
builder methods to their testbench. Read `.value` before comparisons;
`bool(handle)` raises and equality/hash use identity. See
[probes](probes.md#derive-metrics-from-reported-values) for a complete example
and copy, pickle and result-lifetime details.

### Reading currents and DC sweeps

Check these details when reading a current or running a `.dc` sweep:

- **Branch currents come from voltage-source ammeters.** DC/AC-only
  `VoltageSource(name, ...)` emits as `V<name>`; waveform sources use separate
  waveform and DC-offset cards. Use the logical probe `tb.probe('i(vout)')`
  for a source named `vout`, or load it with the `'I'` signal type:
  `ckt.load_signal('vvout', 'I')` for a source named `'vout'`. There is no
  current vector for a node — put a (0 V if needed) voltage source where you
  want an ammeter. For output resistance, normalize the current by the actual
  AC node voltage, then take the reciprocal real admittance near DC (recipe
  above); the ammeter sign matters.
- **`DC` sweeps work — and the swept variable is the signal's x-axis.** After a
  `DC(source='temp', start=-40, stop=125, step=5)`, `load_signal('vref').x` is
  *temperature*, not time. So a swept figure of merit (a temperature coefficient,
  an output resistance from a `V_out` sweep) is an ordinary metric over an
  ordinary testbench — one simulation, one metric, one spec. See
  `bandgap_example.py`'s `tc` testbench.
- **A `.dc` solve does not exercise a startup circuit.** It jumps straight to an
  operating point. If the block has one (bandgap, reference, oscillator), keep a
  transient with a supply ramp alongside the sweep — that is the only thing
  proving the circuit actually *starts*.

### `save_device_currents` is a transient-only opt-in

`mod.save_device_currents = True` makes every testbench run on that module
emit device-current saves on ngspice, which requires exactly one Transient
analysis (`SimulationError` otherwise). Xyce warns and omits device-current
saves. On ngspice, a module that owns both transient and AC tests must
toggle the flag per run (`design_loop_walkthrough.run_test` does exactly
that); it is not a set-once property.

### `bandwidth_3dB` resolution is the AC sweep grid

`bandwidth_3dB` returns the last sweep point still above −3 dB — it does not
interpolate — so its resolution is the grid spacing: 12 % per point at
`points=20` per decade, 1.2 % at `points=200`. A before/after comparison
(routing parasitics, a sizing tweak) that moves bandwidth by a few percent is
invisible at 20/decade and reads as "no change". Sweep finely enough for the
effect you are measuring, and say what the grid is when you report a delta.
If the highest sampled frequency remains at or above the threshold,
`bandwidth_3dB` warns and returns NaN; extend the sweep.

### Port-role inference follows live module definitions

`infer_port_roles(module, ruleset)` fills unset roles on the module's own
`pin_list` after `define_module_ports`. Parents refresh their instance metadata
and net bindings from the live definition, so inference works before or after
`add_module`. Explicit roles are never overwritten. Use `iter_instances()` to
inspect the hierarchy; repeated placements yield distinct records referring
to the same live module definition.
See `docs/authoring-port-role-rules.md` for authoring a ruleset.
