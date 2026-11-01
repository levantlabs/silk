# silk capability roadmap — current capabilities and future directions

This map separates shipped capabilities from possible extensions. Future directions
are ideas, not release commitments. For open work, see the
[issue tracker](https://github.com/levantlabs/silk/issues).

The design loop is **build → simulate → measure → check → store → generate**.
The [agent guide](agent-guide.md) shows how to use it in a circuit design.

## What silk does today

### Build and represent

- **Hierarchical circuits** — `Module`, `add_module`, `define_module_ports`,
  buses, voltage domains, and `SubcircuitInstance` placement handles.
- **Sources** — `VoltageSource` and `CurrentSource`, with waveforms such as
  `Pulse`, `PWL`, and `Sine`. Support depends on the simulator; see
  [sources](api-reference.md#silksources).
- **SPICE netlists** — top-level circuits and subcircuits for ngspice and Xyce,
  selected through the module's simulator setting. Custom SPICE text must use
  syntax supported by the selected simulator.

### Simulate

- **SPICE analyses** — `Transient`, `AC`, `DC`, and `Noise` via
  `Module.simulate` or registered testbenches. ngspice is release-validated;
  Xyce is a preview backend. See
  [backend coverage and limits](known_issues.md#simulator-backends).
- **Behavioral simulation** — `set_behavioral_model` and the behavioral backend
  provide functional transient and DC checks without an external simulator or
  installed PDK models. `simulate_behavioral` is a transient shortcut. See
  [behavioral limits](known_issues.md#behavioral-simulation).
- **Monte Carlo** — process and mismatch sampling for SPICE, plus declared
  behavioral parameter variation. Xyce sampling is supported with restrictions;
  see [Monte Carlo](monte-carlo.md).

### Measure

- **One metric library** — `silk.metrics` is organized by what each metric measures and
  accepts signals for waveform measurements. Examples include `dc_gain_dB`,
  `bandwidth_3dB`, `phase_margin`, `sndr`, `enob`, `sfdr`, `inl`, `dnl`, and
  `period_jitter`. See [metric definitions](metrics-library.md) for input and
  failure contracts.
- **Signals and probes** — `Signal` and `NgspiceSignal`, deferred probe
  expressions, arithmetic, and AC-aware plotting. Noise results use the shared
  names `noise_out` and `noise_in` across SPICE backends. See [probes](probes.md).

### Check designs and explore alternatives

- **Specs and calibration** — `Spec`, `Measurement`, `CalibrationBinding`, and
  `reconcile_specs` connect design targets, measured results, and behavioral
  models. See [the design-stage bridge](design_stage_bridge_user_guide.md).
- **Reusable testbenches** — `ModuleTestbench`, `create_test`, and `run_corners`
  for process, supply-voltage, and temperature checks.
- **Lint and annotation** — interface, voltage-domain, and junction-stress checks;
  `infer_port_roles(module, ruleset)` fills unset roles using configurable rules,
  and `require_annotation_coverage` checks annotation completeness. Junction
  checks use behavioral or SPICE data; missing ratings or saved terminal data
  limit coverage. See [known issues](known_issues.md#junction-stress-coverage-is-limited-by-ratings-and-saved-terminal-data).
- **Fault campaigns** — `FaultSet`, `FaultCampaign`, and faults such as
  `stuck_open` and `vth_shift`. Campaigns require a SPICE backend; see
  [fault injection](api-reference.md#silkfaults--fault-injection-and-monte-carlo-campaigns).
- **Layout intent and estimates** — `SymmetricBlocks`, `SameTemplate`, and
  `GroupBlocks` describe constraints. `estimate_layout` with `RoutingModel`
  estimates footprints and per-net resistance and capacitance; `capture_layout`
  snapshots that state. `inject_parasitics` adds estimated loading to a circuit.
  These are statistical estimates, not geometry extraction; the RC injection
  preserves original terminal connections. See [layout intelligence](api-reference.md#silklayout).
- **Optimization and sweeps** — `VariableSpace`, `Optimizer`, and `Variable`.
  See [optimization](optimizer.md).

### Store evidence and generate code

- **Optional database tracking** — versioned structure, design stages, lineage,
  metric evidence, spec measurements, and selected operating-current sets.
  Characterization profiles identify the producing test and declared parameters;
  evidence can be selected at an exact process, voltage, and temperature point.
  The database does not store raw waveforms or restore parasitics or calibration
  state. See [database](database.md).
- **Code generation and schema export** — `Generator` turns spec JSON into silk
  Python; `export_schema` describes a module for catalog consumers. See
  [code generation](codegen-contract.md) and [schema export](schema-export.md).

## Future directions

### Geometry-aware layout

Connect real layout geometry to circuit instances, then use placement and routing
to estimate systematic device variation and interconnect effects. Existing
statistical estimates and loading injection provide a starting point; geometry
import, position-dependent variation, and extracted-layout accuracy remain future
work.

### A reusable circuit-block library

Package curated, tested composite circuits such as amplifiers, comparators,
current mirrors, and switched-capacitor blocks as an importable library built on
silk. Today, worked designs live in [the examples](agent-guide.md#7-curated-example-index).
The package boundary and distribution name are undecided. Atomic devices already
belong to the PDK catalogs and `ElementLibrary`.

### More automated design decisions

Build higher-level workflows that use measurements to choose the next sweep,
optimization step, or design revision. The metric library, specs, and optimizer
already exist; broader automation can compose them.

### Broader simulator coverage

Extend the backend interface to other simulation engines and increase Xyce
coverage beyond its preview status. Existing Xyce Monte Carlo support does not
remove restrictions such as the lack of per-block mismatch disabling, transient
noise, and multiple analyses in one deck. See
[backend limits](known_issues.md#simulator-backends) before choosing a simulator.

### A richer design-flow record

Build workflows for comparing design revisions, applying release checks, and
round-tripping layout and parasitic evidence. The database already tracks versions
and selected measurement evidence; it is not a complete archive of every run.

### An exploratory natural-language front end

Translate design intent into a silk circuit and its checks. This would build on
the existing circuit, simulation, measurement, and verification APIs.
