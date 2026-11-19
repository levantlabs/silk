# Changelog

Notable changes to **circuitsilk** (`import silk`). Pre-1.0: minor versions may
change APIs.

## 0.4.0 — unreleased

- Generated testbenches save device parameters such as `gm` and `vth` for metrics and, through a new per-test `probes` list, for plotting.
- Device-parameter probes declared on a testbench read back after `evaluate(rawfile=...)`, and the waveform viewer labels them by designer name (`dut.MN:gm`) in results silk produced.
- `tb.device_probe_address('MN:gm')` returns a device parameter's raw vector address and label without simulating.
- Circuit reports summarize corner, Monte Carlo and single-run results, including after reload, show "not measured" where a value is missing, and add pooled mean, standard deviation and worst sample to Monte Carlo tables.
- One test's declared conditions run in parallel with `run(n_workers=...)` or `run_test(..., n_workers=...)`.
- `prepare(corner=..., sample=...)` writes one Monte Carlo sample's deck without simulating.
- Source currents can be named by designer path in probes, metric strings, circuit-backed `load_signal` and viewer selections. In probes and metric strings, a conflicting emitted reading wins with a warning. Bare-`I` probes that name only a source alias, and unsafe current saves, are refused before launch.
- A selected placement's results, viewer and plots use that placement's latest successful run. Its `netlist` is now an adapter that shares the authored configuration; run or load through the selected test, and the adapter cannot be pickled.
- Sweep members and corner, point and sample readers read their own retained run, so those result files must stay available.
- `TestFailure.from_outcome` is removed; use `TestFailure.from_exception(error)`.
- `Module.get_output_signal_list()` and `output_file_available` are removed; use `get_signal_names()` and `sim_results_loaded`.
- SPICE numbers are written with up to 15 significant digits instead of 6, so deck text can change. Generated Python keeps exact literals and rounds other very small or large numbers to 15 digits.
- New read-only testbench declarations (`analyses`, `temperatures`, `supply_voltages`, `variable_values`, `groups`), `silk.signals.SELECTABLE_ANALYSES` and `silk.circuit.validate_port_name`; `TestFailure` and `RunRecord` are included in `from silk.circuit import *`.
- `abs(signal)` and `+signal` work on signals and probes, and `Signal.point()` accepts a corner name or grid key.
- sky130 high-poly resistor (`res_hpo`) body and head Monte Carlo draws are recorded, and other unrecorded PDK device types are listed in `skipped_types`. SPICE Monte Carlo with process variation warns before using device types the PDK manifest flags.
- Lint no longer reports a net driven through a 0 V source as undriven.
- Clearer errors for duplicate spec keys and for invalid AC or Noise point counts in generated testbenches.
- YAML files load faster with PyYAML's C loader when it is available; YAML syntax-error messages may differ.
- Shorter public agent instructions, without maintainer-only rules.
- Fixes: a behavioral `run_all_tests()` records a failed test instead of stopping, though configuration errors still raise; metric warnings are kept when the metric then fails; AC readback of a removed current source is refused the same way on ngspice and Xyce.

## 0.3.0 — 2026-11-02

- Standard sky130 transistors with `m` above 1 now pass `mult=m` to the model, so their Monte Carlo mismatch matches separate parallel devices. Mismatch results for those devices change.
- Standard sky130 transistors accept an optional finger count `nf`.
- Two new known issues: a selected placement's `netlist.view()` can show another placement's run, and a prepared ngspice deck run from another directory can finish without writing its rawfile.

## 0.2.0 — 2026-10-31

- First public release of circuitsilk: custom circuit design in Python, from specs to layout.
- Reusable circuit modules carry testbenches, metrics, specs and optional behavioral models.
- Behavioral simulation and transistor-level SPICE share one characterization workflow.
- PVT corners, Monte Carlo, parameter sweeps and metric-driven optimization explore designs.
- Metrics cover gain, bandwidth, timing, data converters, jitter, noise, distortion and power.
- Typed errors reject missing or incomplete simulation results.
- Analog lint and fault-injection campaigns check circuits beyond nominal simulation.
- Layout is in progress, with parasitic estimation and simulation feedback.
- SQLite design history records versions, metrics and spec results.
- Keyword-only APIs, schema export and an agent guide support automated design workflows.
- sky130 is release-validated; GF180MCU, SG13G2 and Xyce remain preview integrations.
- The earlier 0.1.x packages were yanked.
