# Changelog

Notable changes to **circuitsilk** (`import silk`). Pre-1.0: minor versions may
change APIs.

## 0.3.0 — unreleased

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
