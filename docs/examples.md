# Examples guide

`silk/examples/` ships 82 runnable scripts, from a 30-line behavioral demo up
to a full transistor-level PLL closing lock in SPICE. This guide is the
human-facing map: what to read first, and where to go for a given topic.

If you're building a silk circuit with an AI agent, or you want the
API-level detail behind each example (which classes, which methods, which
gotchas), see [agent-guide.md](agent-guide.md) §7 instead — this page only
tells you what to read and in what order.

## How to run

Every example is a module — run it as:

```bash
python -m silk.examples.<name>
```

e.g. `python -m silk.examples.behavioral_example`. For files in
`silk/examples/metrics/`, use `python -m silk.examples.metrics.<name>`.

For noninteractive runs, prefix the command with `MPLBACKEND=Agg`.

The **Needs** label names the minimum environment required for the workflow
described in each row:

- **PDK-free** — needs neither PDK files nor a simulator.
- **PDK only** — uses the packaged PDK catalog to build, inspect, or emit a
  circuit; needs neither installed model files nor a simulator.
- **ngspice only** — runs an ideal-element SPICE deck without installed PDK
  model files. Requires ngspice and silk's Python dependencies.
- **PDK + ngspice** — runs a SPICE analysis using PDK devices, so needs both
  sky130 and ngspice. These decks include the PDK model library.
- **GF180MCU + ngspice** — the one example that runs on GlobalFoundries 180 nm
  instead of sky130; it needs `gf180mcuD` under `$PDK_ROOT` (same install
  guide) and ngspice.
- **SG13G2 + ngspice** — the one example that runs on IHP SG13G2; it needs the
  SG13G2 PDK with its OSDI plugins compiled under `$PDK_ROOT`, and an
  OSDI-capable ngspice (`$SILK_NGSPICE_BIN` — PATH's stock ngspice-42 cannot
  load SG13G2's plugins) — see the
  [SG13G2 section](installing-the-pdk.md#sg13g2-simulates-end-to-end-with-two-environment-requirements-silk-cannot-supply).

The label covers the advertised workflow, not merely a zero exit status. Some
SPICE examples catch a missing simulator and print a reduced build-only
demonstration; they remain **PDK + ngspice** when the measurements described in
the table require simulation. Optional modes are called out explicitly.

Output files (plots, HTML reports) land in `silk-output/` under the
directory you run the example from. Set `$SILK_EXAMPLES_OUT` to put them
somewhere else; see [docs/cli.md](cli.md#a-note-on-silkexamples).

After a fresh install, the smoke test is `python -m silk.doctor`, then
`python -m silk.examples.behavioral_example` (PDK-free) and
`python -m silk.examples.ota_5t_guide` (PDK + ngspice). To run the whole set
from a source checkout, use `python scripts/run_examples.py` (a contributor
tool, not shipped in the wheel; it takes hours — add `--filter <substring>` to
run a subset, `--timeout` to cap each one). `PDK_ROOT` resolves from the
environment, then the [silk settings file](settings.md), then `~/pdk`.

## Start here — a learning path

Seven examples, read in this order, take you from zero install to a closed
design loop. Each one is also the launching point for its own topic below —
they're not repeated in the topic tables.

1. **[`behavioral_example.py`](../silk/examples/behavioral_example.py)** —
   **PDK-free.** A comparator and an RC filter, wired up and simulated in pure
   Python. Run this first to see the shape of a silk script before you owe
   anything to a toolchain.
2. **[`ota_5t_guide.py`](../silk/examples/ota_5t_guide.py)** — **PDK +
   ngspice.** Builds a five-transistor operational transconductance amplifier
   (OTA) and measures its DC gain, 3 dB bandwidth and 1% settling time.
   The AC test uses `tb.probe('vout')` to select the output signal before
   simulation and `add_metric(...)` to register gain and bandwidth measurements.
   `run()` simulates and evaluates the measurements against specs; the example
   then plots the AC response and transient step response. The settling test
   measures how long the output takes to stay within ±1% of the step swing
   around its final value.
   See [agent-guide.md §4](agent-guide.md)
   for a walkthrough and the [probe guide](probes.md) for signal selection.
3. **[`parameter_sweep_example.py`](../silk/examples/parameter_sweep_example.py)**
   — **PDK only.** Turn one design knob into a `Variable`, sweep a
   `VariableSpace` of values four different ways, and see the design-space picture
   without running a simulator.
4. **[`circuit_tracking_example.py`](../silk/examples/circuit_tracking_example.py)**
   — **PDK only.** Save results to a local database, across corners, and read
   them back — the "don't lose what you measured" step.
5. **[`circuit_report_example.py`](../silk/examples/circuit_report_example.py)**
   — **PDK only.** Turn a run into an HTML/JSON report you can hand to someone
   else, and diff two reports against each other.
6. **[`layout_intelligence_example.py`](../silk/examples/layout_intelligence_example.py)**
   — **PDK only; ngspice optional for the final step.** Estimate a layout from
   the circuit and inject its parasitics back into the netlist. The final
   before/after bandwidth simulation runs when ngspice is available and
   degrades gracefully when it is not.
7. **[`design_loop_walkthrough.py`](../silk/examples/design_loop_walkthrough.py)**
   — **PDK + ngspice.** Sizes a sampling buffer from gain, bandwidth and
   settling requirements, then simulates it at two temperatures and supplies.
   Saves measurements and currents to a database, rebuilds the circuit, and
   reloads one set of results. Uses the saved currents to size wiring,
   estimates parasitics, and measures the resulting bandwidth change.
   Compares reports and generates a netlist from the same circuit.
   `python -m silk.examples.design_loop_walkthrough [OUT_DIR]`.

   **[`design_loop_walkthrough_gf180mcu.py`](../silk/examples/design_loop_walkthrough_gf180mcu.py)**
   — **GF180MCU + ngspice.** Runs the same design process with 3.3 V devices.
   Checks that device dimensions, emitted netlist values and estimated layout
   sizes agree despite the PDK using metres in its netlists.
   `python -m silk.examples.design_loop_walkthrough_gf180mcu [OUT_DIR]`.

   **[`design_loop_walkthrough_sg13g2.py`](../silk/examples/design_loop_walkthrough_sg13g2.py)**
   — **SG13G2 + ngspice.** Runs the process with SG13G2's compiled OSDI models.
   Demonstrates why silk refuses device-current recording with this backend
   configuration (see [known issues](known_issues.md)), then reloads metrics
   and spec measurements without currents. Its layout estimate therefore
   uses minimum wire widths. Compares bandwidth before and after parasitics.
   `python -m silk.examples.design_loop_walkthrough_sg13g2 [OUT_DIR]`.

## By topic

Everything else is grouped by what you're trying to learn. The **Needs** column
uses the dependency labels above.

### Building blocks & sources

The small mechanics every other example is built from: instantiating
devices, building waveforms, wiring hierarchy.

| Example | What you'll learn | Needs |
|---|---|---|
| `factory_api_example.py` | The direct `pdk.<device>(...)` instantiation style, adding devices and sources in one call, and inspecting a device's SPICE line before you add it. | PDK only |
| `source_examples.py` | Every source type silk ships — pulse, sine, PWL, SFFM, AM, and controlled sources. | ngspice only |
| `pwl_example.py` | Seven ways to build a custom (PWL) waveform, from incremental points to loading one from a file. | PDK only |
| `hierarchical_example.py` | Builds nested inverter/buffer hierarchy, reuses buffer instances, and plots top-level signals. | PDK + ngspice |
| `net_object_example.py` | What a `Net` object actually holds — its pins, its fanout — and how rewiring a circuit by attribute assignment works. | PDK only |

### Amplifiers & analog cells

Characterizing real analog building blocks: gain, bandwidth, offset,
matching.

| Example | What you'll learn | Needs |
|---|---|---|
| `current_mirror_example.py` | Measuring copy accuracy and output resistance on a current mirror, including how to read a branch current off a voltage-source ammeter. | PDK + ngspice |
| `source_follower_example.py` | A source-follower buffer's gain, bandwidth, and output resistance, plus a first calibratable behavioral model. | PDK + ngspice |
| `source_follower_calibrate.py` | Writing SPICE-measured values onto a behavioral model, then [restoring observations and reapplying bindings in a fresh process](database.md#source-follower-calibration-across-python-processes). | PDK + ngspice to characterize; catalog + database to restore |
| `folded_cascode.py` | Keeping matched-pair and mirror-ratio sizing invariants correct across many transistors by linking them with variable arithmetic instead of copy-pasting numbers. | PDK + ngspice |
| `two_stage_miller_ota_example.py` | Measuring open-loop gain and phase margin honestly — biased closed-loop, measured open-loop — plus slew rate. | PDK + ngspice |
| `bandgap_example.py` | A real hierarchical design (a bandgap voltage reference) built from smaller amplifier blocks, with temperature coefficient as a first-class spec. | PDK + ngspice |
| `gm_id_lookup.py` | gm/Id sizing lookup curves for sky130 NMOS and PMOS at two L values, built from `tb.probe('M1:gm')` device-parameter probes over one DC sweep and `Signal.versus()` (re-plot one Signal against another's values): gm/Id vs Id/W, gm/gds vs gm/Id, ft vs gm/Id, Id/W vs gm/Id. Reads a design point off the curves (pick gm/Id, get Id/W, gain and W), checks gm/Id against `op_table()` at the same bias, checks the W independence of the curves, and reports where sky130's binned PMOS models make gm/Id non-monotonic. Writes a self-contained HTML page. | PDK + ngspice |
| `amplifier_variants_example.py` | Placing one parameterised amplifier several times with different sizes and load caps, and reading the `.subckt` names that `variants=True` gives each distinct build (`cs_amp_<hash>`). | PDK only |

### Testbenches, metrics & specs

| Example | What you'll learn | Needs |
|---|---|---|
| `strongarm_tb.py` | Builds a StrongARM comparator with named threshold, decision-time and offset tests; `run_all_tests()` also runs a five-process-corner threshold sweep. | PDK + ngspice |
| `output_modes_example.py` | Explicit overwrite (1 → 2 V), retained `new` runs (3 / 4 V), then stable-base overwrite (5 V), with independent saved-file readback and an embedded voltage test. Saves actual paths immediately; explains live reports, labels and failure semantics. Run `MPLBACKEND=Agg .venv/bin/python -m silk.examples.output_modes_example`; outputs under `silk-output/output_modes/`. | ngspice only |
| `testbench_codegen_example.py` | Generates Python testbenches from `TestbenchSpec` with `Generator().generate_testbenches`; invalid builder identifiers and unknown metric types raise `TestbenchSpecError` before code is produced. | PDK-free |
| `mc_diff_pair_offset.py` | A Monte Carlo campaign end to end: the input offset of a matched `nmos1p8` pair at `tt`/`ss`/`ff`, 50 mismatch samples each, declared with `set_corners` + `set_monte_carlo(record_draws=True)` and run by `run_all_tests`. Prints yield per spec with Wilson intervals, the offset sigma per corner, the worst sample, the top rows of the drawn-parameter ranking and the failed count, then saves the HTML report section. Keeps every sample's deck and rawfile under `$SILK_EXAMPLES_OUT/mc_diff_pair_offset/`. Run `MPLBACKEND=Agg .venv/bin/python -m silk.examples.mc_diff_pair_offset`. See [monte-carlo.md](monte-carlo.md). | PDK + ngspice |

### Design-space exploration & optimization

| Example | What you'll learn | Needs |
|---|---|---|
| `ring_oscillator_stage_sweep.py` | The payoff of building tests into the module: sweeping a design knob (stage count) becomes a three-line loop, no rewiring. | PDK + ngspice |
| `source_follower_optimize.py` | Driving a real constrained optimization — minimize bias current subject to bandwidth and gain limits (testbench specs tightened with `add_margin`) — with `Optimizer` over SPICE. | PDK + ngspice |

### Tracking, reports & data

| Example | What you'll learn | Needs |
|---|---|---|
| `ldo_design.py` | Wrapping a design in a reusable parameterized class, with every version saved to the tracking database as you build. | PDK + ngspice |
| `net_current_persistence_example.py` | Saving simulated branch currents once and reusing them later — feeding a layout tool without re-running SPICE. | PDK + ngspice for the initial run; PDK only for reload |
| `composition_graph_demo.py` | Drawing a circuit's own block diagram (a Mermaid flowchart or a PNG) straight from the live `Module` object. | PDK only |

### Layout, EM & parasitics

| Example | What you'll learn | Needs |
|---|---|---|
| `constraints_5t_ota.py` | Declaring layout intent (symmetry, grouping) on a circuit without ever simulating it. | PDK-free |
| `constraints_current_mirror.py` | Same layout-intent pattern, on a current mirror. | PDK-free |
| `constraints_diff_pair.py` | Same layout-intent pattern, on a differential pair. | PDK-free |
| `constraints_external_load.py` | Loading layout constraints from a hand-authored JSON file instead of declaring them in Python. | PDK-free |

### Checkers, lint & robustness (incl. faults)

| Example | What you'll learn | Needs |
|---|---|---|
| `lint_coverage_example.py` | Running silk's analog lint rules against a circuit and reading a coverage report. | PDK-free |
| `bus_ports_example.py` | Wide bus ports, and the lint rule that catches a bus wiring mistake. | PDK-free |
| `port_role_example.py` | Declaring port roles (`Port.power()`, `Port.signal()`, ...) and checking that an interface matches what it claims to be. | PDK-free |
| `port_role_inference_example.py` | Inferring port roles automatically from naming conventions instead of declaring every one by hand. | PDK-free |
| `fault_injection_example.py` | Injecting faults (stuck-open, threshold shift) into a circuit and running a fault campaign to see what breaks. | PDK + ngspice |

### Data converters (flash ADC, SAR chain)

| Example | What you'll learn | Needs |
|---|---|---|
| `abo_th_example.py` | A reusable bootstrapped track-and-hold sampler, and the even-harmonic distortion wall a single-ended sampler runs into. | PDK + ngspice |
| `differential_th_example.py` | How going differential clears the distortion wall the single-ended sampler hit — composing two samplers and measuring the difference. | PDK + ngspice |
| `flash_adc_example.py` | A 4-bit flash ADC composed from the comparator testbench, with SNDR measured by FFT and specs reconciled against it. Pass `--sine` for FFT-based SNDR, or `--no-spice` for the behavioral demonstration. | PDK + ngspice |
| `sar_adc_example.py` | A pure-Python SAR ADC model — cap mismatch and comparator noise, and the DNL/INL/SNDR they produce — with nothing to install. | PDK-free |
| `sar_adc_spice.py` | The same SAR ADC in real transistors: a switched-cap DAC, a StrongARM comparator, and a standard-cell FSM sequencing the whole conversion in one simulation. The default command tests the DAC; use `self`, `transfer` or `sine` for the self-clocked converter. | PDK + ngspice |
| `sar_calibrate.py` | Deciding what's actually worth calibrating from SPICE (DAC gain) versus what needs a Monte-Carlo run instead (comparator offset) — and why. | PDK + ngspice |
| `sar_resolution_sweep.py` | Sweeping SAR resolution to find where extra bits stop buying accuracy, and separating the comparator-noise wall from the cap-mismatch wall. | PDK-free by default; PDK + ngspice for `full`/`quick` |

### Clocking & PLL chain

The full arc: bandgap → LDO → ring oscillator → VCO → PLL, with each block
also worked as a fast calibrated behavioral model.

| Example | What you'll learn | Needs |
|---|---|---|
| `ldo_regulator_example.py` | An LDO regulator characterized for line/load regulation, PSRR, and loop stability — designed to become a downstream block's supply. | PDK + ngspice |
| `ldo_calibrate.py` | Calibrating a 2nd-order behavioral LDO model from SPICE, and the modeling lessons that fall out (slew is not bandwidth). | PDK + ngspice |
| `ring_oscillator_example.py` | A CMOS ring oscillator's frequency, duty cycle, and *real* jitter (from supply noise, not simulator artifact). | PDK + ngspice |
| `ring_calibrate.py` | Calibrating an oscillator's behavioral model — different in kind from calibrating an amplifier, because a VCO's output never settles to a value. | PDK + ngspice |
| `supply_regulated_vco_example.py` | Composing the LDO and the ring oscillator on one shared supply net, and measuring how much the regulator improves the VCO's jitter. | PDK + ngspice |
| `stdcells.py` | The shared standard-cell library (inverter, NAND, NOR, XOR, flop) every digital example below reuses, each with its own delay-characterization testbench. | PDK + ngspice |
| `pfd_example.py` | Building a phase-frequency detector and clock divider up from the standard cells, and measuring the PFD's dead-zone pulse. | PDK + ngspice |
| `charge_pump_example.py` | A charge pump's non-idealities — up/down current mismatch, output compliance range, leakage — measured directly rather than assumed. | PDK + ngspice |
| `pll_example.py` | The full PLL loop composed from a sampled controller and the reusable LDO and ring models, cold-starting and locking. | PDK only |
| `pll_calibrate.py` | Refreshing the PLL's non-ideality constants from SPICE and seeing what actually moves the lock point. | PDK + ngspice |
| `pll_spice_example.py` | The same PLL, every transistor, closing lock in real SPICE — the capstone that verifies the behavioral twins above. | PDK + ngspice |
| `pll_twin_vs_spice.py` | Running the identical lock-recovery experiment on both the SPICE PLL and its behavioral twin, overlaid on one plot. | PDK + ngspice |
| `pll_spec_tree_example.py` | The PLL's block specs (PFD, charge pump, LDO, ring, divider — tests on) and the behavioral PLL's own system specs (declared and measured by its lock test, linked to supporting block specs; links explain, they do not propagate targets or infer results) in one module tree, reconciled by a single `reconcile_specs()` call; one system spec (output jitter) is deliberately never measured, so the report shows `[UNMEASURED]` and `passed` is False. | PDK + ngspice |

### SerDes

PRBS generation through a serializer, an equalized driver, and a lossy
channel.

| Example | What you'll learn | Needs |
|---|---|---|
| `prbs_generator_example.py` | A generalized PRBS/LFSR generator, and why the way you verify it has to scale with the sequence's period. | PDK + ngspice |
| `cml_cells.py` | The CML (current-mode logic) cell library used for high-speed differential paths — a different design style from the standard cells above. | PDK + ngspice |
| `serializer_example.py` | A 10:1 serializer built two ways (CMOS and CML) — the same architecture, and the speed/power trade between the families. | PDK + ngspice |
| `serdes_tx_driver_example.py` | A segmented CMOS TX driver with a single equalization tap, and the difference between an open and a closed eye. | PDK + ngspice |
| `serdes_tx_example.py` | The full TX chain — PRBS → serializer → 4-tap equalizer → driver → channel — with both a CMOS and a CML driver variant. | PDK + ngspice |
| `cmos_ffe_eye.py` | Measuring a CMOS driver's realized equalizer taps from SPICE, calibrating a behavioral twin to them, and checking the two eyes agree. | PDK + ngspice |
| `cml_ffe_eye.py` | The same twin-vs-SPICE eye check, for the CML driver's current-steered equalizer. | PDK + ngspice |

### Waveform explorer & triage

| Example | What you'll learn | Needs |
|---|---|---|
| `eye_centering.py` | Aligning seeded NRZ and evenly spaced PAM4 waveforms with the public `eye_diagram` average-voltage crossing method. Saves before/after plots from original samples, without calculating additional eye measurements. | PDK-free |

### Metric coverage (`silk/examples/metrics/`)

These examples measure simple circuits with `silk.metrics` and compare the
results with calculations. Read them to choose a metric, prepare its input
signal and understand what the reported value means.

| Example | What you'll learn | Needs |
|---|---|---|
| `rlc_filter_ac.py` | The AC filter metrics — peak gain, Q, −3 dB bandwidth, passband ripple, stopband attenuation, group delay — on one series RLC built twice, at Q = 3.16 and at Q = 0.707, because a Q read off a flat response and a ripple read off a peaked one are both meaningless. Every metric also runs on the closed-form `H(jω)` on the same grid, which separates the circuit's error from the sweep grid's. | ngspice only |
| `inverter_timing.py` | The transient timing and primitive family — rise/fall time, slew rate, propagation delay, pulse width, duty cycle, on/off time, crossings, crossing, and the windowed min/max/argmin/argmax/average/rms/stddev/integral primitives, plus average power and energy per cycle — on one CMOS inverter, because on an inverter every number is predictable by hand and the table prints the prediction beside the measurement. Two 0 V ammeter sources supply the current traces the two power metrics require and turn the edge times into charge-conservation predictions; the transient spans exactly five whole periods, which `dynamic_energy_per_cycle` needs. Also prints the `propagation_delay(in_edge='rising')` trap (the default), which on an inverting stage returns half a period instead of a delay, and the `out_edge` recipe that names the output transition explicitly instead. | PDK + ngspice |
| `cs_amp_ac_noise.py` | The open-loop margins and the whole `.noise` family on one NMOS and one load resistor. Why `phase_margin` and `gain_margin` need **three** poles and not one (two poles approach −180° and never cross it, so the margin stays `+inf`), why the margins are read off `−vout` — a common-source stage inverts, and the loop supplies the other 180° — and why noise is measured at the drain rather than after the poles. Separate AC and noise benches keep these measurements independent. The five closed-form noise helpers are fed the numbers this run measured, including ngspice's per-instance contribution from the source resistor. | PDK + ngspice |
| `cs_amp_distortion.py` | Measures harmonic distortion, signal-to-noise ratios and spectral power from a sine wave through a common-source amplifier. Compares FFT and Welch spectra and explains how sampling affects coherence measurements. | PDK + ngspice |
| `diff_pair_rejection.py` | `cmrr_dB` and `psrr_db` — the two AC metrics that need **more than one response of the same circuit** — on a resistively loaded NMOS pair with a tail current source. Four benches, one `.ac` each: differential drive, common-mode drive, supply injection, and a fourth whose DUT holds **two instances of the same subcircuit** so that one rawfile carries both responses and the declarative `add_metric(type='cmrr_dB', signals={'vdiff': …, 'vcm': …})` binding can be shown against the direct call (they agree to 8e-15). Every printed figure has the topology's prediction beside it: CMRR = (1 + 2(gm+gmb)·Z_tail)/2, with the tail impedance measured out of the common-mode sweep, and PSRR = 20·log₁₀\|1 + jω·Rd·CL\|, which needs no measured quantity at all and which predicts that this topology rejects the supply **not at all** below its load pole. | PDK + ngspice |
| `cs_amp_two_tone.py` | Drives a common-source amplifier with two tones and measures third-order intermodulation products. Compares two input amplitudes to check whether compression affects the extrapolated input intercept point. | PDK + ngspice |
| `ring_osc_jitter.py` | Measures period jitter, cycle-to-cycle jitter, time interval error and Allan deviation on a three-stage ring oscillator. Compares a clean supply with a repeatable noisy supply and relates the jitter to supply sensitivity. | PDK + ngspice |
| `comparator_and_sampler.py` | The comparator and sample-and-hold families — `hysteresis`, `propagation_delay` (separate input/output thresholds), `value_at` and the closed-form `kT_over_C`, plus the trip offset, the kickback and the hold droop computed at the call site from `crossing`, `value_at` and `peak_to_peak` (they are not library metrics) — on a two-transistor comparator and a one-transistor track-and-hold. `hysteresis` takes **four** signal slots and is the reason `add_metric` binds by name: it is shown both ways, directly on two separate ramp transients and declaratively on a bench whose DUT holds **four instances of the same subcircuit** (rising and falling at two ramp rates), and the two routes agree to 2e-8. Why the *reference* steps while the input is held still (a static comparator decides when its input moves, so a moving input buries the kickback in its own settling), why the kickback is read on three nodes of one divider — gate, tap and source read 78 mV, 7.8 mV and exactly 0.0, the last being what an ideally driven input always reads — and why the same comparator is 17 % slower when it has something to kick. The hysteresis here is entirely dynamic, `2·Rd·C_L·(dv/dt)` from three drawn numbers, and the 4× bench proves it by changing the one term that is not drawn. | PDK + ngspice |
| `prbs_eye.py` | `eye_diagram` and `evm_constellation` on two sky130 inverter-plus-RC lanes carrying PRBS7, at τ = 0.10, 0.60, 1.00 UI. Uses `auto_center=True` to frame original-sample windows and an explicit mid-swing threshold to measure height and width, with a measured-launch RC convolution as the model. `levels` holds that supplied threshold; the summary compares the selected eye with its model. These point-geometry gaps cannot identify overlapping symbol populations; use a meaningful height margin, not merely `> 0`. Separate mid-slot and rising-crossing-recovered decision samples give a known-bit margin, EVM, wrong-decision counts and an independent sampling check. Decks and rawfiles are retained under `$SILK_EXAMPLES_OUT/metrics/prbs_eye/` (default `silk-output/metrics/prbs_eye/`). | PDK + ngspice |
| `cs_amp_power_sweep.py` | Uses a `VariableSpace` grid to sweep ten source amplitudes and measure 1 dB compression, power-added efficiency and DC-to-RF efficiency. Uses voltage-source ammeters to measure power and explains the input and output power references. | PDK + ngspice |
| `flash_adc_codes.py` | Measures code transitions, INL, DNL, monotonicity and missing codes on a 3-bit flash ADC. Compares resistor-ratio predictions with measured errors, then demonstrates a missing code caused by a miswired comparator. Also measures ENOB and converter figures of merit; the timing uncertainty supplied to the jitter calculation is an illustrative assumption. | PDK + ngspice |
| `rlc_step.py` | `settling_time`, `overshoot`, `undershoot` and `error_norm` on the same series RLC's step response (Q = 3.16, a real ring), each checked against the closed-form second-order step; and all eight waveform transforms (`derivative`, `running_integral`, `sample`, `window`, `lshift`, `flip`, `convolve`, `cross_correlation`), each a step inside a real derived measurement — loop current, delivered charge, edge-to-peak delay, ring period — never a standalone tour of the API. | ngspice only |

## Design notes and results

Per-example detail moved out of the agent guide's curated index (which keeps
pattern, file and one sentence). Each note is the guide's former text for that
example; the example's own docstring remains authoritative.

### `abo_th_example.py`

Bootstrapped track-and-hold (reusable subckt; settling/droop): Abo & Gray 1999 single-NMOS bootstrapped sampler — the **reusable subckt** (`make_abo_th(pdk, out, w_sw, c_hold, with_tests)`). Embedded acquisition-settling + hold-droop tests (`silk.metrics.timing.settling_time` plus a hold-droop figure computed at the call site from `value_at`). Keeps a behavioral+SPICE SFDR demo that shows the **even-harmonic (2nd) wall** (~52 dB) from charge injection — motivating the differential version.

### `differential_th_example.py`

Differential track-and-hold (composition; SFDR ≥60 dB): **Hierarchical composition**: instantiates two `abo_th` subckts (`make_abo_th`) on complementary inputs and measures SFDR on `vhp − vhm`. Differential cancels even harmonics (H2: −53 → −143 dBc), clearing ≥60 dB where single-ended floors ~52 dB. One transient feeds both track-end + hold-end specs; uses `silk.metrics.spectral.sfdr` on a `Signal` difference.

### `constraints_external_load.py`

Loads layout constraints from a JSON file with `silk.constraints.serialization.from_json`. Applies port roles, net properties and grouping or symmetry constraints to a circuit. A second input file demonstrates warnings for references to missing ports or nets.

### `design_loop_walkthrough.py`

Builds a five-transistor OTA with a bias mirror from gain, bandwidth and settling requirements. Measures the circuit at 27 °C/1.8 V and 85 °C/1.62 V, saves metrics, spec results and currents, then rebuilds the design and reloads a selected set of results. Uses those currents to estimate wire widths and parasitics, simulates the bandwidth change, and compares reports. Also demonstrates that results from a different circuit are rejected with `EvidenceMismatchError`.

### `design_loop_walkthrough_gf180mcu.py`

Repeats the design process on GF180MCU using 3.3 V devices. Checks that dimensions in metres in the emitted netlist agree with dimensions in microns used for layout estimation. Reloads the stored run with the highest tail current and uses it for wire sizing and parasitic estimation.

### `design_loop_walkthrough_sg13g2.py`

Repeats the design process on SG13G2 using 1.2 V devices and compiled OSDI models. Demonstrates that device-current recording is incompatible with the required OSDI loading commands (see [known issues](known_issues.md)). Reloads metrics and spec results without currents, estimates parasitics using minimum wire widths, and measures their effect on bandwidth. Results from a changed circuit are rejected.

### `mc_diff_pair_offset.py`

Monte Carlo (mismatch yield, sigma per corner, parameter ranking, HTML section): Matched `nmos1p8` pair, offset = `v(g1) - v(g2)`. `tb.set_corners([...])` + `tb.set_monte_carlo(50, record_draws=True)` declare the campaign on the test; `run_all_tests(n_workers=4)` returns one `MonteCarloResults` (key `module.test`); then `yield_fraction()`, `corner(c).stat(m, 'std')`, `worst`, `sensitivity`, `failed`, and `CircuitReport.add_monte_carlo`. Workflow and limits: [monte-carlo.md](monte-carlo.md).

### `ota_5t_guide.py`

Builds a five-transistor OTA with AC, transient and noise testbenches already registered. Measures DC gain and 3 dB bandwidth from the AC output probe, 1% settling time from the unity-gain step response, and input-referred noise. `add_metric(...)` registers measurements, `run_all_tests()` runs the tests, and `reconcile_specs()` checks the collected results. Saves AC and transient plots. See [agent-guide.md §4](agent-guide.md) for the walkthrough.

### `factory_api_example.py`

Factory API basics (direct device instantiation): `pdk.<device>('name', pin=net, ..., w=.., l=..)` direct instantiation vs the old string-based add; unified `ckt.add(mn1, ..., vdd, ...)` for devices AND sources in one call; direct attribute assignment (`mn1.w = 6.0`, `mn1.d = 'net'`) without reconstructing the device; the declarative source library `sources.vdc`/`sources.vpulse` alongside process-independent `elements.resistor`/`elements.capacitor`; `mn1.to_spice()` to inspect a device's SPICE line before adding it.

### `pwl_example.py`

PWL waveform construction: Seven ways to build a `sources.PWL`: incremental `.add(t, v)`, fluent chaining, from a list of `(t, v)` tuples, from a flat `[t0, v0, t1, v1, ...]` list, with `r=`/`td=` repeat and delay, on a `CurrentSource`, and loaded from a text file. Reference for the waveform builder itself, not a new circuit pattern.

### `source_follower_example.py`

Source follower (buffer; gain/BW + output resistance): Full-loop **testbench-as-part-of-module** for an NMOS common-drain buffer. Two AC testbenches: small-signal gain (V/V) + `bandwidth_3dB`, and **output resistance via AC current injection** (1 A into VOUT → `\|Zout\|`). Diode-connected current-mirror bias (`IBIAS`), parametrized factory, and a calibratable first-order **behavioral model**. Saves a Bode plot + report.

### `source_follower_calibrate.py`

Measures a source follower in SPICE and uses `CalibrationBinding` to set the gain and bandwidth of its behavioral model. Derives offset from a transient and compares the calibrated step response with SPICE. The [separate characterize/store and restore/behavioral commands](database.md#source-follower-calibration-across-python-processes) show how to save measurements and apply the same calibration to a freshly built model in another process.

### `source_follower_optimize.py`

Optimize / minimize power: `silk.optimize` `Optimizer` over `Variable`s (`i_ref`, `w_follower`): **minimize bias current s.t. BW > 100 MHz and gain > 0.8 V/V** (the testbench specs plus `opt.add_margin`), LHS `sweep()`, read `opt.best`. Passes the `create_test` testbench directly to `Optimizer`, which reads its `SpecReport.metrics` (see [optimizer.md](optimizer.md) for the `ModuleTestbench`/`Optimizer` workflow).

### `current_mirror_example.py`

Current mirror (copy accuracy + output resistance): Electrical characterization (vs. layout-only `constraints_current_mirror.py`). Shows two silk mechanics: **branch currents via the voltage-source ammeter** (`load_signal('vvout', 'I')` → `i(vvout)`), and a short `Transient` for the DC operating point. Copy error from a transient settle; Rout from a 1 V AC stimulus (`Rout = 1/\|I_ac\|`), read directly as `1/\|I_ac\|` rather than differentiating a sweep.

### `gm_id_lookup.py`

gm/Id lookup curves (sky130 NMOS/PMOS; device-parameter probes + `Signal.versus`): The first example to use `tb.probe('M1:gm')` / `id` / `gds` / `cgg`: one DC sweep of the gate gives every gm/Id lookup view — gm/Id vs Id/W, gm/gds vs gm/Id, ft (`gm/(2π·cgg)`, an estimate) vs gm/Id, Vov (`vgs - vth` from the `M1:vgs`/`M1:vth` probes) vs gm/Id — by plain Signal arithmetic, drawn with `Signal.plot`, plus `Signal.versus(x)`, which re-parametrises a Signal against another Signal's values (strictly decreasing x is flipped ascending; non-monotonic x raises unless `allow_nonmonotonic=True`). PMOS values are positive (the probes are sign-corrected, so Vov needs no flip). Design read-off: pick gm/Id = 15, read Id/W and gm/gds, size W for gm = 100 µS, re-run at that W (sky130 models are W-binned, so it re-sweeps once) and confirm with an `Op` run + `op_table()`. Also prints the hand check of sweep gm/Id vs `op_table()` at one bias (agrees to ~1e-15) and notes that sky130's PMOS gm/Id is not monotonic at L = 0.15 µm. Writes a self-contained HTML page under `example_output_dir()`. Guard: `tests/test_gm_id_lookup_example.py`.

### `bandgap_example.py`

Bandgap reference (hierarchical Banba topology; TC-defining spec): Builds a hierarchical Banba reference, measures temperature coefficient with a DC temperature sweep, and checks startup with a supply ramp. Two variants make a topology argument: PMOS-input senses V_be directly (57 ppm/C over −40..125°C); NMOS-input needs PMOS source-follower level shifters whose |Vth| tempco breaks the PTAT/CTAT cancellation below +5°C — the instructive counterexample that motivates the textbook's PMOS-input choice.

### `ring_oscillator_example.py`

Ring oscillator (f_osc + duty + *real* jitter): 5-stage CMOS ring, composed from `build_inverter`/`build_nand2` subckts. **Starts with no `.ic`/`uic`**: stage 0 is a NAND whose second input is `EN`, so `EN=0` parks the ring in a defined DC state and the `EN` edge kicks it into oscillation — startup is a property of the *circuit*, not the simulator (same principle as the bandgap's startup cell). The end-to-end user of the **timing/jitter** metrics: `average_period`, `average_duty`, `period_jitter`, `cycle_to_cycle_jitter`. **Jitter needs noise** — a deterministic transient yields ~0.002 ps (numerical, meaningless), so a second testbench modulates VDD with `TRNOISE` and measures the jitter it causes (0.62 ps at 20 mV rms; cycle-to-cycle 0.95 ps ≈ σ·√2, which cross-checks both metrics).

### `ring_oscillator_stage_sweep.py`

Parameter sweep via the embedded testbench: The payoff of **testbench-as-part-of-module**, in ~20 lines: `build_ring_oscillator(stages=n)` returns a DUT with its tests *already attached*, so sweeping a design knob is `build -> run_characterization -> read metrics`. No testbench rebuilt, no source re-wired, no spec re-declared. Recovers `f = 1/(2·N·t_d)` from SPICE: f falls 1743→474 MHz over N=3..11 (3.7×, matching 3.7× more stages) and the extracted per-stage delay `t_d` is flat to **1%** (96.1 ps) — a real check that the model holds rather than a fit.

### `ldo_regulator_example.py`

LDO regulator (NMOS follower pass device): Composable regulator built to become a ring oscillator's supply (→ supply-regulated VCO). **Composes** `build_two_stage_ota(input_type='pmos')` as the error amp (PMOS input senses V_CTRL near VSS; its NMOS-CS output swings high enough to drive the pass gate). Seven testbenches cover DC tracking, small- and large-signal voltage steps, load steps, load-current sweep, PSRR and reliability. A separate broken-loop build measures loop gain and phase margin. The supply is 3.3 V and the control range is 0.5–1.8 V. The reliability check intentionally fails because this demonstration uses 1.8 V devices on the 3.3 V rail.

### `ldo_design.py`

Parameterized LDO (`LDODesign` class; db-tracked): An NMOS-output LDO wrapped as a reusable `LDODesign` class over a `DEFAULT_PARAMS` dict. Built via the factory API (`pdk.nmos1p8(...)`, `ldo.add(...)`); the top-level netlist opens with `db_path=` and calls `create_circuit_version()` for version tracking, then runs a `Transient` and plots supply/bias/output waveforms. A standalone characterization circuit, not part of the LDO→ring→VCO→PLL chain below (see `ldo_regulator_example.py` for that).

### `ldo_calibrate.py`

Behavioral twin, calibrated (LDO): Every param of the LDO's 2nd-order behavioral model written from a measured SPICE spec via **`CalibrationBinding`** (gain, ζ, closed-loop f_n, slew, PSRR, output clamp). Three modeling lessons encoded: a 1st-order model *cannot overshoot* (SPICE does, 19%); **slew ≠ bandwidth** (separate steps measure each); ζ must come from the *small*-signal step (large-step overshoot is slew-induced — a category error). RMS 3.75 mV vs SPICE. NB `CalibrationBinding` attaches to a **Spec**, not a metric — a metric with no spec calibrates nothing, silently.

### `ring_calibrate.py`

Behavioral twin, calibrated (ring VCO): A **phase integrator**, not a filter — an oscillator has no value to settle to. Calibrated from the measured tuning curve (K_vco 1105 MHz/V; **dead below 1.0 V**, and the twin goes to zero rather than inventing a frequency a PLL could 'lock' to). Within 3.5% of SPICE across the range; **177× faster measured** (13.15 s → 0.07 s per µs) — the ratio that makes a PLL lock simulable.

### `supply_regulated_vco_example.py`

Supply-regulated VCO (LDO + ring composed): `build_ldo()` → VREG → `build_ring_oscillator()`, composed on one shared net. Control range is the **intersection** (LDO good to 0.5 V, ring dead below 1.0 V → 1.0–1.8 V). Composed tuning lands on the ring's own curve to ~1% (K_vco 1100 MHz/V) — *after* fixing a trap worth reading: measured in a 40 ns window the curve was monotonic, plausible and wrong (649), because the LDO takes ~0.5 µs to settle. The payoff: 200 mV @ 1 MHz on the rail → jitter 508 ps unregulated vs 3.4 ps regulated = **148× = 43 dB, matching the LDO's PSRR at 1 MHz** (three independent measurements agree). Also needs a *fine* `maxstep` — regulated jitter sits near SPICE's edge-time resolution.

### `composition_graph_demo.py`

Composition graphic (draw a circuit's block diagram from the live Module): `silk.circuit.composition.module_graph(vco, depth=1)` walks the supply-regulated VCO (built `with_tests=False`) into a graph dict, `graph_to_mermaid()` emits pasteable Mermaid `flowchart LR` text, and `save_graph_png()` renders a matplotlib PNG. A static view — no simulation run, so net edges carry names but no voltages, and the LDO/ring boxes report unmeasured (`–`); wiring this into `CircuitReport.add_composition_diagram(mod)` after a real run adds voltages + spec pass/fail to the same picture.

### `pll_example.py`

PLL (full loop over the calibrated twins): The capstone: REF → PFD → charge pump → loop filter → **[LDO twin]** → **[VCO twin]** → /N, closed. The controller, reusable LDO and reusable ring execute as three sample placements; the controller reads the ring's real OUT through a one-point delay. Loop filter **designed, not copied**: type-II R+C1∥C2 with the LDO's measured pole (7.7 MHz) and the PFD sampling delay subtracted from PM *before* committing (62° textbook → 54° real). Cold-start lock from a **dead** ring (the twin's measured 1.0 V floor): pull-in τ = N·f_ref·C/(I_cp·K_vco) ≈ 10 µs predicted lock ~30 µs, **observed 31.7 µs**; locks 25 MHz × 24 → 600.0 MHz, locked VCTRL 1.4053 V vs tuning-curve 1.405 V. The PFD/CP carry the non-idealities **measured by their own files** (reset pulse, up/down mismatch, leakage, compliance — imported as in-code `*_NOM` constants). In the measured 80 us cold-start runs, the existing four lock specs pass at both 0.2 ns and 0.1 ns. Signed phase error is +10.54/+10.76 ps and VCTRL ripple is about 352/215 uVpp respectively. Interpolating sampled OUT crossings does not recover exact edge times; these runs show substantial ripple step sensitivity, not established ripple convergence. Sample execution calls each composed block at every point. With per-run port preparation and scalar pin access, two interleaved comparisons on one machine reduced the registered ring osc selection (1 ps / 55 ns) from 3.63–3.80 s to 1.78–1.88 s, and the 80 us PLL cold-start selection at 0.2 ns from 73.65–74.98 s to 24.56–25.05 s. These are measured comparisons, not runtime guarantees; the larger ripple remains sensitive to the time step.

### `pfd_example.py`

PFD + DFF + divider (the PLL's digital third): Composes the ring's gate cells upward: `build_nand3` → the classic **6-NAND edge-triggered DFF** (7474-style, async clear) → the two-flop **PFD** with NAND reset → a **/8 ripple divider** (toggle-wired DFFs). Testbenches measure what the behavioral PLL depends on: transfer gain + linearity at two phase offsets, the **frequency-detector property** (net output slams positive for a frequency error — what cold-start acquisition leans on), the **anti-dead-zone pulse** (294 ps measured, exported in-code as `T_RESET_NOM_PS` and imported by `pll_example`; zero would mean the VCO wanders freely near lock), and exact **/8 at the PLL's real 600 MHz** (ratio 8.000, duty 49.8%). Completing the feedback path: the **synchronous /3** (`build_div3` — two flops + one NAND2+INV of next-state, illegal state self-corrects in one clock, tested at its in-loop 75 MHz) and the composed **/24** (`build_divider24` — ripple /8 first so the fast stages are the simple toggles): 600 MHz in → **25.000 MHz out = f_ref**, exactly 24.000. The dividers' `RSTB` teaches the latch-chain lesson: multiple valid DC states, so reset is the circuit's obligation. 15 PASS.

### `charge_pump_example.py`

Charge pump (the PLL's analog third): One 10 µA reference mirrored into a PMOS source + NMOS sink with series switches (UP inverted on-chip via `build_inverter`). Measures the three non-idealities the behavioral PLL carries (exported in-code as `*_NOM` constants, imported by `pll_example`): **up/down mismatch** 3.44% (a spur generator — with the PFD's equal-width anti-dead-zone pulses, only equal *currents* cancel; in the PLL model this becomes the +10.4 ps static offset), **compliance** 0.19–1.62 V via real `DC` sweeps (the PLL locks at 1.405 V — *inside*, with 215 mV margin; above 1.62 V the pump stops pumping, which now also caps the cold-start slam), and **off-state leakage** 0.042 nA. 4 PASS, no overstress (1.8 V devices on the 1.8 V CP rail).

### `pll_calibrate.py`

Calibration bridge (PLL non-idealities ← SPICE): Opt-in refresh of the PLL's imported `*_NOM` store (reset pulse, i_up/i_dn, leakage, compliance): re-measures via the PFD/CP testbenches, prints **drift table + paste-ready blocks** per file, then re-runs the PLL lock on the fresh values (`build_pll(pfd_cp=...)`) to show what actually moves — the static phase offset rides on a ~3% current *difference*, so it is the drift-sensitive output. Unlike `ldo_calibrate`/`ring_calibrate` there is **no `CalibrationBinding`**: measurement and model live in different files, so the store is in-code constants and the hand-off is a paste (persistence question backlogged).

### `pll_spice_example.py`

PLL closed in SPICE (capstone verification): Every transistor in one loop: supply-regulated VCO + /24 + PFD + CP + the designed R+C1∥C2 filter. Starts **near lock without raw `.ic` cards**: a test-mode PMOS **precharge switch** forces VCTRL at DC (the operating point solves the whole loop consistent; C1 behind the filter R initializes for free), then a scheduled hand-over — EN 20 ns → RSTB 60 ns → REF 100 ns → release 300 ns. Verifies the twins: VCTRL lands at **1.4016 V vs the behavioral 1.4052 V (−3.7 mV twin error)**, f_out 600.9 MHz tail-average (+0.15%, the release costs ~1.5 µs of phase-acquisition excursion before the tail is clean), VCTRL drift 0.38 mV (**flat integrator = locked** — the sharp criterion). The per-cycle CP correction sawtooth is visible on VCTRL: the reference-spur mechanism, on transistors. 4 PASS in ~2 min (**36 s/µs** — the measured number that justifies the behavioral twins as the design loop). Honest hardware: two-inverter level shift with documented contention; precharge gate parked at 3.3 V (LDO-file overstress precedent, off in mission mode).

### `pll_twin_vs_spice.py`

Twin vs transistors overlay (one axis, same experiment): Runs the SAME near-lock release on both loops — SPICE verbatim, behavioral via `build_pll(v_init=...)` (the precharge switch, behaviorally; trace time-shifted to align releases) — and overlays VCTRL + instantaneous frequency. Landings **2.6 mV apart**; excursions mirror-imaged (release phase is arbitrary in both, sign/depth should NOT match), recovery timescale identical, and the twin reproduces the **per-cycle CP sawtooth** on VCTRL. Teaches the type-II **settling doublet** (125 kHz zero → τ ≈ 1.3 µs): at 3 µs both tails are same-instant snapshots creeping toward 600 from opposite sides, not final values. Saves `pll_twin_vs_spice.png`.

### `two_stage_miller_ota_example.py`

Two-stage Miller OTA (gain / phase margin / slew): Diff-pair + common-source with Cc/Rz Miller compensation. Demonstrates the **open-loop-gain measurement bias** — a huge `Inductor` (DC short / AC open) from VOUT→VINM plus a huge `Capacitor` AC-grounding VINM (`tb.add_passive`) — so the amp is closed-loop for biasing but open-loop for the AC measurement. Uses `phase_margin` + `unity_gain_freq` (`silk.metrics.ac`) and `slew_rate` (`silk.metrics.timing`).

### `folded_cascode.py`

Folded-cascode OTA (Variable-linked matched sizing): `with module:` auto-registration building a folded-cascode amplifier whose matched-pair and mirror-ratio invariants are enforced by `Variable` arithmetic — one leaf value propagates to every transistor referencing it (e.g. `m_bias_nmirr = 2 * m_bias_ndiode`). Reads internal nodes post-sim via the `SubcircuitInstance` handle returned by `add_module()` (`fcamp_inst.vbiasn`), declares a settling-time spec with `declare_spec`/`Measurement`, and plots the bias/cascode/diff-pair nodes as grouped panels.

### `stdcells.py`

Standard-cell library (shared, characterized, behavioral twins): The consolidated cell library the digital examples reuse — nine cell families, including three XOR implementations: `build_inverter` (2T), `build_buf` (4T), `build_nand2` (4T), `build_nand3` (6T), `build_nor2` (4T, PMOS-series dual), `build_nor3` (6T), `build_mux2` (6T transmission-gate 2:1, the serializer's load/shift select), `build_xor2`, `build_dff` (6-NAND flop, 32T). One shared sizing table (`_WN`/`_WP`/`_L` + a stack-widening rule). Every combinational cell has an embedded **truth-table** *and* **propagation-delay** testbench (tp into a 15 fF load: inv 50 / nand2 49 / nor2 64 / xor 75-99 ps); the flop carries **clk-to-Q** (83/130 ps) + a **setup-time sweep** (`sweep_setup_time`: clk-Q flat ~82 ps, stretches to 188 ps at coincidence, then capture fails — setup ≈100 ps by the 10% definition). **`build_xor2` ships three variants** on one rig showing the area/robustness trade: `nand` (16T, full rail), **`static` (12T, full rail — the DEFAULT)**, `ptl` (4T pass-transistor — measured degraded rails VOH 1682 / VOL 198 mV, and *slowest* under load, drive-weak; caveats + "when NOT to use" documented; `variant=` raises `ValueError` on unknown). **Every cell also carries a behavioral twin** (`set_behavioral_model`: boolean fn + a delay *calibrated from this file's own measured tp/clk-Q*); each asserts behavioral-truth == SPICE-truth (0 mismatches). These whole-array twins are characterized for **feed-forward** use. Per-gate vector calls complete over the whole array rather than alternating at each clock edge. Ports-form feedback is admitted in placement order, but wiring these vector models into a loop does not provide explicit point delays or establish causal sequential behavior. Honest limits: no drive strengths yet, single corner. Ring/PFD builders deliberately left untouched. `__main__` prints a cell catalog and saves `silk-output/stdcells.png` (setup sweep + capture + 3-variant XOR levels + NOR2 truth). 29 PASS.

### `prbs_generator_example.py`

PRBS generator family (generalized LFSR; verify scales with the period): `build_prbs(order=7, form="fibonacci")` — a **generalized** LFSR over the ITU-T O.150 two-tap table (orders 7/9/15/23/31), two feedback **forms** (`fibonacci`, `galois`), built from `stdcells` DFF/XOR (default the 12T static XOR) — `build_prbs7` is a thin alias; `prbs_reference(n, order=, form=)` the generalized numpy ground truth. THE reset trap holds at every order: `CLRB` clears to **all-zeros** (the XOR-LFSR's dead lock-up), so **XNOR feedback** makes all-zeros a valid seed — fingerprint `2^(n-1)-1` ones per period (63/255/16383) and a unique run of **zeros** length n. **Galois** = shortest-timing-arc "one-to-many" form; it's a *decimation* (not a simple shift) of Fibonacci, so it's checked against its **own** reference — honestly documented. **The teaching point: the verification MODE scales with the period.** orders **7, 9** — SPICE bit-for-bit at 600 MHz (0 mismatches + exact period 127/511 + balance + run; order 9 = 1040 bits ≈ 1.7 µs in ~56 s); order **15** — period 32767 ≈ 55 µs is un-SPICE-able, so a SPICE **spot-check** (first ~300 bits) anchors a **behavioral twin** (one monolithic LFSR model, delay calibrated from stdcells' clk-Q) run to the full period (32767 / 16383 ones / run 15); orders **23, 31** — tap-table + reference only (14 ms / 3.6 s of chip time, no sim). The twin keeps the verified recurrence in one model. The existing composed toggle-flop probe (`demonstrate_composed_rejection`) now returns `None` with an undelayed-feedback warning: the vector models run once in placement order, so admission does not establish causal toggle dynamics. Composed sequential models need appropriate state and explicit `previous_value` delays. `prbs_reference` exported for the SerDes serializer stand-in. Per-order `verify` test PASSes (4 specs for 7/9, spot for 15).

### `serdes_tx_driver_example.py`

SerDes TX driver + 1-tap FFE (pre-emphasis; eye open vs closed): **Differential** voltage-mode **segmented** CMOS driver — each slice (`build_inverter` reused, sized up) replicated onto a P and an N leg driven from complementary data, so PADP/PADN swing opposite and the eye is read at `V(RXP)-V(RXN)`; the tap slice driven by the **one-UI-delayed inverted bit** (the `-c1·b[n-1]` post-cursor). **Serializer is out of scope**, so the PRBS7 bitstream *and* the delayed/inverted tap sequence are computed in numpy and emitted as `PWL` sources (main/tap + complements) — the honest scoped stand-in. Lumped-RC channel per leg via ideal `passives` (a link is off-chip interconnect, not a PDK device; sized for a ~190 MHz pole, τ≈1.66 UI at 2 Gb/s; *unterminated* here — the full TX adds SST + 100 Ω term). Same PRBS, same hardware, tap off vs on: the differential eye at RX goes from **−310 mV (CLOSED)** to **+983 mV open**, asserted as a first-class **cross-test improvement spec** (`declare_spec` + a hand-recorded `Measurement`, since it spans two testbenches). Teaches folding a transient into an eye in numpy — classify by the *known* TX bit (no blind slicer), scan the sampling delay (the optimum lands ~1.1 UI out because the channel latency is ~1 UI) — and the **min/max-vs-rail trap** in measuring rise/fall on a busy PRBS: a dedicated clean-edge test references thresholds to the rail (diff swing 3.59 V ≈ 2× a leg, rise/fall 138/18 ps per leg at PADP, ignoring gate-coupling overshoot). 5 PASS. Saves `silk-output/serdes_tx_driver_eye.png`.

### `cml_cells.py`

Characterizes current-mode cells and a multi-tap FFE driver. `__main__` characterizes every cell and saves `silk-output/cml_cells.png`.

### `cml_ffe_eye.py`

CML TX eye — SPICE vs calibrated twin: The honest twin-vs-SPICE eye for `cml_cells.build_cml_ffe_driver`'s multi-tap current-steered FFE: drives the transistor driver over a PRBS in SPICE, **measures** its realized taps by least-squares, calibrates the behavioral twin to those measured taps (not an idealized FIR), and pushes both launches through the identical numpy 2-pole+reflection channel. Compares SPICE with a measured-tap twin and reports the remaining eye-height difference. Saves `silk-output/cml_twin_vs_spice_eye.png`.

### `serdes_tx_example.py`

Composes a serializer, FFE driver and channel, then compares CMOS/CML transmission and recovered bits. Saves `silk-output/serdes_tx_eye.png`, `silk-output/serdes_tx_ideal_eye.png` and `composition_serdes_tx_{cmos,cml}.png`.

### `cmos_ffe_eye.py`

CMOS TX eye — SPICE vs calibrated twin: The CMOS analog of `cml_ffe_eye.py`, closing the unit-weighted SST driver fix: `serdes_tx_example.build_cmos_ffe_driver`'s taps are measured from a SPICE launch, the behavioral twin is calibrated to those measured taps, and both eyes (through the identical channel) are compared to report the remaining eye-height difference. Emits both `silk-output/cmos_twin_vs_spice_eye.png` and a self-contained HTML viewer (`_emit_html`, base64-embedded PNG).

### `serializer_example.py`

Builds a 10:1 serializer with CMOS cells and with current-mode logic (CML) cells. Checks both outputs against the intended bit sequence. Compares sampled passing rates and power, then probes clock, data and CML-to-CMOS readout paths to explain differences; the finite sweep does not establish a continuous rate limit.

### `sar_adc_example.py`

SAR ADC behavioral twin (N-bit differential charge-redistribution): The sampled-analog counterpart to the SerDes arc, parameterized by resolution `n`. Pure-numpy **behavioral twin** (the long characterization run — the PLL/SerDes split) modelling the two things that actually limit a SAR: **cap mismatch** — the binary array is built from `2**n` **unit** caps each ~N(1, 0.5%), so bit k (=`2**k` units) has weight sigma `0.5%/√(2**k)` **by construction**, and the DNL spikes fall at the **major-carry codes** (32/64/96/128… at 8-bit) with **no hand-tuning** — the tell the physics is real; and **comparator noise+offset** — fresh per-decision Gaussian noise (sets the dynamic floor) + a fixed input-referred offset (a harmless DC shift for SNDR). `sar_codes` does an honest MSB→LSB binary search over the **actual mismatched** DAC, vectorized over all inputs. Static DNL/INL from a dense DC ramp (noise off = bench practice); dynamic SNDR/ENOB/SFDR from a **coherently-sampled sine + FFT** (prime bin, no window). **8-bit: ENOB 7.83, SNDR 48.9 dB, DNL/INL 0.08 LSB, no missing codes, SFDR 70.5 dB.** `characterize(n)`, `save_plots`, `emit_html`.

### `sar_adc_spice.py`

SAR ADC — real transistors + self-clocking FSM (cap-DAC + StrongARM + stdcell controller): The nominal 8-bit transistor SAR uses **28 µm high-reference NMOS switches**, versus 2 µm low-reference/sample/reset switches, all L=0.15 µm. Lower VGS (~0.9 V versus 1.8 V) and body effect at vrefp require more width: near-rail low/high Ron changes from **317/5087 Ω to 317/328 Ω**. At +810 mV input/trial 243, differential residue change across 13.8→14.0 ns falls from −27.86 to +0.258 mV. The five-code `measure_trip_points` fit is **1.004083675·v_dac + 3.515625 mV**, peak residual 0.03735 mV; after dividing out the dummy factor 256/255, gain is **1.000161473 within ±0.000343130 resolution**. Final trip midpoint uncertainty is ±0.234375 mV; small residuals do not imply finer accuracy. `LOOP_DAC_GAIN`/`LOOP_TRIP_OFFSET` now retain only the schematic dummy terms **256/255 and 0.9/255 V**. At inputs [−810,−540,−270,0,270,540,810] mV, StrongARM codes are **[12,51,89,127,165,203,242]**, versus the former [8,49,88,127,166,205,246]. Ideal twin codes [12,51,89,128,166,204,243] give differences **[0,0,0,−1,−1,−1,−1]**; calibrated twin codes [12,51,89,127,165,204,242] give differences **[0,0,0,0,0,−1,0]**. These are nominal seven-input results, not a corner/mismatch guarantee. Shared nominal benches: DAC-only peak/RMS error **0.414/0.236→0.463/0.250 LSB** (slightly worse); the 300 mV continuous replay's maximum pre-strobe residue error **74.03→2.09 mV**, predicted code 170 unchanged. The 21-point self-clocked transfer (−810 to +810 mV, 81 mV steps) has maximum ideal-twin difference **3→1 code**, measured 300 mV code 170 unchanged. The 16-sample/3-period sine test changes **SNDR 40.68→46.65 dB, ENOB 6.46→7.46**. The [control-waveform measurements](#sar-spice-accuracy-after-the-control-waveform-fix) are the historical 2 µm baseline. Sampling at `VIN_CM=VREFP/2` keeps the comparator near mid-supply. `conv` replays ideal-twin trials in one transient; its displayed code is a twin prediction. `build_sar_fsm(n)` supplies the stdcell sequencer for the separate `self`/`transfer`/`sine` benches. CLI: `trip`/`loop`/`self`/`transfer`/`sine`/`conv`/`diagram`.

### `sar_calibrate.py`

SAR twin, calibrated (what is — and isn't — a calibration target): Measures DAC-only gain and nonlinearity with `dac_transfer_spice`, then compares the real loop with the twin. Its existing DAC-only calibration does **not** cover the loop's historical StrongARM decision gain documented [below](#sar-spice-accuracy-after-the-control-waveform-fix). With the resized switches, `sar_adc_spice.LOOP_DAC_GAIN` is now the schematic dummy-capacitor factor **256/255**; the transistor-loop measurement verifies no further gain within its stated resolution. A nominal common-mode setup error should not be hidden in `COMP_OFFSET`; comparator mismatch requires its own characterization. Measuring the sampling gain in SPICE and incorporating it into the twin is separate follow-up work.

### `sar_resolution_sweep.py`

SAR ADC resolution sweep (where extra bits stop buying accuracy): Sweeps the SAR's resolution to find the **ENOB ceiling** and *attribute* it. Behavioral `n=4..13`: ENOB measured **four ways** — ideal (validates the measurement, recovers ENOB≈n), **cap-mismatch-only**, **comparator-noise-only**, and both — so the two walls are **separated**. Result: the **1 mV comparator-noise wall is hit first** (~9.5-bit ENOB ceiling, saturates near n=9); the 0.5%/unit cap-mismatch wall is milder (near-ideal to ~13 b), so "both" ≈ noise-only (the DNL/INL panel climbs monotonically toward the missing-code threshold where matching finally bites). SPICE confirms the trend in silicon at n=4,6,8 — the real self-clocked SAR (`convert_sar_sine`, on-chip FSM, one transient) digitizes a sine. **The honesty move** (the load-bearing part): the SPICE ENOB is **short-FFT-floor-limited** (32 conversions vs the twin's 4096), so it is compared against the twin run through the **identical short record** (`_matched_record_enob`, nominal caps, no noise) — the fair reference. n=4/6 match the ideal quantizer (+0.2 b), and the **+1.46 b residual at n=8 is REAL** — the StrongARM resolving a 7 mV LSB in a fixed 14 ns settle = the **silicon's own resolution wall**, drawn on the plot with a connector from the matched-twin ✕ to the SPICE ★ so it reads as a circuit limit, not saturation or model error. Gotcha handled: the reused `static_dnl_inl` reports a spurious lone missing code at the **terminal** transition (ramp ends exactly at the top-code level) → endpoint excluded. Saves `silk-output/sar_resolution_sweep.{png,html}` + `sar_resolution_sines.png`. CLI: (none)=behavioral fast, `full`=+SPICE n=4,6,8, `quick`=+SPICE n=4,6.

### `output_modes_example.py`

Simulation output modes + saved-file readback: Ideal load with an embedded voltage test: explicit overwrite (1 → 2 V), owner switched to `new` after test construction (retained 3 / 4 V), then back to the original overwrite path (5 V). Captures actual successful paths immediately and rereads disk after later runs; explains live reports, label grouping and overwrite-on-attempt. Needs ngspice and Python dependencies, **no installed PDK** for this ideal-only deck. Run: `MPLBACKEND=Agg .venv/bin/python -m silk.examples.output_modes_example`; defaults to `silk-output/output_modes/` under the working directory (or `$SILK_EXAMPLES_OUT`), accepts `run_example(output_root=...)`.

### `testbench_codegen_example.py`

Generates Python testbench code from a `TestbenchSpec` dictionary containing a circuit builder, sources, an AC analysis, a gain metric and a spec. Demonstrates input validation: the builder must be a bare Python identifier and the metric type must be known. Generating the code needs no PDK; running the generated circuit does.

### `amplifier_variants_example.py`

One parameterised block, several sizings (`variants=True`): A common-source amplifier built by `build_cs_amp(pdk, *, w_in, l_in, w_load, l_load, c_load)` and placed three times in one top (two settings, one repeat). Shows the `.subckt <module_name>_<8 hex>` suffix per distinct body, the shared definition for the repeated setting, the suffix on a single placement, and the `CircuitError` when the same placements are built without `variants=True`. Netlist only, no simulator.

### `circuit_tracking_example.py`

Database / tracking: The unscoped `db_path=` workflow end to end on the current database schema ([database.md](database.md) names the version); requires sky130 but no ngspice: stable spec keys, a sealed structure/spec identity, metrics and first-class spec measurements with explicit operating context, close/reopen queries through `Database`, an equivalent Python rebuild resolved to the same UUID, spec restoration, promotion with inherited evidence origin, and the enforced post-promotion write lock. Its JSON export is explicitly demonstrated as a lossy projection, not a database backup; the same projection is then imported into a second database, and the example lists the latest sealed snapshot per design stage with `Database.latest_at_stage`. For the profile-scoped path (multiple selected current sets per corner, PVT selection, migration), see `net_current_persistence_example.py` below. Guide: [database.md](database.md).

### `circuit_report_example.py`

Reports: The report flow end to end; requires the sky130 PDK but no ngspice: `generate_report` one-call harvest → `print_summary` → `save_html`/`save_json` → the **partial** `load_json` round-trip shown honestly (lint and coverage reset to `None`; `metadata`, `spec_reconciliation` and `layout_estimate` are restored) → `diff_reports` across two iterations, with a structural change the diff can actually see (a bare parameter change is invisible to it without simulation data). Guide: [reports.md](reports.md).

### `net_current_persistence_example.py`

Reuse simulated currents, metrics, and spec measurements without re-simulating (profile-scoped): Two `ModuleTestbench` characterizations of one DUT — different declared parameters and PVT, same process corner — each storing all three evidence families (`store_report` + `store_net_currents`) under its own characterization profile. Closes the writer, runs a provenance query through a bare `Database`, rebuilds the same Python definition (never simulated), and restores one profile's evidence with `restore_characterization_evidence()` to feed `estimate_layout()`/`RoutingModel` for wire sizing. Shows a current set being superseded (the old `current_set_id` stays retained and readable) and a loud `EvidenceMismatchError` when the requested evidence does not exist for an edited circuit. A final section keeps the original *unscoped* path alive: a plain `Module` calling `store_net_currents()`/`load_net_currents()` directly, no test identity at all. Guide: [database.md](database.md).

### `layout_intelligence_example.py`

Layout estimation: `estimate_layout`; statistical area and wire-length estimates. The base walkthrough uses the packaged sky130 catalog without simulation models. Section 10 demonstrates added RC loading: `inject_parasitics` + a before/after AC sim on the 5T OTA lowers bandwidth (ngspice/model-gated). Original terminals remain connected; this does not verify branch-aware routing or extracted post-layout accuracy.

### SAR SPICE accuracy after the control-waveform fix

Run `python -m silk.examples.sar_adc_spice` for the DAC-only test, or append
`trip` to measure comparator trip points. Use `self`, `transfer` or `sine` for
the self-clocked converter. DAC-only and predetermined-trial results do not
establish self-clocked conversion accuracy; repeat measurements after changing
the switch sizes, unit capacitance or comparator.

## Utility files

`diff_pair_construction_example.py` demonstrates factory construction and structure inspection with a
1.8 V supply, undriven NMOS gates and a PMOS mirror load. It uses the packaged
sky130 catalog without running a simulation. For a broader factory example,
read `factory_api_example.py`.

`compare_outputs.py` requires **PDK + ngspice**. It is a regression check (its own
docstring says so) that runs two independently constructed copies of the same circuit and
diffs the results — not a pattern demonstration. It may exit zero when both
runs fail, so that exit status alone does not prove its comparison ran.

## Going deeper

- [optimizer.md](optimizer.md) — the full `silk.optimize` design-space guide.
- [reports.md](reports.md) — everything `generate_report`/`save_html`/`diff_reports` can do.
- [simulator-save-list.md](simulator-save-list.md) — restricting what a simulator writes, for large rawfiles.
- [quickstart.md](quickstart.md) — the step-by-step walkthrough, if you'd rather read prose than run scripts.
- [agent-guide.md](agent-guide.md) §7 — the examples, indexed with full API-level detail (which classes, which methods, which gotchas each one exercises).
