# circuitsilk

**Custom circuit design in Python, from specs to layout.**

circuitsilk (`import silk`) is a framework for custom IC design. A circuit is one live Python object, and that object carries the design through the whole flow: specs, sizing, simulation, verification, reports, design history and layout. Every step is code with typed, checked results, so intelligence — yours, a script's or an AI agent's — can span the entire design flow, not just the netlist.

- **Testbenches that live with the circuit.** Each block carries its own tests, metrics and specs; `run_all_tests()` measures everything and reports pass or fail.
- **Two engines, one workflow.** Iterate in seconds with the built-in behavioral engine (no simulator, no PDK), then run the same testbenches at transistor level in ngspice. Xyce is a preview backend.
- **Corners, Monte Carlo and design-space search.** PVT corners, Monte Carlo (including behavioral mismatch), sweeps, sampling, and an optimizer that reads testbench metrics directly.
- **Browser waveform viewer.** Explore signals, cursors and derived expressions in the [interactive viewer](https://github.com/levantlabs/silk/blob/main/docs/waveform-explorer.md).
- **A full metric library.** Gain, bandwidth and phase margin; SNDR, ENOB, SFDR and INL/DNL for data converters; delay, slew, settling, jitter, noise, distortion and power.
- **Results you can trust.** silk doesn't take a simulator's exit code on faith. A run that produces no data, or stops before the end of the analysis silk wrote, raises a typed error instead of returning plausible numbers.
- **Checks beyond simulation.** Analog lint rules (interfaces, voltage domains, junction overrange) and fault-injection campaigns.
- **Layout (in progress).** Estimated parasitics can be fed into simulation and optimization, and a design's simulated currents and wiring are available for layout.
- **Design history.** A local SQLite database records versions, metrics and spec results at each PVT point, so a design can be reopened, rebuilt and compared.
- **Built for AI agents.** Keyword-only APIs that fail loudly, a machine-readable schema export, and an agent guide that walks the whole flow.

The whole loop, from system intent to sizing, characterization, reports, database, rebuild and layout feedback, runs in one worked example: `python -m silk.examples.design_loop_walkthrough`.

**PDKs.** The PDK layer is pluggable: a new PDK is two data files, not code. SkyWater 130nm is the only release-validated transistor-simulation PDK; GF180MCU and SG13G2 ship as preview integrations. Install open-source PDKs with [pdksetup](https://github.com/levantlabs/pdksetup) (`pdk install sky130`).

**No PDK yet?** Create and activate a virtual environment, install silk, then
run a behavioral simulation with no PDK and no ngspice:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install "circuitsilk>=0.2"
python -m silk.examples.behavioral_example
```

If `venv` or `ensurepip` is missing, install `python3-venv`
(`sudo apt install python3-venv`) or ask your administrator, then retry.
Continue with the [60-second start](https://github.com/levantlabs/silk/blob/main/docs/quickstart.md#1-60-second-start-no-pdk-no-ngspice).

## A first transistor-level circuit: a CMOS inverter

This example builds a CMOS inverter on sky130, checks its delay against a spec in ngspice, and plots the waveforms. It needs the [Installation](#installation) steps first (ngspice and the sky130 PDK); the behavioral example above needs neither.

```python
from silk import Module, Port, load_pdk, sources
from silk.elements import elements
from silk.metrics import propagation_delay
from silk.simulation import Transient

pdk = load_pdk('sky130')          # finds the models through the PDK_ROOT setting

# The circuit: a CMOS inverter, as a block with typed ports
inv = Module('PDK_skywater130', 'out', 'inverter.cir', pdk_details=pdk,
             module_name='inverter')
inv.define_module_ports({'VDD': Port.supply(), 'VSS': Port.ground(),
                         'IN': Port.analog_input(), 'OUT': Port.analog_output()})
with inv:                         # devices created here join the inverter
    pdk.nmos1p8('mn', d='OUT', g='IN', s='VSS', b='VSS', w=1.0, l=0.15)
    pdk.pmos1p8('mp', d='OUT', g='IN', s='VDD', b='VDD', w=2.0, l=0.15)

# Its testbench: drive it, load it, probe it, and check it against a spec
tb = inv.create_test('step')
tb.add_module_under_test({'VDD': 'vdd', 'VSS': '0', 'IN': 'vin', 'OUT': 'vout'})
tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
tb.add_source(sources.VoltageSource('vin', positive='vin', negative='0',
              pulse=sources.Pulse(v1=0, v2=1.8, td=1e-9, tr=0.5e-9,
                                  tf=0.5e-9, pw=4e-9, per=10e-9)))
tb.add(elements.capacitor('cl', p='vout', n='0', value=10e-15))
tb.add_sim_type(Transient(step=0.01e-9, stop=10e-9))
tb.set_supply_voltage(1.8, source='vdd')

vin, vout = tb.probe('vin'), tb.probe('vout')
tb.add_metric('delay', propagation_delay, sig_in=vin, sig_out=vout,
              out_edge='falling')
tb.add_spec('delay < 150 ps', 'delay', '<', 150e-12)

report = tb.run()                 # ngspice, typical corner
print(f"delay {report.metrics['delay'] * 1e12:.0f} ps: {'PASS' if report.passed else 'FAIL'}")
print(f'vout swings {vout.y.min():.3f} V to {vout.y.max():.3f} V')
tb.plot(vin, vout)
```

For interactive exploration after the run, open the browser viewer:

```python
tb.netlist.view()
```

For a headless run, use `html_path = tb.netlist.view(open=False)` to return the
HTML path without opening a browser. By default, `view.html` is written beside
the rawfile and may have companion files. See the
[viewer guide](https://github.com/levantlabs/silk/blob/main/docs/waveform-explorer.md).
Keep `tb.plot(vin, vout)` for a static plot.

Expected output:

```text
delay 91 ps: PASS
vout swings -0.004 V to 1.816 V
```

Continue with [docs/quickstart.md](https://github.com/levantlabs/silk/blob/main/docs/quickstart.md) for the step-by-step workflow.

## Installation

```bash
# Debian/Ubuntu system pythons are externally managed (PEP 668) — work in a venv:
python3 -m venv .venv
source .venv/bin/activate

pip install "circuitsilk>=0.2"
# — or from source —
git clone https://github.com/levantlabs/silk.git
cd silk
pip install -e ".[dev]"
```

**System requirements:**

- Python 3.10–3.13
- ngspice 42 or newer (for SPICE simulation): check `ngspice --version`.
  If missing or older, install or update it with `sudo apt install ngspice`
  on Debian/Ubuntu, or ask your administrator; then check again.
  It is the release-validated simulator; Xyce is an optional **preview** backend
  ([what is and is not validated](https://github.com/levantlabs/silk/blob/main/docs/known_issues.md#simulator-backends))
- SkyWater 130nm PDK models at `$PDK_ROOT` (for simulations using sky130 devices) —
  see [docs/installing-the-pdk.md](https://github.com/levantlabs/silk/blob/main/docs/installing-the-pdk.md)

The `PDK_ROOT` setting should point to a directory that contains
`sky130A/libs.tech/ngspice/sky130.lib.spice` (the standard open_pdks layout).
[docs/installing-the-pdk.md](https://github.com/levantlabs/silk/blob/main/docs/installing-the-pdk.md) walks through getting
there with [pdksetup](https://github.com/levantlabs/pdksetup)
(`pdk install sky130`), plus Docker and open_pdks alternatives.

Tool and PDK locations (`PDK_ROOT`, `SILK_NGSPICE_BIN`, `SILK_XYCE_BIN`, …) are
silk **settings**: set them as environment variables, or once in a settings
file at `~/.config/silk/silk.env` so every shell and IDE picks them up. An
environment variable overrides the file. See
[docs/settings.md](https://github.com/levantlabs/silk/blob/main/docs/settings.md)
for the format and a copy-paste example.

`load_pdk('sky130')` loads the packaged catalog even without the model files.
SPICE circuits made only from ideal elements, sources and passives need
ngspice but no installed PDK; see the
[PDK-free RC walkthrough](https://github.com/levantlabs/silk/blob/main/docs/quickstart.md#the-middle-rung-real-spice-with-no-pdk).
Model files are checked when a simulation needs them; a missing library raises
`SimulationError` with installation advice.

Behavioral simulation needs neither ngspice nor PDK models — see the
[60-second start](https://github.com/levantlabs/silk/blob/main/docs/quickstart.md#1-60-second-start-no-pdk-no-ngspice).

To check all of the above at once — interpreter, dependencies, your silk
settings, ngspice, and the model library of every registered PDK — run the
environment doctor (exit 0 when
the setup is ready, and it prints a fix for anything that isn't):

```bash
python -m silk.doctor            # add --json for machine-readable output
```

## Quick start

See [docs/quickstart.md](https://github.com/levantlabs/silk/blob/main/docs/quickstart.md) for a step-by-step walkthrough.

Find answers by question in the [documentation index](https://github.com/levantlabs/silk/blob/main/docs/README.md), or browse these guides:

- [docs/examples.md](https://github.com/levantlabs/silk/blob/main/docs/examples.md) — human-facing guide to the worked
  examples: a learning path plus a by-topic index
- [docs/cli.md](https://github.com/levantlabs/silk/blob/main/docs/cli.md) — every `python -m` tool (doctor, PDK validator,
  schema exports, waveform explorer) and the `$SILK_NGSPICE_BIN` /
  `$SILK_XYCE_BIN` simulator overrides
- [docs/database.md](https://github.com/levantlabs/silk/blob/main/docs/database.md) — design-history tracking,
  characterization profiles and PVT selection, close/reopen queries and
  restore, identity, promotion, migration, and JSON scope
- [docs/known_issues.md](https://github.com/levantlabs/silk/blob/main/docs/known_issues.md) — current limitations, honestly
- [docs/probes.md](https://github.com/levantlabs/silk/blob/main/docs/probes.md) — declare signals before simulation,
  bind metrics without lambdas, then read or plot results across analyses and corners
- [CONTRIBUTING.md](https://github.com/levantlabs/silk/blob/main/CONTRIBUTING.md) — dev install (venv, editable, tests) and
  conventions

Working examples live in `silk/examples/` and ship in the package — run any of
them as `python -m silk.examples.<name>`, no checkout needed:

| File | What it shows |
|---|---|
| `silk/examples/diff_pair_construction_example.py` | Construction-only factory and structure inspection |
| `silk/examples/factory_api_example.py` | Full factory API: devices, passives, SPICE inspection |
| `silk/examples/parameter_sweep_example.py` | Design-space sweep with `VariableSpace`/`Variable` |
| `silk/examples/ldo_design.py` | Parameterized LDO with `Module` + database tracking |
| `silk/examples/source_examples.py` | All source types: Pulse, Sine, PWL, SFFM, AM, controlled |
| `silk/examples/pwl_example.py` | PWL waveform: incremental, chaining, tuples, flat list, file |
| `silk/examples/hierarchical_example.py` | Multi-level subcircuits: inverter → buffer → testbench |

## Core concepts

### `Module` — the circuit object

In silk, a circuit is code. A `Module` is a live Python object holding a block's devices, nets, ports, sources and analyses. Every step of the flow reads that same object: netlisting, simulation, measurement, reports, the design database and layout estimation.

Modules are built to be reused. The usual way to write one is a factory function whose design knobs (widths, lengths, bias currents, loads) are keyword arguments, so one function produces every sizing of the block. Each block carries:
- its **ports**, with direction and role (supply, ground, analog input, output);
- its **testbenches** (`mod.create_test(...)`), each with its own sources, probes, metrics and specs, so the block knows how to test itself (`mod.run_all_tests()`);
- optionally a **behavioral model** (`set_behavioral_model`), so the block runs in the fast Python engine before its transistors exist, or stands in for them inside a larger system;
- its **specs**, which compare designed intent with measured results.

Blocks compose with `add_module`, so a system is built from blocks that each bring their own tests and models. And because it is all Python, a design can be diffed, reviewed and versioned like any other code, and read and written by an AI agent as easily as by a person.

```python
from silk import Module
ckt = Module('PDK_skywater130', 'out_dir', 'my_circuit.cir', pdk_details=pdk)
ckt.add(device1, device2)          # add pre-constructed device objects
ckt.add_source(vsource)            # add a voltage or current source
ckt.add(capacitor)                 # ideal R, C, L are added with add() too
ckt.add_sim_type(Transient(...))   # set the simulation type
ckt.simulate(corner='tt')          # generate netlist + run ngspice + load results
                                   # optional: timeout=300 (seconds)
ckt.plot('vout', 'vin')            # quick multi-panel plot
sig = ckt.load_signal('v(out)')    # retrieve a waveform for custom analysis
```

### Factory API — device instantiation

Devices are instantiated directly from the PDK object. Pins and parameters are
keyword arguments — IDE autocomplete works on them.

```python
mn = pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=1.0, l=0.35, m=4)
mp = pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=2.0, l=0.35)

# Parameters can be updated by direct attribute assignment after construction:
mn.w = 2.0
mn.d = 'new_out'
print(mn.to_spice())               # inspect the SPICE line
```

### PDK loading

```python
# Option 1: load_pdk (uses the PDK_ROOT setting: env var, settings file, or ~/pdk)
from silk import load_pdk
pdk = load_pdk('sky130')

# Option 2: override the model-library path
pdk = load_pdk('sky130', library_location='/path/to/sky130.lib.spice')
```

### Simulation types

```python
from silk.simulation import Transient, AC, DC, Noise, Corner

ckt.add_sim_type(Transient(step=1e-9, stop=100e-9))
ckt.add_sim_type(AC('dec', 100, 1e3, 1e9))
ckt.add_sim_type(DC(source='vin', start=0, stop=1.8, step=0.01))

# Corner is a str-subclass enum — plain strings also work
ckt.simulate(corner=Corner.TT)
ckt.simulate(corner='ff')          # equivalent
```

### Sources

```python
from silk import sources

vs = sources.VoltageSource('vdd', positive='vdd', negative='gnd', dc=1.8)
cs = sources.CurrentSource('ibias', positive='vdd', negative='vnbias', dc=100e-6)

# Waveforms
vs_pulse = sources.VoltageSource('vin', positive='in', negative='gnd',
                                  pulse=sources.Pulse(v1=0, v2=1.8, td=0,
                                                      tr=100e-12, tf=100e-12,
                                                      pw=5e-9, per=10e-9))
vs_sine  = sources.VoltageSource('vin', positive='in', negative='gnd',
                                  sine=sources.Sine(voffset=0, vampl=1, freq=1e6))

# Piecewise-linear (PWL) waveform
pwl = sources.PWL().add(0, 0).add(1e-9, 0).add(2e-9, 1.8).add(10e-9, 1.8)
vs_pwl = sources.VoltageSource('vin', positive='in', negative='gnd', pwl=pwl)
# Also accepts: pwl=[(0, 0), (1e-9, 1.8)] or pwl=[0, 0, 1e-9, 1.8]
```

### Ideal elements (ElementLibrary)

```python
from silk.elements import elements

r = elements.resistor('r1',  p='out', n='gnd', value=10e3)
c = elements.capacitor('c1', p='out', n='gnd', value=100e-12)
l = elements.inductor('l1',  p='a',   n='b',   value=1e-9)
ckt.add(r, c, l)
```

Construct voltage and current sources with `silk.sources`.

### Design-space exploration

```python
from silk import Module, load_pdk
from silk.optimize import VariableSpace

pdk = load_pdk('sky130')
ckt = Module('PDK_skywater130', 'out', 'sweep.cir', pdk_details=pdk)

# Create variables with bounds
w_n = pdk.nmos1p8.w.variable(name='w_n', value=1.0, bounds=(0.42, 20.0))
w_p = pdk.pmos1p8.w.variable(name='w_p', value=2.0, bounds=(0.42, 20.0))
l   = pdk.nmos1p8.l.variable(name='l',   value=0.15, bounds=(0.15, 1.0))

# Pass Variable objects directly as device parameters
with ckt:
    mn = pdk.nmos1p8('mn', d='out', g='in', s='gnd', b='gnd', w=w_n, l=l)
    mp = pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=w_p, l=l)

# Build a VariableSpace and sample it
space = VariableSpace([w_n, w_p, l])
points = space.sample(20, method='lhs')    # Latin hypercube sampling
# also: method='random', 'halton', 'grid'

for pt in points:
    space.apply(pt)                        # updates variable .value fields
    ckt.generate_netlist(corner='tt')      # picks up new values automatically
```

### Schema export APIs

```python
from silk import export_schema, module_schema
import json

# Export all PDK devices as a machine-readable dict
schema = export_schema(pdk)
print(json.dumps(schema, indent=2))

# Export a single Module's interface
mod_schema = module_schema(my_module)
```

### Database / version tracking

Pass `db_path` to `Module` to enable database tracking. Python remains the
authoritative circuit/spec definition; the SQLite file preserves sealed
snapshot identity, declarations, metric/spec evidence, current sets (multiple
selected sets per corner, distinguished by characterization profile and
PVT — see [docs/database.md](https://github.com/levantlabs/silk/blob/main/docs/database.md)), lineage, and provenance.
Rebuild the same Python definition to restore live state. Unsupported or
malformed files are rejected without modification; see the database guide's
[compatibility matrix and migration guidance](https://github.com/levantlabs/silk/blob/main/docs/database.md#file-compatibility).

```python
ckt = Module('PDK_skywater130', 'out', 'ldo.cir', pdk_details=pdk,
             module_name='ldo_nmos', db_path='designs.db')

ckt.add(mn1, mp1)
ckt.declare_spec('DC gain', 'gain_dB', '>=', 40.0,
                 units='dB', key='gain')
version_id = ckt.create_circuit_version('Initial attempt')
ckt.store_results(
    {'gain_dB': (42.5, 'dB')}, source='spice', sim_type='ac',
    corner='tt', temperature=27.0, voltage=1.8,
    tool_name='ngspice-44', conditions={'load_pf': 2.0},
)
ckt.close()

rebuilt = build_ldo(pdk, db_path='designs.db')
assert rebuilt.create_circuit_version() == version_id
rows = rebuilt.query_results(parameter='gain_dB', corner='tt')
rebuilt.promote('stable')           # evidence remains at its origin UUID
rebuilt.close()
```

JSON export is explicitly a lossy interoperability projection, not a database
backup or circuit archive. See the canonical [database guide](https://github.com/levantlabs/silk/blob/main/docs/database.md)
and the simulator-free `python -m silk.examples.circuit_tracking_example`.

### Code generation

Generate runnable silk Python from a JSON-able spec dict (a circuit description):

```python
from silk.codegen import Generator

gen  = Generator()                        # no constructor args
code = gen.generate({
    'circuit_name': 'diff_pair',
    'pdk': 'sky130',
    'voltage_domains': {'AVDD': {'nominal': 1.8, 'tolerance': 0.1}},
    'ports': {
        'vin_p': {'dir': 'input',  'role': 'analog',  'domain': 'AVDD'},
        'vout':  {'dir': 'output', 'role': 'analog',  'domain': 'AVDD'},
    },
    'devices': [...],
})
print(code)
```

Output mode depends on the payload: absent or `null` `ports` emits a standalone simulation script; `ports: {}` emits a zero-pin `Module` subcircuit; non-empty `ports` emits a reusable `Module` subcircuit with those pins and `define_module_ports`. See the [codegen contract](https://github.com/levantlabs/silk/blob/main/docs/codegen-contract.md). Optional `voltage_domains` and `bus_nets` blocks are emitted only when the spec uses them.

Full JSON shape and field reference: **[docs/codegen-contract.md](https://github.com/levantlabs/silk/blob/main/docs/codegen-contract.md)**.

## Full API reference

See [docs/api-reference.md](https://github.com/levantlabs/silk/blob/main/docs/api-reference.md).

## Package layout

```
silk/
  __init__.py         top-level re-exports
  exceptions.py       CircuitError hierarchy

  circuit/            Module, ModuleTestbench, schema export, port roles, composition graph
  pdk/                BasePDK, load_pdk, PDK manifests + validator
  elements/           ElementLibrary, passives, AnalogElement
  sources/            VoltageSource, CurrentSource, waveforms
  signals/            Signal, NgspiceSignal, rawfile reader, waveform metrics
  simulation/         Transient, AC, DC, Noise, Corner, simulator backends
  simulators/         per-simulator YAML configs (ngspice, xyce)
  specs/              Spec objects, reconciliation, calibration binding
  metrics/            measurement library (ADC, timing, jitter, distortion, noise, …)
  lint/               design-rule framework (stress, port/domain checks, reports)
  faults/             fault injection: FaultSet, FaultCampaign, reports
  layout/             layout estimation, parasitics injection, EM budget
  optimize/           VariableSpace, Variable, Optimizer
  constraints/        constraint vocabulary
  db/                 Database, CircuitTracker, CircuitImporter, CircuitExporter, DesignStage
  codegen/            Generator
  utils/              internal helpers
  examples/           runnable example scripts (see docs/agent-guide.md §7)
```

## Development

The test suite lives in the repository, not in the source package. Run these
commands from the root of a repository clone:

```bash
pip install -e ".[dev]"
pytest tests/ -q
```
