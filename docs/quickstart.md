# circuitsilk — Quick Start Guide

This guide walks through the most common workflows step by step.

## 1. 60-second start (no PDK, no ngspice)

Behavioral simulation uses Python/NumPy and the packaged catalogs; no simulator
or installed models are needed. See [installing-the-pdk.md](installing-the-pdk.md)
for the transistor-simulation support boundary.

```bash
python3 -m venv .venv && source .venv/bin/activate   # Debian/Ubuntu pip refuses outside a venv (PEP 668)
pip install "circuitsilk>=0.2"
python -m silk.examples.behavioral_example
```

Or from source:

```bash
git clone https://github.com/levantlabs/silk.git
cd silk
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The example wires a comparator into an RC low-pass filter and prints per-signal
stats:

```
Available signals: ['time', 'v(gnd)', 'v(vcomp)', 'v(vfilt)', 'v(vin)', 'v(vref)']
  vin     min=0.000 V  max=2.000 V  mean=1.041 V  (121 samples)
  vcomp   min=0.000 V  max=1.800 V  mean=0.937 V  (121 samples)
  vfilt   min=0.000 V  max=0.721 V  mean=0.419 V  (121 samples)
Behavioral simulation completed successfully.
```

Here is the same idea from scratch — a behavioral inverter driven by a pulse,
runnable as-is on a fresh install:

```python
import numpy as np
from silk import Module, Port, sources, load_pdk

pdk = load_pdk('sky130')

# Module args: PDK label, output directory (created if missing), netlist file name
inv = Module('PDK_skywater130', 'out', 'inv.cir', pdk_details=pdk, module_name='inv')
inv.define_module_ports({'vin': Port.analog_input(), 'vout': Port.analog_output()})
def model(ports, t, params):
    ports.vout.v = np.where(ports.vin.v > 0.9, 0.0, 1.8)

inv.set_behavioral_model(model, execution='vector')

top = Module('PDK_skywater130', 'out', 'top.cir', pdk_details=pdk, module_name='top')
top.add_source(sources.VoltageSource('vin_src', positive='vin', negative='0',
                                     pulse=sources.Pulse(v1=0, v2=1.8, td=5e-9, tr=1e-9,
                                                         tf=1e-9, pw=20e-9, per=40e-9)))
top.add_module(inv, 'x1', nets={'vin': 'vin', 'vout': 'vout'})
top.simulate_behavioral(t_step=1e-9, t_stop=80e-9)

vout = top.load_signal('vout')
print(f"vout: min={vout.y.min():.2f} V  max={vout.y.max():.2f} V  ({len(vout.y)} samples)")
# vout: min=0.00 V  max=1.80 V  (81 samples)
```

Section 4b covers the behavioral engine in full — stateful models, port
directions, subcircuit composition.

### The dependency ladder

| What you want to do | What you need |
|---|---|
| Behavioral simulation, schema export, design-space setup | `pip install "circuitsilk>=0.2"` — nothing else |
| **Real SPICE** on ideal elements, passives and sources (RC filters, ideal-source testbenches, `silk.metrics` on real waveforms) | **+ ngspice. No PDK.** |
| Transistor-level design with real device models | ngspice + a PDK on disk |

**+ ngspice:**

```bash
sudo apt install ngspice        # Debian / Ubuntu
brew install ngspice            # macOS
```

**+ sky130 PDK:** see [installing-the-pdk.md](installing-the-pdk.md) for the
recommended [pdksetup](https://github.com/levantlabs/pdksetup) route
(`pdk install sky130`), the Docker alternative, and the verification steps.
`python -m silk.doctor` checks the whole environment in one shot.

### The middle rung: real SPICE with no PDK

A testbench built only from ideal elements, passives and sources is valid SPICE
on its own — it names no foundry model, so it needs no model library. silk
emits it that way: the `.lib`/`.include` block is written **only** when the
deck needs PDK models, or contains custom content whose model requirements
cannot be determined. So a bare `pip install` plus `apt install ngspice` already runs a real simulation:

```python
from silk import Module, sources, load_pdk
from silk.elements import elements
from silk.simulation import Transient

# load_pdk works with nothing installed: the device catalog, units and corners
# all ship inside silk. Only the model library lives on disk, and this deck
# never asks for it.
pdk = load_pdk('sky130')

tb = Module('PDK_skywater130', 'out', 'rc.cir', pdk_details=pdk)
tb.add_source(sources.VoltageSource(
    'vin', positive='vin', negative='0', dc=0,
    pulse=sources.Pulse(v1=0, v2=1.0, td=1e-9, tr=1e-10, tf=1e-10,
                        pw=5e-9, per=10e-9)))
with tb:
    elements.resistor('r1', p='vin', n='vout', value=1e3)      # 1 ns RC
    elements.capacitor('c1', p='vout', n='0', value=1e-12)
tb.add_sim_type(Transient(1e-11, 20e-9))

tb.simulate()                                # real ngspice, no PDK files read
vout = tb.load_signal('vout')
print(f"vout: max={vout.y.max():.3f} V  ({len(vout.y)} samples)")
# vout: max=0.994 V  (2032 samples)
```

The emitted `rc.cir`, in a run directory under `out/`, says so in place of
the library block:

```
*** No PDK library included: this deck instantiates no PDK
*** device (ideal elements, passives and sources only), so it
*** runs with no PDK installed. One PDK device anywhere in
*** the hierarchy, or any add_custom_line() text, restores it.
```

PDK devices and custom netlist content retain model includes. Missing models
are reported when simulation starts.

## 2. Loading the PDK

Transistor-level work needs the PDK installed — see
[installing-the-pdk.md](installing-the-pdk.md) if `PDK_ROOT` is not set up
yet. `load_pdk` itself works either way (see section 1's middle rung).

`PDK_ROOT` is a silk setting: set it once in your settings file
(`~/.config/silk/silk.env`, one line `PDK_ROOT=/path/to/pdk`) or as an
environment variable, which overrides the file. [settings.md](settings.md)
explains the file; `python -m silk.doctor` shows the value silk resolved and
where it came from.

The recommended way is `load_pdk`, which resolves the device catalog, lib,
corners, and extra includes from the PDK manifest — no yaml or lib path in your
code:

```python
# PDK_ROOT comes from the environment or your settings file (settings.md);
# load_pdk('sky130', pdk_root='/path/to/pdk') overrides it for one call.
from silk import load_pdk
pdk = load_pdk('sky130')                   # silk.pdk.available() lists registered names
```

`load_pdk` loads the packaged catalog and resolves the library path without
requiring installed model files. A deck with PDK devices reports missing models
at `simulate()`; `python -m silk.doctor` checks their presence before a run.
Overrides: `load_pdk('sky130', pdk_root=..., library_location=...,
corner=..., user_units='si')`.

For a custom catalog or model-library layout, see the low-level PDK
constructors in the [API reference](api-reference.md). For shipped PDKs, prefer
`load_pdk()` so units, corners and extra includes come from the manifest.

## 3. Building a circuit

A `Module` holds devices, sources, passives, and simulation commands.
The following construction examples use `pdk` from section 2. They omit input
and tail-bias sources; complete the testbench before simulating it. W/L values
are in micrometres by default: `w=4.0` means 4 µm. For a runnable continuation,
use the complete RC testbench above or the amplifier factory in
`ota_5t_guide.py`; this differential-pair fragment needs its input and bias
sources first.

A `Module` creates its output directory when needed. Inside its with-block,
device, source and passive constructors register with the module automatically:

```python
from silk import Module, sources
from silk.simulation import Transient

ckt = Module('PDK_skywater130', 'out', 'diffpair.cir', pdk_details=pdk)
with ckt:
    pdk.nmos1p8('mn1', d='voutn', g='vinp', s='tail', b='0', w=4.0, l=0.15, m=2)
    pdk.nmos1p8('mn2', d='voutp', g='vinn', s='tail', b='0', w=4.0, l=0.15, m=2)
    pdk.pmos1p8('mp1', d='voutp', g='voutp', s='vdd', b='vdd', w=4.0, l=0.15)
    pdk.pmos1p8('mp2', d='voutn', g='voutp', s='vdd', b='vdd', w=4.0, l=0.15)
    sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)

ckt.add_sim_type(Transient(step=1e-9, stop=100e-9))
```

**Alternative — explicit `add()`:**

Outside a with-block, construct devices and add them explicitly to a fresh module:

```python
ckt = Module('PDK_skywater130', 'out', 'diffpair.cir', pdk_details=pdk)
mn1 = pdk.nmos1p8('mn1', d='voutn', g='vinp', s='tail', b='0', w=4.0, l=0.15, m=2)
mn2 = pdk.nmos1p8('mn2', d='voutp', g='vinn', s='tail', b='0', w=4.0, l=0.15, m=2)
mp1 = pdk.pmos1p8('mp1', d='voutp', g='voutp', s='vdd', b='vdd', w=4.0, l=0.15)
mp2 = pdk.pmos1p8('mp2', d='voutn', g='voutp', s='vdd', b='vdd', w=4.0, l=0.15)
ckt.add(mn1, mn2, mp1, mp2)
ckt.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
ckt.add_sim_type(Transient(step=1e-9, stop=100e-9))
```

Both styles are fully supported and can be mixed. See the
[Module API reference](api-reference.md) for details.

## 4. Simulating

`simulate()` runs the full pipeline — netlist generation, ngspice, result loading:

```python
ckt.simulate(corner='tt')          # bare string works
# or use the Corner enum for IDE completion:
from silk.simulation import Corner
ckt.simulate(corner=Corner.FF)
```

You can run the steps individually if needed:

```python
ckt.generate_netlist(corner='tt')        # writes the .cir file
ckt.run_simulator()                      # calls ngspice
ckt.load_file(ckt.results_file)          # use the actual path from this run
ckt.read_raw_file()                      # parses into signal objects
```

## 4b. Behavioral (Python) Simulation

Behavioral simulation runs entirely in Python/NumPy — no ngspice or PDK required.
It is useful for rapid architecture exploration, algorithm prototyping, and
unit-testing circuit blocks before committing to a real SPICE model.

### Defining ports and attaching a behavioral model

```python
from silk import Module, Port, load_pdk

pdk = load_pdk('sky130')  # packaged catalog; no installed models needed
sub = Module('PDK_skywater130', 'out', 'comp.cir',
             pdk_details=pdk, module_name='comparator')

# Declare port directions so the engine knows which nets are inputs/outputs
sub.define_module_ports({
    'vin':  Port.analog_input(),
    'vref': Port.analog_input(),
    'vout': Port.analog_output(),
})

# Attach a stateless function: fn(ports, t, params) publishes by assignment
import numpy as np

def comparator(ports, t, params):
    ports.vout.v = np.where(ports.vin.v > ports.vref.v, 1.8, 0.0)

sub.set_behavioral_model(comparator, execution='vector')
```

A stateful class works just as well — useful for models with memory (filters,
integrators, state machines):

```python
class RCFilter:
    def __init__(self, R=1e3, C=100e-12):
        self.R, self.C = R, C

    def __call__(self, ports, t, params):
        dt = t[1] - t[0] if len(t) > 1 else 1e-9
        alpha = dt / (self.R * self.C + dt)
        vin = np.asarray(ports.vin.v)
        v, vout = 0.0, np.zeros_like(vin)
        for i, vi in enumerate(vin):
            v += alpha * (vi - v)
            vout[i] = v
        ports.vout.v = vout

rc_sub = Module('PDK_skywater130', 'out', 'rc_filter.cir',
                pdk_details=pdk, module_name='rc_filter')
rc_sub.define_module_ports({'vin': Port.analog_input(), 'vout': Port.analog_output()})
rc_sub.set_behavioral_model(RCFilter(R=1e3, C=100e-12), execution='vector')
```

### Connecting subcircuits in a top-level circuit

```python
from silk import sources

top = Module('PDK_skywater130', 'out', 'behavioral.cir', pdk_details=pdk)
top.add_source(sources.VoltageSource(
    'vin_src', positive='vin', negative='0',
    pulse=sources.Pulse(v1=0, v2=2.0, td=5e-9, tr=1e-9, tf=1e-9,
                        pw=20e-9, per=40e-9)))
top.add_source(sources.VoltageSource(
    'vref', positive='vref', negative='0', dc=0.9))

top.add_module(sub,    'comp1', nets={'vin': 'vin', 'vref': 'vref', 'vout': 'vcomp'})
top.add_module(rc_sub, 'rc1',   nets={'vin': 'vcomp', 'vout': 'vfilt'})
```

### Running the simulation

```python
# Behavioral — Python only, no ngspice
top.simulate_behavioral(t_step=1e-9, t_stop=120e-9)
```

### Accessing results

`load_signal()` and `plot()` use the same results workflow for both engines.
The blocks above contain only Python models, so they cannot run as SPICE.
To use SPICE, give each block actual circuit contents, select a SPICE backend
(for example, `top.simulator = 'ngspice'`), and register an analysis on the
top-level circuit with `top.add_sim_type(Transient(step=1e-9, stop=120e-9))`
(import `Transient` from `silk.simulation`). Then call `top.simulate(corner='tt')`.
This needs the selected simulator and, for PDK devices, installed models.
The behavioral shortcut restores the previous analyses and backend; it does
not configure a later SPICE run. See the
[complete two-engine example](agent-guide.md#one-workflow-two-engines) for a
block that already has both implementations. Shared results do not imply
identical engine capabilities.

After the behavioral run above:

```python
names = top.get_signal_names()   # list available signal names
print(names)

vcomp = top.load_signal('vcomp')
vfilt = top.load_signal('vfilt')

print(vcomp.x)    # time vector (numpy)
print(vcomp.y)    # voltage vector (numpy)

top.plot('vcomp', 'vfilt')   # multi-panel plot
```

See `silk/examples/behavioral_example.py` for a complete runnable demonstration.

## 5. Reading and plotting results

These fragments assume a simulated circuit with the named signals and voltage
source. Replace the names with those from `ckt.get_signal_names()`.

```python
# Quick plot — one line after simulation
ckt.plot('vout', 'vin')                # multi-panel, shared x-axis
ckt.plot('vout', 'vin', overlay=True)  # all signals on one axis
ckt.plot('vout', 'vvsupply:I')         # mix voltage and current (source 'vsupply' → vector i(vvsupply))

# Manual signal access for custom analysis
ckt.get_signal_names()               # returns the list of available signal names
vout = ckt.load_signal('v(voutp)')  # returns a Signal
ibias = ckt.load_signal('vvsupply', signalType='I')   # branch current of the source named 'vsupply'

print(vout.name)   # signal name
print(vout.x)      # time / frequency array (numpy)
print(vout.y)      # value array (numpy)

vout.plot()        # single signal matplotlib plot
```

Branch currents come from voltage sources: a `VoltageSource('vsupply', ...)`
emits as `Vvsupply`, and ngspice saves its current as `i(vvsupply)`, so the
signal name is the source name with a `v` prefix, loaded with `signalType='I'`.
There is no current vector for a bare node — insert a 0 V source where you want
an ammeter. A testbench reads a branch current with `tb.probe('i(vs)')`
for a source named `vs`; see [current probes](probes.md).

Signals support arithmetic:

```python
vin = ckt.load_signal('vin')
voutn = ckt.load_signal('voutn')
gain = vout / vin              # element-wise division
diff = vout - voutn            # differential output
```

### Looking at *everything*, interactively

`ckt.plot(...)` above needs you to already know which signals matter. A
SPICE run saves every node (`save all`) — opt out with
`ckt.set_save_list([...])` when a rawfile is too big to store, see
[simulator-save-list.md](simulator-save-list.md). Browse saved signals with
`ckt.view()` after simulation:

```python
ckt.view()                                 # emits view.html beside the rawfile and opens it
ckt.view(open=False)                       # emit only (headless); returns the html path
```

Use `emit_view()` to choose the HTML output path:

```python
from silk.signals.view import emit_view
emit_view(ckt.results_file, 'view.html')   # open view.html in any browser
```

By default, the viewer writes the HTML plus sibling `<stem>.data.js` and
`<stem>.stamp.js` files. Keep all three files together when moving or sharing
the viewer; this mode supports refresh and scope switching. For one shareable
HTML file with its waveform data embedded, use:

```python
ckt.view(open=False, inline=True)
# Or choose the output path directly:
emit_view(ckt.results_file, 'share.html', inline=True)
```

The viewer needs no installation and has a searchable signal picker,
real zoom (data underneath it, not a static image), and cursor/viewport
measurements. On a large hierarchical design, `--split-by-block` on the CLI
(or `emit_views_by_block`) writes one page per block instead of a single
huge one, with a `Scope ▾` control to switch between them. See
[cli.md](cli.md#silksignalsview--waveform-explorer) for the full CLI,
signal-filtering flags, and `context=` for adding run provenance (module,
corner, PDK, spec results) to the page.

## 6. Sources and waveforms

```python
from silk import sources

# DC source
vs = sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8)

# Pulse waveform
vclk = sources.VoltageSource(
    'vclk', positive='clk', negative='0',
    pulse=sources.Pulse(v1=0, v2=1.8, td=0,
                        tr=100e-12, tf=100e-12,
                        pw=5e-9, per=10e-9))

# Sine waveform
vsin = sources.VoltageSource(
    'vin', positive='in', negative='0',
    sine=sources.Sine(voffset=0, vampl=1, freq=1e6))

# PWL waveform — build incrementally or from a list
pwl = sources.PWL().add(0, 0).add(10e-9, 1.0).add(20e-9, 0.0)
vpwl = sources.VoltageSource('vin', positive='in', negative='0', pwl=pwl)
# Also accepts: pwl=[(0, 0), (10e-9, 1.0)] or pwl=[0, 0, 10e-9, 1.0]

# Current source
cs = sources.CurrentSource('ibias', positive='vdd', negative='vnbias', dc=100e-6)

# Controlled sources
vcvs = sources.VoltageControlledVoltageSource(
    'e1', out_pos='vout', out_neg='0',
    ctrl_pos='vin', ctrl_neg='0', gain=10)
```

## 7. Passives and ideal elements

This fragment assumes a module `ckt` without instances named `r1`, `c1`,
`l1` or `vdd`.

```python
from silk.elements import elements

# Ideal, process-independent primitives via the ElementLibrary
# Use silk.sources for voltage and current sources.
r = elements.resistor('r1', p='out', n='0', value=10e3)
c = elements.capacitor('c1', p='out', n='0', value=100e-12)
l = elements.inductor('l1', p='a', n='b', value=1e-9)
ckt.add(r, c, l)
ckt.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
```

The old `silk.elements.passives.Resistor`/`Capacitor`/`Inductor` constructors
are deprecated; prefer `elements.resistor`/`capacitor`/`inductor` with
`Module.add()` above. `Module.add_passive()` remains supported.

## 8. Hierarchical subcircuits with Module

`Module` doubles as a subcircuit definition. This fragment assumes
`nmos_output` and `nmos_bias` are devices you built and `top` is the parent
module:

```python
from silk import Module, Port, PortDir

sub = Module('PDK_skywater130', 'out', 'ldo.cir', pdk_details=pdk)
sub.define_module_name('ldo_nmos')
sub.define_module_ports({'vdd': Port.supply(), 'vss': Port.ground(),
                         'vout': Port.analog_output()})

sub.add(nmos_output, nmos_bias)

# Instantiate the subcircuit inside a top-level netlist
top.add_module(sub, 'ldo_inst',
               nets={'vdd': 'vdd3', 'vss': '0', 'vout': 'vldo_out'})
```

The ground port is named `vss`, not `gnd`: ngspice folds a port *named* `gnd`
onto the global SPICE node `0` and silently discards whatever the parent wired
to that port position, while Xyce honours it — so silk rejects the
declaration (`STRUCT-012` at lint time, and a guard in `generate_netlist` that
raises `CircuitError` before any netlist is written). Use `vss` for a ground
port and connect it to `0` in the parent when global
ground is intended. A formal port named `0` is also rejected. Use `0` for
internal global-ground connections too: ngspice treats `gnd` as an alias,
while silk rejects internal `gnd` nets in Xyce decks.

## 8a. Characterize with a testbench

`Module.create_test()` returns a `ModuleTestbench` that wires up a DUT, drives
it with sources, runs a simulation, and evaluates metrics against specs — through
the steps below.

```python
from silk import sources
from silk.simulation import AC
from silk.metrics import dc_gain_dB

# Prerequisite: mod is an amplifier subcircuit with the six ports below.
# See ota_5t_guide.py for a complete amplifier and its embedded testbenches.
tb = mod.create_test('ac')
tb.add_module_under_test({'VDD': 'VDD', 'VSS': '0', 'VINP': 'vinp',
                          'VINM': 'vinm', 'VOUT': 'vout', 'IBIAS': 'ibias'})
# Drive EVERY input: supply, bias, and the AC stimulus. A testbench with only
# a supply "runs" but measures -inf dB gain.
tb.add_source(sources.VoltageSource('vdd',   positive='VDD',   negative='0', dc=1.8))
tb.add_source(sources.VoltageSource('vbias', positive='ibias', negative='0', dc=0.9))
tb.add_source(sources.VoltageSource('vinp',  positive='vinp',  negative='0', dc=0.9, ac=1.0))
tb.add_source(sources.VoltageSource('vinm',  positive='vinm',  negative='0', dc=0.9))
tb.add_sim_type(AC(variation='dec', points=20, fstart=1, fstop=1e9))
vout = tb.probe('vout')                 # names a signal; no data until the run
tb.add_metric('gain_dB', dc_gain_dB, sig=vout)
tb.add_spec('DC gain > 30 dB', 'gain_dB', '>', 30.0)
tb.set_temperature(27.0)
tb.set_supply_voltage(1.8, source='vdd')  # configure this rail, not the bias sources
report = tb.run()
vout.plot()                             # the probe reads the run that just finished
```

Declare probes, bind metrics, run, then plot (see the
[practical probe guide](probes.md) for result lifetime, current probes,
multi-analysis tests and retained corner waveforms). Arithmetic on a probe
defers, so `sig=(tb.probe('vp') + tb.probe('vm')) / 2` measures a derived
signal. The older callable form,
`tb.add_metric('gain_dB', lambda ckt: dc_gain_dB(ckt.load_signal('vout')))`,
also works when extraction needs additional logic, such as scaling the metric.

`tb.run()` defaults to the PDK's typical corner; pass `corner='ff'` for
another. Configure temperature and the intended supply with the setters above
before running; the report records the resulting operating conditions.
The legacy `run(temperature=..., voltage=...)` keywords are deprecated
assertions of independently configured conditions. They neither set temperature
or supply voltage nor attach arbitrary PVT labels; a mismatch raises.
Recorded conditions let the tracking database in §12 distinguish PVT points —
see [database.md](database.md#characterization-profiles). Testbench names are unique
per module: a second `mod.create_test('ac')` raises `DuplicateNameError`.

The full worked version of this pattern (with transient and noise testbenches
too) is `silk/examples/ota_5t_guide.py`.

`tb.run()` is simulate-then-evaluate, and the second step is public. If the
simulator runs somewhere else — a remote machine or CI — emit
the deck here, run it there, and evaluate the rawfile it produced:

```python
deck_path = tb.prepare(corner='tt')   # apply bench settings; return the deck path
# ... run it elsewhere with the simulator the deck was generated for ...
report = tb.evaluate('tt', rawfile='rawfile.raw')   # metrics + specs, no simulator
```

Make the referenced models and output directories available remotely. The
deck may write a prepared rawfile path; pass the actual returned rawfile to
`evaluate()`.

`evaluate()` returns a `SpecReport` and evaluates the metrics and specs on
that rawfile. For an external file, pass the corner you simulated; the
rawfile does not record it. Declare `temperature=` and `voltage=` too when
known. Omitted external conditions remain unknown, so the report may be
unscoped; this call does not recover the operating conditions of a local run.

If the remote run aborted before producing analysis data, its rawfile may
contain only ngspice's synthetic `constants` plot, which reports one point
and would otherwise be measured as if it were results. `evaluate()` raises
`SimulationAborted` on such a file (`data_rows=0`) before any metric runs, so
no measurement is recorded on any spec; `tb.netlist.load_file()` +
`read_raw_file()` raise the same way. External files receive a data-presence
check, not a completion verdict: an aborted run may contain partial analysis
data that passes this check. Verify external completion separately. Supplying
`log=` enriches a detected no-data error; it does not by itself reject partial
data. Hand over the remote log to get the reason:

```python
from silk.exceptions import SimulationAborted

try:
    report = tb.evaluate('tt', rawfile='rawfile.raw', log='output.log')
except SimulationAborted as e:
    print(e.simulator, e.data_rows, e.reason)   # ngspice 0 Timestep too small...
```

`report` is a `SpecReport`; check `report.passed` for a boolean summary or
iterate `report.results` for per-spec detail.  See
[api-reference.md](api-reference.md) for `ModuleTestbench` and `SpecReport` full
signatures.

> **Metrics note:** `dc_gain_dB` lives in `silk.metrics.ac`, re-exported from
> `silk.metrics` — the one metric library. Metrics are grouped by what they
> measure; each function documents the input signal kinds it accepts. See
> [docs/metrics-library.md](metrics-library.md).

To revise the teaching OTA, follow [revise, rebuild, rerun](agent-guide.md#revise-rebuild-rerun):
change `build_5t_ota(pair_w=...)`, build fresh embedded tests, and save the
measurements, report and netlist from that revised design. Existing benches
contain DUT copies; changing the original does not update them.

## 8b. Emit a SPICE netlist

`generate_netlist` has two modes depending on whether `define_module_name()` has
been called on the module:

```python
# Subcircuit mode — module_name is set: returns .subckt string, no file written
mod.define_module_name('my_amp')
subckt_str = mod.generate_netlist(corner='tt')
print(subckt_str)          # starts with ".subckt my_amp ..."

# Top-level mode — module_name is None: writes the netlist and returns its path
top = Module('PDK_skywater130', 'out', 'top.cir', pdk_details=pdk)
deck_path = top.generate_netlist(corner='tt')   # returns the actual path under out/
```

By default, top-level output goes into a new run directory under `out/`.
Use the returned path; `set_output_options(mode='overwrite')` instead writes
directly to the configured directory.

### A block with embedded testbenches

A named block's `.subckt` text excludes its testbenches. Generate one separate
top-level deck per testbench:

```python
from pathlib import Path
from silk.examples.ota_5t_guide import build_5t_ota

mod = build_5t_ota()  # registers ac, tran and noise tests
for name in mod.list_tests():
    tb = mod.get_test(name)
    deck_path = tb.netlist.generate_netlist(corner='tt')
    print(name, Path(deck_path).resolve())
```

This writes AC, transient and noise decks without running a simulator.
For this example, the common output directory is
`silk-output/ota_5t_guide/`, relative to where you run Python. Its
`test_ota_5t~ac/`, `test_ota_5t~tran/` and `test_ota_5t~noise/` directories
hold the separate tests. `SILK_EXAMPLES_OUT` changes the `silk-output` base
directory.
Each generation creates a fresh `run-*` directory; use the returned path.
`tb.run()` generates another deck before simulation.

Use `corner='tt'` for typical, `'ff'`/`'ss'` for fast/slow corners, or pass a
`Corner` enum value. For design-space sweeps see §9 below.

## 9. Design-space exploration

See [optimizer.md](optimizer.md) for bounds, sampling methods, the
`Optimizer` harness, and its limitations.

```python
from silk import Module, sources, load_pdk
from silk.simulation import Transient
from silk.optimize import VariableSpace

pdk = load_pdk('sky130')

# Create Variable objects via the PDK parameter descriptor
w_n = pdk.nmos1p8.w.variable(name='w_n', value=1.0, bounds=(0.42, 20.0))
w_p = pdk.pmos1p8.w.variable(name='w_p', value=2.0, bounds=(0.42, 20.0))
l   = pdk.nmos1p8.l.variable(name='l',   value=0.15, bounds=(0.15,  1.0))

# Pass variables directly as device parameters — links them to the space
mn = pdk.nmos1p8('mn', d='out', g='in', s='0', b='0', w=w_n, l=l)
mp = pdk.pmos1p8('mp', d='out', g='in', s='vdd', b='vdd', w=w_p, l=l)

# Use a fresh circuit containing the devices linked to these variables.
ckt = Module('PDK_skywater130', 'out', 'inverter.cir', pdk_details=pdk)
ckt.add(mn, mp)
ckt.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0', dc=1.8))
ckt.add_source(sources.VoltageSource(
    'vin', positive='in', negative='0',
    pulse=sources.Pulse(v1=0, v2=1.8, td=5e-9, tr=1e-9, tf=1e-9,
                        pw=20e-9, per=40e-9)))
ckt.add_sim_type(Transient(step=1e-9, stop=80e-9))

space = VariableSpace([w_n, w_p, l])
print(space.names)          # ['w_n', 'w_p', 'l']
print(space.to_bounds())    # [(0.42, 20.0), (0.42, 20.0), (0.15, 1.0)]

# Sampling strategies
pts_random = space.sample(20, method='random')
pts_lhs    = space.sample(20, method='lhs')    # Latin hypercube
pts_halton = space.sample(20, method='halton') # Halton low-discrepancy
pts_grid   = space.sample(4,  method='grid')   # 4^3 = 64 points

# Apply a point and regenerate the netlist
for pt in pts_lhs:
    space.apply(pt)                             # sets variable .value fields
    ckt.generate_netlist(corner='tt')            # picks up new values
```

## 10. Multi-corner simulation

Continue with the complete inverter from section 9, which has supply and
input sources, a transient analysis, and the output node `out`. Section 9
only generated decks; this loop runs the final sampled design at each corner.
It requires ngspice and the installed sky130 models.
For testbenches, use `tb.run_corners(corners)` to characterize each corner.

```python
from silk.simulation import Corner

corners = Corner.standard()   # (Corner.TT, Corner.FF, Corner.SS, Corner.SF, Corner.FS)

results = {}
for c in corners:
    ckt.simulate(corner=c)
    results[c] = ckt.load_signal('v(out)')
```

## 11. Schema export

The module example assumes `my_module` is a module you have built.

`export_schema` produces a machine-readable dict describing all devices — useful
for inspection tools, code generators or documentation.

```python
from silk import export_schema, module_schema
import json

# All PDK devices
schema = export_schema(pdk)
print(json.dumps(schema, indent=2))

# All ideal elements
from silk.elements import elements as elem_lib
elem_schema = export_schema(elem_lib)

# A single Module's interface
mod_schema = module_schema(my_module)
```

## 12. Database / version tracking

Track circuit versions and measurements in SQLite. Existing older databases
need an explicit copy migration; see the [copy-migration route](database.md#migrating-a-schema-v1-database)
and the [database guide](database.md#file-compatibility) for compatibility.
Runnable end-to-end demo (no ngspice needed):
`python -m silk.examples.circuit_tracking_example`.

Pass `db_path` to `Module`, declare stable `key=` values, and seal before
writing evidence. The fragment below assumes `mn1` and `mp1` are devices
you built, and `build_ldo()` reconstructs exactly the same circuit and spec:

```python
ckt = Module('PDK_skywater130', 'out', 'ldo.cir', pdk_details=pdk,
             module_name='ldo_nmos', db_path='designs.db')

ckt.add(mn1, mp1)
ckt.declare_spec('DC gain', 'gain_dB', '>=', 40.0,
                 units='dB', key='gain')
version_id = ckt.create_circuit_version('Initial attempt')
ckt.store_results(
    {'gain_dB': (42.5, 'dB'), 'status': ('converged', None)},
    source='spice', sim_type='ac', corner='tt', temperature=27.0,
    voltage=1.8, tool_name='ngspice-44', conditions={'load_pf': 2.0},
)
ckt.close()

# Later process: rebuild the same Python definition and restore explicitly.
rebuilt = build_ldo(pdk, db_path='designs.db')
assert rebuilt.create_circuit_version() == version_id
rows = rebuilt.query_results(parameter='gain_dB', corner='tt')
# If the first session also used store_report(...):
# rebuilt.restore_spec_measurements(source='spice', corner='tt')
# If it used store_net_currents(...):
# rebuilt.load_net_currents('tt')
stable_id = rebuilt.promote('stable')
assert rows[0].origin_version_id == version_id
rebuilt.close()
```

`store_report(tb.run(...))` is the canonical bridge for restorable first-class
spec measurements. `store_net_currents()`/`load_net_currents()` bridge
characterized currents into EM/layout sizing. JSON export is a marked, lossy
projection—not a backup, archive, Python reconstruction, or raw-run history.

## 13. Exceptions

```python
from silk.exceptions import (
    CircuitError,        # base class
    DuplicateNameError,  # duplicate element/source/passive name, or a duplicate test name
    UndeclaredNetError,  # strict_nets mode: net not declared
    SimulationError,     # ngspice failed or signal not found
    SimulationAborted,   # no usable data, or incomplete certified analysis (subclass of SimulationError)
    PDKError,            # bad YAML / duplicate device
    NotSupportedError,   # requested capability is unsupported
)
```

Enable strict net checking on a circuit:

```python
ckt = Module('PDK_skywater130', 'out', 'ckt.cir',
             pdk_details=pdk, strict_nets=True)
```

## Next steps

- Human-facing examples guide (learning path + by-topic index): [docs/examples.md](examples.md)
- Full API reference: [docs/api-reference.md](api-reference.md)
- Circuit reports (`generate_report`, `diff_reports`): [docs/reports.md](reports.md)
- Analog lint and coverage rules: [docs/lint_and_coverage_user_guide.md](lint_and_coverage_user_guide.md)
- Design-stage bridge (hierarchical specs + reconciliation): [docs/design_stage_bridge_user_guide.md](design_stage_bridge_user_guide.md)
- Working examples: `silk/examples/`
- Project README: [../README.md](../README.md)
