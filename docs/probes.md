# Probes: declare, measure, run, read back

A testbench probe names a signal before simulation. It is a `Signal`, not a
new kind of port: arithmetic and transforms can be declared before data exists,
then registered metrics and plots resolve the expression against the results.
Use the same workflow for tests with several analyses and for corner reads.
The [API reference](api-reference.md#moduletestbench) lists the full signatures.

## A runnable example

This ideal RC circuit needs silk and ngspice, but no installed PDK models.
`load_pdk` supplies the packaged catalog; no PDK device is instantiated.
Run with `MPLBACKEND=Agg` on a headless machine. Output goes under `out/probes/`
relative to the working directory.

```python
from silk import Module, Port, load_pdk, sources
from silk.elements import elements
from silk.metrics.statistics import average
from silk.metrics.waveform import peak
from silk.simulation import Transient


def build_rc(*, resistance=1e3, capacitance=1e-9):
    dut = Module('PDK_skywater130', 'out/probes', 'rc.cir',
                 pdk_details=load_pdk('sky130'), module_name='rc')
    dut.define_module_ports({'a': Port.analog_input(), 'b': Port.analog_output()})
    with dut:
        elements.resistor('r1', p='a', n='b', value=resistance)
        elements.capacitor('c1', p='b', n='0', value=capacitance)

    tb = dut.create_test('step')
    tb.add_module_under_test({'a': 'vin', 'b': 'vout'})
    tb.add_source(sources.VoltageSource(
        'vs', positive='vin', negative='0', dc=0,
        pulse=sources.Pulse(v1=0, v2=1, td=0, tr=10e-9, tf=10e-9,
                            pw=5e-6, per=10e-6)))
    tb.add_sim_type(Transient(step=10e-9, stop=10e-6))
    vout = tb.probe('vout')
    drop = tb.probe('vin') - vout
    tb.add_metric('out_avg', average, sig=vout)
    tb.add_metric('drop_peak', peak, sig=drop)
    tb.add_spec('output responds', 'out_avg', '>', 0.4)
    return dut


dut = build_rc()                   # owns its characterization test
tb = dut.get_test('step')
vin, vout = tb.probe('vin'), tb.probe('vout')
assert not vout.bound              # no samples yet
report = tb.run()
print(report.metrics)
print(vout.y[-1])                  # reads the completed run
snapshot = vout.snapshot()         # ordinary Signal carrying these samples
fig = tb.plot(vin, vout, show=False)
fig.savefig('out/probes/step.png')
```

The two metrics and the spec are evaluated during `run()`. For this
single-analysis run, `snapshot.x` and `snapshot.y` hold the same samples that
`vout` reads at resolution time. Declaring a derived waveform such as
`(vin + vout) / 2`, `vout.derivative()` or `vout.window(1e-6, 4e-6)` also defers
until results are requested. Normal signal compatibility checks still apply.

`sig.versus(x_signal)` re-plots one Signal against another's values — for
example `(gm / id).versus(id / w)` for a gm/Id lookup curve. It defers like the
other transforms. Strictly decreasing x is reversed to ascending; x that is not
strictly monotonic raises `ValueError` unless `allow_nonmonotonic=True`, which
keeps sample order (a parametric trace, for plotting).

## Metrics: bind signal slots by name

The function form takes a catalog or plain Python function and its named signal
slots: `tb.add_metric('out_avg', average, sig=vout)`. The slot is named by the
function signature, not always `sig`: for example, `average_power` takes
`sig_v=` and `sig_i=`. Extra metric parameters are ordinary keywords. Probes
bound to a metric must belong to that same testbench, including the leaves of
a derived expression; cross-bench metric bindings are rejected.

A probe taken from a selected placement keeps that placement's identity.
Registering it directly on another bench raises `TypeError`. Shared metric
declarations are checked again before evaluation extracts any metric: a probe
pinned to a different placement raises `MetricBindingError`, even when both
placements share the same authored test. No successful report is produced.
Use an authored/local probe (`tb.probe(...)` on the authored bench) or a
declarative signal name (`type='average', signal='vout'`) to measure each
placement's own node. Bound reference waveforms remain valid inputs; opaque
lambda captures are outside the inspectable metric-slot checks.

The declarative and callable forms remain supported:

```python
# Add these before running the bench above.
tb.add_metric('out_avg_declarative', type='average', signal='vout')
tb.add_metric('out_avg_mV', lambda ckt: average(ckt.load_signal('vout')) * 1e3)
```

Use the callable form when an extractor needs the simulated `Module`
directly; it receives that `Module`, not a `ModuleTestbench`, and should load
signals from that argument. For arithmetic on reported metric numbers, use
[metric handles](#derive-metrics-from-reported-values). Capturing an outer probe in a lambda is not a replacement for
registered binding: copying a testbench rebinds its registered probe slots,
but cannot rewrite a Python closure.

## Names and paths

Authored net, port and instance names use ASCII letters, digits and `_`,
start with a letter or `_`, and may end in one numeric index such as `data[3]`.
The ground net `0` is the only leading-digit exception; instances cannot be `0`.
Names such as `a:b`, `a.b`, `v-bias`, `bus[en]`, `net[3]bias`, `d[1][2]`
and `gnd!` are refused at construction and rechecked at netlist emission.
`gnd!` is an ordinary node in SPICE, not ground. Use `0`; inside a block,
declare `Port.ground()` (conventionally `vss`) and wire it to `0` at the bench.
The `.` and `:` separators remain legal in hierarchy selectors and probe paths,
including `tb.probe('m1:gm')`; they are not part of an authored name.

## Arithmetic on transformed probes

Declare spectrum arithmetic before running, then read it implicitly:

```python
vin = tb.probe('vin')
vout = tb.probe('vout')
in_spectrum = vin.fft(tclk=1e-9, n_samples=1024)
out_spectrum = vout.fft(tclk=1e-9, n_samples=1024)
# Choose a band with nonzero denominator values (FFT suppresses DC).
ratio = out_spectrum.window(1e6, 100e6) / in_spectrum.window(1e6, 100e6)
product = out_spectrum * in_spectrum
squared = out_spectrum ** 2
tb.run()
frequencies, values = ratio.x, ratio.y
ratio.plot()
```

The recipe's declared type remains None, including after a run or read. Each
evaluation computes a separate result with the metadata supplied by its
operands and transforms; it does not update the recipe. Reads, plots
and registered metrics evaluate the expression against the current bench
results; no explicit `resolve()` is needed. Both operands must have compatible
axis labels and exactly equal coordinate shapes and values. This applies to
`+`, `-`, `*`, `/`, bus arithmetic and `codes_from_bits()` (binary and
thermometer decoding), including expressions evaluated after a run. Equal
copied axes work regardless of array identity or dtype. Near-equal coordinates
and singleton-versus-longer grids raise `ValueError`; arithmetic never
interpolates or snaps coordinates. The error names the signals and lengths,
and the first differing coordinate for equal-shaped grids.

Use matching recorded coordinates. Independently constructed `linspace` or
`arange` axes can differ from recorded coordinates in their last bits and are
then refused. When resampling is appropriate, explicitly align one signal
first with `silk.metrics.sample(other, x_new=reference.x)`. Division preserves
ordinary zero-denominator behavior.

This ratio divides stored spectrum values. The default FFT stores squared
magnitudes, so their ratio is a power-spectrum ratio, without phase; it does
not automatically calculate a complex transfer function. FFT scaling remains
controlled by `scaling=`.

## Xyce capture defaults

Declare node probes before `tb.run()`. With no explicit save list, Xyce
captures the bench's top-level nodes plus the nodes requested by those probes,
including DUT-internal and nested arrayed paths such as
`tb.probe('dut.X1[0].g[1]')`. Connected ports resolve to their parent nodes;
duplicate requests add only one output vector. ngspice keeps its `save all`
default. A probe added after a run needs another run if its node was not saved.

Declarative metrics bound before the run request the same node capture:
`tb.add_metric('level', type='average', signal='dut.mid')` and every path
in `signals={slot: path, ...}` use the same resolution and validation as node
probes. This includes generated testbenches and DUT bus bits such as
`xdut.y[2]`. A refusal names the metric and its path. Current, device-parameter
and noise-spectrum inputs retain their existing routes; `analysis=` still
selects the metric's analysis.

Xyce validates these requests before launch: unknown or ambiguous nodes and
unused declarations, unsupported literal punctuation, and opaque model-internal
paths that cannot be resolved from the circuit raise a typed `CircuitError`
naming the probe. Ground has no captured node vector. A noise-only analysis
writes spectra, so node-voltage probes require a separate AC, DC, operating-point
or transient test.
An explicit `set_save_list(...)` remains authoritative and may deliberately
omit a declared node probe; reading an omitted node still raises. Reset it
with `set_save_list(None)` to restore automatic capture. See
[simulator save lists](simulator-save-list.md) for selection rules.

## Arrayed instance names and raw paths

Designer names keep square brackets: `X1[0].M1[0]:gm` is a device-parameter
probe. SPICE instance components use angle brackets instead, so its raw
hierarchy contains `xX1<0>.XMM1<0>`. This avoids ngspice's ambiguity between
an instance index and the parameter delimiter in `@instance[param]` saves.
The same instance spelling is used by Xyce and behavioral result paths.
Net and port names keep their spelling: an internal `g[0]` becomes a vector
such as `v(xX1<0>.g[0])`. Generated `.control save` arguments are double-quoted;
the quotes are not part of the raw vector name.

For a bench source named `vs[0]`, both `tb.probe('i(vs[0])')` and a metric's
`signal='i(vs[0])'` resolve to raw `i(vvs<0>)`. The bracket spelling of a raw
path is an alias again: `i(vvs[0])` reads the same vector, and
`xx1[0].xy[1].g[0]` resolves like `xx1<0>.xy<1>.g[0]` (instance parts convert,
the node leaf `g[0]` does not; matching stays case-insensitive).
The same `[k]` spelling is accepted by every selector (`set_save_list`,
`emit_view(signals=/exclude=)`, `silk.view --signals`, `tb.netlist.view(signals=)`) and
is what the viewer, `get_signal_names()` and the annotation report display;
rawfile names remain the data keys. Only when two distinct vectors differ
by `[k]` versus `<k>` alone does silk show the raw names, and the remedy of an
ambiguous request names the raw spelling (`i(Vbias<0>__silk_pwl)`) or asks you to
rename; see [simulator save list](simulator-save-list.md#selectors).
`i(<name>)` considers the source's designer name and the normalised emitted
name together: two different sources reached that way (`vs[0]` beside a source
named `vvs[0]`; `v` beside `vv`) raise
`AmbiguousBranchCurrentError`. A raw current name that names no source is
handed to the one signal collector with its original spelling, so an
`i(vvs[0])` row, an `i(vvs<0>)` row, or both (ambiguous) are each found or
refused as they are, never read as a missing current. When a
signal request matches two distinct rows of the plot (for example a literal
top-level net `xx1[0].mid` beside the hierarchical `xx1<0>.mid`, which is also
its alias) silk raises `AmbiguousSignalError` naming the request, the plot and
the candidates; it never picks one. The unambiguous simulator spelling (instance parts as `<k>`, for example `xx1<0>.mid`) selects the hierarchical vector; if a literal net's name collides with it, rename the net. Literal `<` and `>` in a
designer instance name are reserved and refused; use square brackets for
array indices. A literal designer neighbor `X1_0` remains distinct.

Every waveform source has two physical instances: dedicated `pwl=` uses
`Vbias<0>__silk_pwl` / `Vbias<0>__silk_dc` (or `I...` for current); other waveform slots
use `__silk_wave` / `__silk_dc`. A voltage source's private series node is
`__silk_src_Vbias[0]`; designer brackets remain net characters. The
case-insensitive `__silk_` net prefix is reserved for silk at the beginning
of each hierarchy component. Designer instance names refuse `__silk_`
anywhere, preventing generated-card collisions with authored instances.
Use the public `silk.circuit.validate_instance_name` and `validate_net_name`
checks to validate names before adding them. Logical probes are unchanged;
raw waveform card references use the reserved suffixes. Raw references in
opaque lines and external includes must respect the emitted spelling and
reserved namespace; included file contents are not recursively validated.
Private source nodes can be selected explicitly in `set_save_list`, including
hierarchical paths such as `xleaf[0].__silk_src_Vbias[0]`.

`i(<source name>)` reads any source by the name you gave it, including a source
silk emits as two cards; the `__silk_` cards are only for looking inside.

For a waveform source, `tb.probe('i(bias[0])')` and a metric string with the
same path read its logical current: the series branch for voltage, the signed
sum of both measured parallel currents for current sources. Both operands are
saved, including with a selective node save list. A physical name such as
`i(Ibias<0>__silk_dc)` reads only that card. Missing operands raise; authored source
values are never substituted. Current annotations sum samples before computing
statistics and store `current_vector=None` for a sum; persistence retains
that absence of a single-vector provenance.

**AC current-source reads are refused** with `NotSupportedError`, including
logical names, physical cards, metrics and annotations. An ideal current
source's AC current is its `ac=` value; to measure it, place a 0 V
`VoltageSource` in series (silk's ammeter pattern), then probe that voltage
source's current. Transient and DC current-source readback use saved simulator
vectors. AC voltage-source branch measurements remain available.

Create device-parameter probes before `run()`; a probe absent from that run's
request set is refused even if a matching vector exists.

Missing or empty explicitly requested device-parameter vectors emit an
`op_parameter_missing` warning naming the designer instance, parameter,
address and intended plot. The requests belong to the run that produced the
results, including shared and worker results. Loading an external rawfile
does not infer requests from the current circuit. A present numerical NaN
alone does not trigger this missing-vector warning.

## Designer node paths

`add_module_under_test()` places the DUT as instance `dut` (`xdut` in the
netlist). Node-voltage strings start at the receiving testbench:

| Probe | Node selected |
|---|---|
| `tb.probe('vin')` | Bench net `vin` |
| `tb.probe('stage.mid')` | Internal net of the helper placed as `stage` |
| `tb.probe('dut.mid')` | Internal net of the DUT |
| `tb.probe('dut.x1[0].mid')` | Internal net in the DUT's child `x1[0]` |
| `tb.probe('dut.out')` | Bench net wired to DUT port `out` |

`v(dut.x1[0].mid)` works too. The emitted spellings
`xdut.xx1[0].mid` and `xdut.xx1<0>.mid` still work. Designer probe lookup uses
actual placement/net keys, including their case and declared array numbers.
Authored names cannot contain dots. In older circuits or external data, a
joined path may still identify two different targets; use a placement reference
or an unambiguous emitted path when the error requests one. Port aliases
reaching the same physical node count once.

An existing emitted/raw reading wins over a different designer target, with a
`UserWarning` naming the designer spelling and emitted alternatives. Multiple
raw rows still raise `AmbiguousSignalError`. The probe keeps its original
request and resolves it on each read against the selected reader's circuit
and complete raw plot. This applies to external files, `analysis=`, retained
corners, reruns, copies and selected placements. It does not retain a wiring
snapshot: a changed graph is used on the next read. A removed designer target
and a valid target missing from the rawfile fail with different diagnostics;
neither returns a fabricated value. Missing DUT-relative paths suggest the
`dut.` prefix when that resolves to a known node.

Save selection lives on **Module**, for example
`tb.netlist.set_save_list(['dut.x1[0].mid', 'dut.out'])`. It resolves against
that receiving module at netlist generation; a setting on the DUT definition
does not configure its characterization benches. Use bare selectors, not
`v(...)` wrappers. Existing case-insensitive full-name, glob (`*`, `?`) and
hierarchy-segment matching applies; square brackets are literal array syntax.
Globs may select multiple nodes. Include and exclude both check the full
inventory, and an emitted match set wins over a different designer set with
a warning. Depth limits apply to physical node paths after port wiring; ground
has no saved vector. Equivalent designer and emitted selections write identical
save commands. Private source nodes keep their emitted spelling.

Device-parameter names such as `st1.MN:gm`, terminal probes such as
`st1.MN:d`, and `i(vs)` keep their existing routes. Designer node strings do
not change rawfile keys or netlist names. Viewer selectors and labels use emitted names with bracket
aliases by default. Checked `node_context` enables designer selection and, with
`labels='designer'`, designer display labels; see the checked-node metadata
guidance in [checked designer node metadata](explorer-integration.md#checked-designer-node-metadata). There is no `ModuleTestbench.view` or
`ModuleTestbench.set_save_list` method.

## Probe a placement by object

Placement handles carry structural net references before simulation:

```python
stage = tb.add_module(bias, 'stage', nets={'vbias': 'bias_net'})
p = tb.probe(stage.vbias)          # connected port reads bias_net
q = tb.probe(stage.net('vbias[0]'))   # explicit lookup for an internal net
# Register metrics, then run the bench as usual.
```

`stage.vbias` acquires no samples. `tb.probe` validates the actual placement
and returns the same deferred `Signal` route used by string probes. An internal
net reads its hierarchical path; a port reads its connected parent net. Nested
placements and separate copies retain their placement identity. A bare
`Net` from a definition is refused: use the placement handle to identify the
copy. Structural references are voltages; use `i(source)` or a terminal-current
probe for currents. Colon-containing authored net names are refused; hierarchy and device probe syntax is unchanged.

After a run, `stage.vbias.y`, `.plot()` and Signal arithmetic work when exactly
one live result context is eligible. Several benches or selected owner paths
require `p = selected_tb.probe(stage.vbias); p.y`, even when only one has run.

A selected probe registered in a metric keeps its root, owner and authored-test
binding when pickled with that root graph. Pickling the probe or authored bench
without the selected root leaves it detached and refuses reads; it never adopts
another owner's results.

A removed/replaced placement invalidates a held structural reference. Live
rewiring is honored when binding; an already bound Signal follows its bench
through reruns and rebuilds. `p.snapshot()` retains a fixed result. An explicit
`analysis=` creates a new probe without changing another probe. Object probes
follow the same capture rules as string probes: Xyce includes them in automatic capture, but an explicit save list can omit them.

Use `stage.net(name)`, `stage.bus(name)`, `stage.child(name)` and
`stage.device(name)` for explicit lookups when those tools are available.
Declared names come first: a child called `child` is reached as `stage.child`.
An item named `bus`, `child`, `device`, `instance_name`, `net`, `path` or
`subcircuit` shadows the handle's tool with that name, making that tool
unavailable through that handle. A name matching several item kinds retains
the existing ambiguity error. See
[declared names and dot shortcuts](known_issues.md#declared-names-and-dot-shortcuts).

Bus and array subscripts name **declared numbers** on Net, Signal and placement
handles. For `bits=range(8, 12)`, `stage.d[8]` selects `d[8]`; `[0]` refuses.
`stage.d[8:11]` selects bits 8, 9, 10 and preserves those numbers; iteration
preserves the selected order. `stage.X[8]` and `stage.M[3]` select actual module
and device keys `X[8]` and `M[3]`. Two keys such as `M[03]` and `M[3]` refuse
rather than overwrite the same declared number. A width-one bus is still an
aggregate, including for integer decoding and report rendering. Decoding
weights follow iteration positions, so declared bit 8 is not weighted as 256.

Slice bounds are numerical, with inclusive start and exclusive stop. A positive
step traverses the stored order; a negative step reverses it. Omitted bounds
impose no bound. Steps select numbers congruent to the explicit start (or the
first traversed declaration) modulo the step magnitude. Negative indices are
literal declared numbers, rather than positions from the end. Empty selections
refuse with declared indices; a zero step raises `ValueError`.

| Declarations | Selection | Declared numbers returned |
| --- | --- | --- |
| 8, 9, 10, 11 | `[8:11]`, `[:10]`, `[10:]` | 8,9,10; 8,9; 10,11 |
| 8, 9, 10, 11 | `[::-1]`, `[11:7:-2]` | 11,10,9,8; 11,9 |
| -3, -1, 2, 5 | `[-1]`, `[-3:6:2]` | -1; -3,-1,5 |
| 11, 10, 9, 8 (reversed handle) | `[8:11]`, `[::-1]` | 10,9,8; 8,9,10,11 |
| 8 (width-one bus) | `[8]`, `[:]` | bit 8; aggregate containing bit 8 |

Ordinary lists and waveform sample arrays retain positional indexing.

### Bus maths: element-wise, like numpy

A bus Signal behaves like a numpy array under arithmetic. `+ - * / **`, unary minus and the Signal transforms (`mag()`,
`real()`, `db20()`, `derivative()`, `shift()` and the rest) act on every bit
and return a bus, so `d.mag()` is a bus. `as_int()`, `len`, iteration and
indexing are unchanged.

```python
d = tb.probe(stage.d[4:7])         # bits 4, 5, 6
ref = tb.probe(stage.dref[4:7])
err = (d - ref).mag()              # a bus of three bits
scaled = d * 2 - 0.5               # a number applies to every bit
centred = d - tb.probe('vcm')      # so does a scalar Signal
```

For `+ - * /`, two buses of **equal length combine by position**, whatever their declared
numbers: with `a = c[1:3]` and `b = c[2:0:-1]`, `a - b` holds `c1 - c2` then
`c2 - c1`. A declared bus keeps its bit numbers (`d[8]` is the net `d[8]`;
indexing and slicing are unchanged, and a Python slice excludes its stop, so
bits 8, 9, 10 are `d[8:11]`). **A one-bus maths result keeps its source's
bit numbers:** `(d * 2)[8]` is `2 * d[8]`, and `d.mag()[8]` and
`(d * 2).mag()[8]` also select the bit made from `d[8]`; `[0]` raises.
**A two-bus result is numbered from 0 by position.** With `d` declared
from 8 and `c` from 0, `e = d + c` gives `e[0] = d[8] + c[0]`,
`e[1] = d[9] + c[1]`, `e[2] = d[10] + c[2]`; `c + d` also starts at 0.
Buses of different length raise `silk.exceptions.BusPairingError` (a `ValueError`),
naming both widths. `+ - * /` also broadcast a number or scalar Signal on either side; `**` takes a
real-number exponent and applies it to every bit (no bus exponent). There is no pairing by declared bit number; to pair bits by
number, loop over them. A transform's own arguments are forwarded unchanged to
every bit, and each method keeps the argument types it accepts for a scalar
Signal; a bus-valued argument is refused with `TypeError`.

Deferred probes behave the same as bound ones: the expression is built before
the run and each bit resolves against its own bench's results.

A module can expose only part of an internal bus as pins. Declare the port
first, then call `add_net` with a range containing every port bit: for example,
`define_module_ports({'x': Port.analog_input(bits=4)})` followed by
`add_net('x', bits=8)`. The port keeps its four formal bits and metadata;
`mod.nets.x` and a placement's `stage.x` cover all eight internal bits.
`tb.probe(stage.x)` follows the port bits to the wired parent nets and the
remaining bits to the placement's internal nets. Both naming styles work;
the port's naming style applies. A narrower range or one missing any port bit
raises `CircuitError`. Declaring the wider internal bus before the narrower
port is unsupported. Parent wiring still requires the formal port width
(or the existing scalar broadcast for input/inout ports).

`tb.bus(name)` is the bench-level counterpart of `stage.bus(name)`: it
returns a reference to a bus declared on the test bench's own netlist, with
the same declared-number subscripts (`tb.bus('d')[5:7]` is bits 5 and 6). It
keeps its bench, and its selected owner when taken from `top.get_test(...)`,
through slicing, arithmetic and probing; `tb.bus('d')[5]` and iteration give
scalar `BenchBitReference`s with the same ownership, and binding any of them on
another bench or owner raises. Bind it with
`tb.probe(tb.bus('d'))`. An unknown name raises `CircuitError` listing the
bench's buses. `stage.X[8].bus('rout')` works on one copy of an arrayed
module; two copies of the same bus combine by position, like any two buses of
equal length.

`tb.probe_bus(names)` builds a bus from named signals when there is no declared
bus to point at. `names` is a list or tuple of path strings; element `i` is
`tb.probe(names[i])`, so every spelling and every error of `probe` applies per
element. The result is a **new ordered aggregate numbered 0..N-1** in list order,
whose leaves keep their probe paths: `d = tb.probe_bus(['d[10]', 'd[9]'])` has
`d[0]` the probe of net `d[10]`. It does not rename or redeclare a circuit bus.
A bare string raises `TypeError` (use `probe`); `[]` and `()` raise
`ConfigurationError`; a non-string element raises `TypeError` naming its index.
The codegen JSON writes one of these per bus: see
[buses in the codegen contract](codegen-contract.md#buses-a-list-of-paths-for-one-bus).

## What can be probed?

| Spelling | Meaning / requirement |
|---|---|
| `tb.probe('vout')` or `tb.probe('v(vout)')` | Voltage on a testbench net. Plain names are checked against known bench nets. |
| `tb.probe('xdut.mid')` | Voltage on an internal DUT node, if that vector was saved. |
| `tb.probe('i(vs)')` | Branch current through `VoltageSource('vs', ...)`; silk maps the source name to its emitted vector. In the RC example, this is the driving source's current, with the simulator's source-current sign convention. |
| `tb.probe('vvs', 'I')` | Low-level spelling of that same current vector. |
| `tb.probe('st1.MN:d')` | Drain-terminal current of the DUT device you named `MN` inside submodule instance `st1` — the same designer name `tb.probe('st1.MN:gm')` takes. Same vector, same requirements as the emitted-path form below. |
| `tb.probe('xdut.m1:d')` | The same terminal current by EMITTED path (still accepted, unchanged): drain-terminal current of device `m1` through silk's ammeter wrapper; requires the DUT's `save_device_currents = True`, `terminal_current_method = 'ammeter_wrapper'`, and exactly one transient analysis. |

If a prefix reads as both an emitted path and a designer name, and the two reach different devices, the emitted-path device is used and a `UserWarning` names the spelling (`'xdut.<path>.<name>:d'`) that selects the DUT device.

There is no current vector *for a node*. Insert a voltage-source ammeter
(0 V if appropriate) at the branch you want to measure. Device-current
wrappers are a separate opt-in; they are not required for the RC example's
voltage-source current. Xyce automatically captures declared node probes when
no explicit save list is set; ngspice defaults to saving all nodes. With
[selective saves](simulator-save-list.md), retain
every voltage node the expression needs. Selective saves support node voltages
only: current selectors are rejected, and combining a save list with
`save_device_currents` raises. Voltage-source branch probes are saved even
with a selective node list.
Use the default capture settings for device-current wrappers.


## Behavioral and SPICE probes

Select the backend using the [same testbench workflow](agent-guide.md#one-workflow-two-engines).
The table describes resolved data, after a run. Device probes can refuse at
declaration when the behavioral backend is already selected.

| Spelling / request | SPICE | Behavioral |
|---|---|---|
| `tb.probe('vout')` or `tb.probe('v(vout)')` | Saved node voltage | The net's model/source array, typed voltage |
| `tb.probe('xdut.n')`, inside an entered block | Saved internal voltage | Internal net array; blocks without a model are entered |
| `tb.probe('xdut.n')`, inside a modelled leaf | Saved internal voltage | `NotSupportedError`: the model replaces the block's internals |
| `tb.probe('vout', 'I')` | Not-found `SimulationError` | Same error: a node has no branch current |
| `tb.probe('i(b)')` for `CurrentSource('b', ...)` | Transient/DC: measured current, summing both cards when present. AC: refused; use a voltage-source ammeter | Transient: the independent source's own defined waveform, typed current; DC: no branch-current vector |
| `tb.probe('vs', 'I')` for `VoltageSource('s', ...)`, or `tb.probe('i(s)')` / `tb.probe('i(vs)')` | Saved voltage-source branch current | `NotSupportedError`: needs an electrical solve |
| `tb.probe('xdut.m1:d')`, `tb.probe('m1:gm')`, report `op_table()` | Device currents/parameters with the corresponding SPICE saves | `NotSupportedError`: no device currents, parameters or operating-point table |
| `(vp + vm) / 2`, derived signals and named metric slots | Deferred until results exist | Same workflow |

In transient results, the independent current-source waveform is separate from its net's array,
including under entered hierarchy (`xdut.ib`). A model output remains typed
voltage even if the model author interprets its numbers as current; requesting
`'I'` on that net does not convert it. Model quantity typing remains deferred.

Use `analysis='tran'` or `analysis='dc'` for behavioral data. The selector is
honored, not ignored. `simulate_behavioral()` uses these same read rules;
changing the backend setting after a run does not relabel its evidence.
For DC the x-axis is the swept source value (`v(v-sweep)` or `i(i-sweep)`),
not time. DC currently writes net arrays and the sweep axis only: asking for
an independent source's branch-current vector gives not-found `SimulationError`
even when that current source is the sweep target. A single-point real plot
follows the shared reader convention:
`xlabel='op'`, `x=[0]`, while the rawfile retains the actual sweep value.

Verification: `.venv/bin/python -m pytest -q tests/test_behavioral_probes.py`
covers each behavioral row and the shortcut;
`.venv/bin/python -m pytest -m slow -q tests/test_behavioral_probes.py`
covers the SPICE rows (ngspice + sky130 required).
`.venv/bin/python -m pytest -q tests/test_behavioral_dc.py` covers DC axes
and selection. Remaining [refusals](known_issues.md#behavioral-simulation)
include AC/noise/op, device saves and ambient temperature.

## More than one analysis

For a bench declaring both transient and AC analyses, write the selection on
the probe. This is a separate authoring pattern, not an extra analysis to add
to the already-configured RC test above:

```python
from silk.metrics import dc_gain_dB

# tb is a wired bench declaring both Transient(...) and AC(...).
v_time = tb.probe('vout', analysis='tran')
v_freq = tb.probe('vout', analysis='ac')
tb.add_metric('mean_voltage', average, sig=v_time)
tb.add_metric('gain_dB', dc_gain_dB, sig=v_freq)
```

The accepted selectors are `tran`, `ac`, `dc`, `noise` and `op`. A registered metric
can infer the analysis when exactly one declared analysis supplies its signal
kind; otherwise binding fails with `MetricBindingError`. For instance,
`average` accepts either plot in a tran+AC bench, so it needs a selection.
Explicit probes also make standalone `.y` reads and plots unambiguous. For the
declarative form, put `analysis=` on `add_metric`; for a callable, put it on
`ckt.load_signal(...)`. See [multi-analysis tests](multi-analysis-tests.md).

For an AC Bode magnitude plot use `v_freq.plot()` after the run: `Signal.plot`
handles complex values and a log-frequency axis. Ordinary single-run `tb.plot()`
panels use the same Bode rendering; retained-corner overlays use their separate
corner plotting path. The multi-probe example above uses transient signals.

## Derive metrics from reported values

`add_metric` returns a `MetricHandle` in every form. Arithmetic (`+`, `-`,
`*`, `/`, `**`, unary minus and `abs`, including numbers on either side)
builds an expression. Register that expression to report it and check a spec.
For a wired charge-pump bench with its output held by `VoltageSource("vout", ...)`
and a transient spanning both current windows:

```python
from silk.metrics import average

i = tb.probe('vvout', 'I') * 1e6
up = tb.add_metric('up_uA', average, sig=i.window(0.3e-6, 0.9e-6).mag())
down = tb.add_metric('down_uA', average, sig=i.window(1.3e-6, 1.9e-6).mag())
mismatch = tb.add_metric('mismatch_pct', 100 * abs(up - down) / ((up + down) / 2))
mismatch.add_spec('current matching', 'mismatch_pct', '<', 10)
report = tb.run()
print(mismatch.value)  # same number as report.metrics['mismatch_pct']
```

See [charge_pump_example.py](../silk/examples/charge_pump_example.py) for the
complete circuit and testbench. Inputs are measured once per evaluation; derived metrics consume those reported
values, in registration order. Handles in an expression must belong to the same
testbench. Scalar division by zero (including Python and NumPy signed zero)
produces NaN regardless of NumPy's divide warning/error mode. NaN inputs
propagate to NaN, which fails a spec. A failed input
extractor makes its dependent metrics fail with an error, without re-measuring.

`.value` reads the latest completed evaluation,
not a saved corner snapshot; retain the report for a particular corner.
Reading it before evaluation raises. `bool(handle)` raises `TypeError` with a
run-first message; compare `handle.value` after the run. Equality and hashing
use object identity; ordering comparisons are unsupported. Public testbench
methods forward through handles, so `.add_metric(...).add_spec(...)`,
`.add_source(...)` and other builder chains still work. Derived registrations
survive composition and pickling under the same restrictions as their input
metrics. A deep copy starts without measured handle values.

A handle returned by `selected_tb.add_metric(...)` keeps that selected receiver:
`.value`, arithmetic and forwarded calls such as `.evaluate()` and `.run()`
use its placement, including inherited operating conditions. Combining selected
handles requires the same hierarchy root, owner path and authored test. Register
selected expressions and specs on that matching selected test. Authored handles
remain usable for shared declarations; derived recipes store metric names and
evaluate with each placement's local values.

Selected metric handles have the same lifetime and copying limits as selected
tests. Keep the root alive. To copy a selection, deep-copy the root and handle
together and retain the copied root. Once a selected root is collected, or its
placement is removed, placement-dependent use raises `ConfigurationError` rather
than reading another placement's values. **Breaking:** pickling a selected metric handle is refused,
even when included with its root, just as for the selected test. Ordinary
authored-handle copying and pickling retain their existing behavior.

## Keep and select corner waveforms

Use `.point((corner, temperature, supply))` for deterministic grids, and
`.sample((corner, index))` or `.sample((point, index))` for Monte Carlo.
To extend the runnable RC example, sweep the corners:

```python
reports = tb.run_corners(['tt', 'ff', 'ss'])
print(tb.sweep_corners)             # corner -> retained rawfile path
tt = vout.corner('tt')              # bound Signal for one retained corner
print(tt.y[-1])
vout.plot()                        # overlay all retained corners
tb.plot(vin, vout, corners=['tt', 'ss'], show=False)
```

An ideal RC has no process-dependent devices: these corner traces coincide.
On a PDK-backed design, the same workflow selects the PDK's process models.

The corner name is a **label** (the keys of `sweep_corners`, `SpecReport.corner`,
the output folder), not evidence of what ran. This RC has no PDK device, so
silk records its corner as **none** in `report.operating_conditions`
(`corner_kind == 'none'`, `corner is None`) and `run_corners` with distinct
corners emits one `UserWarning` that the runs are identical. A `None` corner
on a snapshot means none *or* unknown: read `corner_kind`, and never fill it
from the label. On a PDK-backed design the corner is a value, basis
`configured-and-emitted`. `silk.simulation.has_no_pdk_device(module)` answers
the PDK-free question before a launch. See
[operating conditions](api-reference.md#operating-conditions-none-unknown-and-the-consumer-contract).

Each corner gets its own output folder and an entry in `sweep_corners`,
for both serial and parallel runs (`run_corners(..., n_workers=N)`),
regardless of `set_output_options`. With several corners
retained, bare `vout.x`/`.y` raises `UnboundSignalError` and `vout.bound` is
false until you select a corner.
`vout.corner('tt')` returns a bound Signal; it does not change the original
probe or the bench's current results. Derived expressions can select a corner
in exactly the same way. Keep the rawfiles if you want to read them later.

## Selecting a placed test

Use the hierarchy root to select a test's placement before declaring probes:

```python
selected = top.get_test('step', instance='amp.bias')
vout = selected.probe('out')
print(vout.y[-1])          # this owner's latest retained run
print(selected.metrics)
```

`top.list_test_entries()` supplies these paths and selected handles without
running anything. The local `dut.get_test('step')` remains the original bench.
After `run_all_tests()`, it reads normally when its latest invocation had one
distinct simulation. If inherited conditions produced different runs, numeric
reads raise `UnboundSignalError` naming the selections; `vout.plot()` and
`tb.plot()` overlay the distinct runs with placement and corner/sample labels.
Aliases sharing a simulation contribute one trace with all their labels.

Ordinary runs default to a fresh `run-.../` folder; select
`set_output_options(mode='overwrite')` explicitly to reuse fixed paths.

Selected probes retain loaded run data even when another root or placement
reuses the same overwrite-mode filename. Each selection keeps its latest run,
not unlimited history. A request refused before result replacement preserves
its report, metrics, corner/sample selection and bound probes. This includes
invalid arguments to `run()` or `evaluate(rawfile=...)`; `prepare()` does not
replace results either. Once replacement starts, a failed run or explicit file
load clears current readback. A worker error with unknown replacement state
does not by itself clear evidence. Successful replacement exposes the new run,
including its waveform when a metric reports an extraction error. Previously
held reports and stored history survive these live readback transitions. Keep
both the hierarchy root and selected handle alive while using selected probes.

If batch evaluation stops on a configuration error, probes reflect confirmed
result replacements; other probes retain their previous results.
An empty `run_corners()`
request returns an empty collection and leaves the retained evidence together.

## Freezing data with `snapshot()`

`vout.snapshot()` returns an ordinary Signal holding the samples as they are
now. It takes the same optional arguments as a read (`module`, `analysis=`,
`corner=`), and `Signal.resolve()` remains as an alias with identical
behaviour. There are two reasons to use it:

1. **Keep an earlier run while re-running the same bench**, for example in a
   loop over a parameter: `runs.append(vout.snapshot())` before the next `run()`.
2. **Keep data after the bench is gone**, for example returning results from a
   function whose bench is local. A probe does not keep its bench alive.

Reading `.x`/`.y`, metrics (`add_metric(..., sig=vout)`) and plots never need
it; they read the bench's current results directly.

## Result lifetime and limits

- Before a run, reading probe data raises `UnboundSignalError`; declaring
  arithmetic or registering a metric does not require samples yet.
- A probe follows the bench's results at each read. Use `snapshot()` (see
  above) to obtain a plain Signal holding the current samples, or `corner(name)` to select retained
  samples. Use `.point((corner, temperature, supply))` for deterministic grids
  and `.sample((corner, index))` or `.sample((point, index))` for Monte Carlo.
  None creates a persisted run archive. The corner table describes
  only the latest sweep; a subsequent single `run()` or `evaluate()` replaces it.
- Keep the bench alive: probes hold weak references to it.
  `tb.plot()` draws all registered probes; pass derived expressions explicitly.
- Report metrics are captured numbers. Hierarchy reports retain readers for
  their run data; ordinary unselected direct reports retain their existing
  live-Module behavior. A retained reader is not an independent circuit.
- Report caching is disabled: `ModuleTestbench(cache=True)`
  emits a `UserWarning` at construction and then behaves exactly like
  `cache=False`: every `run()` simulates. A report holds the metrics from
  its evaluation; a probe follows the bench's latest results.

The embedded AC/transient/noise workflow in
[`ota_5t_guide.py`](../silk/examples/ota_5t_guide.py) shows probes in a transistor
design; [agent-guide.md](agent-guide.md#4-worked-spine--silkexamplesota_5t_guidepy)
walks through its construction. Existing `load_signal()` code remains supported.

### Copying a checked viewer node

A circuit-backed viewer may copy `tb.probe('v(dut.x1[0].Mid)')`, preserving
case and literal punctuation. That code is for the bench/result reader that
produced the displayed data. AC copies retain `.mag()` or
`.phase_deg_unwrapped()` and multi-analysis copies include `analysis=`.
Designer spelling is used only when it uniquely resolves to the exact source
row; otherwise copy uses a proven emitted spelling or refuses. Hidden duplicate
rows and unsaved competing nodes cannot make an ambiguous name safe. A label
alone never establishes probe identity. Colon and dot are hierarchy separators
in the viewer; authored-name validation belongs to the naming contract, not
viewer inference.
