# One test, several analyses

A silk test may declare more than one analysis. The short snippets below
assume a testbench named `tb` and imported analysis classes; the
[complete example](#a-complete-example) supplies both. Metric snippets show
alternative bindings, so use each on a fresh testbench:

```python
tb.add_sim_type(Transient(step=1e-8, stop=1.6e-5))
tb.add_sim_type(AC(variation='dec', points=10, fstart=1, fstop=1e9))
```

Both run, both land in the rawfile, and a metric says which one it reads.
Use ngspice for this workflow. Xyce accepts one analysis per test; split
analyses into separate tests when using that backend.

## The rule: silk never guesses the plot

For an ordinary single-analysis test, omit `analysis=`. With several
analyses, select the result explicitly whenever more than one could supply
the metric input.

With more than one, silk asks what each metric's signal inputs actually need.
Every metric declares the **kind** of Signal each input takes, and every kind
declares which analyses produce it (`produced_by` in
[`schema-export.md`](schema-export.md)). Given the analyses this test
declares:

| how many declared analyses could feed the metric | what you write |
| --- | --- |
| exactly one | nothing — silk reads it |
| more than one | `analysis='…'`, and omitting it **raises** |
| none | add a compatible analysis; see the validation limits below |

So in a `Transient` + `AC` test:

```python
# dc_gain_dB takes a 'frequency_response', which comes from .ac and from
# nothing else in this test. No field.
tb.add_metric('gain', type='dc_gain_dB', signal='vout')

# average takes 'any' Signal, so BOTH plots could feed it. Say which.
tb.add_metric('vavg', type='average', signal='vout', analysis='tran')
```

Leave the second one off and the run refuses rather than measuring something:

```
ValueError: add_metric('vavg', type='average'): 'vout' could come from more
than one analysis in this test: 'ac', 'tran'. Add analysis='ac' or
analysis='tran' to the metric.
```

The metric-schema analysis names are `'tran'`, `'ac'`, `'dc'` and `'noise'`.
Operating-point results can also be selected with `analysis='op'`; import
`Op` from `silk.simulation` to register that analysis.

**A declarative metric selects one plot for its signal inputs.** To also average the AC
magnitude, bind a magnitude projection under its own name:

```python
from silk.metrics import average
tb.add_metric('ac_magnitude_avg', average, sig=tb.probe('vout', analysis='ac').mag())
```

One registration does not automatically evaluate the metric once per plot.

**`domain` is not consulted.** A metric's `domain` is its home/grouping label
(`average`'s is `'tran'`), and grouping is not the same question as which plot
holds the data — `average`'s *input kind* is `any`. Selection reads kinds.

## The probe form

`add_metric`'s function form has no `analysis=` of its own: the **probe**
carries it, because the probe is the thing that names a signal. See the
[practical probe guide](probes.md#more-than-one-analysis) for selecting the
analysis read by each probe.

```python
from silk.metrics import average, dc_gain_dB

vout_t = tb.probe('vout', analysis='tran')   # kind 'any': say which
tb.add_metric('vavg', average, sig=vout_t)

tb.add_metric('gain', dc_gain_dB, sig=tb.probe('vout'))   # only .ac makes one
```

Each signal slot selects an analysis from its input kind; ambiguous slots raise
`MetricBindingError` and identify the required probe selection:

```
MetricBindingError: add_metric('vavg', average): 'vout' could come from more
than one analysis in this test: 'ac', 'tran'. Name the analysis on the probe:
tb.probe(..., analysis='ac') or tb.probe(..., analysis='tran').
```

It is a `MetricBindingError` (a `ConfigurationError`) rather than the
declarative form's `ValueError` so that it propagates through
`run_all_tests()` like every other binding failure instead of becoming that
test's `{'error': …}` entry.

A probe that names an analysis reads that plot everywhere — `vout_t.y`,
`vout_t.snapshot()` and `tb.plot(vout_t)`. `tb.plot()` draws signals of
different analyses as separate panels, each with its own x-axis;
`overlay=True` refuses a mixed set, because one axis cannot be seconds and
hertz at once. A probe that names none, read
directly off a multi-plot run, meets the loader's refusal below. An expression
over probes (`(vp + vm) / 2`) is settled once for the whole recipe from its
slot's kind, and any leaf that named its own analysis keeps it.

Passing `analysis=` to `add_metric` itself alongside a function or a lambda
raises and names the place that does take it.

## The callable form

`add_metric`'s callable form selects a plot where it loads the signal:

```python
from silk.metrics import average

tb.add_metric('vavg', lambda ckt: average(
    ckt.load_signal('vout', analysis='tran')))
```

`analysis=` on `add_metric` itself is for the declarative form only; passing
it with a lambda raises and says this.

The same keyword works anywhere you hold the module, including after the run:

```python
report = tb.run()
vout_t = tb.netlist.load_signal('vout', analysis='tran')
vout_f = tb.netlist.load_signal('vout', analysis='ac')
```

Attribute access on a subcircuit instance (`inst.vout`) has no place for the
keyword, so on a multi-plot rawfile it hits the same refusal by design; the
way through is `ckt.load_signal(inst.path('vout'), analysis=…)`.

## A complete example

An RC low-pass measured in the time domain and the frequency domain from one
test and one simulation. This uses ideal components: ngspice is required,
but installed PDK model files are not. Run from a writable working directory:

```python
from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.simulation import AC, Transient

pdk = load_pdk('sky130')

rc = Module('sky130', 'silk-output/multi-analysis', 'rc.cir', pdk_details=pdk)
rc.define_module_name('rc_lpf')
rc.define_module_ports({'vin': 'input', 'vout': 'output'})
rc.add(elements.resistor('r1', p='vin', n='vout', value=1e3))
rc.add(elements.capacitor('c1', p='vout', n='0', value=1e-9))

tb = rc.create_test('rc')
tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
tb.add(sources.vpulse('in', positive='vin', negative='0', v1=0, v2=1,
                      td=0, tr=1e-9, tf=1e-9, pw=4e-6, per=8e-6, ac=1))

tb.add_sim_type(Transient(step=1e-8, stop=1.6e-5))
tb.add_sim_type(AC(variation='dec', points=10, fstart=1, fstop=1e9))

tb.add_metric('gain', type='dc_gain_dB', signal='vout')                 # ac
tb.add_metric('vavg', type='average', signal='vout', analysis='tran')   # tran

report = tb.run()
print(report.metrics)

tb.netlist.view(open=False)  # writes the explorer HTML with a plot switcher
```

Expect a gain close to 0 dB and an average voltage close to 0.499 V.

The relevant parts of the emitted deck are shown below (`...` omits other
control commands, and `<dir>` stands for the test output directory):

```
.tran 10n 16u
.ac dec 10 1 1G
.control
save all
run
...
write <dir>/rawfile.raw tran1.all ac1.all
quit 0
.endc
```

`tran1` / `ac1` / `dc1` / `noise1` are nutmeg's own names for the plot each
analysis leaves behind, and the **argument order sets the order of the plots
in the file** — so the rawfile reads in the order you declared the analyses,
not the order ngspice ran them in (it runs `.ac` before `.tran` whatever the
deck says). A deck with one analysis uses the unqualified `write <path>` form,
unless `save_device_params` adds an operating-point plot and qualifies the write.
The sole declared non-op analysis remains the default signal read.
Operating-point plots use `op1`.

## What it looks like when it goes wrong

Analysis selection is checked at these points:

| where | when |
| --- | --- |
| `generate_testbenches` | a declarative `TestbenchSpec` — refused before any Python is emitted (`TestbenchSpecError`) |
| `run()` / `evaluate()` | a hand-built test, declarative (`ValueError`) or probe (`MetricBindingError`) — checked **before** extraction, because the extraction loop turns exceptions into report strings |
| `load_signal` | a multi-plot rawfile without a named analysis, or multiple plots matching the named kind (`SimulationError`) |

```
SimulationError: this rawfile holds 2 analyses, so a signal cannot be loaded
without naming one: 'Transient Analysis' (tran), 'AC Analysis' (ac). Pass
analysis='<kind>' to load_signal (or to add_metric) — silk does not guess
which analysis a metric meant.
```

The code generator rejects a metric whose input kind has no compatible
declared analysis. A hand-built test does not always reject that mismatch
before running: with no explicit `analysis=`, selection can reach the loader
or metric extraction instead. Add the analysis required by the metric rather
than relying on that validation.

## Multiple plots of the same kind

High-level signal and probe reads require exactly one plot matching the named
kind: `tran`, `ac`, `dc`, `noise` or `op`. Custom or concatenated ngspice
rawfiles and repeated Xyce step plots can contain several matches.
Selection among multiple plots of that same kind is not currently supported
by silk and is a future extension. Save the intended result in a separate
rawfile. For example:

```text
SimulationError: this rawfile holds 2 'tran' plots. Selection among multiple
plots of the same kind is not currently supported by silk and is a future
extension. Save the intended result in a separate rawfile.
```

Equal waveform values do not remove ambiguity. A noise spectral-density plot
and an integrated-noise plot both count as `noise`. A unique AC result beside
two transient plots can still be read with `analysis='ac'`. Metric-slot
resolution wraps the selection error in `MetricBindingError`, retaining the
original error as its cause.

Rawfile loading preserves every plot in file order. The viewer lets you
switch between them by plot index. Low-level code
can choose a `RawPlot` explicitly. Continuing the RC example:

```python
from silk.signals.rawfile import read_rawfile
from silk.signals.spice import NgspiceSignal

raw = read_rawfile(tb.netlist.results_file)
sig = NgspiceSignal.from_rawfile(raw, 'vout', plot=raw.plots[0])
```

This `plot` argument belongs to the low-level reader; high-level probes select
by analysis kind. Ordinary native analysis registration remains one entry per
analysis class.

## Limits

- **One of each.** A test holds at most one Transient, one AC, one DC, one
  Noise and one Op — `add_sim_type` files an analysis under its class, so
  adding a second Transient replaces the first.
- **Xyce runs one analysis per deck.** silk raises `SimulationError` before
  writing or launching a multi-analysis Xyce deck. Split the analyses into
  separate tests, or select ngspice; see
  [known issues](known_issues.md#simulator-backends).
- **Incomplete results are rejected.** A missing named plot can prevent
  ngspice's qualified write, but an analysis that ends early can leave partial
  data in a rawfile. Certified silk runs reject incomplete results with
  `SimulationAborted` before publishing them. See the
  [analysis-completion limits](known_issues.md#analysis-completion) for
  external files and opaque requests.
- **`save_device_currents`** requires exactly one Transient analysis and is
  refused on a multi-analysis deck (it emits no `.control` block at all).
