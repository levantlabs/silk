# Monte Carlo

silk runs a compatible registered test as a Monte Carlo campaign (device mismatch,
process variation, or both) through the same machinery as corners, then
analyses it: statistics, yield with intervals, the worst sample, per-sample
re-run, a ranking of the drawn device parameters, and an HTML section.

Worked example: [`silk/examples/mc_diff_pair_offset.py`](../silk/examples/mc_diff_pair_offset.py)
(input offset of a matched pair at `tt`/`ss`/`ff`, 50 samples each).
Signatures: [api-reference.md](api-reference.md).

## The idea

A campaign runs the same test repeatedly with random parameter values. Each
sample represents one manufactured circuit. Register metrics and specs on the
testbench first; silk then reports statistics and yield across the samples.
With corners declared, it repeats the campaign at each corner.

For SPICE, variation comes from the PDK's statistical models. For behavioral
simulation, declare variation on the model's parameters with `Normal` or
`Uniform`; no external simulator or installed PDK models are needed.

## Behavioural models: declaring variation

This complete example runs without a simulator. It models a fixed input offset,
measures it at the last transient point, and checks a 4 mV limit. Run it from
your working directory; results go under `silk-output/mc-offset/`.

```python
from silk import Module, Normal, PortDir, sources
from silk.metrics import final_value
from silk.simulation import Transient


def offset_model(ports, t, params):
    ports.vout.v = ports.vin.v + params.offset


dut = Module(None, 'silk-output/mc-offset', 'offset.cir',
             pdk_details=None, module_name='offset')
dut.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
dut.set_behavioral_model(offset_model,
                         params={'offset': Normal(0.0, 3.2e-3)},
                         execution='vector')
dut.simulator = 'behavioral'
tb = dut.create_test('offset')
tb.add_module_under_test({'vin': 'vin', 'vout': 'vout'})
tb.add_source(sources.VoltageSource(
    'input', positive='vin', negative='0', dc=0.0))
tb.add_sim_type(Transient(step=1e-9, stop=10e-9))
offset = tb.add_metric('offset', final_value, sig=tb.probe('vout'))
tb.add_metric('abs_offset', abs(offset))
tb.add_spec('offset limit', 'abs_offset', '<', 4e-3)
tb.set_monte_carlo(200, seed=7, record_draws=True)
mc = tb.run()
print(mc.summary())
print(mc.stat('offset', 'std'))
```

For an existing comparator model, attach several declarations like this
(`comp` and `comparator_model` must already exist):

```python
from silk import Normal, Uniform

comp.set_behavioral_model(comparator_model, params={
    'offset_voltage': Normal(nominal=0.0, sigma=3.2e-3),
    'gain_error': Normal(nominal=0.0, sigma=0.01,
                         variation='process', name='comparator_gain'),
    'hysteresis': Uniform(low=0.5e-3, high=1.5e-3),
    'vdd': 1.8,
}, execution='vector')
```

Every ordinary behavioural run passes nominal floats to the model: `0.0`,
`0.0`, and `0.001` for these declarations. Values resolve once per run and hold
across transient and DC analyses. The module keeps the declarations; runs do
not overwrite them or their `Variable` fields. Models with declarations get a
parameter view over a fresh resolved dictionary per instance per run; models
without declarations read their attached live dictionary through the same view.
Store callable state on the model or in its closure.

Mismatch is the default. A process declaration names a shared source and must
have a nonempty string name. Two executed blocks declaring that name with
different kinds or resolved fields raise `VariationError` before any model
runs, including in nominal simulations. `Variable` and expression fields are
read at run preparation; invalid fields raise `VariationError`, and unsupported
field types raise `TypeError`.

Declarations must be top-level params values. Nesting them inside containers,
using them as SPICE device/source/waveform values, or calibrating a whole declaration is
refused. Calibration can target an individual distribution field, as shown
[below](#calibrating-a-behavioral-declaration-from-a-campaign). A block with both a behavioural model and SPICE contents can still run
its SPICE circuit; its model params are unused in that simulation.

Behavioural tests use the same campaign calls, with no installed PDK models
or external simulator:

```python
# Attach declarations to the DUT's model parameters before placing it.
dut.simulator = 'behavioral'
tb.set_monte_carlo(n=200, seed=7, variation='both')
mc = tb.run()                         # or dut.run_all_tests()[...]
print(mc.stat('offset', 'std'))       # use a metric registered on this bench
```

`tb.run_monte_carlo(200, seed=7, variation='both')` runs a one-off campaign.
`variation='mismatch'` draws independent values per physical instance;
`'process'` draws one shared value per named source; `'both'` draws both.
Unselected declarations stay nominal. `block.set_mismatch(False)` before
placement keeps mismatch nominal throughout that hierarchy; process still
draws. No active declaration (including zero spread or all mismatch disabled)
raises `MonteCarloError` before the first sample.

Each value is a deterministic function of the full base seed, sample index and
instance path or process name (draw scheme 1). Adding unrelated blocks does
not shift other draws. Root and child tests invoked from the same root agree
on the same physical instance when their seeds match; a block run independently
is its own root. Bench helpers have a separate test-specific namespace.
`run_all_tests` checks conflicting process definitions across all participating
behavioural tests before simulating any test, even nominal ones.

Resolved declaration fields are frozen when a campaign is prepared. One sample
holds its values across every DC point and every analysis. Behavioural samples
run in-process even with `n_workers=2`. Corners repeat the same chips under
different labels and issue the identical-corners warning. Every sample writes
to `<corner>/mc_NNNN/` with the overwrite default, or
`<corner>/run-.../mc_NNNN/` with `output_mode='new'`, and records no
design-stage measurement on the module's `Spec` objects.

Models must be deterministic: randomness or external mutable state inside a
callable is outside this guarantee. The digest mapping reaches approximately
±8.2 sigma for Normal tails. These are fixed per-run variations, not
time-varying noise. For field calibration, see
[Calibrating a behavioral declaration from a campaign](#calibrating-a-behavioral-declaration-from-a-campaign).
See the [declaration API](api-reference.md#silksimulationvariation--behavioural-parameter-declarations).

With `record_draws=True`, behavioural campaigns record actual parameter values
(`mc.draws.raw is True`), with one column per active mismatch declaration per
instance, such as `('dut.c0', 'offset')`, and one `('process', name)` column per
active shared process source. Each corner gets a `draws.csv`. A bench mismatch
column that collides with a process column is refused before sampling.

Sensitivity rows use names such as `dut.c0:offset` and `process:shift`.
The default `by='parameter'` keeps each process source separate;
`by='device'` combines all process sources into one row named `process`.

Behavioural `mc.rerun(key)` recomputes draws with or without `record_draws`.
A snapshot of the simulation inputs, captured before the first sample, guards
replay with a
`MonteCarloError` naming a changed model or plain parameter, declaration
(including a numeric or `Variable` sigma), effective mismatch setting,
draw scheme, analysis, source, condition or backend. This snapshot is kept in memory only. Replay assumes
deterministic models: captured closure values,
globals and other external Python state are outside the identity contract.

## The workflow

For an existing SPICE module `mod`, first build the testbench's sources,
analyses, metrics and specs. The following outline assumes a module named
`pair` and metrics named `offset_V` and `abs_offset_V`:

```python
from silk import generate_report

tb = mod.create_test('offset')
...                                             # sources, analysis, metrics, specs
tb.set_corners(['tt', 'ss', 'ff'])              # optional: the corner list
tb.set_monte_carlo(50, variation='mismatch',   # declare the campaign
                   seed=0, record_draws=True)

results = mod.run_all_tests(n_workers=4)        # corners x samples
mc = results['pair.offset']                     # a MonteCarloResults

print(mc.summary())                             # run / failed / used per corner, yields
mc.yield_fraction()                             # {spec name: (fraction, (lo, hi))}
mc.corner('ss').stat('offset_V', 'std')         # one corner's sigma
mc.percentile('offset_V', 5)
mc.worst('abs_offset_V')                        # value, corner, sample index
mc.corner('tt').sensitivity('offset_V').rows[:3]  # per corner; needs record_draws=True
mc.table(); mc.to_csv('samples.csv'); mc.plot_histogram('offset_V')
mc.failed                                       # samples that produced no result
mc.rerun(('ss', 17))                            # re-simulate one sample from its seed

report = generate_report(mod)
report.add_monte_carlo(mc)                      # yield, histograms, sensitivity, failures
report.save_html('mc.html')
```

`tb.run_monte_carlo(n, ...)` runs once with the arguments you give and leaves
the declared setting alone. `tb.run()` on a declared test returns the same
`MonteCarloResults`, running serially; `run_all_tests(n_workers=N)`
can run up to N SPICE samples (or chunks, below) concurrently. Behavioral
samples always run in-process.

## Operating grids

The declared plan accepts temperature and supply lists alongside corners:

```python
ota.set_temperatures([-40, 27, 125])
ota.set_supply_voltages([1.62, 1.8, 1.98])
ac = ota.get_test('ac').set_corners(['tt', 'ss', 'ff'])
ac.set_monte_carlo(2, seed=17, min_yield=0.95, record_draws=True)
mc = ac.run()                        # 27 points, two samples each
point = ('tt', 125.0, 1.62)
report = mc[(point, 0)]
waveform = ac.probe('vout').sample((point, 0))
sigma = mc.point(point).stat('gain_dB', 'std')
measurement = mc.point(point).measurement('gain_dB', 'mean')
mc.to_csv('ac_pvt_mc.csv')
```

If the supply is ambiguous, pass `source='name'` to `set_supply_voltages`;
the source is resolved separately for each executing test.

The grid key is `(corner, temperature, supply)` and a sample key is
`(point, index)`. An unspecified other axis uses its effective scalar setting,
or `None` when unspecified. Without temperature or supply lists, keys remain corner strings
and `(corner, index)` pairs. `mc.corner('tt')` selects the physical process
corner across all its points; `mc.corners` lists physical corner strings.
`mc.point(point)` selects one full point, including samples that produced no
report. Measurement and sensitivity require exactly one planned full point;
use `mc.point(point).sensitivity(metric)`. Captured condition-consistency checks
still apply. Aggregate measurement attribution keeps a physical corner string.
`worst(metric).corner` carries the full point tuple for grids and the original
string for corner-only results.

`yield_by_point()` maps each point to its per-spec fraction and Wilson interval.
**Every point must meet `min_yield`**, including each corner of a corner-only
campaign. A good point cannot hide a bad one. `yield_fraction()` remains pooled:
it counts passing point/sample evaluations over all planned evaluations,
including failed samples in the denominator. It is informational and does not
determine the grid verdict. It is not the probability that the same device
passes every condition. Seeds repeat by sample index across points, so these
are not additional independent device draws; the existing pooled Wilson bounds
are not an all-PVT manufacturing confidence interval. `stat`, `percentile` and
histograms remain pooled; use a point view for per-point statistics. Standard
deviation is sample standard deviation (`ddof=1`).

`summary()` shows each point's yield verdict, informational pooled yield and
scalar metric statistics with complete worst-point coordinates. Counts show planned samples, reports and failures, for example `planned=4: 3 reports (2 PASS, 1 FAIL), 1 failed and
produced no report`. Failed points and unavailable measurements never disappear
from coverage or yield denominators. A wholly failed point remains selectable
when another point succeeded; an entirely failed plan still raises
`SimulationError`. Structured and complex metric outputs remain in the table;
only suitable real scalar outputs receive numeric summaries.
Behavioral simulation supports supply grids but refuses ambient-temperature
settings; model temperature effects through a behavioral parameter.

`table()` and `to_csv()` use namespaced columns throughout, including corner-only
Monte Carlo tables: `run.corner`, `run.temperature`, `run.supply`,
`run.supply_source`, `run.sample`, `run.seed`; captured `observed.*` value,
`*_state` and `*_basis` fields; temperature/supply `*_sweep.start/stop/step`;
`observed.supply_source`; `run.status`, `run.error`; `metric.<registered output>`;
and `check.<stable spec key>`. The check's display name is retained as metadata,
so identical display labels with distinct keys remain distinct in stored tables.
Live Monte Carlo campaigns still refuse duplicate spec display names. `run.status`
is `PASS` or `FAIL` for a completed report and lowercase `failed` for a run that
produced no report. Check cells use captured `PASS`, `FAIL`, or `UNAVAILABLE`
evidence; a failed boolean alone cannot identify an unavailable measurement.

Fixed observations have a value and state `fixed`; explicit unknown/none
observations have an empty value and state `unknown`/`none`. Sweeps have an empty
scalar value, state `sweep`, and their captured bounds. Bases and source identity
come from the captured record. If no report was captured, all observation cells
are empty, including state; this differs from captured unknown evidence. CSV
quotes names normally and leaves missing values empty. Rows and columns retain
plan/declaration order independently of subsequent bench edits.

A point campaign uses folders like
`tt/temperature-125.0/supply-1.62/mc_0000/`; new output mode inserts `run-.../`
between the supply and sample components. Float components use readable
round-trip spellings, `unset` means `None`, and existing path handling escapes
unsafe characters. Each point has its own wrapper and `draws.csv`, beside its
samples. Grid draw files use separate `corner,temperature,supply,sample,seed`
columns before parameter columns; corner-only draw files keep their original
format. Chunks, retries and replay stay within a full point. Replay refuses
changes to effective axes and source selection as well as existing inputs.

## Variable lists and grouped run plans

Declare lists on a testbench when sources, device parameters or behavioral
model parameters share a `Variable`. Every reference and expression follows
its value for that point. For example, with Variables named `vdd`, `f_clk`,
`b0`, `b1` and `b2` already wired into the circuit:

```python
tb.set_corners(['ss', 'ff'])
tb.set_temperatures([-40, 125])
tb.set_variable_values(vdd, [1.62, 1.98])
tb.set_variable_values(f_clk, [0.6e9, 1.2e9, 2.4e9])
tb.set_variable_values(b0, [1, 1, 1])
tb.set_variable_values(b1, [1, 0, 1])
tb.set_variable_values(b2, [0, 1, 1])
tb.set_groups([['corner', 'temperature', vdd], [f_clk, b0, b1, b2]])
results = tb.run()  # two operating rows crossed with three frequency/trim rows
```

This makes six runs, in order: all three frequency/trim rows at `ss`, -40 °C,
1.62 V, followed by the same three rows at `ff`, 125 °C, 1.98 V. Trim codes are
3, 5 and 7. A group steps its lists together without deduplicating entries.
Groups cross with one another and with ungrouped lists. The axis order is
corner, temperature, supply, then Variables in declaration order; each group
occupies its earliest member's position, and later axes vary fastest.
Replacing a list keeps its position; removing and redeclaring a Variable
appends it. `set_variable_values(var, None)` clears that list;
`set_groups(None)` clears the groups.

All lists in one group must have equal lengths. A group member needs a declared
list, and a list can belong to only one group. Setters validate individual
entries immediately; planning checks the complete experiment before any deck
is written. Ungrouped repeats and repeated whole runs refuse. Grouped
`run(corner='ss')` and `run_all_tests(corner='ss')` select all complete matching
rows and refuse an absent corner. `prepare(corner=...)` requires exactly one
deterministic point; Monte Carlo and multiple decks still refuse.

Numeric Variable values must be finite real scalars, excluding booleans. They
use the Variable's existing bounds and resolution. Categorical entries must be
members of its categories and JSON scalars: strings, booleans, finite integers
or floats. Unset entries, complex numbers and container/object categories
refuse. Coordinates use effective snapped values, and Python key equality
applies: `True`, `1` and `1.0` compare equal; the string `'1'` is distinct.

Variable plans use keys `(corner, temperature, supply, ((name, value), ...))`,
where the supply coordinate remains the configured scalar/list value, or
`None` if a Variable alone drives the rail. `observed.supply` records the
verified voltage the Variable gives an identifiable supply source. A supply
setter that would overwrite that same Variable-driven source is refused,
including expressions such as `dc=2*vdd`. Unknown or ambiguous supply sources
retain their usual rules.

Each run keeps its own report. Tables, CSV and report grids include a
`run.<variable name>` column for every swept Variable, including failed and
Monte Carlo sample rows. Select retained waveforms with the complete point
key, or `(point, sample_index)` for Monte Carlo. Samples expand inside each
point, retaining the same sample-index seed scheme. Replaying a sample checks
the entire declared experiment for changes.

Authored Variable values and source bindings are restored after preparation,
simulation and evaluation, including failures. A run report keeps a detached
snapshot in `parameters['silk_run_variables']`; that parameter name is reserved
on Variable plans. Storing individual reports preserves this attribution in
the existing result conditions. Readback with `latest_only=True` retains each
distinct coordinate set. Producing-deck comparison leaves Variable-list plans
`not checked`, with a reason; it does not reconstruct their decks.
Scalar spec restoration refuses ambiguous Variable
points or an unprovable stored measurement link before changing any live
observations or currents. `run_test(store=True)` still refuses declared plans;
store each returned report separately instead. `run_corners()` ignores the
declared lists/groups and leaves them intact.

Saved collections containing groups or Variable coordinates use test-results
JSON v4, including PVT-only groups, nested collections and detached round trips.
Older readers refuse that version. Plans without these declarations keep their
existing keys, folders, columns and saved encoding.

## Campaign folders

A campaign overwrites by default even when the module writes new folders.
Use `tb.set_monte_carlo(200, seed=1, output_mode='new')` (or the one-off
`run_monte_carlo(..., output_mode='new')`) to keep each campaign in its own
`<testbench dir>/<corner>/run-.../` folder. This folder holds the samples,
`draws.csv` when `record_draws=True`, and `mc_both.lib.spice` for SPICE
campaigns with `variation='both'`. `output_mode` accepts `'overwrite'` and `'new'`; other
values raise `ValueError`.

**Optimization:** `Optimizer.evaluate()` and
`Optimizer.sweep()` refuse a bench declaring `set_monte_carlo(...)` with
`NotSupportedError` before any point is applied, deck is written or simulation
launches. They consume one report per point, not Monte Carlo collections.
Declared `set_corners(...)` collections are refused too, even for one corner.
Clear the declarations with `set_monte_carlo(None)` / `set_corners(None)` /
`set_temperatures(None)` / `set_supply_voltages(None)`, including inherited
declarations, to optimize ordinary runs. Yield optimization is not supported.

## Several samples in one ngspice process

By default every sample runs in its own ngspice process, and each process loads
the model library first. For short simulations, repeated library loading
can dominate the runtime. `samples_per_process=s` (a positive int, default 1,
on `set_monte_carlo` and `run_monte_carlo`) lets up to `s` consecutive samples of one corner share a
process, which loads the library once:

```python
tb.set_monte_carlo(200, seed=0, samples_per_process=50)
mod.run_all_tests(n_workers=4)         # 4 processes of 50 samples each
```

- **Per-sample results.** Each sample keeps its own folder, `rawfile.raw`,
  `output.log` and seed. Grouping samples changes execution, while retaining
  the per-sample analyses, metrics, draws and yield calculation.
  `mc.rerun(key)` runs one sample alone and still works after you change
  `samples_per_process`.
- **Chunks run in parallel.** A chunk never spans corners; `n_workers` chunks
  run at a time. `samples_per_process=200` makes one chunk, so `n_workers`
  then has nothing to parallelise: pick `ceil(n / s)` at least `n_workers`.
  A very long simulation may still prefer 1.
- **A failed sample** is recorded in `mc.failed`; the remaining samples in
  its chunk can continue. Missing data and incomplete analysis grids are
  simulation failures, just as in ordinary runs. Chunks additionally refuse
  ambiguous repeated plots because completion cannot be verified.
- **A chunk that dies or times out.** Samples it finished are kept; the first
  unfinished sample and every later one in the chunk are rerun once, each in
  its own process. They are listed in `mc.rerun_samples` (`(corner, index)`
  keys, empty when nothing was rerun; a rerun that failed again is listed too,
  and is also in `mc.failed`), shown in `summary()` and warned once. Ctrl-C
  starts no reruns. A Monte Carlo campaign passes no timeout today, so nothing
  stops a hung sample. A low-level timeout, when one is supplied, budgets the
  whole chunk (the timeout times its number of samples); it is not a
  per-sample watchdog. A sample the simulator reported as failed (a non-zero
  status) is a completed failure: it is never rerun.
- **The chunk's own files** (the loop deck, its transcript) are in
  `<testbench dir>/<corner>/.mc_chunk_<first index>/` in overwrite mode.
  With `output_mode='new'`, they are in the campaign's
  `<testbench dir>/<corner>/run-.../.mc_chunk_<first index>/` folder.
  They are kept like sample files.
  Each sample's `output.log` is the chunk's shared start-up text, that
  sample's segment, and (when the process ended normally) the closing banner.
  A sample finished before a crash records no simulator version.
- **A ceiling, not a demand.** Before writing anything, silk checks that the
  test's analyses are ones it can certify for completion (the same rules as an
  ordinary run, applied to the declared analyses). If not (no analysis, a
  repeated analysis, a sweep whose final point cannot be resolved such as
  `AC('lin', 2, 1, 1)`), one warning names why and every sample runs in its
  own process. An op-only test is checked against its emitted deck. If a sample
  deck still turns out not to be certifiable once emitted, that sample alone is
  run in its own process and listed in `mc.rerun_samples`; the warning
  keeps the reason.
- **Refused before any deck is written:** a simulator other than ngspice
  (`NotSupportedError` naming ngspice; Xyce and behavioral), custom
  netlist content or emitters, `save_device_params` and PDKs with OSDI models
  (`NotSupportedError`), and `save_device_currents` (`MonteCarloError`). silk
  never silently falls back to one process per sample.
- **Memory.** Preparing a chunk holds its samples' netlists at once, so it
  scales with `samples_per_process` times `n_workers`.

## What varies in SPICE

| `variation=` | Each sample | Library section |
|---|---|---|
| `'mismatch'` (default) | every device draws its own mismatch | the corner's mismatch section from the PDK manifest |
| `'process'` | one draw shared by every device | the PDK's process section, **only at the corners the PDK lists for it** (typical only on sky130) |
| `'both'` | both | silk writes `mc_both.lib.spice` in `<testbench dir>/<corner>/` in overwrite mode, or in the campaign's `<testbench dir>/<corner>/run-.../` folder with `output_mode='new'` |

Which sections exist, which switches turn them on and which model parameters
`record_draws` can read are PDK facts in the manifest's `library.statistical` block
([adding-a-pdk.md](adding-a-pdk.md)); none is in silk's code. A PDK without
the block raises `NotSupportedError` naming it; an install whose model library
lacks a needed section raises `MissingLibrarySectionError`, and
`python -m silk.doctor` warns about it beforehand.

## Rules worth knowing

- **SPICE seeds.** `seed=0` by default, never the clock. Sample *k* of a campaign
  with base seed *s* gets a seed derived from *s* and *k*, so the same call
  gives the same samples, and `mc.rerun(key)` re-simulates exactly one.
  `rerun` refuses with `MonteCarloError`, naming every changed managed input,
  before writing a wrapper, deck or sample file. See [Replay inputs](#replay-inputs).
  The simulators take a 31-bit seed, so in a very large campaign two samples
  can rarely share one ([known issues](known_issues.md#monte-carlo)).
- **Corners without Monte Carlo.** `set_corners` alone makes `run_all_tests`
  run each corner and return a `CornerResults`. A corner that fails is listed
  in `.failed` and skipped by the statistics with one counted `MetricWarning`;
  the run raises only when *every* corner fails.
- **Monte Carlo without corners.** The samples run at the test's corner.
- **Yield.** `yield_fraction()` counts a failed sample as failing every spec,
  and reports a 95 % Wilson interval. With `min_yield=0.95`, `passed` means
  every spec's yield at every operating point is at least that (per corner
  without temperature or supply lists). The pooled `yield_fraction()` is informational. With the
  default `min_yield=None`,
  every sample must pass.
- **Samples record no design-stage measurement** on the first-class `Spec`
  objects (they would overwrite each other).
- **Mismatch off for one block (ngspice).** `block.set_mismatch(False)` on a
  subcircuit block turns mismatch off for it and everything below it (for
  example, a bias network you want held ideal while the pair varies). Call it
  before instantiating the block.
- **Sample files.** See [Campaign folders](#campaign-folders) for sample placement and preservation.
- **Disk use is yours.** silk keeps every sample's deck and rawfile and
  **never deletes them**; there is no option that does. 150 samples are 150
  folders. Clear the output directory yourself when you are done.
- **SPICE refusals before the first sample.** Before printing the campaign
  size or writing a deck, silk rejects: a PDK with no statistical block
  (`NotSupportedError`); a deck with no PDK device (all samples would be
  identical), two specs sharing one name, `'process'`/`'both'` off the PDK's
  process corners (`MonteCarloError`); a missing library section
  (`MissingLibrarySectionError`). Bad arguments (`n=0`, `n=2.5`,
  `min_yield=0`, ...) raise `MonteCarloError` at the setter.
- **`run_test(store=True)`** raises for a test with either setting declared:
  database storage of samples is not built yet.
- **The settings belong to the test.** They do not inherit from the owner
  Module as `set_temperature` does.

## Failed corners and declared metrics

A campaign retains its registered spec and metric inventory, including
unspecced metrics and structured outputs such as `eye.width`. Selecting
`mc.corner('ss')` keeps that inventory even when every sample in `ss` failed.
`yield_fraction()` and the report show zero yield for every declared spec in
that corner; table/CSV output keeps the metric columns with missing values.
Report metric metadata and histogram limits also retain the declarations.

For a known metric, `stat()` and `percentile()` with no usable observations
return `nan` with one `MetricWarning` explaining the failed/unavailable and
used counts. This also applies to a manually constructed result with zero
observations. `worst()` returns `SampleWorstCase(nan, None, None)` when no
usable check remains. Unknown metrics, invalid statistic/percentile requests,
and `worst()` without a declared spec still raise `ValueError`.
`sensitivity()` still refuses a ranking with no usable samples after the
counted warning. Running a campaign whose **every** sample fails still raises
`SimulationError`; selecting an all-failed corner of a returned campaign does
not.

For hand-built `MonteCarloResults`, optional keyword-only `metric_names` and
`spec_declarations` supply the inventory explicitly. Each spec declaration is
a mapping with `name`, `metric`, `op` and `target`, for example
`dict(name='gain > 1', metric='gain', op='>', target=1.0)`. Spec-referenced
metrics are included automatically. These inputs are copied into an immutable
snapshot. If omitted, each inventory is inferred once from the supplied
reports for compatibility; an entirely failed manual root needs explicit
declarations because no names can be inferred. Reusing a spec name with a
different metric, operator or target raises `ValueError` naming both
definitions, including conflicts in supplied reports. A declared spec with no
successful checks counts as zero successes. With zero observations, its yield
fraction and confidence interval are `nan`.

## Ranking what causes the spread

**Rank per operating point** (per corner when no temperature or supply lists are declared): call `mc.corner('tt').sensitivity(metric)`, not
`mc.sensitivity(metric)` on a multi-corner run. Pooling corners adds each
corner's shift to every device's draws, which makes the matched devices'
columns nearly collinear. Rank each corner separately to avoid attributing
corner shifts to individual devices. The pooled call raises `ValueError`.

With `record_draws=True`, silk reads each device's drawn model parameters
(on ngspice from `showmod`; on Xyce the normalised draws from the `.res`
file) and `mc.sensitivity(metric)` fits the metric on the standardised draws
by least squares. When varying parameters are at least as numerous as usable
samples, it falls back to correlation only: effects, shares and R squared
are `None`. Use more samples than varying parameters for the fit. Each row gives the parameter's effect per sigma, its normalized
squared-coefficient share (`b² / sum(b²)`) and its correlation, sorted by share,
with the fit's R squared. `by='device'` sums shares and combines effects by
root-sum-square. Only devices
whose type has a `record` entry in the manifest are recorded;
`mc.draws.skipped_types` lists the others. On ngspice, `record_draws=True`
conflicts with `save_device_currents`; recording requires at least one device
type listed by the manifest's `record` entries. Ranking `correlate=` quantities is
not implemented and raises `NotSupportedError`.

## Limitations

See also [known issues](known_issues.md#monte-carlo).

- **Loop engine limits.** `samples_per_process` is ngspice only; see
  [Several samples in one ngspice process](#several-samples-in-one-ngspice-process).
- **No W/L check for the process section.** A pre-run check that the devices
  sit inside the process model's valid range is not built; process results
  outside it are on you.
- **Mismatch off per device.** `set_mismatch(False, devices=[...])` raises
  `NotSupportedError`. Per block is supported on ngspice.
- **Mismatch off per block on Xyce.** Xyce ignores the subcircuit-local
  switch, so the request raises `NotSupportedError` rather than silently
  varying the block.
- **Database storage** of samples.
- **Ranking operating-point (`correlate=`) quantities.**

## Calibrating a behavioral declaration from a campaign

`mc.measurement('offset_v', how='std', corner='tt')` creates a `Measurement`
from one corner's usable observations when that corner has one full point.
For a grid, use `mc.point(point).measurement('offset_v', how='std')`; exactly
one planned point must be selected, including points whose samples all failed.
`std` uses sample standard deviation (`ddof=1`). Insufficient usable data returns `nan` with
a `MetricWarning`; missing source attribution raises
`MonteCarloError`.

```python
from silk.specs import CalibrationBinding

sigma_spec.calibration = CalibrationBinding(
    target_module=comparator, param_name='offset_voltage', field='sigma')
sigma = mc.measurement('offset_v', corner='tt')
sigma_spec.record_measurement(sigma)
comparator.reconcile_specs(prefer_source='spice')
```

In this example, `comparator` is an existing behavioral module whose
`offset_voltage` parameter is a `Normal` declaration, `sigma_spec` is a spec
owned by that module, and `mc` is a SPICE campaign with an `offset_v` metric.
The target parameter must already be a `Normal` declaration for `field='sigma'`. Numeric fields
are replaced by a validated declaration copy. A `Variable` field is updated
through that Variable; expression fields refuse because there is no inversion
policy. `Normal` supports `nominal` and `sigma`; `Uniform` supports `low` and
`high`. Multiple field updates compose into one final declaration, so changing
`Uniform(0, 1)` to limits 2 and 3 succeeds. All final declarations and Variable
assignments are validated before any parameter write. Two bindings onto the
same Variable refuse. The default source filter excludes behavioral evidence.

`Measurement.aggregate` is an immutable `AggregateMeasurement` from
`silk.specs`. It records the metric, statistic, variation, selected campaign
corner label, sample counts, used indices, selected seeds, full base seed,
campaign ID, requested sample count, structure hash and captured behavioral
draw-scheme version. Its nested collections are immutable too. Failed samples
and reports with unavailable values have separate counts; a usable value that
fails its specification still participates. Corner views keep the campaign ID;
repeated campaigns have different IDs. The selected label is separate from
`Measurement.corner`, the effective condition (which can explicitly be NONE).
Unknown conditions stay unknown. A failed run's planned settings
are not proof that a simulator applied them.

This evidence is attribution, not replay reconstruction or a raw-sample
archive. `measurement()` neither updates a Spec nor saves to a database.
See the [explicit aggregate save workflow](database.md#aggregate-measurements-schema-v6).

Selected attempts with usable measurements retain the aggregate record when
reconciliation exports a report. Report JSON validates the metric and campaign
identity in these snapshots as well as in the current measurement.
An unavailable statistic (including an all-failed corner view) instead retains
its attribution in `SpecAttempt.aggregate`. The attempt stays `unavailable`,
its `measurement` stays `None`, and reconciliation reports `FAIL`; a NaN
statistic is never displayed as a measurement value. This optional field uses
the same aggregate codec, accepts only unavailable attempts, and rejects
malformed payloads or conflicting attempt merges. Older reports without the
field load with `aggregate=None`.

## Replay inputs

`mc.rerun(key)` compares the current experiment with an immutable signature
captured before the campaign prepares its first sample. The guard covers:

- Circuit structure, ordered formal-port declarations and instance bindings,
  device parameters and connections (including legacy `add_element` devices),
  nested sources and passives, block mismatch settings, and custom deck text.
- Analyses and their order; source DC, AC, waveform and controlled-source
  parameters and connections, using the effective configured supply voltage.
- Temperature, primary supply voltage and source selection, simulator selection
  and executable path, and effective simulator options, including inherited settings
  and bench overrides.
- Corner declarations; Monte Carlo sample count, seed, variation, draw recording,
  correlation probes and minimum yield; and the recorded sample seed.
- Model-library paths, includes, corner sections, statistical sections/switches
  and settings, configured compiled-model paths, and the PDK unit convention.
- Authored device-current/device-parameter capture, terminal-current method and
  waveform save-list configuration.
- The selected test placement and the original run's temperature/voltage assertions.

A one-off `run_monte_carlo(...)` retains its own request and leaves the bench's
Monte Carlo declaration alone. Replay detects subsequent changes to that
bench declaration too. Unchanged replay uses the original sample key and seed;
the existing output-path policy still overwrites that sample's files. Corner
subsets and failed samples use the same guard. Worker count,
`samples_per_process`, `output_mode` and generated output state are not
experiment inputs.

Keep the originating bench alive: `rerun(key)` returns a fresh report without
replacing the stored sample or recalculating the collection's yield.

Replay evaluates with the metrics and specs registered on the bench **now**;
evaluation is outside the replay guard. Changing an extractor or a spec can
therefore change the returned report even when the simulation inputs match.

The guarantee covers everything silk configures, including model-library
paths, sections and statistical settings. It assumes the simulator and model
files on disk are unchanged. silk does **not** read or fingerprint model-file
contents, and an in-place edit of a model file is outside this guarantee.
Put updated models at a different path or section so the guard detects the
change. Replay does not archive simulator or model files.
