# Design-space exploration — `silk.optimize`

`silk.optimize` samples and drives circuit design points; it does not search
for the best one. Pass Variables into the circuit, collect them in a VariableSpace,
and evaluate sampled or externally proposed points with Optimizer. A `Variable` is created —
usually via `pdk.<device>.<param>.variable(name=, value=, bounds=)`, which
pulls its bounds from the PDK catalog, or directly via `Variable(name,
value=, bounds=)` for a behavioral-model knob — and passed as a device
parameter at construction time. A `VariableSpace` collects a list of variables;
`space.sample(n, method=...)` generates joint `{name: value}` design points,
and `space.apply(point)` writes each value back onto its variable. Because
the variable object *is* the parameter (not a copy of its value), the next
`generate_netlist()` picks up the new value automatically — no re-wiring
step. `Optimizer` wraps a testbench and a `VariableSpace` into a single
`evaluate(point) → record` call.

```python
from silk import Variable, load_pdk
from silk.optimize import VariableSpace

pdk = load_pdk('sky130')

w_n = pdk.nmos1p8.w.variable(name='w_n', value=1.0, bounds=(0.42, 20.0))
l   = pdk.nmos1p8.l.variable(name='l',   value=0.15, bounds=(0.15, 1.0))

# w_n / l are passed as device parameters at construction time — the link
# is the object identity, not a snapshot of .value at this point.
mn = pdk.nmos1p8('mn', d='out', g='in', s='0', b='0', w=w_n, l=l)

space = VariableSpace([w_n, l])
```

Use `VariableSpace(mod.variables())` to collect variables already attached to
a circuit and its placed child modules. This includes device parameters,
source and waveform parameters, ideal passives, expression inputs and
behavioral-model parameters. Shared variables appear once; distinct variables
with the same name need distinct dictionary keys when constructing the space.
Behavioral models still read a plain variable parameter through
`params['k'].value`.

Discovery reads stored values without running models or drawing statistical
samples. It includes variables inside `Normal.nominal`/`sigma` and
`Uniform.low`/`high`, and inside standard containers and object-bearing NumPy
arrays. Nested statistical declarations are still unsupported when attaching
or running a model. Arbitrary object attributes, callable internals and SPICE
text are not searched. Unplaced `add_submodule` registrations and legacy
`add_element` parameter lists or `PassiveBase.value` storage are also excluded.

## Bounds

`pdk.<device>.<param>.variable(...)` derives `bounds` from the PDK catalog
when you don't pass one explicitly. A catalog entry with no declared upper
limit comes back as `(lo, None)`. `validate()` and `clip()` treat a
`None` end as unbounded. Sampling needs a finite interval, and calling `space.sample()` on a
variable with an open bound raises a `ValueError` naming the variable and
telling you to supply finite bounds:

```python
unbounded_w = pdk.nmos1p8.w.variable(name='w_n', value=1.0)
print(unbounded_w.bounds)          # (0.42, None)

try:
    VariableSpace([unbounded_w]).sample(5)
except ValueError as exc:
    print(exc)
# Cannot sample — variable(s) with unbounded end(s): 'w_n'
# (bounds=(0.42, None)). Sampling needs finite (min, max); PDK catalogs
# often set no upper limit, so pass bounds explicitly, e.g.
# pdk.nmos1p8.w.variable(name='w_n', value=1.0, bounds=(0.42, 20.0)).
```

The fix is to pass `bounds=(lo, hi)` explicitly at `.variable()` time, as in
the example above.

## Sampling methods

`Variable.sample(n, method='random', *, seed=0)`,
`VariableSpace.sample(n, method='random', *, seed=0)` and `Optimizer.sweep(..., seed=0)`
use a fresh NumPy generator per call. The same integer seed reproduces the
same points; the default is **0**, so repeating a call repeats its points.
Pass a different seed to select another reproducible draw, or `seed=None`
for fresh OS entropy. Booleans and non-integers raise `TypeError`; negative
integers raise `ValueError`. Python and NumPy global random states are untouched.
One generator is shared across all variables and groups in a space sample;
`sweep` forwards its seed to that sample.

Halton/grid values are independent of the seed for numeric and categorical
variables. Categorical random/LHS retain seeded random draws in both APIs.
For positive `n`, categorical grid enumerates every category once in declared
order in both `Variable.sample` and `VariableSpace.sample`, regardless of `n`;
`n=0` returns no samples. A space grid combines these with other variables in
the cross-product of groups and ungrouped columns.

For related numeric variables, pass `groups=[[w_n, l]]` to the
`VariableSpace` constructor. Variables in a group share the LHS stratum
permutation and the Halton base. Variables outside groups are independent.
Grid sampling steps grouped variables together through their grid columns by
index. Each group must have equal actual column lengths; mismatches raise
`ValueError` naming every variable and its count in that group. This includes
categorical columns (all categories) and resolution-limited numeric columns
(which may have fewer than `n` values). Groups and ungrouped variables cross in
the order of their first member in the variable declaration; each output dict
keeps the variable declaration order. For example, three variables
`[w_n, w_p, l]` with `groups=[[w_n, w_p]]` and three values each produce nine
points: each width pair is held while `l` traverses its three values. Random
sampling remains independent.

Categorical Halton maps `floor(u * category_count)` to the declared categories,
using the same sequence as discrete numeric Halton: base 2 for `Variable.sample`
and the group's prime base for `VariableSpace.sample`. It does not guarantee
all categories appear in a finite prefix, even when `n` reaches the category
count. For example, five base-2 draws over five categories visit indices
`[2, 1, 3, 0, 3]`, missing index 4. Use grid for exhaustive category coverage.

`space.sample(n, method=...)` supports four methods. All four respect a
variable's `resolution` (if `enforce_resolution=True`, the default): sampled
values snap to the discrete grid `[lo, lo+res, lo+2*res, ...]` instead of
landing at arbitrary floats.

- **`'random'`** — independent draws in the variable's scale coordinates.
- **`'lhs'`** — Latin Hypercube: each variable is independently stratified
  into `n` equal-width strata, one sample per stratum.
- **`'halton'`** — a deterministic sequence designed to cover the interval
  evenly with relatively few points.
- **`'grid'`** — cross-product of groups and ungrouped columns. `n` is a
  requested count per numeric variable; the output size is the product of
  actual axis lengths, including all categorical choices. Grouped columns
  step together and must have equal lengths. The number of points grows
  quickly with the number of independent axes.

Set `scale='log'` on a numeric `Variable` to sample all four methods in log
coordinates. Draws are then projected onto the resolution or integer domain.
For example, `Variable('x', bounds=(1, 1000), scale='log').sample(4, 'grid')`
gives approximately `[1, 10, 100, 1000]`. Log bounds must be finite and satisfy
`0 < lo <= hi` when sampling; PDK-derived incomplete bounds are allowed at
construction. Log scale rejects categorical variables.

`var_type` accepts exactly `'continuous'`, `'integer'`, or `'categorical'`;
`scale` accepts exactly `'linear'` (default) or `'log'`. Both options validate
on assignment too. Unknown sampling methods raise `ValueError`, even for `n=0`.

`resolution` is the variable's quantization property; `n` chooses samples.
Linear resolution and integer grids select up to `n` distinct values across
the full permitted domain, using evenly spaced domain indices. An enforced
resolution grid anchors at the lower bound. Without enforced resolution, linear integers use
`ceil(lo)..floor(hi)`: bounds `(1.2, 4.8)` have exactly `[2, 3, 4]`.
`n=0` returns none; `n=1` selects the first value. A requested count above the
available domain emits one `UserWarning` per affected variable per call,
naming the request and actual count. It points at the public sample/sweep caller
and follows Python's filters, including warnings-as-errors before evaluation.
Warnings appear in terminals, editors and notebooks; they are not persisted in
`SpecReport` or optimizer results.

Resolution controls both sampling and assignment. Set
`enforce_resolution=False` to bypass it. Linear grids select distinct values
when possible; log grids retain `n` draws and may repeat after rounding or
snapping. Random, LHS and Halton also retain `n` draws and may repeat.
Sampling and `apply()` keep projected values within bounds.

Integer variables require a whole-number `resolution` when one is set and a
whole-number lower bound when both are set. Construction and assignment reject
incompatible grids with `ValueError`, naming the variable, resolution, bounds
and remedy. Use an integer resolution and lower bound, or
`var_type='continuous'`. This rule applies to both scales, even with
`enforce_resolution=False`. An open lower bound is allowed until sampling,
when it must be supplied. Integral floats such as `2.0` are accepted; sampled
integer values are Python integers.

```python
points = space.sample(3, method='random', seed=7)
assert points == space.sample(3, method='random', seed=7)

space.sample(3, method='lhs', seed=None)  # fresh entropy each call

halton_points = space.sample(3, method='halton')
assert len(halton_points) == 3

space.sample(2, method='grid')      # 2 variables -> 2**2 = 4 points
# [{'w_n': 0.42, 'l': 0.15}, ..., {'w_n': 20.0, 'l': 1.0}]
```

## `apply()` is atomic

`space.apply(point)` sets each variable's `.value` from a `{name: value}`
dict. Points reach `apply()` from *outside* `sample()` too — an external
optimizer proposes an out-of-bounds candidate, a hand-written sweep dict has
a typo, a point loaded from a file is stale — so every value is validated
against its variable's bounds (or categories) **before any value is
assigned**. A bad point raises `ValueError` naming the variable, the
offending value, and its bounds, and leaves every variable in the space
unchanged:

```python
before = (w_n.value, l.value)
try:
    space.apply({'w_n': 5.0, 'l': 999.0})
except ValueError as exc:
    print(exc)
# Invalid value for variable 'l': 999.0 is not valid (bounds=(0.15, 1.0))
assert (w_n.value, l.value) == before
```

Values that are in-bounds but off the resolution grid are not rejected —
they're legal input, and the value setter's snapping handles them. Unknown
keys raise `KeyError`. A partial point leaves omitted variables unchanged.

## The `Optimizer` harness

`Optimizer(test, space)` accepts a `ModuleTestbench` directly and reads `.metrics` from the `SpecReport` returned by
`test.run(corner=...)`. Custom tests returning the legacy dictionary shape
`{'metrics': {...}}` remain supported. The following fragment assumes `test`
is an already configured testbench with the named metrics. Declare objectives and limits, then evaluate points:

```python
from silk.optimize import Optimizer

opt = Optimizer(test, space)  # test is a wired ModuleTestbench
opt.add_objective('gain_dB',  maximize=True)
opt.add_objective('power_uW', minimize=True)
opt.add_constraint('phase_margin', min=45.0)
opt.add_constraint('power_uW',     max=500.0)

record = opt.evaluate({'w_n': 2.0, 'l': 0.3})
# {'id': 0, 'params': {...}, 'corner': 'tt', 'status': 'success',
#  'metrics': {...}, 'feasible': True, 'error': None, 'error_type': None,
#  'timestamp': ..., 'applied_params': {...}, 'constraints': [...]}
```

`Optimizer.evaluate(point, corner=...)` applies the design point and calls
`test.run(corner=...)` once. `ModuleTestbench.evaluate(rawfile=...)` instead
measures existing simulation data without launching a simulator.

A report with valid metrics produces a successful evaluation even when its
specs fail (`report.passed == False`). Feasibility uses the testbench's
declared specs by default, with optimizer margins and explicit overrides
applied as described below. An extractor's
`"Error: ..."` metric value still produces `metric_error`.

### Variables, default specs, and margins

`add_objective(name, ...)` and `add_constraint(name, ...)` accept either a
registered metric name or a `Variable` in the optimizer's `VariableSpace`.
With an `i_ref` variable already included in the space:

```python
opt.add_objective(i_ref, minimize=True)  # no current-reporting metric needed
opt.add_constraint(i_ref, max=40e-6)     # sample value in amperes
```

The variable must be the same object used to construct the VariableSpace. Its sample
key is used even when the VariableSpace was constructed with a different dictionary
key. Ranking reads the recorded sample, not the
variable's current value. `record['params']` retains the requested point;
`record['applied_params']` snapshots the complete applied sample after
resolution snapping (including unchanged variables from partial requests).
Variable objectives and limits use this applied snapshot.
Older history records without `applied_params` remain readable, using their
recorded `params`; absent historical limit provenance reads as `None`.
Strings always refer to measured metrics.
A metric and variable with the same sample key cannot both be objectives
(or both be explicit constraints); use a distinct VariableSpace dictionary key.

All specs declared on the optimized testbench become feasibility limits,
including strict `<` / `>` and equality `==`. Multiple specs on one metric
all apply. An explicit `add_constraint('metric', min=..., max=...)` replaces
**all** spec-derived limits on that metric. Explicit min/max bounds remain
inclusive. This affects optimization only; it does not edit the testbench's
spec declarations or its report's pass/fail result.

`add_margin(spec_name, amount)` tightens a single uniquely named spec.
For a testbench declaring the following spec names and targets:

```python
opt.add_margin('gain > 60 dB', 3)      # > 63 dB
opt.add_margin('Vos < 5 mV', 1e-3)    # < 0.004, when target is 0.005 volts
```

Use the target's numeric unit; labels are not parsed for unit conversion.
Margins must be finite and nonnegative. Lower limits increase, upper limits
decrease, and the comparison operator is preserved. Equality specs and
unknown or ambiguous names raise `ValueError` with a remedy. Repeating a
margin call replaces that spec's margin. Explicit metric constraints take
precedence regardless of call order. There is no global or percent margin.

`record['constraints']` snapshots the effective limits used for that point;
`summary()['constraints']` describes the current configuration. Each limit
includes `source` (`spec`, `spec+margin`, or `explicit constraint`), its metric
or sample key and `value_source`, and its operator/target or min/max bounds.
Spec limits also include `spec_name`, `spec_target`, `margin`, and declared
`units`; explicit limits list the spec names they `overrides`.
For standalone checks use `opt.feasible(metrics, params=sample)` when a
variable is constrained. Missing constrained values, including omitted
sample keys, are infeasible.

A scalar NaN measurement remains unchanged in history and a completed
extraction still has `status='success'`, so the sweep continues. NaN fails
any constraint declared for that metric. NaN in any declared objective
excludes the record from `opt.best`; if every candidate is excluded, `best`
is `None`. An objective-only NaN leaves constraint feasibility unchanged,
and an unrelated NaN diagnostic affects neither feasibility nor selection.
These rules also apply to NumPy floating scalars. Positive and negative
infinity use ordinary bound comparisons and objective scoring.

Every declared objective must have a recorded value for a candidate to be
`best`: absent, `None`, and string values also exclude it. This applies to
metric objectives and Variable objectives (using `applied_params`, or `params`
for legacy records), without changing history, status or constraint feasibility.
Variable-only objectives allow empty metrics. `summary()['best']` is
the same record as `best`, `None` when no candidate is eligible.

`best` checks successful records against the current testbench specs, margins
and explicit metric/Variable constraints each time it is read. Changing only
limits can immediately exclude an old winner or admit a previously infeasible
record. It compares recorded measurements and applied parameters again without
rerunning simulations or applying parameters; legacy records use `params`.
It returns the original record, preserving historical `feasible` verdicts and
`constraints` snapshots. A newly admitted winner can therefore still have
`feasible=False`. `summary()['best']` follows current selection, while
`summary()['feasible_count']` counts historical feasible verdicts.
This rechecks acceptance limits; it does not establish that old measurements
remain valid after circuit, backend, stimulus, model or extractor changes.

`Optimizer.add_constraint(name, min=None, max=None)` sets numeric limits
on measured metrics or variables. `Module.add_constraint(obj)`, available
after `import silk.constraints`, attaches layout constraints to a circuit.

`evaluate()` and `sweep()` require one ordinary report per design point.
A testbench with declared Monte Carlo runs, corners, temperature lists,
supply lists, Variable lists or groups (including inherited or one-element
lists) raises
`NotSupportedError` before parameters, history, decks or simulations change.
This includes a single declared corner and an explicit `corner=` argument.
Clear these declarations with `set_monte_carlo(None)`, `set_corners(None)`,
`set_temperatures(None)`, `set_supply_voltages(None)`,
`set_variable_values(var, None)` and `set_groups(None)`
before optimizing; the optimizer can then select one corner with `corner=`.

After this preflight, `evaluate()` records completed evaluations and handled
run/bounds failures in `opt.history`, tagged with one of six `status` strings.
Objective and constraint values must be real; input errors such as complex
constraint values propagate before a record is appended. The statuses are listed in
`silk/optimize/optimizer.py`:

| status | meaning |
|---|---|
| `'parameter_error'` | `space.apply()` raised `KeyError` or `ValueError` — bad point, out of bounds; other exception types propagate. |
| `'netlist_error'` | `CircuitError` from the run, except `SimulationError`, which is classified separately. |
| `'simulation_failure'` | `SimulationError`, including missing, aborted or incomplete simulation data. |
| `'metric_error'` | Simulation ran, but one or more metric extractors raised (stored as `"Error: ..."` strings in the metrics dict). |
| `'unexpected_error'` | Any other exception from `test.run()`, caught and logged the same way. |
| `'success'` | All steps completed; `feasible` reflects the declared constraints. |

`opt.sweep(n, method='lhs', corner='tt', max_failures=None, n_workers=None, *, seed=0)`
samples points from the space and evaluates each, continuing through
failures by default. For grid sampling, `n` is per numeric variable.
`max_failures` stops early when the cumulative failure count reaches the
limit, which can save time when many points fail simulation.

### Parallel design points (`n_workers`)

For a `ModuleTestbench` test, `sweep()` can run multiple
design points at once: `n_workers=None`/`1` (default) is serial and
in-process, an int caps at the CPU count, `'all'` uses every CPU (resolved
through `silk.simulation.executor.resolve_n_workers` — same semantics as
`run_corners`/`run_all_tests`; behavioral is always forced to serial). Each
worker is a full extra OS process with its own parsed PDK model library —
size `n_workers` for memory, not just CPU count.

Parallel sweeps use the `fork` multiprocessing start method. Where `fork`
is unavailable, silk falls back to serial execution with a `RuntimeWarning`.

Every point is applied to the space in the parent process with
`space.apply(point)`. Device parameters and metrics that read a variable's
`.value` therefore see the value used for that point.
Only the simulator launch itself moves to a worker, through the shared
`silk.simulation.executor.run_tasks` pool — the same executor
`run_corners`/`run_all_tests` use — and metric extraction happens back in
the parent, one point at a time, in submission order. Results (metrics,
`feasible`, `best` and NaN handling) are therefore identical
between `n_workers=None` and `n_workers>1`; only wall-clock time and each
point's output folder differ.

Each point in a `ModuleTestbench` sweep gets its own output folder with
`run_label='point_<id>'`, including serial runs. The testbench keeps the same
netlist object and ends with the state of the last evaluated point.
A refusal before result replacement preserves earlier loaded results; a
failure after replacement starts clears current results and implicit reload
paths. Metric extraction failures retain the new simulation data. An
executor error with unknown replacement state does not itself erase results.

A custom test, including a `ModuleTestbench` subclass overriding `run()`,
always runs through its own `run()` serially; passing `n_workers > 1` raises
`TypeError`.

`max_failures` is checked after each *chunk* of `n_workers` points (1 for
serial, so serial behaviour — including how much compute is skipped — is
unchanged), not after every single point when running in parallel: a
chunk already dispatched to the pool runs to completion before the count
is checked. History keeps only the prefix through the failure limit; the
live bench and variables may describe a later point than the last history entry.

Read results back off the harness:

```python
opt.history      # every record, in order
opt.successes    # status == 'success'
opt.failures     # any non-success status
opt.best         # best feasible record (equal-weight average of signed objectives), or None
opt.summary()    # counts, best, objectives and current constraints
```

`opt.best` does not apply the winner. When it is not `None`, use
`space.apply(opt.best['applied_params'])`, then rerun the test before
inspecting its waveforms.

`best` negates maximization objectives and averages all objective values
with equal weights, without normalizing units. Choose metric scales carefully
when combining objectives. Use `sweep()` for a fixed sample set or drive
`evaluate()` from your search algorithm.
`VariableSpace.to_bounds()` / `to_list()` / `from_list()`
exist specifically to interoperate with array-based optimizer APIs.

## Reading results for an AI agent or surrogate model

There are no dedicated export methods. Everything is public:

- `opt.history` / `opt.successes` / `opt.failures` / `opt.best` — records,
  each with `params`, `applied_params`, `metrics`, `status` and `feasible`.
- `opt.summary()` — counts, `best`, `objectives` and `constraints`.
  `summary()['objectives']` is a list in declaration order (the order `best`
  uses), one dict per objective: `name`, `goal` (`'minimize'` or
  `'maximize'`) and `value_source` (`'metric'` or `'variable'`, the same key
  as the constraint descriptions).
- `opt.space.variables()` — a new `{space key: Variable}` dict in
  `opt.space.names` order, holding the very `Variable` objects (bounds,
  `var_type`, `scale`, categorical choices, current `value`).

Building `(X, y)` for a surrogate in user code:

```python
import numbers

import numpy as np

names = opt.space.names
objs = [o['name'] for o in opt.summary()['objectives']]
rows = [r for r in opt.successes
        if all(isinstance(r['metrics'].get(n), numbers.Real) for n in objs)]
X = np.array([[r['applied_params'][n] for n in names] for r in rows], dtype=float)
y = np.array([[r['metrics'][n] for n in objs] for r in rows], dtype=float)
```

This sketch assumes metric objectives and numeric variables; for a Variable
objective read `r['applied_params'][name]` instead, and map categorical
values yourself. Decide your own policy for NaN objectives (they remain in
history as successes).

## Worked examples

- `silk/examples/parameter_sweep_example.py` — the sampling half only,
  using the packaged sky130 catalog, without installed models or a simulator:
  `Variable` + `VariableSpace`,
  all four sampling methods compared, `apply()` → `generate_netlist()`
  sweep loop, coverage plot. Run from the repo root with
  `MPLBACKEND=Agg .venv/bin/python -m silk.examples.parameter_sweep_example`.
- `silk/examples/source_follower_optimize.py` — the full constrained
  loop: `Optimizer` minimizing bias current subject to bandwidth and gain
  constraints, `sweep(method='lhs')`, and `opt.best`, passing the AC
  `ModuleTestbench` directly to `Optimizer`.
  Needs ngspice and installed sky130 models resolved through silk settings
  (see [settings](settings.md)). Run from the repo root with
  `MPLBACKEND=Agg .venv/bin/python -m silk.examples.source_follower_optimize`.

## Limitations

- Use `sweep()` for a fixed sample set or drive `evaluate()` from your search algorithm.
- Evaluation history (`opt.history`) is in-memory only. It is not persisted
  to silk's tracking database — capture it yourself (e.g. `json.dump` on
  `{'history': opt.history, 'summary': opt.summary()}`, with a `default=`
  hook: records may hold numpy scalars, and `best` is a record that also
  appears in `history`, so the output repeats it) before the process exits if you need it later.
