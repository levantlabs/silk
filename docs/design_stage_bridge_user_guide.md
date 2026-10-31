# Silk Design-Stage Bridge User Guide

Declare a target on a module, record a measurement, and reconcile the two.
A calibration binding can also copy the measured value into a behavioral
model parameter, so a later behavioral run uses the characterized value.

Start with [Quickstart](quickstart.md) for circuit construction and simulation.
This guide focuses on specs and calibration. Behavioral simulation needs no
simulator or installed PDK models. SPICE simulation needs a simulator and,
when the circuit contains PDK devices, the corresponding model library.

## Quickstart: record, reconcile, calibrate

Run this complete example from the repository root with `.venv/bin/python`.
It records a supplied measurement; it does not run SPICE or claim to measure
the gain itself.

```python
import tempfile
import time
from silk import Module, PortDir, Measurement, CalibrationBinding

outdir = tempfile.mkdtemp()
block = Module(None, outdir, 'gain.cir', None, module_name='gain_block')
block.define_module_ports({'vin': PortDir.IN, 'vout': PortDir.OUT})
params = {'gain': 10.0}

def gain_model(ports, t, params):
    ports.vout.v = ports.vin.v * params.gain

block.set_behavioral_model(gain_model, params=params, execution='vector')
gain_spec = block.declare_spec(
    name='gain', metric='gain', op='>=', target=12.0,
    key='voltage_gain',
    calibration=CalibrationBinding(target_module=block, param_name='gain'),
)
gain_spec.record_measurement(Measurement(
    source='spice', value=11.5, timestamp=time.time(),
))
result = block.reconcile_specs()
print(result.summary())
assert result.fail_count == 1
assert params['gain'] == 11.5
```

Output:

```text
Spec reconciliation: 0 PASS, 1 FAIL, 0 UNMEASURED
  [FAIL] gain: 11.5 >=12
Calibrations applied:
  gain_block.gain = 11.5 (from gain, source=spice)
```

The measured gain fails the target, but it still calibrates the model.
Calibration is independent of whether the measurement meets the target.
The next behavioral run reads `11.5` from the same parameter dictionary.

## Declaring specs

`Module.declare_spec(name, metric, op, target, units='', description='',
parent=None, calibration=None, *, key=None)` returns a `Spec` owned by the
module. Supported comparisons are `<`, `<=`, `>`, `>=` and `==`.
`units` is a display label; silk does not convert measurement units.

The stable local `key` defaults to the exact `name`. Reusing a key on one
module raises `DuplicateNameError`; repeated display names need distinct
keys. A non-real target (including a boolean) or a non-string key raises
`TypeError`; an empty or whitespace-only key raises `ValueError`.
`metric`, `op`, `target` and `key` are read-only after construction;
changing or deleting them raises `AttributeError`. Edit the declaration
and rebuild the circuit to change them. `declared_at` records the caller's
filename and line number.

Keep the returned spec handle to record measurements. A plain metric name,
such as `gain`, does not make reconciliation run a metric extractor.
Record it manually, or register the metric and spec on a testbench.
Dotted metrics read object attributes as described below.

## Recording measurements

`Measurement` and `Spec` are available from `silk`. A measurement supplies
`source` and `value`; the ordinary sources are `behavioral`, `spice`,
`extraction` and `user`. Optional fields include `corner`, `timestamp`,
`sim_run_id`, `notes`, `passes`, `temperature`, `voltage`, `sim_type`,
`tool_name`, `origin_version_id`, `profile_id` and `aggregate`.

`timestamp` defaults to `0.0`, meaning unknown. Manual
`record_measurement()` does not fill in the current time; supply
`time.time()` when recording a new observation. It does not reconcile or
calibrate. It records an attempt with an evaluated verdict; reconciliation
also updates the live measurement's `passes` cache.

Ordinary live measurements occupy one slot per `(source, corner)` and a
new publication replaces that slot. Different sources coexist. A
`profile_id` scopes a slot by profile, source, corner, temperature and
voltage. Placement-scoped evidence also stays separate. Prefer the public
methods over manipulating the slot dictionary:

- `spec.latest()` returns the live measurement with the greatest timestamp.
- `spec.latest(source='spice')` restricts that lookup to one source.
- `spec.evaluate(measurement)` compares a value with the target.
- `spec.attempt_history()` returns recorded attempts;
  `spec.latest_attempt()` returns the latest attempt. Both accept `source=`.
  Latest means greatest observation timestamp; later publication breaks ties.

A NaN observation is recorded as an `unavailable` attempt rather than a
usable numeric observation. Testbench extractor failures and non-numeric
results also produce failed attempts, with reasons. A failed latest attempt
must remain visible even if an older measurement passed.
To record failure manually, import `time` and import `SpecAttempt` from `silk.specs`,
then call `spec.record_attempt(SpecAttempt(source='user', timestamp=time.time(),
kind='extractor_error', reason='extraction failed'))`.
See [Metrics](metrics-library.md) for metric failure conventions.

## Hierarchy

Pass `parent=parent_spec` when declaring a child spec. The child is appended
to `parent_spec.children`. This records a relationship; silk does not sum
budgets, derive child targets, invalidate dependent specs, or change parent
targets automatically. Children may use different metrics and need not live
on descendant modules.

`top.reconcile_specs()` walks the module placements, including the root,
not the spec-parent relationships. It reports specs on nested modules too.
`add_module(block, instance_name, nets=...)` places a live reference to
`block`. Reusing that object shares its declarations and parameter dictionary;
build separate modules when placements need independent design parameters.
Outcomes carry `owner_path` so placement evidence can be distinguished.
Callable and captured state are copied per placement for each run;
module-level globals remain shared and are not reset; see [BehavioralEngine](api-reference.md#behavioralengine).

## Reconciliation

`module.reconcile_specs(prefer_source=None)` returns a
`SpecReconciliationReport`. It resolves dotted attributes, selects evidence,
evaluates each spec, validates calibration bindings, then applies eligible
calibrations after all specs have been staged.

The latest recorded attempt determines the result. With
`prefer_source='spice'`, evidence from that source is preferred when present;
otherwise selection falls back to the latest evidence from any source.
A failed attempt from the preferred source is still a failure. When no
attempt exists, legacy live measurements provide the fallback.

| Evidence | Outcome |
|---|---|
| Usable measurement meeting the target | `PASS` |
| Usable measurement missing the target | `FAIL` |
| Latest attempt failed | `FAIL`, with no current measurement |
| No attempt or measurement | `UNMEASURED` |

A failed-attempt outcome retains `last_successful_attempt` separately when
available. An older success does not become the current verdict or calibrate
in place of that failure. For a numeric outcome, `margin` is
`abs(value - target)`, not a signed pass margin.

Read `report.outcomes`, `pass_count`, `fail_count`, `unmeasured` (a list),
`calibrations_applied`, `summary()` and `as_dict()`. Each outcome includes
`spec`, `measurement`, `status`, `margin`, `source_used`, `calibrated`,
`owner_path`, `report_key`, `latest_attempt` and `last_successful_attempt`.

A preferred behavioral source skips the default calibration only when
behavioral evidence exists and is selected. A preference alone does not
disable calibration, because source selection can fall back.

## Calibration and behavioral parameters

`set_behavioral_model(fn, params=None, *, execution='vector')` attaches a
ports-form callable `fn(ports, t, params)`. Read inputs and assign outputs
through attributes such as `ports.vin.v` and `ports.vout.v`; return `None`.
The parameter view supports both `params.gain` and `params['gain']`.
The supplied parameter dictionary is retained by reference. With no dictionary,
the model receives an empty parameter view, but calibration is unavailable.

`execution='vector'` calls the model with the full transient time array;
`execution='sample'` calls it once per time point. Ports need directions.
Use a top-level testbench to supply inputs and run the reusable block.
See [Behavioral simulation](agent-guide.md#one-workflow-two-engines) for supported circuits
and analysis limits.

`CalibrationBinding(target_module, param_name,
source_filter=frozenset({'spice', 'extraction'}), field=None)` names the write
location. The binding is skipped when there is no eligible measurement,
the latest attempt failed, the source is outside the filter, or the value is
NaN. A valid measured value can calibrate even when its spec fails.

Binding validation happens before any parameter is changed:

| Problem | Error |
|---|---|
| Target has no attached parameter dictionary | `RuntimeError` |
| Parameter key was not declared | `KeyError` |
| Bindings write conflicting values to one parameter | `ConfigurationError` |
| Distribution field does not match the target declaration | `VariationError` |
| Statistical declaration field contains a `Variable` expression | `NotSupportedError` |

For statistical parameters, use `field='nominal'` or `'sigma'` for `Normal`,
or `'low'` or `'high'` for `Uniform`. Multiple statistical-field bindings
targeting the same `Variable` are refused with `ConfigurationError`. Plain
parameter entries, including Variables and expressions, are replaced by the
measured number.
See [Monte Carlo](monte-carlo.md) for statistical declarations.

For a plain scalar, `module.calibrate_behavioral('gain', value)` is the
manual write path. It raises `RuntimeError` without a parameter dictionary
and `KeyError` for an undeclared key. It does not create missing keys.

## Integration with ModuleTestbench

On an existing testbench, register a metric with `tb.add_metric(...)`, then
use `tb.add_spec(name, metric_name, op, target, *, key=None)`. The metric can
be its registered name or the handle returned by `add_metric()`.
`add_spec()` returns a `Spec`; capture it to inspect evidence or attach a
calibration binding. `spec.add_spec(...)` can forward another declaration
to the same testbench, with a distinct key.

`tb.run()` returns a `SpecReport` and publishes evidence to attached specs.
Successful measurements use `source='behavioral'` for a behavioral run and
`source='spice'` otherwise. An extractor error, unavailable value or
non-numeric result is recorded as a failed attempt with its reason.
Calling `reconcile_specs()` then evaluates that evidence and can calibrate.
See [Testbenches](multi-analysis-tests.md) for a complete runnable workflow.

## Reports and comparisons

`generate_report(module)` returns a `CircuitReport` with a
`spec_reconciliation` report, including an empty report when there are no
specs. Generating it invokes reconciliation and can therefore calibrate.

`report.save('report.md')`, `save_html('report.html')` and
`save_json('report.json')` write Markdown, HTML and JSON respectively.
HTML uses a `section.spec-reconciliation` wrapper and `spec-pass`,
`spec-fail` and `spec-unmeasured` classes on table rows.
Markdown and HTML omit the spec section when there are no outcomes.
JSON includes `spec_reconciliation` using the reconciliation report's
`as_dict()` representation.

`diff_reports(before, after).spec_transitions` compares outcomes by
`(owner_path, spec.key)`. It includes changes in status, value, source or
attempt evidence, plus additions and removals. Entries expose `spec_name`,
`spec_key`, `owner_path`, `report_key`, `before_status`, `after_status`,
`before_value`, `after_value` and `source_used`. An absent side has `None`
status and value. Ambiguous duplicate identities raise `ConfigurationError`.
See [Reports](reports.md) for report generation and comparison.

## Dotted-path metrics

A metric of exactly three dot-separated parts can read an attribute:

- `net.<name>.<attr>` reads an attribute on `module.net_dict[name]`.
- `device.<name>.<attr>` reads the device in `module.elem_dict[name]`
  (the entry's `device` value when stored as a dictionary).

For example, `net.vout.peak_current` can constrain a populated current
annotation. Retained annotation evidence preserves its identity and time.
Legacy numeric attributes without attempt evidence are exposed as
`source='user'` measurements with unknown time. Resolution does not erase a
recorded failure. Missing objects, absent attributes, `None` or non-numeric
values cannot supply a measurement; without other evidence the outcome is
`UNMEASURED`. Declaring a dotted metric does not run current extraction.
See [Current annotation](api-reference.md#current-annotation) for producing current evidence.

## The flash ADC example

[The flash ADC example](../silk/examples/flash_adc_example.py) combines a
track-and-hold, resistor ladder, comparators and behavioral encoder. Its
characterization testbenches register metrics and specs, including comparator
offset specs. A comparator model reads its offset parameter like this.
The snippet requires `np`, `VDD` and a comparator module `comp` with input
ports `vp`, `vm` and output port `vout`:

```python
def _comparator_model(ports, t, params):
    offset = params['offset_voltage']
    ports.vout.v = np.where(ports.vp.v + offset > ports.vm.v, VDD, 0.0)

comp.set_behavioral_model(_comparator_model, params={'offset_voltage': 0.0}, execution='vector')
```

The example also supports SPICE and offset campaigns. Its results depend on
the selected mode and measured circuit; inspect the actual reconciliation
summary rather than assuming every spec passes. Example artifacts go to the
`SILK_EXAMPLES_OUT` setting, defaulting to `silk-output/` under the working
directory. See [the example index](agent-guide.md#7-curated-example-index) for context.

## Database persistence

Construct the module with `db_path=...` and give long-lived specs stable
`key=` values. `module.create_circuit_version()` seals a circuit snapshot.
`module.store_report(spec_report, ...)` explicitly stores a testbench's
`SpecReport` as typed metric observations and first-class spec measurements.
Neither `test.run()` nor `reconcile_specs()` writes that report automatically.

`get_stored_specs(include_measurements=True)` queries stored declarations and
measurements; `restore_spec_measurements(...)` applies compatible evidence to
a rebuilt live circuit. Metrics and currents require matching structure;
restoring spec measurements also requires matching declaration identities.
Call `module.close()` when finished. See [Database tracking](database.md) for
a complete close/reopen example, condition selection and compatibility rules.

## Limits

Specs express targets and compare evidence. They do not automatically allocate
budgets, attribute a system result to individual blocks, propagate changed
assumptions, or negotiate new targets. Reconciliation selects evidence; it
does not compute an all-corner worst-case envelope. Use the campaign APIs for
condition and sample aggregation.

Outcome statuses are `PASS`, `FAIL` and `UNMEASURED`; approval and signoff
states belong in the calling workflow. Declarations live in Python. There is
no separate spec-file import format or interactive spec editor in this bridge.

For signatures and return types, see the [API reference](api-reference.md).
