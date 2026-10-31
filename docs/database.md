# Database tracking

`silk.db` stores a circuit's declared design intent, measured results, and
operating currents used by layout estimation. Pass `db_path=` to a `Module`
to enable it. It is off by default.

The SQLite file is the canonical persistent tracking record. Python remains
authoritative for constructing the circuit, declaring specs, defining tests
and extractors, and configuring behavioral models. The database verifies and
restores evidence onto an equivalent rebuild; it does not reconstruct a
`Module`.

## Start with a tracked circuit

Pass `db_path=` when building the circuit, declare its specs and tests in
Python, then call `create_circuit_version()` to seal a snapshot before saving
results. Use `run_test(..., store=True)` to run and save one registered test,
or call `store_report(report)` after a run. Close the module when finished.
Later, rebuild the same circuit from Python to restore compatible evidence;
use `Database` directly when you only need to inspect stored history.

The complete [run-and-store example](#run-and-store-in-one-call-run_teststoretrue)
below needs ngspice and sky130 models. For a simulator-free walkthrough, see
[Runnable references](#runnable-references). Shorter snippets assume the
circuit, test, report or stored IDs introduced in their surrounding text.

## What silk.db preserves

- immutable, sealed circuit-version snapshots with parent lineage and locks;
- independently versioned structure and spec-set fingerprints;
- the stored structure representation: devices, passives, nets, hierarchy,
  parameters, and content-addressed module definitions;
- spec declarations, including stable owner paths/keys, targets, units,
  descriptions, parents, and declaration provenance;
- append-only numeric or text metric observations with source, analysis,
  corner, temperature, voltage, object, units, tool, timestamp, and JSON
  conditions;
- first-class numeric spec measurements with operating context, pass/fail,
  and the version where they were observed;
- **characterization profiles** — a reusable, derived identity for "which
  registered test, on which live owner, with which declared parameters"
  produced a piece of evidence (see below);
- **selected current sets** — one or more complete, independently stable
  operating-current sets per circuit version, corner, and (when
  profile-scoped) exact temperature/supply, each with a stable ID and stored
  rows that a replacement does not overwrite;
- **run provenance** (schema v4) — the `silk.simulation.provenance` record of the run
  each metric row, spec measurement and current set came from (silk,
  simulator, PDK library files, options, prepared deck), stored once per
  distinct record (see [Run provenance](#run-provenance-schema-v4)).

This is a tracking database, not a raw simulation warehouse. It does not
preserve raw runs, waveform samples, simulator logs, Python reconstruction,
parasitic restore, or calibration restore. A stored `results_file` or
`sim_run_id` is provenance/correlation text, not embedded run data.
Behavioral parameters and `CalibrationBinding` state remain in Python. There
is no run ledger — silk stores only evidence a caller explicitly chose to
keep, not every simulation that ran.

## File compatibility

Silk validates an existing non-empty path through a read-only connection before
opening it normally. A rejected file is not modified.

| File at `db_path` | Behavior |
|---|---|
| Missing or zero bytes | Create a new schema-v7 database. |
| Current, complete schema v7 | Open normally. |
| Schema v1, v2, v3, v4, v5 or v6 | Raise `UnsupportedSchemaVersionError`, naming the explicit copy migration (see below). Never upgraded implicitly. |
| Other older/newer declared schema version | Raise `UnsupportedSchemaVersionError`; preserve the original. |
| Unversioned legacy silk database | Raise `DatabaseCompatibilityError`; preserve the original. |
| Other SQLite file, wrong format ID, malformed v7 shape (including a same-column impostor missing a required foreign key or unique index), or corrupt file | Raise `DatabaseCompatibilityError`; preserve the original. |
| Supported schema with an unknown stored fingerprint scheme | Raise `UnsupportedFingerprintSchemeError` before comparing identities or mutating live objects. |

Silk opens database schema version 7. Versions 1 to 6 require explicit
copy migrations to new files; none is upgraded in place. Database version
3 changed only `spec_measurements.value` from `REAL NOT NULL` to nullable
`REAL`. Database version 4 adds only the `run_provenance` table
and a nullable `provenance_id` column on `simulation_results`,
`spec_measurements` and `net_current_sets`. Database version 5
stores the state of each operating condition on those three tables (see
[Condition states](#condition-states-schema-v5)). Version 6 adds nullable aggregate
measurement evidence (see [Aggregate measurements](#aggregate-measurements-schema-v6)).
These database versions are independent of
**metric schema format 2.0**, which is unchanged.

### Condition states (schema v5)

`simulation_results`, `spec_measurements` and `net_current_sets` carry
`corner_state`, `temperature_state` and `voltage_state` beside the
`corner`, `temperature` and `voltage` columns:

| State | Value column | Python scalar |
|---|---|---|
| `fixed` | set | the value (`str` corner, `float`) |
| `none` (the condition provably does not apply) | `NULL` | `ConditionState.NONE` |
| `unknown` | `NULL` | `None` |
| `sweep` (temperature or supply) | `NULL` | `ConditionState.SWEEP` |

`CHECK` constraints tie each state to its value, and a profile-scoped row
needs every fact `fixed` or `none`. Writers (`add_simulation_result`,
`insert_spec_measurement`, `replace_net_current_set`, `store_report`,
`store_net_currents`) take the Python scalar and derive the state; no writer
takes a state separately. Records and the `corner=`/`temperature=`/`voltage=`
selectors use the same scalars: `ConditionState.NONE` selects none rows and
never unknown ones, and `None` as a selector still means "no filter".
`ConditionState` is a plain Enum from `silk.simulation`, so a none corner
never equals a corner named `'none'`. `net_current_sets.corner` is nullable
in v5, but `replace_net_current_set` and `store_net_currents` still refuse a
`None` (unknown) corner; `ConditionState.NONE` is accepted.

### Unavailable measurements

A `spec_measurements` row whose `value IS NULL` is an **evaluated but
unavailable** observation. `get_stored_specs()` and
`restore_spec_measurements()` return Python `float('nan')` for that value;
it fails all five supported spec comparisons. No row remains no observation.
For `get_simulation_results()`, a row decodes to NaN only when **both**
`value` and `value_text` are NULL. Text/error rows retain `value=None` and
their original text. Finite values, including real zero, are unchanged.
NULL context fields such as temperature, voltage and profile remain `None`.

Writers still require a numeric value (which may be NaN) or text; passing
neither is an error. `store_report()` retains its extractor-error convention:
the failed spec's compatibility-status text includes `Error: ...`, its raw
nonnumeric metric is reported as skipped, and no first-class numeric
measurement is fabricated. Mixed finite/NaN/error reports save atomically.

CircuitReport JSON uses a separate **schema version 5** contract. `CircuitReport.save_json()`
uses `silk._json_codec.encode_json` with `allow_nan=False`: a NaN measurement
is encoded as `{"value": null, "status": "unavailable"}`, and infinities use
signed `infinite` tags. Finite values (including zero) and error strings keep
their ordinary JSON representations. `CircuitReport.load_json()` restores
these tags to Python floats. See [report JSON encoding](reports.md#saving-and-loading)
for the full wire format and legacy-reader compatibility. Readers capped at v4
refuse v5 files. V5 preserves all five violation location types and fields;
complete legacy structured net/terminal locations still load. Text-only legacy
locations cannot be recovered and refuse with regeneration advice; empty or
missing locations also refuse.

These report tags are distinct from SQL NULL storage described above and from
JSON metadata stored in database text columns, which also uses the strict
codec. Database schema **7**, CircuitReport JSON schema **5**, and metric-catalog
schema format **2.0** are independent versioned contracts. Metric-warning
capture and persistence are already supported; see the
[per-metric diagnostic records](metrics-library.md#report-capture-and-persistence-of-metricwarning).

## Identity and stable spec keys

A snapshot has two explicit fingerprints:

- `silk.structure/v4`: circuit structure, DUT-owned sources, device parameters normalized to
  PDK-native SPICE text, composition, and each child's port-to-parent-net
  bindings at every depth;
- `silk.spec/v1`: individual spec definitions and their set at stable owner
  paths.

Fingerprints are always compared together with their scheme. A changed device,
passive, source, net, parameter, or hierarchy changes structure identity. A changed
spec key, owner path, metric, operator, target, units, or parent relationship
changes spec identity.

Give every long-lived declaration an explicit keyword-only `key=`:

```python
gain = amp.declare_spec(
    'DC gain', 'gain_dB', '>=', 60.0, units='dB', key='gain'
)
```

The display name and description can then evolve without changing identity.
If `key=` is omitted, the display name is the key, so renaming it intentionally
creates a different declaration. `declared_at` is provenance, not identity.

`create_circuit_version()` assembles structure and declarations in one
transaction, verifies their fingerprints, and seals the snapshot. Sealed
structure/declarations cannot be rewritten. Evidence writers require a sealed,
unlocked target.

## Characterization profiles

A **characterization profile** groups evidence — metrics, first-class spec
measurements, and a selected current set — under the declared facts that
produced it: *which registered test, on which live module, with which
declared parameters*. It has a third fingerprint, `silk.profile/v1`, over:

- `owner_path` — the root/instance path of the live `Module` that owns the
  registered test (`$`, `$/amp1`, ...), not `module_name` and not the
  generated testbench module's filename;
- `test_name` — the exact, unmodified test name (a `Module` fails loud on a
  duplicate test name, so this is reliable identity, not a guess);
- `parameters` — the test's complete declared parameter map, defaults
  materialized (`{}` is valid — it means "no declared parameters," not
  "unknown"). A `Variable` or Variable expression declared as a parameter is
  recorded as its value when the run's context is built, so a sweep stores
  one profile per value and the profile names the value the run used. The
  bench's own `tb.parameters` keeps the live Variable.

**There is no separate user-chosen profile label.** Identity is derived
entirely from these three declared facts.
Two runs with the same owner/test/parameters are the same profile, whatever
they are informally called; a changed owner, test name, or parameter value is
a different profile, with no rename or in-place update.

Evidence storage registers profiles automatically; no separate registration
step is required. `Database.store_characterization_profile(version_id, context)`
can also register a profile explicitly. Storing the same
canonical identity again on the same circuit version is idempotent (the same
`profile_id` comes back); a digest match with a different stored owner/test/
parameters payload is corruption or a hash collision and raises
`EvidenceMismatchError` rather than silently updating anything.

### PVT is selection beneath a profile, not profile identity

Process corner, temperature, and primary supply voltage (PVT) are **not**
part of a profile's identity — they are an exact selection key *beneath* one:

```text
$/amp : load_step : {cload_f=2e-12} @ (tt, 27 C, 1.8 V)
$/amp : load_step : {cload_f=2e-12} @ (tt, 85 C, 1.62 V)
```

Both rows above share one profile (same owner/test/parameters) but are two
distinct evidence selections — the same profile can have selected evidence at
more than one PVT point. Symmetrically, two different profiles (different
declared parameters, or a different test entirely) can each have selected
evidence at the very same process corner. All three PVT fields — corner,
temperature, voltage — are required together whenever newly stored evidence
is profile-scoped; a partial PVT point is rejected before any write. Their
values are compared exactly (no tolerance), and `corner`/`temperature`/
`voltage` select the operating point. Declared test parameters may not use
the reserved names `process_corner`, `temperature_c` or `supply_voltage_v`.

### Declaring a scoped test

The runtime descriptor remains metadata and never changes a deck. Temperature
configuration lives separately on the executing Module/testbench:

```python
test = amp.create_test('load_step', parameters={'cload_f': 2e-12})
# Wire the DUT, load, stimulus and a 1.8 V supply first.
test.set_temperature(27.0)
test.set_supply_voltage(1.8)
report = test.run(corner='tt')
```

A complete descriptor requires each of process corner, temperature and supply
to be fixed or proven not to apply (`ConditionState.NONE`). A run configured
with only temperature can therefore be scoped when corner and supply are
proven not to apply. An unknown or swept fact leaves the report unscoped.
`run(temperature=...)` and `run(voltage=...)` are both deprecated and only
assert matching explicit setup; neither sets anything. External-file
`evaluate(rawfile=..., temperature=..., voltage=...)` accepts either declaration
alone and records external origin. On a fresh file load, omitted facts remain
unknown. These declarations do not configure the simulator or supply. See
[operating conditions](api-reference.md#operating-conditions) for migration,
backend limits, cache behavior and the distinction between TEMP and TNOM.

For evidence with an `operating_conditions` snapshot, `store_report` and the
executing Module's `store_net_currents` use exactly that available PVT. Storage
arguments may confirm known values but cannot override them or fill unknowns;
contradictions raise `EvidenceMismatchError` before writes. `set_corner` and
prior live measurements cannot supply missing values. For a report without a
snapshot, supported explicit `store_report()` arguments still supply temperature,
voltage, analysis type and tool; existing corner resolution and configured
temperature/voltage storage defaults still apply.
Missing captured observations never authorize a lookup of live measurements,
including measurements recorded by hand before building the report.

The compact snapshot is stored in metric and compatibility rows under
`conditions['silk_operating_conditions']`, merged with other caller conditions.
A conflicting value of this reserved key fails. It describes provenance and
does not change profile identity or schema. Spec/current rows retain numeric PVT
but no separate origin/basis field; reading those rows alone does not prove
that silk executed the run. Historical rows lacking the marker remain
provenance-unknown.

A second reserved key, `conditions['silk_metric_warnings']`, holds every
`silk.metrics.diagnostics.MetricWarning` raised while that row's own metric
was extracted (see [report capture and
persistence](metrics-library.md#report-capture-and-persistence-of-metricwarning)),
as a list of `{metric, code, message, details, suggestion, output}` dicts.
Unlike `silk_operating_conditions` (one shared snapshot copied onto every
metric row in the report), this key is per row: only the metric that actually
warned carries it, and a metric with no warnings writes exactly the
`conditions` every other metric in the report gets. Setting it explicitly on
`store_report(conditions=...)` raises `EvidenceMismatchError` — it is
computed from `SpecReport.metric_warnings`, never caller-supplied. Older rows
may omit the warning key. Warnings do not change how
`latest_only=True` groups observations.

`store_report()` then persists metrics, compatibility pass/fail rows, and
first-class spec measurements together, all under the same derived/reused
profile and exact PVT, in one transaction:

```python
amp.store_report(report, tool_name='ngspice-44')
```

The default `source=None` takes its value from the report evidence:
`behavioral` for a recorded behavioral backend, otherwise `spice`. Changing
the module's simulator after a run does not relabel that report. An explicit
`source=` that contradicts its recorded backend raises `EvidenceMismatchError`.
Reports without backend evidence still accept caller-supplied sources.

Passing an explicit `corner=`/`temperature=`/`voltage=` to `store_report()`
that disagrees with the run's own context raises `EvidenceMismatchError`
rather than silently preferring one value.

For an **unscoped** report with an operating-conditions snapshot, storage
uses the snapshot's condition values, including `ConditionState.NONE`,
unknowns and sweeps. `report.corner` records the run label, which can differ
from the evidence's corner when process corner does not apply.

For a report without a snapshot, `store_report()` resolves the corner from
its explicit `corner=` argument, else `report.corner`, else the tracker's
current corner (`set_corner()`, `'tt'` by default). An explicit corner that
conflicts with a recorded report corner raises `EvidenceMismatchError`.
A SPICE run at `'ff'` therefore cannot silently be stored under `'tt'`.

Currents follow the same context. Call `store_net_currents()` on the
generated testbench netlist (`test.netlist` for a `ModuleTestbench` run) once
it carries `save_device_currents = True` and has been annotated:

```python
current_report = test.netlist.annotate_currents(tstart=1e-6)
test.netlist.store_net_currents(
    current_report, corner='tt', db_path='designs.db',
    tool_name='ngspice-44',
)
```

PVT comes from the executing Module's snapshot, including unknown values.
Without a snapshot, manual/legacy Modules retain the existing optional declared
`CharacterizationContext` and storage defaults. A descriptor conflicting with a
snapshot is rejected before writes; a non-`CharacterizationContext` descriptor
also fails.

Always close a module that opened tracking state:

```python
amp.close()
test.netlist.close()
```

## Run and store in one call: `run_test(store=True)`

Iterate with no database writes, then persist the runs you decide to keep.
`Module.run_test(name, ..., store=True)` runs one registered testbench and
files its `SpecReport` through the **same** `store_report()` contract shown
above — metrics, compatibility pass/fail rows and first-class spec
measurements, one transaction — and returns the `SpecReport` with the
storage receipt on `report.storage`. The default (`store=False`, or omitting
it) runs without database writes. You can still call `store_report(report)`
afterwards.

```python
import os

from silk import Module, load_pdk, sources
from silk.db import Database
from silk.elements import elements
from silk.metrics import dc_gain_dB
from silk.simulation import AC

OUT = os.path.join(os.environ.get('SILK_DOC_OUT', '.'), 'run_store_demo')
DB_PATH = os.path.join(OUT, 'designs.db')
os.makedirs(OUT, exist_ok=True)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)                     # disposable demo file

pdk = load_pdk('sky130')

# 1. The authoritative design: a common-source stage with one declared spec.
amp = Module('PDK_skywater130', OUT, 'cs_amp.cir', pdk_details=pdk,
             module_name='cs_amp', db_path=DB_PATH)
amp.define_module_nets({'vin': None, 'vout': None, 'vdd': None})
with amp:
    pdk.nmos1p8('mn', d='vout', g='vin', s='0', b='0', w=4.0, l=0.5)
amp.add(elements.resistor('rl', p='vdd', n='vout', value=50e3))

tb = amp.create_test('ac')
tb.add_module_under_test({'vin': 'vin', 'vout': 'vout', 'vdd': 'vdd'})
tb.add_source(sources.VoltageSource('vsup', positive='vdd', negative='0', dc=1.8))
tb.add_source(sources.VoltageSource('vs', positive='vin', negative='0',
                                    dc=0.65, ac=1.0))
tb.add_sim_type(AC(variation='dec', points=20, fstart=1, fstop=1e9))
tb.add_metric('gain_dB', dc_gain_dB, sig=tb.probe('vout'))
gain = tb.add_spec('DC gain', 'gain_dB', '>=', 10.0, key='gain')

# 2. Explicit version preparation -- run_test never does this for you.
version_id = amp.create_circuit_version('CS stage, first cut')

# 3. Select the supply explicitly: vsup and vs are both grounded DC rails.
tb.set_supply_voltage(1.8, source='vsup')
tb.set_temperature(85)
report = amp.run_test('ac', corner='ff', store=True)
print(report.summary())
print('receipt:', report.storage)

# 4. Supply and TEMP remain configured at 1.8 V and 85 C. Omitting corner
#    selects the PDK default (tt), not the previous run's ff corner. Both
#    runs are scoped. Every store appends; nothing is deduplicated.
default_report = amp.run_test('ac', store=True)
print('default corner:', default_report.corner, default_report.storage)

# 5. Read it back through the same module ...
(scoped,) = amp.query_results(parameter='gain_dB', corner='ff',
                              temperature=85, voltage=1.8)
print(f'scoped: gain_dB={scoped.value:.2f} @ {scoped.corner}/'
      f'{scoped.temperature} C/{scoped.voltage} V, '
      f'profile={scoped.profile_id}')
(default_scoped,) = amp.query_results(parameter='gain_dB', corner='tt',
                                      temperature=85, voltage=1.8)
print(f'default scoped: gain_dB={default_scoped.value:.2f} @ '
      f'{default_scoped.corner}/{default_scoped.temperature} C/'
      f'{default_scoped.voltage} V')
stored_gain = next(r for r in amp.get_stored_specs() if r.spec_key == 'gain')
print('stored spec measurements:',
      [(m.corner, round(m.value, 2), m.passes, m.profile_id is not None)
       for m in stored_gain.measurements])
amp.close()
tb.netlist.close()

# ... or later, from the file alone, with only the version UUID.
with Database(DB_PATH) as db:
    rows = db.get_simulation_results(version_id, parameter='gain_dB')
print('from the file:', [(r.corner, round(r.value, 2)) for r in rows])
```

Run as-is with installed sky130 models and ngspice, resolved through silk's
settings (`.venv/bin/python -m silk.doctor` shows the paths; `SILK_DOC_OUT` picks the
output directory). It prints the `PASS` summary, the receipt
`{'metrics_stored': 1, 'specs_stored': 1, 'skipped': {}, 'cached': False}`,
then the two stored measurements — observed with ngspice 42 as
`('ff', 17.48, True, True)` scoped under a derived profile at
(ff, 85 °C, 1.8 V) and `('tt', 14.69, True, True)` scoped at
(tt, 85 °C, 1.8 V) — and the same two rows read back from the closed file.
Omitting run arguments does not clear the testbench's supply or temperature
configuration; the second run retains both setters while selecting the PDK's
default corner. Numeric gain values can vary with model/simulator versions.

What the flag does, and does not, do:

- **Explicit version preparation.** `Module(db_path=...)` and
  `create_circuit_version()` are yours to call. `run_test(store=True)` never
  creates, rebinds or promotes a version. Missing `db_path` raises
  `RuntimeError`; a missing bound version raises `DatabaseError` naming
  `create_circuit_version()`. Both fail **before** the simulator starts, and
  so do a locked version (`LockedVersionError`), an
  unsealed one (`DatabaseError`) and a live design that has drifted from the
  bound snapshot since it was sealed (`EvidenceMismatchError`) — every
  readiness failure that can be known without a report is rejected before
  the expensive run.
- **Temperature comes from execution.** Use the testbench's `set_temperature`
  before running. Legacy `temperature=` only asserts matching explicit setup.
  Verified fixed TEMP plus an established supply gives complete scoped PVT;
  otherwise partial conditions are stored unscoped, without tracker fallback.
  The report's recorded corner is authoritative. The supply comes from
  `set_supply_voltage` and is confirmed against the launched deck; legacy
  `voltage=` only asserts it.
- **One source of provenance.** `store_options` forwards `store_report`'s
  `tool_name`, `units`, `sim_type`, `source` and `conditions`; a `corner`,
  `temperature` or `voltage` key there is rejected (`ValueError`) rather than
  letting it agree or disagree with the run.
- **The receipt.** `report.storage` is `store_report()`'s
  `metrics_stored`/`specs_stored`/`skipped` plus `cached`. It is `None` on
  every report that was not stored through the flag — including a report
  you later pass to `store_report()` yourself.
- **Failing specs are evidence.** A failed spec check is stored with its
  `fail` status row and numeric measurement. An extractor that raised leaves
  its metric in `skipped` and a `fail` status row, with **no** numeric
  measurement fabricated; an aborted simulation raises `SimulationAborted`
  from the run and nothing is stored.
- **A failed save raises `ReportStorageError`.** The run completed but
  `store_report()` failed (a stale-report conflict, a lock taken meanwhile,
  a disk error). The exception carries the completed report on `.report`
  and the original error as `__cause__`; the per-report transaction has
  rolled back every row it wrote, and the live `Measurement` objects the run
  already recorded on the module's Specs stand — the database did not make
  them and cannot unmake them.
- **Append on repeat.** Storing the same test twice — two `store=True` runs,
  or a `store=True` run followed by `store_report(report)` — appends two
  observations, exactly as two `store_report()` calls would.
- **Report caching is disabled.** `ModuleTestbench(cache=True)`
  warns at construction and then behaves exactly like `cache=False` — every
  `run_test` simulates, and its `store=True` receipt always says
  `cached=False`. The parameter, `clear_cache()` and the `cached` receipt
  field are kept for a future working cache.
- **What is not stored.** Numeric metrics named like `i_supply_uA` are
  stored as metrics like any other. Annotated net-current profiles are a
  separate table with their own lifecycle: run `annotate_currents()` and
  call `store_net_currents()` on the testbench netlist (see "Selected current
  sets and their lifecycle"). Raw waveforms are never persisted.
- **Out of scope.** `run_all_tests()`, `run_corners()` and the explicit
  `store_report(report)` path are unchanged; passing `store=True` to the
  batch helpers is rejected by `ModuleTestbench.run` like any unknown kwarg.

## Selected current sets and their lifecycle

Every selected current set — profile-scoped or unscoped — has its own stable
`current_set_id`. Storing a new set for the same selection key does not
overwrite the old one in place:

1. the old set's rows are never rewritten; its state changes to
   `'superseded'`;
2. a brand-new current set is inserted with a fresh `current_set_id` and a
   `supersedes_current_set_id` link back to the one it replaces;
3. exactly one current set stays `state='current'` at each selection key.

The selection key is `(circuit_version_id, corner)` for unscoped evidence, or
`(circuit_version_id, profile_id, corner, temperature, voltage)` for
profile-scoped evidence — so a profile-scoped replacement at one PVT point
never disturbs another PVT point's selected set, and an unscoped replacement
at one corner never disturbs another corner.

The practical effect: **a `current_set_id` any code already holds — in a
variable, in a printed report, in a future downstream table — keeps reading
back correctly even after the set it names is no longer the selected one.**
Nothing deletes it and nothing repoints it. See "Migrating a schema-v1 database" below for how existing v1 currents
become stable current sets.

```python
from silk.db import Database

with Database('designs.db') as db:
    origin_version_id, rows = db.get_net_current_set(
        version_id, current_set_id=old_current_set_id
    )
```

An explicit `current_set_id` reads exactly that historical set — superseded
or current — without any corner/profile/PVT selector.

## Reopen and query history without a `Module`

Keep the returned version UUID. It is the unambiguous historical handle:

```python
from silk.db import Database

with Database(db_path) as db:
    metrics = db.get_simulation_results(
        source_id, parameter='gain_dB', corner='tt'
    )
    declarations = db.get_stored_specs(source_id)
    current_origin, current_rows = db.get_net_current_set(source_id, 'tt')

    # Profile-scoped: resolve the profile from the same declared facts a
    # caller remembers (owner path, test name, declared parameters), or by
    # its exact profile_id.
    from silk.db.context import CharacterizationContext
    context = CharacterizationContext(
        owner_path='$', test_name='load_step', parameters={'cload_f': 2e-12},
        corner='tt', temperature=27.0, voltage=1.8,
    )
    profile = db.get_characterization_profile(source_id, context=context)
    scoped_metrics = db.get_simulation_results(
        source_id, profile_id=profile.profile_id, corner='tt'
    )
```

Metric queries return immutable `SimulationResultRecord` objects. Declaration
queries return immutable `SpecDeclarationRecord` objects containing
`SpecMeasurementRecord` history. Current lookup returns
`(origin_version_id, rows)` with `sqlite3.Row`
entries; use keys such as `row['profile_id']` and `row['state']` to inspect
their shared operating context and selection state. Metric and measurement
records carry a nullable `profile_id` (declaration records do not): `None`
means explicitly unscoped/legacy/manual
evidence, never a guessed identity.

`get_simulation_results()`, `get_stored_specs()`, `query_results()`, and
`restore_spec_measurements()` all accept a `profile_id` filter that is
three-state: omit it (the default) for no filter — both scoped and unscoped
evidence match, exactly as before profiles existed;
`silk.db.UNSCOPED_PROFILE` for explicitly unscoped rows only; an exact
`profile_id` string for one profile only.

## Rebuild and restore onto live Python

Rebuild the same authoritative definition, bind it by resolving the sealed
identity, then query or restore:

```python
rebuilt = build_amp(pdk, db_path='designs.db')
assert rebuilt.create_circuit_version() == source_id

metrics = rebuilt.query_results(
    parameter='gain_dB', source='spice', corner='tt', latest_only=True
)
restored_specs = rebuilt.restore_spec_measurements(
    source='spice', corner='tt', tool_name='ngspice-44'
)
restored_currents = rebuilt.load_net_currents('tt')
```

`load_net_currents(corner)` loads only unscoped current sets; use
`restore_characterization_evidence(context=...)` for profile-scoped currents.

`latest_only=True` returns the latest row per complete context, including
profile, tool and conditions; one metric at one PVT can still yield several rows.

Metric applicability depends on structure identity. Spec-measurement restore
also requires the same stable declaration location and definition. Current
restore validates every row and every live pin before mutating anything.

`load_net_currents()` preserves existing user/SPICE annotations unless
`force=True`. It clears stale values from an earlier DB load when a later
complete set no longer contains those rows. A partial net has pin rows but no
aggregate row, so its `Net.avg_current`, `rms_current`, and `peak_current`
remain `None`—never a plausible-looking zero or partial sum.

### Source-follower calibration across Python processes

The [source-follower calibration example](../silk/examples/source_follower_calibrate.py)
composes measurement persistence with the behavioral calibration bridge. From
the repository root, run these as **two separate commands** (each Python
process exits):

```bash
MPLBACKEND=Agg .venv/bin/python -m silk.examples.source_follower_calibrate characterize --database follower.db
MPLBACKEND=Agg .venv/bin/python -m silk.examples.source_follower_calibrate restore --database follower.db
```

Characterization requires ngspice and sky130 models. Both phases rebuild the
same nominal `build_source_follower()` and attach the same `CalibrationBinding`
recipe. They explicitly select root owner `$`, tests `ac` and `beh_check`,
27 C, 1.8 V, and the `--corner` argument (default `tt`). The transient profile
also declares its low input and averaging-window parameters. An explicit
27 C records ngspice's default ambient without changing the operating point.

The database holds SPICE AC gain in V/V and bandwidth in Hz, their spec
observations, and the transient's average VOUT in V over [10 ns, 19 ns], before
the 1 V input steps. The stored transient conditions describe the derivation:
`offset = gain * input_low_V - settled_vout_V`. Offset is derived from those
primitives; it is not a separately simulator-certified observation.

Restore resolves the complete producing profiles, queries their latest SPICE
observations at that exact point, restores AC spec measurements, reconciles the
authored gain/bandwidth bindings, and derives offset with the same formula.
Then it runs only the behavioral step bench. It needs neither an installed
simulator nor readable model libraries, but keeps the configured library path
for deck comparison. Changed library-path/settings or bench decks can therefore
refuse as stale even when no simulation is requested. Deck checks compare
emission, not model-file contents or simulator installations. Their `checked`
or `not checked` state and warnings remain visible. Missing, incompatible,
incomplete or ambiguous evidence raises; restore never characterizes as a
fallback. Repeated characterization appends evidence; latest compatible rows
within the explicitly selected profiles are used.

This restores observations onto rebuilt Python and reapplies authored bindings;
it does not reconstruct Python or arbitrary fitted models. No parameters are
loaded from a sidecar. Optional `--validation-output reference.json` on either
phase writes parameter and waveform comparison evidence only; restore never
reads it. Running without arguments produces a SPICE/behavioral overlay
in one session.

### Profile-scoped selection and restore

Four methods on a bound `Module` or its `CircuitTracker` cover profile-scoped
selection and restoration. The first three also have `Database` counterparts
for inspecting stored history without a live module. Restoration requires the
bound live module:

```python
profiles = rebuilt.list_characterization_profiles()
profile = rebuilt.resolve_characterization_profile(context=context)
evidence = rebuilt.select_characterization_evidence(
    context=context, corner='tt', temperature=27.0, voltage=1.8,
)
result = rebuilt.restore_characterization_evidence(
    context=context, corner='tt', temperature=27.0, voltage=1.8,
)
```

- `list_characterization_profiles()` lists every profile compatible with the
  rebuilt structure, nearest lineage first — with no selector at all; filter
  the returned list client-side when you only know a test name or part of a
  parameter map, rather than guessing at a restore selector.
- `resolve_characterization_profile()` resolves exactly one profile — by
  exact `profile_id` or by a *complete* `CharacterizationContext`, never by a
  partial one. If the profile exists only on a structurally incompatible
  circuit version elsewhere in the lineage, it raises `EvidenceMismatchError`
  rather than silently skipping to a plausible-looking result.
- `select_characterization_evidence()` returns one exact, read-only
  `CharacterizationEvidenceRecord` bundling that profile's compatible
  metrics, spec measurements, and selected current set at the exact PVT —
  without touching the live module. It requires **exactly one** of
  `profile_id=` or `context=` (both or neither raises
  `EvidenceAmbiguityError`/`ValueError`), and it requires the profile to have
  a *selected* (not merely superseded) current set at that exact PVT.
- `restore_characterization_evidence()` does the same selection, then
  applies it atomically: every measurement and current row is validated
  against the live design *before* any live object is mutated, so a failure
  partway through never leaves the module half-updated.

`select_characterization_evidence()` and `restore_characterization_evidence()`
both expose the concrete `profile_id`, `current_set_id`, and
`origin_version_id` that produced the answer — so downstream code (a future
layout evidence table, a printed report) can say exactly which selected
evidence it consumed.

## Four things you can always ask

The typed records above exist specifically to answer these four questions
without ever guessing:

**Which declared characterization context produced this metric or spec
measurement?** Every `SimulationResultRecord`/`SpecMeasurementRecord` carries
`profile_id`; resolve it with `get_characterization_profile()`/
`resolve_characterization_profile()` to get back the owner path, test name,
and declared parameters that produced it. `profile_id is None` is an honest
"unscoped," never a fabricated test identity.

**What selected evidence exists for profile P at this exact PVT?**
`select_characterization_evidence(profile_id=P, corner=..., temperature=...,
voltage=...)` (or the `context=` form) returns the compatible metrics, spec
measurements, and current set together — the profile/PVT current-set unique
index guarantees at most one selected current parent for that key.

**Where did restored evidence originate after promotion?**
Metric, spec-measurement and current-set records carry `origin_version_id`.
An aggregate `CharacterizationEvidenceRecord` also names the queried
`target_version_id`. Promotion preserves each observation's origin (see
"Promotion and evidence origin" below).

**Was current set S replaced, and can a downstream reference still use
it?** Read it directly by ID: `db.get_net_current_set(version_id,
current_set_id=S)` returns `(origin_version_id, rows)`; each row's
`row['state']` is `'current'` or `'superseded'`. A superseded set's `current_set_id`
never becomes invalid and its rows are never deleted or rewritten.

## Promotion and evidence origin

`CircuitTracker.promote()` advances a stored design stage and does not require
passing specs. It is not a readiness check; inspect test and spec evidence
before deciding to advance.

Promotion creates a sealed child UUID with the same snapshot identity and a
parent link. Repeating the same promotion from the same source with the same
target label reuses that UUID; conflicting source or label combinations raise
`DatabaseError`. It does not copy evidence and pretend it was observed again:

```python
stable_id = rebuilt.promote('stable')
record = rebuilt.query_results(
    parameter='gain_dB', source='spice', corner='tt', latest_only=True
)[0]
assert record.origin_version_id == source_id
```

Compatible reads search the bound version first, then its parent lineage. A
spec-only successor may reuse ancestor metrics and currents because structure
is unchanged; it may reuse a spec measurement only when that declaration's
stable location and definition are unchanged. A structural successor cannot
reuse ancestor evidence. Promotion locks the child by default, and a later
write raises `LockedVersionError`; reads remain available.

Profile-scoped evidence follows the identical rule: a child version does not
get its own copy of a parent's profile. `list_characterization_profiles()`/
`resolve_characterization_profile()`/`select_characterization_evidence()`
walk compatible lineage and return the profile's true origin version. If a
child stores new evidence under the same owner/test/parameters, it gets its
own new profile row (same scheme/digest, new `profile_id`, new origin
version) — exact-ID selection is then unambiguous, and a descriptor-based
lookup selects the nearest compatible version in the lineage.

## Unscoped and legacy evidence

You can retain report A, run B, then call `store_report(A)`: storage uses A's
detached original spec observation, including its precise timestamp and attempt
identity, whether B has equal or different values. Saving/loading A as standalone
report JSON preserves that evidence. Repeated stores append rows; they do not
rewrite observations. Circuit/spec declaration and provenance checks still apply.

Any report without captured observations remains storable, including manual
reports, Monte Carlo samples, fault-campaign scoring reports and JSON saved by
older versions. Its supplied values, operating conditions, provenance and
supported explicit storage arguments are retained. Storage never borrows live
measurement time, run ID, notes, temperature, voltage, analysis type or tool,
even when the live value is equal. Without report-owned evidence or supported
explicit arguments or configured storage defaults, context fields are SQL NULL;
missing observation time and run ID are SQL NULL, and missing notes are empty
text. There is no observation timestamp argument, and save time is never
substituted for observation time.

Unknown observation times sort before known times and restore to the existing
live unknown representation, `0.0`. Storing an untimestamped report later does
not make it the latest evidence ahead of timestamped observations. This does not
change the separate simulation-result row insertion timestamp. Existing snapshot
validation still prevents explicit arguments from overriding report-owned
conditions or filling conditions that the snapshot records as unknown.

A plain top-level `Module`, a direct `store_results()` call, and every v1
database migrated with `migrate_v1_to_v2()` (below) stay unscoped — every
associated `profile_id` is `NULL`. Silk never substitutes `default_test`, a
filename, a module name, `$`, an empty parameter map, or PVT defaults to make
an unscoped call look scoped. `store_results()`/`store_result()` never grow a
`profile_id` parameter, and an arbitrary `conditions=` dict shaped like a
characterization context (even one that reuses the reserved field names)
never fabricates a profile — it is stored as free-form text, nothing more.

Unscoped and profile-scoped evidence coexist safely. Measurements from
different profiles or placements remain separate; use `record_measurement()`
and the lookup methods rather than constructing slot keys.

## Automatic deck checks through a built design

`Module.query_results`, `select_characterization_evidence`,
`restore_spec_measurements` and `restore_characterization_evidence` (and their
bound tracker delegates) check stored evidence automatically. The promise is: **This stored row's producing test, at its recorded
launch corner, can emit the same normalized deck under the rebuilt design's
current settings, with the same backend.** Resolve the exact root-relative
test on the rebuilt design;
silk never reconstructs tests or absent parents from stored data.

For temperature/supply lists, the checker expands the current test plan at the
recorded launch corner and compares against its current point decks. Adding or
reordering other points preserves an unchanged point's evidence. Removing or
replacing a point makes its evidence stale unless a remaining configuration
emits the same deck. Stored observed temperatures and voltages are not replayed
as configuration: an observed default temperature does not become an explicit
TEMP setting. Inherited lists and explicit clears follow the same rules as a
run. Planning and point configuration use isolated emission views; lookups do
not prepare decks, simulate, create output folders or change the live bench.

To replace stale grid evidence, run the current plan and store its reports:

```python
for report in rebuilt.get_test('gain').run(corner='tt').values():
    rebuilt.store_report(report)
# For a nested test, select it with instance='parent.cell'.
```

`run_test(store=True)` accepts one scalar report and refuses declared
collections; use the plan/store loop above for a grid or declared corner list.

```python
history = rebuilt.query_results(parameter='gain_dB')  # latest_only=False
for row in history:
    print(row.value, row.deck_check.state, row.deck_check.reason)
latest = rebuilt.query_results(parameter='gain_dB', latest_only=True)
restored_count = rebuilt.restore_spec_measurements(source='spice', corner='tt')
```

The immutable `silk.db.DeckCheck` has `state` and `reason`. Returned metric and
spec-measurement records, current sets and aggregate records carry a trailing
`.deck_check`. It is a snapshot from that lookup, excluded from record equality
and hashing. Silk does not persist it in SQL, provenance, JSON/CSV exports or
report schemas; caller `dataclasses.asdict` or pickle may still include it.
Raw `Database` reads leave it `None`, remain unchecked history, and emit no new
warning. Other historical APIs such as `get_stored_specs` do not run this check.

| State | Meaning and behavior |
| --- | --- |
| `checked` | Current producing deck and backend match the accepted launch. |
| `stale` | Deck or backend differs. Default history returns flagged rows; `latest_only=True` raises if a selected latest row is stale. |
| `not checked` | No comparable launch/test/deck, with an actionable reason. Values stay usable with one combined `UserWarning` per public call. |

Latest-result queries do not fall back to an older checked row. An older stale row superseded by a matching latest row does not
block that latest request. Read-only aggregate selection returns flagged
history even when a member is stale. Restores check the complete selected set
before staging, hydrating or applying: one stale member raises
`EvidenceMismatchError` naming the producer and rerun command, without partial
mutation and before any not-checked warning. Uncheckable observations restore
with the combined warning. A restored Spec may PASS its target even when the
simulation is not checked; this policy is disclosure, not a reconciliation gate.

Every aggregate selection and successful aggregate restore emits that one
warning, because current annotations are not frozen at launch and current sets
are always `not checked`. The aggregate is `stale` if a metric/spec member is
stale, otherwise `not checked`; it is never wholly `checked`. The legacy
`load_net_currents` API remains outside this automatic check.

Coverage excludes manual values/reports, external rawfiles, behavioral,
Monte Carlo, stochastic and opaque/custom decks, missing/renamed/ambiguous tests
and registered tests that are unwired in this session (no DUT placement or
netlist; wire the bench to check it). Missing root context, old evidence
without launch facts, unsupported normalization
schemes, and probe requests needing a preflight simulation also prevent
checking. Failed or incomplete
launches do not publish an accepted digest. Genuine invalid current settings
keep their typed error instead of being treated as unknown. Nothing backfills
or relabels old provenance; rerun a covered SPICE bench to obtain checkable data.

Eligibility is checked before definition validation: opaque/custom waveform
renderers or source waveform selectors are never invoked to decide a lookup's
coverage. Unmodified source subclasses retain the built-in selector. Covered current
emission uses the same whole-deck definition validation and name
resolution as ordinary preparation, on an isolated view. Conflicting bodies
claiming one subcircuit name raise `CircuitError`; they cannot be certified by
silently emitting one representative. Identical bodies and declared variants
retain their ordinary preparation behavior.

Only silk-generated file locations and the fallback title are normalized.
An identical rebuild in another output folder can check; moved absolute PDK/
OSDI paths or changed silk emission may make old rows stale. Metric definitions
and user Python, simulator versions and model-file contents are outside the
comparison; existing provenance still records its usual version/model facts.
Current deck emission happens once per current point configuration for each
distinct test/placement/launch label/backend per public call, with no persistent
or report-cache reuse of verdicts.

## Run provenance (schema v4)

Every testbench-built `SpecReport` carries `report.provenance`, a
`silk.simulation.provenance.RunProvenance` (field contract in
[reports.md](reports.md#provenance)). `store_report()` and
`run_test(store=True)` store that record once in `run_provenance` and set
`provenance_id` on every `simulation_results` and `spec_measurements` row
the call writes; `store_net_currents()` does the same for the current set it
writes. Current annotations' storage provenance does not establish a
launch-frozen producing identity, so current sets remain not checked.

| Table / column | Meaning |
|---|---|
| `run_provenance(provenance_id, scheme, digest, record_json, created_at)` | One row per distinct record. `(scheme, digest)` is unique: `digest` is the record's `RunProvenance.digest()` (sha256 of its canonical JSON, `timestamp_utc` excluded), so every row of one run, and re-stores of the same run, share one row. `record_json` is the full record as stored first. |
| `simulation_results.provenance_id`, `spec_measurements.provenance_id`, `net_current_sets.provenance_id` | Nullable foreign key to `run_provenance` (`ON DELETE RESTRICT`). `NULL` = no provenance recorded: a hand-built report, or any row that predates schema v4. |

Circuit versions carry no provenance column: a version is a structure that
many runs attach evidence to, and it reaches its provenances through its
measurement rows.

Reading it back: `SimulationResultRecord.provenance_id`,
`SpecMeasurementRecord.provenance_id` and `NetCurrentSetRecord.provenance_id`
name the row, and `Database.get_run_provenance(provenance_id)` returns the
`RunProvenance` (`None` for a `None` id). `Database.store_run_provenance(record)`
is the content-addressed writer the store paths use.

Provenance `silk.provenance/v3` adds `normalized_deck_sha256` (versioned
`silk.normalized-deck/v1:<hex>`), `producing_test` (`owner_path`, `test_name`)
and `launch_corner_label`; backend uses the existing `simulator` field. These
facts travel with the accepted launch and each logical report, including worker
and shared runs. Running A, preparing B, and later storing or re-evaluating A
still records A. `deck_sha256` keeps its exact prepared-file meaning. There is
no SQL migration for this addition; older provenance reads with absent facts
and remains uncheckable, preserving its original scheme and history.
For v1/v2, serialization and `digest()` retain the pre-v3 field inventory and
omit the three producing fields. Records stored with those pre-v3 content
addresses retain their original IDs and SQL JSON when loaded and re-stored.
In v3, those facts contribute to the provenance content address, which is
distinct from the normalized deck digest itself.

## Migrating a schema-v4 database

All copy migrations open the source read-only and leave it byte-identical.
The destination must be distinct and initially nonexistent. Each migration
validates the temporary copy, including foreign-key integrity, before atomic
publication; failure leaves no partial published database. Publication refuses any existing destination, including one created concurrently.

```python
from silk.db import Database, migrate_v4_to_v5, migrate_v5_to_v6, migrate_v6_to_v7

migrate_v4_to_v5('designs_v4.db', 'designs_v5.db')
migrate_v5_to_v6('designs_v5.db', 'designs_v6.db')
migrate_v6_to_v7('designs_v6.db', 'designs_v7.db')
with Database('designs_v7.db') as db:
    pass  # Query or restore evidence through the normal API.
```

The first migration creates a
fresh v5 file, with the same tables as a newly created v5 database; every row
keeps its rowid, and indexes or triggers the source adds beyond silk's own
are recreated. Each evidence row gets its states by one rule: a non-`NULL`
corner, temperature or voltage is `fixed` with its value kept, and `NULL` is
`unknown` -- **never `none`**, because none is a proof made at launch and a
v4 row keeps no deck to re-prove it from. A `NULL` temperature on a
`simulation_results` row whose recorded operating conditions say
`temperature_kind == 'sweep'` becomes `sweep`. Each recorded
`silk_operating_conditions` marker gains the keys derivable from it:
`corner_kind` and `supply_kind` (`fixed` when the marker's value is set,
otherwise `unknown`) and `corner_basis` (`legacy` for a fixed corner, whose
basis was never recorded). Keys a marker already has are kept.

Characterization profiles are copied unchanged, and every scoped row was
complete in v4, so each existing profile keeps its identity and its evidence
is selected at the same corner/temperature/voltage as before. After migration, `get_simulation_results(latest_only=True)` groups rows by their
recorded operating conditions, so a pre-upgrade row (`corner_basis` `legacy`)
and a re-run of the same evidence after the upgrade show side by side once.

## Migrating a schema-v3 database

```python
from silk.db import Database, migrate_v3_to_v4, migrate_v4_to_v5, migrate_v5_to_v6, migrate_v6_to_v7

migrate_v3_to_v4('designs_v3.db', 'designs_v4.db')
migrate_v4_to_v5('designs_v4.db', 'designs_v5.db')
migrate_v5_to_v6('designs_v5.db', 'designs_v6.db')
migrate_v6_to_v7('designs_v6.db', 'designs_v7.db')
with Database('designs_v7.db') as db:
    pass  # Query or restore evidence through the normal API.
```

A private copy
gains the empty `run_provenance` table and the nullable `provenance_id`
column on the three tables, with every existing row left `NULL` -- silk never
invents provenance for evidence recorded before it existed. All rows, rowids,
identities, foreign keys and indexes survive. A migrated file declares exactly
the same tables as a fresh v4 file.

## Migrating a schema-v2 database

```python
from silk.db import Database, migrate_v2_to_v3, migrate_v3_to_v4, migrate_v4_to_v5, migrate_v5_to_v6, migrate_v6_to_v7

migrate_v2_to_v3('designs_v2.db', 'designs_v3.db')
migrate_v3_to_v4('designs_v3.db', 'designs_v4.db')
migrate_v4_to_v5('designs_v4.db', 'designs_v5.db')
migrate_v5_to_v6('designs_v5.db', 'designs_v6.db')
migrate_v6_to_v7('designs_v6.db', 'designs_v7.db')
with Database('designs_v7.db') as db:
    pass  # Query or restore evidence through the normal API.
```

A private temporary backup is rebuilt with nullable `spec_measurements.value`,
preserving existing rows, rowids, foreign keys, indexes and
characterization-profile identities.

## Migrating a schema-v1 database

Opening an existing schema-v1 file with `Database(v1_path)` raises
`UnsupportedSchemaVersionError` and names the fix; it is never upgraded
implicitly. Migrate it explicitly to a **distinct, non-existing** destination:

```python
from silk.db import migrate_v1_to_v2, migrate_v2_to_v3, migrate_v3_to_v4, migrate_v4_to_v5, migrate_v5_to_v6, migrate_v6_to_v7

migrate_v1_to_v2('designs_v1.db', 'designs_v2.db')
migrate_v2_to_v3('designs_v2.db', 'designs_v3.db')
migrate_v3_to_v4('designs_v3.db', 'designs_v4.db')
migrate_v4_to_v5('designs_v4.db', 'designs_v5.db')
migrate_v5_to_v6('designs_v5.db', 'designs_v6.db')
migrate_v6_to_v7('designs_v6.db', 'designs_v7.db')
```

- Every existing UUID and every existing fingerprint scheme/digest pair is
  copied exactly — nothing is relabelled or recomputed.
- Every v1 metric and spec measurement becomes an honestly **unscoped**
  (`profile_id IS NULL`) row. v1 has no authoritative owner/test/parameter
  descriptor to derive a profile from, so silk does not invent one.
- Every v1 corner's `net_currents` rows become one stable, unscoped current
  set (a new `current_set_id`, `state='current'`, no predecessor) —
  preserving exactly the evidence a v1 caller already treated as one
  selected set. If a corner's rows are internally inconsistent (mismatched
  analysis, window start, results-file path, temperature, voltage, tool, or
  structure identity within the same version/corner group — a sign of file
  corruption, since the v1 write path itself only ever wrote one consistent
  group per corner), the whole migration aborts atomically before writing
  anything, naming every offending `(circuit_version_id, corner)` group and
  its inconsistent field(s).
- Validation includes a zero-profiles check.

`migrate_v1_to_v2` retains its original name, signature and actual v2 result,
including `spec_measurements.value NOT NULL`. Use the later copy steps above
to open the evidence with the current `Database` API. All source and
intermediate files survive.
Schema migration preserves historical `silk.structure/v1` and
`silk.structure/v2` tags verbatim; it does **not** upgrade structure identity
or make old evidence compatible with a live v4 design. Identity-dependent reads of those rows raise as described
below. Rows already using the current fingerprint scheme retain their normal
readback behavior.

<a id="migrating-stored-structure-identities-to-v2"></a>

## Source ownership and structure v4 compatibility

Persistent identity includes every source in the tracked root module and its
placed descendants: independent voltage/current, controlled E/G/H/F, and SPICE
behavioral voltage/current B sources. Type, output/control bindings, controlling
source reference, values, waveform fields and expression text participate.
Variables resolve through the existing device value resolver; each source field
retains at least the precision its emitter uses. Exact discrete waveform fields
remain exact. Registered testbench objects are outside the DUT traversal, so
external bench stimuli do not enter the DUT fingerprint. A caller explicitly
hashing the whole bench includes its sources: the Monte Carlo campaign hash
combines DUT and bench structure and is distinct from persistent DUT identity.

This does not establish an automatic stored-evidence guarantee for external
stimulus changes. Profiles identify the owner, test and declared parameters; PVT selects evidence
beneath a profile. Profiles do not include an automatic source inventory. If an external bench source
changes without a corresponding declaration change, its stored profile and DUT
identity may remain unchanged. Replay retains its separate managed-source guard.

Root and nested source facts are stored as flattened `devices` rows with
`type='silk_source'`, concrete source type in `model`, and `kind='source'` plus
canonical connections/parameters/waveform facts in JSON `properties`.
`get_circuit_summary()['devices']` includes these rows, and flattened SQL row
counts include sources. Local module `device_count` keeps its existing meaning,
excluding separate sources/passives. Reopening exposes source facts through
these rows; promotion preserves them with new device row IDs. Existing reusable
module JSON also retains its local source facts and a definition hash that includes them. Live compatibility
compares the rebuilt circuit's fingerprint with the stored scheme and digest;
it does not reconstruct the circuit or recompute identity from stored rows.
Before storing a module definition, silk checks that its payload matches
its claimed hash.

The lossy JSON exporter writes tracked sources to a per-circuit `sources` list
(the stored source entry: name, type, nets, parameters, waveform) and
SPICE-imported voltage and current sources to the same list as `name`, `kind`,
`positive`, `negative`, `dc` and `ac`, and passive
devices (resistors, capacitors, inductors) to a `passives` list with `name`,
`kind`, `value`, `pos` and `neg`. The kind, value and terminals are copied as
stored, and transistor width, length and multiplier stay the canonical SPICE
text the tracker stored (for example `"4"`, meaning 4 um on sky130). Import
stores every record. A record it cannot use (a passive missing `name`, `kind`,
`value`, `pos` or `neg`, or a transistor with no width or length) raises
`DatabaseError` naming the record and field (a tracked source without `nets`, or a parsed source without `positive` or `negative`, is refused the same way); nothing is skipped silently. The
projection still is neither a circuit archive nor a database backup.

Opaque/custom emitters, unresolved Variables and unsupported source values
cannot authorize persistent snapshots, module identity or evidence reuse.
Snapshot creation and evidence reuse raise typed errors for these cases. Executable custom-source paths
keep their existing behavior: variant naming may use a source-omitting naming
hint, but identical definitions still require emitted-body comparison; run
sharing declines when source identity is unprovable. Opaque Monte Carlo campaigns
record `structure_hash=None`; separately captured replay source facts and their
existing refusals remain. Source-card emission and public source APIs are unchanged.

`silk.structure/v4` is independent of SQL schema 7. Historical v1/v2/v3 tags,
snapshots and evidence remain untouched and cannot be relabeled or automatically
transferred to v4. Fully source-free digest bytes remain byte-identical, but a
matching digest with an older tag still refuses compatibility. Missing source
facts in an old snapshot do not mean the circuit had no sources. Preserve the
database, rebuild the Python design in a new database or under a new circuit
name, then seal a fresh v4 snapshot and rerun characterization. Source-bearing
`variants=True` names/references intentionally change with the corrected digest;
source-free names and source-card bytes stay unchanged.

<a id="device-unit-canonicalization-introduced-in-structure-v3"></a>

## Device units in stored identities

The current `silk.structure/v4` fingerprint includes device-unit
normalization introduced in v3. It normalizes a copy of extracted device
parameters using the same unit conversion and rendering as SPICE emission.
For example, sky130 `w=1` in `user_units='u'` agrees with
`w=1e-6` in `user_units='si'`, and differs from `w=1` in SI mode. Equality
applies within the same construction path, PDK/model, topology, and other
identity-bearing content; it does not equate different PDKs or models.

Structure fingerprint versions are independent of the database schema,
`silk.spec/v1`, and `silk.profile/v1`.
The current scheme retains child port bindings in flattened and recursive identities.
A physical leaf change or child-port rewire changes every affected ancestor;
the same physical definition remains reusable. Definition reuse still
independently checks emitted subcircuit text before deduplicating.

The normalization contract is:

- Factory devices use their class's parameter scales and caller-supplied
  bookkeeping. Legacy `add_element()` devices use their immediate owning
  module's PDK and parameter quantity/unit metadata, including nested leaves.
- Only caller-supplied numeric values are converted. Assigned `Variable`
  values follow their numeric equivalents. Omitted catalog defaults are
  already PDK-native and are never rescaled.
- An ideal passive's value is stored and hashed as given, except that a
  `Variable` or Variable expression (`2 * r_load`) is replaced by its current
  number, so the identity follows the value and the `devices` row holds
  the number.
- Every extracted factory/legacy device parameter, including dimensionless
  values, is stored as its canonical SPICE **text**. The shared renderer
  removes conversion roundoff using the same precision as emission. Explicit
  strings, including catalog string defaults, remain verbatim: literals and
  expressions are neither parsed nor evaluated for identity equivalence.
- Live parameters, variable objects, supplied/default bookkeeping, PDK state,
  and emitted SPICE bytes do not change. A normalization failure raises an
  actionable `CircuitError` naming the device and parameter instead of
  returning an incomplete fingerprint.

New `devices.properties` and stored module payloads therefore contain
PDK-native parameter strings, not original author-input numbers. For example,
sky130's 1 µm width is `"1"`; GF180MCU's metre-native representation of 2.2 µm
is `"2.2u"`; a multiplier of two is `"2"`; `"width_param"` stays literal.
`CircuitExporter.export_to_json()` forwards these W/L/m strings into its
existing width/length/multiplier fields. That unit change left the projection format unchanged
(`silk.circuit-projection/v2`). The projection is not a circuit reconstruction
or database backup. Historical device properties and exports retain their
stored types and values.

Existing v1/v2 snapshots, module payloads, results, spec measurements, and net
currents keep their original tags, digests, and data. V1 omitted child port
bindings, and both old schemes omitted unit interpretation. That information
cannot be inferred reliably from historical rows. Silk does not rewrite,
relabel, rehash, or automatically transfer old evidence. Matching digest
bytes alone never authorize reuse under the current v4 scheme. V3 snapshots
also retain their original tags and cannot authorize v4 reuse because they
omitted DUT-owned sources.

To continue work:

1. Preserve the original database for historical inspection through SQLite.
   Schema-v1 files may be copied through `migrate_v1_to_v2()`,
   `migrate_v2_to_v3()`, `migrate_v3_to_v4()`, `migrate_v4_to_v5()`, `migrate_v5_to_v6()` and `migrate_v6_to_v7()` above
   to reach the current database format, preserving original identity tags,
   digests, and evidence while changing table shape.
2. Rebuild the design from its Python source and choose a new `db_path`, or a
   new circuit name in an existing schema-v7 database. Same-name, same-stage
   snapshot lookup rejects stored v1/v2/v3 rows with
   `UnsupportedFingerprintSchemeError` and a preserve-and-rebuild diagnostic.
3. Call `create_circuit_version()` to seal a fresh v4 snapshot and rerun
   characterization to establish fresh evidence. Equivalent v4 rebuilds
   reuse that snapshot normally. Do not attach historical evidence or an old
   parent snapshot based on a v1/v2/v3 digest match.

Identity-dependent reads, comparisons, and evidence restores reject v1/v2/v3;
the supported SQL schema still opens for inspection and the stored data stays
untouched. There is no automatic fingerprint or evidence migration.

## Grouping saves: the transaction boundary

Every save that wraps its own writes in `Database.transaction()` is **one
atomic operation**: `store_results()`, `store_result()`, `store_report()`,
`store_net_currents()`, version creation and `import_from_spice()` each group
all of their writes. Tracker compatibility-status writes and promotion also
have their own transaction scope. (`import_from_json()` is
atomic *per circuit*, not across a multi-circuit file.) Call one on its own
and it commits before it returns; wrap several in `db.transaction()` and they
defer to a single commit at the outermost exit. Only the operations listed
here guarantee that all their writes roll back together.

Tracker bindings support **read-your-own-writes**. Inside a group,
`create_circuit_version()`, `ensure_version()` and `promote()` immediately
update the tracker's id, stage and version label to the stored row. A later
tracker write targets that row, so promotion with `lock=True` immediately
refuses subsequent writes. A changed module can create a version and store
results in the same group:

```python
with tracker.db.transaction():
    version_id = tracker.create_circuit_version()
    tracker.store_result("gain", 40.0, "dB")  # targets version_id
```

A successful outer commit keeps the binding. A rollback restores the tuple
from entry to the failing scope, including bindings changed by successful
nested scopes. A failed inner scope restores its parent's binding while
leaving earlier work intact. Failed savepoint release, commit failure and a
poisoned group also restore bindings. SQL rollback failures still propagate;
restoring Python state does not certify that a refused SQL rollback succeeded.
Restores are discarded before after-commit measurement callbacks run, so a
callback failure cannot undo a committed binding. User edits to the circuit
itself are not rolled back.

All tracker evidence targets pass a shared full structure-and-declaration
identity check before writing. `store_pass_fail()` retains its compatibility
`spec_<name>` text and metadata and rejects stale identity. Manual
`add_transistor()` and `add_net_to_db()` also reject stale identity; they still
cannot mutate sealed snapshots. A missing binding raises `DatabaseError`
naming `create_circuit_version()` for these methods, result/report storage,
promotion and report-storage preflight. Current storage preserves its
unbound-testbench workflow: it resolves and validates each child-definition
snapshot; if the testbench tracker is bound, it also rejects a stale parent.

`transaction()` is re-entrant, and every scope — not just the outermost one —
is its own rollback boundary. Nested scopes run on a SQLite `SAVEPOINT`:
releasing one merges it into the enclosing transaction (nothing reaches the
file yet), and an exception that escapes one undoes every SQL change made
under it, including changes from deeper scopes that already exited
successfully.

That is what makes a *caught* failure safe:

```python
with db.transaction():                      # one outer group
    tracker.store_results({"gain": (60.0, "dB")})
    try:
        tracker.store_results({              # raises on the second entry
            "bandwidth": (60.0, "dB"),
            "phase_margin": (object(), "deg"),
        })
    except TypeError:
        pass                                 # caught OUTSIDE that save
    tracker.store_results({"slew": (12.0, "V/us")})
# Committed: gain and slew. The failed save left no rows at all.
```

- **Caught inside the enclosing transaction.** The failed operation's own SQL
  is rolled back, and the pending live-object updates it registered are
  discarded. Unrelated successful work in the enclosing group still commits.
- **Escaping the outermost context.** The whole group rolls back, exactly as
  before.
- **The exception has to reach the context manager.** One swallowed inside the
  very same `with db.transaction():` block, before that block exits, is
  invisible to it — nothing is rolled back for it. Put work you intend to
  recover from in its own `transaction()`, or in a save that already has one.
- **Post-commit actions are scope-owned.** Live `Spec.measurements` updates
  created by storage publish only after the real outermost SQL commit, once,
  in registration order. Rolling back a scope drops the actions registered
  under it — including by an inner save that had already returned — and leaves
  its siblings' actions pending and ordered. An action that raises *after* a
  successful commit propagates its error without claiming a rollback, and the
  actions behind it are abandoned rather than replayed later.
- **When SQLite itself fails mid-scope.** A nested scope whose `RELEASE`
  SQLite refuses never merged into the enclosing transaction, so it is cleaned
  up exactly like a failure inside it — catching that error cannot resurrect
  its rows or its actions. A nested rollback SQLite will not confirm is worse:
  those rows may still be live, so the group is poisoned and the outermost
  context rolls everything back and raises `DatabaseError` ("a nested scope's
  rollback could not be confirmed") with the SQLite error as `__cause__`,
  rather than commit state it could not verify. The poison does not outlive
  that group.
- **A group SQLite has already discarded accepts no more nested saves.** If
  SQLite abandons the enclosing transaction underneath the group — an
  interrupted write, a full disk — the next nested save raises `DatabaseError`
  instead of opening a savepoint. Catching the refusal does not rescue the
  group: it is marked unsafe, so the outermost context still rolls back and
  raises `DatabaseError`.
- **What this does not claim.** Only the saves listed above are covered:
  raw-connection work and caller-issued transaction-control SQL are outside
  the contract, and a simulation measurement already recorded on a live `Spec`
  is never unmade by a database rollback.

## Failure modes

| Exception | Meaning |
|---|---|
| `DatabaseCompatibilityError` | The file is not a valid supported silk database, or its declared shape is malformed. |
| `UnsupportedSchemaVersionError` | The file declares a schema version other than 7 (v1/v2/v3/v4/v5/v6 files name the explicit copy migration path). |
| `UnsupportedFingerprintSchemeError` | Stored identity uses an algorithm this release cannot interpret. |
| `EvidenceNotFoundError` | No compatible metric/spec measurement/profile/current evidence exists for the requested filters. |
| `EvidenceMismatchError` | Relevant evidence exists but its structure, declaration, profile identity, cached pass result, or shared context is incompatible — including a profile/context that exists only on an incompatible structure elsewhere in the version lineage. |
| `EvidenceAmbiguityError` | An evidence selection supplied both mutually-exclusive selectors (`profile_id` and `context`) instead of exactly one. |
| `CharacterizationContextError` | A profile-scoped call supplied a partial context — some but not all of owner path/test name/parameters/corner/temperature/voltage — or a reserved PVT name as a declared parameter key. |
| `DatabaseError` | A tracker write has no bound version (call `create_circuit_version()`), or a database state/transaction precondition failed. |
| `LockedVersionError` | A write targeted a locked promoted version. |
| `ReportStorageError` | `run_test(store=True)` ran the test but `store_report()` failed afterwards; `.report` holds the completed `SpecReport`, `__cause__` the underlying error. Rows are rolled back; the run's live Spec measurements are not undone. |
| `RuntimeError` from current restore | No matching version/current corner exists, a row is stale, or a stored pin no longer exists live. Nothing is partially applied. |
| `ValueError` for `db_path` | A method was pointed at a different file than the module's existing binding. |

These failures are deliberate. Silk does not return empty or plausible-looking
evidence when applicability cannot be proved.

## JSON is a projection, not a backup

`export_to_json()` writes `projection_format: silk.circuit-projection/v2` and
an explicit `projection_scope`. It includes a circuit summary, transistor/net
projection, and metric values and their recorded PVT, source, tool, conditions and timestamp;
characterization profiles and run-provenance links are omitted. Version 2 writes each metric's `corner`, `temperature` and `voltage` as a
value, `{"state": "none"}` / `{"state": "sweep"}` or `null` (unknown), plus
`corner_state` / `temperature_state` / `voltage_state`. A version-1 reader
cannot read a tagged value, so the version changed rather than letting an
older silk misread one.

`import_from_json()` accepts `silk.circuit-projection/v2`, and
`silk.circuit-projection/v1` imported by the schema v4-to-v5 rule: a value is
fixed, `null` is unknown (never none), and a `null` temperature recorded as a
sweep is a sweep. Any other `projection_format` is refused before anything
is written. A v1 result carrying a state key or tagged value is refused too;
its circuit's import rolls back, while circuits earlier in the same file stay
imported, as for any other invalid result. A file with
no `projection_format` is hand-authored input in the documented shape and is
accepted as before. It excludes
circuit-version and evidence-row UUIDs, specs, currents, parasitics, complete
module connectivity, raw runs, waveforms, logs, and calibration.

`CircuitImporter.import_from_json()` can ingest that projection, but recomputes
the snapshot fingerprint from parsed content and creates new version/result
UUIDs. It neither preserves nor impersonates the source database's evidence
identity. If the imported name, stage, and computed identity already match a
sealed snapshot in the destination, import rejects the collision before adding
evidence; use a different database or circuit name. Keep and back up the closed
SQLite file when you need the persistent record; JSON is for lossy interchange
and inspection, not archival round-trip.

## Runnable references

```bash
# Close/reopen/query/restore/promotion flow; sky130 catalog, no simulator. It also
# imports the lossy JSON projection into a second database and lists the
# latest sealed snapshot per design stage:
MPLBACKEND=Agg .venv/bin/python -m silk.examples.circuit_tracking_example

# Two profile-scoped characterizations (metrics + spec measurements +
# currents) at one corner, close/reopen provenance query, current-set
# supersession, EM sizing from restored evidence, a loud mismatch failure,
# and the unscoped path unchanged:
MPLBACKEND=Agg .venv/bin/python -m silk.examples.net_current_persistence_example

# The flagship design loop: two characterization contexts stored (metrics +
# spec measurements + currents), a bare-Database provenance query, a fresh
# rebuild restoring one exact context with profile/current-set/origin IDs,
# restored currents driving EM sizing and parasitic feedback, and typed
# EvidenceMismatchError on an edited or post-layout structure:
MPLBACKEND=Agg .venv/bin/python -m silk.examples.design_loop_walkthrough
```

The executable contract is also covered by `tests/db/test_release_flow.py`
(the unscoped flow, running against the current database schema v7) and
`tests/db/test_v1_migration_public_contract.py` (a real v1 file copied through
v2, v3, v4, v5 and v6 to v7 and read back through the public surface above).

### Aggregate measurements (schema v6)

Schema v6 adds one nullable `spec_measurements.aggregate_json` column.
Ordinary rows and copied historical rows contain SQL NULL. The shared
`silk.measurement.aggregate/v1` codec is also used by CircuitReport JSON v4 and later;
there is no database-specific aggregate dialect. Invalid payloads refuse with
the measurement row ID. The aggregate stores summary evidence rather than raw samples.

Saving an aggregate is explicit. After constructing the circuit and declaring
`sigma_spec` with the same metric as the campaign statistic:

```python
from silk.db import Database

version = dut.create_circuit_version()
measurement = mc.measurement('offset_v', corner='tt')
sigma_spec.record_measurement(measurement)  # live evidence only
stored = next(s for s in dut.get_stored_specs()
              if s.owner_path == '$' and s.spec_key == sigma_spec.key)
with Database(db_path) as db:
    db.insert_spec_measurement(
        'offset-sigma-observation', stored.spec_id,
        measurement.source, measurement.value,
        corner=measurement.corner, temperature=measurement.temperature,
        voltage=measurement.voltage, timestamp=measurement.timestamp,
        sim_run_id=measurement.sim_run_id, aggregate=measurement.aggregate)
dut.close()

with Database(db_path) as db:
    evidence = db.get_stored_specs(version)  # typed records retain aggregate
# Rebuild the same circuit/spec declarations with the same db_path, then:
assert rebuilt.create_circuit_version() == version
rebuilt.restore_spec_measurements()
```

Use a unique measurement ID per insertion. Pass `profile_id` and existing
`provenance_id` explicitly when applicable. The writer checks the aggregate
metric against the selected stored declaration and requires `campaign_id` to
match `sim_run_id`. Live typed readback repeats this check and names the
measurement ID on refusal, even when the connection was opened before a row
was changed. Existing snapshot identity checks and transactions remain.
`store_report()` takes an ordinary `SpecReport` and does not borrow aggregate
provenance from a live measurement with the same value. Both measurement
restoration and characterization restoration retain explicitly saved aggregate
evidence. `CircuitExporter.export_to_json()` is a lossy projection that omits
specs; it is not the aggregate save path.

Upgrade an existing v5 database with an explicit copy:

```python
from silk.db import migrate_v5_to_v6, migrate_v6_to_v7
migrate_v5_to_v6('design-v5.db', 'design-v6.db')
migrate_v6_to_v7('design-v6.db', 'design-v7.db')
```

IDs, profiles, seals, conditions, provenance and history are preserved. Earlier migrations
retain their named destination versions: follow the chain through v5, then
copy v5 to v6 and v6 to v7. Normal opens never migrate an existing file in place.


### Supply-sweep evidence (schema v7)

The current schema is v7. A designated-supply DC sweep stores
`voltage_state='sweep'` and `voltage=NULL` on evidence rows. Its configured
start/stop/step in volts lives in the existing operating-conditions/report
JSON as `supply_sweep`, not SQL range columns. Summaries say supply sweep.
Sweeps remain unscoped: characterization profiles require every fact fixed
or none, so a sweep cannot create or match a fixed-voltage profile.

Upgrade a v6 database explicitly into a new destination:

```python
from silk.db import Database, migrate_v6_to_v7
migrate_v6_to_v7('designs_v6.db', 'designs_v7.db')
with Database('designs_v7.db') as db:
    pass
```

Earlier migration chains must finish with this v6-to-v7 step before opening
with `Database`.
Historical rows and provenance are preserved without reinterpretation; old
exports without a supply-sweep marker do not acquire one during import.
