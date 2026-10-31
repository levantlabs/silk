# Validating a PDK device-details YAML

Use `silk.pdk.validate` to compare catalog names, terminal counts and emission
prefixes with installed models. `--simulate` also builds and runs a small
circuit per device; neither pass verifies performance or pin order.

Only sky130 is release-validated for transistor simulation in 0.2.0.
GF180MCU and SG13G2 are registered preview/catalog integrations; a local
validation result does not change that support boundary.

## Run the validator

From the repository root, with the virtual environment installed:

```bash
# Static checks: installed PDK models required, no simulator required.
.venv/bin/python -m silk.pdk.validate

# Also build and simulate each device: requires ngspice.
.venv/bin/python -m silk.pdk.validate --simulate
```

Both commands default to the registered `sky130` PDK. Model and simulator
locations use silk's [settings](settings.md): environment variables override
the settings file, which overrides built-in defaults. Run
`.venv/bin/python -m silk.doctor` to inspect the resolved locations.

Useful options:

- `--pdk NAME` selects a registered PDK. Its manifest supplies the catalog,
  library, corners and extra includes. See `silk.pdk.available()` for names.
- `--yaml PATH` loads a device-details YAML through `BasePDK` instead of the
  registry. Without `--lib`, this mode uses
  `<PDK_ROOT>/sky130A/libs.tech/ngspice/sky130.lib.spice`, even for a catalog
  intended for another PDK. It does not load manifest corner or extra-include
  configuration. Prefer `--pdk` once a manifest exists.
- `--lib PATH` overrides the library file. For an ad-hoc catalog, supply this
  along with `--yaml` to select the intended model library. It must name a
  readable file, rather than a directory.

Exit codes:

- `0`: no issues found in the checked scope.
- `1`: validation issues, including invalid catalog names, catalog loading
  errors reported as `PDKError`, or YAML parsing errors.
- `2`: the PDK name, catalog file or library cannot be resolved.

For the default command, successful output starts with `OK: sky130:`. Other
runs print `OK: <label>:`, where the label is the PDK name or YAML path. The
line includes the device count and `no issues (scope: ...)`. For device
validation issues, failure output names each device, issue kind and detail.
Counts and paths depend on the catalog and model installation.

## What the static pass checks

The validator follows `.include`, `.inc` and file-based `.lib` references
from the library file. It also starts from the PDK's extra includes and files
listed in **all** its corners. This set of reachable files is called the
include closure. It collects definitions across sections, rather than proving
that every definition is available in a particular selected corner. The root
library must be readable, but missing or unreadable included files are skipped;
a clean static verdict does not prove the include tree is complete.

Each catalog device's `PDKDevice` name, `pinList` and `spice_prefix` are checked
against the collected `.subckt` and `.model` definitions:

| Issue kind | Meaning |
| --- | --- |
| `missing_definition` | The `PDKDevice` name has no definition in the checked files. |
| `arity_mismatch` | The `pinList` count matches none of the terminal counts found for that subcircuit. |
| `wrong_emission` | A subcircuit uses a prefix that does not start with `X`, or a bare model uses a prefix that starts with `X`. |

The emission check distinguishes subcircuit calls from primitive lines; it
does not verify that every primitive uses the correct device letter. The
static pass also does not check parameter values against model bins.

### Junction ratings and their sources

The catalog-only `check_junction_provenance(pdk)` function needs no model
library or simulator. It checks `junction_ratings` and
`junction_ratings_provenance`:

- `unsourced_rating`: a rating lacks both a source citation and a heuristic
  derivation rule.
- `bad_provenance`: the provenance block is malformed or a derivation rule
  disagrees with the rating. A rule must begin with `= <key>_max` or
  `<factor> x nominal <V> V`. Equality compares both ends of the voltage range;
  the factor form derives a symmetric magnitude.
- `rating_convention`: a magnitude is not positive and finite, or a signed
  range `{min: lo, max: hi}` has non-finite ends or `lo >= hi`.

A source citation covers ratings without individual heuristic rules. An
unrated device needs no provenance; a provenance block on an unrated device
is an issue. See [Junction ratings](adding-a-pdk.md#junction-ratings-junction_ratings--junction_ratings_provenance) for the
rating convention. The full validator includes this check, but still requires
a model-library scope to complete its static pass.

### Read the scope with the verdict

Completed validation verdicts report the scope used. Loading failures return
earlier without a scope:

- `closure of <lib file>, N files`: definitions collected from the library
  and its reachable files, including the additional starting files above.
- `unscoped walk of $PDK_ROOT (<root>) — no lib file`: a recursive search
  used when no library is declared, including the `/dev/null` placeholder.
  This needs an explicitly configured `PDK_ROOT` pointing at an existing
  directory. Definitions from unrelated PDKs under that root may hide errors.

A missing, unreadable or directory-valued library path causes exit `2`;
it does not trigger the recursive fallback. This restriction applies to
validation. `load_pdk('sky130', library_location='/fake/lib')` can still be
used for netlist generation without reading installed models.

In the Python API, an explicit `lib_root=` overrides the closure and searches
that tree recursively. Use it deliberately: its definitions may differ from
those visible to a netlist using the PDK's library configuration.

## What the simulation pass checks

With `--simulate`, the validator builds one temporary circuit per device.
Every pin gets a separate node and an independent DC `VoltageSource` to ground
(`0`), with voltages `0.1`, `0.2`, … V in pin order. Each parameter uses its
`minval`. The circuit runs a transient analysis with a 1 ns step and 2 ns stop
at the PDK's default corner (`tt` if unspecified).

Build or simulation exceptions produce `sim_failed`; the simulation runner’s
normal failure checks apply. After a completed run, the validator reads that
run's `output.log` using its recorded run-file path and reports `sim_failed`
for `unknown subckt`, `singular matrix`, or error/fatal lines, even when the
simulator returned data. A missing or unreadable run log also produces
`sim_failed`. Temporary circuit directories are removed after each device.
Save terminal output if you need a record; the CLI does not write a report file
or retain temporary decks and logs.

This is a basic build-and-run check at one bias and corner. It does not
characterize performance, validate every parameter combination or test every
corner. For binned devices, ensure the chosen `minval` values form a supported
model combination. Run the simulation pass before shipping catalog changes;
it costs one ngspice run per device.

## Python API

This example runs the static pass with the configured sky130 models:

```python
from silk import load_pdk
from silk.pdk.validate import validate_pdk

pdk = load_pdk('sky130')
issues = validate_pdk(pdk)
for issue in issues:
    print(issue.device, issue.kind, issue.detail)
assert issues == []
```

`validate_pdk(pdk, *, lib_root=None, simulate=False)` returns a list of
`DeviceIssue(device, kind, detail)` entries. Set `simulate=True` to add the
simulation pass. `kind` is one of the seven issue kinds described above.
An unusable declared library or unavailable fallback raises
`silk.pdk.validate.PDKValidationError`. Validate an explicit `lib_root` yourself:
an empty or unreadable tree contributes no definitions.

The CLI checks catalog identifiers before loading the PDK, reporting all bad
device, pin, alias and parameter names, including `paramConstraints` names and
`source_class` references. Python loaders (`BasePDK` and `load_pdk`) instead
raise `PDKError` on the first invalid name.

## Regression checks and authoring checklist

`tests/test_pdk_validate.py` includes the installed-sky130 static guard
`test_real_pdk_has_no_validation_issues` and the slow simulation guard
`test_all_devices_simulate`. A `PDK_ROOT` from the environment or settings file satisfies these guards;
the built-in default alone does not. The simulation guard also checks for
the configured ngspice executable.
The simulation guard compares the exact set of failing devices against
`KNOWN_SIM_UNSUPPORTED`, currently empty. A new failure or unexpected recovery
fails that assertion.

When adding devices or a new PDK:

- Set `PDKDevice` to the model's exact name, `pinList` to its actual terminal
  order, and parameter bounds to supported values.
- Add rating sources or derivations where ratings are declared.
- Run the static pass using `--pdk NAME`, or `--yaml PATH --lib LIBRARY_FILE`
  before registering a manifest.
- Run the same command with `--simulate` and investigate every issue.
- Add equivalent static and simulation regression guards for a new PDK.
  If a simulation limitation cannot be resolved, document it and explicitly
  assert the expected failing-device set instead of ignoring failures.
