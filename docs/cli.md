# silk command-line tools

silk installs no `silk` console command. Run its tools as Python modules,
using the environment where you installed silk. From a repository checkout,
use `.venv/bin/python` from the repository root; the commands below assume
that layout.

Start with the environment check:

```bash
.venv/bin/python -m silk.doctor
```

Tool and model locations come from [silk settings](settings.md): environment
variables override the settings file, which overrides defaults. Catalog
exports and waveform viewing need neither a simulator nor installed PDK
models. PDK validation needs the model library; its optional simulation check
also needs ngspice.

## Tools

| Module (run with `.venv/bin/python -m`) | Purpose | Detailed guide |
|---|---|---|
| `silk.doctor` | Check dependencies, settings, simulators and model installations | [Environment check](#silkdoctor--environment-check) |
| `silk.pdk.validate` | Check a PDK catalog against its model library | [PDK validation](pdk-validation.md) |
| `silk.circuit.rolecheck` | Check a port-role ruleset and classify sample names | [Port-role rules](authoring-port-role-rules.md) |
| `silk.elements.schema` | Export an ideal-element or PDK catalog as JSON | [Schema export](schema-export.md) |
| `silk.sources.schema` | Export source and waveform schemas as JSON | [Schema export](schema-export.md) |
| `silk.signals.schema` | Export scalar metric schemas as JSON | [Schema export](schema-export.md) |
| `silk.codegen.generator` | Generate Python from a circuit-spec JSON file | [Code generation](codegen-contract.md) |
| `silk.signals.view` | Emit an interactive HTML waveform explorer from a nutmeg rawfile | [Waveform explorer](waveform-explorer.md) |

### `silk.doctor` — environment check

The doctor reports the Python version (requires 3.10 or later), settings file
and resolved values with their sources, imports of `numpy`, `matplotlib`,
`yaml` and `pydantic`, operating-region threshold policy, ngspice and its
version, `PDK_ROOT`, and every registered PDK's model installation. Optional
checks cover Xyce and the matplotlib backend.

Settings include `PDK_ROOT`, `SILK_NGSPICE_BIN`, `SILK_XYCE_BIN`,
`SILK_EXAMPLES_OUT`, `SILK_FIREFOX`, `SILK_CHROME`, and
`SILK_REGION_THRESHOLDS`. Unknown settings-file keys produce warnings;
an unreadable or invalid settings file fails, as do checks that cannot
resolve the settings they need. A selected region-threshold file is validated
before simulation; an invalid file fails the check.

The [`PDK_ROOT`](settings.md#the-settings) row passes when its directory exists,
including the built-in `~/pdk` default, and fails with a fix when it does not.

Each row has a status:

- `[ ok ]`: check passed.
- `[FAIL]`: required check failed; a `fix:` line explains what to do.
- `[warn]`: advisory, with a fix hint where available. Missing registered PDKs,
  missing statistical sections, and missing or incompatible OSDI plugins are
  warnings. An installed catalog does not imply working simulation models.
- `[skip]`: an optional component or policy is not selected or available.

Only sky130 is release-validated for transistor simulation;
GF180MCU and SG13G2 are preview integrations. A PDK's `[ ok ]` reports
installation completeness, not release support.

Exit `0` means all required checks passed; exit `1` means at least one failed.
Warnings and skipped optional checks do not affect the exit code.

```bash
.venv/bin/python -m silk.doctor --json
```

`--json` writes a JSON list to stdout, with `{name, status, detail, fix}` per
check. Status values are `ok`, `FAIL`, `warn`, or `skip`. Paths, dependency
versions, row counts and the final summary depend on your environment.

`python -m silk.doctor` imports silk, which imports numpy. Doctor also needs
numpy in direct-script mode; install it in the selected environment before
running the doctor.

### `silk.pdk.validate` — PDK vetting

Static validation checks catalog names, pin counts and emission against the
library's `.subckt`/`.model` definitions, along with catalog consistency checks.
It needs installed models but no simulator:

```bash
.venv/bin/python -m silk.pdk.validate --pdk sky130
```

The verdict includes the library-search scope used for the check.

| Option | Meaning |
|---|---|
| `--pdk NAME` | Registered PDK; default `sky130` |
| `--yaml PATH` | Ad-hoc device-details YAML, bypassing the registry |
| `--lib PATH` | Override the model library location |
| `--simulate` | Also run a per-device ngspice smoke test |

Exit `0` means no issues, `1` means catalog/validation issues (including
invalid names or YAML), and `2` means the catalog or library could not be
resolved or used. Invalid device, pin, parameter and alias names are listed
before library validation. See [PDK validation](pdk-validation.md) for the
check catalog and library-search rules.

### `silk.circuit.rolecheck` — port-role ruleset checker

Use the shipped ruleset by default, or provide `--file PATH`. `--names NAME…`
classifies sample port names without building a circuit:

```bash
.venv/bin/python -m silk.circuit.rolecheck --names vdd clk vout
```

```text
OK: default (port-roles.default.yaml): 7 rule(s)
  vdd              -> power
  clk              -> clock
  vout             -> (unmatched)
```

Exit `0` means valid, `1` means malformed ruleset content, and `2` means the
file was missing or arguments were invalid. Other read errors, such as a
directory path or a permission failure, escape as a traceback with exit `1`.
Use the `rolecheck` wrapper rather than
running `silk.circuit.port_roles` directly.

### `silk.elements.schema` — device schema export

Exactly one of `--elements` or `--pdk NAME_OR_YAML_PATH` is required.
Registered names and aliases load the packaged catalog without requiring
installed models. A YAML path loads an ad-hoc device catalog.

```bash
.venv/bin/python -m silk.elements.schema --elements
.venv/bin/python -m silk.elements.schema --pdk sky130
```

JSON goes to stdout, or to `-o FILE` / `--output FILE`; `--indent N` sets
indentation (default `2`). Metadata includes `_meta` and, when the catalog
has corners, `_library`. Diagnostics go to stderr, keeping stdout parseable.

### `silk.sources.schema` — source library schema export

```bash
.venv/bin/python -m silk.sources.schema
```

Writes JSON for `VoltageSource`, `CurrentSource`, four controlled-source
types, behavioral voltage/current sources, their waveforms, and `_meta`. It has no argument parser: arguments, including `--help`, are ignored.

### `silk.signals.schema` — metric schema export

```bash
.venv/bin/python -m silk.signals.schema
```

Writes the scalar metric catalog from `silk.metrics`, plus `_meta` with the
format version and signal-kind table. Each metric describes its domain, unit,
signal inputs and parameters. Metrics returning result objects are omitted.
It has no argument parser: arguments, including `--help`, are ignored.

### `silk.codegen.generator` — circuit-spec → Python codegen

Provide a circuit-spec JSON file as the required positional argument.
Python goes to stdout by default; `-o FILE` / `--output FILE` writes it to a
file. Missing the positional argument produces argparse usage and exit `2`.
The [code-generation contract](codegen-contract.md) describes complete inputs
and generated code; generation itself needs no simulator or installed models.

For example, create a zero-pin subcircuit spec and generate its builder:

```bash
printf '%s\n' '{"circuit_name":"empty_block","ports":{},"devices":[]}' > /tmp/silk-cli-spec.json
.venv/bin/python -m silk.codegen.generator /tmp/silk-cli-spec.json -o /tmp/silk-cli-generated.py
```

### `silk.signals.view` — waveform explorer

The viewer emits an HTML page with a searchable signal picker, zoom, cursors
and viewport measurements. It also accepts dropped rawfiles or **Open .raw…**
in the browser. See [waveform explorer](waveform-explorer.md) for browser
controls and [browser waveform core](browser-waveform-core.md) for the
standalone JavaScript API.

This example uses a fixture shipped in the repository and needs no simulator:

```bash
.venv/bin/python -m silk.signals.view tests/fixtures/rawfiles/a_ac_bin.raw -o /tmp/cli_doc_demo/view.html
```

```text
wrote view.html (1 plots, 6 signals, 5 pts) -> /tmp/cli_doc_demo/view.html
```

Open the resulting page in a browser, or add `--open`. By default it has
`view.data.js` and `view.stamp.js` sidecars; keep these beside the HTML.
`--inline` instead writes a single file with no sidecars or auto-reload.
The six signals here are magnitude/phase pairs for three complex AC vectors.

| Argument | Meaning and default |
|---|---|
| `rawfile` | Required nutmeg `.raw` input |
| `-o`, `--output OUT` | HTML path; default `view.html` beside the rawfile. With `--split-by-block`, only its parent directory is used; block filenames are chosen automatically. |
| `--title TEXT` | Page title |
| `--inline` | Single file; incompatible with `--serve` |
| `--serve` | Serve on `http://127.0.0.1`, re-emit when the rawfile changes; blocks until Ctrl-C |
| `--port N` | Server port; default `0` chooses an available port and prints it |
| `--open` | Open the emitted page in the default browser |
| `--poll-ms N` | Sidecar polling interval; default `2000`, `0` disables polling |
| `--view-state PATH` | Embed an existing saved view; incompatible with `--no-view-state` |
| `--no-view-state` | Disable automatic loading of `<HTML stem>.silkview.json` beside the output |
| `--commands JSON` | Embed an array of commands to replay after rendering |
| `--context-title TEXT` | Run title; takes precedence over `--title` |
| `--context-subtitle TEXT` | Run subtitle |
| `--context-field LABEL=VALUE` | Repeatable information field; splits at the first `=` |
| `--signals PAT` | Repeatable inclusion selector; matches any supplied selector |
| `--exclude PAT` | Repeatable exclusion selector; exclusion wins |
| `--max-depth N` | Maximum hierarchy depth; `0` keeps top-level names |
| `--list-blocks [DEPTH]` | Print block counts without emitting pages; default depth `1` |
| `--split-by-block [DEPTH]` | Emit one page per block, plus `top.html` when top-level signals exist; default depth `1` |

Selectors match raw vector names before AC magnitude/phase expansion. They
accept exact names, globs (`*`, `?`), or contiguous dot-separated hierarchy
segments anywhere in the parsed path, all case-insensitive. Instance indices
accept `[k]` or `<k>`; node leaves remain literal. Brackets are not glob character
classes.
See [selector rules](simulator-save-list.md#selectors).

```bash
.venv/bin/python -m silk.signals.view tests/fixtures/rawfiles/a_ac_bin.raw -o /tmp/cli_doc_demo/scoped.html --signals 'v(out)'
```

```text
wrote scoped.html (1 plots, 2 of 6 signals, 5 pts) -> /tmp/cli_doc_demo/scoped.html
```

A selector matching nothing, or an empty final selection, fails with a message.
Overlapping selectors are allowed. Served pages retain their selection on
re-emission. `--list-blocks` cannot be combined with output or other
emit-affecting flags. `--split-by-block` cannot be combined with `--serve`,
`--signals`, `--exclude`, or `--max-depth`: each page selects its own block.
Block counts can overlap because device internals may belong to both a
device-class root and an instance hierarchy. A non-inline split with multiple
pages provides a Scope control for switching between them. `--open` opens `top.html`,
or the first page if there are no top-level signals.

For example, embed initial commands and run details:

```bash
.venv/bin/python -m silk.signals.view tests/fixtures/rawfiles/a_ac_bin.raw -o /tmp/cli_doc_demo/context.html --context-title 'AC response' --context-field corner=tt --commands '[{"cmd":"addPanel","args":{"signals":["v(out) (mag)"]}}]'
```

Python validates command structure; unknown command names fail in the browser.
Saved view files embed layout, cursors and markers; automatic loading looks
beside the output HTML, not the input rawfile. A loaded view adds
`[+view-state]` to the success line.

Exit `0` means success, `1` means input, output, selection or payload failure
(with a diagnostic on stderr), and `2` means argument usage errors, including
incompatible flags, malformed `LABEL=VALUE`, or missing `rawfile`.

From Python, use `Module.view(open=False)` to emit without opening a browser,
`silk.view(source)` for a rawfile or simulated module, or
`report.add_waveform_view(source)` to emit and link a page from a report.
`Module.view()` opens the browser by default. Emission is explicit; a
simulation alone does not write a waveform explorer.

## Other module invocations

Some schema and code-generation commands emit a cosmetic `runpy`
`RuntimeWarning` on stderr because their parent package has already imported
them. Their stdout remains usable and the warning does not change exit codes
under default warning handling.

There is no `silk.db.cli`. Use the [database Python API](database.md).
`silk.db.circuit_db` has no runnable demonstration; `silk.db.structure_tracker`
prints a static usage snippet. Neither is a database-management CLI.

## A note on `silk.examples.*`

Examples are runnable demonstrations rather than general command-line tools.
See the [example index](agent-guide.md#7-curated-example-index) and each example's
module docstring for inputs and options. Run examples with `MPLBACKEND=Agg`
when no graphical display is available.

Examples using `silk.examples.example_output_dir()` write under `silk-output/`
in the current working directory by default. The `SILK_EXAMPLES_OUT` setting
(environment or settings file) redirects this root. Some examples use temporary
scratch directories or offer their own output-directory arguments; check the
individual example. The helper raises `ExampleOutputError` naming the path
and setting when its destination cannot be created or is not writable.

## ngspice release support

silk supports ngspice 42 and later. The doctor resolves the same binary as the
runner: `SILK_NGSPICE_BIN`, then `ngspice` on `PATH`. Set the environment
variable for a single command to select another installation. Xyce similarly
uses `SILK_XYCE_BIN`, then `Xyce` on `PATH`.

The doctor warns about releases below the minimum and reports update advice
for supported releases with known bugs fixed later. Simulation launch emits
the same advice as `NgspiceVersionWarning`, once per process per canonical
binary path. This advisory does not stop simulation, including when warnings
are treated as errors. Restart the process after replacing a binary in place.
See [known issues](known_issues.md) for simulator limitations, including slow
sky130 loading on ngspice 45 and 46, and [installing the PDK](installing-the-pdk.md)
for preview PDK requirements.
