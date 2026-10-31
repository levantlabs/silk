# silk settings — where silk finds its tools and PDKs

Configure PDK and tool locations, example output and operating-region policy
through environment variables or an optional settings file.

## The settings

| Setting | What it is | Built-in default |
|---|---|---|
| `SILK_REGION_THRESHOLDS` | Your per-PDK operating-region YAML for `op_table()` and HTML reports; see [operating-point.md](operating-point.md#project-and-customer-threshold-files) | not set (shipped per-PDK policy) |
| `PDK_ROOT` | The directory that holds your installed PDKs (`sky130A/`, `gf180mcuD/`, `ihp_sg13g2/` — see [installing-the-pdk.md](installing-the-pdk.md)) | `~/pdk` |
| `SILK_NGSPICE_BIN` | The ngspice binary | `ngspice` on `PATH` |
| `SILK_XYCE_BIN` | The Xyce binary (optional second simulator) | `Xyce` on `PATH` |
| `SILK_EXAMPLES_OUT` | Output root for examples using `example_output_dir()`; individual examples may provide other destinations | `./silk-output` under the current directory |
| `SILK_FIREFOX` | A Firefox that can run headless, for the optional browser smoke test | not set (browser smoke tests can use `SILK_CHROME` instead) |
| `SILK_CHROME` | A Chrome/Chromium that can run headless, for the optional browser tests | not set (the optional Playwright end-to-end test looks on `PATH`) |

When `SILK_REGION_THRESHOLDS` is set, doctor parses the selected file and
reports `FAIL` with the file/key diagnostic and a fix hint if it is invalid.
Threshold paths expand a leading `~`; relative paths use the working directory.

## Where the file lives

silk selects **one** settings-file path; it does not search the other paths
if the selected file is absent:

1. the path in `$SILK_SETTINGS`, if that variable is set;
2. `$XDG_CONFIG_HOME/silk/silk.env`, if `XDG_CONFIG_HOME` is nonempty;
3. `~/.config/silk/silk.env`.

With neither variable set, silk uses the third path. `python -m silk.doctor`
prints the path silk is using (see [below](#check-it-with-the-doctor)).

## Create a settings file

If you already have a settings file, edit it instead of replacing it. For a
new installation using the default path:

```bash
mkdir -p ~/.config/silk
cat > ~/.config/silk/silk.env <<'EOF'
# silk settings — one KEY=value per line; see docs/settings.md
PDK_ROOT=~/pdk

# Only if your simulators are not on PATH:
# SILK_NGSPICE_BIN=/opt/ngspice/bin/ngspice
# SILK_XYCE_BIN=/opt/xyce/bin/Xyce

# Optional:
# SILK_EXAMPLES_OUT=~/silk-runs
EOF
python -m silk.doctor
```

The doctor may report missing tools or PDKs until you install them and set
their locations. Creating this file does not install them.

## File format

```bash
# Comments start with '#'. Blank lines are ignored.
PDK_ROOT=~/pdk                         # a leading ~ is expanded
export SILK_XYCE_BIN=/opt/xyce/bin/Xyce   # optional 'export ' prefix
SILK_EXAMPLES_OUT="/data/silk runs"    # quote a value that contains spaces or '#'
```

- One `KEY=value` per line, using exactly the names in the table above.
- An unquoted value ends at whitespace followed by `#` (a trailing comment).
  A quoted value (`"…"` or `'…'`) is taken literally, apart from a leading `~`.
- An empty value (`PDK_ROOT=`) counts as not set.
- Nothing else is interpreted: no `$VARIABLE` expansion, commands or backslash
  escapes. silk reads the file directly; you do not need to source it.
- Leading `~` expansion applies to file values. Environment values are returned
  as supplied; use an absolute path when setting them outside a shell.
- Relative file/directory paths use the process working directory, not the
  directory containing `silk.env`; prefer absolute paths for settings shared
  across projects. Bare executable names are searched on `PATH`.

silk refuses a file it cannot read cleanly rather than guessing. A line that is
not `KEY=value`, an unterminated quote, or the same key given twice raises
`silk.exceptions.SettingsError`, naming the file and line number:

```
SettingsError: /home/you/.config/silk/silk.env:4: expected KEY=value, got 'PDK_ROOT /pdk'
```

A key silk does not know (a misspelling such as `PDKROOT`) does not stop silk,
but the doctor reports it as a `[warn]` with its line number, so it is never
silently ignored.

## Precedence

For each setting, the first of these wins:

1. **the environment variable**, if set and not empty;
2. **the settings file**;
3. **the built-in default**.

So the file holds your everyday locations, and an environment variable
overrides one of them for a single command, for example to run against a
second ngspice build:

```bash
SILK_NGSPICE_BIN=/opt/ngspice-47/bin/ngspice python -m silk.doctor
```

Keep a per-command choice like that out of the file: whatever the file says
applies to **every** run.

## Turning the file off (CI and test runs)

Set `SILK_SETTINGS` to an empty value, or to `/dev/null`, and silk reads no
settings file at all: environment variables and built-in defaults only.

```bash
SILK_SETTINGS= python -m pytest -q
```

Do this in CI and any run that must be reproducible from its own environment,
so a machine's personal file cannot change the result. silk's own CI workflow
does. Setting `SILK_SETTINGS` to a path that does not exist is an error, not a
silent fallback.

silk's test suite reads your settings once at start-up, exports what the file
sets into the environment, then turns the file off for the rest of the run.
Tests therefore see the same tools silk does, and a test that unsets a variable
gets the built-in default rather than your file.

## Check it with the doctor

```bash
python -m silk.doctor
```

Near the start, the report shows the file and every setting with its value
and where the value came from — `env`, `file` or `default`. For example (paths and
values depend on your machine):

```
[ ok ] settings file                   /home/you/.config/silk/silk.env (found)
[skip] setting SILK_REGION_THRESHOLDS  not set (per-PDK operating-region thresholds YAML; optional)
[ ok ] setting PDK_ROOT                /home/you/pdk  (source: file)
[ ok ] setting SILK_NGSPICE_BIN        ngspice  (source: default)
[ ok ] setting SILK_XYCE_BIN           /opt/xyce/bin/Xyce  (source: file)
[ ok ] setting SILK_EXAMPLES_OUT       /home/you/work/silk-output  (source: default)
[skip] setting SILK_FIREFOX            not set (Firefox for the optional browser smoke test; optional)
[skip] setting SILK_CHROME             not set (Chrome/Chromium for the optional browser tests; optional)
```

The file line reads `(not found)` when there is no file, and `disabled by
SILK_SETTINGS` when it is turned off. When a check fails because a location is
wrong or missing, its `fix:` line names the settings file path to edit.
`python -m silk.doctor --json` carries the same rows. The full doctor contract
is in [cli.md](cli.md#silkdoctor--environment-check).

## From Python

```python
from silk import settings

settings.get('PDK_ROOT')         # the resolved value
settings.lookup('PDK_ROOT')      # Setting(name='PDK_ROOT', value='/home/you/pdk', source='file')
settings.settings_path()         # the file silk reads, or None when disabled
```

The simulator runners, `load_pdk`, the doctor, provenance and the test-suite
probes use `silk.settings` to resolve these locations. Provenance records
the simulator setting name, `PATH` or `in-process`; it does not distinguish
environment overrides from file values.
