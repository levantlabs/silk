# AGENTS.md — contract for AI agents working in this repo

silk (`import silk`, PyPI name `circuitsilk`) is for custom circuit
design in Python, from specs to layout: describe a circuit, emit SPICE, simulate,
measure, and check the result against specs.

**If you are building a circuit example, read [docs/agent-guide.md](docs/agent-guide.md)
first.** That is the designed front door: the build → characterize → report →
netlist workflow, the composition patterns, the gotchas, and a curated index of
every worked example. This file is the *repo contract* — environment, test
commands, and the conventions you can break without noticing. The two do not
overlap; you need both.

---

## Environment contract

**Run Python as `.venv/bin/python` from the repo root.** A bare `python` may
resolve to a different environment that lacks silk's dependencies (matplotlib in
particular). Every command in this file assumes `.venv/bin/python`.

```bash
cd /path/to/silk
.venv/bin/python -m pytest -q
```

**For sky130 transistor simulation, `PDK_ROOT`** must point at a directory containing
`sky130A/libs.tech/ngspice/sky130.lib.spice` (the standard open_pdks layout).
`load_pdk('sky130')` resolves the library path through it, falling back to
`~/pdk` when unset. Loading the packaged catalog and generating a netlist do
not require the model library to exist. A simulation that needs missing PDK
models raises `SimulationError` with the library path and installation advice;
set `PDK_ROOT` or pass `library_location=` to `load_pdk()` after installing them.

**ngspice 42 or newer** is supported and is the default SPICE simulator, required
by the core slow profile and most SPICE examples. Versions 45, 45.2 and 46 load
sky130 slowly and emit `NgspiceVersionWarning`; see
[slow sky130 loading](docs/known_issues.md#slow-sky130-library-loading-on-ngspice-45-and-46).
**Xyce** is an optional preview backend; select it
per-module with `mod.simulator = 'xyce'` (needs the Xyce CLI on `PATH` or
the `SILK_XYCE_BIN` setting).

**Tool and PDK locations are silk settings, not facts in this file.**
`PDK_ROOT`, `SILK_NGSPICE_BIN`, `SILK_XYCE_BIN`, `SILK_EXAMPLES_OUT`,
`SILK_FIREFOX` and `SILK_CHROME` resolve as environment variable > the user's
settings file (`~/.config/silk/silk.env` by default) > built-in default; see
[docs/settings.md](docs/settings.md). On the owner's machine the settings file
holds the real locations (Xyce is installed but not on `PATH`, for example).
**Run `.venv/bin/python -m silk.doctor` to see them:** it prints the settings
file path and every setting with its value and source (`env`, `file` or
`default`). Do not record "Xyce unavailable" or skip a capability profile
because a tool is not on `PATH`; ask the doctor where silk finds it. Never
create or edit the owner's settings file yourself.

The optional headless-browser smoke needs a working unpacked browser, selected
with `SILK_FIREFOX` or `SILK_CHROME`; the required Node differential gate does
not. A distro Firefox snap or system Chrome may fail headless; prefer an
unpacked browser and check that the executable exists before naming it.

**Ideal-device SPICE needs a simulator, but no installed PDK models.**
A circuit built entirely from ideal elements, sources and passives (including
ideal-only subcircuits) omits the PDK library includes. A circuit containing
PDK devices needs both the simulator and the models. Custom/opaque netlist
content conservatively retains the PDK includes rather than guessing that it
is model-free. See [the PDK-free RC walkthrough](docs/quickstart.md#the-middle-rung-real-spice-with-no-pdk).

The genuinely dependency-free path is **behavioral simulation** — no PDK, no
simulator. Use it to sanity-check an install:
`.venv/bin/python -m silk.examples.behavioral_example`, or
[quickstart §1](docs/quickstart.md#1-60-second-start-no-pdk-no-ngspice).

Install sky130 with [pdksetup](https://github.com/levantlabs/pdksetup)
(`pdk install sky130`). PDK setup is documented in [docs/installing-the-pdk.md](docs/installing-the-pdk.md).

**The 0.2.0 PDK support boundary is explicit:** sky130 is the only
release-validated transistor-simulation PDK. GF180MCU and SG13G2 remain
registered preview/catalog integrations. `silk.pdk.available()` means the
catalog ships, not that its real-model simulation is release-gated.

**Check all of the above with one command** before you debug anything
environmental:

```bash
.venv/bin/python -m silk.doctor        # --json for machine-readable output
```

It shows the silk settings (file path, each value and its source), then checks
the interpreter version, each runtime dependency, ngspice, `PDK_ROOT`,
every registered PDK's library file (one line per manifest, nothing hardcoded),
and the matplotlib backend, printing a `fix` hint per failure or warning.
**Exit 0 = ready, 1 = at least one required check failed.** A registered PDK
that is absent or missing compiled OSDI models reports `[warn]` and does not
gate readiness; `[ ok ]` proves installation completeness, not release support.
Optional tools (Xyce) report `[skip]` and never affect the exit code, so "any
`[FAIL]` ⇒ exit 1" holds.

Two details worth knowing: `--json` emits only a JSON list of
`{name, status, detail, fix}` with `status` one of
`ok`/`FAIL`/`warn`/`skip`; and
because `-m silk.doctor` imports `silk` (hence numpy), a *missing numpy* gives
you numpy's traceback instead of a report. Direct-script mode also needs numpy;
install it in the selected environment before running the doctor. There is no `silk` console command; never write
`silk doctor`. Full contract in [docs/cli.md](docs/cli.md#silkdoctor--environment-check).

---

## Test commands and expectations

| Command | Scope | Requirements / release role |
|---|---|---|
| `.venv/bin/python -m pytest -q` | Fast suite — no simulator required; external-dependency tests skip when unavailable | Must pass in bare and configured profiles |
| `.venv/bin/python -m pytest -m "slow and not requires_xyce and not requires_gf180 and not requires_sg13g2" -q` | Core integration + wheel | Needs ngspice + sky130 (the `PDK_ROOT` setting, as above) |
| `.venv/bin/python -m pytest -m "slow and requires_xyce" -q` | Optional Xyce integration | Non-gating, but **required when changing Xyce paths**. Needs the `SILK_XYCE_BIN` setting or Xyce on `PATH` (the doctor shows which) |
| `.venv/bin/python -m pytest -m "slow and requires_gf180" -q` | GF180 preview integration | Needs ngspice + GF180MCU; non-gating for 0.2.0 |
| `SILK_NGSPICE_BIN=<ngspice 44 or newer> .venv/bin/python -m pytest -m "slow and requires_sg13g2" -q` | SG13G2 preview integration | Needs compatible ngspice + compiled OSDI models; non-gating for 0.2.0. The newer ngspice is a per-command override, never the settings file (that would change ngspice for every run); the owner's settings file names it in a comment |

Test counts change as work lands; they are not a fixed baseline in this contract.
Record the commit, environment, command and actual pass/skip/deselection counts
with each verification. Historical measurements do not establish that the
current candidate has passed.

`pyproject.toml` sets `addopts = "-m 'not slow'"`, so the plain `pytest -q`
invocation deselects the slow tests automatically — you do not need `-m 'not
slow'` yourself. The `slow` marker covers opt-in integration and artifact
tests. Capability markers split optional simulators and preview PDKs from the
sky130 core release profile so an unavailable dependency is never reported as
a pass and a preview check is never mistaken for a support guarantee.

The fast suite must also run on a bare checkout. Tests that genuinely need a
PDK or simulator carry `needs_pdk` / `needs_ngspice` from `tests/_env.py` and
skip with a reason when that dependency is unavailable. CI runs both the bare
profile and a configured sky130/ngspice matrix. A failure in the bare profile
usually means a new external dependency was not declared; do not hide it by
installing the dependency in that job.

**The fast suite must stay green.** A task agent verifies its change with the
test files that cover it (see Rules for task agents); it does not run the full
suites. The reviewer sizes each landing's checks to the change:

- **Backlog or documentation only:** the documentation tests the change
  touches, nothing more.
- **Code, examples or tests:** the inventory tests, the covering test files
  (slow included), and before/after output of the examples the change
  reaches.

The four fast chunks of `scripts/fast_chunks.py` run once every five landed
tasks, over everything landed since their last run, not for each task. The Python 3.10,
Xyce and browser legs run when the change touches what they test. The full
profiles (core slow, the bare profile, and the capability profile of any Xyce,
GF180 or SG13G2 change) run once every 20 landed tasks and before every release.

---

## Conventions you must know

### Units are PDK-native by default

Every shipped manifest sets `user_units: u` — circuit code writes microns on
every PDK (`load_pdk`'s bare-constructor default `UNIT_PDK` applies only when
constructing a PDK class directly). sky130 W/L are in **µm**: `w=4.0` means
4 µm, not 4 metres. Emitted netlists write W/L as plain numbers in the PDK
unit (sky130: `w=4 l=0.5`, meaning µm); other values such as times and
capacitances carry SI suffixes (`40n`, `10f`). **These are correct — do not
"fix" them to SI.** Pass
`load_pdk('sky130', user_units='si')` if you actually want metres.

### `add_sim_type` takes analysis *objects*

```python
from silk.simulation import Transient, AC, DC, Noise
ckt.add_sim_type(Transient(step=1e-9, stop=100e-9))   # correct
ckt.add_sim_type('transient')                          # TypeError
```

The legacy string form was removed and now raises `TypeError`. It also raises
`ValueError` if called on a subcircuit (a module with `module_name` set) —
subcircuits emit a `.subckt` block and do not own an analysis; put the analysis
on a top-level testbench module.

### Only `name` is positional on device factories

PDK device factories and the ideal `ElementLibrary` elements accept exactly one
positional argument — the instance name. Pins and parameters are
keyword-only:

```python
pdk.nmos1p8('m1', d='out', g='in', s='gnd', b='gnd', w=1.0, l=0.35)  # correct
pdk.nmos1p8('m1', 'out', 'in', 'gnd', 'gnd')                         # TypeError
elements.resistor('r1', p='out', n='gnd', value=10e3)                # correct
```

Devices created inside a module's with-block join that module automatically:

```python
with inv:
    pdk.nmos1p8('mn', d='OUT', g='IN', s='VSS', b='VSS', w=1.0, l=0.15)
    pdk.pmos1p8('mp', d='OUT', g='IN', s='VDD', b='VDD', w=2.0, l=0.15)
```

This is deliberate: a mis-typed pin fails at construction instead of surviving
inside a dict. Note the deprecated exception — `silk.elements.passives.Resistor` /
`Capacitor` / `Inductor` warn on construction and still take `pos`, `neg`, and
the value positionally
(`passives.Resistor('r1', 'a', 'b', 1e3)` works). Prefer keywords anyway.

### The embedded-testbench pattern

A block is a reusable, parametrizable `Module` carrying its ports, testbenches,
an optional behavioral model and specs. A `build_X()` / `make_X()` factory returns the block
with its characterization testbenches **already registered** via
`mod.create_test(...)`, so the module knows how to test itself and
`mod.run_all_tests()` drives the whole suite. Design knobs (W/L, bias currents,
load caps) are keyword arguments of the factory, defaulting to module-level
constants — never read from globals inside helpers, or importers cannot
override them.

`silk/examples/ota_5t_guide.py` is the teaching spine and is walked
step-by-step in [docs/agent-guide.md §4](docs/agent-guide.md).
`silk/examples/strongarm_tb.py` is the canonical multi-testbench version.

**Read the curated example index in [docs/agent-guide.md §7](docs/agent-guide.md)
before writing a new example.** The pattern you
need almost certainly exists, and the index says which file to copy.

### Declare probes before running, read them after

`tb.probe('vout')` returns a `Signal` that resolves against the testbench's
results on access. Bind registered metrics with their named signal slots,
for example `tb.add_metric('gain_dB', dc_gain_dB, sig=vout)`. Arithmetic such
as `(vp + vm) / 2` defers too. Callables/lambdas and `load_signal()` remain
supported; use them for custom extraction or post-metric scaling.

Probes are not saved waveform snapshots. Keep the bench alive and select
`analysis=` when ambiguous. Distinct corners already use separate output
folders; use `dut.set_output_options(mode='new')` to preserve results across
repeated runs. See the
[probe workflow](docs/probes.md) for runnable code, current naming, explicit
corner selection and the opt-in report-cache caveat.

### Golden tests assert byte-identical emission

`tests/refactor/` replays a fixture battery against frozen goldens in
`tests/refactor/goldens/`: byte-identical SPICE netlists, `export_schema` JSON,
structure hashes, and fault enumerations, plus a frozen `dir(Module)` name list
with per-callable signatures.

Consequences:

- **Do not reformat netlist emission.** Whitespace, value formatting, and line
  order are all under test.
- **Do not "tidy" the `_Base` trick.** Each `Module` part-mixin does:

  ```python
  if TYPE_CHECKING:
      from silk.circuit._module_protocol import _ModuleAttrs
      _Base = _ModuleAttrs
  else:
      _Base = object
  ```

  Collapsing this into ordinary class-body annotations leaks
  `__annotations__` into `dir(Module)` and stringifies signatures under PEP
  563 — both break the API golden. The reason is documented in
  `silk/circuit/_module_protocol.py`; read it before touching a part-mixin.

If a golden change is genuinely intended, regenerate it deliberately (see
[CONTRIBUTING.md](CONTRIBUTING.md)) and say so in the commit message.

### Behavioral simulation is the fast iteration loop

Select `dut.simulator = 'behavioral'` (or `tb.netlist.simulator` for one test)
for the pure-numpy engine — no external simulator or installed PDK models.
`tb.run()`, `run_all_tests()`, probes, metrics, specs, reports, plots and the
rawfile viewer use the same workflow as SPICE. Behavioral tests run in-process
regardless of `n_workers`. See [one workflow, two engines](docs/agent-guide.md#one-workflow-two-engines)
for executable examples and verification commands.

`mod.simulate_behavioral(t_step, t_stop)` is a shortcut through that backend,
restoring the caller's analyses and simulator selection afterwards. **BREAKING:**
the grid includes `t_stop`, `results_file` is populated, signal names use
rawfile spelling, and node-current requests raise instead of returning voltage.
`Net.voltage` is still populated. Behavioural Monte Carlo accepts `Normal` /
`Uniform` model parameter declarations and holds each sample's draws for the
whole run; see [the workflow](docs/monte-carlo.md#behavioural-models-declaring-variation).
Limits that bite:

- A modelled block needs port directions (`define_module_ports`) and a model
  (`set_behavioral_model`). It is a leaf replacing all its contents, including
  transistors. Blocks without models are entered (`xdut.<net>`); empty ones
  raise `SimulationError`, and unmodelled transistors raise `NotSupportedError`
  naming the path. Modelled-leaf internal nets cannot be probed.
- Transient is fixed-step. Ports-form feedback loops run in placement order:
  sample models once per point, vector models once over the whole array.
  Use `previous_value` and `initial_value` for explicit point delays; undelayed
  feedback warns. Mixed sample/vector loops raise `LoopError`; DC feedback raises
  `LoopError` after the point.
  DC source sweeps
  evaluate models once per point in order; `t` is the point index, and callable
  state can retain hysteresis. AC, noise, op and ambient temperature refuse.
- Outside modelled leaves, R/C loads to ground or declared supply rails are
  ignored; single-driver series R/L paths are wires; E sources evaluate gain
  times differential control voltage. Ignored/wired elements appear in one
  warning per run and in the spec report. Dividers, ambiguous R/L paths,
  series C, G/F/H and B elements raise `NotSupportedError`.
- Transient stimuli support `pulse`, `sine`, `pwl` and `dc`. `exponential`,
  `sffm`, `am`, `trnoise` and `trrandom` raise `NotImplementedError`; DC uses
  independent sources' DC values. Independent current sources expose their
  defined current in transient results (DC has no source branch-current
  vector), but voltage-source branch currents and device probes need an
  electrical solve. See the complete [refusal list](docs/known_issues.md#behavioral-simulation)
  and [probe table](docs/probes.md#behavioral-and-spice-probes).

### Fail loudly

silk raises typed errors instead of returning empty or plausible-looking
results. The hierarchy is in `silk/exceptions.py`: `CircuitError` →
`DuplicateNameError`, `UndeclaredNetError`, `SimulationError`
(→ `SimulationAborted`), `PDKError`, `NotSupportedError`. Notably silk does
**not** trust a simulator's exit code — a run that produces no data raises
`SimulationAborted` even when the simulator exited 0. For certified requests
emitted and executed by silk, ngspice/Xyce transient/DC/AC/noise results must
also reach their final grid point. The relative tolerance is `1e-7` of the
largest requested axis magnitude or span; discrete sweeps require the expected
point count and cap tolerance at one quarter of their final resolvable interval.
Sub-decade ngspice DEC AC sweeps use the version-specific grid from the
simulator YAML (stretched through 42, plain steps from 43); only unknown
versions retain no endpoint verdict. AC/Noise constructors refuse DEC sweeps
shorter than one whole step, including zero span. Logs enrich the reason while data decides completion.

External/direct runner/backend decks and unprovable requests retain only the
data-presence gate, with no endpoint verdict. This includes opaque custom
content, caller control commands, repeated analyses/ambiguous repeated plots,
`.step`, continuation cards, and nested DC (silk currently emits one DC
dimension). Opaque API entries are `add_custom_line()`/`custom_lines` anywhere
in the hierarchy, custom analysis subclasses passed to `add_sim_type()`,
replacement netlisters, and custom `to_spice()` emitters supplied through
`add()`/`add_source()`/`add_passive()` (including custom source waveform
emitters), custom PDK emission providers, and Module emission overrides.
See [analysis completion](docs/known_issues.md#analysis-completion).
Preserve this: when you add a failure path, raise a typed error with an
actionable message rather than falling back to a default and carrying on.

---

## Repo map

| Path | What lives there |
|---|---|
| `silk/circuit/` | `Module` (split across `modules_*.py` part-mixins), `ModuleTestbench`, `Net`, ports/roles, netlisters, reports, composition graph |
| `silk/pdk/` | `BasePDK`, `load_pdk`, PDK manifests (`data/*.pdk.yaml`), the `validate` CLI |
| `silk/elements/` | `ElementLibrary`, ideal `passives` (R/L/C), `AnalogElement`, device-class factory |
| `silk/sources/` | `VoltageSource`, `CurrentSource`, controlled sources, waveform shapes |
| `silk/signals/` | `Signal`, deferred probe expressions, `NgspiceSignal`, the MIT nutmeg rawfile reader, waveform plotting/explorer |
| `silk/simulation/` | `Transient`, `AC`, `DC`, `Noise`, `Corner`, the behavioral engine, backend dispatch |
| `silk/simulators/` | Per-simulator YAML configs (`ngspice.yaml`, `xyce.yaml`) — options syntax and presets |
| `silk/specs/` | `Spec`, `Measurement`, `CalibrationBinding` — the design-stage bridge |
| `silk/metrics/` | ADC / spectral / timing / jitter / power / distortion / noise metric library |
| `silk/lint/` | Analog design-rule framework: `RuleRegistry`, rules, waivers, reports |
| `silk/lint_markers/` | `NoConnect` / `NoConnectPort` — intentional-exception markers for lint |
| `silk/faults/` | Fault injection: `FaultSet`, `FaultCampaign`, `CampaignReport` |
| `silk/layout/` | Layout is in progress: `estimate_layout`, `RoutingModel`, `capture_layout`, `inject_parasitics`, EM budget |
| `silk/constraints/` | Layout-intent constraints (`SymmetricBlocks`, `GroupBlocks`, …) |
| `silk/optimize/` | `VariableSpace`, `Variable`, `Optimizer` — design-space sampling and search |
| `silk/db/` | SQLite tracking: `Database`, `CircuitTracker`, importer/exporter, `DesignStage` |
| `silk/codegen/` | `Generator` (spec JSON → silk Python) |
| `silk/utils/` | Shared SPICE value formatting |
| `silk/examples/` | Runnable worked examples — see docs/agent-guide.md §7 |

---

## Where to look

| Question | Doc |
|---|---|
| How do I build a circuit example? | [docs/agent-guide.md](docs/agent-guide.md) — read this first |
| What's the signature of X? | [docs/api-reference.md](docs/api-reference.md) |
| What command-line tools exist? | [docs/cli.md](docs/cli.md) — silk installs **no `silk` console command**; every tool is `python -m silk.<module>` |
| What's broken or unfinished? | [docs/known_issues.md](docs/known_issues.md) — honest, verified limitations |
| What work is open? | In the maintainers' repository, open work is tracked with Backlog.md: run `backlog task list --plain`. The public release does not include `backlog/`; there, see the issue tracker at github.com/levantlabs/silk/issues. [docs/capability-roadmap.md](docs/capability-roadmap.md) is the capability view. |
| What are the style rules? | [docs/coding-conventions.md](docs/coding-conventions.md) |
| How do I contribute a change? | [CONTRIBUTING.md](CONTRIBUTING.md) |
| How do I add a PDK? | [docs/adding-a-pdk.md](docs/adding-a-pdk.md) |
| A first walkthrough | [docs/quickstart.md](docs/quickstart.md) |
| Which doc answers my question? | [docs/README.md](docs/README.md) — the full triage table |

---

## Gotchas

- **Examples write to `silk-output/` under the working directory** (gitignored),
  or to the `SILK_EXAMPLES_OUT` setting. Get the path from
  `silk.examples.example_output_dir()`; never from `os.path.dirname(__file__)`,
  which for an installed user is inside `site-packages`
  (`tests/test_example_output_paths.py` rejects it).
- **`Module(pdk_name, dir_folder, netlistname, …)` creates `dir_folder`** if it
  is missing (`os.makedirs(..., exist_ok=True)`, nested paths included).
- **Simulator configs are `silk/simulators/*.yaml`.** Options syntax, presets,
  and the known-options allowlist are data, not code. Assigning an unregistered
  name raises immediately — `mod.simulator = 'bogus'` gives
  `ValueError: No simulator backend registered for 'bogus'. Registered:
  ['ngspice', 'xyce']`.
- **PDK manifests are `silk/pdk/data/*.pdk.yaml`.** Adding a PDK is a manifest
  plus a device-details YAML, not a code change. Both are shipped as package
  data — if you add a data file, check `[tool.setuptools.package-data]` in
  `pyproject.toml` covers it.
- **`import silk.constraints` is a side-effect import.** It monkey-patches
  `Module.add_constraint` / `Module.list_constraints` onto `Module`. Without the
  import you get `AttributeError`.
- **`silk.metrics` is the ONE metric library.** It is organised by purpose
  (`waveform`, `statistics`, `timing`, `jitter`, `eye`, `power`, `spectral`,
  `rf`, `converter`, `fom`, `ac`, `noise`) and owns
  every metric, gain/bandwidth/phase-margin included. Import metrics from
  `silk.metrics`.
- **Branch currents come from voltage-source ammeters.** There is no current
  vector for a node — insert a (0 V if needed) `VoltageSource` and read
  `tb.probe('i(vs)')` for a source named `vs`.
- **Run examples under `MPLBACKEND=Agg`** so plotting does not try to open a
  window.

---

## Working agreement

**Verify before you claim.** This repo's docs are written to be trusted
verbatim: `docs/cli.md` quarantines anything that could not be confirmed by
running it, and `docs/known_issues.md` states limitations rather than leaving
them to be discovered. Hold new text to the same bar — run the command, read the
source, then write it down.

silk is MIT-licensed. Do not copy code from other EDA projects into it (several
nearby ones are GPL or unlicensed), and do not vendor a dependency without
checking its license.

---

This section and the Backlog.md workflow below apply only to agents using the
maintainers' private task board on the development machines. A public clone has
neither that board nor those machines; everything above applies to every clone.

**Historical task tags.** Tags such as `TASK-NN` cite the unpublished development
board. Keep these historical references; explanations belong in the comments
alongside them.

## Rules for task agents

These apply to every agent working a backlog task on a `task/task-N` branch.
The reviewer gates and merges; you implement, record and stop.

**Where you are.** Confirm `pwd` and `git rev-parse --abbrev-ref HEAD` before
your first edit and abort if either is wrong. Several agents share this repo
and its git index.

**Running code.**
- The worktree's `.venv` may link to a shared environment whose silk install
  may point elsewhere or be absent. pytest from the worktree root is fine. A standalone script or example run needs
  `PYTHONPATH=<worktree>` and should print `silk.__file__` once to prove it
  imports your tree.
- Tool and PDK locations come from the silk settings: run
  `.venv/bin/python -m silk.doctor` to see them, and do not export your own
  values over them. SG13G2 runs need a per-command `SILK_NGSPICE_BIN` (ngspice
  44 or newer, see the test table above). Use a 10-minute Bash timeout for
  long runs.
- The machine is shared. If `~/.cache/silk/heavy.sh` exists, run every
  heavy run through it, inside its own memory scope. A heavy run is any
  pytest invocation that launches a simulator (ngspice or Xyce), even a single
  test file or a single test; any pytest over more than one file; any example;
  and any batch of simulations. The scope goes OUTSIDE heavy.sh, never the
  other way round:

  ```bash
  DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus systemd-run --user --scope -q \
    -p MemoryMax=10G -p MemorySwapMax=0 bash ~/.cache/silk/heavy.sh <command>
  ```

  The scope caps the whole process tree (pytest, workers, simulators) at
  10 GB. Other projects on this machine hold their own heavy steps while any
  silk heavy run, or heavy.sh's background watcher, sits outside such a scope.
  A run already started cannot be moved into a scope; let it finish and scope
  the next one. heavy.sh may hold a run back while another project's gate or
  check holds the machine; that wait is expected, so never bypass the wrapper
  to avoid it. Always single-process pytest (no `-n`), simulations one at a
  time.
- Never `pkill`/`killall` by pattern (`pkill -f pytest`, `pkill -f heavy.sh`):
  on a shared machine that kills other agents' and the reviewer's runs. Stop
  only processes you started, by the PID you recorded when you started them.
- Verify with the tests that cover your change, not the full suites: every
  test file that imports, runs or names a module you changed, plus the tests
  you added, with `-m "slow or not slow"`. Run them one file per pytest
  invocation (a single simulator-free file needs no `heavy.sh`; a file that
  launches a simulator always goes through it, as above), and record each file's
  pass/fail/skip in the task notes. Do not run the four fast chunks or the core
  slow profile: the reviewer runs them on the schedule in Test commands and
  expectations. An acceptance criterion that names them stays unticked; the
  reviewer ticks it at the gate. For reference, the chunks are
  `$(.venv/bin/python scripts/fast_chunks.py N)` for N = 1, 2, 3, 4 (table
  `CHUNKS`; `tests/test_fast_chunks.py` fails if a test file is in no chunk or
  two).

**What you may change.**
- `silk/examples/` is joint work with the owner: change only the example files
  your task names, and only as the task says.
- Examples use silk's API. Before writing a helper in an example, search
  `silk.sources`, `silk.metrics`, `silk.signals` and the example index in
  docs/agent-guide.md §7, and call the existing function (for example
  `sources.BitPattern` / `sources.PRBS` / `prbs_bits` for bit stimulus,
  `silk.metrics.sample` for resampling). If silk lacks a general helper, do
  not hide a private copy in the example: list it in the task notes as a
  candidate silk API. The reviewer checks every example diff for new helpers.
- Never name a commercial EDA tool or vendor in anything you add: code,
  docstrings, comments, tests, docs, commit messages. State definitions in
  silk's own terms.
- Add a `CHANGELOG-dev.md` entry for any user-visible change, marking breaking
  changes as such. The public `CHANGELOG.md` gets one section per release,
  drafted by the reviewer and reviewed by the owner.
- Docstrings, rendered `#:` documentation comments, docs and CHANGELOG state
  behaviour: no task numbers, reviews, rulings or Q numbers. Put process history
  in task notes and commit messages.
- Do not widen scope. For a decision outside the acceptance criteria, write the
  question in your pane and in the task notes, commit, and stop on that point.

**Git and backlog.**
- Every new tracked file needs an explicit `MANIFEST.public` entry: `include`
  if it ships publicly, `exclude` otherwise. The public export refuses an
  unclassified tracked path.
- **Worktree / backlog split-brain (agents).** The `backlog` CLI writes task
  files into the checkout it runs in. An agent working in a git worktree edits
  its worktree's copy of the task; the claim, plan and notes are not visible on
  `main` until that branch merges, and a concurrent edit to the same task on
  `main` conflicts on merge. Treat the claim commit on `main` as the
  authoritative signal, keep plan/notes edits on the task branch, and expect to
  resolve the task-file merge by hand. Nothing enforces this; it is recorded
  here because it was previously only a buried task comment.
- Stage files by name. Never `git add -A` or `git add .`; never commit the
  `.venv` link. Read back `git show --stat HEAD` before reporting a sha.
- Read the whole task, including its implementation notes: the owner's rulings
  live there. Use `--append-notes`, never `--notes` (it replaces them). Never
  put backticks or `$(...)` inside a double-quoted backlog argument; write long
  notes to a file and pass `"$(cat file)"`.
- Do not merge your branch or delete it.

**When done.** Tick each acceptance criterion you verified, append the evidence
(commands, results, per-file test counts), write a final summary, set the
status to Done, and commit code and task file together on your branch. Then
report in your pane: the commit sha, the exact file list and the test counts.

The Backlog workflow below applies only in the maintainers' development
checkout with its private `backlog/` board. Public clones use the contributor
instructions and public issue tracker; no Backlog command is required.

<!-- BACKLOG.MD GUIDELINES START -->
<!-- backlog.md-instructions-version: 1.50.1 -->
<CRITICAL_INSTRUCTION>

## Backlog.md Workflow

This workflow applies only in the maintainers' development checkout with its
private `backlog/` board. Public clones use the contributor instructions and
public issue tracker; no Backlog command is required.

**In that development checkout, for every user request, run `backlog instructions overview` before answering or taking action.**

Use the overview to decide whether to search, read, create, or update Backlog tasks.

Before task lifecycle actions, read the matching detailed guide:
- `backlog instructions task-creation` before creating or splitting tasks
- `backlog instructions task-execution` before planning, changing status or assignee, adding a plan or implementation notes, or implementing task work
- `backlog instructions task-finalization` before checking acceptance criteria, writing final summaries, or moving tasks to terminal statuses

Use `backlog <command> --help` before running unfamiliar commands. Help shows options, fields, and examples.

Do not edit Backlog task, draft, document, decision, or milestone markdown files directly. Use the `backlog` CLI so metadata, relationships, and history stay consistent.

</CRITICAL_INSTRUCTION>
<!-- BACKLOG.MD GUIDELINES END -->
