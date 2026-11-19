# AGENTS.md — contract for AI agents working in this repo

silk (`import silk`, PyPI package `circuitsilk`) describes circuits in Python, emits SPICE, simulates, measures and checks specs.

**Before building a circuit example, read [the agent guide](docs/agent-guide.md#1-what-silk-is--the-build--characterize--report--netlist-loop) and its [example index](docs/agent-guide.md#7-curated-example-index).** This file covers the environment and rules that are easy to miss.

## Environment contract

**Use `.venv/bin/python` from the repo root.** Bare `python` may lack dependencies. Before debugging the environment, run:

```bash
.venv/bin/python -m silk.doctor
```

Doctor shows tool locations, their setting sources, dependencies and model installations. Exit 0 means required checks passed; exit 1 means a required check failed. Warnings and optional skips are not failures or support guarantees. For JSON output and the missing-numpy caveat, see [doctor's contract](docs/cli.md#silkdoctor--environment-check). There is no `silk` console command: use `python -m silk.<module>`.

Settings resolve **environment variable → settings file → default**. Check doctor before declaring a tool unavailable because it is absent from `PATH`. Never create or edit the owner's settings file yourself. See [settings and defaults](docs/settings.md#the-settings) and [override precedence](docs/settings.md#precedence).

ngspice 42+ is the default simulator. Versions 45, 45.2 and 46 have a [sky130 loading slowdown](docs/known_issues.md#slow-sky130-library-loading-on-ngspice-45-and-46). Xyce is an optional preview backend: select `mod.simulator = 'xyce'`; see [backend selection](docs/agent-guide.md#choosing-a-simulator--ngspice-default-or-xyce).

For sky130, `PDK_ROOT` must contain `sky130A/libs.tech/ngspice/sky130.lib.spice`. It defaults to `~/pdk`. Catalog loading and netlist generation need no installed models; transistor simulation does. See [installation](docs/installing-the-pdk.md#recommended-pdksetup) and [load_pdk overrides](docs/api-reference.md#silkpdk). Only sky130 is release-validated for transistor simulation. GF180MCU and SG13G2 are preview/catalog integrations; `silk.pdk.available()` means registered catalogs, not validated simulations.

Ideal-only SPICE needs a simulator but no PDK. Custom netlist content conservatively keeps model includes; see the [PDK-free walkthrough](docs/quickstart.md#the-middle-rung-real-spice-with-no-pdk). Behavioral simulation needs neither external simulator nor PDK. Run `.venv/bin/python -m silk.examples.behavioral_example` or try the [quickstart](docs/quickstart.md#1-60-second-start-no-pdk-no-ngspice).

The optional headless-browser smoke needs a working unpacked browser selected by `SILK_FIREFOX` or `SILK_CHROME`; the required Node differential gate does not. Snap/system browsers may fail headless. Check the executable exists before naming it.

## Test commands and expectations

Run these from the repo root:

| Check | Command |
|---|---|
| Fast | `.venv/bin/python -m pytest -q` |
| Core integration and wheel: ngspice + sky130 | `.venv/bin/python -m pytest -m "slow and not requires_xyce and not requires_gf180 and not requires_sg13g2" -q` |
| Xyce | `.venv/bin/python -m pytest -m "slow and requires_xyce" -q` |
| GF180MCU | `.venv/bin/python -m pytest -m "slow and requires_gf180" -q` |
| SG13G2 | `.venv/bin/python -m pytest -m "slow and requires_sg13g2" -q` |

Plain pytest excludes `slow` automatically. Slow tests cover opt-in integration and artifacts. The fast suite must pass in both bare and configured environments. Dependency tests use `needs_pdk` / `needs_ngspice` from `tests/_env.py` and skip with a reason. A bare-profile failure usually means a new external dependency was not declared; do not hide it by installing that dependency in the bare job. CI checks every supported Python in the bare profile and Python 3.12 with sky130/ngspice.

Optional profiles are non-gating; run the Xyce profile when changing Xyce paths. GF180MCU needs ngspice and its models. SG13G2 needs compiled OSDI models and ngspice 44+: set `SILK_NGSPICE_BIN` for that command, not in the owner's settings file. See [test profiles](CONTRIBUTING.md#running-tests).

Record the commit, environment, command and actual pass/skip/deselection counts. Counts change; an older run does not verify today's candidate.

## Conventions you must know

### Units are PDK-native by default

Shipped manifests use `user_units: u`: W/L are in **µm**, so `w=4.0` means 4 µm. Emission such as `w=4 l=0.5` is correct; do not convert it to metres. Times and capacitances use SI suffixes such as `40n` and `10f`. Use `load_pdk('sky130', user_units='si')` for metres. [The PDK API](docs/api-reference.md#silkpdk) explains manifest defaults and bare `BasePDK` construction.

### `add_sim_type` takes analysis objects

```python
from silk.simulation import Transient
ckt.add_sim_type(Transient(step=1e-9, stop=100e-9))
```

Strings raise `TypeError`. Analyses belong on top-level testbenches; calling this on a subcircuit (`module_name` set) raises `ValueError`.

### Only `name` is positional on device factories

PDK and `ElementLibrary` factories require keyword pins and parameters:

```python
pdk.nmos1p8('m1', d='out', g='in', s='gnd', b='gnd', w=1.0, l=0.35)
elements.resistor('r1', p='out', n='gnd', value=10e3)
```

Positional pins raise `TypeError`; misspelled pins fail at construction. Inside `with mod:`, constructors register automatically; see [auto-registration](docs/agent-guide.md#context-manager-auto-registration-with-mod). Deprecated `silk.elements.passives.Resistor`, `Capacitor` and `Inductor` still accept positional pins/value and warn. Prefer the keyword factories.

### The embedded-testbench pattern

Return a `Module` with tests already registered through `mod.create_test(...)`; run them with `mod.run_all_tests()`. Make design knobs keyword arguments with constant defaults, not globals read inside helpers. Copy the [worked OTA pattern](docs/agent-guide.md#4-worked-spine--silkexamplesota_5t_guidepy); the [example index](docs/agent-guide.md#7-curated-example-index) includes `strongarm_tb.py` for multiple tests. See [factory conventions](CONTRIBUTING.md#adding-an-example).

### Declare probes before running, read them after

`vout = tb.probe('vout')` returns a deferred `Signal`. Arithmetic defers too. Bind metric slots by name: `tb.add_metric('gain_dB', dc_gain_dB, sig=vout)`. Callables, lambdas and `load_signal()` remain supported; see [metric binding](docs/probes.md#metrics-bind-signal-slots-by-name).

Probes follow results; they are not snapshots. Keep the bench alive and choose `analysis=` when ambiguous. Corners have separate folders. Use `dut.set_output_options(mode='new')` to preserve repeated runs. Read [selection](docs/probes.md#keep-and-select-corner-waveforms) and [result lifetime](docs/probes.md#result-lifetime-and-limits) before retaining waveforms.

### Golden tests assert byte-identical emission

`tests/refactor/goldens/` pins SPICE, schema JSON, structure hashes, fault enumeration and `dir(Module)` signatures. **Do not reformat emission:** whitespace, values and order matter. Regenerate intended changes deliberately and explain them in the commit; follow [the golden-test rule](CONTRIBUTING.md#the-golden-test-rule).

Keep the `Module` part-mixin `_Base` pattern:

```python
if TYPE_CHECKING:
    from silk.circuit._module_protocol import _ModuleAttrs
    _Base = _ModuleAttrs
else:
    _Base = object
```

Ordinary class-body annotations leak `__annotations__` and stringify signatures under PEP 563. Read `silk/circuit/_module_protocol.py` before changing a mixin; see [internal gotchas](docs/architecture.md#5-internal-gotchas).

### Behavioral simulation

Set `dut.simulator = 'behavioral'`, or `tb.netlist.simulator` for one test. It uses the same test/probe/report workflow and runs in-process regardless of `n_workers`. A model needs port directions and replaces the block's contents; a transistor in a block without a model raises `NotSupportedError`. See [one workflow, two engines](docs/agent-guide.md#one-workflow-two-engines), including the `simulate_behavioral(t_step, t_stop)` shortcut and restored settings.

This is not an electrical solver. Transient is fixed-step; DC evaluates points without settling. Read the [limits and refusals](docs/known_issues.md#behavioral-simulation) for feedback, approximated passives, sources and unsupported analyses, and the [probe limits](docs/probes.md#behavioral-and-spice-probes). [Behavioral Monte Carlo](docs/monte-carlo.md#behavioural-models-declaring-variation) supports `Normal`/`Uniform` model parameters, held throughout each sample run.

### Fail loudly

Use typed errors with actionable messages, not fallback values that look valid. See the [exception hierarchy](docs/quickstart.md#13-exceptions). Simulator exit 0 does not prove success: no result data raises `SimulationAborted`.

Silk verifies final grid points for certified generated transient/DC/AC/noise requests. Unverified or external decks retain only data-presence checks. Preserve this distinction; [analysis completion](docs/known_issues.md#analysis-completion) defines tolerances, version-dependent grids and certification exclusions. Saved data decides completion; logs explain failures.

## Gotchas

- Examples write to `silk-output/` under the working directory, or `SILK_EXAMPLES_OUT`. Use `silk.examples.example_output_dir()`, never `os.path.dirname(__file__)`, which points into the installed package. `silk-output/` is gitignored; `tests/test_example_output_paths.py` checks the path rule. Run examples with `MPLBACKEND=Agg`.
- `Module(..., dir_folder, ...)` creates missing output directories, including parents; see [output paths](docs/schema-export.md#d2-output_dir).
- Simulator options, presets and allowlists are data in `silk/simulators/*.yaml`. Unknown backend names raise `ValueError` immediately; see [backend selection](docs/agent-guide.md#choosing-a-simulator--ngspice-default-or-xyce).
- PDK manifests and device catalogs live in `silk/pdk/data/`. Follow the [PDK checklist](docs/adding-a-pdk.md#7-checklist-adding-a-new-pdk), including package-data coverage in `pyproject.toml`.
- Import `silk.constraints` before `Module.add_constraint` or `list_constraints`; they are installed by that [side-effect import](docs/agent-guide.md#import-silkconstraints-is-a-side-effect-import).
- Import measurements from `silk.metrics`. Check the [metric library](docs/metrics-library.md#layout) before writing a measurement helper.
- A node has no branch-current vector. Insert a voltage-source ammeter (0 V if needed), then read `tb.probe('i(vs)')` for source `vs`; see [current probes](docs/probes.md#what-can-be-probed).

## Where to look

The [package map](docs/architecture.md#1-package-map) explains the code directories and ownership. Use the [documentation index](docs/README.md#start-here) for guides, [API reference](docs/api-reference.md), [known issues](docs/known_issues.md), [coding conventions](docs/coding-conventions.md) and [contributor instructions](CONTRIBUTING.md). Open work is in the [issue tracker](https://github.com/levantlabs/silk/issues); the [roadmap](docs/capability-roadmap.md) describes capabilities.

## Working agreement

Verify before claiming success: read the source, run the relevant command and record what happened. Keep docs factual and state limitations.

Silk is MIT-licensed. Do not copy code from other EDA projects. Check licenses before vendoring dependencies; see [license policy](CONTRIBUTING.md#license).

Tags such as `TASK-NN` in source comments cite the private development board. Keep them; explain them in the comment alongside.

