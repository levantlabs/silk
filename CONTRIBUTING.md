# Contributing to silk

silk is pre-release (`0.2.x`, Development Status :: 3 - Alpha). The API still
moves. Contributions are welcome; this page covers setup, tests, and the few
rules that are easy to break by accident.

If you are an AI agent, read [AGENTS.md](AGENTS.md) — it is the repo contract —
and [docs/agent-guide.md](docs/agent-guide.md) if you are writing an example.

---

## Dev setup

```bash
git clone https://github.com/levantlabs/silk.git
cd silk
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

`[dev]` adds `pytest`, `pytest-cov`, `build`, and `twine` on top of the runtime
deps (numpy, matplotlib, pyyaml, pydantic). Python 3.10–3.13.

**Simulator.** ngspice is required for SPICE tests and most examples:

```bash
sudo apt install ngspice        # Debian/Ubuntu
```

Xyce is an optional preview backend (`mod.simulator = 'xyce'`, needs the Xyce CLI on `PATH` or
`$SILK_XYCE_BIN`).

**PDK.** Install sky130 through [pdksetup](https://github.com/levantlabs/pdksetup)
with `pdk install sky130`. The `PDK_ROOT` setting must point to a directory
containing `sky130A/libs.tech/ngspice/sky130.lib.spice`. It resolves from the
environment, then the user's settings file, then `~/pdk`; see
[docs/settings.md](docs/settings.md) and the full
[PDK installation guide](docs/installing-the-pdk.md).

Ideal-device SPICE needs a simulator but no installed PDK models: circuits
built entirely from ideal elements, sources and passives (including ideal-only
subcircuits) omit PDK includes. Circuits containing PDK devices need both the
simulator and the models. Custom/opaque netlist content conservatively retains
PDK includes. Behavioral simulation
(`.venv/bin/python -m silk.examples.behavioral_example`) needs neither.

**Run Python as `.venv/bin/python`.** A bare `python` may resolve to a
different environment and produce confusing import or matplotlib failures.

**Verify the whole setup in one command:**

```bash
.venv/bin/python -m silk.doctor
```

It checks the interpreter, every dependency, ngspice, `PDK_ROOT`, each
registered PDK library, and the matplotlib backend, printing a fix hint per
failure. Exit 0 means you are ready to work.

---

## Running tests

```bash
.venv/bin/python -m pytest -q  # fast suite
# Core release profile: ngspice/sky130 integration plus wheel validation
.venv/bin/python -m pytest \
  -m "slow and not requires_xyce and not requires_gf180 and not requires_sg13g2" -q
```

`pyproject.toml` sets `addopts = "-m 'not slow'"`, so the plain invocation
already deselects the slow tests. The fast suite needs neither a simulator nor
a PDK: tests that genuinely depend on one use `needs_ngspice` / `needs_pdk`
from `tests/_env.py` and skip honestly when it is absent. CI runs a scrubbed
bare profile to enforce that contract, plus configured jobs that exercise the
dependency-gated tests. The suite **must stay green** — run it before opening a
PR. A single-process run of the whole fast suite can be killed for memory on a
shared machine, so run it in four chunks instead:
`.venv/bin/python -m pytest -q $(.venv/bin/python scripts/fast_chunks.py N)` for
N = 1 to 4 (the table is `CHUNKS` in `scripts/fast_chunks.py`). Run the core slow profile as well if your change touches simulation,
netlist emission, or the PDK layer.
Capability-specific profiles are explicit:

```bash
.venv/bin/python -m pytest -m "slow and requires_xyce" -q
.venv/bin/python -m pytest -m "slow and requires_gf180" -q
SILK_NGSPICE_BIN=/path/to/ngspice-44-or-newer \
    .venv/bin/python -m pytest -m "requires_sg13g2" -q      # add "slow and" for corners x temperature
```

Xyce is an optional preview backend and non-gating. GF180MCU and SG13G2 are
preview/catalog integrations; the GF180 capability profile requires
its PDK and is deliberately non-gating, and so is the SG13G2 one, which also
needs compiled OSDI plugins and an ngspice ≥ 44 that loads them (see
[installing-the-pdk.md](docs/installing-the-pdk.md)). Any future release that promotes one
of those PDKs to release-validated simulation support must install it in CI and
gate the matching real-model profile.

Coverage:

```bash
.venv/bin/python -m pytest -q --cov=silk --cov-report=term-missing
```

---

## The golden-test rule

`tests/refactor/` guards `Module` against silent drift by replaying a fixture
battery against frozen goldens in `tests/refactor/goldens/`:

- byte-identical SPICE netlists,
- byte-identical `export_schema` JSON,
- structure hashes,
- device-path fault enumerations,
- a frozen `dir(Module)` name list with per-callable signatures.

**Do not regenerate goldens to make a test pass.** A golden failure means your
change altered observable output. Either that was unintended — fix the code — or
it was intended, in which case:

1. Confirm the new output is right by reading the diff, not by trusting the tool.
2. Regenerate: `.venv/bin/python scripts/capture_module_goldens.py`
3. Commit the regenerated goldens **in their own commit**, and say in the message
   what changed and why the new bytes are correct.

Two specific traps:

- **Netlist formatting is under test.** Whitespace, value formatting, and line
  order all count. Do not tidy emission code cosmetically.
- **Do not collapse the `_Base` trick** in the `Module` part-mixins
  (`if TYPE_CHECKING: _Base = _ModuleAttrs else: _Base = object`) into ordinary
  class-body annotations. It leaks `__annotations__` into `dir(Module)` and
  stringifies signatures, which breaks the API golden. The reason is written
  down in `silk/circuit/_module_protocol.py`.

---

## Code style

Read [docs/coding-conventions.md](docs/coding-conventions.md) before writing
code. In brief:

- PEP 8, `snake_case` for functions/methods/attributes, `CapWords` for classes,
  `UPPER_SNAKE_CASE` for module constants, leading `_` for internals.
- **No camelCase in new code.** The pre-2026-04 API used `addElement`,
  `getSignalNames`, etc.; those were renamed and must not come back.
- Absolute imports (`from silk.circuit.net import Net`). Function-local imports
  are fine to break cycles — comment why.
- Type hints on public functions and methods; numpy-style docstrings on public
  APIs.
- **Comments explain *why*, not *what*.** Default to no comment. Never reference
  the PR or task that added a line.
- Raise silk's typed exceptions (`CircuitError`, `DuplicateNameError`,
  `UndeclaredNetError`, `SimulationError`, `PDKError`) for silk-specific
  failures; `ValueError`/`TypeError` for generic programming errors. Never
  `assert` for user-facing validation — it is stripped under `-O`.
- Prefer behavior tests over implementation tests, and use the public API
  (`ckt.add`, `add_element`, `add_net`) rather than hand-building internal dict
  state.

---

## Adding an example

Examples live in `silk/examples/` and follow the conventions in
[docs/agent-guide.md](docs/agent-guide.md). **Read §7's example index first** —
the pattern you need probably exists and should be copied rather than reinvented.

An example contributes four things:

1. **A `build_X()` factory** returning a `Module` with its testbenches already
   registered via `mod.create_test(...)`. Design knobs (W/L, bias currents, load
   caps) are keyword arguments defaulting to module-level constants — not
   globals read inside helpers.
2. **A fast build test** (no marker) asserting structure: device count, ports,
   registered testbenches, declared specs, and that `generate_netlist()` emits.
3. **A slow SPICE test** marked `@pytest.mark.slow` and skipped when ngspice
   or the PDK is absent:

   ```python
   import shutil
   import pytest
   from silk.simulation.runner import ngspice_executable
   from tests._env import needs_pdk

   @needs_pdk
   @pytest.mark.slow
   @pytest.mark.skipif(shutil.which(ngspice_executable()) is None,
                       reason="ngspice not installed")
   def test_x_characterizes_itself(tmp_path):
       ...
   ```

   `tests/test_ota_5t_guide.py` is the model to copy.
4. **A row in the agent-guide §7 index** describing what the example teaches —
   including its honest limits. The index is how the next agent finds your work.

Get your output directory from `silk.examples.example_output_dir()` —
`$SILK_EXAMPLES_OUT` when set, else `silk-output/` under the working directory.
Never build the path from `os.path.dirname(__file__)`: the examples ship in the
wheel, so that writes inside `site-packages`. `tests/test_example_output_paths.py`
rejects it. Run the example end to end under `MPLBACKEND=Agg` before
submitting.

---

## Commits

Describe user-visible changes in your pull request.

Use conventional-commit-style prefixes: `feat:`,
`fix:`, `docs:`, `test:`, `refactor:`, `chore:`. An optional scope is common —
`feat(examples):`, `fix(lint):`, `docs(agent-guide):`.

```
feat(examples): SAR ADC resolution sweep — behavioral + SPICE + sine
fix: two first-run footguns — auto-create output dir, fail loudly on unmodeled waveforms
```

- The message explains *why*; the diff shows *what*.
- Keep commits focused. Renaming a method is one commit; renaming it, refactoring
  it, and adding tests is three.
- Never `--no-verify` without an explicit reason, and never force-push a branch
  someone else may be on.

---

## License

silk is MIT-licensed (© 2026 Levant Labs) — see [LICENSE](LICENSE). By
contributing you agree your contribution ships under the same terms.

Do not copy code from other EDA projects into silk, whatever their license, and
do not vendor a dependency without checking its license first. Several nearby
open-source EDA projects are GPL or unlicensed; silk stays clean-room by
policy. Third-party components silk does vendor (currently just uPlot, MIT,
for the waveform explorer) are tracked in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
