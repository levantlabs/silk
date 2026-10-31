# silk coding conventions

Short, practical rules for code added to silk. Follow PEP 8 unless a rule
below says otherwise.

---

## Naming

**Modules, packages, functions, methods, variables, attributes**: `snake_case`.

```python
def compute_parasitics(module): ...      # good
def computeParasitics(module): ...       # bad — camelCase is not our style

class Net:
    def __init__(self, name, module):
        self.fanout_limit = 0             # good
        self.fanoutLimit  = 0             # bad
```

**Classes**: `CapWords`.

```python
class PyCircuitElements: ...              # good
class Net: ...                             # good
```

**Constants (module-level)**: `UPPER_SNAKE_CASE`.

```python
DEFAULT_PDK_ROOT = '/opt/pdk'              # good
default_pdk_root = '/opt/pdk'              # bad — looks like a variable
```

**Enums**: class name in `CapWords`, members in `UPPER_CASE`.

```python
class PortDir(str, Enum):
    IN     = 'input'
    OUT    = 'output'
    INOUT  = 'inout'
```

**"Private" / internal names**: prefix with a single underscore `_`. Double
underscore is reserved for name-mangled attributes (rare — think twice).

```python
class Module:
    def __init__(self, ...):
        self._behavioral_results = {}     # internal, snake_case + underscore
        self._module_ref = None           # internal, snake_case + underscore

    def _bind_pin_to_net(self, ...):     # internal method
        ...
```

**Dict / list attributes**: use a consistent suffix.

```python
self.elem_dict = {}          # dict of elements
self.net_dict = {}           # dict of nets
self.module_dict = {}        # dict of submodules
self.custom_lines = []       # list — either a semantic name or a _list suffix
```

Use meaningful plural names; add `_dict` or `_list` only when it clarifies
the collection.

---

## Imports

- Stdlib first, then third-party, then silk modules. One blank line between
  groups.
- Prefer absolute imports (`from silk.circuit.net import Net`) over
  relative imports.
- Local imports inside a function are acceptable (and sometimes required) to
  break circular dependencies — always comment why:

```python
def __setattr__(self, name, value):
    if name in self.__class__._pin_names:
        # Local import: defer circuit construction dependencies.
        from silk.circuit.net import PortBinding
        ...
```

- Preserve the runtime annotation/signature pattern in Module part-mixins;
  follow `silk/circuit/_module_protocol.py`. Elsewhere, future annotations are encouraged
  in files that hold forward references or type hints that would otherwise
  cause import cycles.

---

## Type hints

- Public methods and public module-level functions: include type hints.
- Internal methods: hints are nice-to-have but not required.
- Use `Optional[X]` for nullable parameters, not `Union[X, None]`.
- Use `from __future__ import annotations` to avoid having to quote forward
  references, except in Module part-mixins (follow `_module_protocol.py`).
- If a hint would force a circular import, use `TYPE_CHECKING`:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from silk.circuit.modules import Module

class Net:
    module: 'Module'         # forward reference via future annotations
```

---

## Docstrings

- Use numpy-style for public APIs. Short, one-line description, then sections
  (`Parameters`, `Returns`, `Raises`, `Examples`) as needed.
- Internal helpers: a one-line docstring explaining *what* and *why* is fine.
- Avoid describing what the code literally does — describe intent and
  non-obvious constraints.

```python
def _bind_pin_to_net(self, net_name: str, binding: PortBinding) -> Net:
    """Register a pin binding on the Net side.

    This is the single chokepoint for pin-to-net binding. Every add/connect
    path funnels through here so that ``Net.pins`` stays consistent with
    owner-side state. Handles rewires automatically via the
    ``_pin_to_net`` reverse index.
    """
```

---

## Comments

- Default: **no comment**. Well-named code doesn't need narration.
- Only add comments for the *why* when it's non-obvious: a hidden constraint,
  a subtle invariant, a workaround for a bug, a defensive choice explained.
- Don't describe what the next line does — that's what the code says.
- Don't reference the PR or task that added the line ("added for issue
  #123"). That belongs in the commit message.

```python
# Bad — narrates obvious behavior
i = i + 1   # increment i

```

---

## Error handling

- Raise one of silk's typed exceptions (`CircuitError`, `DuplicateNameError`,
  `UndeclaredNetError`, `SimulationError`, `PDKError`) when the failure is a
  silk-specific concept. Use `ValueError` / `TypeError` for generic
  programming errors.
- Don't swallow exceptions silently. If a defensive fallback is intentional,
  comment why and what the fallback provides.
- Don't use assertions for user-facing validation — `assert` may be stripped
  under `-O`. Use `raise` with a typed exception.

---

## Testing

- New code: write tests. Favor **behavior tests** (does `add_module` populate
  `Net.pins` correctly?) over **implementation tests** (does `_bind_pin_to_net`
  call `_ensure_net` exactly once?).
- Follow the TDD cycle (red → green) when a skill or plan says to. Otherwise
  tests-alongside-implementation is fine.
- Avoid fixtures that hand-construct internal mock state (e.g.
  `module_dict[name] = {'device': ...}` or `net_dict[name] = []`). Use the
  public API (`add_module`, `add_net`, `ckt.add(device)`) so the test
  exercises real code paths.
- Use the shared capability markers when a required simulator or PDK is
  unavailable; do not skip a failing supported case. Track unfinished work
  outside docstrings.

---

## Git / commits

- Conventional-commit-style prefixes are encouraged: `feat:`, `fix:`,
  `refactor:`, `test:`, `docs:`, `chore:`.
- Commit messages describe *why*, not *what*. The diff shows the *what*.
- Keep behavior changes and their regression tests together; separate
  unrelated renames and refactors.
- Never skip hooks (`--no-verify`) without an explicit reason.
- Never force-push to a branch someone else may be working on.

---

## Deprecated patterns (do not use in new code)

- Avoid camelCase method and attribute names; use snake_case.
- Avoid flags that repeat type information; prefer type dispatch.
- **Mock dict-based mutation** of `module.net_dict[name] = [...]` or
  `module_dict[name] = {'device': dev}` in tests or examples. Use the
  public `add_module` / `add_net` / `ckt.add` API instead.

---

## Reviewing new code

Quick checklist before merging:

- [ ] Naming follows snake_case (methods, attributes, variables).
- [ ] New classes are `CapWords`.
- [ ] Type hints on public functions and methods.
- [ ] Docstrings describe intent, not mechanics.
- [ ] No commented-out dead code.
- [ ] Tests exist and exercise public APIs.
- [ ] Run the applicable test profiles with `.venv/bin/python`; see AGENTS.md
  for commands and requirements.
- [ ] If the change touches `silk/examples/`, the example still runs end
  to end under `MPLBACKEND=Agg`.
