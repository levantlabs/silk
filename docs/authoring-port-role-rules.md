# Authoring a port-role ruleset

How to write a `port-roles.yaml` that tells silk how to infer a port's
**role** from its **name**. silk ships an opt-in default
(`silk/circuit/data/port-roles.default.yaml`); this doc is for writing your own
— a house/company convention that *replaces or extends* it.

## 1. What a ruleset is

A ruleset is an ordered list of `name-pattern -> role` rules. Patterns are data, so naming conventions can be changed without modifying silk. A
ruleset is applied by `infer_port_roles(module, ruleset)`, which fills the role
of every port whose role is still unset (§6).

Nothing infers automatically: you must pass a ruleset. The shipped default is
opt-in via `RoleRuleset.default()`; your own comes from
`RoleRuleset.from_yaml("your-rules.yaml")`.

## 2. The role vocabulary

`role:` must be one of the five `PortRole` values (`silk/circuit/ports.py`).
Unknown role strings raise `RoleRulesetError` when loaded.

| `role:` value | Meaning |
|---|---|
| `power` | A supply rail (`vdd`, `vcc`, `avdd`, `dvdd18`, …). |
| `ground` | A ground/return rail (`gnd`, `vss`, SPICE node `0`, …). |
| `clock` | A clock or strobe (`clk`, `ck`, `phi1`, …). |
| `analog` | A continuous-valued signal (`vin`, `vout`, `vbias`, `iref`, …). |
| `digital` | A logic-valued signal (`d[3]`, `en`, `sel`, …). |

Roles feed the annotation-coverage check (`require_annotation_coverage`) and,
downstream, `Net.classification`. They do **not** carry a direction — that's
`PortDir`, set separately in `define_module_ports`.

## 3. The YAML schema

```yaml
version: 1                                   # required; only 1 is defined today
rules:                                       # ordered list, most specific wins
  - {match: "*vdd*",         role: power}    # a glob rule
  - {match: "0",             role: ground}   # an exact-literal glob (no wildcards)
  - {regex: "^d\\[[0-9]+\\]$", role: digital} # a regex rule (escape hatch)
```

| Key | Required | Meaning |
|---|---|---|
| `version` | yes | Schema version. Must be `1`. Missing/other → error. |
| `rules` | no | List of rule mappings. Omitted, empty or null means no rules. |
| `match` | per-rule\* | A **glob** pattern (§4.1). |
| `regex` | per-rule\* | A **regex** pattern (§4.2). |
| `role` | yes | One of the §2 values. |

\* Each rule has **exactly one** of `match:` or `regex:` — neither or both is an
error.

## 4. How names are matched

Matching is **case-insensitive** on both sides (`*VDD*` matches `avdd`). No
bus-index stripping is done in v1 — a per-bit port `d[3]` is matched literally
(use the anchored regex below to match numbered bus bits).

### 4.1 Glob (`match:`) — the default

Standard shell globbing (`fnmatch`): `*` = any run of characters, `?` = one
character; square brackets introduce character classes, so `d[*]` matches
`d*`, not `d[3]`. Substring rules are the common case:

```yaml
- {match: "*vdd*", role: power}     # anything containing "vdd"
- {match: "avdd?", role: power}     # avdd0, avdd1, …
- {match: "vref",  role: analog}    # exactly "vref" (no wildcard = exact literal)
```

Prefer simple globs for substring matching. For bracketed bus names, prefer
the anchored regex below.

### 4.2 Regex (`regex:`) — the escape hatch

For patterns globs can't express. A full Python regex, matched
case-insensitively, searched against the name:

```yaml
- {regex: "^d\\[[0-9]+\\]$", role: digital}   # d[0], d[12], … but not "addr"
```

In a double-quoted YAML string, write a backslash as `\\`. Single-quoted
YAML strings preserve backslashes literally.

### 4.3 When several rules match — most-specific-first

If more than one rule matches a name, the **most specific** wins, decided in
this order:

1. an **exact literal** pattern (a `match:` with no `*`/`?`, or a `regex:` with
   no metacharacters) beats any wildcarded rule;
2. else the pattern with **more literal characters** (non-wildcard chars) wins;
3. else the one **declared first** in the file.

So put more specific rules earlier when they'd otherwise tie. Example — a bias
pin that also contains `vdd`:

```yaml
rules:
  - {match: "vdd_bias_ref", role: analog}   # exact -> wins for that exact name
  - {match: "*vdd*",        role: power}     # everything else vdd-ish -> power
```

The shipped default uses the declaration-order tiebreaker: `*vdd*` (power)
is before `*gnd*` (ground), so `vddgnd` resolves to power.

Specificity is a character-count heuristic, not a measure of regex meaning.
For globs, only `*` and `?` count as metacharacters for ranking; square
brackets still have their usual `fnmatch` matching behavior. For regexes,
characters in `.^$*+?{}[]\|()` are excluded from the literal count. Even a
regex without metacharacters uses search matching, not whole-name matching.

## 5. Worked example — the shipped default

The default in `silk/circuit/data/port-roles.default.yaml` recognizes common supply,
ground and clock names; analog and digital signal conventions remain yours:

```yaml
version: 1
rules:
  - {match: "*vdd*", role: power}
  - {match: "*vcc*", role: power}
  - {match: "*vss*", role: ground}
  - {match: "*gnd*", role: ground}
  - {match: "0",     role: ground}
  - {match: "*clk*", role: clock}
  - {match: "*clock*", role: clock}
```

Extend this with your own analog/digital conventions using `.add()`, or copy
the YAML and add rules to it. Loading a file replaces the default ruleset.

**Check monitor inputs.** Substring patterns also match names such as
`vdd_sense` and `gnd_sense`. An inferred supply role can affect lint checks,
net classification and annotation coverage. The no-driver check (INT-004)
considers a supply exemption only when every role-bearing binding on the net
has a POWER or GROUND role. Under that condition, an explicit POWER role or
a GROUND role (inferred or explicit) exempts the net; inferred POWER roles
alone do not. A net with ANALOG and GROUND child inputs and no driver or
other connected party receives no supply exemption and can trigger INT-004.
Thus `gnd_sense` inferred as GROUND can hide a missing driver. Add an exact
signal rule such as `{match: "gnd_sense", role: analog}`, or narrow the supply
patterns. Role inference annotates ports; it does not connect a ground-named
port to SPICE node `0`.

## 6. Loading and using a ruleset

```python
from silk import Module, PortDir, PortRole, RoleRuleset, infer_port_roles

module = Module("TEST", "/tmp/silk-port-roles", "block.cir", pdk_details=None)
module.define_module_name("block")
module.define_module_ports({"vdd": PortDir.IN, "ibias": PortDir.IN,
                            "data": PortDir.OUT})
rs = RoleRuleset.default().add("*bias*", PortRole.ANALOG)
report = infer_port_roles(module, rs)
assert report.filled == [("vdd", PortRole.POWER), ("ibias", PortRole.ANALOG)]
assert report.unmatched == ["data"]
assert rs.classify("VDD") == PortRole.POWER
assert rs.classify("unknown") is None
```

To load your own file instead, use `rs = RoleRuleset.from_yaml("rules.yaml")`.
`report.filled` lists the names and roles set by this call;
`report.unmatched` lists ports whose roles remain unset. These still count as
unannotated when checking annotation coverage.

`RoleRuleset.from_yaml` and `.default()` return an independent ruleset — you can
`.add()` house rules onto `.default()` without mutating anything shared.
Extending a ruleset does not revise earlier inferred roles: inference skips
every port whose role is already set.

### Inference and placement

Run `infer_port_roles` after declaring the module's ports. Placed modules are
live references, so roles inferred before or after placement reach the parent
when it reads its graph. Existing wiring remains instance-local. Inferred
roles retain their provenance; explicit roles are never overwritten.
Inference visits only this module's own ports; apply it to each required
definition in a hierarchy.

## 7. What fails loudly

`RoleRulesetError` is a `ValueError` subclass, available from `silk.circuit.port_roles`.
Loading raises it for unknown role strings, missing roles, neither or both of
`match:`/`regex:`, missing or unsupported versions, invalid regexes, a top-level
value that is not a mapping, nonempty `rules` values that are not lists, or
rules that are not mappings. Use `version: 1` and `rules: []` for an empty file.

YAML syntax errors propagate as `yaml.YAMLError`; file-access errors propagate
as `OSError` subclasses, including `FileNotFoundError`. These are separate
from ruleset schema validation errors.

## 8. Checking a ruleset (CLI)

From the repository root, run the checker with its virtual environment.
For an installed package, use the Python interpreter where silk is installed:

```
.venv/bin/python -m silk.circuit.rolecheck [--file rules.yaml] [--names NAME ...]
```

- **Validate** (default): loads the ruleset (silk's shipped default if no
  `--file`) and reports `OK: <label>: N rule(s)`, or the specific error. This is
  exactly the §7 loud validation, wrapped with an exit code.
- **Dry-run** (`--names`): also classifies each name, so you can confirm the
  rules do what you intend before wiring them into a build.

```console
$ .venv/bin/python -m silk.circuit.rolecheck --names vdd avss clk vinp d3
OK: default (port-roles.default.yaml): 7 rule(s)
  vdd              -> power
  avss             -> ground
  clk              -> clock
  vinp             -> (unmatched)
  d3               -> (unmatched)
```

**Exit codes** make it a CI / pre-commit gate: `0` = valid, `1` = a caught
`RoleRulesetError`, `2` = file not found (or invalid CLI arguments). Other
file-access failures and YAML syntax errors are not caught by the CLI and
print a traceback.

Equivalent in Python: `RoleRuleset.from_yaml("acme-roles.yaml")` — a successful return means
the file loaded.
