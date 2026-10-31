# Silk Lint and Coverage User Guide

Silk's lint and coverage system checks analog designs against a catalog
of rules and reports coverage — ordinary rules count scope dispatches;
STR-001 counts catalog devices examined for stress.

## Quick start

```python
from silk.circuit.report import generate_report
from silk import LintReport

report = generate_report(my_module)

# Inspect
print(report.lint.summary())          # error/warning/info counts
print(report.coverage.rule_coverage.root.aggregate)

# Save artifacts
report.save_html("full_dashboard.html")        # CircuitReport HTML
report.lint.save_html("lint_only.html")        # focused lint
report.lint.save_junit_xml("junit.xml")        # for CI
report.lint.save_json("lint.json")             # structured archive
report.save_json("circuit_report.json")        # full circuit snapshot

# Diff between runs
prev = LintReport.load_json("previous_lint.json")
diff = report.lint.diff(prev)
print(f"{len(diff.added)} new violations, {len(diff.removed)} fixed")
```

Lint diff raises `ValueError` if the silk version, PDK name/version or module
name differs between the reports.

## V1 rule catalog (22 default + 1 opt-in)

Run `silk.help_rule("INT-001")` to see any rule's rationale and whether it
is in the default catalog; `silk.list_rules()` returns the default IDs and
`list_rules(include_optional=True)` every shipped ID. The rule catalog contains
22 default rules and one opt-in rule.

### Default catalog

What `RuleRegistry.default()` runs (`silk.lint.rules.DEFAULT_CATALOG`):

| ID | Name | Category | Severity | Needs sim |
|---|---|---|---|---|
| INT-001 | DRIVER_CONFLICT | interface | ERROR | no |
| INT-002 | POWER_GROUND_SHORT | interface | ERROR | no |
| INT-003 | ROLE_MISMATCH | interface | ERROR | no |
| INT-004 | NO_DRIVER | interface | ERROR | no |
| INT-005 | CLOCK_ON_ANALOG | interface | ERROR | no |
| DOM-001 | RAIL_MISMATCH | domain | ERROR | no |
| DOM-002 | DOMAIN_CROSSING | domain | ERROR | no |
| DOM-003 | SUPPLY_DISABLED_STRESS | domain | ERROR | no |
| STR-001 | JUNCTION_OVERRANGE | stress | ERROR | yes |
| STRUCT-001 | DANGLING_PORT | structural | WARNING | no |
| STRUCT-002 | POWER_PORT_UNCONNECTED | structural | ERROR | no |
| STRUCT-003 | FLOATING_NET | structural | WARNING | no |
| STRUCT-004 | FLOATING_GATE | structural | ERROR | no |
| STRUCT-005 | BODY_NOT_TIED | structural | WARNING | no |
| STRUCT-006 | MISSING_PARAM | structural | ERROR | no |
| STRUCT-007 | SIGNAL_ON_SUPPLY_NET | structural | ERROR | no |
| STRUCT-008 | UNUSED_INSTANCE | structural | WARNING | no |
| STRUCT-011 | SINGLE_CONNECTION_NET | structural | WARNING | no |
| STRUCT-012 | UNPORTABLE_GROUND_PORT | structural | ERROR | no |
| STRUCT-013 | HIDDEN_SUPPLY_NET | structural | ERROR | no |
| STRUCT-010 | PORT_DIRECTION_SANITY | structural | WARNING | no |
| BUS-002 | BUS_PARTIAL_WIRING | structural | WARNING | no |

STR-001 requires simulation data; every rated maximum comes from the PDK
catalog's `junction_ratings`, the rule adds no margin of its own. Each catalog
value is the foundry's stated maximum operating voltage, with its source (or
the heuristic rule it was derived by) recorded beside it — see "Junction
ratings" in `docs/adding-a-pdk.md`.

It also requires the *right kind* of simulation data. STR-001 scores terminal
voltages against ratings in volts, so it runs on behavioral, transient and
`.dc`-sweep results and **declines** AC and noise results, whose "voltages"
are small-signal transfer ratios and V/&radic;Hz densities rather than node
voltages. A module carrying only AC results records STR-001 as *unchecked* in
the coverage scorecard rather than as a clean pass — a stress report that
measured nothing must not read as one that passed. The decision is made from
the rawfile plot actually loaded, not from the analyses the testbench
declared, so a testbench that declares both a transient and an AC sweep is
still checked against its transient data.

Inspect `report.lint.stress_coverage`: missing ratings or terminal waveforms
leave devices partially checked or unchecked; behavioral stress cannot inspect
nested device terminals.

Findings name the worst-case point on whatever axis the run swept:
`at t=4.04ns` for a transient, `at v(v-sweep)=3.000` for a DC sweep.

STRUCT-003 and STRUCT-011 split the "this net is not a circuit" case between
them: STRUCT-003 fires on a net with ZERO connections, STRUCT-011 on a net
with exactly ONE. The one-pin shape is what a net-name typo produces under
the default `strict_nets=False` mode, and what an undriven DUT input or a
lone transistor gate looks like. A net that is a declared port of the module
being checked is exempt from STRUCT-011 — the parent supplies its second
connection. Both rules are silenced per net by
`Module.add_no_connect(net_name, reason=...)`.

STRUCT-012 fires on a module that **declares** a port named `gnd` (any case).
Formal port `0` is rejected at declaration; STRUCT-012 and setup/emission
guards also catch restored or mutated interfaces containing it.
ngspice-42 folds that node name onto the global SPICE node `0`; Xyce 7.10 folds nothing, at any spelling. Node `0` is global and pierces
subcircuit boundaries, so on ngspice the fold discards whatever the parent
wired to that port position — `.subckt blk a gnd` instantiated as
`Xblk top mid blk` ties the block's ground to global `0` instead of to `mid`.
A `gnd` formal port can discard the parent's wiring on ngspice; use `vss` and wire it explicitly.

The trigger is the declaration, not the wiring: a reusable block is
instantiated many times, possibly by code its author does not control, so a
check keyed on the instantiation sites present in one netlist could never
protect the block's interface. Only *ports* are reported by STRUCT-012.
Internal nets named `gnd` can silently give wrong results on Xyce, so silk
separately refuses any such net (case-insensitively) before writing a Xyce
deck. Use `0` for global ground or an explicitly wired `vss` port. ngspice
and behavioral nets are unchanged. Names `vss`, `VSS`, `agnd`, `ground`,
`GROUND` remain allowed as ports: ngspice folds none of them and
they are legitimately distinct
nets. Fix by renaming the port, e.g. to `vss`.

The interface guard also runs during static simulation setup and netlist
emission: `Module.generate_netlist()` (and therefore
`simulate()`) raises `CircuitError` with the same message, before any backend
is consulted and before any deck is written. That guard is not waivable —
the lint rule is, through the ordinary `rule_id` + port-location waiver.

STRUCT-013 fires when a subcircuit wires MOSFET source and body terminals to
an internal net it never declares as a port — a supply rail the author forgot
to export. The block then simulates dead with no diagnostic: a sky130
inverter whose body wires the PMOS to `vdd` while the port block declares
only `vin`/`vout`/`gnd` returns `v(vout) = 0.0` from ngspice-42, no error.

The trigger is structural, never a net's *name* — silk never infers a role
from what a net is called. A net is reported only when **every** pin on it is
a MOSFET source or body terminal, because that is the one shape with no other
reading: a gate, a drain, a resistor end, a submodule port or a source
terminal all give the net a job other than "rail". Declared ports are skipped
(the parent wires them) and so is the global SPICE node `0`. Fix by declaring
the net as a supply port and wiring it at the parent level.

A block that genuinely has no supply of one polarity — a PMOS-only current
bank, a diode-connected bias reference, a resistor ladder, a cap-DAC switched
between references — already has every device terminal on a declared port, so
the rule is silent and needs no marker: that is what "supply-less by design"
means here. Boundaries: a net with one pin is STRUCT-011's finding and one
with none is STRUCT-003's, so neither is repeated; a hidden rail that also
carries a decoupling cap or a back-annotated parasitic is *not* reported,
because the extra terminal removes the certainty (a cap-DAC bottom plate is
exactly a cap plate plus two NMOS sources). Suppress a finding with a waiver
on `STRUCT-013` and a net matcher — **not** with `Module.add_no_connect()`,
which declares a net intentionally *unconnected*: a hidden rail has two or
more real bindings, so that marker would be reported as stale. For the
stricter demand that the ports be *declared* regardless of wiring, opt in to
STRUCT-009 below.

STRUCT-010 warns when a POWER- or GROUND-role port is declared with a
direction other than the one a supply receiver has: explicit IN, or the bare
`PortDir.POWER` / `PortDir.GROUND`. A supply port declared OUT usually means
the direction conventions got mixed up and the hierarchical netlist will
mislead whoever reads it. The one legitimate shape is a block that
*produces* a rail from one it *receives* — a regulator, a charge pump, a
header or footer switch, a bias generator — whose output is exactly a
POWER-role OUT port (`Port.supply(direction='out')`, the LDO example's
`vout`). So the rule exempts a supply-role OUT port whenever the block also
declares a port of the **same role** that receives: POWER in licenses POWER
out, GROUND in licenses GROUND out, and a mixed-up inverter whose `vdd` was
declared OUT is still reported because its only supply input is a GROUND.
The exemption keys on `PortRole` and `PortDir` alone, never on a port's
name. A supply-role OUT port in a block with no same-role input is reported
with a message that names the missing role; INOUT is reported as before,
because a rail that both sources and sinks is rare enough that the intent
should be on record (waive it on `STRUCT-010` and the port when it is). Fix
by changing the direction to IN, or, for a rail the block genuinely emits,
by declaring the supply it is derived from as a same-role IN port.

### Opt-in rules

Implemented and tested, but **not** run by default
(`silk.lint.rules.OPTIONAL_CATALOG`):

| ID | Name | Category | Severity | Needs sim |
|---|---|---|---|---|
| STRUCT-009 | POWER_GROUND_PORT_REQUIRED | structural | ERROR | no |

STRUCT-009 requires every device-bearing subcircuit to declare POWER- and
GROUND-role ports. It keys on `PortRole`; a bare `PortDir.POWER` /
`PortDir.GROUND` carries the matching role (see the contract
below), so a block declared that way is *not* reported as unpowered. It
remains opt-in because some valid blocks intentionally omit a POWER or GROUND port.

The *defect* STRUCT-009 was written for — a block wiring devices to a supply
it never exports — is covered by default by STRUCT-013, which keys on the
wiring instead of on the declaration and is therefore silent on every one of
those blocks. Reach for STRUCT-009 when you want the interface contract
itself enforced: every block must name its rails, whether or not it happens
to wire them. Opt in with one flag:

```python
from silk.lint.registry import RuleRegistry

reg = RuleRegistry.default(include_optional=True)   # 22 default + STRUCT-009
```

or add just the one you want:

```python
from silk.lint.rules.structural import PowerGroundPortRequired

reg = RuleRegistry.default()
reg.add_rule(PowerGroundPortRequired())
```

### Role and domain contracts the rules assume

Role-keyed rules (INT-002, INT-004, STRUCT-002, STRUCT-007, STRUCT-009,
STRUCT-010) use `PortRole` for supply classification; driver and direction
checks also inspect `PortDir`. **Supply directions imply the supply
role**: a port declared with a bare `PortDir.POWER` /
`PortDir.GROUND` (or the `'power'` / `'ground'` string, or a `Port(dir=...)`
with no `role`) is given `PortRole.POWER` / `PortRole.GROUND` at
`define_module_ports` time, so the examples' idiom
`{"vdd": PortDir.POWER, "vss": PortDir.GROUND}` is visible to every one of
those rules, to `Net.classification`, to the `PortRoles` constraint export
and to the annotation-coverage gate. (The ground port is spelled `vss`, not
`gnd`: ngspice folds a port *named* `gnd` onto the global node `0` and
silently drops the parent's wiring to it while Xyce honours it, so
STRUCT-012 and the emission guard reject that declaration — see the
STRUCT-012 paragraph under [Default catalog](#default-catalog).)
`(PortDir.POWER, PortRole.POWER)`, `Port.supply(...)` and `Port.ground()`
remain equivalent spellings. An
explicit role is never overridden — `(PortDir.POWER, PortRole.ANALOG)` is
stored as written. Signal directions (`IN` / `OUT` / `INOUT`) imply nothing;
their role stays unset until declared or filled by `infer_port_roles`. The
ground net of a testbench built from direction-only blocks plus voltage
sources is therefore a supply net to INT-004 and raises no NO_DRIVER finding.

**An inferred role is a guess, a declared one a statement**. Roles
filled by `infer_port_roles` are marked (`ModulePin.role_inferred`,
`PortBinding.role_inferred`), and INT-004 reads the mark: an undriven net whose
only supply evidence is an *inferred* POWER role — the default `*vdd*` glob on
a monitor input such as `vdd_sense` — is reported like any signal net, where it
used to be silenced by the supply skip. When all role-bearing bindings have
supply roles, a *declared* POWER role keeps the skip
(a real supply is fed from outside the block), and a GROUND role keeps it
whether declared or inferred (nothing ever drives a return rail in the OUT
direction, so the rule has nothing to say about it). Calling
`infer_port_roles` before or after `add_module` reaches parent bindings when
the parent reads its graph.

**The net named `'0'` is ground**. It is the global ground node of
every simulator silk targets, so the interface rules treat it as ground by
spelling — the one net name they recognise that way. INT-002 reports a
POWER-role port wired to `'0'` as a grounded supply (`Net '0' is the
simulator ground node — supply is grounded: u1.vdd(power)`) even when no
GROUND-role port shares the net, and INT-004 never reports `'0'` as undriven.
`gnd`, `vss`, `ground` and friends are *not* recognised by spelling: ngspice
folds `gnd` to node 0 but Xyce does not, so those names are conventions that
belong in your `RoleRuleset`, not in the rules.

**A same-level primitive, or the block's own port, is the other party on a
net**. Device and passive pins carry no direction (a MOSFET drain
is bidirectional, a resistor end is either end), so INT-004 treats their
presence as "something else is on this net" and stays silent on a net
driven by a top-level MOSFET or a resistor-divider tap and read by several
submodule inputs. Presence is not a driver: INT-001 still does not
see a device shorted against a submodule output. Inside a hierarchical block,
the block's own IN / POWER / GROUND / INOUT port pin is the other party on
its pass-through net (the parent drives it through the port), so a port
fanned out to two children satisfies INT-004; the block's own OUT port is
*not*: OUT ports provide no external-drive evidence; these checks do not
prove every output is driven. The own-port binding carries the port's
declared role: INT-002 reads it, so a child's GROUND port wired
onto the net that carries the block's own POWER port (or POWER onto GROUND)
is reported as a supply short with the block's port named — `Net 'vdd'
connects POWER and GROUND ports: u1.gnd(ground), blk.vdd(power)`. INT-003 and
INT-005 do *not* compare the block's own port against its children: they ask
whether two children disagree about what a net carries, and the block's own
port is not a peer — a DIGITAL or CLOCK port fanned into ANALOG-declared
stdcell pins (the shipped SAR FSM shape) is the parent's abstraction
boundary, not a wiring error, and stays silent.

DOM-002 DOMAIN_CROSSING fires on any signal net whose ports sit in
different voltage domains. DOM-003 SUPPLY_DISABLED_STRESS only fires when
the domain being driven can actually be powered down, which you declare on
the rail: `VoltageDomain('dvdd', nominal=1.8, switchable=True)`. An
unannotated crossing therefore produces one DOM-002 finding and nothing
from DOM-003; a crossing into a switchable rail adds one DOM-003 finding
naming the live rail that would forward-bias the powered-down block.

Rule behaviour is covered by the real sky130 fixtures under `tests/lint/`.

## Waiving false positives

Create `silk_waivers.yaml` and load it via the registry:

```yaml
waivers:
  - rule_id: INT-001
    matcher:
      location_type: net
      fields:
        net_name: vout_ref
    reason: "Intentional dotted-OR for test multiplexer"
    author: jdoe
```

Load and apply:

```python
from silk.lint.registry import RuleRegistry
from silk.circuit.report import generate_report

reg = RuleRegistry.default()
reg.load_waivers("silk_waivers.yaml")

report = generate_report(my_module, registry=reg)  # applies policies and detects simulation data
```

Waivers require a non-empty `reason`. `expires` is optional but
recommended; choose a future date when setting it. Waivers without expiration tend to become permanent
technical debt.

A waiver suppresses exactly the findings whose location matches every
field listed, so it can be as narrow as one pin or one terminal pair:

```yaml
waivers:
  - rule_id: STRUCT-001            # one dangling port on one instance
    matcher:
      location_type: port
      fields: {port_name: en, module_name: u1}
    reason: "enable is driven from the pad ring"
  - rule_id: STR-001               # one terminal pair on one device
    matcher:
      location_type: terminal
      fields: {instance_path: mn, terminal: Vgs}
    reason: "gate clamped by the ESD diode not modelled in this bench"
```

`location_type` is one of `net` (`net_name`, `port_names`), `port`
(`port_name`, `module_name`), `instance` (`instance_path`), `terminal`
(`instance_path`, `terminal`), `module` (`module_name`).
Use `terminal: Vgs` in YAML; tuple-based `terminal_pair` matching requires
a Python `WaiverMatcher`, since YAML sequences load as lists.
Every waived finding still appears in `waivers_applied` and in the
compliance matrix — waivers document, they do not hide.

## Excluding vendor IP

Create `silk_lint_exclusions.yaml`:

```yaml
exclusions:
  - target:
      type: instance
      path: ldo.sram_macro
    reason: "Vendor SRAM, black box"
    owner: "SRAM_vendor_v3.1"
```

Load and query:

```python
reg.load_exclusions("silk_lint_exclusions.yaml")
print(reg.is_excluded("ldo.sram_macro"))      # ScopeExclusion(...)
print(reg.is_excluded("ldo.sram_macro.cell")) # ScopeExclusion(...) — subtree match
```

Exclusions match location `instance_path` or `module_name`; net-located
findings lack these and are not removed by subtree exclusions.
Exclusions filter matching findings; they do not change rule execution or
coverage counts.

## Declaring intentional floating nets

Use the `NoConnect` marker instead of waiving STRUCT-003 FLOATING_NET:

```python
my_module.add_net("vout_diff_test")
my_module.add_no_connect("vout_diff_test", reason="differential test tap")
```

The NoConnect attaches a `kind='lint_marker'` PortBinding to the net.
FloatingNetRule treats nets whose only pins are lint markers as
intentional and skips them. Markers are per-net: marking `probe_a`
leaves a floating `probe_b` reported.

### Declaring an intentionally unwired submodule pin

For STRUCT-001 DANGLING_PORT and BUS-002 BUS_PARTIAL_WIRING use the
per-pin `NoConnectPort` marker rather than a waiver:

```python
top.add_no_connect_port("u1", "en", reason="enable tied off at chip level")
top.add_no_connect_port("u2", "dbg[3]", reason="wired in the TB only")  # one bus bit
top.add_no_connect_port("u2", "dbg", reason="wired in the TB only")  # whole bus
```

The instance and port are validated when the marker is registered (a typo
raises with the known names listed), and the marker covers exactly that
pin on exactly that instance. `DanglingPortRule(config={"ignore_patterns":
["debug_*"]})` is the blunt alternative for PDK-mandated unused pins.

No-connect markers affect lint only; an unwired submodule port still prevents
SPICE instance emission.

### Markers in the report, and stale markers

Both marker kinds reach the report: `report.lint.no_connects` holds the
`(instance_name, net_name, NoConnect)` tuples and
`report.lint.no_connect_ports` the `NoConnectPort` objects. JSON lists
them together under `no_connects` (`kind: net` / `kind: port`) and the
HTML "Intentionally Unconnected" table has a Port column.

Remove stale markers: they can still affect lint even when their target is
connected, and have no `expires` date to make them fall out on their own. So
`generate_report` runs one pass over the module's markers and lists the
stale ones in `report.lint.stale_markers` (one `StaleMarker` per marker,
naming `net probe_a` or `port u1.en` plus what it is wired to); the count
is `summary()["stale_markers"]` and the HTML renders them as WARNING rows
under "Stale Markers". A stale marker is a hygiene finding, not a
violation: the wired target is fine, the claim on it is not. The rule:

- a `NoConnect` is stale when its net has **two or more** real
  connections. Zero is the STRUCT-003 case and exactly one the STRUCT-011
  case — both are the marker's job (a probe point, a test tap), so a
  one-pin net with a `NoConnect` is **not** stale;
- a `NoConnectPort` is stale as soon as the pin carries a net.

```python
from silk.lint_markers import find_stale_markers
for m in find_stale_markers(top):        # same pass generate_report runs
    print(m.describe())
# stale NoConnectPort on port u1.en ('enable unused'): wired to en_wired
```

## Compliance matrix

```python
matrix = report.lint.compliance_matrix()
# {'INT-001': {'violations': 3, 'waivers': 0}, 'STRUCT-001': {...}, ...}
```

Useful for tapeout audit checklists and CI dashboards.
