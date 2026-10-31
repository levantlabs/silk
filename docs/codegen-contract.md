# Codegen JSON Contract

This document defines the JSON payload any tool can produce for `silk.codegen.Generator` to emit a runnable Python module definition.

Start with a zero-pin reusable module. Generation and construction need no
simulator or installed model libraries:

```python
from silk.codegen import Generator
payload_dict = {"circuit_name": "my_circuit", "pdk": "sky130", "ports": {}}
gen = Generator()
code = gen.generate(payload_dict)   # returns Python source as a string
gen.generate_to_file(payload_dict, 'my_circuit.py')
```

The field examples below are fragments to merge into a circuit payload, unless
shown as a complete example. Python emission snippets show the corresponding
generated lines; they assume the surrounding module or testbench setup.

---

## Top-level structure

```json
{
  "circuit_name":    "diff_amp",
  "pdk":             "sky130",
  "output_dir":      "./subckt",
  "voltage_domains": { ... },
  "ports":           { ... },
  "bus_nets":        [ ... ],
  "variables":       [ ... ],
  "devices":         [ ... ],
  "ideal_elements":  [ ... ],
  "sources":         [ ... ],
  "simulation":      { ... }
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `circuit_name` | string | no | `"my_circuit"` | Used as module name and output filename. **Must be a valid Python identifier and not a Python keyword** (`my_amp`, not `my amp` / `diff-amp` / `class`); it is an identity (SPICE `.subckt` name, tracking-DB structure identity, `module_name=` literal, `<name>.cir`). Anything else raises `ValueError` naming the field before any code is produced. |
| `pdk` | string | no | `"sky130"` | PDK identifier — see [PDK values](#pdk-values) |
| `output_dir` | string | no | `"./sim"` | Output directory the generated `Module(...)` writes to |
| `corner` | string | no | `"tt"` | Process corner name from the selected PDK's manifest (sky130: `"tt"` / `"ff"` / `"ss"` / `"sf"` / `"fs"` and composed forms such as `"ff_hh_mm"`; gf180mcu and sg13g2: `"typical"` / `"ss"` / `"ff"`). Standalone mode emits `ckt.simulate(corner=...)`; subcircuit mode does not use it. At generation time only the corner’s **syntax** is checked — `[A-Za-z0-9_-]+`, else `ValueError` naming the field — so codegen works without a resolvable PDK install; membership in the manifest is enforced by `Module.simulate`, which rejects an unknown corner and lists the available ones. |
| `voltage_domains` | object | no | `{}` | Named supply rails — see [voltage_domains](#voltage_domains) |
| `ports` | object or null | no | absent | Port map (dict form). Three cases: key absent or `null` → standalone simulation script; `{}` → zero-pin `Module` subcircuit (`mod.define_module_ports({})`); non-empty → `Module` subcircuit with those pins |
| `bus_nets` | array | no | `[]` | Internal bus net declarations — see [bus_nets](#bus_nets) |
| `variables` | array | no | `[]` | Module variables shared by numeric parameters — see [variables](#variables--module-variables-shared-by-parameters) |
| `devices` | array | no | `[]` | PDK transistors, resistors, caps, diodes, BJTs |
| `ideal_elements` | array | no | `[]` | Process-independent `resistor`/`capacitor`/`inductor` via `ElementLibrary`. Sources belong in `sources[]`. |
| `sources` | array | no | `[]` | The single home for every V/I source: `VoltageSource`/`CurrentSource` with optional waveforms, the VCVS/VCCS/CCVS/CCCS dependent sources, and `BehavioralVoltage`/`BehavioralCurrent` |
| `simulation` | object | no | `{}` | Analysis request for standalone scripts; omission emits no analysis call |
| `submodules` | array | no | `[]` | Hierarchical child instances — see [submodules](#submodules) |

**Module vs standalone** (three cases): if `ports` is non-empty, the generator produces a `Module` subcircuit definition (`mod.define_module_ports(...)`). If `ports` is present but `{}`, it produces the same module form with no pins (`mod.define_module_ports({})`) — a pin-less block, like `define_module_ports({})` in the Python API. When the `ports` key is **absent** or `null`, it produces a full standalone simulation script (the Module variable is named `ckt` instead of `mod`). Sending `ports: {}` for a testbench is therefore wrong: omit the key. Both modes consume `submodules` and `bus_nets`; the standalone mode is the right shape for testbench-style top schematics with no I/O ports but multiple child instances.

---

## PDK values

| Value | PDK |
|---|---|
| `"sky130"` | SkyWater 130 nm; release-validated transistor simulation |
| `"gf180mcu"` | GF180MCU; preview catalog integration |
| `"sg13g2"` | SG13G2; preview catalog integration |

The generator accepts registered PDK names. Catalog loading and code generation
do not require installed model libraries; transistor simulation does. The default
`corner` remains `"tt"`, so standalone GF180MCU and SG13G2 scripts should
explicitly set `"corner": "typical"`.

---

## `voltage_domains`

Named supply rails, defined once and referenced by name from `ports`. In subcircuit mode, the generator emits one `VoltageDomain(name, ...)` line per entry; ports declaring `domain: "<name>"` attach to that instance.

```json
"voltage_domains": {
  "AVDD": {"nominal": 1.8, "tolerance": 0.1},
  "DVDD": {"nominal": 0.9, "tolerance": 0.1},
  "IO":   {"nominal": 3.3, "min": 3.0, "max": 3.6}
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `nominal` | number | yes | — | Nominal supply voltage in V |
| `tolerance` | number | no | `null` | Symmetric fractional tolerance (e.g. `0.1` → ±10 %) |
| `min` | number | no | `null` | Explicit lower bound (overrides tolerance) |
| `max` | number | no | `null` | Explicit upper bound (overrides tolerance) |
| `switchable` | boolean | no | `false` | Rail can be powered down while others stay live; the only input lint DOM-003 SUPPLY_DISABLED_STRESS keys on |

**Domain names** must be valid Python identifiers, excluding Python keywords, punctuation and spaces. The authored string remains the domain identity. Its generated Python variable may differ: for example, `Module` uses `domain_Module`, with a numeric suffix if needed to avoid other bindings. Ports use the same shared domain object automatically; callers do not supply an alias. Python identifier normalization also counts as a binding collision; distinct authored strings such as `K` and `K` retain distinct domain objects. Ordinary nonconflicting names retain their existing output.

Generator-local domain names previously refused — `pdk`, `mod`, `ckt`, `os`, `analysis`, `sources`, `elements`, `_pdk_root`, `_lib_path` — are now accepted with safe bindings, as are names colliding with imports such as `Module` or `VoltageDomain`. The generator avoids domain bindings, authored variables, submodule import aliases and per-copy wiring locals. A domain name that conflicts with an authored variable is still refused. Both JSON input and direct `CircuitSpec` objects are checked before writing output; direct objects use the domain object's actual `name`, even when its mapping key differs. Standalone mode emits no domain bindings; unused domain metadata in a direct standalone spec leaves its output unchanged.

The `tolerance` / `min` / `max` fields drive silk's stress checks (`check_voltage_domains`, `check_junction_overrange`); `switchable: true` is emitted as `switchable=True`. Generated output omits any optional kwarg that isn't set on the input entry.

---

## `ports`

Maps each port net name to a dict describing direction and optional role / domain / bus width. `dir` is the only required key; the rest are optional. Used to call `mod.define_module_ports()`.

```json
"ports": {
  "clk":   {"dir": "input"},
  "vh":    {"dir": "output", "role": "analog",  "domain": "AVDD"},
  "vdd":   {"dir": "input",  "role": "power",   "domain": "AVDD"},
  "code":  {"dir": "input",  "role": "digital", "bits": 4},
  "lanes": {"dir": "input",  "role": "analog",  "bits": [8, 12]},
  "dbg":   {"dir": "output", "role": "digital", "bits": 4,
            "naming": "underscore"}
}
```

### Port fields

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `dir` | string | yes | — | One of `"input"` / `"output"` / `"inout"` / `"power"` / `"ground"` |
| `role` | string | no | `null` | One of `"analog"` / `"digital"` / `"clock"` / `"power"` / `"ground"` |
| `domain` | string | no | `null` | Must reference a key in `voltage_domains` |
| `bits` | int or list | no | `null` (scalar) | Bus width — int `N` → bits `0..N-1`; list `[a, b]` → bits `a..b-1` |
| `naming` | string | no | `"bracketed"` | `"bracketed"` (`d[3]`) or `"underscore"` (`d_3`) |

### Emission

The generator picks the **shortest legal Python form** per port:

- Only `dir` set → bare `PortDir.IN` / `PortDir.OUT` / `PortDir.INOUT` / `PortDir.POWER` / `PortDir.GROUND`.
- `dir` + `role` only → `(PortDir.IN, PortRole.ANALOG)` 2-tuple.
- Any of `domain`, `bits`, or non-default `naming` set → full `Port(dir=..., role=..., domain=..., bits=..., naming=...)` constructor, with only the fields actually set emitted as kwargs.

Imports for `PortRole` and `Port` are conditional on the spec needing them; `PortDir` is always imported in subcircuit mode.

### Notes

- The string-only form (`"vin": "input"`) raises `TypeError`. Always use the dict form.
- `null` for `role`, `domain`, `bits`, or `naming` is treated the same as omitting the key (falls back to the dataclass default).
- Unknown `dir` or `role` strings raise `ValueError`. A `domain` string not present in `voltage_domains` raises `ValueError` naming both the port and the missing domain.
- **Never name a port `gnd`** (any case) — name the ground port `vss`, as the examples below do. ngspice folds the port *name* `gnd` onto the global SPICE node `0`, silently discarding whatever the parent wired to that port position, while Xyce honours it: two different circuits, no warning from either. silk rejects the declaration — `STRUCT-012` at lint time and a backend-independent guard in `Module.generate_netlist`, so a spec declaring a `gnd` port produces code that raises `CircuitError` instead of a netlist. This port restriction applies to every backend. Xyce additionally rejects any net named `gnd` (any case), including internal nets, before writing a deck: it can otherwise silently produce wrong results. Use `0` for global ground or an explicitly wired `vss` return port. `gnd!` is refused by the shared name validator; use `0` for ground, or a `Port.ground()` port wired to `0` at the bench.

---

## `bus_nets`

Internal bus net declarations. Most internal nets self-create when device pins reference them, so `bus_nets` is **only** needed when an internal net must be a bus (otherwise individual bits auto-create as scalars and the bus aggregator can't form). Bus *ports* declared in the `ports` section do not need a corresponding `bus_nets` entry.

```json
"bus_nets": [
  {"name": "count", "bits": 4},
  {"name": "addr",  "bits": [4, 12], "naming": "underscore"}
]
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `name` | string | yes | — | Logical bus name; per-bit nets are derived from `naming` |
| `bits` | int or list | yes | — | Width: int `N` → bits `0..N-1`; list `[a, b]` → bits `a..b-1` |
| `naming` | string | no | `"bracketed"` | `"bracketed"` (`count[3]`) or `"underscore"` (`count_3`) |

Each entry emits one `<parent_var>.add_net(name, bits=..., naming=...)` line. The block is emitted **after** the Module is constructed (and after `define_module_ports` in subcircuit mode) and **before** any device or submodule, so downstream code finds the bus aggregator already registered. `<parent_var>` is `mod` in subcircuit mode and `ckt` in standalone mode; both output modes consume `bus_nets`.

---

## `submodules`

Each entry instantiates a child Module inside the parent. silk emits
one `from <import_path> import <module_var> as <alias>` line per unique
`(import_path, module_var)` pair plus one `<parent_var>.add_module(...)`
call per entry — except an entry that carries `parameters`, which still
emits `<parent_var>.add_element(..., parameters=...)` (see "Parameters:
explicit dict, never `**kwargs`" below). `<parent_var>` is `mod` in
subcircuit mode (`ports` present, including `{}`) or `ckt` in standalone mode (`ports`
absent) — both output modes consume `submodules`.

### Submodule fields

| Key | Type | Required | Default | Notes |
|---|---|---|---|---|
| `instance_name` | string | yes | — | Must be a valid Python identifier and not a keyword. Unique across all entries. |
| `module_ref.import_path` | string | yes | — | Emitted verbatim in the `from … import …` line. |
| `module_ref.module_var` | string | yes | — | Variable name inside the imported file (silk-emitted children use `mod`). |
| `nets` | object or list | yes | — | `{child_port: parent_net_name}`, where a per-copy child pin may be a port name or one bit (`"d[2]"`), or one dict per array copy — see [per-copy nets lists](#per-copy-nets-lists). Values may be scalar names, bus-net names (whole-bus form), or expanded bits (e.g. `"data[3]"`). |
| `parameters` | object | no | `null` | Per-instance parameter overrides — passed as `parameters={...}` to `add_element` (not `add_module`, which has no `parameters` argument; see below). Use this, **not** loose kwargs. |
| `array` | int or list | no | `null` | If set, this entry is an array. Int N → `range(N)`; list `[a, b]` → `range(a, b)`. Net values may contain `$i` / `$(i±N)` templates — see [Index templates](#index-templates-i--in). |
| `naming` | string | no | `"bracketed"` | When `array` is set: `"bracketed"` emits `U[0]`, `U[1]`, ...; `"underscore"` emits `U_0`, `U_1`, .... Ignored if `array` is absent. |

### Whole-bus wiring

When a value in `nets` matches a name declared in the parent's
`bus_nets` or a bus-typed port, silk emits a Net handle
(`<parent_var>.nets.<name>`, i.e. `mod.nets.<name>` in subcircuit mode
or `ckt.nets.<name>` in standalone mode) so silk runtime takes the
bus-expansion branch. Python keywords and non-identifiers cannot follow a dot,
so only those names use bracket lookup with a repr key (`mod.nets['in']`,
for example). Soft keywords keep dot lookup. Plain net names emit as strings.
Expanded bits
(`"data[0]"`) also emit as strings.

### Alias rule (silk-side)

The spec never specifies the import alias. silk computes it as
`<module_var>_<sanitized_import_path>`, sanitizes `.`/`/`/`-` to
underscores, strips leading dots, and appends a numeric suffix when
the result collides with a generator-reserved name (`mod`, `ckt`,
`pdk`, …) or with another alias already chosen for this spec.

### Parameters: explicit dict, never `**kwargs`

`add_module`'s signature is `add_module(module, name, nets=None,
**ports)` — there is no `parameters` argument, and `**ports` would
silently swallow a `parameters` override as a port name. So an entry
that carries `parameters` is emitted against `add_element` instead,
whose signature is `add_element(elemDevice, elemName, nets=None,
parameters=None, **port_kwargs)`. silk always emits
`parameters={...}` there, never as a loose kwarg. That form raises at
run time for a Module child; see [known issues](known_issues.md).
Entries without `parameters` always emit `add_module`.

---

## Hierarchical generation gotchas

Importing a generated child file constructs its `mod` object and creates
its `output_dir` if needed. It does not write a `.cir` file. Calling
`mod.generate_netlist()` on that subcircuit returns SPICE text without writing
it to disk; the parent writes the complete deck when it generates a top-level
netlist. Use distinct circuit names for distinct child definitions so their
SPICE subcircuit names remain unambiguous.

---

## Per-copy nets lists

On an arrayed `submodules`, `devices`, or `ideal_elements` entry, `nets`
may be a list of dicts instead of a single dict. This includes devices,
ideal elements and helper submodules in both rigs and tests. The list supplies
one dict per copy in ascending index order. For example:

```json
{
  "instance_name": "amp1",
  "module_ref": {"import_path": ".amp", "module_var": "amp"},
  "array": 3,
  "nets": [
    {"inp": "n3", "vb": "vbias", "out": "o[0]"},
    {"inp": "n0", "vb": "vbias", "out": "o[1]"},
    {"inp": "clk", "vb": "vbias", "out": "o[2]"}
  ]
}
```

The generator emits one list named `__silk_amp1_nets` and one `for` loop
that selects the dict for each copy. Names remain `amp1[0]`, `amp1[1]`, and
`amp1[2]` by default; `naming: "underscore"` remains available. With
`array: [4, 7]`, the three dicts connect copies `amp1[4]` through `amp1[6]`.

Submodule per-copy pin keys may be identifier-shaped port names or one bit
of a bus port, `name[k]`, where `k` is a non-negative decimal integer.
For example, three copies may connect `"d[2]"` to `"p"`, `"q"`, and `"r"`.
This also applies to rig/test helper submodules. Devices and ideal elements
require plain identifier pin keys because they become Python keyword arguments.
Malformed bit keys and slices (`"d[]"`, `"d[-1]"`, `"d[1:0]"`, `"d [2]"`)
are refused, and slice net values remain refused.

A list requires `array` and exactly as many dicts as copies. All dicts must
have the same pin set, and each value must be a plain net name, including
expanded bus bits or `"0"`. Index templates (`$i` or `$(i±N)`) are refused
inside the list. Wrong item/value types raise `TypeError`; wrong counts,
pin sets or plain net names raise `ValueError`, naming the entry.
Rig/test generation
wraps malformed entries in `TestbenchSpecError` with the group and entry name.
Its net-name inventory includes every copy when checking derived-signal names.
The single-dict form keeps its existing broadcast and template behavior.

## Index templates (`$i` / `$(i±N)`)

A string-valued net in a single `nets` dict (or scalar source field) on an arrayed entry may
contain placeholders that expand per-instance:

| Token | Expands to |
|---|---|
| `$i` | The loop variable. |
| `$(i)` | Same as `$i` (explicit parens). |
| `$(i+N)` | `i + N` (positive integer offset). |
| `$(i-N)` | `i - N` (positive integer offset). |

Strings with NO placeholder are *broadcast*: the same net is wired to
that port on every instance in the array. Multiple placeholders in one
string are allowed (`'tap[$i]_tap[$(i+1)]'`).

silk emits the rewritten string as a Python f-string inside the
generated array expression (a list comprehension for devices/elements/sources,
a one-line `for i in range(...)` loop for submodules). Result strings like `tap[-1]` (when `i=0` and the
template is `$(i-1)`) are emitted verbatim. Generation does not validate the
expanded net names; normal runtime name checks still apply.

---

## `variables` — module variables shared by parameters

A variable is one value several parameters share: change it and every
parameter that references it changes on the next netlist. Matching two
devices is a shared variable; a 2:1 ratio is one device using a scaled
reference. `CircuitSpec` only — a `TestbenchSpec` declares no
variables, and a reference in a rig or test raises `TestbenchSpecError`.

```json
{
  "variables": [
    {"name": "w_in", "value": 2.0}
  ],
  "devices": [
    {"type": "nmos1p8", "instance": "m1",
     "nets": {"d": "a", "g": "inp", "s": "tail", "b": "0"},
     "params": {"w": {"variable": "w_in"}, "l": 0.15}},
    {"type": "nmos1p8", "instance": "m2",
     "nets": {"d": "b", "g": "inn", "s": "tail", "b": "0"},
     "params": {"w": {"variable": "w_in", "scale": 2.0}, "l": 0.15}}
  ]
}
```

### Declaration fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | yes | A Python identifier, not a keyword, unique. It may not be a name the generated code binds or reads (`pdk`, `mod`, `ckt`, `os`, `analysis`, `sources`, `elements`, `_pdk_root`, `_lib_path`, `Module`, `Variable`, `Port`, `PortDir`, `PortRole`, `VoltageDomain`, `Net`, `load_pdk`, the array index `i`, `range`, `slice`), a `voltage_domains` name, a computed submodule import alias, or a generated per-copy wiring-list name. |
| `value` | number | yes | A finite int or float (not a boolean, not an SI string). |

No other key is accepted: bounds and step are not part of the contract yet
and a declaration carrying them raises `ValueError`, so a later additive key
is never silently ignored.

### References

Any numeric parameter slot may hold a reference object instead of a number:

| Form | Emits |
|---|---|
| `{"variable": "w_in"}` | `w=w_in` |
| `{"variable": "w_in", "scale": 2}` | `w=2.0 * w_in` |

The scale is a finite int or float, emitted first as a float literal and
never simplified (`"scale": 1` still emits `1.0 * w_in`). There is no offset
and no string form: a string is an ordinary literal, never parsed as an
expression.

Slots that take a reference: every `devices[].params` and
`ideal_elements[].params` value, a source's `dc` and `ac`, electrical waveform parameters (`params`
form or legacy flat form), every `pwl` entry, and a dependent source's
`gain`/`gm`/`transresistance`. A reference on an arrayed entry applies to
every element of the array. Bit sequences, `RandomSource.type`,
`BitPattern.repeat` and PRBS `order`/`length`/`seed` require literals.

### Emission

The generated file imports `from silk import Variable` and declares
the variables in one block, in declaration order, after the PDK setup (and
the voltage domains in module mode) and before the `Module` is created:

```python
# --- module variables ---
w_in = Variable('w_in', 2.0)
```

The value is emitted with Python `repr` (an int stays an int). A spec
without `variables` emits no variable declarations.

`mod.variables()` returns original Variable objects once by identity, including
unset Variables and expression leaves. Distinct same-name Variables remain
distinct. For each module it visits devices, sources (including VCVS gain,
VCCS gm, CCVS transresistance and CCCS gain), `passives_dict` in insertion
order, attached behavioral params in key order, then placed children depth
first in placement order. A shared Variable takes its first occurrence.
Copying and placing a module preserves the shared Variable objects; embedded
testbenches include their placed DUT.

Discovery includes stored waveform parameters, parameter containers, and
statistical parameters (`Normal.nominal`/`sigma`, `Uniform.low`/`high`). It
searches dictionary keys and values, lists, tuples, sets, frozensets,
object-bearing NumPy arrays and object-bearing structured NumPy scalars.
It does not resolve values, draw samples, execute models or read derived
properties. Cyclic containers terminate; set and frozenset iteration order is
undefined. Arbitrary object attributes and callable internals are not searched.
Nested statistical declarations in behavioral parameters are refused at
attachment and run time, even though variable discovery can traverse them.
Discovery excludes deprecated `add_submodule` registrations, device parameters
stored in `add_element`'s legacy `param_list`, and legacy `PassiveBase.value`
storage. Use the spec's declaration order when displaying variables to a user.

### Errors

All raise `ValueError` at generate time and name the variable and, for a
reference, the entry and parameter:

- a reference to an undeclared variable —
  `device 'm2' param 'w' references undefined variable 'w_inn' (declared: [...])`;
  waveform and PWL slots read `param 'waveform v2'` / `param 'pwl[3]'`;
- a reference object with a key other than `variable`/`scale`, or without
  `variable`;
- a duplicate, non-identifier, keyword or clashing name;
- a non-numeric, boolean or non-finite value or scale;
- a declared variable that no parameter references. This is an error, not
  a warning: the declaration would be dead code that no device uses, and it
  usually means the spec lost a reference. Remove the declaration.

---

## `devices` — PDK devices

Each entry instantiates one device from the PDK.

```json
{
  "type":     "nmos1p8",
  "instance": "mn1",
  "nets":     {"d": "vout", "g": "vin", "s": "0", "b": "0"},
  "params":   {"w": 4.0, "l": 0.15, "m": 2}
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | Device type key — see [sky130 device types](#sky130-device-types) |
| `instance` | string | yes | Instance name passed to the factory (SPICE emission adds its device prefix) |
| `nets` | object or list | no (`{}`) | `{pin: net}` — pin names depend on device type. Arrays also accept [per-copy nets lists](#per-copy-nets-lists). |
| `params` | object | no | `{param: value}` — parameters depend on device type |
| `array` | int or list | no | If set, this entry is an array. Int N → `range(N)`; list `[a, b]` → `range(a, b)`. Net values may contain `$i` / `$(i±N)` templates — see [Index templates](#index-templates-i--in). |
| `naming` | string | no | When `array` is set: `"bracketed"` emits `U[0]`, `U[1]`, ...; `"underscore"` emits `U_0`, `U_1`, .... Affects emitted names only for arrays; a supplied value must still be `"bracketed"` or `"underscore"`. |

### sky130 device types

The tables below list the shipped sky130 catalog. For device-specific
parameter defaults and limits, export the live schema without loading models:

```python
from silk import load_pdk
from silk.elements.schema import export_schema
schema = export_schema(load_pdk('sky130'))
print(schema['nmos1p8']['pins'])
```

W/L values use the PDK's native micrometres, not metres. Other units follow
the schema's `unit` fields.

#### MOSFET (pins: `d`, `g`, `s`, `b` — params: `w` µm, `l` µm, `m`; `nmos20v_iso` additionally requires pin `sub`)

| Type | Description |
|---|---|
| `nmos1p8` | NMOS 1.8 V standard |
| `nmos1p8_lvt` | NMOS 1.8 V low-Vt |
| `nmos3p3_native` | NMOS 3.3 V native |
| `nmos5v` | NMOS 5 V |
| `nmos5v_native` | NMOS 5 V native |
| `nmos20v` | NMOS 20 V |
| `nmos20v_native` | NMOS 20 V native |
| `nmos20v_zvt` | NMOS 20 V zero-Vt |
| `nmos20v_iso` | NMOS 20 V isolated |
| `nmos_esd` | NMOS ESD protection |
| `pmos1p8` | PMOS 1.8 V standard |
| `pmos1p8_lvt` | PMOS 1.8 V low-Vt |
| `pmos1p8_hvt` | PMOS 1.8 V high-Vt |
| `pmos5v` | PMOS 5 V |
| `pmos20v` | PMOS 20 V |
| `pmos_esd` | PMOS ESD protection |

#### Resistors (pins: `a`, `b`, `c` — params: `w` µm, `l` µm, `m`)

| Type | Description |
|---|---|
| `res_hpo` | High-precision poly resistor |
| `res_xhpo` | Extra-high-precision poly resistor |
| `res_generic_nd` | Generic n-diffusion resistor |
| `res_generic_pd` | Generic p-diffusion resistor |
| `res_iso_pw` | Isolated p-well resistor |

#### Capacitors

MiM capacitors use pins `plus`, `minus` and parameters `w` µm, `l` µm, `m`.
`cap_vpp_11p5x11p7` uses pins `c0`, `c1`, `b` and parameter `m`.

| Type | Description |
|---|---|
| `cap_mim1` | MIM capacitor (type 1) |
| `cap_mim2` | MIM capacitor (type 2) |
| `cap_vpp_11p5x11p7` | VPP capacitor |

#### Varactors (pins: `plus`, `minus`, `bulk` — params: `w` µm, `l` µm, `m`)

| Type | Description |
|---|---|
| `varactor_nmos` | NMOS varactor |
| `varactor_pmos` | PMOS varactor |

#### Diodes (pins: `a`, `k` — params: `m`, `area` µm², `perim` µm)

| Type | Description |
|---|---|
| `diode_std` | Standard diode |
| `diode_pwnd` | P-well/n-diff diode |
| `diode_nw` | N-well diode |

#### BJTs (pins: `c`, `b`, `e` — NPN also has `s`; parameter: `m`)

| Type | Description |
|---|---|
| `npn_std` | NPN standard |
| `npn_11v` | NPN 11 V |
| `pnp_small` | PNP small |
| `pnp_large` | PNP large |

---

## `ideal_elements` — process-independent elements

`ideal_elements[]` is the home for process-independent **passives** —
`resistor`/`capacitor`/`inductor` — via `ElementLibrary`. Every V/I source,
including a plain DC or AC source, belongs in [`sources[]`](#sources--the-single-home-for-every-source)
below. Source entries in `ideal_elements[]` raise `ValueError`, naming the
entry and directing the caller to `sources[]`.

Each entry creates an element via `ElementLibrary`.

```json
{
  "type":     "resistor",
  "instance": "r_load",
  "nets":     {"p": "vout", "n": "0"},
  "params":   {"value": 10000}
}
```

### ideal_elements fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | Element type key — see table below |
| `instance` | string | yes | Instance name passed to the factory (SPICE emission adds its device prefix) |
| `nets` | object or list | no (`{}`) | `{pin: net}` — pin names depend on element type. Arrays also accept [per-copy nets lists](#per-copy-nets-lists). |
| `params` | object | no | `{param: value}` — parameters depend on element type |
| `array` | int or list | no | If set, this entry is an array. Int N → `range(N)`; list `[a, b]` → `range(a, b)`. Net values may contain `$i` / `$(i±N)` templates — see [Index templates](#index-templates-i--in). |
| `naming` | string | no | When `array` is set: `"bracketed"` emits `U[0]`, `U[1]`, ...; `"underscore"` emits `U_0`, `U_1`, .... Affects emitted names only for arrays; a supplied value must still be `"bracketed"` or `"underscore"`. |

### Element types

| Type | Pins | Params |
|---|---|---|
| `resistor` | `p`, `n` | `value` (Ω) |
| `capacitor` | `p`, `n` | `value` (F) |
| `inductor` | `p`, `n` | `value` (H) |


---

## `sources` — the single home for every source

`sources[]` is the single home for every V/I source in a spec: independent
`VoltageSource`/`CurrentSource` (with optional waveforms, below), the four
dependent (controlled) sources, and the two behavioral sources. Every entry
is emitted as one inline `ckt.add(<factory call>)`/`mod.add(<factory call>)`
(see [Emitted component lines](#emitted-component-lines);
`add_source(...)` remains a valid public alias — `Module.add()` already routes a source object to it
internally, so emitted SPICE decks are identical either way).

Each independent-source entry creates a `VoltageSource` or `CurrentSource`.

```json
{
  "type":     "VoltageSource",
  "instance": "vbias",
  "positive": "vbias",
  "negative": "0",
  "dc":       1.2,
  "ac":       0,
  "waveform": { ... },
  "pwl":      [ ... ]
}
```

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `type` | string | yes | — | `"VoltageSource"` or `"CurrentSource"` |
| `instance` | string | yes | — | Instance name passed to the factory (SPICE emission adds its device prefix) |
| `positive` | string | yes | — | Positive terminal net |
| `negative` | string | yes | — | Negative terminal net |
| `dc` | number | no | `0` | DC value (V or A) |
| `ac` | number | no | `0` | AC small-signal magnitude |
| `waveform` | object | no | omitted (`None`) | Transient waveform — see [waveforms](#waveforms) |
| `pwl` | array | no | omitted (`None`) | Piecewise-linear: flat `[t0, v0, t1, v1, ...]` |
| `array` | int or list | no | `null` | If set, this entry is an array. Int N → `range(N)`; list `[a, b]` → `range(a, b)`. Net values may contain `$i` / `$(i±N)` templates — see [Index templates](#index-templates-i--in). |
| `naming` | string | no | `"bracketed"` | When `array` is set: `"bracketed"` emits `U[0]`, `U[1]`, ...; `"underscore"` emits `U_0`, `U_1`, .... Affects emitted names only for arrays; a supplied value must still be `"bracketed"` or `"underscore"`. |

Supply at most one non-null `waveform` or `pwl`. Omission and `null` mean
absent; empty objects and arrays, and malformed fields, are rejected. This applies to scalar, array and testbench generation.
`dc` adds to the entire waveform through two emitted source cards, including
when zero. Use zero or omit it for waveform levels that are already absolute.
`ac` is placed only on the DC card. A constant source still emits one card.

### Dependent (controlled) sources — VCVS/VCCS/CCVS/CCCS

`sources[]` entries also accept the four dependent (controlled) sources —
`VoltageControlledVoltageSource`/`VCVS`, `VoltageControlledCurrentSource`/
`VCCS`, `CurrentControlledVoltageSource`/`CCVS`,
`CurrentControlledCurrentSource`/`CCCS` (`type` accepts either the full
class name or the short alias). These are structural — a linear function of
a control node-pair voltage (VCVS/VCCS) or of a *controlling source's
branch current* (CCVS/CCCS) — not stimulus generators: no `dc`/`ac`/
`waveform`/`pwl`. All four accept `array` and `naming`.

```json
{"type": "VCVS", "instance": "e1", "out_pos": "vout", "out_neg": "0",
 "ctrl_pos": "vin", "ctrl_neg": "0", "gain": 10}
```

```json
{"type": "CCCS", "instance": "f1", "out_pos": "icopy", "out_neg": "0",
 "gain": 2, "ctrl_vname": "vmeas"}
```

| Field | Type | Required | Applies to | Notes |
|---|---|---|---|---|
| `type` | string | yes | all four | Full class name or alias (`VCVS`/`VCCS`/`CCVS`/`CCCS`) |
| `instance` | string | yes | all four | Instance name passed to the factory (SPICE emission adds its device prefix) |
| `out_pos`, `out_neg` | string | yes | all four | Output terminal nets |
| `ctrl_pos`, `ctrl_neg` | string | yes | VCVS, VCCS | Control node-pair nets |
| `gain` | number | yes | VCVS (V/V), CCCS (A/A) | Dimensionless |
| `gm` | number | yes | VCCS | Transconductance, A/V |
| `transresistance` | number | yes | CCVS | Transresistance, V/A |
| `ctrl_vname` | string | yes | CCVS, CCCS | **Instance name** of the controlling source — see below |
| `array` | int or list | no | all four | N → `range(N)`; `[a, b]` → `range(a, b)` |
| `naming` | string | no | all four | `"bracketed"` (default) → `name[i]`; `"underscore"` → `name_i` |

Every pin (`out_pos`, `out_neg`, and the VCVS/VCCS control pins) accepts the
same `$i` / `$(i±N)` index templates as independent-source arrays. Scalar
entries retain literal pin names.

For an arrayed CCVS/CCCS, `ctrl_vname` names the controller entry's base
`instance`. A scalar voltage-branch controller is shared by all elements.
An arrayed controller must have the same half-open range (`array: 3` equals
`array: [0, 3]`); element i binds to controller i using the **controller's**
`naming`, which may differ from the consumer's. A scalar consumer cannot name
an entire controller array. Mismatched ranges and non-voltage-branch targets
raise `ValueError` naming the entries and both ranges (`scalar` where absent).
The same rules apply in testbench and rig `sources[]`; testbench generation
wraps invalid inputs in `TestbenchSpecError`, a `ValueError` subclass.
The source catalog has no array-support field.


`ctrl_vname` names the controlling voltage source — the zero-valued ammeter
whose branch current CCVS/CCCS reads (SPICE's H/F elements control on a
*source's current*, not a node voltage) — by its `instance`, e.g. a
`VoltageSource` placed purely to carry the sensed current. It is not a pin and not a numeric parameter (see
`export_source_schema()`'s `references` field, `docs/schema-export.md`).
For `CircuitSpec` JSON, the generator requires `ctrl_vname` to name an
instance in the same `sources[]` list, raising `ValueError` otherwise; a
missing required field raises `KeyError` naming the field. Scalar testbench
sources have a broader reference scope, described below.

#### Eligible and ineligible ctrl_vname targets

Only a source with a voltage-source branch current can be sensed:

| Eligible (has a branch current — V, E, H, B `V=`) | Ineligible (no branch current — I, G, F, B `I=`) |
|---|---|
| `VoltageSource` | `CurrentSource` |
| `VoltageControlledVoltageSource` / `VCVS` | `VoltageControlledCurrentSource` / `VCCS` |
| `CurrentControlledVoltageSource` / `CCVS` | `CurrentControlledCurrentSource` / `CCCS` |
| `BehavioralVoltage` (below) | `BehavioralCurrent` (below) |

In `CircuitSpec` JSON, a `ctrl_vname` naming an ineligible target raises
`ValueError`, naming the referencing instance and the target instance and type:

```
ValueError: sources entry 'f1' (CurrentControlledCurrentSource): ctrl_vname='ibias'
targets source 'ibias' (CurrentSource), which has no voltage-source branch
current to sense. Choose a voltage source (V, E, H or B V=), or insert a
zero-volt ammeter source in the current path.
```

Worked example — a CCVS sensing a `BehavioralVoltage` ammeter (eligible):

```json
{"sources": [
  {"type": "BehavioralVoltage", "instance": "iprobe",
   "positive": "isense", "negative": "0", "expression": "0"},
  {"type": "CurrentSource", "instance": "idrive",
   "positive": "vdd", "negative": "isense", "dc": 0.001},
  {"type": "CCVS", "instance": "h1", "out_pos": "vout", "out_neg": "0",
   "transresistance": 1000, "ctrl_vname": "iprobe"}
]}
```

`iprobe` (a zero-volt `BehavioralVoltage` ammeter, `expression: "0"`) carries
the 1 mA `idrive` current through its branch; `h1` senses that branch current
and reports `v(vout) = transresistance * i(iprobe) = 1000 * 0.001 = 1.0` V.

#### `ctrl_vname` at deck emission: instance name → SPICE element name

Use the Python/JSON **instance name** for `ctrl_vname`. At deck emission,
silk translates a matching instance to the controlling source's SPICE element
name: a `VoltageSource`/`sources.vdc(...)` instance `vmeas` becomes `Vvmeas`,
and a `BehavioralVoltage` instance `iprobe` becomes `Biprobe`.

`CircuitSpec` JSON requires a matching eligible instance in its own
`sources[]`; do not add a SPICE prefix to that reference. Scalar
`TestbenchSpec` consumers preserve the literal reference at generation time
when the named controller is absent or scalar. They can therefore reference
a test-local source from a rig, or one explicitly named expanded element
such as `vdd[0]`. The reference resolves against the assembled module at deck
emission; if no instance matches, it passes through unchanged, as in the
handwritten Python API. Same-list membership and eligibility checks are not
applied to these scalar testbench references at generation time. References
involving an arrayed consumer or a controller found as an array entry retain
the array membership, eligibility and matching-range checks described above.

For the worked example above, the emitted H card resolves `ctrl_vname:
"iprobe"` to `iprobe`'s own SPICE element name (`Biprobe`, from its
`BehavioralVoltage` prefix `B`), not the instance name verbatim:

```
Hh1 vout 0 Biprobe 1000
```

Emission calls the class directly — `sources.VoltageControlledVoltageSource(
'e1', out_pos=..., out_neg=..., ctrl_pos=..., ctrl_neg=..., gain=...)` —
rather than through the `sources.v*`/`sources.i*` declarative alias surface
that `VoltageSource`/`CurrentSource` use, since these classes take no dc/ac/
waveform and so have no alias/quantity entry to build one from.

### Behavioral sources — BehavioralVoltage/BehavioralCurrent

`sources[]` entries also accept the two behavioral sources —
`BehavioralVoltage` (SPICE `B V=`) and `BehavioralCurrent` (SPICE `B I=`).
Like the controlled sources above, these are structural, not stimulus
generators: no `dc`/`ac`/`waveform`/`pwl`. Both accept `array` and `naming`
(see below).

```json
{"type": "BehavioralVoltage", "instance": "bv1",
 "positive": "vout", "negative": "0", "expression": "v(vin) * 2"}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | `"BehavioralVoltage"` or `"BehavioralCurrent"` (full class name only — no short alias) |
| `instance` | string | yes | Instance name passed to the factory (SPICE emission adds its device prefix) |
| `positive` | string | yes | Positive terminal net |
| `negative` | string | yes | Negative terminal net |
| `expression` | string | yes | The `B` element's `V=`/`I=` expression, passed through to SPICE verbatim (e.g. `"v(vin) * 2"`) |
| `array` | int or list | no | N → `range(N)`; `[a, b]` → `range(a, b)` |
| `naming` | string | no | `"bracketed"` (default) → `name[i]`; `"underscore"` → `name_i` |

**Arrays.** The `$i`, `$(i)`, `$(i+N)` and `$(i-N)` index
templates are rewritten per element in `positive`, `negative` **and**
`expression`, in the circuit, module, testbench and rig `sources[]` paths
(the same shared call). Braces in an expression (e.g. `{param}`) survive the
rewrite, and an expression with no template is emitted unchanged for every
element. Scalar entries keep their text literally. silk does not parse
expressions, so:

- `i()` takes the **emitted SPICE element name**, as returned by
  `silk.circuit.emitted_names.emitted_instance_name`; silk does not rewrite
  it. Spell it with a template so each element names its own source.
- Templates are expanded using the element's own index (`$(i+1)` is that
  element's index plus one); silk does not resolve or bounds-check the
  resulting SPICE references, and it adds no expression parser.
- An out-of-range reference (e.g. `v(in[$(i+1)])` at the last element) is
  caught only by the simulator, loudly, at simulation time.

The source catalog (`export_source_schema()`) has no array-support field.

A missing `positive`/`negative`/`expression` raises `KeyError` naming the
field and the instance, mirroring the controlled sources above. Emission
calls the class directly — `sources.BehavioralVoltage('bv1',
positive='vout', negative='0', expression='v(vin) * 2')` — the same
reasoning as the controlled sources (no dc/ac/waveform alias surface).

Both classes are schematized by `export_source_schema()`, the
same as `VoltageSource`/`CurrentSource`/the controlled sources above —
`_meta.unschematized_sources` is empty. `expression` is not a
`parameters[]` entry (it carries no unit — not a physical quantity) and not
a `references[]` entry (it isn't resolved against other `sources[]`
entries); it is listed under its own `expressions` key
(`{name: "expression", kind: "expression", required: true, default: null}`)
so a consumer can tell it apart from both — see `docs/schema-export.md`'s
"Behavioral sources" subsection for the full shape and a worked example.
A `BehavioralVoltage` is an eligible `ctrl_vname` target (it has a branch
current); a `BehavioralCurrent` is not — see the eligibility table above.

### Source schema composition and parameter kinds

`export_source_schema()` reports the constructor slot in each
`waveforms[kind].kwarg`. A dedicated kind keeps its existing keyword (for
example, `pulse` or `pwl`); a registered kind without a dedicated source
keyword reports `waveform`. The registry key still identifies the kind, and
`class` and `code_template` describe construction of its waveform object.
Consumers must pass that object through the reported `kwarg`, rather than
assuming the registry key is a source constructor argument. This applies to
both `VoltageSource` and `CurrentSource`.

The optional top-level `param_kinds` mapping in `source_metadata.yaml` supplies
parameter input hints by name, alongside `param_units`. For example, an
integration declaring `bits: bits` exports `kind: bits` on a waveform's
`bits` parameter; consumers can distinguish bit input from numeric input.
Unmapped parameters omit `kind` entirely. Names, defaults, required flags and
enumeration choices remain reflected from Python signatures; the mapping
must not duplicate them. The shipped `bitpattern` kind declares `bits: bits`;
its bits are literal sequence input, while `ui`, levels and edges are numeric
parameters. On both `VoltageSource` and `CurrentSource`, `kind: integer`
marks `prbs.order`, `prbs.length`, `prbs.seed` and `bitpattern.repeat`.
Consumers must send literal integers: floats (including whole-valued floats)
and numeric strings raise `TypeError`. Dimensionless float inputs such as
`mdi` remain untagged. The name-keyed mapping applies to every waveform.
Both kinds report `kwarg="waveform"` and share finite PWL
timing; see the [finite-source API](api-reference.md#finite-bit-pattern-and-prbs-sources).
Waveform parameter hints and constructor slots propagate to the
`vsource`/`isource` device catalog; its templates still use generated source
factories. These input hints do not change waveform validation or SPICE emission.


### Waveforms

The classes below are common waveform choices. `BitPattern` and `PRBS` are
also accepted through `waveform` with nested `params`; see the
[finite-source API](api-reference.md#finite-bit-pattern-and-prbs-sources) for
their required bits or generator settings, levels, unit interval and edges.
Use `export_source_schema()` for the complete live waveform catalog.

Set `waveform.type` to one of the class names below and provide the
parameters either as additional keys (the flat form used by the examples
below) or nested under a `params` key — the same `{"type": ..., "params":
{...}}` shape `devices[]`/`ideal_elements[]` already use:

```json
"waveform": {
  "type": "Pulse",
  "params": { "v1": 0, "v2": 1.8, "tr": 1e-9, "tf": 1e-9, "pw": 10e-9, "per": 20e-9 }
}
```

Both forms generate identically; a dict combining them (`params` plus a flat
key beside `type`) is rejected. The `params` form exists so a waveform whose
own parameter is named `type` — currently only `RandomSource` — can be
expressed at all: without nesting, that parameter's key collides with the
`type` key that names the waveform class. See [`RandomSource`](#randomsource-trrandom)
below.

#### `Pulse`

```json
"waveform": {
  "type": "Pulse",
  "v1": 0, "v2": 1.8,
  "td": 0, "tr": 1e-9, "tf": 1e-9, "pw": 10e-9, "per": 20e-9
}
```

| Param | Default | Description |
|---|---|---|
| `v1` | — | Initial value |
| `v2` | — | Pulsed value |
| `td` | `0` | Delay (s) |
| `tr` | required | Rise time (s); explicitly supply a positive value |
| `tf` | required | Fall time (s); explicitly supply a positive value |
| `pw` | `0` | Pulse width (s) |
| `per` | `0` | Period (s) |
| `np` | `null` | Number of pulses (null = unlimited) |

#### `Sine`

```json
"waveform": {
  "type": "Sine",
  "voffset": 0, "vampl": 0.1, "freq": 1e6,
  "td": 0, "theta": 0, "phase": 0
}
```

| Param | Default | Description |
|---|---|---|
| `voffset` | `0` | Waveform offset (V or A), added to source `dc` |
| `vampl` | — | Peak amplitude (V or A) |
| `freq` | — | Frequency (Hz) |
| `td` | `0` | Delay (s) |
| `theta` | `0` | Damping factor |
| `phase` | `0` | Phase (degrees) |

#### `Exponential`

```json
"waveform": {
  "type": "Exponential",
  "v1": 0, "v2": 1.8,
  "td1": 0, "tau1": 10e-9, "td2": 20e-9, "tau2": 10e-9
}
```

| Param | Default | Description |
|---|---|---|
| `v1` | — | Initial value |
| `v2` | — | Target value |
| `td1` | `0` | Rise delay (s) |
| `tau1` | `0` | Rise time constant (s) |
| `td2` | `0` | Fall delay (s) |
| `tau2` | `0` | Fall time constant (s) |

#### `SFFM` (single-frequency FM)

```json
"waveform": {
  "type": "SFFM",
  "vo": 0, "va": 1.0, "fc": 1e6, "mdi": 5, "fs": 10e3
}
```

| Param | Default | Description |
|---|---|---|
| `vo` | — | DC offset |
| `va` | — | Amplitude |
| `fc` | — | Carrier frequency (Hz) |
| `mdi` | — | Modulation index |
| `fs` | — | Signal frequency (Hz) |
| `phasec` | `0` | Carrier phase (degrees) |
| `phases` | `0` | Signal phase (degrees) |

#### `AM` (amplitude modulation)

```json
"waveform": {
  "type": "AM",
  "va": 1.0, "vo": 0, "mf": 0.5, "fc": 1e6
}
```

| Param | Default | Description |
|---|---|---|
| `va` | — | Amplitude |
| `vo` | — | DC offset |
| `mf` | — | Modulating frequency (Hz) |
| `fc` | — | Carrier frequency (Hz) |
| `td` | `0` | Delay (s) |
| `phases` | `0` | Phase (degrees) |

#### `TransientNoise`

```json
"waveform": {
  "type": "TransientNoise",
  "na": 1e-6, "nt": 1e-9
}
```

| Param | Default | Description |
|---|---|---|
| `na` | — | Noise amplitude (V or A RMS) |
| `nt` | — | Time step (s) |
| `nalpha` | `0` | 1/f exponent |
| `namp` | `0` | 1/f amplitude |
| `rtsam` | `0` | RTS amplitude |
| `rtscapt` | `0` | RTS capture time (s) |
| `rtsemt` | `0` | RTS emission time (s) |

#### `RandomSource` (TRRANDOM)

`RandomSource`'s own constructor parameter is named `type` — the same name
as the waveform dict's class-selector key — so it can only be reached
through the nested `params` form:

```json
"waveform": {
  "type": "RandomSource",
  "params": { "type": 1, "ts": 1e-9 }
}
```

| Param | Default | Description |
|---|---|---|
| `type` | — | Distribution: `1`=uniform, `2`=gaussian, `3`=exponential, `4`=poisson (passed through to SPICE `TRRANDOM(...)` as-is — an integer code, not a name) |
| `ts` | — | Time step (s) |
| `td` | `0` | Delay (s) |
| `param1` | `1` | Distribution parameter 1 |
| `param2` | `0` | Distribution parameter 2 |

`RandomSource.type` accepts an integer code or `RandomType` member and emits an integer.
The source schema exports the enum as `choices` on the `type` parameter:

```json
{"name": "type", "unit": null, "default": null, "required": true,
 "choices": [{"value": 1, "label": "uniform"},
             {"value": 2, "label": "gaussian"},
             {"value": 3, "label": "exponential"},
             {"value": 4, "label": "poisson"}]}
```

Only enumerated source parameters include `choices`; other parameters omit it.
The [metric schema](schema-export.md#part-f--metric-catalog-export) uses a different `parameters[].choices` shape: the key is always present and is `null` when there is no permitted set.

#### PWL (piecewise-linear)

Use flat `pwl` for points alone; for timing, use
`waveform: {"type": "PWL", "params": {"points": [...], "r": ..., "td": ...}}`;
see [PWL timing limits](api-reference.md#waveform-shapes).

```json
"pwl": [0, 0, 1e-9, 1.8, 2e-9, 1.8, 3e-9, 0]
```

Flat array alternating time (s) and value (V or A).

---

## `simulation` — simulation command

Used by standalone scripts (when the `ports` key is absent or `null`). Omission or `{}` emits no `add_sim_type` call, although the script still calls `simulate`; supply an analysis for a useful simulation. Ignored for module definitions, including zero-pin ones (`ports: {}`). The `type` is case-insensitive and defaults to `"Transient"` in a non-empty request.

### Transient

```json
"simulation": {
  "type": "Transient",
  "step": 1e-9,
  "stop": 100e-9
}
```

| Param | Default | Description |
|---|---|---|
| `step` | `1e-9` | Time step (s) |
| `stop` | `100e-9` | Stop time (s) |

This JSON form forwards only `step` and `stop`; Python Transient options
`start`, `maxstep` and `uic` are not forwarded.

### AC

```json
"simulation": {
  "type": "AC",
  "variation": "dec",
  "points": 10,
  "fstart": 1,
  "fstop": 1e9
}
```

| Param | Default | Description |
|---|---|---|
| `variation` | `"dec"` | Sweep type: `"dec"`, `"oct"`, `"lin"` |
| `points` | `10` | Points per decade/octave or total |
| `fstart` | `1` | Start frequency (Hz) |
| `fstop` | `1e9` | Stop frequency (Hz) |

### DC

```json
"simulation": {
  "type": "DC",
  "source": "vdd",
  "start": 0,
  "stop": 1.8,
  "step": 0.01
}
```

| Param | Default | Description |
|---|---|---|
| `source` | required | Source instance name in `sources[]` to sweep (an array needs a single expanded name); `"temp"` requests a temperature sweep |
| `start` | `0` | Start value (V or A) |
| `stop` | `1.8` | Stop value (V or A) |
| `step` | `0.01` | Step size |

### Noise

```json
"simulation": {
  "type": "Noise",
  "output": "vout",
  "ref": "0",
  "source": "vin",
  "variation": "dec",
  "points": 10,
  "fstart": 1,
  "fstop": 1e9
}
```

| Param | Default | Description |
|---|---|---|
| `output` | `"vout"` | Output node for noise measurement |
| `ref` | `"0"` | Reference node (usually ground) |
| `source` | required | Input source instance name in `sources[]` for input-referred noise (an array needs a single expanded name) |
| `variation` | `"dec"` | Sweep type: `"dec"`, `"oct"`, `"lin"` |
| `points` | `10` | Points per decade/octave or total |
| `fstart` | `1` | Start frequency (Hz) |
| `fstop` | `1e9` | Stop frequency (Hz) |

---

### Operating point

`"simulation": {"type": "Op"}` emits `analysis.Op()` with no parameters.

## Complete example — differential pair module with annotated ports

```json
{
  "circuit_name": "diff_pair",
  "pdk": "sky130",
  "output_dir": "./subckt",
  "voltage_domains": {
    "AVDD": {"nominal": 1.8, "tolerance": 0.1}
  },
  "ports": {
    "vdd":    {"dir": "input",  "role": "power",  "domain": "AVDD"},
    "vss":    {"dir": "input",  "role": "ground", "domain": "AVDD"},
    "vin_p":  {"dir": "input",  "role": "analog", "domain": "AVDD"},
    "vin_n":  {"dir": "input",  "role": "analog", "domain": "AVDD"},
    "vout_p": {"dir": "output", "role": "analog", "domain": "AVDD"},
    "vout_n": {"dir": "output", "role": "analog", "domain": "AVDD"},
    "vbias":  {"dir": "input",  "role": "analog", "domain": "AVDD"}
  },
  "devices": [
    {
      "type": "nmos1p8", "instance": "mn_p",
      "nets": {"d": "vout_n", "g": "vin_p", "s": "tail", "b": "vss"},
      "params": {"w": 4.0, "l": 0.15, "m": 2}
    },
    {
      "type": "nmos1p8", "instance": "mn_n",
      "nets": {"d": "vout_p", "g": "vin_n", "s": "tail", "b": "vss"},
      "params": {"w": 4.0, "l": 0.15, "m": 2}
    },
    {
      "type": "pmos1p8", "instance": "mp_p",
      "nets": {"d": "vout_p", "g": "vout_p", "s": "vdd", "b": "vdd"},
      "params": {"w": 4.0, "l": 0.15, "m": 1}
    },
    {
      "type": "pmos1p8", "instance": "mp_n",
      "nets": {"d": "vout_n", "g": "vout_p", "s": "vdd", "b": "vdd"},
      "params": {"w": 4.0, "l": 0.15, "m": 1}
    },
    {
      "type": "nmos1p8", "instance": "mn_tail",
      "nets": {"d": "tail", "g": "vbias", "s": "vss", "b": "vss"},
      "params": {"w": 8.0, "l": 0.5, "m": 4}
    }
  ],
  "sources": [
    {
      "type": "CurrentSource", "instance": "ibias",
      "positive": "vdd", "negative": "vbias", "dc": 100e-6
    }
  ]
}
```

### Mixed-feature example — bus port + bus_net + multiple domains

For features absent from the diff-pair example above:

```json
{
  "circuit_name": "encoder",
  "pdk": "sky130",
  "voltage_domains": {
    "AVDD": {"nominal": 1.8, "tolerance": 0.1},
    "DVDD": {"nominal": 0.9, "tolerance": 0.1}
  },
  "ports": {
    "clk":  {"dir": "input"},
    "vh":   {"dir": "output", "role": "analog",  "domain": "AVDD"},
    "code": {"dir": "input",  "role": "digital", "bits": 4, "domain": "DVDD"}
  },
  "bus_nets": [
    {"name": "count", "bits": 4}
  ]
}
```

---

## What the generator produces

Given the diff-pair payload above, `gen.generate(payload)` returns a Python string:

```python
"""Generated subcircuit: diff_pair"""

from silk import Module
from silk.circuit.circuit_elements import PortDir, PortRole, Port
from silk.circuit.voltage_domain import VoltageDomain
from silk import sources

# --- PDK setup ---
from silk import load_pdk
pdk = load_pdk('sky130')

# --- voltage domains ---
AVDD = VoltageDomain('AVDD', nominal=1.8, tolerance=0.1)

# --- module definition ---
mod = Module('PDK_sky130', './subckt', 'diff_pair.cir', pdk_details=pdk,
             module_name='diff_pair')
mod.define_module_ports({'vdd': Port(dir=PortDir.IN, role=PortRole.POWER, domain=AVDD), 'vss': Port(dir=PortDir.IN, role=PortRole.GROUND, domain=AVDD), 'vin_p': Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD), 'vin_n': Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD), 'vout_p': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=AVDD), 'vout_n': Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=AVDD), 'vbias': Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD)})

# --- PDK devices ---
mod.add(pdk.nmos1p8('mn_p', d='vout_n', g='vin_p', s='tail', b='vss', w=4.0, l=0.15, m=2))
mod.add(pdk.nmos1p8('mn_n', d='vout_p', g='vin_n', s='tail', b='vss', w=4.0, l=0.15, m=2))
mod.add(pdk.pmos1p8('mp_p', d='vout_p', g='vout_p', s='vdd', b='vdd', w=4.0, l=0.15, m=1))
mod.add(pdk.pmos1p8('mp_n', d='vout_n', g='vout_p', s='vdd', b='vdd', w=4.0, l=0.15, m=1))
mod.add(pdk.nmos1p8('mn_tail', d='tail', g='vbias', s='vss', b='vss', w=8.0, l=0.5, m=4))

# --- sources ---
mod.add(sources.idc('ibias', positive='vdd', negative='vbias', value=0.0001))
```

### Emitted component lines

Every PDK device, ideal element and source is emitted as **one inline
`mod.add(...)` line** (`ckt.add(...)` in a standalone script). The instance
name is the first positional argument of the factory call, a string — it is
never a Python variable, so an instance named `pdk`, `mod`, `ckt`, `tb`,
`sources` or `elements` is legal and cannot shadow a generator local. (Instance
names must still be valid non-keyword Python identifiers.) Arrays emit the
list comprehension inside `add(*[...])`; an arrayed submodule is a one-line `for i in range(N): mod.add_module(...)` loop.

```python
mod.add(pdk.nmos1p8('M1', d='out', g='in', s='0', b='0', w=2.0, l=0.18))       # PDK device
mod.add(*[pdk.nmos1p8(f'MA[{i}]', d=f'out[{i}]', g='in', s='0', b='0', w=2.0, l=0.18) for i in range(2)])  # arrayed device
mod.add(elements.resistor('R1', p='out', n='0', value=1000.0))                    # ideal element
mod.add(sources.vdc('vdd', positive='vdd', negative='0', value=1.8))              # independent source
mod.add(sources.VoltageControlledVoltageSource('e1', out_pos='vout', out_neg='0', ctrl_pos='a', ctrl_neg='0', gain=10))  # controlled source
mod.add_module(child_alias, 'xinv', nets={...})                                     # submodule
for i in range(2): mod.add_module(child_alias, f'xa[{i}]', nets={...})          # arrayed submodule (range(a, b) for a tuple; add_element(..., parameters=...) when parameters are set)
```

Imports for `PortRole`, `Port`, and `VoltageDomain` are only emitted when the spec actually uses them. A spec with `bus_nets` adds an `# --- internal bus nets ---` block between `define_module_ports` and the devices block, with one `mod.add_net(name, bits=..., naming=...)` per `bus_nets` entry.

---

## Testbench codegen

`silk.codegen.Generator.generate_testbenches(spec)` is the testbench
counterpart of `generate`: a JSON-able **`TestbenchSpec`** becomes generated
Python that exercises a circuit through silk's public testbench API. silk owns
every emitted line; the caller owns the spec. Module-level equivalents:
`from silk.codegen.testbench_generator import generate_testbenches,
TestbenchSpec, TestbenchSpecError`. Runnable generation-only demo (no PDK or simulator):
`.venv/bin/python -m silk.examples.testbench_codegen_example`.

```json
{
  "circuit_name":    "ac_gain",
  "circuit_builder": "_build_ac_gain",
  "rigs": [
    { "name": "supply",
      "sources": [ {"type": "VoltageSource", "instance": "vdd",
                    "positive": "vdd", "negative": "0", "dc": 1.8} ] }
  ],
  "tests": [
    { "name": "ac",
      "dut_portmap": {"vout": "vout", "vin": "vin"},
      "applied_rigs": ["supply"],
      "sources": [ {"type": "VoltageSource", "instance": "vin",
                    "positive": "vin", "negative": "0", "dc": 0.5, "ac": 1.0} ],
      "analyses": [ {"kind": "ac", "variation": "dec", "points": 10, "fstart": 1, "fstop": 1e9} ],
      "metrics": [ {"name": "gain", "type": "dc_gain_dB", "signal": "vout", "params": {}} ],
      "specs":   [ {"name": "gain > 0.7 dB", "metric": "gain", "op": ">", "target": 0.7} ] }
  ]
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `circuit_name` | string | yes | Names the generated `build_<circuit_name>_with_tests(pdk)` assembler when `circuit_builder` is set — then it must be a valid, non-keyword Python identifier (same rule as [`circuit_name`](#top-level-structure) above). |
| `circuit_builder` | string | no | The caller's circuit-builder function name (e.g. `_build_ac_gain`). See the contract below. Absent/`null`/`""` → no assembler is emitted; the caller writes the one-line assembler itself. |
| `rigs` | array | no | Shared groups: `{"name", "sources": [SourceSpec…], "devices": […], "ideal_elements": […], "submodules": [SubmoduleSpec…]}` → `def _apply_<name>_rig(tb)`. `name` must be a Python identifier. |
| `tests` | array | no | One per testbench: `{"name", "dut_portmap", "applied_rigs", "sources", "devices", "ideal_elements", "submodules", "analyses", "metrics", "specs", "corner", "temperature_c"}` → `def _attach_<name>(mod)` calling `mod.create_test('<name>')`. `name` must be a Python identifier; `dut_portmap` maps DUT port → testbench net. `corner` and `temperature_c` are optional; see [Corner, temperature and op tables](#corner-temperature-and-op-tables). |
| `save_device_params` | boolean | no | `true` turns on operating-point device-parameter capture for the DUT, so `report.op_table()` works on a test with an `op` analysis. See [Corner, temperature and op tables](#corner-temperature-and-op-tables). Absent, `null` or `false` emits nothing. |

- **Sources** (`rigs[].sources` and `tests[].sources`) are the same
  [`SourceSpec`](#sources--the-single-home-for-every-source) shape as
  `CircuitSpec.sources` — independent, dependent/controlled
  (VCVS/VCCS/CCVS/CCCS, see "Dependent (controlled) sources" above) and
  behavioral (see "Behavioral sources" above) alike. An
  independent source emits as `tb.add(sources.v<kind>(...))`; a dependent
  or behavioral source emits as `tb.add(sources.<ClassName>(...))` — the
  same constructor calls the circuit generator emits inline, inside `tb.add(...)`.
- **Arrayed sources.** An independent or controlled source with `array` / `naming` in
  `rigs[].sources` / `tests[].sources` emits all N sources inline, with the
  circuit generator's instance/net semantics (`$i` templates included):
  `tb.add(*[sources.vdc(f'vb[{i}]', positive=f'p{i}', negative='0', value=1.8) for i in range(3)])`.
  Controlled-source pins and current-control references follow the rules
  above. Behavioral sources follow the same array rules.
- **Devices and ideal elements** (`rigs[].devices` / `.ideal_elements`,
  `tests[].devices` / `.ideal_elements`) — a test is a circuit, so it holds
  what a circuit holds. Entries are exactly the circuit spec's
  [`devices`](#devices--pdk-devices) and
  [`ideal_elements`](#ideal_elements--process-independent-elements) entries
  (`type`, `instance`, `nets`, `params`, optional `array` / `naming`), emitted
  by the circuit generator's own call-expression emitters and added inline,
  one `tb.add(...)` per entry, with **no variable bound to the instance name**
  (so a part named `tb`, `pdk` or `mod` cannot clobber the generated function's
  locals; no name is reserved). Order in the function: sources, devices, ideal
  elements, helpers. Example (test scope):

  ```python
      pdk = tb.pdk_details          # only when the group has devices
      tb.add(pdk.nmos1p8('Mload', d='y', g='a', s='0', b='0', w=1.0, l=0.15))
      tb.add(*[pdk.nmos1p8(f'MA[{i}]', d='y', g='a', s='0', b='0', w=1.0, l=0.15) for i in range(2)])
      tb.add(elements.resistor('rload', p='y', n='0', value=1000.0))
  ```

  `from silk.elements import elements` is added to the file imports only when
  some rig/test has `ideal_elements`. Omitting these keys emits no such parts.

  **Duplicate names.** Within one test (its applied rigs share its namespace)
  or one rig, a device or ideal element sharing a name with another device,
  element, source or helper instance — or with `dut` — is refused at
  generation, compared case-insensitively (SPICE), on the expanded names of
  arrayed entries. It raises `TestbenchSpecError` naming both offenders.
- **Analyses** use the same per-kind fields as [`simulation`](#simulation--simulation-command)
  plus `"kind"`: `transient` / `ac` / `dc` / `noise` / `op`; they emit as
  `tb.add_sim_type(analysis.<Kind>(...))`.
- **Metrics** are declarative — `{"name", "type", "signal" | "signals",
  "analysis", "params"}`
  — and emit as `tb.add_metric(name, type=..., signal=..., **params)` or
  `tb.add_metric(name, type=..., signals={...}, **params)`. `type` must be a
  registered metric; enumerate the live catalog with
  `silk.signals.export_metric_schema()`. A signal path is a **rawfile signal
  name**: the testbench net a DUT port is mapped to in `dut_portmap` (`"vout"`)
  for anything on a port, or `"xdut.<net>"` for a DUT-internal node —
  `add_module_under_test` always instances the DUT as `dut`, and SPICE aliases
  its ports to the testbench nets, so `xdut.<port>` is *not* present in the
  rawfile.

  Terminal-current probes require `save_device_currents=True` and
  `terminal_current_method='ammeter_wrapper'` on the DUT, and exactly one
  Transient analysis on the test.

  **One signal-path vocabulary**, shared by a metric's `signal` /
  every `signals` slot and a derived signal's `inputs`: a net (`"vout"`), a
  DUT-internal path (`"xdut.<net>"`), a device probe (`"xdut.m1:d"`), and a
  source's branch current as `"i(<source name>)"`, e.g. `"i(vs)"` for the
  source whose `instance` is `vs` (emitted as `Vvs`). The raw vector name
  `"i(vvs)"` is also accepted and is documented behaviour. Both readers
  (`tb.probe` and a metric's path) use one lookup: `i(x)` where `x` is a
  source name reads that source; otherwise `x` is the raw vector name,
  verbatim. If `x` is both a source name and the emitted instance of a
  *different* source (sources `vs` and `vvs`: `i(vvs)` is `Vvvs` or `Vvs`),
  the run raises `AmbiguousBranchCurrentError` naming both readings; it never
  picks one. A current that is not in the results raises, listing these
  spellings. The path is emitted as a string and resolved at run time.
  - **One input → `"signal": "<path>"`.** Use this for a one-input metric.
  - **N inputs → `"signals": {"<slot>": "<path>", …}`**, keyed by the metric's
    own signal-input names as published in
    `export_metric_schema()[type]["signal_inputs"][].name`.
    Bind inputs by name, not by position. For example, `cmrr_dB`,
    `propagation_delay`, `average_power`, `dynamic_energy_per_cycle` and
    `hysteresis` accept this form. `error_norm` is refused for a different
    reason: its `goal` slot takes a waveform no analysis produces (see the
    [error contract](#error-contract)). `"signals": {"sig": "<path>"}` on a
    one-input metric is accepted as the explicit spelling of `"signal"`;
    passing both keys is an error.

  ```jsonc
  {"name": "cmrr", "type": "cmrr_dB",
   "signals": {"vdiff": "voutd", "vcm": "voutc"}, "params": {"freq": 1e3}}
  ```
  - **`"analysis"`: which analysis's plot the metric reads** — `"tran"`,
    `"ac"`, `"dc"` or `"noise"`. Optional, and **required
    exactly when the choice is ambiguous**, so a picker should surface it only
    then. A test that declares one analysis writes one plot: omit the field.
    A test that declares several writes one plot per analysis, and silk
    **never guesses** which one a metric meant — it reads the metric's own
    `signal_inputs[].kind` against `_meta.signal_kinds[kind].produced_by`, the
    same table the [unproducible-kind refusal](#error-contract) reads:

    | declared analyses | metric | field |
    | --- | --- | --- |
    | one | any | omit |
    | `transient` + `ac` | `dc_gain_dB` (kind `frequency_response`, `produced_by: ["ac"]`) | omit — only `ac` can feed it |
    | `transient` + `ac` | `average` (kind `any`, `produced_by:` all four) | **required** — both could |

    `domain` is never consulted for this: it is the metric's home/grouping
    label, and `average`'s `domain` is `"tran"` while its input
    kind is `any`. Naming an analysis the test does not declare, or one the
    metric's kinds cannot come from, is refused at generate time.

  ```jsonc
  {"name": "vavg", "type": "average", "signal": "vout", "analysis": "tran"}
  ```
- **Specs** — `{"name", "metric", "op", "target"}` — emit as `tb.add_spec(...)`;
  `metric` must name a metric declared in the same test. Runtime specs support
  `<`, `<=`, `>`, `>=` and `==`; the target must be a real number in the metric's
  units, not a boolean. Codegen does not validate all runtime requirements.

### Corner, temperature and op tables

Three optional keys cover a run at one corner and one
temperature, and an operating-point table. Each emits one public silk call;
omitting them emits no corresponding setup calls.

```jsonc
{ "circuit_name": "nfet", "circuit_builder": "_build_nfet",
  "save_device_params": true,
  "tests": [
    { "name": "op", "dut_portmap": {"d": "vd", "g": "vg", "s": "0", "b": "0"},
      "corner": "ff", "temperature_c": 85,
      "sources": [ /* vd, vg */ ],
      "analyses": [ {"kind": "op"} ] } ] }
```

```python
def _attach_op(mod):
    mod.save_device_params = True
    tb = mod.create_test('op')
    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
    tb.set_corner('ff')
    tb.set_temperature(85.0)
    ...
```

| Key | Where | Emits | Checked at generation | Checked when the code runs |
|---|---|---|---|---|
| `corner` | `tests[]` | `tb.set_corner('<corner>')` after `add_module_under_test` | The same syntax as the CircuitSpec `corner`: a non-empty string of `[A-Za-z0-9_-]`, else `TestbenchSpecError` naming the test. A TestbenchSpec names no PDK, so membership is not checked here. | `tb.set_corner` checks the name against the PDK's corners (`load_pdk(name).corners`) when the bench is built, and raises `silk.exceptions.UnknownCornerError` listing them. It never falls back to the default corner. |
| `temperature_c` | `tests[]` | `tb.set_temperature(<float>)` after `set_corner` | A finite number above -273.15 °C. A string, a boolean, `NaN`/`Infinity` or a lower value raises `TestbenchSpecError` naming the test. | As `tb.set_temperature`: the test's own TEMP, which wins over a temperature set on the DUT. |
| `save_device_params` | top level | `mod.save_device_params = True`, first line of every `_attach_<test>` function | JSON `true`/`false`/`null` only; anything else raises `TestbenchSpecError`. | The DUT captures device operating-point parameters on every run of every test (ngspice; on Xyce `op` must be the test's only analysis, see `docs/operating-point.md`). |

**Which corner a test runs at**, highest first:

1. a corner passed to the call — `tb.run(corner='ss')`, `mod.run_test('op', corner='ss')`,
   or `mod.run_all_tests(corner='ss')`, which forwards it to every test;
2. the test's `corner` key (`tb.set_corner`);
3. the PDK's `default_corner` (sky130: `tt`).

`tb.run_corners()` is an explicit sweep and ignores the test's `corner`. The
CircuitSpec top-level `corner` is not in the list: it only sets the
standalone script's `ckt.simulate(corner=...)`, and a subcircuit generated
for use as a DUT (a spec with `ports`) carries no corner.

`save_device_params` belongs to the DUT, not to one test, so it is a top-level
key. Without it, `report.op_table()` raises `SimulationError` naming
`save_device_params`.

### Derived signals (waveform math)

A test can declare **derived signals**: waveform math written as
probe → operation → metric → spec and saved with the test, so it is
recomputed from the simulated traces on every run and a spec judges a
recomputed value, never a trace drawn in a viewer. There is no expression
language and no formula parser: each derived signal names **one operation
from silk's fixed list**, its inputs and the operation's parameters.

```jsonc
"derived_signals": [
  {"name": "vdrop",  "operation": "difference", "inputs": ["vin", "vout"]},
  {"name": "vdrop2", "operation": "scale", "inputs": ["vdrop"],
   "params": {"gain": 2.0, "offset": 0.0}}
],
"metrics": [ {"name": "drop2_avg", "type": "average", "signal": "vdrop2"} ]
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `name` | string | yes | A non-keyword Python identifier, unique among the test's derived signals. It is emitted as a local variable of `_attach_<test>`, so the names that function reads (`tb`, `mod`, `pdk`, `analysis`, `sources`, `elements`, `metrics`, `i`, `range`, the generated `_apply_*_rig` / `_attach_*` / builder names) are refused. It must not equal, case-insensitively, a DUT port or a net of the test (`dut_portmap` nets, and the nets of the test's and its applied rigs' sources, devices, ideal elements and helpers): a metric's `signal` could not tell the two apart. |
| `operation` | string | yes | One of the operation names below. |
| `inputs` | array of strings | yes | Each one is a **signal path**, in the same vocabulary as a metric's `signal` (the testbench net a DUT port maps to, `xdut.<net>`, `i(vs)`, a device probe such as `xdut.m1:d`), or the **name of a derived signal declared earlier in the list**. Order matters for `difference` and `quotient`. |
| `params` | object | no | The operation's parameters; only `scale` has any. Values are numbers. Omitted parameters take their defaults. |
| `analysis` | string | no | `"tran"`, `"ac"`, `"dc"`, `"noise"` or `"op"`: the plot this derived signal's path inputs read, in a test that declares more than one analysis. It must be an analysis the test declares. A derived signal inherits the analysis of the derived signals it uses; inputs that read different analyses are refused. |

The operation list is silk's, published as JSON-safe data by
`silk.signals.export_signal_operations()` — one entry per operation with
`name`, `min_inputs`, `max_inputs` (`null` = no upper limit), `ordered`,
`parameters` (`name`, `type`, `default`, `required`), `python` (the silk
Python it maps to) and `description`. The generator validates against that
same list, and the waveform explorer's trace math uses the same names and
definitions.

| operation | inputs | parameters | silk Python |
|---|---|---|---|
| `sum` | 2 or more | — | `a + b (+ ...)` |
| `difference` | exactly 2, ordered | — | `a - b` |
| `product` | 2 or more | — | `a * b (* ...)` |
| `quotient` | exactly 2, ordered | — | `a / b` |
| `mean` | 2 or more | — | `(a + b + ...) / n` |
| `scale` | 1 | `gain` (default 1.0), `offset` (default 0.0) | `gain * a + offset` |
| `abs` | 1 | — | `a.mag()` (\|y\|) |
| `derivative` | 1 | — | `a.derivative()` (central differences) |
| `integral` | 1 | — | `a.integral()` (cumulative trapezoid, `y[0] = 0`) |

**Emission.** Each derived signal is one line after the analyses, built from
`tb.probe(...)` and Signal arithmetic/methods; the probes defer, so nothing is
computed until the bench runs. A metric whose `signal` (or a `signals` slot)
names a derived signal is emitted in `add_metric`'s function form over the
registered metric function, with any path slots as `tb.probe(...)`, and the
file imports `from silk import metrics` only then. Metrics that bind no
derived signal keep the declarative line, and a spec without
`derived_signals` emits no derived-signal setup. For the spec above:

```python
from silk import metrics
...
    tb.add_sim_type(analysis.Transient(step=1e-08, stop=1e-06))
    vdrop = tb.probe('vin') - tb.probe('vout')
    vdrop2 = 2.0 * vdrop + 0.0
    tb.add_metric('drop2_avg', metrics.average, sig=vdrop2)
```

**Analysis selection.** A metric that binds a derived signal reads the plot
its derived signal pins. If the metric also carries `"analysis"`, every
derived signal it binds must pin the same one (its path slots are emitted as
`tb.probe(path, analysis=...)`). In a test with several analyses where the
choice is open, the refusal tells you to set `"analysis"` on the derived
signal.

**Refusals** (`TestbenchSpecError`, naming the test and the derived signal):
`derived_signals` not a list or an entry not an object; a missing, non-
identifier, keyword, reserved, duplicate or net/port-clashing `name`; an
unknown key; an unknown `operation`; the wrong number of inputs; an unknown
input (not a non-empty string, or the name of this or a later derived
signal); an unknown, missing required, or non-numeric parameter; an
`analysis` that is not a silk analysis or not declared by the test; inputs
reading different analyses; a metric `analysis` that disagrees with a
derived signal it binds.

### Buses: a list of paths for one bus

A signal entry may be an **ordered list of path strings** instead of one string.
It is a bus, written as one line, and Signal maths acts on its elements the way
numpy acts on an array. The list is accepted in a derived signal's `inputs` and
in **one** slot of a metric (`signal`, or one entry of `signals`). A plain string
is one scalar signal; a spec with no list emits scalar probes.

```jsonc
"derived_signals": [
  {"name": "dmag", "operation": "abs", "inputs": [["d[0]", "d[1]", "d[2]", "d[3]"]]}
],
"metrics": [ {"name": "dc", "type": "average", "signal": "dmag"} ]
```

```python
from silk import metrics
...
    dmag = tb.probe_bus(['d[0]', 'd[1]', 'd[2]', 'd[3]']).mag()
    for __silk_k, __silk_bit in enumerate(dmag):
        tb.add_metric(f'dc[{__silk_k}]', metrics.average, sig=__silk_bit)
```

- **Order.** Elements are the bus in list order; nothing is sorted. A descending
  bus is the list written descending. Two buses in one expression need equal
  length and combine by position; a scalar signal or number applies to every
  element. Unequal lengths are refused when the spec is generated.
- **Path spelling.** Elements use the designer vocabulary, `[k]` on every Python
  surface and `dut` for the module under test: `d[0]`, `dut.stage[2].rout[0]`.
  `<k>` is SPICE-only; an element with a `<k>` instance spelling or an `xdut.`
  prefix is refused. This check applies to list elements only.
- **A metric over a bus** is expanded by the loop shown: one scalar metric per
  element named `name[k]`, `k` counting from 0 by **list position** (not from the
  bit number in the path). The other slots of that metric must be scalar and are
  repeated in every call. `params` and `analysis` are emitted as for any
  function-form metric. `from silk import metrics` is added for such a metric.
  The loop counter is `enumerate`; its locals use the `__silk_` prefix and are
  renamed (`__silk_k1`) if a derived signal already has that name.
- **Refusals** (`TestbenchSpecError`, naming the test, entry and element): an
  empty list; an element that is not a non-empty string; a SPICE or emitted
  spelling; the same path twice (a repeated net is one signal: use the string
  form); unequal widths in one expression; two bus slots in one metric; a spec
  that names a bus metric; a generated name `name[k]` equal to another metric's
  name; a derived signal named `enumerate` in a test that emits a bus metric.
- **No graph is built at generation.** A path that does not exist fails when the
  generated code runs, with the same error a scalar `tb.probe(path)` gives.

### Testbench helper submodules

`tests[].submodules` and `rigs[].submodules` are optional lists, each defaulting
to `[]`. Helper subcircuits are placed beside the DUT. A rig
can contain sources, devices, ideal elements and helpers; `applied_rigs` places
its contents into each test that applies it. An empty rig emits `pass`.
Omitting the helper lists, or supplying empty lists, emits no helper imports
or placements.

Each helper entry has exactly this reference/wiring vocabulary:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `instance_name` | string | yes | Non-keyword Python identifier; unique within the assembled test across all applied rigs and test-local helpers. `dut` is reserved. |
| `module_ref` | object | yes | Explicit import reference; no lookup in the DUT schematic's submodule list. |
| `module_ref.import_path` | string | yes | Valid absolute or package-relative Python import path, e.g. `library.biasgen` or `.biasgen`. Relative paths resolve from the generated file's package. |
| `module_ref.module_var` | string | yes | Exported **Module object**, e.g. `mod`; a non-keyword Python identifier. The Module must have a `module_name` and declared ports. |
| `nets` | object or list | yes | Helper port name → testbench net-name string, or one dict per array copy in index order — see [per-copy nets lists](#per-copy-nets-lists). Required ports must be connected before generating the deck. |
| `array` | int or list | no | Same vocabulary and parser as circuit-level `submodules[].array` (see [`submodules`](#submodules)): int N → `range(N)`; list `[a, b]` → `range(a, b)`. Net values may contain `$i` / `$(i±N)` templates (see [Index templates](#index-templates-i--in)). |
| `naming` | string | no | `"bracketed"` (default) emits `instance_name[0]`, `instance_name[1]`, ...; `"underscore"` emits `instance_name_0`, `instance_name_1`, .... Ignored if `array` is absent. |

`parameters` is **not** supported on helper entries and raises
`TestbenchSpecError`: silk parameterizes a Module at build time (builder
arguments), and `tb.add_module` has no `parameters` argument — unlike
circuit-level `submodules[].parameters`, which calls `add_element(...,
parameters=...)` for a device element.

For example, a caller supplies this spec alongside its `_build_chip`
builder function:

```json
{
  "circuit_name": "chip",
  "circuit_builder": "_build_chip",
  "rigs": [{
    "name": "bias",
    "submodules": [{
      "instance_name": "bias1",
      "module_ref": {"import_path": ".biasgen", "module_var": "mod"},
      "nets": {"out": "vbias", "vss": "0"}
    }]
  }],
  "tests": [{
    "name": "startup",
    "dut_portmap": {"bias": "vbias", "vss": "0"},
    "applied_rigs": ["bias"],
    "submodules": [{
      "instance_name": "bias2",
      "module_ref": {"import_path": ".biasgen", "module_var": "mod"},
      "nets": {"out": "aux_bias", "vss": "0"}
    }]
  }]
}
```

Silk emits the import once for the whole spec and a call for each instance:

```python
from .biasgen import mod as mod_biasgen

def _apply_bias_rig(tb):
    tb.add_module(mod_biasgen, 'bias1', {'out': 'vbias', 'vss': '0'})

def _attach_startup(mod):
    tb = mod.create_test('startup')
    tb.add_module_under_test({'bias': 'vbias', 'vss': '0'})
    _apply_bias_rig(tb)
    tb.add_module(mod_biasgen, 'bias2', {'out': 'aux_bias', 'vss': '0'})

def build_chip_with_tests(pdk):
    mod = _build_chip(pdk)
    _attach_startup(mod)
    return mod
```

The caller owns the explicit reference and wiring fields; silk owns imports,
aliases, and API calls. Repeated `(import_path, module_var)` pairs share an
import across rigs and tests. Aliases follow the schematic generator's
`<module_var>_<sanitized_import_path>` convention, with numeric suffixes to
avoid collisions with other references, generated functions, local bindings,
and the builder function name.

An arrayed helper (`"array": 3` on the `bias1` entry above) emits a
`for i in range(3):` loop calling `tb.add_module(...)` once per expanded
instance, equivalent to a caller writing
`for i in range(3): tb.add_module(mod_biasgen, f'bias1[{i}]', {...})`
in Python by hand:

```python
def _apply_bias_rig(tb):
    for i in range(3):
        tb.add_module(mod_biasgen, f'bias1[{i}]', {'out': f'vbias{i}', 'vss': '0'})
```

Instance-name uniqueness and the `dut` reservation are checked on these
**expanded** names (`bias1[0]`, `bias1[1]`, `bias1[2]`), not the
pre-expansion `instance_name`.

Malformed helper entries and instance-name collisions across an assembled
test raise `TestbenchSpecError` with the rig/test and offending (expanded)
name. Silk imports the helper when the generated Python is executed; ordinary
Python import failures remain visible. Connections use the existing
`Module.add_module` checks: unknown ports fail at placement; undeclared nets
fail under `strict_nets`; missing required ports fail before an invalid deck
is written. Declare strict testbench nets before attachment using the Python
setup API.

Handwritten Python can use the same convenience method:

```python
from .biasgen import mod as biasgen

tb = dut.create_test('startup')
helper = tb.add_module(biasgen, 'bias1', {'out': 'vbias', 'vss': '0'})
tb.add_module_under_test({'bias': 'vbias', 'vss': '0'})
```

`helper` is the existing `SubcircuitInstance` handle. A Python caller may
instead call a factory and pass its returned Module. The executing netlist is
created lazily, and `tb.netlist.add_module(...)` remains supported. Adding
helpers does not replace `tb.module` or satisfy the requirement to add the DUT
exactly once before running. Specs, metrics and reports retain their DUT
ownership.

### `circuit_builder` emission contract

Define or import the builder in the generated file's namespace before calling
the assembler; codegen does not import it.

`circuit_builder` is emitted **bare** as a call target — `mod =
<circuit_builder>(pdk)` — so it must be a **bare Python identifier that is not a
keyword**: the plain name of the builder function. Dotted attribute paths
(`pkg.mod.build`), call expressions (`_build_c()`) and any statement-shaped text
(`evil(); x`, quotes, newlines, `#` comments) raise `TestbenchSpecError` naming
the field and the accepted form **before any code is produced**, for dict and
direct `TestbenchSpec` input alike. The circuit name follows the same
identifier rule; see the [top-level structure](#top-level-structure) table.

### Error contract

`generate_testbenches` raises `TestbenchSpecError` — a `ValueError` subclass —
at generation time for the validation cases below. Messages identify the
offending test, metric, spec or rig where applicable. Checked: missing `circuit_name`; a rig, test,
metric or spec without `name`, or a test without `dut_portmap`; duplicate rig
or test names; a `spec.metric` that names no metric in its test; an
`applied_rigs` entry naming no rig; a metric with no binding or an unknown
`type`; a source entry missing required fields; an unknown analysis `kind`;
a malformed test `corner`, a test `temperature_c` that is not a finite number
above -273.15 (strings and booleans included), and a non-boolean
`save_device_params`. Missing required keys encountered during validation or
assembly raise `TestbenchSpecError` rather than escaping as bare `KeyError`.

Supply arrays of objects for `rigs` and `tests`. Malformed top-level
containers are not consistently wrapped: for example, `rigs: [1]` raises
`AttributeError`, and `tests: null` raises `TypeError`. Validation does not
check whether the circuit implements the intended design.

Metric validation checks whether the declared signals and analyses can
supply every input. The following refusals each
name the test, the metric, its `type`, and the offender:

| refused | example message |
| --- | --- |
| an unbound or unknown slot | `test 't' metric 'tpd' (type 'propagation_delay'): 'signals' is missing 'sig_out'. This metric takes 2 signal inputs: 'sig_in' (kind 'transient'), 'sig_out' (kind 'transient'); bound: 'sig_in'.` |
| `"signal"` on a multi-input metric | `test 't' metric 'cmrr' (type 'cmrr_dB'): 'signal' binds a single input, but this metric takes 2: 'vdiff', 'vcm'. Use "signals": {"vdiff": ..., "vcm": ...}.` |
| an **unproducible kind** — an input whose `kind` has an empty `produced_by` in `_meta.signal_kinds`, so no probe can ever fill it | `test 't' metric 'e' (type 'error_norm'): input 'goal' needs a Signal of kind 'reference_waveform', which no silk analysis produces — it must be constructed and passed through add_metric's callable form.` |
| a param outside its declared `choices` | `test 't' metric 'tpd' (type 'propagation_delay'): param 'in_edge' is 'rsing', which is not one of ['rising', 'falling', 'any']` |
| a metric input whose kind cannot be produced by any analysis in the test (the diagnostic calls it `domain`) | `test 't' metric 'g' (type 'dc_gain_dB'): domain 'ac' has no matching analysis in this test; declared: 'transient'. Add an 'ac' analysis, or measure this metric in a test that has one.` |
| a metric more than one declared analysis could feed, with no `"analysis"` | `test 't' metric 'vavg' (type 'average'): 'vout' could come from more than one analysis in this test: 'ac', 'tran'. Add analysis='ac' or analysis='tran' to the metric.` |
| `"analysis"` naming an analysis the test does not declare | `test 't' metric 'vavg' (type 'average'): analysis='noise', but this test declares 'tran', 'ac'. Add 'noise' to the test with add_sim_type, or name one of the analyses it has.` |
| `"analysis"` naming one the metric's input kinds cannot come from | `test 't' metric 'g' (type 'dc_gain_dB'): analysis='tran' cannot feed this metric's input ('sig' (kind 'frequency_response')). Analyses in this test that can: 'ac'.` |

The missing-required-param (`gain_at_freq` without `freq`) and unknown-param
refusals also apply. Analysis matching translates the two
vocabularies: `produced_by: ["tran"]` matches an
analysis of `"kind": "transient"`. The metric's `domain` groups metrics;
its signal input kinds determine which analyses can feed it. Metrics with an input
whose `produced_by` is empty are refused; construct that input in Python
and bind it through `add_metric`'s callable form; see
[`schema-export.md`](schema-export.md#schema-v2--signal-input-kinds).
