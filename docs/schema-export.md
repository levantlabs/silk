# Schema Export APIs

Use these APIs to discover silk's devices, sources, metrics and module
interfaces. They return dictionaries describing names, pins, parameters and
code templates. Catalog export needs neither a simulator nor installed PDK
models.

For the JSON that a tool **sends** to `silk.codegen.Generator`, see
[`codegen-contract.md`](codegen-contract.md). This page describes what a tool
**reads** from silk before building that JSON.

## Recommended architecture

Keep the circuit's JSON spec as the editor's saved document. Use the catalogs
below to populate component pickers and parameter fields, then call
`Generator().generate(spec)` to get Python source. The call returns a string;
`Generator().generate_to_file(spec, path)` writes it to a file.

Importing existing Python circuits by executing them and inspecting their
`Module` objects requires additional work. `Module.export_schema()` describes
an interface, not a complete circuit serialization; see [Limitations](#limitations).

## Part A — Device catalog export

### A1. Signatures

```python
from silk import export_schema, export_schema_json, load_pdk, module_schema
from silk.elements import elements
from silk.elements.schema import device_schema

pdk = load_pdk('sky130')
catalog = export_schema(pdk)
resistor = export_schema(elements)['resistor']
text = export_schema_json(elements, indent=2, include_metadata=False)
assert resistor == device_schema(elements.resistor)
```

The signatures are `export_schema(source, include_metadata=True)` and
`export_schema_json(source, indent=2, **kwargs)`. The JSON wrapper forwards
`kwargs` to `export_schema`.

`source` may be a loaded PDK, an `ElementLibrary` instance such as `elements`,
or a `Module` instance. A PDK or element library returns a map of device names
to device descriptions. A module returns one entry keyed by `module_name`,
or `unnamed_module` when no module name is set.

With metadata enabled, `_meta` contains `source_type`, `device_count` and
`format_version: "1.0"`. PDKs with corner metadata also expose `_library`
with `pdk_name`, `default_corner` and `corners`. Read entries whose names do
not start with `_` as devices; do not depend on their discovery order.
Discovery order, unspecified fields and metadata details beyond the stated
format version are not compatibility promises.

Each device entry contains:

| Field | Meaning |
|---|---|
| `name` | Device type name; use it as `devices[].type` or `ideal_elements[].type` in a generator spec |
| `is_pdk_device` | Whether this is a PDK device |
| `pdk_model` | SPICE model name, or `null` for an ideal element |
| `spice_prefix` | Netlist instance prefix, such as `XM`, `R` or `C` |
| `category` | Classification such as `transistor`, `passive`, `source`, `diode` or `other` |
| `pins` | Ordered array of `{name, description}`; names are constructor keyword arguments |
| `parameters` | Ordered array of `{name, default, unit, min, max}` |
| `code_template` | Python format string illustrating construction; see A3 |
| `source_class` | Rich source class name for a source entry, otherwise `null` |

Module entries have a different shape: `name`, `type: "module"`,
`is_pdk_device: false`, `pdk_model: null`, `spice_prefix: "X"`,
`category: "module"`, `pins` with names and directions, `devices` with element
instance names, and `code_template`. Call `module_schema(module)` or
`module.export_schema()` for that description directly.

Catalog device, pin and parameter names are Python identifiers and cannot be
Python keywords. Invalid names in PDK device-details files raise `PDKError`;
invalid names in ideal-element or source metadata raise `CircuitError`.
These checks also cover aliases and source factory/waveform names. The public
helper `silk.utils.require_python_identifier(value, context)` raises
`ValueError` for an invalid identifier. A PDK's SPICE model name is separate
and need not be a Python identifier.

### A2. Kwarg canonicalization

Use `parameters[].name` directly for constructor keywords and generator
`params` keys. For example:

```python
from silk.elements import elements

r1 = elements.resistor('r1', p='vout', n='0', value=1000)
```

The deprecated `silk.elements.passives.Resistor` constructor uses `resistance`
or `res`; these names are absent from the catalog. New generated code
should use `elements.resistor(..., value=...)`. Pins likewise use the names
in `pins[]`.

Parameter `default` may be a number, string or `null`. `unit` describes the
parameter's native unit as a string, such as `u` (µm), `Ohm`, `F` or `H`,
or is `null` when no unit is declared. With the default
`load_pdk` settings, device dimensions use the catalog's PDK-native units.
An explicit `user_units='si'` changes the device input convention; see
[PDK units](api-reference.md#load_pdkname--pdk_rootnone-library_locationnone-cornernone-user_unitsnone).

`min` and `max` may be strings, numbers or `null`; `"-1"` denotes an
unbounded endpoint in PDK constraints. Caller-supplied numeric PDK parameters
are checked against bounds during netlist emission; catalog defaults and SPICE
string values bypass that check. A parameter's unit label is not an instruction
for an editor to rescale values automatically.

### A3. `code_template`

Templates illustrate constructor calls; they are not a promise of
byte-identical `Generator` output. Build a generator spec from names, pins
and parameters when possible. Template formatting and argument order can
change.

Device templates use `{instance_name}`, `{net_<pin>}` and
`{param_<parameter>}`. For example, a resistor template is:

```text
{instance_name} = elements.resistor('{instance_name}', p={net_p}, n={net_n}, value={param_value})
```

Substituted net names and string parameter values need valid Python literal
quoting, for example `repr('vout')`. Templates do not quote those substitutions
for you. Module templates show the constructor and `define_module_ports`
call, but the caller must replace the constructor's `...` with real arguments.

### A4. Worked examples

Inspect the installed catalog rather than keeping a separate device list:

```python
from silk import export_schema, load_pdk
from silk.elements import elements

nmos = export_schema(load_pdk('sky130'))['nmos1p8']
assert nmos['pdk_model'] == 'sky130_fd_pr__nfet_01v8'
assert [p['name'] for p in nmos['pins']] == ['d', 'g', 's', 'b']
assert [p['name'] for p in nmos['parameters']] == ['m', 'w', 'l']

ideal = export_schema(elements, include_metadata=False)
assert set(ideal) == {'capacitor', 'inductor', 'isource', 'resistor', 'vsource'}
assert ideal['resistor']['parameters'] == [
    {'name': 'value', 'default': 1000, 'unit': 'Ohm', 'min': None, 'max': None}
]
```

The sky130 `nmos1p8` parameter values are:

| Parameter | Default | Unit | Minimum | Maximum |
|---|---|---|---|---|
| `m` | `1` | `null` | `"1"` | `"-1"` |
| `w` | `1.0` | `u` | `"0.42"` | `"-1"` |
| `l` | `0.15` | `u` | `"0.15"` | `"-1"` |

The `vsource` and `isource` entries describe `VoltageSource` and
`CurrentSource`. Their parameters include `dc` and `ac`; they also carry
`symbol` (`VSource` and `ISource`, respectively), `waveforms` and
`waveform_exclusive: true`. Each waveform has its
own parameters and declarative template, such as `sources.vpulse(...)`.
The scalar template uses `sources.vdc` or `sources.idc`, mapping the catalog's
`dc` parameter to the factory's `value` argument. Put sources in the generator
spec's `sources[]`, not `ideal_elements[]`.

A source waveform parameter annotated with an `IntEnum` has
`choices: [{"value": ..., "label": ...}]`; for example `trrandom.type` uses
`RandomType`. Source parameters without an enumeration omit `choices`.
Metric parameter choices use a different shape, described in Part F.

#### CLI

Run from the repo root with its environment:

```bash
.venv/bin/python -m silk.elements.schema --pdk sky130 -o pdk_schema.json
.venv/bin/python -m silk.elements.schema --elements -o elements_schema.json
```

Omit `-o` to write JSON to standard output. `--pdk` accepts a registered name
or alias through `load_pdk`, or an existing device-details YAML path through
`BasePDK`. `--indent` defaults to `2`. Exporting a catalog does not load
simulator models.

## Part B — BusSpec contract

`silk.circuit.bus_spec.BusSpec` is frozen metadata for a declared bus port
or internal bus net. Module bus declarations store it in `Module._buses`.
Its fields are `name`, `indices`, `direction`, `role`, `domain` and `naming`.
`direction` is a `PortDir`; `role` is a `PortRole` or `None`; `domain` is a
voltage-domain name or `None`, rather than a live `VoltageDomain` object.

Bus declarations normalize indices to a nonempty, ascending, contiguous tuple
of integers. `naming` defaults to `bracketed`; `underscore` is also accepted.
Other naming styles raise `ValueError` when expanded.

```python
from silk.circuit.bus_spec import indices_from_bits, expand_bits

assert indices_from_bits(4) == (0, 1, 2, 3)
assert indices_from_bits(range(8, 12)) == (8, 9, 10, 11)
assert indices_from_bits(slice(0, 4)) == (0, 1, 2, 3)
assert expand_bits('count', 2) == ('count[0]', 'count[1]')
assert expand_bits('dbg', 2, naming='underscore') == ('dbg_0', 'dbg_1')
```

`spec.width` returns `len(spec.indices)` and `spec.expanded_names()` returns
the tuple of scalar pin names. `indices_from_bits` rejects zero or negative
widths, empty ranges/slices and steps other than one with `ValueError`;
unsupported input types raise `TypeError`. Direct `BusSpec` construction
stores the supplied fields without this normalization.

For JSON input, see `ports[].bits` and `bus_nets` in
[`codegen-contract.md`](codegen-contract.md).

## Part C — VoltageDomain contract

```python
from silk import VoltageDomain

avdd = VoltageDomain('avdd', nominal=1.8, tolerance=0.1)
io = VoltageDomain('io', nominal=3.3, min=3.0, max=3.6)
core = VoltageDomain('core', nominal=0.9, switchable=True)
assert round(avdd.v_min(), 2) == 1.62
assert round(avdd.v_max(), 2) == 1.98
assert io.contains(3.3)
assert core.v_low() == 0.0
```

`VoltageDomain` is a mutable dataclass with required `name` and `nominal`
fields. `min`, `max` and `tolerance` default to `None`; `switchable` defaults
to `False`. `switchable=True` marks a rail that may be powered down while
other rails remain live for domain-aware lint checks.

| Method | Result |
|---|---|
| `v_min()` | Explicit `min`, otherwise `nominal * (1 - tolerance)`, otherwise `nominal` |
| `v_max()` | Explicit `max`, otherwise `nominal * (1 + tolerance)`, otherwise `nominal` |
| `v_low()` | `0.0`, the disabled voltage |
| `would_stress(rating)` | Whether `v_max()` exceeds the rating, excluding equality within relative tolerance `1e-9` |
| `contains(voltage)` | Whether the voltage is between `v_min()` and `v_max()`, inclusive |
| `is_compatible(other)` | Same object, or matching names with overlapping ranges |

Explicit bounds take precedence separately for each endpoint. Construction
does not validate consistency among the fields. Dataclass equality compares
field values; it is distinct from physical-rail compatibility.

## Part D — Corner, output_dir, other spec fields

### D1. `corner`

```python
from silk import Corner

assert Corner.TT == 'tt'
assert tuple(str(c) for c in Corner.standard()) == ('tt', 'ff', 'ss', 'sf', 'fs')
```

`Corner` is a string enum for five common process corners: typical, fast/fast,
slow/slow, slow-NMOS/fast-PMOS and fast-NMOS/slow-PMOS. The selected PDK's
manifest defines its actual corner set; the enum is a convenience, not the
complete list for every PDK. Simulation accepts corner strings or enum members.

For a standalone spec, `Generator.generate` emits
`ckt.simulate(corner=spec.corner)` with the requested value (default `tt`).
For a subcircuit spec (`ports` is present), the generated block does not
simulate itself; select the corner on its testbench. A PDK-containing netlist
uses the manifest's corner includes, or its library section when no explicit
includes are declared. Ideal-only circuits omit PDK model includes.

### D2. `output_dir`

`CircuitSpec.output_dir` defaults to `./sim` and becomes the `Module`
constructor's `dir_folder`. **The constructor creates this directory,
including missing parents.** No separate `os.makedirs` call is needed.

The netlist name is `<circuit_name>.cir`. Output mode defaults to `new`,
which puts each run in a fresh child folder below the configured directory.
Top-level `generate_netlist()` returns the actual deck path; a subcircuit
returns SPICE text without writing a file. Use
`mod.set_output_options(mode='overwrite')` to reuse the configured folder.
Simulation writes results and simulator logs as well as the deck; artifact
names depend on the backend. Read `mod.results_file` for the current result
path rather than assuming a fixed location. See [output options](api-reference.md#testbench-output-directories-and-run-labels).

### D3. Other CircuitSpec fields

`silk.codegen.generator.CircuitSpec` has the following fields. Container
defaults are fresh for each instance.

| Field | Default | Purpose |
|---|---|---|
| `circuit_name` | Required by the dataclass; `from_dict` defaults to `my_circuit` | Circuit identity and netlist basename |
| `pdk` | `sky130` | Registered PDK selection |
| `output_dir` | `./sim` | Output directory |
| `corner` | `tt` | Standalone simulation corner |
| `voltage_domains` | `{}` | Named `VoltageDomain` objects |
| `ports` | `None` | Absent means standalone; a map, including `{}`, means subcircuit |
| `bus_nets` | `[]` | Internal bus declarations |
| `devices` | `[]` | PDK devices |
| `ideal_elements` | `[]` | Process-independent passive elements |
| `sources` | `[]` | Independent, controlled and behavioral sources |
| `simulation` | `{}` | Standalone analysis description |
| `submodules` | `[]` | Imported child-module placements |
| `variables` | `[]` | Module variable declarations |

`Generator.generate` accepts either a dictionary parsed by
`CircuitSpec.from_dict`, or a constructed `CircuitSpec`. See
[`codegen-contract.md`](codegen-contract.md) for the nested input shapes and
validation errors.

## Part E — Testbench source catalog export

```python
from silk.sources import export_source_schema, source_schema

sources = export_source_schema()
voltage = source_schema('VoltageSource')
assert [p['name'] for p in voltage['parameters']] == ['dc', 'ac']
assert voltage['waveform_exclusive'] is True
assert sources['_meta']['unschematized_sources'] == []
```

`export_source_schema(include_metadata=True)` returns a map keyed by class
name. `_meta` includes `format_version: "1.0"` and
`unschematized_sources`, which lists source classes without metadata.
`source_schema(class_name, meta=None)` returns one entry; an unknown class
name raises `KeyError`. Names, defaults and required flags are reflected
from constructors; units, symbols, pin names and waveform mappings come
from `silk/sources/source_metadata.yaml`.

Each entry has `name`, `category`, `symbol`, `unit`, `pins`, `parameters`,
`waveforms`, `waveform_exclusive` and `code_template`. Scalar and waveform
parameters include `name`, `unit`, `default` and `required`; some also carry
`kind` or enum `choices`. `required` distinguishes an argument without a
default from one whose default is `null`.

Independent voltage/current sources provide `dc`, `ac` and waveform choices.
Their `symbol` values are `VSource` and `ISource`, respectively.
Each waveform entry contains `class`, `parameters`, `kwarg` and
`code_template`. Amplitude units follow the source (`V` or `A`); other
parameters can have units such as `s`, `Hz`, `deg`, or `null`.

The device catalog's `vsource`/`isource` entries in Part A expose this same
waveform information using declarative factories and `p`/`n` pins. The source
catalog uses constructor names and `positive`/`negative` pins.

### Dependent (controlled) sources

The catalog includes `VoltageControlledVoltageSource` (VCVS),
`VoltageControlledCurrentSource` (VCCS), `CurrentControlledVoltageSource`
(CCVS) and `CurrentControlledCurrentSource` (CCCS). They have no `ac`
parameter and an empty `waveforms` map. Their templates call the classes
directly.

All use `out_pos`/`out_neg`; voltage-controlled sources also use
`ctrl_pos`/`ctrl_neg`. Parameters are `gain` (VCVS: V/V; CCCS: A/A),
`gm` (VCCS: A/V), or `transresistance` (CCVS: V/A).

CCVS/CCCS additionally expose `references`, containing
`{name: "ctrl_vname", kind: "source", required: true, default: null}`.
This names a controlling voltage source's branch current, not a pin or scalar
parameter. Its template placeholder is `{ref_ctrl_vname}`. Other source
classes omit `references`. The generator validates references against eligible
sources in the same spec; see [reference eligibility](codegen-contract.md#eligible-and-ineligible-ctrl_vname-targets).

### Behavioral sources

`BehavioralVoltage` and `BehavioralCurrent` have `positive`/`negative` pins,
empty `parameters` and `waveforms`, and an `expressions` list containing
`{name: "expression", kind: "expression", required: true, default: null}`.
Their symbols are `BV`/`BI`, and units are `V`/`A`.
Source classes that do not declare expressions omit `expressions`; they do
not publish it as `null`.

The expression is a SPICE behavioral expression, not a Python callable.
The template uses `{expr_expression}`; substitute a quoted Python string.
The expression body is passed to SPICE rather than resolved as circuit-spec
references. A behavioral voltage source may control CCVS/CCCS; a behavioral
current source may not. Controlled and behavioral sources also support arrays;
see [source arrays](codegen-contract.md) for their input and reference rules.

Run the source catalog CLI from the repo root:

```bash
.venv/bin/python -m silk.sources.schema
```

It writes JSON to standard output.

## Part F — Metric catalog export

```python
from silk import export_metric_schema
from silk.signals import metric_schema
from silk.metrics import gain_at_freq

metrics = export_metric_schema()
assert metrics['_meta']['format_version'] == '2.0'
gain = metric_schema(gain_at_freq)
assert gain['unit'] == 'V/V'
assert gain['signal_inputs'] == [{'name': 'sig', 'kind': 'frequency_response'}]
```

`export_metric_schema(include_metadata=True)` imports `silk.metrics` and
returns schemas for registered scalar metrics. Functions registered with
`outputs=` return result objects and are omitted from this catalog.
`metric_schema(fn)` describes one decorated function; an undecorated function
raises `TypeError`.

Each metric has `name`, `domain`, `unit`, `signal_inputs` and `parameters`.
`domain` is `tran`, `ac`, `dc` or `noise`. `_meta.metric_count` counts exported
entries; query it rather than relying on a fixed catalog count.
`_meta.source_type` is `"metrics"`.

When storing metric outputs with `store_report`, unit inference uses
case-sensitive metric-name suffixes. For dotted result outputs, pass explicit
units, for example `units={'eye.height': 'V', 'eye.width': 's'}`.

Names, scalar types, choices and signal-input kinds are reflected from
function signatures. The decorator supplies analysis domain, return unit and
parameter units. Read returned units literally: `gain_at_freq`, for example,
returns linear V/V. `''` means dimensionless; `[y]` and `[x]` mean the input's
axis units; expressions such as `[y]*sqrt([x])` describe derived units.

### Parameter types and choices

Each scalar parameter contains:

| Field | Meaning |
|---|---|
| `name` | Python keyword argument |
| `unit` | Declared unit, or `null` |
| `default` | Default value, or `null` when required |
| `required` | No Python default exists |
| `type` | `float`, `int`, `str`, `bool` or `complex_points` |
| `optional` | Annotation permits `None` |
| `choices` | Literal values accepted by the annotation, or `null` |

`optional` and `required` are independent. For example,
`spectral_power.window` defaults to `hann` and accepts `None` for no window.
Use `choices` as a menu where present and additionally allow `null` when
`optional` is true. Source enum choices contain value/label objects; metric
choices are plain values.

Some parameter menus are:

| Parameter | Choices |
|---|---|
| `error_norm.kind`, `relative_error_norm.kind` | `max`, `rms`, `mean_abs` |
| `inl_max_lsb.method` | `endpoint`, `best_fit` |
| `spectral_power.window` | `hann`, `hamming`, `blackman`, `rectangular`; also `None` |

The type vocabulary includes `complex_points` for the `ComplexPoints`
annotation, an array-like sequence of complex points. Result-object metrics
can use it in Python even though they are omitted from the scalar catalog.
`ComplexPoints` describes inputs without coercing them. Built-in scalar types
are matched independently by identity, so `bool` is distinct from `int` and
subclasses are not accepted. Mixed-type `Literal` annotations and unions of
multiple non-`None` types are unsupported.

Unsupported scalar annotations raise `UnsupportedParamTypeError`
(a `TypeError`) at `@metric` decoration time. A bare `Signal` annotation
without a kind raises `UnsupportedSignalKindError` (also a `TypeError`).
Unsupported scalar return annotations raise `NonScalarMetricError`;
result-object metrics must declare their scalar `outputs`.

### Schema v2 — signal-input kinds

Check `_meta.format_version` before consuming a metric catalog.
In version `2.0`, `signal_inputs` is an ordered array of `{name, kind}`
objects. Version `1.0` used an array of names; a migrating consumer reads
`entry['name']` instead of treating `entry` itself as a name.

`_meta.signal_kinds` maps each kind to a `produced_by` list:

| Kind | `produced_by` | Input meaning |
|---|---|---|
| `transient` | `["tran"]` | Time trace |
| `frequency_response` | `["ac"]` | Small-signal frequency response |
| `noise_density` | `["noise"]` | Noise density in V/√Hz or A/√Hz |
| `any` | `["tran", "ac", "dc", "noise"]` | Axis-independent input; each metric still defines its sample requirements |
| `code_transitions` | `[]` | ADC transition-level table |
| `power_sweep` | `[]` | Output power versus input power, both in dBm |
| `reference_waveform` | `[]` | User-authored target waveform |

The catalog uses `tran` in `produced_by`; a testbench spec uses `transient`
in `analyses[].kind`.

For a tool that binds analysis outputs directly, offer a metric only when
**every input kind has a producer among the analyses the tool can author**.
`domain` groups metrics; admission depends on input kinds rather than this
label. An empty `produced_by` means an analysis cannot supply the input
directly; it may still be constructed in Python. For example,
`extract_code_transitions` derives a
transition table from transient signals.

This excludes `error_norm`, `relative_error_norm`, `p1db` and the
transition-table metrics from direct probe binding. `generate_testbenches`
raises `TestbenchSpecError` for such bindings. It also checks named signal
slots, parameter choices and each input kind's producers against the test's
declared analyses; see the
[error contract](codegen-contract.md#error-contract).

Kinds describe input slots, not a stored tag on a `Signal`. Runtime guards
can reject an incompatible axis or sample type, but cannot always distinguish
a real-valued AC response from a noise density. Consumers must use the
schema's producer information when wiring analysis results to metric inputs.

Run the metric catalog CLI from the repo root:

```bash
.venv/bin/python -m silk.signals.schema
```

It writes JSON to standard output.

## Limitations

`Module.export_schema()` exports scalar port names and directions and element
instance names. It does not export bus groupings, port voltage domains, full
connectivity, or component parameter values. A bus appears as separate scalar
pins. Its template illustrates port declaration, not reconstruction of the
whole module.

Keep the generator spec as the saved document when these details must survive
a save/load cycle. See [`codegen-contract.md`](codegen-contract.md) for that
input format.
