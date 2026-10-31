# Generated device and element classes

The reference lists 36 classes that are not documented one by one: the 33
sky130 device classes (`nmos1p8`, `res_hpo`, `cap_mim1`, ...) and the 3 ideal
elements (`resistor`, `capacitor`, `inductor`).
The ideal catalog also contains source entries for schema export; construct
sources with `silk.sources`.
A shared factory builds these `AnalogElement` subclasses from YAML. The catalog
defines each class's pins, parameters, bounds and defaults.

Three pieces build and expose the classes:

* {py:class}`silk.elements.ElementLibrary` builds the ideal elements from
  `silk/elements/analog_devices.yaml`. The ready-made instance is
  `silk.elements.elements`.
* {py:func}`silk.pdk.load_pdk` builds the device classes of a PDK from the
  device-details YAML that the PDK's manifest names. The result is a PDK
  object ({py:class}`silk.pdk.BasePDK`, or a custom subclass) whose attributes
  are the device classes.
* The device-class factory, `silk.elements.factory.make_device_class`, turns one
  catalog entry into one class. Both of the above call it. It is internal and
  not part of the public reference.

## How a catalog becomes classes

A PDK is registered by a manifest, `silk/pdk/data/<name>.pdk.yaml`, which names
the PDK's model library, corners, unit convention and (`device_details:`) the
device-details YAML. The registry scans `silk/pdk/data/*.pdk.yaml` the first
time it is used. {py:func}`silk.pdk.available` lists the canonical names and
{py:func}`silk.pdk.manifest` returns one parsed manifest:

```python
from silk.pdk import available, load_pdk

available()                      # ['gf180mcu', 'sg13g2', 'sky130']
pdk = load_pdk('sky130')         # an alias such as 'skywater130' also works
pdk.model_names[:3]              # ['nmos1p8', 'nmos1p8_lvt', 'nmos3p3_native']
```

`load_pdk` resolves the library paths against `PDK_ROOT` but does not look for
the files: the catalog ships inside silk, so loading, netlist generation and
behavioural simulation work with no models installed. A simulation that needs
the models fails later with a `SimulationError` naming the missing library.

For each entry of the device-details YAML, `BasePDK` records the entry for the
legacy `add_element` path ({py:class}`silk.pdk.PDKDevice`) and calls the
factory to generate a class, which it stores as an attribute named after the
entry. A catalog entry looks like this (an excerpt of `nmos1p8` in
`silk/pdk/data/skywater_130_device_details.yaml`):

```yaml
nmos1p8:
  PDKDevice: sky130_fd_pr__nfet_01v8      # the model/subcircuit name in the deck
  paramList:
    m: {minval: 1, maxval: -1, resolution: 1}
    w: {minval: 0.42, maxval: -1, default: 1.0, pdk_unit: u, resolution: 0.005}
    l: {minval: 0.15, maxval: -1, default: 0.15, pdk_unit: u, resolution: 0.005}
  pinList: {d: drain, g: gate, s: source, b: bulk}
```

* `pinList` gives the pins, in the order of the model's terminals.
* `paramList` gives each parameter's bounds (`-1` means unbounded), default,
  the unit prefix the model takes (`pdk_unit`) and its manufacturing grid
  (`resolution`).
* `PDKDevice` is the model name written into the netlist. The instance-name
  prefix defaults to `XM` (a subcircuit call) and is overridden by
  `spice_prefix:`.
* Optional blocks carry junction ratings, fault-injection bounds and
  operating-point declarations; they are described in
  the {py:class}`silk.pdk.PDKDevice` fields.

The ideal catalog has the same shape with a simpler parameter list
(`pinList: [p, n]`, `paramConstraints: [{name: value, default: 1000, unit: Ohm}]`),
a `spice_prefix` for every entry, and optional `pinAliases` and `paramAliases`
(`pos` for `p`, `resistance` for `value`). Device, pin and parameter names must
be valid Python identifiers; otherwise loading raises `PDKError` (PDK catalogs)
or `CircuitError` (ideal catalogs), naming the file and the key.

The generated class is a subclass of {py:class}`silk.elements.AnalogElement`
whose `__module__` is `silk.elements.factory`. Each call to `load_pdk` builds
its classes again, so two loads of one PDK hold distinct class objects.

## The constructor contract

The instance name is the only positional argument. Pins and parameters are
keyword-only:

```python
from silk.pdk import load_pdk

pdk = load_pdk('sky130')
pdk.nmos1p8('m1', d='out', g='in', s='0', b='0', w=4.0, l=0.5)

try:
    pdk.nmos1p8('m1', 'out', 'in', '0', '0')
except TypeError:
    print('Pins must be keyword arguments.')
```

Every pin must be given, and what is given must be one node token: a net name
or a {py:class}`silk.circuit.Net`. Use the string `'0'` for SPICE ground;
Avoid `'gnd'`: ngspice treats it as a ground alias, behavioral simulation
recognizes lowercase `'gnd'` as global ground, and silk refuses it in Xyce
decks. For a reusable ground port, use `vss` and wire it to `'0'` in the parent.

A missing pin, a parameter without a default, or an unknown keyword raises
`TypeError`. An invalid instance name raises `CircuitError`. A pin value of
the wrong type raises `TypeError`; a malformed net name raises `ValueError`.
Pin-value errors name the instance and pin. Parameters you leave out take the catalog
default (for a PDK parameter with no explicit default, its minimum is used).
Inside a `with module:` block the new instance adds itself to the
module. The ideal elements also accept their catalog's aliases
(`elements.resistor('r1', pos='out', neg='0', res=1e3)`); giving a name and its
alias together raises `ValueError`.

PDK parameter values are checked against the catalog when the netlist is emitted
(`to_spice()`): a caller-supplied number outside the bounds, or a fractional
value for a dimensionless parameter such as `m`, raises `CircuitError` naming the
device. A single-line string value is passed through to the netlist as a
SPICE literal; line breaks are rejected. The
legacy `add_element` path keeps its older behaviour: it prints a warning and
emits the value.

The class attribute of a parameter is an accessor that builds a
`Variable` already configured from the catalog's bounds and resolution:

```python
from silk.pdk import load_pdk

pdk = load_pdk('sky130')
pdk.nmos1p8.w.variable(name='w_in', value=2.0)    # bounds (0.42, None), resolution 0.005
```

## Units

W and L are written in micrometres on every shipped PDK. Every shipped manifest
sets `user_units: u`, meaning the circuit author writes micrometres, and silk
converts each caller-supplied number to the unit that PDK's models take (the
`pdk_unit` of the parameter). The same call therefore emits a different number
on a PDK whose models take metres:

```text
sky130   nmos1p8   w=1.0, l=0.5   ->  XMm1 o i g g sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.5
gf180mcu nmos3p3   w=1.0, l=0.5   ->  XMm1 o i g g nfet_03v3 m=1 w=1u l=500n
```

Catalog defaults are already in the PDK's own unit and are never converted. To
change the convention for one load, pass `user_units=` to `load_pdk`: `'si'`
(metres), `'pdk'` (each parameter in the unit its catalog tag names, no
conversion; a bare PDK constructor defaults to this and prints a notice to
standard error) or a prefix token (`'u'`, `'n'`, `'p'`, `'m'`, `'base'`). Do not
rewrite `w=4 l=0.5` in an emitted sky130 netlist as SI: those numbers are
micrometres. {py:func}`silk.pdk.device_geometry_um` returns a device's geometry
in micrometres whatever the PDK's unit.

## Listing what a PDK offers

Nothing needs a simulator or installed models:

```python
from silk.pdk import load_pdk
from silk import export_schema

pdk = load_pdk('sky130')
len(pdk.model_names)             # 33 (gf180mcu: 61, sg13g2: 4)

schema = export_schema(pdk)      # {device name: spec dict, ...} plus '_meta', '_library'
schema['nmos1p8']['pins']        # [{'name': 'd', 'description': 'drain'}, ...]
schema['nmos1p8']['parameters']  # name, default, unit, min and max of m, w and l
schema['nmos1p8']['pdk_model']   # 'sky130_fd_pr__nfet_01v8'
```

`pdk.model_names` is the list of device names, and {py:func}`silk.circuit.export_schema`
returns each device's pins, parameters (default, unit, bounds), model name, SPICE
prefix and category as plain data. `export_schema(silk.elements.elements)` does the
same for the ideal elements, including the source entries `vsource` and `isource`.
Catalog loading and schema export need no installed models.
`.venv/bin/python -m silk.pdk.validate --pdk NAME` reads the installed model
library to vet pin lists and emission; add `--simulate` for a device smoke test.
`silk.pdk.manifest(name)` returns the parsed manifest of a PDK.

## Where to read more

* The catalogs themselves: `silk/pdk/data/*.pdk.yaml` (manifests),
  `silk/pdk/data/skywater_130_device_details.yaml`,
  `silk/pdk/data/gf180mcu_device_details.yaml`,
  `silk/pdk/data/sg13g2_device_details.yaml` and
  `silk/elements/analog_devices.yaml`.
* `docs/adding-a-pdk.md` for the manifest and catalog formats and how to
  register a new PDK.
* The loaded PDK and its records: {py:func}`silk.pdk.load_pdk`,
  {py:class}`silk.pdk.BasePDK`, {py:class}`silk.pdk.PDKDevice`,
  {py:class}`silk.pdk.ParameterConstraint` and
  {py:class}`silk.elements.AnalogElement`.
