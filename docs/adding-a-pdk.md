# Adding a PDK to silk

Add a process design kit (PDK) by describing its devices and model libraries
in YAML. The examples below use the shipped manifests in `silk/pdk/data/`,
with sky130 as the starting point.

## 1. Overview

A PDK integration needs two YAML files, with an optional Python class:

1. A **manifest** — `silk/pdk/data/<name>.pdk.yaml` — that declares the PDK's
   name, where its device catalog and model library live, and its process
   corners.
2. A **device-details YAML** — the existing per-device catalog format (one
   entry per device: which real `.subckt`/`.model` it maps to, its pins, its
   parameter ranges).
3. An optional `BasePDK` subclass for custom behavior. No shipped PDK needs
   one: each manifest describes its corner-file layout for `BasePDK`.

Drop the manifest (and device-details YAML, if it isn't already colocated)
into `silk/pdk/data/`, and the PDK is **registered**:
`silk.pdk.available()` lists it and `load_pdk('<name>')` loads it. There is no
separate registration step — the registry (`silk/pdk/registry.py`) auto-scans
`silk/pdk/data/*.pdk.yaml` on first use, cached after that.

Add a complete entry under the canonical PDK name in
`silk/circuit/data/region-thresholds.default.yaml` as well. Every registered
PDK must have an explicit shipped region-policy set;
`tests/test_region_thresholds.py` checks this coverage. These thresholds are
designer policy, separate from device model parameters. See
[operating-point.md](operating-point.md#project-and-customer-threshold-files)
for the schema and customer overrides.

Registration is catalog availability, not a release support tier. For 0.2.0,
sky130 is the only release-validated simulation PDK; the registered GF180MCU
and SG13G2 catalogs are preview integrations. Promoting a PDK requires a gated
real-model profile covering its major families, result readback, and a
non-default corner in addition to the authoring checks below.

## 2. The manifest (`<name>.pdk.yaml`)

Field reference, from `silk/pdk/registry.py` and `silk/pdk/_loader.py`:

| Field | Required | Default | Meaning |
|---|---|---|---|
| `name` | yes | — | Canonical PDK name, must be unique across all manifests. |
| `aliases` | no | `[]` | List of alternate names that resolve to the same manifest. |
| `class` | no | `silk.pdk.basepdk:BasePDK` | `module:ClassName` of the `BasePDK` subclass to instantiate. |
| `device_details` | yes | — | Path to the device-details YAML, **relative to the manifest's own directory**. |
| `user_units` | no | `pdk` | The unit the circuit AUTHOR writes in: `pdk` (whatever this PDK wants), `si`, or a prefix such as `u` ("I write microns"). Converted per the `pdk_unit:` each parameter declares. See [Units](#units). |
| `vendor` | no | — | Free-text vendor name, informational. |
| `library` | yes | — | See below. |

`library` block:

| Field | Required | Meaning |
|---|---|---|
| `lib_path` | yes | Path to the model library file, **relative to `$PDK_ROOT`** (or an override root — see §5), using the directory layout installed by [pdksetup](https://github.com/levantlabs/pdksetup). |
| `default_corner` | yes | Corner used when none is passed at simulate time. Should be one of the keys in `corners` — note this is **not validated** at load time; a typo surfaces only when the corner is used. |
| `extra_includes` | no | List of extra cell files (relative to the root), `.include`d after the corner `.lib` line. Use this when a device's model lives in a file the corner lib's own include chain doesn't pull in. |
| `corners` | yes | Map of `corner_name -> {kind, desc}`. `kind` is one of `process`, `rc`, `statistical`, `composed` — and it is **functional**, not just descriptive: `run_corners()` builds its default sweep from the corners with `kind: process`, so mislabeling a corner silently changes the PDK's default sweep. |
| `xyce_lib_path` | no | Xyce model-library path relative to the root, replacing `lib_path` for Xyce; omitted means use `lib_path`. |
| `xyce_extra_includes` | no | Xyce replacement for `extra_includes`, relative to the root; omitted uses `extra_includes`, while `[]` means none. |
| `statistical` | no | Monte Carlo sections, switches and recorded parameters; see [statistical declarations](#optional-librarystatistical-block-monte-carlo). |
| `osdi` | no | Root-relative compiled-model paths; see [compiled Verilog-A models](#compiled-verilog-a-models-osdi). |

Each `library.corners.<name>` entry accepts:

| Field | Required | Meaning |
|---|---|---|
| `kind` | no | Corner category; use `process` for corners in the default sweep. |
| `desc` | no | Description of the corner. |
| `includes` | no | List of `{file, section}` entries replacing the single library line; files resolve against the root and an omitted section emits a plain `.include`. |
| `xyce_includes` | no | Xyce-specific replacement for `includes`, using the same entry shape and root-relative paths. An empty list falls back to `includes`; it does not suppress them. |


**Corners are author-declared, not derived.** `silk` does not infer valid
corners from the library file — whatever you list under `corners` is the
authoritative set for that PDK, reported by `pdk.corners`.

Trimmed excerpt from the real `sky130.pdk.yaml`:

```yaml
name: sky130
aliases: [skywater130, skywater_130]
device_details: skywater_130_device_details.yaml
user_units: u
vendor: SkyWater
library:
  lib_path: "sky130A/libs.tech/ngspice/sky130.lib.spice"
  default_corner: tt
  extra_includes:
    - "sky130A/libs.ref/sky130_fd_pr/spice/sky130_fd_pr__cap_vpp_11p5x11p7_m1m2_noshield.model.spice"
  corners:
    tt: { kind: process, desc: "typical" }
    ff: { kind: process, desc: "fast-n fast-p" }
    ss: { kind: process, desc: "slow-n slow-p" }
    mc: { kind: statistical, desc: "Monte Carlo sampling (one random draw per run)" }
    ff_hh_mm: { kind: composed, desc: "fast-n fast-p, high R, high C, mismatch" }
    # ... 51 corners total for sky130 (process/rc/statistical/composed)
```

Omitting `class` uses `BasePDK`, as all shipped manifests do. A custom
subclass can be selected with a `module:ClassName` entry, for example:

```yaml
class: my_pdk.devices:CustomPDK
```

The module must be importable and `CustomPDK` must accept the `BasePDK`
constructor arguments.

`extra_includes` is exactly how sky130's vpp capacitor is made simulatable:
its model lives in a cell file the `tt` corner's own `.include` chain never
pulls in, so it's listed here and `.include`d unconditionally alongside the
corner `.lib` line (`PySpiceNetlister.generate_pdk_include` in `silk/circuit/spice_netlister.py`).

### Optional `library.statistical:` block (Monte Carlo)

A PDK that supports SPICE Monte Carlo declares which library sections and switches
produce statistical variation. silk hardcodes none of it; a PDK without the
block has `pdk.statistical is None` and SPICE Monte Carlo on it raises
`NotSupportedError`.

```yaml
library:
  statistical:
    mismatch:
      sections: { tt: tt_mm, ss: ss_mm }   # corner -> library section with mismatch
      switch: mc_mm_switch                 # .param the section turns on
      record:                              # model params to record per device,
        nmos1p8: [vth0, toxe, voff]        # keyed by device-details YAML names
    process:
      section: mc                          # one draw per sample, all devices
      corners: [tt]                        # corners it is valid at
      switch: mc_pr_switch
      includes:                            # PDK-relative; resolved to absolute
        - "sky130A/libs.tech/ngspice/parameters/critical.spice"
```

`silk.pdk.statistical.require_library_sections(pdk, sections)` raises
`MissingLibrarySectionError` (a `PDKError` and a `ConfigurationError`) naming
each section the installed library lacks; `silk.doctor` reports the same
sections as `[warn]`. Set `min_open_pdks:` to add a hint to the message (the
check itself is by section, never by version).

## 3. The device-details YAML

The device catalog describes the callable device factories. Each entry declares a `PDKDevice` (the real `.subckt`/`.model` name),
a `pinList` (in the subckt's real positional terminal order), and a
`paramList` of parameter constraints. See `docs/pdk-validation.md` for the
validation workflow, and
`silk/pdk/data/skywater_130_device_details.yaml` for a complete real example.
Use those entries as a starting point and validate your catalog (§6).

**Catalog naming rule:** device keys,
`pinList` / `pinAliases` keys and `paramList` / `paramAliases` keys must be
Python identifiers that are not keywords. For example, use `nfet_01v8`,
not `nfet-01v8`, and `w_min`, not `w-min`; `class` and `1device` are invalid.
`load_pdk` / `BasePDK` raise `PDKError` at file load with the file, device and
bad key in the message. `python -m silk.pdk.validate` reports that refusal
and exits 1, listing every invalid catalog name in one pass. Nothing translates these names downstream. The separate
`PDKDevice` field still carries the real model name unchanged; terminals
remain positional. Ideal-element `paramConstraints[*].name` values follow
the same rule. Check a name independently with
`silk.utils.require_python_identifier(value, context)`.

### Junction ratings (`junction_ratings` + `junction_ratings_provenance`)

A device may carry a `junction_ratings:` block — the per-terminal-pair
maximum voltages the STR-001 lint rule (`silk.lint.rules.stress`) scores
simulated waveforms against:

```yaml
nmos1p8:
  # Other catalog fields omitted.
  junction_ratings:
    Vgs_max: 1.95
    Vds_max: 1.95
    Vgd_max: 1.95
    Vbs_max: {min: -1.95, max: 0.3}   # signed range: asymmetric published limits
    Vbd_max: 1.95
  junction_ratings_provenance:
    source: >-
      $PDK_ROOT/skywater-pdk/docs/rules/device-details/nfet_01v8/index.rst
      ("Operating Voltages where SPICE models are valid"): VDS 0 to 1.95 V,
      VGS 0 to 1.95 V, VBS +0.3 to -1.95 V. Vbs_max is the signed range
      {min: -1.95, max: 0.3} (Vbs = V(b) - V(s)).
    heuristic:
      Vgd_max: >-
        = Vds_max; no separate Vgd operating limit is published ...
      Vbd_max: >-
        = Vds_max; no separate Vbd operating limit is published ...
```

Keys are `<pair>_max` in volts: `Vgs/Vds/Vgd/Vbs/Vbd` for a `d/g/s/b`
device, `Vak` for an `a/k` diode, `Vbe/Vbc/Vce` for a `c/b/e[/s]` BJT.
Leave the block out for a device the foundry publishes no ratings for
(passives) — an absent block means "unrated", and STR-001 skips the device.

A value takes one of two forms, and the loader normalises both to a signed
`(lo, hi)` range (`PDKDevice.junction_ranges`) that STR-001 scores the
**signed** pair voltage against:

- a bare number `m` — shorthand for the symmetric range `[-m, +m]`. This is
  the right form wherever the foundry states one magnitude;
- a mapping `{min: lo, max: hi}` — a signed range, for a pair whose
  published limits differ by direction. **Sign convention:** the pair named
  `Vxy` is `V(x) - V(y)` — `Vbs = V(b) - V(s)`, `Vgs = V(g) - V(s)`, `Vak =
  V(a) - V(k)`, `Vbe = V(b) - V(e)` — so sky130's NMOS "VBS = +0.3 to
  -1.95 V" is `{min: -1.95, max: 0.3}`: the body may sit at most 0.3 V above
  the source and 1.95 V below it. Mind the device polarity when reading the
  foundry's sentence: a PMOS n-well body normally sits *above* its source,
  so sky130's PMOS "VBS = -0.1 to +1.95 V" is `{min: -0.1, max: 1.95}` —
  +1.95 is its reverse-bias end and -0.1 its forward one, the mirror image
  of the NMOS entry. Write `min` as the lower (more negative) end
  regardless of the order the foundry's sentence lists them in;
  `{min: 0.3, max: -1.95}` is rejected by the validator.

**The convention — read before writing a number.**

1. **A rating is the foundry's stated *operating* limit for that pair, and
   nothing else.** silk adds no margin: STR-001 compares the signed pair
   voltage against `(lo, hi)` — `|V_pair| > m` for the magnitude form — so
   the number you write is the number that fires.
   Where the foundry's figure already carries its own tolerance (SkyWater's
   1.95 V for a 1.8 V device, GF180MCU's "nominal + 10 % supply overshoot"),
   that figure is the rating. Do not add an author-chosen margin.
2. **Never a breakdown voltage.** `BVDSS`, `BVox`, a diode model's `bv`,
   "points of failure" in a reliability section — these are where the device
   *dies*, not where it may operate, and publishing one as a rating would let
   STR-001 wave through a circuit the foundry says is broken.
3. **Cite it.** `junction_ratings_provenance.source` names the document (a
   path under `$PDK_ROOT`, a public URL, or a model-file parameter such as
   the `VGS_MAX` SOA parameters in an IHP `.lib`) and quotes the statement.
   `source` covers every key not listed under `heuristic`.
4. **Derive only what is not published, and say how.** When the foundry
   states no number for a pair, list the key under `heuristic:` with the rule
   its value was derived by. The rule text must *open* with one of two forms
   the validator can re-check against the numbers — `= <key>_max` (equal to
   another rating of the same device) or `<factor> x nominal <V> V` — and may
   continue with the justification. The shipped catalogs use two derivations
   worth copying: `Vgd_max = Vds_max` and `Vbd_max = Vds_max` (gate/body at
   source potential with the drain at its maximum — the normal off-state).
   `= <key>_max` means the same *signed range*, both ends — a magnitude
   cannot be claimed equal to an asymmetric range — and `<factor> x nominal`
   derives a magnitude, so it can only vouch for the symmetric form.
5. **A published asymmetric range is stored as one.** `VBS +0.3 to -1.95 V`
   is `Vbs_max: {min: -1.95, max: 0.3}`, not the reverse-bias magnitude
   1.95 — that shorthand would admit a body forward-biased by 1.95 V when
   the foundry allows 0.3. Restate the range in `source` in the signed
   convention above. Where the two directions genuinely share one limit,
   the bare magnitude is the form to use; a signed range with `min == -max`
   is the same thing spelled longer.

`python -m silk.pdk.validate --pdk <name>` checks numeric ranges and
provenance declarations mechanically:
`unsourced_rating` (a value with no source and no rule), `bad_provenance` (a
rule that does not open with a recognised form, refers to a pair the device
does not rate, or disagrees with the numbers it claims to derive) and
`rating_convention` (a magnitude that is not a positive finite number, or a
signed range that is not finite at both ends with `min < max`). It cannot
check that a citation is *true* — that is the author's job, and the
device characterization.

### Operating-point device parameters

To support `Module.save_device_params`, declare the parameters and internal
instance name verified against the installed model, and the simulator backends
you tested:

```yaml
nmos3p3:
  # Other catalog fields omitted.
  op_params: [id, vgs, vds, vbs, vth, vdsat, gm, gds, gmbs, cgg, cgs, cgd, capbd, capbs]
  op_internal_name: m0
  op_backends: [ngspice]
  polarity: n
```

All three `op_params`, `op_internal_name` and `op_backends` must be nonempty
for the device to be eligible for saves. Missing declarations skip that device;
a design with no eligible devices produces a `UserWarning`.

`op_params` must be a subset of `id`, `vgs`, `vds`, `vbs`, `vth`, `vdsat`,
`gm`, `gds`, `gmbs`, `cgg`, `cgs`, `cgd`, `cgso`, `cgdo`, `cgbo`, `capbd`,
`capbs`. Unknown parameters or backend names raise `ValidationError` at load
time. Declare only parameters you have read back successfully: an unavailable
save vector can prevent a simulator from producing results.

`op_internal_name` is the literal internal instance name from the model's
`.subckt`, not something silk can infer from its model name. sky130's
`nfet_01v8` uses `msky130_fd_pr__nfet_01v8`; GF180MCU's `nfet_03v3` uses
`m0`. Inspect the model and verify a one-parameter probe before declaring the
full list. The decks and logs in `tests/fixtures/op_point/gf180mcu/` show
such checks.

Test each backend separately before listing it in `op_backends`. Xyce uses
`N(<path>:PARAM)` and `ID(<path>)` on a single-point DC solve; see
`tests/fixtures/op_point/xyce_probe_available.cir`. Its library compatibility
also needs checking independently of the device address.

`polarity: n` or `polarity: p` independently declares the channel type used by
`SpecReport.op_table()`. It does not require the other three fields. See
[operating-point.md](operating-point.md) for save requests and table behavior.

## 4. When you need a `BasePDK` subclass

**No shipped PDK requires a subclass.** GF180MCU
(`silk/pdk/data/gf180mcu.pdk.yaml`) ships with **no `class:` key at all** — two
YAML files and nothing else — and drives ngspice end to end.

The default `class` is `silk.pdk.basepdk:BasePDK`: device-details parsing, unit
handling, parameter validation and netlist emission all come for free.

### What silk expresses

For a deck that needs PDK models, with no per-corner `includes:`, silk emits:

```spice
.lib "<library_location>" <corner>
.include "<extra_include>"
```

The `.include` line repeats for each extra include, or is absent when none
are declared. That fits a PDK whose models are one file with a `.LIB <corner>` section per
corner — sky130. A corner's `includes:` list covers everything else, and both
GF180MCU and SG13G2 use it:

* **Several sections of the *same* file** — GF180MCU: `.LIB typical` pulls in
  the FETs only, while resistors, MIM caps, MOS caps, diodes and BJTs each
  live in their own top-level section (`res_typical`, `mimcap_typical`,
  `diode_typical`, `bjt_typical`, …), so its `typical` corner is seven `.lib`
  lines into one file.
* **One model file per device family (or per corner)** — SG13G2, below.

### Several lib files for one corner

A corner entry may declare its own `includes:` list, and each entry becomes one
`.lib "<file>" <section>` line. This is what SG13G2 needs — its device families
live in six separate corner files:

```yaml
corners:
  typical:
    kind: process
    includes:
      - { file: "…/cornerMOSlv.lib", section: mos_tt }
      - { file: "…/cornerMOShv.lib", section: mos_tt }
      - { file: "…/cornerRES.lib",   section: res_typ }
      - { file: "…/cornerCAP.lib",   section: cap_typ }
      - { file: "…/cornerDIO.lib",   section: dio_tt }
      - { file: "…/cornerHBT.lib",   section: hbt_typ }
```

`file` resolves against `$PDK_ROOT` exactly like `lib_path`. Omit `section` and
the entry is emitted as a plain `.include` instead.

Why explicit pairs rather than deriving the section from the corner name:

* **The section name is not the corner name.** MOS says `mos_tt`, resistors say
  `res_typ`, caps `cap_typ`, diodes `dio_tt`, HBTs `hbt_typ`. No suffix rule
  covers that.
* **`cornerMOSlv.lib` and `cornerMOShv.lib` use identical section names.** The
  *file* is the only thing distinguishing low- from high-voltage devices, so a
  section name alone is ambiguous.

A corner with no `includes:` emits `.lib "<lib_path>" <corner>`,
as sky130 does.

### Compiled Verilog-A models (`osdi:`)

If the PDK's device models are Verilog-A (SG13G2's MOSFETs are PSP 103),
ngspice runs them only as compiled OSDI plugins. Declare them:

```yaml
library:
  osdi:
    - "ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/osdi/psp103.osdi"
    - "ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/osdi/psp103_nqs.osdi"
```

Paths resolve against `$PDK_ROOT` like everything else. Declaring them states
a **fact about the PDK**, not a build step — silk never compiles Verilog-A;
that is environment setup (openvaf + the PDK's own compile script), the same
category as installing ngspice. What the declaration buys:

* ngspice decks that need PDK models load the plugins with `pre_osdi <path>` lines at the top
  of their `.control` blocks — ngspice executes these **before parsing the
  circuit**, so no `.spiceinit` or global state is involved;
* missing compiled plugins raise `SimulationError` when simulation starts,
  with installation instructions; netlist generation does not require them;
* `python -m silk.doctor` warns when declared OSDI plugins are missing;
* Xyce (which has no OSDI loader) and batch-write mode
  (`save_device_currents`, which emits no `.control` block) refuse with the
  reason.
* a compiled plugin must also be loadable by the selected ngspice binary.
  `silk.doctor` tests this with a circuit-free `pre_osdi` deck and reports
  loading errors with a `SILK_NGSPICE_BIN` hint. A simulation that encounters
  an incompatible plugin raises `SimulationError`. See
  [installing-the-pdk.md](installing-the-pdk.md) for setup.

**Cross-family corner pairing is yours to decide.** The PDK does not state
which resistor corner belongs with slow MOS; IHP's own xschem templates ship
only typical pairings. silk's `sg13g2.pdk.yaml` pairs slow MOS with worst-case
sheet R/C, and says so in a comment. Check it against your signoff.

### Units

`pdk_unit` describes model units; `user_units` describes author units.

| | `pdk_unit:` | `user_units:` |
|---|---|---|
| **file** | device catalog | manifest, or `load_pdk(user_units=…)` |
| **scope** | one per parameter | one per PDK load |
| **answers** | what unit the **PDK's models** expect | what unit the **circuit author** writes |
| **values** | `base` `u` `n` `p` `m`, or absent | `pdk`, `si`, or any of those prefixes |

`pdk_unit:` is a **fact about the foundry models** — read it off the `.subckt`
lines, it never changes. sky130's take µm (`pdk_unit: u`); GF180MCU's and
SG13G2's take metres (`pdk_unit: base`). Declare `pdk_unit` explicitly for
dimensional factory parameters; the legacy constraint path can infer it
from a suffixed minimum.

`user_units:` is the unit **the team writes in**. It lives in the manifest, not
at each call site, so every engineer using a PDK writes the same unit by
default — the shared file is what makes a team consistent.
`load_pdk(user_units=…)` overrides the manifest for that load.

**All silk manifests declare `user_units: u`** — microns. That is what lets one
circuit source target any of them.

**`pdk_unit:` values are PREFIXES, not units** — the same table serves lengths
(metres → µm) and capacitances (farads → pF), hence `u`/`p` rather than
`micrometre`/`picofarad`. **`base` means no prefix**, the SI base unit as-is. It
avoids confusion with `m`, which means *milli*.

#### One interpreter, one accessor

`BasePDK` is the **only** code that reads a device catalog and interprets
these two declarations. Everything downstream that needs a device's geometry
— `silk.layout`'s area and wire models first — asks the PDK layer for it in
canonical micrometres through `device_geometry_um(device, pdk)` (or
`pdk.device_geometry_um(device)`), which applies the same author→PDK factor
emission applies to caller-supplied values, leaves catalog defaults alone,
and then converts the `pdk_unit:` to µm. A geometry parameter with no
recognised `pdk_unit:` is a catalog defect and raises there; nothing in silk
assumes "untagged means microns". So: tag every W/L (`u` or `base`), and do
not write a second unit table anywhere — see
[api-reference.md §BasePDK](api-reference.md#basepdk).

#### Writing one unit everywhere

Nothing to pass — the manifests already declare microns:

```python
from silk import load_pdk

pdk = load_pdk('sky130')
device = pdk.nmos1p8('M1', d='out', g='in', s='0', b='0', w=1.0, l=0.3)
print(device.to_spice())
```

For the corresponding MOS factories, `w=1.0, l=0.3` emits geometry as follows
(the complete device lines also contain pins, model names and other parameters):

```text
sky130:   w=1 l=0.3
gf180mcu: w=1u l=300n
sg13g2:   w=1u l=300n
```

Other settings:

* `user_units='si'` — the author writes SI base units (`w=1e-6`). Equivalent to
  `user_units='base'`.
* `user_units='pdk'` — the author writes whatever that particular PDK wants, so
  nothing is scaled. With this setting, `w=1` means a micron on sky130
  and a metre on GF180MCU; use explicit author units for portable code.

#### `quantity:` — prefixes apply per dimension

A length in µm scales by 1e6; an **area** in µm² by 1e6 **squared**. Declare
`quantity: area` on such a parameter:

```yaml
area:
  pdk_unit: p
  quantity: area      # default is `length`
```

sky130 diode area uses `pdk_unit: p` because its model expects square
micrometres: the conversion from m² is 10¹². `quantity: area` makes
`area=0.09` written in µm² emit as `0.09`, rather than applying a length
conversion. The supported quantities are `length` (the default) and `area`.

Common geometry units do not make device names or electrical behavior
interchangeable. Select the appropriate factory (`nmos1p8`, `nmos3p3`, or
`nmos_lv`) and resize the circuit for the target process.

## 5. Loading & discovery

```python
import silk.pdk

print(silk.pdk.available())        # ['gf180mcu', 'sg13g2', 'sky130']
silk.pdk.manifest('sky130')         # -> the parsed manifest dict

pdk = silk.pdk.load_pdk('sky130')   # resolves lib_path against $PDK_ROOT
```

`load_pdk(name, *, pdk_root=None, library_location=None, corner=None,
user_units=None)` overrides, all optional:

- `pdk_root` — root to resolve library paths against. By default, the
  `PDK_ROOT` setting comes from the environment, then the user settings file,
  then `~/pdk`; see [settings.md](settings.md).
- `library_location` — replaces the main library path only. Extra includes,
  per-corner include files and OSDI paths still resolve against the PDK root.
  Its existence is checked when running a deck that references PDK models.
- `corner` — overrides the manifest's `default_corner`; membership in the
  declared corner map is not checked at load time.
- `user_units` — `'pdk'`, `'si'`, `'base'` or a prefix such as `'u'` or `'n'`,
  overrides the manifest's `user_units`.

`load_pdk` resolves the library path without requiring the model file to exist;
the packaged catalog remains usable for behavioral simulation and netlisting.
Running a deck containing a PDK device with `simulate()` raises `SimulationError`
if its library is missing, naming the path and installation advice. This applies
to both resolved and explicitly supplied library paths. Run
`python -m silk.doctor` first to report missing libraries before simulation.

Once loaded:

```python
from silk import load_pdk

pdk = load_pdk('sky130')
pdk.corners          # {'tt': {...}, 'ff': {...}, ...} — the full declared map
pdk.default_corner    # 'tt'
```

`Module.simulate(corner=...)` defaults to `pdk.default_corner` when `corner`
isn't passed.

## 6. Validate before you ship

The validator works on **any registered PDK by name** (`--pdk`), resolving the
catalog/lib/corners from its manifest — or on a raw device-details yaml
(`--yaml`) before you've written the manifest:

Run from the silk source checkout with its environment activated, or use
`.venv/bin/python`. Replace `mypdk` and the example paths with your own values. Static validation
needs the installed model files to check definitions, but does not run a simulator.

```bash
# Once your manifest exists — validate the whole PDK by name:
python -m silk.pdk.validate --pdk mypdk            # static (fast, no simulator)
python -m silk.pdk.validate --pdk mypdk --simulate # + per-device ngspice smoke-sim

# Before the manifest exists — validate a device-details yaml directly:
python -m silk.pdk.validate --yaml path/to/my_device_details.yaml --lib /path/to/lib.spice
```

`--pdk` defaults to `sky130`. Exit code `0` = clean, `1` = issues reported
(including `missing_definition`, `arity_mismatch`, `wrong_emission`, `sim_failed`, and the
catalog-only `unsourced_rating` / `bad_provenance` / `rating_convention` from
§3's ratings convention), `2` =
the PDK name, catalog file or model library cannot be resolved. Full details, the
checklist for individual devices, and the pytest regression-guard pattern
(`test_real_pdk_has_no_validation_issues` / `test_all_devices_simulate`) are in
`docs/pdk-validation.md` — read that before authoring a new device-details YAML;
this guide only covers wrapping it in a manifest.

## 7. Checklist: adding a new PDK

1. Write (or point to) the device-details YAML — see §3 and
   `docs/pdk-validation.md`. Any `junction_ratings` you declare need a
   `junction_ratings_provenance` block (§3, "Junction ratings").
2. Write `silk/pdk/data/<name>.pdk.yaml`: `name`, `device_details` path,
   `library.lib_path`, `library.default_corner`, `library.corners` (every
   corner you support, author-declared).
3. Omit `class` to use `BasePDK`; no shipped PDK needs a subclass. If you
   add custom behavior, the optional `class:` field selects its subclass.
   Corner-path resolution belongs in the manifest includes.
4. Drop both files under `silk/pdk/data/` (device-details YAML can live
   elsewhere if `device_details` in the manifest points at it correctly —
   the path is relative to the manifest's directory).
5. Confirm discovery: `python -c "import silk.pdk; print(silk.pdk.available())"`
   should list your PDK; `silk.pdk.load_pdk('<name>')` should load it. Restart
   Python if discovery was already cached before you added the manifest.
6. Add the region-policy entry described in §1 and ensure the YAML files
   are covered by `[tool.setuptools.package-data]` in `pyproject.toml`.
7. Run `python -m silk.pdk.validate --pdk <name>` (static, then add
   `--simulate`) and fix everything it reports before shipping. Don't drop
   the `--pdk` flag — the bare command defaults to validating **sky130**,
   not your new PDK.
