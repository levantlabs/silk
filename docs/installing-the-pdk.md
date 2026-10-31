# Installing a PDK

A process design kit (PDK) supplies the device models a simulator needs for
transistor circuits. Start with **sky130**, the release-validated PDK.

silk includes the device catalogs and manifests for three PDKs in
`silk/pdk/data/`; the simulator model files are a separate installation.
Registration means `silk.pdk.available()` can discover
the catalog; it is not, by itself, a simulation-support claim:

| silk name | Foundry / node | Registered catalog | 0.2.0 simulation status |
|---|---|---|---|
| `sky130` | SkyWater 130nm bulk CMOS | yes | **Release-validated** with ngspice and installed models |
| `gf180mcu` | GlobalFoundries 180nm bulk CMOS | yes | **Preview/catalog**; not release-gated |
| `sg13g2` | IHP 130nm BiCMOS + SiGe HBT | yes | **Preview/catalog**; MOSFET-only catalog, not release-validated |

Catalog/schema use and behavioral simulation need only `pip install`; neither
uses the real foundry models. **Nor does real SPICE on a testbench of ideal
elements, passives and sources** — silk omits PDK library includes when it
can prove the entire circuit uses only those elements, so an RC filter or an
ideal-source bench runs on ngspice alone. See the
[PDK-free SPICE walkthrough](quickstart.md#the-middle-rung-real-spice-with-no-pdk).
Custom netlist content conservatively retains the PDK includes.
What needs the real models on disk is a deck with PDK **devices** in it:
transistor-level `simulate()`, corner runs, `python -m silk.pdk.validate`.

What silk expects, concretely, one directory per PDK under a single
`$PDK_ROOT`:

- `sky130A/libs.tech/ngspice/sky130.lib.spice` (the standard open_pdks layout)
- `gf180mcuD/libs.tech/ngspice/sm141064.ngspice`
- `ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib` (one of six
  files SG13G2 needs — see below)

plus ngspice, found on `PATH` or selected with `SILK_NGSPICE_BIN`. Install
only the PDKs you need.

Tell silk where that directory is with the `PDK_ROOT` setting — once, in your
silk settings file (`~/.config/silk/silk.env`, a line `PDK_ROOT=/path/to/pdk`),
or as an environment variable, which overrides the file. Without either, silk
looks in `~/pdk`. [settings.md](settings.md) has the format and a copy-paste
example.

If you only want to try silk out, skip this page and start with the
[60-second start in the quickstart](quickstart.md#1-60-second-start-no-pdk-no-ngspice).

## Recommended: pdksetup

[pdksetup](https://github.com/levantlabs/pdksetup) is a small Apache-2.0 CLI
that installs and version-tracks open PDKs. It wraps
volare for sky130 and gf180mcu, so you get the same on-disk layout EDA tools
expect; SG13G2 uses its own installer since it isn't volare-managed.

**Prerequisites:** Python 3.11+, `git`.

```bash
git clone https://github.com/levantlabs/pdksetup.git
cd pdksetup
python3 -m venv .venv
.venv/bin/python -m pip install -e .
source .venv/bin/activate   # makes the `pdk` command available
```

Use this installer environment for the `pdk` commands below. Before running
any silk command, run `deactivate` and return to the Python environment where
silk is installed. The installer environment does not include silk.

Install sky130:

```bash
export PDK_ROOT="$HOME/pdk" # choose the same root for pdksetup and silk
pdk install sky130          # latest version
```

For the preview PDKs, use `pdk install gf180mcu` or
`pdk install ihp_sg13g2`. silk accepts both `sg13g2` and `ihp_sg13g2` for
SG13G2. See the [preview requirements](#preview-integration-details-and-known-caveats)
before trying to simulate it.

### Where it installs

`pdk` resolves the install root from four sources, **highest wins**:

1. `--pdk-root PATH` on the command line (or `--here` for `$PWD`)
2. the `PDK_ROOT` environment variable
3. `pdk_root` in `~/.config/pdksetup/config.toml`
4. `~/.pdk` (default)

**pdksetup and silk have different defaults:** pdksetup uses `~/.pdk`,
while silk uses `~/pdk`. They also have separate configuration files.
Setting `pdk_root` in pdksetup does not configure silk. Use the same
`PDK_ROOT` environment variable for both, or set the matching path in each
program's configuration. A `PDK_ROOT` value in silk's settings file does not
export a shell variable for `pdk` or `ls`.

For sky130 and gf180mcu, pdksetup uses volare to download and enable a
version. The library locations silk reads are:

```text
$PDK_ROOT/
├── sky130A/libs.tech/ngspice/sky130.lib.spice
├── gf180mcuD/libs.tech/ngspice/sm141064.ngspice
└── ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib
```

### Status and updates

```bash
pdk status                  # table of all known PDKs
pdk status sky130           # just one
pdk check-updates sky130    # compare installed version against the release list
```

If an installed PDK is reported missing, compare the root used by
`pdk status` with the root printed by `python -m silk.doctor`.

## Preview integration details and known caveats

<a id="sg13g2-simulates-end-to-end-with-two-environment-requirements-silk-cannot-supply"></a>

### SG13G2: compiled models and compatible ngspice

SG13G2 requires compiled Verilog-A models, loaded through ngspice's OSDI
plugin interface. Install [OpenVAF-Reloaded](https://github.com/OpenVAF/OpenVAF-Reloaded)
and make `openvaf-r` available on `PATH` before installing the PDK. Current
[pdksetup](https://github.com/levantlabs/pdksetup#ihp-sg13g2--130nm-bicmos--sige-hbt)
compiles these models during installation when `openvaf-r` or `openvaf` is
available. Older installations may need the PDK's compile script.

The upstream script expects `PDK_ROOT` to name the IHP repository itself,
which is below silk's shared PDK root. For the layout shown above, run it
with a command-local root:

```bash
PDK_ROOT="$PDK_ROOT/ihp_sg13g2" python "$PDK_ROOT/ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/install.py"
```

This command assumes the shared `PDK_ROOT` is exported. The
[upstream script](https://github.com/IHP-GmbH/IHP-Open-PDK/blob/main/ihp-sg13g2/libs.tech/ngspice/install.py)
also tries to create `~/.spiceinit`; silk does not require that file, because
its generated decks include the library paths and plugin-loading commands.

silk's manifest requires four files in
`$PDK_ROOT/ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/osdi/`:
`psp103.osdi`, `psp103_nqs.osdi`, `r3_cmc.osdi` and `mosvar.osdi`.
Missing plugins raise `SimulationError` when a simulation needs them; silk
does not compile them.

Plugins built by openvaf-r use OSDI v0.4, which requires ngspice 44 or newer
with OSDI support. ngspice 42 cannot load them. Use `SILK_NGSPICE_BIN` to
select a compatible executable, then run `python -m silk.doctor`. When that
executable is available, the doctor checks that it can load the plugins and
warns if it cannot. Also read the separate ngspice row: if the executable is
missing, the plugin-load check is skipped. A failed simulation reports the plugin-loading cause in
`SimulationError`.

SG13G2's typical corner includes six model libraries: `cornerMOSlv.lib`,
`cornerMOShv.lib`, `cornerRES.lib`, `cornerCAP.lib`, `cornerDIO.lib` and
`cornerHBT.lib`. They live under
`$PDK_ROOT/ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/models/` alongside their
supporting model files. The catalog currently exposes only MOSFETs.
SG13G2 transistor simulation requires ngspice; silk's Xyce backend refuses
decks that need its OSDI models.

For temperature characterization, configure `tb.set_temperature(85)` before
running the testbench. The deprecated `run(temperature=...)` argument only
asserts the configured temperature; it does not set it.

The optional integration tests exercise device sweeps and corners. Select a
compatible ngspice through the settings above, then run this command from the
root of a silk source checkout, using its silk environment:

```bash
.venv/bin/python -m pytest tests/test_sg13g2_simulation.py -m "slow or not slow" -q
```

The tests skip with a reason if the PDK, compiled plugins or compatible
simulator is missing. A skip does not validate the integration.

### GF180MCU: statistical-corner limitation

The manifest includes model sections for MOSFETs, resistors, capacitors,
diodes and bipolar transistors, plus `design.ngspice`. Catalog coverage does
not establish simulation support for every family.

The model library's `res_statistical` section omits the sheet resistance for
`rm4`. A circuit containing `res_rm4` can therefore abort in the
`statistical` corner. Use another metal layer or omit that resistor for
that corner.

Device names differ across PDKs: `nmos1p8` (sky130), `nmos3p3` (GF180MCU)
and `nmos_lv` (SG13G2). Geometry values use the same author-facing units, as
explained below.

## Geometry units for registered PDKs

Every shipped manifest defines its author-facing geometry in microns, including
the two preview integrations. **Lengths such as W/L are in microns; areas
are in square microns.** silk
converts to whatever that PDK's own models actually expect before it emits the
netlist:

```python
from silk import load_pdk

pdk = load_pdk('sky130')
device = pdk.nmos1p8('M1', d='d', g='g', s='0', b='0', w=1.0, l=0.3)
print(device.to_spice(), end='')
```

Output:

```text
XMM1 d g 0 0 sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.3
```

For the same W/L, GF180MCU and SG13G2 emit `w=1u l=300n` because their
models expect metres.

This demonstrates the tested unit-emission contract for the registered
catalogs. For GF180MCU and SG13G2 it does not imply end-to-end simulation
support.

The manifest sets author units; each parameter's `pdk_unit` determines emission.
See [Units in adding-a-pdk.md](adding-a-pdk.md#units) for conversion rules.

## Install ngspice

```bash
sudo apt install ngspice     # Debian / Ubuntu
brew install ngspice         # macOS
```

For sky130, use ngspice 44 or a release from 47 onward to avoid [slow library loading in 45, 45.2 and 46](known_issues.md#slow-sky130-library-loading-on-ngspice-45-and-46).

silk supports ngspice 42 and newer for sky130, though newer versions fix
known simulator bugs listed by `python -m silk.doctor`. Preview PDKs may add
tool or model requirements —
notably SG13G2's compiled OSDI plugins. If you keep a specific build outside
`PATH`, set **`SILK_NGSPICE_BIN`** to the binary — in your
[settings file](settings.md) for every run, or as an environment variable for
one command. silk and `python -m silk.doctor` both resolve it the same way, so
the doctor always reports the binary your simulations will actually use. The
optional Xyce backend resolves **`SILK_XYCE_BIN`** identically.

## Verify

Monte Carlo needs the statistical sections of the sky130 library (`tt_mm`,
`ss_mm`, `ff_mm`, `sf_mm`, `fs_mm` and `mc`). An older open_pdks install lacks
them: `python -m silk.doctor` reports each as `[warn] pdk sky130 section <name>`,
and a Monte Carlo run raises `MissingLibrarySectionError` before any simulator
starts. Update the install (see above) to get them. Ordinary nominal-corner
simulation does not need these sections.

Run these checks in the Python environment where silk is installed. In a
source checkout, use `.venv/bin/python` in place of `python`.

Check the library file for each PDK you installed. Substitute your PDK root
if it comes from the settings file rather than the environment (`python -m silk.doctor` prints it):

```bash
echo "$PDK_ROOT"
ls "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice"
ls "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice"
ls "$PDK_ROOT/ihp_sg13g2/ihp-sg13g2/libs.tech/ngspice/models/cornerMOSlv.lib"
```

```bash
python -c "from silk import load_pdk; load_pdk('sky130'); print('PDK OK')"
```

Expected:

```
PDK OK
```

`load_pdk` succeeding is *not* by itself proof the models are installed: since
the catalog, units and corners ship inside silk, `load_pdk` resolves the
library path without stat-ing it, so that behavioral mode and
[PDK-free SPICE](quickstart.md#the-middle-rung-real-spice-with-no-pdk) work on
a bare `pip install`. The `ls` checks above, and `python -m silk.doctor` below,
are what tell you the library is really there.

A simulation that needs missing PDK models raises `SimulationError`, naming
the PDK and resolved library path and suggesting installation or path settings.
Netlist generation alone does not require the files.

For the whole environment at once — Python version, ngspice, `$PDK_ROOT`,
every *registered* PDK, optional extras — run:

```bash
python -m silk.doctor
```

It reports a library check for each registered PDK (`pdk sky130`,
`pdk gf180mcu`, `pdk sg13g2`). Missing model libraries produce `[warn]`.
For SG13G2, it also checks that the four compiled plugins exist. When the
selected ngspice executable is available, it checks that the plugins load,
warning with a fix if loading fails. If the executable is missing, the
plugin-load check is skipped; read the separate ngspice row for that failure.

A doctor's `[ ok ] pdk ...` result is an installation check: the primary
model library and declared OSDI plugin files exist. It proves plugin loading
only when the selected ngspice executable is available, so also read the
separate ngspice row. It does not check every file transitively included by
the library, validate device behavior,
or change a preview PDK's support tier.

## Alternatives

**Container setup.** [IIC-OSIC-TOOLS](https://github.com/iic-jku/IIC-OSIC-TOOLS)
provides an open-source EDA environment with PDKs. Point `PDK_ROOT` at the
container's PDK directory and run the verification steps above. SG13G2 still
needs compatible ngspice and compiled plugins.

**open_pdks from source.** Building
[open_pdks](https://github.com/fossi-foundation/open-pdks) yourself works fine
too — silk cares only about the final layout, not how it got there. Point
`PDK_ROOT` at the directory containing `sky130A/` (or the equivalent for
another PDK) and run the verify steps above.

## Next steps

- [Quickstart](quickstart.md) — building and simulating a circuit
- [Adding a PDK](adding-a-pdk.md) — the manifest format, corners, includes, and
  the full unit-conversion mechanism, for a PDK silk doesn't ship yet
- [PDK validation](pdk-validation.md) — `python -m silk.pdk.validate` checks
  every device in the manifest against the installed models
