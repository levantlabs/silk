"""Load externally authored constraint JSON onto a circuit.

The circuit has a resistor-loaded NMOS stage and a clocked reset device.
PortRoles entries set port roles, NetIntent entries set net budgets and
voltage domains, and device constraints enter Module.list_constraints().
The records illustrate loading; no layout is generated or validated.

Run ``.venv/bin/python -m silk.examples.constraints_external_load``.
It prints the loaded fields and warnings for a second payload containing
missing port and net references. Only the packaged sky130 catalog is needed.
External consumers must support silk's overlay and constraint types or
adapt the records before importing them.
"""
import warnings

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.examples import example_output_dir
from silk.elements import elements
from silk import load_pdk

# The package installs the constraint methods on Module.
import silk.constraints
from silk.constraints import serialization


pdk = load_pdk('sky130')


EXTERNAL_FILE = """
[
  {"constraint": "PortRoles",
   "roles": {"VDD": "power", "VSS": "ground", "CLK": "clock",
             "VINP": "analog", "VOUT": "analog"}},

  {"constraint": "NetIntent", "net": "vdd_int",
   "max_ir_drop_mV": 30.0, "max_j_factor": 0.7,
   "domain": {"name": "avdd_1v8", "nominal": 1.8, "tolerance": 0.1}},

  {"constraint": "NetIntent", "net": "clk_int",
   "max_length_um": 75.0, "is_critical": true},

  {"constraint": "SymmetricBlocks", "pairs": [["M1", "M2"]], "direction": "V"},
  {"constraint": "GroupBlocks",     "name": "diff_pair",
                                    "instances": ["M1", "M2"],
                                    "generator": {"name": "MOS",
                                                  "parameters": {"pattern": "cc"}}},
  {"constraint": "SameTemplate",    "instances": ["M1", "M2"]},
  {"constraint": "ProximityConstraint",
   "instance_a": "M1", "instance_b": "M2", "max_distance_um": 4.0}
]
"""


def build_shell() -> Module:
    """Build the ports, nets and devices referenced by the JSON payload.

    Missing port or net references warn during loading. The loader updates
    existing objects; it does not create missing ports or nets.
    """
    mod = Module(
        "PDK_skywater130", example_output_dir("constraints_external_load"),
        "external.cir",
        pdk_details=pdk, module_name="external_demo",
    )
    mod.define_module_ports({
        "VDD": PortDir.IN,
        "VSS": PortDir.IN,
        "CLK": PortDir.IN,
        "VINP": PortDir.IN,
        "VOUT": PortDir.OUT,
    })
    mod.add_net("vdd_int")
    mod.add_net("clk_int")

    mod.add(elements.resistor("R_FILT", p="VDD", n="vdd_int", value=100.0))
    mod.add(elements.resistor("R_LOAD", p="vdd_int", n="VOUT", value=10e3))

    mod.add(elements.resistor("R_CLK", p="CLK", n="clk_int", value=50.0))
    mod.add(pdk.nmos1p8("M1", d="VOUT", g="VINP", s="VSS", b="VSS", w=1.0, l=0.15))
    mod.add(pdk.nmos1p8("M2", d="VOUT", g="clk_int", s="VSS", b="VSS", w=1.0, l=0.15))
    return mod


def main() -> None:
    print("=" * 72)
    print("Loading a silk-canonical JSON file into a silk Module")
    print("=" * 72)

    print("\n── Source JSON (3 categories) ─────────────────────────────────────")
    print(EXTERNAL_FILE.strip())

    mod = build_shell()
    print("\n── Module shell *before* loading ──────────────────────────────────")
    print(f"  ports : {list(mod.pin_list)}    "
          f"(roles: {[p.role for p in mod.pin_list.values()]})")
    print(f"  nets  : {list(mod.net_dict)}    "
          f"(intent: all None / False)")
    print(f"  constraints attached: {len(mod.list_constraints())}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        serialization.from_json(mod, EXTERNAL_FILE)

    print("\n── Module shell *after* loading — overlays routed everything ──────")

    print("\n  Port roles (5 of 5 ports got a role from the file):")
    for name, pin in mod.pin_list.items():
        print(f"    {name:6s}  role={pin.role!r}")

    print("\n  Net layout intent:")
    for name, net in mod.net_dict.items():
        dom = net.domain.name if net.domain else None
        print(f"    {name:10s}  ir_mV={net.max_ir_drop_mV}  "
              f"j_factor={net.max_j_factor}  "
              f"len_um={net.max_length_um}  "
              f"critical={net.is_critical}  domain={dom}")

    print("\n  Module.list_constraints() — only the device-relational types:")
    for c in mod.list_constraints():
        print(f"    [{c.constraint:21s}] severity={c.severity}  source={c.source}")

    if w:
        print("\n  Warnings emitted:")
        for entry in w:
            print(f"    ! {entry.message}")
    else:
        print("\n  (no warnings — every reference resolved)")

    print("\n" + "=" * 72)
    print("Forgiving load: missing references warn, don't fail")
    print("=" * 72)

    sloppy = '''[
        {"constraint": "PortRoles", "roles": {"GHOST_PORT": "power"}},
        {"constraint": "NetIntent", "net": "ghost_net", "max_ir_drop_mV": 99},
        {"constraint": "SameTemplate", "instances": ["M1", "M2"]}
    ]'''

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        serialization.from_json(mod, sloppy)

    print("\n  Warnings emitted:")
    for entry in w:
        print(f"    ! {entry.message}")
    print(f"\n  list_constraints() contains "
          f"{len(mod.list_constraints())} entries — the SameTemplate "
          f"still landed even though the overlay entries failed.")


if __name__ == "__main__":
    main()
