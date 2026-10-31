"""Layout intent for an NMOS differential pair and tail device.

The input pair requests vertical symmetry and a common-centroid group.
A proximity constraint keeps the tail device near one input device.
Port roles and a length budget on the shared tail net complete the intent.
These are constraint records, not a generated or checked layout.

Run ``.venv/bin/python -m silk.examples.constraints_diff_pair``.
It prints ports, net intent, device constraints and serialized overlays.
Only the packaged sky130 catalog is needed; no simulation runs.
"""

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.examples import example_output_dir
from silk import load_pdk

# The package installs the constraint methods on Module.
import silk.constraints
from silk.constraints import (
    GroupBlocks, ProximityConstraint, SymmetricBlocks, serialization,
)


pdk = load_pdk('sky130')


def build_diff_pair() -> Module:
    mod = Module(
        "PDK_skywater130", example_output_dir("constraints_diff_pair"), "diff_pair.cir",
        pdk_details=pdk, module_name="diff_pair",
    )

    mod.define_module_ports({
        "VDD":    Port.supply(),
        "VSS":    Port.ground(),
        "VBIAS":  Port.analog_input(),
        "VIN_P":  Port.analog_input(),
        "VIN_M":  Port.analog_input(),
        "VOUT_P": Port.analog_output(),
        "VOUT_N": Port.analog_output(),
    })

    mod.add_net("tail")
    mod.net_dict["tail"].is_critical = True
    mod.net_dict["tail"].max_length_um = 25.0

    mod.add(pdk.nmos1p8("M1", d="VOUT_N", g="VIN_P", s="tail", b="VSS", w=2.0, l=0.5, m=1))
    mod.add(pdk.nmos1p8("M2", d="VOUT_P", g="VIN_M", s="tail", b="VSS", w=2.0, l=0.5, m=1))
    mod.add(pdk.nmos1p8("M_tail", d="tail", g="VBIAS", s="VSS", b="VSS", w=4.0, l=1.0, m=1))

    mod.add_constraint(SymmetricBlocks(
        pairs=[["M1", "M2"]], direction="V",
        rationale="differential input pair across vertical axis",
    ))
    mod.add_constraint(GroupBlocks(
        name="input_pair", instances=["M1", "M2"],
        generator={"name": "MOS", "parameters": {"pattern": "cc"}},
        rationale="common-centroid finger layout for matching",
    ))
    mod.add_constraint(ProximityConstraint(
        instance_a="M1", instance_b="M_tail", max_distance_um=8.0,
        severity="soft",
        rationale="keep tail close to pair for thermal matching and CMRR",
    ))
    return mod


def summarize(mod: Module) -> None:
    print("=" * 72)
    print(f"Module: {mod.module_name}    "
          f"({len(mod.elem_dict)} devices, {len(mod.net_dict)} nets)")
    print("=" * 72)

    print("\n── Ports (with PortRole — silk-native) ────────────────────────────")
    for name, pin in mod.pin_list.items():
        print(f"  {name:8s}  dir={pin.direction!s:18s}  role={pin.role!s}")

    print("\n── Nets with layout state set ─────────────────────────────────────")
    interesting = [
        (n, net) for n, net in mod.net_dict.items()
        if net.is_critical or net.max_ir_drop_mV is not None
        or net.max_j_factor is not None or net.max_length_um is not None
    ]
    if not interesting:
        print("  (none)")
    for name, net in interesting:
        print(f"  {name:10s}  is_critical={net.is_critical}  "
              f"max_length_um={net.max_length_um}")

    print("\n── Constraints (device-relational) ────────────────────────────────")
    for c in mod.list_constraints():
        if c.constraint == "SymmetricBlocks":
            body = f"pairs={c.pairs} dir={c.direction}"
        elif c.constraint == "GroupBlocks":
            body = f"name={c.name!r} instances={c.instances} gen={c.generator}"
        elif c.constraint == "SameTemplate":
            body = f"instances={c.instances}"
        elif c.constraint == "ProximityConstraint":
            body = f"{c.instance_a}↔{c.instance_b} ≤{c.max_distance_um}µm"
        else:
            body = ""
        print(f"  [{c.constraint:21s}] severity={c.severity}  {body}")
        if c.rationale:
            print(f"    why: {c.rationale}")


def show_overlay_entries(mod: Module) -> None:
    """Print PortRoles and NetIntent entries synthesized from the module."""
    print("\n── Overlay entries (synthesized from PortRole / Net) ──────────────")
    payload = serialization.to_obj(mod)
    overlays = [o for o in payload if o["constraint"] in ("PortRoles", "NetIntent")]
    for entry in overlays:
        print(f"  {entry}")


if __name__ == "__main__":
    mod = build_diff_pair()
    summarize(mod)
    show_overlay_entries(mod)
