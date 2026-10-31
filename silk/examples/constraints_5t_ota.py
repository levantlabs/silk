"""Layout intent for a five-transistor OTA.

The NMOS input pair has a PMOS mirror load and an NMOS tail device.
The example declares port roles, net budgets, symmetry, device groups,
matching templates and proximity. SameTemplate requests identical PMOS
load templates; this script does not generate or validate a layout.

Run ``.venv/bin/python -m silk.examples.constraints_5t_ota``.
It prints the intent, selected serialized fields, and fields reloaded on
a fresh shell. Device references are not validated on that shell.
Only the packaged sky130 catalog is needed; no simulation runs.
"""
import json

from silk._json_codec import encode_json
from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.examples import example_output_dir
from silk import load_pdk

# The package installs the constraint methods on Module.
import silk.constraints
from silk.constraints import (
    GroupBlocks, ProximityConstraint, SameTemplate, SymmetricBlocks,
    serialization,
)


pdk = load_pdk('sky130')


def build_5t_ota() -> Module:
    mod = Module(
        "PDK_skywater130", example_output_dir("constraints_5t_ota"), "ota5t.cir",
        pdk_details=pdk, module_name="ota_5t",
    )

    mod.define_module_ports({
        "VDD":   Port.supply(),
        "VSS":   Port.ground(),
        "VINP":  Port.analog_input(),
        "VINM":  Port.analog_input(),
        "VOUT":  Port.analog_output(),
        "IBIAS": Port.analog_input(),
    })

    for n in ("vint", "tail"):
        mod.add_net(n)

    out_net = mod.add_net("VOUT")
    out_net.is_critical = True
    out_net.max_length_um = 30.0
    out_net.max_j_factor = 0.5

    mod.net_dict["vint"].max_length_um = 40.0

    mod.add(pdk.nmos1p8("M1", d="vint", g="VINP", s="tail", b="VSS", w=4.0, l=0.5, m=1))
    mod.add(pdk.nmos1p8("M2", d="VOUT", g="VINM", s="tail", b="VSS", w=4.0, l=0.5, m=1))

    mod.add(pdk.pmos1p8("M3", d="vint", g="vint", s="VDD", b="VDD", w=8.0, l=0.5, m=1))
    mod.add(pdk.pmos1p8("M4", d="VOUT", g="vint", s="VDD", b="VDD", w=8.0, l=0.5, m=1))

    mod.add(pdk.nmos1p8("M5", d="tail", g="IBIAS", s="VSS", b="VSS", w=6.0, l=1.0, m=1))

    mod.add_constraint(SymmetricBlocks(
        pairs=[["M1", "M2"]], direction="V",
        rationale="NMOS input pair across the OTA's vertical axis",
    ))
    mod.add_constraint(SymmetricBlocks(
        pairs=[["M3", "M4"]], direction="V",
        rationale="PMOS mirror load mirrored across the same axis",
    ))

    mod.add_constraint(GroupBlocks(
        name="input_pair", instances=["M1", "M2"],
        generator={"name": "MOS", "parameters": {"pattern": "cc"}},
        rationale="common-centroid input pair — primary matching driver",
    ))
    mod.add_constraint(GroupBlocks(
        name="mirror_load", instances=["M3", "M4"],
        generator={"name": "MOS", "parameters": {"pattern": "interdigitated"}},
        rationale="interdigitated PMOS load — secondary matching",
    ))

    mod.add_constraint(SameTemplate(
        instances=["M3", "M4"],
        rationale="PMOS load fingers must use identical templates so "
                  "the mirror ratio is set by W only, not layout drift",
    ))

    mod.add_constraint(ProximityConstraint(
        instance_a="M5", instance_b="M1",
        max_distance_um=10.0, severity="soft",
        rationale="tail current source close to the input pair's common "
                  "source for CMRR + thermal coupling",
    ))

    return mod


def _constraint_body(c) -> str:
    if c.constraint == "SymmetricBlocks":
        return f"pairs={c.pairs} dir={c.direction}"
    if c.constraint == "GroupBlocks":
        return (f"name={c.name!r} instances={c.instances} "
                f"gen={c.generator}")
    if c.constraint == "SameTemplate":
        return f"instances={c.instances}"
    if c.constraint == "ProximityConstraint":
        return f"{c.instance_a}↔{c.instance_b} ≤{c.max_distance_um}µm"
    return ""


def summarize(mod: Module) -> None:
    print("=" * 72)
    print(f"Module: {mod.module_name}    "
          f"({len(mod.elem_dict)} devices, {len(mod.net_dict)} nets, "
          f"{len(mod.list_constraints())} constraints)")
    print("=" * 72)

    print("\n── Ports (PortRole — silk-native) ─────────────────────────────────")
    for name, pin in mod.pin_list.items():
        print(f"  {name:6s}  dir={pin.direction!s:18s}  role={pin.role!s}")

    print("\n── Nets with layout intent ────────────────────────────────────────")
    interesting = [
        (n, net) for n, net in mod.net_dict.items()
        if net.is_critical or net.max_ir_drop_mV is not None
        or net.max_j_factor is not None or net.max_length_um is not None
    ]
    print(f"  {'net':10s} {'critical':9s} {'max_ir_mV':10s} "
          f"{'max_j':6s} {'max_len_um':10s}")
    for name, net in interesting:
        print(f"  {name:10s} {str(net.is_critical):9s} "
              f"{str(net.max_ir_drop_mV):10s} {str(net.max_j_factor):6s} "
              f"{str(net.max_length_um):10s}")

    print("\n── Constraints (device-relational) ────────────────────────────────")
    for c in mod.list_constraints():
        print(f"  [{c.constraint:21s}] severity={c.severity}  "
              f"{_constraint_body(c)}")
        if c.rationale:
            print(f"    why: {c.rationale}")


def show_serialization(mod: Module) -> None:
    """Print selected serialized fields, omitting record envelope metadata."""
    print("\n── Selected serialized fields (compact) ───────────────────")
    payload = serialization.to_obj(mod)
    for entry in payload:

        slim = {k: v for k, v in entry.items()
                if k not in ("id", "created_at", "prompt_hash",
                             "source", "confidence")}
        print(f"  {json.dumps(encode_json(slim), allow_nan=False, separators=(',', ':'))}")
    port_overlay = sum(1 for e in payload if e["constraint"] == "PortRoles")
    net_overlay  = sum(1 for e in payload if e["constraint"] == "NetIntent")
    print(f"\n  ({len(payload)} total entries: "
          f"{port_overlay} PortRoles, {net_overlay} NetIntent, "
          f"{len(mod.list_constraints())} real constraints)")


def round_trip_check(mod: Module) -> None:
    """Reload port roles, net intent and constraint records on a fresh shell.

    Device references are not validated here.
    """
    print("\n── Round-trip into a fresh shell ──────────────────────────────────")

    # Only ports and nets are rebuilt; device references remain unchecked.
    fresh = Module(
        "PDK_skywater130", example_output_dir("constraints_5t_ota"), "ota5t_reload.cir",
        pdk_details=pdk, module_name="ota_5t",
    )

    from silk.circuit.circuit_elements import PortDir
    fresh.define_module_ports({n: PortDir.IN for n in
                               ("VDD", "VSS", "VINP", "VINM", "IBIAS")})
    fresh.define_module_ports({"VOUT": PortDir.OUT})
    for n in ("vint", "tail", "VOUT"):
        fresh.add_net(n)

    payload = serialization.to_json(mod)
    serialization.from_json(fresh, payload)

    print(f"  fresh.pin_list['VDD'].role        = {fresh.pin_list['VDD'].role!r}")
    print(f"  fresh.pin_list['VOUT'].role       = {fresh.pin_list['VOUT'].role!r}  "
          f"(survives round-trip via the PortRoles overlay entry)")
    print(f"  fresh.pin_list['VINP'].role       = {fresh.pin_list['VINP'].role!r}")
    print(f"  fresh.net_dict['VOUT'].max_j_factor    = "
          f"{fresh.net_dict['VOUT'].max_j_factor}")
    print(f"  fresh.net_dict['VOUT'].max_length_um   = "
          f"{fresh.net_dict['VOUT'].max_length_um}")
    print(f"  fresh.net_dict['vint'].max_length_um   = "
          f"{fresh.net_dict['vint'].max_length_um}")
    print(f"  fresh.list_constraints() count    = "
          f"{len(fresh.list_constraints())}  "
          f"(expected: {len(mod.list_constraints())})")


if __name__ == "__main__":
    mod = build_5t_ota()
    summarize(mod)
    show_serialization(mod)
    round_trip_check(mod)
