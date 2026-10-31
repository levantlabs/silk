"""Serialize and restore layout intent for a two-NMOS current mirror.

Port roles describe the interface, Net fields hold wiring budgets, and
SymmetricBlocks and GroupBlocks record device placement intent.
The example reloads those records onto a rebuilt circuit and displays them.
It does not generate or validate a layout.

Run ``.venv/bin/python -m silk.examples.constraints_current_mirror``.
It prints the intent, JSON payload and selected restored fields.
Only the packaged sky130 catalog is needed; no simulation runs.
"""

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.examples import example_output_dir
from silk import load_pdk


# The package installs the constraint methods on Module.
import silk.constraints  # noqa: F401
from silk.constraints import GroupBlocks, SymmetricBlocks, serialization


pdk = load_pdk('sky130')


def build_current_mirror() -> Module:
    """Build two NMOS devices and attach port roles, net intent and constraints."""
    mod = Module(
        "PDK_skywater130", example_output_dir("constraints_current_mirror"),
        "current_mirror.cir",
        pdk_details=pdk, module_name="current_mirror",
    )

    mod.define_module_ports({
        "VDD":  Port.supply(),
        "VSS":  Port.ground(),
        "IBIAS": Port.analog_input(),
        "IOUT":  Port.analog_output(),
    })

    mod.add(pdk.nmos1p8("M1", d="IBIAS", g="IBIAS", s="VSS", b="VSS", w=1.0, l=0.5, m=1))
    mod.add(pdk.nmos1p8("M2", d="IOUT", g="IBIAS", s="VSS", b="VSS", w=1.0, l=0.5, m=1))

    mod.net_dict["IBIAS"].max_length_um = 50.0
    mod.net_dict["IBIAS"].is_critical = True

    mod.add_constraint(SymmetricBlocks(
        pairs=[["M1", "M2"]], direction="V",
        rationale="reference and output mirror, axis through their shared gate",
    ))
    mod.add_constraint(GroupBlocks(
        name="mirror", instances=["M1", "M2"],
        generator={"name": "MOS", "parameters": {"pattern": "interdigitated"}},
        rationale="finger-interdigitated layout for matching",
    ))
    return mod


def show_intent(mod: Module) -> None:
    """Inspect the layout intent silk holds in memory."""
    print("=" * 72)
    print(f"Module: {mod.module_name}")
    print("=" * 72)

    print("\n── Port roles (silk-native, via PortRole) ─────────────────────────")
    for name, pin in mod.pin_list.items():
        print(f"  {name:6s}  dir={pin.direction!s:18s}  role={pin.role!s}")

    print("\n── Per-net layout state (silk-native, on Net) ─────────────────────")
    print(f"  {'net':10s} {'classification':16s} {'critical':9s} "
          f"{'max_ir_mV':10s} {'max_j':6s} {'max_len_um':10s}")
    for name, net in mod.net_dict.items():
        cls = net.classification.value if net.classification else "—"
        print(f"  {name:10s} {cls:16s} {str(net.is_critical):9s} "
              f"{str(net.max_ir_drop_mV):10s} {str(net.max_j_factor):6s} "
              f"{str(net.max_length_um):10s}")

    print("\n── Constraints (device-relational, via add_constraint) ────────────")
    for c in mod.list_constraints():
        print(f"  [{c.constraint:20s}] severity={c.severity}  "
              f"source={c.source}  conf={c.confidence}")
        if c.rationale:
            print(f"    why: {c.rationale}")


def show_serialized(mod: Module) -> str:
    print("\n── Serialized to silk JSON ─────────────────────────────────")
    payload = serialization.to_json(mod)
    print(payload)
    print("\n  Serialized entries:")
    print("    Port roles → one PortRoles entry")
    print("    Net fields → one NetIntent entry per net with intent")
    print("    Device constraints → records with metadata (id, source, ...)")
    return payload


def round_trip(payload: str) -> Module:
    print("\n── Reload port roles, net intent and device constraints ─────")
    fresh = Module(
        "PDK_skywater130", example_output_dir("constraints_current_mirror"),
        "current_mirror_reloaded.cir",
        pdk_details=pdk, module_name="current_mirror",
    )

    fresh.define_module_ports({
        "VDD":  Port(dir=Port.supply().dir),
        "VSS":  Port(dir=Port.ground().dir),
        "IBIAS": Port.analog_input(),
        "IOUT":  Port.analog_output(),
    })
    fresh.add(pdk.nmos1p8("M1", d="IBIAS", g="IBIAS", s="VSS", b="VSS", w=1.0, l=0.5, m=1))
    fresh.add(pdk.nmos1p8("M2", d="IOUT", g="IBIAS", s="VSS", b="VSS", w=1.0, l=0.5, m=1))
    serialization.from_json(fresh, payload)

    print(f"  fresh.pin_list['VDD'].role        = {fresh.pin_list['VDD'].role!r}")
    print(f"  fresh.pin_list['VSS'].role        = {fresh.pin_list['VSS'].role!r}")
    print(f"  fresh.net_dict['IBIAS'].max_length_um = "
          f"{fresh.net_dict['IBIAS'].max_length_um}")
    print(f"  fresh.list_constraints() yields {len(fresh.list_constraints())} "
          f"device-relational constraint(s)")
    return fresh


if __name__ == "__main__":
    mod = build_current_mirror()
    show_intent(mod)
    payload = show_serialized(mod)
    round_trip(payload)
