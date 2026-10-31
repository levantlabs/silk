"""Round-trip tests for silk-canonical serialization.

The wire format is silk's own. It carries three kinds of entries in
one flat array:

  1. **Real constraints** — the 4 device-relational types from
     ``silk.constraints.types``. Carry the full audit envelope (id,
     source, …). One entry per constraint.
  2. **PortRoles overlay** — exactly one entry per module, carrying a
     ``{port_name: role_string}`` dict that captures every port with
     a non-None ``PortRole``. Replaces the old per-role-type entries
     (PowerPorts / GroundPorts / ClockPorts / AnalogPorts /
     DigitalPorts).
  3. **NetIntent overlay** — one entry per net that has any
     silk-native layout intent set (``max_ir_drop_mV``,
     ``max_j_factor``, ``max_length_um``, ``is_critical=True``, or
     ``domain``). Only-set fields are emitted.

The discriminator and audit envelope remain flat keys on each constraint.
Port and net overlays preserve silk's native state.
"""
import json

import pytest


@pytest.fixture(autouse=True)
def _patch():
    import silk.constraints  # installs Module patch


# ===========================================================================
# Real constraints — the 4 device-relational types
# ===========================================================================

def _populate_device_constraints(mod):
    from silk.constraints import (
        SymmetricBlocks, GroupBlocks, SameTemplate, ProximityConstraint,
    )
    mod.add_constraint(SymmetricBlocks(pairs=[["mn1", "mp1"]], direction="V"))
    mod.add_constraint(GroupBlocks(name="diff", instances=["mn1", "mp1"]))
    mod.add_constraint(SameTemplate(instances=["mn1", "mp1"]))
    mod.add_constraint(
        ProximityConstraint(instance_a="mn1", instance_b="mp1", max_distance_um=5)
    )


def test_to_json_emits_real_constraint_types(basic_netlist):
    from silk.constraints import serialization
    _populate_device_constraints(basic_netlist)
    data = json.loads(serialization.to_json(basic_netlist))
    types = {obj["constraint"] for obj in data}
    # Real constraints (overlays absent because no port roles or net
    # intent set on basic_netlist).
    assert {"SymmetricBlocks", "GroupBlocks", "SameTemplate",
            "ProximityConstraint"}.issubset(types)


def test_envelope_fields_present_on_real_constraints(basic_netlist):
    from silk.constraints import serialization
    _populate_device_constraints(basic_netlist)
    data = json.loads(serialization.to_json(basic_netlist))
    real = next(o for o in data if o["constraint"] == "SymmetricBlocks")
    for k in ("id", "source", "confidence", "severity", "created_at"):
        assert k in real


def test_real_constraint_round_trip(basic_netlist, pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.constraints import serialization
    _populate_device_constraints(basic_netlist)
    payload = serialization.to_json(basic_netlist)
    fresh = Module("PDK_skywater130", tmp_dir, "round.cir", pdk_details=pdk)
    serialization.from_json(fresh, payload)
    assert fresh.list_constraints() == basic_netlist.list_constraints()


# ===========================================================================
# PortRoles overlay
# ===========================================================================

def _module_with_typed_ports(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import Port
    mod = Module("PDK_skywater130", tmp_dir, "ports.cir", pdk_details=pdk)
    mod.define_module_ports({
        "VDD":  Port.supply(),
        "VSS":  Port.ground(),
        "CLK":  Port.clock_input(),
        "VINP": Port.analog_input(),
        "DOUT": Port.digital_output(),
    })
    return mod


def test_port_roles_overlay_is_single_entry(pdk, tmp_dir):
    from silk.constraints import serialization
    mod = _module_with_typed_ports(pdk, tmp_dir)
    data = json.loads(serialization.to_json(mod))
    role_entries = [o for o in data if o["constraint"] == "PortRoles"]
    assert len(role_entries) == 1


def test_port_roles_overlay_carries_dict(pdk, tmp_dir):
    from silk.constraints import serialization
    mod = _module_with_typed_ports(pdk, tmp_dir)
    [entry] = [o for o in json.loads(serialization.to_json(mod))
               if o["constraint"] == "PortRoles"]
    assert entry["roles"] == {
        "VDD":  "power",
        "VSS":  "ground",
        "CLK":  "clock",
        "VINP": "analog",
        "DOUT": "digital",
    }


def test_no_port_roles_entry_when_no_roles_set(empty_netlist):
    from silk.constraints import serialization
    data = json.loads(serialization.to_json(empty_netlist))
    assert not any(o["constraint"] == "PortRoles" for o in data)


def test_port_roles_only_includes_ports_with_a_role(pdk, tmp_dir):
    """Ports declared with PortDir but no role do NOT appear in the entry."""
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import Port, PortDir
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "mix.cir", pdk_details=pdk)
    mod.define_module_ports({
        "VDD":     Port.supply(),
        "UNTYPED": PortDir.IN,        # no role
    })
    [entry] = [o for o in json.loads(serialization.to_json(mod))
               if o["constraint"] == "PortRoles"]
    assert "UNTYPED" not in entry["roles"]
    assert entry["roles"] == {"VDD": "power"}


def test_port_roles_import_sets_pin_role(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "imp.cir", pdk_details=pdk)
    mod.define_module_ports({n: PortDir.IN for n in
                             ("VDD", "VSS", "CLK", "VINP", "DOUT")})
    payload = json.dumps([
        {"constraint": "PortRoles",
         "roles": {"VDD": "power", "VSS": "ground", "CLK": "clock",
                   "VINP": "analog", "DOUT": "digital"}},
    ])
    serialization.from_json(mod, payload)
    assert mod.pin_list["VDD"].role  == "power"
    assert mod.pin_list["VSS"].role  == "ground"
    assert mod.pin_list["CLK"].role  == "clock"
    assert mod.pin_list["VINP"].role == "analog"
    assert mod.pin_list["DOUT"].role == "digital"
    # Overlay never appears in list_constraints().
    assert mod.list_constraints() == []


def test_port_roles_round_trip_lossless(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir
    from silk.constraints import serialization
    mod = _module_with_typed_ports(pdk, tmp_dir)
    payload = serialization.to_json(mod)

    fresh = Module("PDK_skywater130", tmp_dir, "rt.cir", pdk_details=pdk)
    fresh.define_module_ports({n: PortDir.IN for n in
                               ("VDD", "VSS", "CLK", "VINP", "DOUT")})
    serialization.from_json(fresh, payload)

    for name, original in mod.pin_list.items():
        assert fresh.pin_list[name].role == original.role


def test_port_roles_round_trip_preserves_inferred_provenance(pdk, tmp_dir):
    """Pre-fix (measured on the pre-TASK-97 code): ``_apply_port_roles``
    set ``pin.role`` directly with no notion of provenance at all, so a
    role that ``infer_port_roles`` filled from a '*vdd*' name-pattern
    guess (``role_inferred=True``, TASK-42) came back
    ``role_inferred=False`` -- byte-identical to a user's declared
    ``Port.ground()`` -- after nothing but an export/import round trip.
    That silently defeats INT-004's guessed-POWER guard (see
    ``test_inferred_provenance_survives_round_trip_for_int004`` in
    test_interface_rules.py). Now the additive, sparse ``inferred`` map
    (TASK-97) carries the guess across, and a declared role keeps
    ``role_inferred=False``."""
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import Port, PortDir
    from silk.circuit.port_roles import RoleRuleset, infer_port_roles
    from silk.constraints import serialization

    mod = Module("PDK_skywater130", tmp_dir, "prov.cir", pdk_details=pdk)
    mod.define_module_ports({
        "VDD": PortDir.IN,      # '*vdd*' glob -> inferred POWER
        "VSS": Port.ground(),   # declared
    })
    infer_port_roles(mod, RoleRuleset.default())
    assert mod.pin_list["VDD"].role_inferred is True
    assert mod.pin_list["VSS"].role_inferred is False

    payload = serialization.to_json(mod)
    [entry] = [o for o in json.loads(payload) if o["constraint"] == "PortRoles"]
    # Sparse, like ``roles`` itself: only the guessed port is listed.
    assert entry["inferred"] == {"VDD": True}

    fresh = Module("PDK_skywater130", tmp_dir, "rt.cir", pdk_details=pdk)
    fresh.define_module_ports({"VDD": PortDir.IN, "VSS": PortDir.IN})
    serialization.from_json(fresh, payload)
    assert fresh.pin_list["VDD"].role == "power"
    assert fresh.pin_list["VDD"].role_inferred is True
    assert fresh.pin_list["VSS"].role == "ground"
    assert fresh.pin_list["VSS"].role_inferred is False


def test_port_roles_import_without_inferred_key_is_all_declared(pdk, tmp_dir):
    """A pre-TASK-97 file has no ``inferred`` key at all -- the documented
    backward-compat contract: every role in it imports as declared."""
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import PortDir
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "old.cir", pdk_details=pdk)
    mod.define_module_ports({"VDD": PortDir.IN})
    payload = json.dumps([
        {"constraint": "PortRoles", "roles": {"VDD": "power"}},
    ])
    serialization.from_json(mod, payload)
    assert mod.pin_list["VDD"].role == "power"
    assert mod.pin_list["VDD"].role_inferred is False


def test_port_roles_warns_on_unknown_port(pdk, tmp_dir, recwarn):
    from silk.circuit.modules import Module
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "warn.cir", pdk_details=pdk)
    payload = json.dumps([
        {"constraint": "PortRoles", "roles": {"GHOST": "power"}},
    ])
    serialization.from_json(mod, payload)
    msgs = [str(w.message) for w in recwarn.list]
    assert any("GHOST" in m for m in msgs)


# ===========================================================================
# NetIntent overlay
# ===========================================================================

def test_net_intent_one_entry_per_net_with_intent(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "ni.cir", pdk_details=pdk)
    mod.add_net("vdd_int")
    mod.add_net("clk_int")
    mod.add_net("untouched")
    mod.net_dict["vdd_int"].max_ir_drop_mV = 50.0
    mod.net_dict["vdd_int"].max_j_factor   = 0.7
    mod.net_dict["clk_int"].max_length_um  = 75.0
    mod.net_dict["clk_int"].is_critical    = True

    data = json.loads(serialization.to_json(mod))
    intents = [o for o in data if o["constraint"] == "NetIntent"]
    assert {e["net"] for e in intents} == {"vdd_int", "clk_int"}


def test_net_intent_only_emits_set_fields(pdk, tmp_dir):
    """An entry must NOT contain keys for fields that are at default."""
    from silk.circuit.modules import Module
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "ni2.cir", pdk_details=pdk)
    mod.add_net("clk_int")
    mod.net_dict["clk_int"].max_length_um = 75.0
    mod.net_dict["clk_int"].is_critical = True

    [entry] = [o for o in json.loads(serialization.to_json(mod))
               if o["constraint"] == "NetIntent"]
    # Only the set fields should be present.
    assert entry == {
        "constraint": "NetIntent",
        "net": "clk_int",
        "max_length_um": 75.0,
        "is_critical": True,
    }


def test_net_intent_carries_voltage_domain(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.circuit.voltage_domain import VoltageDomain
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "nd.cir", pdk_details=pdk)
    avdd = VoltageDomain("avdd_1v8", nominal=1.8, tolerance=0.1)
    mod.add_net("vdd_int")
    mod.assign_net_domain("vdd_int", avdd)

    [entry] = [o for o in json.loads(serialization.to_json(mod))
               if o["constraint"] == "NetIntent"]
    assert entry == {
        "constraint": "NetIntent",
        "net": "vdd_int",
        "domain": {"name": "avdd_1v8", "nominal": 1.8, "tolerance": 0.1},
    }


def test_no_net_intent_entry_when_nothing_set(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "ni3.cir", pdk_details=pdk)
    mod.add_net("plain")
    data = json.loads(serialization.to_json(mod))
    assert not any(o["constraint"] == "NetIntent" for o in data)


def test_net_intent_import_routes_to_net_fields(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "ni_in.cir", pdk_details=pdk)
    mod.add_net("vdd_int")
    mod.add_net("clk_int")
    payload = json.dumps([
        {"constraint": "NetIntent", "net": "vdd_int",
         "max_ir_drop_mV": 25.0, "max_j_factor": 0.5,
         "domain": {"name": "avdd_1v8", "nominal": 1.8, "tolerance": 0.1}},
        {"constraint": "NetIntent", "net": "clk_int",
         "max_length_um": 100.0, "is_critical": True},
    ])
    serialization.from_json(mod, payload)

    n_vdd = mod.net_dict["vdd_int"]
    assert n_vdd.max_ir_drop_mV == 25.0
    assert n_vdd.max_j_factor == 0.5
    assert n_vdd.domain.name == "avdd_1v8"
    assert n_vdd.domain.nominal == 1.8

    n_clk = mod.net_dict["clk_int"]
    assert n_clk.max_length_um == 100.0
    assert n_clk.is_critical is True

    assert mod.list_constraints() == []


def test_net_intent_round_trip_lossless(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.circuit.voltage_domain import VoltageDomain
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "nirt.cir", pdk_details=pdk)
    avdd = VoltageDomain("avdd_1v8", nominal=1.8, tolerance=0.1)
    mod.add_net("vdd_int")
    mod.add_net("clk_int")
    mod.net_dict["vdd_int"].max_ir_drop_mV = 50.0
    mod.net_dict["vdd_int"].max_j_factor   = 0.7
    mod.assign_net_domain("vdd_int", avdd)
    mod.net_dict["clk_int"].max_length_um  = 75.0
    mod.net_dict["clk_int"].is_critical    = True

    payload = serialization.to_json(mod)

    fresh = Module("PDK_skywater130", tmp_dir, "nirt2.cir", pdk_details=pdk)
    fresh.add_net("vdd_int")
    fresh.add_net("clk_int")
    serialization.from_json(fresh, payload)

    for name in ("vdd_int", "clk_int"):
        for attr in ("max_ir_drop_mV", "max_j_factor",
                     "max_length_um", "is_critical"):
            assert getattr(fresh.net_dict[name], attr) == \
                   getattr(mod.net_dict[name], attr), (name, attr)
    assert fresh.net_dict["vdd_int"].domain.name == "avdd_1v8"


def test_net_intent_warns_on_unknown_net(pdk, tmp_dir, recwarn):
    from silk.circuit.modules import Module
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "niw.cir", pdk_details=pdk)
    payload = json.dumps([
        {"constraint": "NetIntent", "net": "ghost", "max_ir_drop_mV": 10},
    ])
    serialization.from_json(mod, payload)
    msgs = [str(w.message) for w in recwarn.list]
    assert any("ghost" in m for m in msgs)


# ===========================================================================
# Mixed payload + helpers
# ===========================================================================

def test_mixed_payload_round_trip(pdk, tmp_dir):
    """A combined JSON use: device constraints + port roles + net
    intents + voltage domain in one file. After round-trip everything
    lands in its silk-native location."""
    from silk.circuit.modules import Module
    from silk.circuit.circuit_elements import Port, PortDir
    from silk.circuit.voltage_domain import VoltageDomain
    from silk.constraints import SymmetricBlocks, serialization

    mod = Module("PDK_skywater130", tmp_dir, "mix.cir", pdk_details=pdk)
    mod.define_module_ports({"VDD": Port.supply(), "VSS": Port.ground(),
                             "VOUT": Port.analog_output()})
    mod.add_net("vdd_int")
    mod.assign_net_domain("vdd_int",
                          VoltageDomain("avdd_1v8", nominal=1.8, tolerance=0.1))
    mod.net_dict["vdd_int"].max_ir_drop_mV = 30.0
    mod.add(pdk.nmos1p8("MN0", d="vdd_int", g="vdd_int", s="vdd_int", b="vdd_int", w=1, l=0.15))
    mod.add(pdk.nmos1p8("MN1", d="vdd_int", g="vdd_int", s="vdd_int", b="vdd_int", w=1, l=0.15))
    mod.add_constraint(SymmetricBlocks(pairs=[["MN0", "MN1"]], direction="V"))

    payload = serialization.to_json(mod)

    fresh = Module("PDK_skywater130", tmp_dir, "mix_fresh.cir", pdk_details=pdk)
    fresh.define_module_ports({n: PortDir.IN for n in ("VDD", "VSS")})
    fresh.define_module_ports({"VOUT": PortDir.OUT})
    fresh.add_net("vdd_int")
    fresh.add(pdk.nmos1p8("MN0", d="vdd_int", g="vdd_int", s="vdd_int", b="vdd_int", w=1, l=0.15))
    fresh.add(pdk.nmos1p8("MN1", d="vdd_int", g="vdd_int", s="vdd_int", b="vdd_int", w=1, l=0.15))
    serialization.from_json(fresh, payload)

    assert len(fresh.list_constraints()) == 1
    assert fresh.list_constraints()[0].constraint == "SymmetricBlocks"
    assert fresh.net_dict["vdd_int"].max_ir_drop_mV == 30.0
    assert fresh.net_dict["vdd_int"].domain.name == "avdd_1v8"
    assert fresh.pin_list["VDD"].role  == "power"
    assert fresh.pin_list["VOUT"].role == "analog"


def test_from_obj_returns_only_real_constraints():
    """from_obj is side-effect-free; returns only typed constraints,
    drops overlay entries silently (no module to apply them to)."""
    from silk.constraints import serialization
    out = serialization.from_obj([
        {"constraint": "SymmetricBlocks", "pairs": [["A", "B"]], "direction": "V"},
        {"constraint": "PortRoles", "roles": {"VDD": "power"}},
        {"constraint": "NetIntent", "net": "n", "max_ir_drop_mV": 10},
    ])
    assert len(out) == 1
    assert out[0].constraint == "SymmetricBlocks"


def test_to_json_indent_default_is_pretty(basic_netlist):
    from silk.constraints import serialization
    _populate_device_constraints(basic_netlist)
    s = serialization.to_json(basic_netlist)
    assert "\n" in s


def test_net_intent_round_trips_switchable_domain(pdk, tmp_dir):
    from silk.circuit.modules import Module
    from silk.circuit.voltage_domain import VoltageDomain
    from silk.constraints import serialization
    mod = Module("PDK_skywater130", tmp_dir, "sw.cir", pdk_details=pdk)
    mod.add_net("vdd_gated")
    mod.add_net("vdd_on")
    mod.assign_net_domain("vdd_gated", VoltageDomain("dvdd", nominal=0.9, switchable=True))
    mod.assign_net_domain("vdd_on", VoltageDomain("avdd", nominal=1.8))

    payload = serialization.to_json(mod)
    by_net = {o["net"]: o for o in json.loads(payload) if o["constraint"] == "NetIntent"}
    assert by_net["vdd_gated"]["domain"]["switchable"] is True
    assert "switchable" not in by_net["vdd_on"]["domain"]   # default omitted

    fresh = Module("PDK_skywater130", tmp_dir, "sw2.cir", pdk_details=pdk)
    fresh.add_net("vdd_gated")
    fresh.add_net("vdd_on")
    serialization.from_json(fresh, payload)
    assert fresh.net_dict["vdd_gated"].domain.switchable is True
    assert fresh.net_dict["vdd_on"].domain.switchable is False
