"""Self-contained circuit battery + observation helpers for the Module
file-decomposition golden harness.

This module is **deliberately decoupled from silk/examples/**: the
example builders are under active editing in a parallel workstream, so binding
the refactor goldens to them would let unrelated edits spuriously break the
safety net. Instead we build compact, deterministic representatives of the
five named battery shapes (diff pair, 5T OTA, flash core, hierarchical
inverter→buffer→tb, bus) right here.

Both ``scripts/capture_module_goldens.py`` (writes goldens on the pre-split
commit) and ``tests/refactor/test_module_golden.py`` (replays them) import
``build_battery`` and the ``*_of`` observers from here, guaranteeing identical
circuits on both sides of the split.

Determinism notes:
- SPICE netlist emission carries no timestamps/RNG (verified); the ``.lib``
  line embeds a *fixed* fake path (``_PDK_LIB``) that is never opened.
- ``silk.constraints`` is imported so ``Module.add_constraint`` /
  ``list_constraints`` are installed consistently for the ``dir(Module)``
  snapshot.
"""
from __future__ import annotations

import inspect
import json
import os
from collections import OrderedDict

import silk.constraints  # noqa: F401 — installs Module.add_constraint/list_constraints
from silk.constraints import GroupBlocks, ProximityConstraint, SymmetricBlocks
from silk.circuit.modules import Module
from silk.circuit.circuit_elements import Port, PortDir
from silk.pdk.basepdk import BasePDK
from silk import sources
from silk.elements import elements
from silk.simulation import analysis
from silk.db.structure_tracker import CircuitStructureTracker

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
_PDK_YAML = os.path.join(
    _REPO, "silk", "pdk", "data", "skywater_130_device_details.yaml"
)
# Fixed, deterministic, never-opened path — keeps the ``.lib`` line stable and
# machine-independent in the committed goldens.
_PDK_LIB = "/fake/sky130.lib.spice"

N_BITS = 4


def _pdk():
    return BasePDK(_PDK_YAML, library_location=_PDK_LIB)


# ---------------------------------------------------------------------------
# 1. Differential pair (subckt) — Port roles + constraints (monkeypatched API)
# ---------------------------------------------------------------------------
def build_diff_pair(pdk) -> Module:
    mod = Module(
        "PDK_skywater130", "/tmp", "diff_pair.cir",
        pdk_details=pdk, module_name="diff_pair",
    )
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "VBIAS": Port.analog_input(),
        "VIN_P": Port.analog_input(), "VIN_M": Port.analog_input(),
        "VOUT_P": Port.analog_output(), "VOUT_N": Port.analog_output(),
    })
    mod.add_net("tail")
    mod.net_dict["tail"].is_critical = True
    mod.net_dict["tail"].max_length_um = 25.0

    mod.add_element(pdk.nmos1p8, "M1",
                    nets={"d": "VOUT_N", "g": "VIN_P", "s": "tail", "b": "VSS"},
                    parameters={"w": 2.0, "l": 0.5, "m": 1})
    mod.add_element(pdk.nmos1p8, "M2",
                    nets={"d": "VOUT_P", "g": "VIN_M", "s": "tail", "b": "VSS"},
                    parameters={"w": 2.0, "l": 0.5, "m": 1})
    mod.add_element(pdk.nmos1p8, "M_tail",
                    nets={"d": "tail", "g": "VBIAS", "s": "VSS", "b": "VSS"},
                    parameters={"w": 4.0, "l": 1.0, "m": 1})

    mod.add_constraint(SymmetricBlocks(
        pairs=[["M1", "M2"]], direction="V",
        rationale="differential input pair across vertical axis"))
    mod.add_constraint(GroupBlocks(
        name="input_pair", instances=["M1", "M2"],
        generator={"name": "MOS", "parameters": {"pattern": "cc"}},
        rationale="common-centroid finger layout for matching"))
    mod.add_constraint(ProximityConstraint(
        instance_a="M1", instance_b="M_tail", max_distance_um=8.0,
        severity="soft", rationale="keep tail close to pair"))
    return mod


# ---------------------------------------------------------------------------
# 2. 5T OTA (subckt) — plus a sim type + a spec (Sim/Spec subsystems)
# ---------------------------------------------------------------------------
def build_ota_5t(pdk, module_name="ota_5t") -> Module:
    mod = Module(
        "PDK_skywater130", "/tmp", f"{module_name}.cir",
        pdk_details=pdk, module_name=module_name,
    )
    mod.define_module_ports({
        "vdd": Port.supply(), "vss": Port.ground(),
        "vbias": Port.analog_input(),
        "vinp": Port.analog_input(), "vinn": Port.analog_input(),
        "vout": Port.analog_output(),
    })
    mod.add_net("tail")
    mod.add_net("nd1")
    # NMOS input pair
    mod.add_element(pdk.nmos1p8, "M1",
                    nets={"d": "nd1", "g": "vinp", "s": "tail", "b": "vss"},
                    parameters={"w": 2.0, "l": 0.5, "m": 1})
    mod.add_element(pdk.nmos1p8, "M2",
                    nets={"d": "vout", "g": "vinn", "s": "tail", "b": "vss"},
                    parameters={"w": 2.0, "l": 0.5, "m": 1})
    # PMOS mirror load
    mod.add_element(pdk.pmos1p8, "M3",
                    nets={"d": "nd1", "g": "nd1", "s": "vdd", "b": "vdd"},
                    parameters={"w": 4.0, "l": 0.5, "m": 1})
    mod.add_element(pdk.pmos1p8, "M4",
                    nets={"d": "vout", "g": "nd1", "s": "vdd", "b": "vdd"},
                    parameters={"w": 4.0, "l": 0.5, "m": 1})
    # NMOS tail
    mod.add_element(pdk.nmos1p8, "M5",
                    nets={"d": "tail", "g": "vbias", "s": "vss", "b": "vss"},
                    parameters={"w": 4.0, "l": 1.0, "m": 1})

    # Spec exercises the Spec subsystem; sim types are testbench-only and are
    # covered by the hierarchical tb below.
    mod.declare_spec("dc_gain", metric="dc_gain_dB", op=">=", target=40.0,
                     units="dB", description="minimum open-loop gain")
    return mod


# ---------------------------------------------------------------------------
# 3. Flash core (subckt) — resistor ladder (passives) + 3 comparator instances
#    (hierarchy + multi-instance of one subckt)
# ---------------------------------------------------------------------------
def _build_comparator(pdk) -> Module:
    return build_ota_5t(pdk, module_name="comparator")


def build_flash_core(pdk) -> Module:
    comp = _build_comparator(pdk)
    mod = Module(
        "PDK_skywater130", "/tmp", "flash_core.cir",
        pdk_details=pdk, module_name="flash_core",
    )
    mod.define_module_ports({
        "vin": Port.analog_input(), "vref": Port.analog_input(),
        "vbias": Port.analog_input(),
        "vdd": Port.supply(), "vss": Port.ground(),
        "out0": Port.analog_output(), "out1": Port.analog_output(),
        "out2": Port.analog_output(),
    })
    # Reference ladder: vref → t1 → t2 → t3 → vss
    taps = ["t1", "t2", "t3"]
    for t in taps:
        mod.add_net(t)
    ladder_nodes = ["vref"] + taps + ["vss"]
    with mod:
        for i in range(len(ladder_nodes) - 1):
            elements.resistor(f"Rlad{i}", p=ladder_nodes[i],
                              n=ladder_nodes[i + 1], value=1e3)
    # Comparators tap the ladder
    for i, t in enumerate(taps):
        mod.add_module(comp, f"comp{i}", nets={
            "vinp": "vin", "vinn": t, "vout": f"out{i}",
            "vbias": "vbias", "vdd": "vdd", "vss": "vss",
        })
    return mod


# ---------------------------------------------------------------------------
# 4. Hierarchical top-level testbench — inverter → buffer → tb (top netlist)
# ---------------------------------------------------------------------------
# The ground PORT on these two blocks is "vss", not "gnd": since TASK-36 a
# declared port named "gnd" is rejected at netlist emission (ngspice folds the
# port name onto global node 0 and silently discards the parent's wiring). The
# ground NET stays "gnd" everywhere, so only the two .subckt headers changed
# when the goldens were recaptured.
def _build_inverter(pdk) -> Module:
    inv = Module("PDK_skywater130", "/tmp", "inv.cir",
                 pdk_details=pdk, module_name="inverter")
    inv.define_module_ports({
        "vin": PortDir.IN, "vout": PortDir.OUT,
        "vdd": PortDir.POWER, "vss": PortDir.GROUND,
    })
    with inv:
        pdk.nmos1p8("mn", d="vout", g="vin", s="vss", b="vss", w=1.0, l=0.15)
        pdk.pmos1p8("mp", d="vout", g="vin", s="vdd", b="vdd", w=2.0, l=0.15)
    return inv


def _build_buffer(pdk) -> Module:
    buf = Module("PDK_skywater130", "/tmp", "buf.cir",
                 pdk_details=pdk, module_name="buffer")
    buf.define_module_ports({
        "vin": PortDir.IN, "vout": PortDir.OUT,
        "vdd": PortDir.POWER, "vss": PortDir.GROUND,
    })
    inv = _build_inverter(pdk)
    buf.add_module(inv, "inv1",
                    nets={"vin": "vin", "vout": "vmid", "vdd": "vdd", "vss": "vss"})
    buf.add_module(inv, "inv2",
                    nets={"vin": "vmid", "vout": "vout", "vdd": "vdd", "vss": "vss"})
    return buf


def build_hier_tb(pdk, dir_folder) -> Module:
    """Top-level testbench (module_name=None → writes a full sim netlist)."""
    buf = _build_buffer(pdk)
    tb = Module("PDK_skywater130", dir_folder, "hier_tb.cir",
                pdk_details=pdk, module_name=None)
    tb.add_module(buf, "xbuf",
                   nets={"vin": "vin", "vout": "vout", "vdd": "vdd", "vss": "gnd"})
    with tb:
        sources.VoltageSource("vdd_src", positive="vdd", negative="0", dc=1.8)
        sources.VoltageSource("vin_src", positive="vin", negative="0",
                              pulse=sources.Pulse(0, 1.8, td=0, tr=1e-9,
                                                  tf=1e-9, pw=10e-9, per=20e-9))
        elements.capacitor("cload", p="vout", n="0", value=10e-15)
    tb.add_sim_type(analysis.Transient(step=1e-9, stop=40e-9))
    return tb


# ---------------------------------------------------------------------------
# 5. Bus testbench — counter (bus out) → dac (bus in) (bus expansion, no PDK)
# ---------------------------------------------------------------------------
def build_bus_tb() -> Module:
    counter = Module(pdk_name="sky130", dir_folder="/tmp",
                     netlistname="counter4.cir", pdk_details=None,
                     module_name="counter4")
    counter.define_module_ports({
        "clk": Port.clock_input(),
        "count": Port.digital_output(bits=N_BITS),
    })
    dac = Module(pdk_name="sky130", dir_folder="/tmp",
                 netlistname="dac4.cir", pdk_details=None, module_name="dac4")
    dac.define_module_ports({
        "code": Port.digital_input(bits=N_BITS),
        "vout": Port.analog_output(),
    })
    tb = Module(pdk_name="sky130", dir_folder="/tmp",
                netlistname="bus_tb.cir", pdk_details=None,
                module_name="bus_demo_tb")
    tb.add_net("clk")
    tb.add_net("bus", bits=N_BITS)
    tb.add_net("vout")
    tb.add_module(counter, "xcount",
                   clk=tb.nets.clk, count=tb.nets.bus)
    tb.add_module(dac, "xdac",
                   code=tb.nets.bus, vout=tb.nets.vout)
    return tb


# ---------------------------------------------------------------------------
# Battery assembly (deterministic order)
# ---------------------------------------------------------------------------
def build_battery(dir_folder) -> "OrderedDict[str, Module]":
    """Build all battery circuits. ``dir_folder`` receives the top-level
    testbench netlist file (the only circuit that writes to disk)."""
    pdk = _pdk()
    battery = OrderedDict()
    battery["diff_pair"] = build_diff_pair(pdk)
    battery["ota_5t"] = build_ota_5t(pdk)
    battery["flash_core"] = build_flash_core(pdk)
    battery["hier_tb"] = build_hier_tb(pdk, dir_folder)
    battery["bus_tb"] = build_bus_tb()
    return battery


# Deterministic battery order (mirrors build_battery insertion order).
BATTERY_NAMES = ("diff_pair", "ota_5t", "flash_core", "hier_tb", "bus_tb")
# Which circuits emit a SPICE netlist we can byte-compare. The bus tb is
# behavioral with pdk_details=None → no SPICE emission.
NETLIST_CIRCUITS = ("diff_pair", "ota_5t", "flash_core", "hier_tb")
# Circuits whose device-path fault enumeration is meaningful (have devices).
FAULT_CIRCUITS = ("diff_pair", "ota_5t", "flash_core", "hier_tb")


# ---------------------------------------------------------------------------
# Observers — all deterministic
# ---------------------------------------------------------------------------
def netlist_of(name, mod, dir_folder) -> str:
    """Return the circuit's SPICE netlist as a string.

    Subckt modules (``module_name`` set) return the ``.subckt`` block
    directly; the top-level tb (``module_name is None``) writes to
    the returned deck path and we read it back.
    """
    mod.set_output_options(mode='overwrite')
    out = mod.generate_netlist()
    if mod.module_name is None:
        with open(out, "r") as fh:
            out = fh.read()
    # The top-level tb embeds its results directory in the `.control`
    # `write .../rawfile.raw` line. That directory is a per-run temp path, so
    # normalize it to a fixed token to keep the golden machine-independent.
    return out.replace(dir_folder, "<TBDIR>")


def schema_of(mod) -> str:
    return json.dumps(mod.export_schema(), sort_keys=True, indent=2, default=str)


def hash_of(mod) -> str:
    tracker = CircuitStructureTracker(None, None)
    struct = tracker.extract_structure(mod)
    structure_hash, _module_hashes = tracker.compute_structure_hash(struct)
    return structure_hash


def faults_of(mod) -> str:
    return "\n".join(sorted(mod._silk_fault_all_device_paths()))


def module_api_snapshot() -> dict:
    """Frozen ``dir(Module)`` name list + per-callable signature.

    Properties (``nets``/``backend``/``simulator``/``netlister``) and any
    non-signature-bearing attrs are recorded by name with signature ``None``
    (``inspect.signature`` raises on property objects). ``silk.constraints`` is
    imported at module load, so ``add_constraint``/``list_constraints`` are
    present and captured consistently.

    Some inherited ``object``/``type`` dunders are excluded or normalized
    because CPython varies them by version, and silk supports a range
    (``requires-python >= 3.10``). None are part of silk's own API, so freezing
    them would make the golden fail purely on interpreter version, not on any
    real API change — a genuine drop of a silk-owned method is still caught.

    - **Excluded from the name list** (their *presence* is version-dependent):
      ``__getstate__`` (added to ``object`` in 3.11, bpo-26579); and
      ``__firstlineno__`` / ``__static_attributes__`` (added to every class in
      3.13). ``Module`` pickles via ``__reduce_ex__``/``__setstate__`` and
      never touches any of them.
    - **Signature normalized to None** (they are present on every version, but
      their signature *repr* is version-dependent): ``__init_subclass__`` and
      ``__subclasshook__`` — implicit classmethods on ``object`` that
      ``inspect`` reports as ``None`` on 3.12 and as a signature on 3.14. The
      golden already records them as ``None``, so pinning them there keeps it
      version-stable.
    """
    _VERSION_VARIANT_DUNDERS = {
        "__getstate__", "__firstlineno__", "__static_attributes__",
    }
    _VERSION_VARIANT_SIGNATURES = {"__init_subclass__", "__subclasshook__"}
    names = sorted(n for n in dir(Module) if n not in _VERSION_VARIANT_DUNDERS)
    sigs = {}
    for n in names:
        if n in _VERSION_VARIANT_SIGNATURES:
            sigs[n] = None
            continue
        attr = inspect.getattr_static(Module, n, None)
        if isinstance(attr, (staticmethod, classmethod, property)):
            sigs[n] = None
            continue
        try:
            sigs[n] = str(inspect.signature(getattr(Module, n)))
        except (TypeError, ValueError):
            sigs[n] = None
    return {"names": names, "signatures": sigs}
