"""Shared transistor cells, characterization tests and behavioral models.

Builders provide inverters, buffers, NAND and NOR gates, a multiplexer, XOR
variants and a positive-edge D flip-flop with active-low clear. Combinational
cells include truth-table and propagation-delay tests. The flip-flop includes
capture and clock-to-Q tests; ``sweep_setup_time()`` measures capture and
clock-to-Q across data-to-clock spacings.

``build_xor2(variant="static")`` uses 12 transistors, including two input
inverters. The NAND form uses 16, and the pass-transistor form uses four.
The static form has 25% fewer transistors than the NAND form; device count
does not measure layout area. The pass-transistor form can degrade output
levels. Its behavioral model emits ideal rails, so compare levels in SPICE.

Behavioral models apply fixed delays. The script calibrates them from the
measured timing. These models support feed-forward paths; composed sequential
feedback needs a model with state and explicit point delays.

Characterization defaults to sky130 tt and 1.8 V. Combinational tests use a
15 fF load. Builders expose
sizing parameters, but there is no drive-strength ladder or load sweep. These
are electrical tests, without layout or power characterization.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.stdcells`` from the
repository root with ngspice and sky130 models installed. It prints truth and
timing checks, timing drift, XOR levels, a setup sweep and a cell catalog.
It saves ``stdcells.png`` in the configured examples output directory."""
import itertools
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk.elements import elements
from silk import load_pdk
from silk import sources
from silk.simulation import Transient
from silk.examples import example_output_dir
from silk.metrics import propagation_delay, value_at

pdk = load_pdk('sky130')

# --- the shared sizing table (one place; every cell reads from here) -----------
_VDD = 1.8
_L = 0.15              # channel length (um) — min-L: these are speed cells
_WN = 1.0             # unit NMOS width (um)
_WP = 2.0             # unit PMOS width (um) — ~2x for mobility-matched rise/fall
# Series stacks are widened to hold on-resistance ~constant as devices stack
# (a 2-high stack has ~2x the series R of a lone device). Multiple = #-in-series.
# NANDs stack NMOS (widen NMOS); NORs stack PMOS (widen the already-2x PMOS).
_STACK_WIDEN = {1: 1, 2: 2, 3: 3}

# Supported implementation variants.
_CELL_VARIANTS = {
    "inverter": ("x1",), "buf": ("x1",),
    "nand2": ("x1",), "nand3": ("x1",),
    "nor2": ("x1",), "nor3": ("x1",),
    "dff": ("x1",),
    "mux2": ("tg",),                      # transmission-gate 2:1
    "xor2": ("nand", "static", "ptl"),   # topology variants — see module docstring
}

# Nominal propagation delays (ps), measured into the 15 fF characterization
# load at sky130 tt, 1.8 V and 27 C. The script prints measured drift.
_TP_NOM_PS = {
    "inverter": 50.0, "buf": 71.0,
    "nand2": 49.0, "nand3": 55.0,
    "nor2": 64.0, "nor3": 90.0,
    "xor2_nand": 75.0, "xor2_static": 97.0, "xor2_ptl": 99.0,
    "mux2": 78.0,
}
_CLKQ_NOM_PS = 106.0             # DFF clk-to-Q (avg of rise/fall) for the twin


def _require_variant(variant, cell):
    allowed = _CELL_VARIANTS[cell]
    if variant not in allowed:
        raise ValueError(
            f"unknown {cell} variant {variant!r}; implemented: {allowed}. "
            f"xor2 supports static, NAND and pass-transistor topologies.")


# --- behavioral model helpers ---------------------------------------------------

def _delay_signal(y, t, tp_s):
    """Shift a waveform right by ``tp_s`` seconds (fixed propagation delay)."""
    y = np.asarray(y, dtype=float)
    if tp_s <= 0:
        return y
    return np.interp(t - tp_s, t, y, left=float(y[0]))


def _make_comb_model(in_ports, out_port, boolfn):
    """Return a rail-to-rail boolean model with a fixed propagation delay.

    ``boolfn`` takes one numpy boolean array per input port. The delay is
    ``params['tp_s']``. The model does not represent pass-transistor level loss."""
    def model(ports, t, params):
        vdd = float(params["vdd"])
        args = [np.asarray(getattr(ports, p).v, dtype=float) > vdd / 2 for p in in_ports]
        out = np.where(boolfn(*args), vdd, 0.0)
        getattr(ports, out_port).v = _delay_signal(out, t, float(params['tp_s']))
    return model


def _dff_model(ports, t, params):
    """Capture D on rising CLK edges, with asynchronous active-low clear.

    Delay the output by ``params['tcq_s']``. This whole-waveform model supports
    feed-forward use; composed sequential feedback needs explicit point delays."""
    vdd = float(params["vdd"])
    clk = np.asarray(ports.CLK.v, dtype=float) > vdd / 2
    d = np.asarray(ports.D.v, dtype=float) > vdd / 2
    clrb = np.asarray(ports.CLRB.v, dtype=float) > vdd / 2
    q_bool = np.zeros(len(t), dtype=bool)
    state = False
    for i in range(len(t)):
        if not clrb[i]:
            state = False
        elif clk[i] and i > 0 and not clk[i - 1]:      # rising edge
            state = bool(d[i])
        q_bool[i] = state
    tcq = float(params["tcq_s"])
    q = _delay_signal(np.where(q_bool, vdd, 0.0), t, tcq)
    qb = _delay_signal(np.where(q_bool, 0.0, vdd), t, tcq)
    ports.Q.v = q
    ports.QB.v = qb


# --- truth-table logic ----------------------------------------------------------

def inverter(a: int) -> int:
    return 1 - a


def buf(a: int) -> int:
    return a


def nand2(a: int, b: int) -> int:
    return 1 - (a & b)


def nand3(a: int, b: int, c: int) -> int:
    return 1 - (a & b & c)


def nor2(a: int, b: int) -> int:
    return 1 - (a | b)


def nor3(a: int, b: int, c: int) -> int:
    return 1 - (a | b | c)


def xor2(a: int, b: int) -> int:
    return a ^ b


def mux2(a: int, b: int, s: int) -> int:
    return b if s else a


# --- combinational cells --------------------------------------------------------

def build_inverter(*, variant="x1", wn=_WN, wp=_WP, l=_L, m=1, name="inv",
                   with_tests=True) -> Module:
    """CMOS inverter: ``Y = NOT A``.

    ``m`` sets the number of parallel devices. Give differently sized cells in
    one parent distinct ``name`` values, or enable ``variants`` on the cells.
    Conflicting bodies under one module name raise ``CircuitError`` at emission."""
    _require_variant(variant, "inverter")
    m_mod = Module("PDK_skywater130", example_output_dir(), f"{name}.cir", pdk_details=pdk,
                   module_name=name)
    m_mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "Y": Port.analog_output(),
    })
    with m_mod:
        pdk.pmos1p8("MP", d="Y", g="A", s="VDD", b="VDD", w=wp, l=l, m=m)
        pdk.nmos1p8("MN", d="Y", g="A", s="VSS", b="VSS", w=wn, l=l, m=m)
    m_mod.set_behavioral_model(_make_comb_model(["A"], "Y", lambda a: ~a),
                               params={"vdd": _VDD, "tp_s": _TP_NOM_PS["inverter"] * 1e-12}, execution='vector')
    if with_tests:
        attach_truth_test(m_mod, in_ports=["A"], out_port="Y", fn=inverter)
        attach_delay_test(m_mod, in_ports=["A"], out_port="Y", toggle="A", hold={})
    return m_mod


def build_buf(*, variant="x1", wn=_WN, wp=_WP, l=_L, with_tests=True) -> Module:
    """Non-inverting buffer made from two series inverters: ``Y = A``."""
    _require_variant(variant, "buf")
    m = Module("PDK_skywater130", example_output_dir(), "buf.cir", pdk_details=pdk,
               module_name="buf")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "Y": Port.analog_output(),
    })
    inv = build_inverter(wn=wn, wp=wp, l=l, with_tests=False)
    m.add_module(inv, "i1", nets={"VDD": "VDD", "VSS": "VSS", "A": "A", "Y": "mid"})
    m.add_module(inv, "i2", nets={"VDD": "VDD", "VSS": "VSS", "A": "mid", "Y": "Y"})
    m.set_behavioral_model(_make_comb_model(["A"], "Y", lambda a: a),
                           params={"vdd": _VDD, "tp_s": _TP_NOM_PS["buf"] * 1e-12}, execution='vector')
    if with_tests:
        attach_truth_test(m, in_ports=["A"], out_port="Y", fn=buf)
        attach_delay_test(m, in_ports=["A"], out_port="Y", toggle="A", hold={})
    return m


def build_nand2(*, variant="x1", wn=_WN, wp=_WP, l=_L, with_tests=True) -> Module:
    """2-input NAND: ``Y = NOT(A AND B)``. PMOS parallel, NMOS series (2x wide)."""
    _require_variant(variant, "nand2")
    m = Module("PDK_skywater130", example_output_dir(), "nand2.cir", pdk_details=pdk,
               module_name="nand2")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "B": Port.analog_input(),
        "Y": Port.analog_output(),
    })
    ws = wn * _STACK_WIDEN[2]
    with m:
        pdk.pmos1p8("MPA", d="Y", g="A", s="VDD", b="VDD", w=wp, l=l)
        pdk.pmos1p8("MPB", d="Y", g="B", s="VDD", b="VDD", w=wp, l=l)
        pdk.nmos1p8("MNA", d="Y", g="A", s="nx", b="VSS", w=ws, l=l)
        pdk.nmos1p8("MNB", d="nx", g="B", s="VSS", b="VSS", w=ws, l=l)
    m.set_behavioral_model(
        _make_comb_model(["A", "B"], "Y", lambda a, b: ~(a & b)),
        params={"vdd": _VDD, "tp_s": _TP_NOM_PS["nand2"] * 1e-12}, execution='vector')
    if with_tests:
        attach_truth_test(m, in_ports=["A", "B"], out_port="Y",
                          fn=nand2)
        attach_delay_test(m, in_ports=["A", "B"], out_port="Y", toggle="A",
                          hold={"B": 1})
    return m


def build_nand3(*, variant="x1", wn=_WN, wp=_WP, l=_L, with_tests=True) -> Module:
    """3-input NAND: ``Y = NOT(A AND B AND C)``. NMOS triple stack (3x wide)."""
    _require_variant(variant, "nand3")
    m = Module("PDK_skywater130", example_output_dir(), "nand3.cir", pdk_details=pdk,
               module_name="nand3")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "B": Port.analog_input(),
        "C": Port.analog_input(), "Y": Port.analog_output(),
    })
    ws = wn * _STACK_WIDEN[3]
    with m:
        pdk.pmos1p8("MPA", d="Y", g="A", s="VDD", b="VDD", w=wp, l=l)
        pdk.pmos1p8("MPB", d="Y", g="B", s="VDD", b="VDD", w=wp, l=l)
        pdk.pmos1p8("MPC", d="Y", g="C", s="VDD", b="VDD", w=wp, l=l)
        pdk.nmos1p8("MNA", d="Y", g="A", s="nx1", b="VSS", w=ws, l=l)
        pdk.nmos1p8("MNB", d="nx1", g="B", s="nx2", b="VSS", w=ws, l=l)
        pdk.nmos1p8("MNC", d="nx2", g="C", s="VSS", b="VSS", w=ws, l=l)
    m.set_behavioral_model(
        _make_comb_model(["A", "B", "C"], "Y", lambda a, b, c: ~(a & b & c)),
        params={"vdd": _VDD, "tp_s": _TP_NOM_PS["nand3"] * 1e-12}, execution='vector')
    if with_tests:
        attach_truth_test(m, in_ports=["A", "B", "C"], out_port="Y",
                          fn=nand3)
        attach_delay_test(m, in_ports=["A", "B", "C"], out_port="Y", toggle="A",
                          hold={"B": 1, "C": 1})
    return m


def build_nor2(*, variant="x1", wn=_WN, wp=_WP, l=_L, with_tests=True) -> Module:
    """Two-input NOR: ``Y = NOT(A OR B)``.

    PMOS devices are in series and widened by ``_STACK_WIDEN[2]``.
    NMOS devices are in parallel."""
    _require_variant(variant, "nor2")
    m = Module("PDK_skywater130", example_output_dir(), "nor2.cir", pdk_details=pdk,
               module_name="nor2")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "B": Port.analog_input(),
        "Y": Port.analog_output(),
    })
    wps = wp * _STACK_WIDEN[2]
    with m:
        pdk.pmos1p8("MPA", d="px", g="A", s="VDD", b="VDD", w=wps, l=l)
        pdk.pmos1p8("MPB", d="Y", g="B", s="px", b="VDD", w=wps, l=l)
        pdk.nmos1p8("MNA", d="Y", g="A", s="VSS", b="VSS", w=wn, l=l)
        pdk.nmos1p8("MNB", d="Y", g="B", s="VSS", b="VSS", w=wn, l=l)
    m.set_behavioral_model(
        _make_comb_model(["A", "B"], "Y", lambda a, b: ~(a | b)),
        params={"vdd": _VDD, "tp_s": _TP_NOM_PS["nor2"] * 1e-12}, execution='vector')
    if with_tests:
        attach_truth_test(m, in_ports=["A", "B"], out_port="Y",
                          fn=nor2)
        attach_delay_test(m, in_ports=["A", "B"], out_port="Y", toggle="A",
                          hold={"B": 0})
    return m


def build_nor3(*, variant="x1", wn=_WN, wp=_WP, l=_L, with_tests=True) -> Module:
    """3-input NOR: ``Y = NOT(A OR B OR C)``. PMOS triple series (3x wide)."""
    _require_variant(variant, "nor3")
    m = Module("PDK_skywater130", example_output_dir(), "nor3.cir", pdk_details=pdk,
               module_name="nor3")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "B": Port.analog_input(),
        "C": Port.analog_input(), "Y": Port.analog_output(),
    })
    wps = wp * _STACK_WIDEN[3]
    with m:
        pdk.pmos1p8("MPA", d="px1", g="A", s="VDD", b="VDD", w=wps, l=l)
        pdk.pmos1p8("MPB", d="px2", g="B", s="px1", b="VDD", w=wps, l=l)
        pdk.pmos1p8("MPC", d="Y", g="C", s="px2", b="VDD", w=wps, l=l)
        pdk.nmos1p8("MNA", d="Y", g="A", s="VSS", b="VSS", w=wn, l=l)
        pdk.nmos1p8("MNB", d="Y", g="B", s="VSS", b="VSS", w=wn, l=l)
        pdk.nmos1p8("MNC", d="Y", g="C", s="VSS", b="VSS", w=wn, l=l)
    m.set_behavioral_model(
        _make_comb_model(["A", "B", "C"], "Y", lambda a, b, c: ~(a | b | c)),
        params={"vdd": _VDD, "tp_s": _TP_NOM_PS["nor3"] * 1e-12}, execution='vector')
    if with_tests:
        attach_truth_test(m, in_ports=["A", "B", "C"], out_port="Y",
                          fn=nor3)
        attach_delay_test(m, in_ports=["A", "B", "C"], out_port="Y", toggle="A",
                          hold={"B": 0, "C": 0})
    return m


def build_xor2(*, variant="static", wn=_WN, wp=_WP, l=_L, with_tests=True) -> Module:
    """Build a two-input XOR; see the module docstring for variant tradeoffs.

    ``static`` uses an eight-transistor complementary core and two input
    inverters. The pull-down conducts when A equals B; the pull-up conducts
    when A differs from B. ``nand`` uses four NAND2 gates. ``ptl`` uses an
    inverter for B-bar, an NMOS passing B-bar when A is high, and a PMOS
    passing B when A is low. The pass devices can degrade output levels."""
    _require_variant(variant, "xor2")
    mod_name = {"nand": "xor2", "static": "xor2_static", "ptl": "xor2_ptl"}[variant]
    m = Module("PDK_skywater130", example_output_dir(), f"{mod_name}.cir", pdk_details=pdk,
               module_name=mod_name)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "B": Port.analog_input(),
        "Y": Port.analog_output(),
    })

    if variant == "nand":
        nand2 = build_nand2(wn=wn, wp=wp, l=l, with_tests=False)
        m.add_module(nand2, "g1", nets={"VDD": "VDD", "VSS": "VSS",
                                         "A": "A", "B": "B", "Y": "n1"})
        m.add_module(nand2, "g2", nets={"VDD": "VDD", "VSS": "VSS",
                                         "A": "A", "B": "n1", "Y": "n2"})
        m.add_module(nand2, "g3", nets={"VDD": "VDD", "VSS": "VSS",
                                         "A": "B", "B": "n1", "Y": "n3"})
        m.add_module(nand2, "g4", nets={"VDD": "VDD", "VSS": "VSS",
                                         "A": "n2", "B": "n3", "Y": "Y"})
        tp_key = "xor2_nand"
    elif variant == "static":
        # 8T core + two inverters for the input complements = 12T.
        inv = build_inverter(wn=wn, wp=wp, l=l, with_tests=False)
        m.add_module(inv, "gia", nets={"VDD": "VDD", "VSS": "VSS",
                                        "A": "A", "Y": "abar"})
        m.add_module(inv, "gib", nets={"VDD": "VDD", "VSS": "VSS",
                                        "A": "B", "Y": "bbar"})
        wns = wn * _STACK_WIDEN[2]     # 2-high series stacks -> widen
        wps = wp * _STACK_WIDEN[2]
        with m:
            # PDN: Y->0 iff A==B.  branch (A,B): A=1&B=1;  branch (Ab,Bb): A=0&B=0
            pdk.nmos1p8("MNA1", d="Y", g="A", s="nd1", b="VSS", w=wns, l=l)
            pdk.nmos1p8("MNB1", d="nd1", g="B", s="VSS", b="VSS", w=wns, l=l)
            pdk.nmos1p8("MNA0", d="Y", g="abar", s="nd2", b="VSS", w=wns, l=l)
            pdk.nmos1p8("MNB0", d="nd2", g="bbar", s="VSS", b="VSS", w=wns, l=l)
            # PUN: Y->1 iff A!=B.  branch (Ab,B): A=1&B=0;  branch (A,Bb): A=0&B=1
            pdk.pmos1p8("MPA1", d="pd1", g="abar", s="VDD", b="VDD", w=wps, l=l)
            pdk.pmos1p8("MPB1", d="Y", g="B", s="pd1", b="VDD", w=wps, l=l)
            pdk.pmos1p8("MPA0", d="pd2", g="A", s="VDD", b="VDD", w=wps, l=l)
            pdk.pmos1p8("MPB0", d="Y", g="bbar", s="pd2", b="VDD", w=wps, l=l)
        tp_key = "xor2_static"
    else:   # ptl
        inv = build_inverter(wn=wn, wp=wp, l=l, with_tests=False)
        m.add_module(inv, "ginv", nets={"VDD": "VDD", "VSS": "VSS",
                                         "A": "B", "Y": "bbar"})
        with m:
            # A=1 -> NMOS on -> Y = B-bar (weak 1);  A=0 -> PMOS on -> Y = B (weak 0)
            pdk.nmos1p8("MNP", d="bbar", g="A", s="Y", b="VSS", w=2 * wn, l=l)
            pdk.pmos1p8("MPP", d="B", g="A", s="Y", b="VDD", w=2 * wp, l=l)
        tp_key = "xor2_ptl"

    m.set_behavioral_model(
        _make_comb_model(["A", "B"], "Y", lambda a, b: a ^ b),
        params={"vdd": _VDD, "tp_s": _TP_NOM_PS[tp_key] * 1e-12}, execution='vector')
    if with_tests:
        attach_truth_test(m, in_ports=["A", "B"], out_port="Y",
                          fn=xor2, with_levels=True)
        attach_delay_test(m, in_ports=["A", "B"], out_port="Y", toggle="A",
                          hold={"B": 0})
    return m


def build_mux2(*, variant="tg", wn=_WN, wp=_WP, l=_L, with_tests=True) -> Module:
    """Six-transistor multiplexer: ``Y = S ? B : A``.

    Two transmission gates select A or B, and an inverter produces S-bar.
    The output is unbuffered and driven through the selected transmission gate.
    Only ``variant="tg"`` is supported."""
    _require_variant(variant, "mux2")
    m = Module("PDK_skywater130", example_output_dir(), "mux2.cir", pdk_details=pdk,
               module_name="mux2")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "B": Port.analog_input(),
        "S": Port.analog_input(), "Y": Port.analog_output(),
    })
    inv = build_inverter(wn=wn, wp=wp, l=l, with_tests=False)
    m.add_module(inv, "ginv", nets={"VDD": "VDD", "VSS": "VSS",
                                     "A": "S", "Y": "sbar"})
    with m:
        # TG0: on when S=0 (NMOS gate = Sbar high, PMOS gate = S low) -> Y = A.
        pdk.nmos1p8("MN0", d="A", g="sbar", s="Y", b="VSS", w=wn, l=l)
        pdk.pmos1p8("MP0", d="A", g="S", s="Y", b="VDD", w=wp, l=l)
        # TG1: on when S=1 -> Y = B.
        pdk.nmos1p8("MN1", d="B", g="S", s="Y", b="VSS", w=wn, l=l)
        pdk.pmos1p8("MP1", d="B", g="sbar", s="Y", b="VDD", w=wp, l=l)
    m.set_behavioral_model(
        _make_comb_model(["A", "B", "S"], "Y",
                         lambda a, b, s: np.where(s, b, a)),
        params={"vdd": _VDD, "tp_s": _TP_NOM_PS["mux2"] * 1e-12}, execution='vector')
    if with_tests:
        attach_truth_test(m, in_ports=["A", "B", "S"], out_port="Y",
                          fn=mux2)
        attach_delay_test(m, in_ports=["A", "B", "S"], out_port="Y", toggle="A",
                          hold={"B": 0, "S": 0})
    return m


# --- sequential cell ------------------------------------------------------------

def build_dff(*, variant="x1", wn=_WN, wp=_WP, l=_L, with_tests=True) -> Module:
    """Positive-edge D flip-flop with active-low clear, built from six NANDs.

    The connections are::

        n1 = NAND2(n4, n2);  n2 = NAND3(n1, CLRB, CLK)
        n3 = NAND3(n2, CLK, n4);  n4 = NAND3(n3, CLRB, D)
        Q  = NAND2(n2, QB);  QB = NAND3(Q, n3, CLRB)

    ``CLRB = 0`` forces Q low asynchronously. Two NAND2s and four NAND3s use
    32 transistors. The behavioral model captures D with a fixed clock-to-Q
    delay and supports feed-forward use."""
    _require_variant(variant, "dff")
    m = Module("PDK_skywater130", example_output_dir(), "dff.cir", pdk_details=pdk,
               module_name="dff")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "D": Port.analog_input(), "CLK": Port.analog_input(),
        "CLRB": Port.analog_input(),
        "Q": Port.analog_output(), "QB": Port.analog_output(),
    })
    nand2 = build_nand2(wn=wn, wp=wp, l=l, with_tests=False)
    nand3 = build_nand3(wn=wn, wp=wp, l=l, with_tests=False)
    m.add_module(nand2, "g1", nets={"VDD": "VDD", "VSS": "VSS",
                                     "A": "n4", "B": "n2", "Y": "n1"})
    m.add_module(nand3, "g2", nets={"VDD": "VDD", "VSS": "VSS",
                                     "A": "n1", "B": "CLRB", "C": "CLK", "Y": "n2"})
    m.add_module(nand3, "g3", nets={"VDD": "VDD", "VSS": "VSS",
                                     "A": "n2", "B": "CLK", "C": "n4", "Y": "n3"})
    m.add_module(nand3, "g4", nets={"VDD": "VDD", "VSS": "VSS",
                                     "A": "n3", "B": "CLRB", "C": "D", "Y": "n4"})
    m.add_module(nand2, "g5", nets={"VDD": "VDD", "VSS": "VSS",
                                     "A": "n2", "B": "QB", "Y": "Q"})
    m.add_module(nand3, "g6", nets={"VDD": "VDD", "VSS": "VSS",
                                     "A": "Q", "B": "n3", "C": "CLRB", "Y": "QB"})
    m.set_behavioral_model(_dff_model,
                           params={"vdd": _VDD, "tcq_s": _CLKQ_NOM_PS * 1e-12}, execution='vector')
    if with_tests:
        attach_dff_timing_test(m)
    return m


# --- truth-table characterization (combinational cells) -------------------------

_TT_WIN = 3e-9          # per-combination hold (a gate settles in well under 1 ns)
_C_LOAD = 15e-15        # characterization output load (F)


def attach_truth_test(mod, *, in_ports, out_port, fn, w=_TT_WIN, with_levels=False):
    """Walk every input combination in one transient; output must equal ``fn``.

    ``fn`` takes one argument per input bit (in ``in_ports`` order) and returns the
    expected output bit. The metric samples the output near the end of each
    window and counts disagreements; one "0 mismatches" spec covers the table.
    With ``with_levels`` (the XOR variants), also reports worst-case VOH/VOL — the
    knob that exposes pass-transistor level degradation.
    """
    combos = list(itertools.product((0, 1), repeat=len(in_ports)))
    tb = mod.create_test("truth")
    mut = {"VDD": "VDD", "VSS": "0"}
    for p in in_ports:
        mut[p] = p.lower()
    mut[out_port] = out_port.lower()
    tb.add_module_under_test(mut)
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    for i, p in enumerate(in_ports):
        levels = [c[i] for c in combos]
        tb.add_source(sources.VoltageSource(
            p.lower(), positive=p.lower(), negative="0",
            waveform=sources.BitPattern(levels, ui=w, v1=0, v2=_VDD,
                                        tr=100e-12, tf=100e-12)))
    tb.add(elements.capacitor("cload", p=out_port.lower(), n="0",
                                      value=_C_LOAD))
    tb.add_sim_type(Transient(step=50e-12, stop=len(combos) * w, maxstep=200e-12))
    out = tb.probe(out_port.lower())
    tb.add_metric("truth_mismatches", _truth_mismatches, sig=out,
                  combos=combos, fn=fn, w=w)
    tb.add_spec(f"{mod.module_name} truth table exact (0 mismatches)",
                "truth_mismatches", "==", 0.0)
    if with_levels:
        tb.add_metric("voh_mv", lambda ckt, cs=combos, o=out_port.lower(), f=fn,
                      ww=w: _levels(ckt, cs, o, f, ww)[0])
        tb.add_metric("vol_mv", lambda ckt, cs=combos, o=out_port.lower(), f=fn,
                      ww=w: _levels(ckt, cs, o, f, ww)[1])
        # Check that each variant resolves past the decision threshold.
        tb.add_spec(f"{mod.module_name} VOH > VDD/2", "voh_mv", ">", _VDD / 2 * 1e3)
        tb.add_spec(f"{mod.module_name} VOL < VDD/2", "vol_mv", "<", _VDD / 2 * 1e3)
    return tb


def _truth_mismatches(sig, combos, fn, w, *, vdd=_VDD):
    bad = 0
    for i, combo in enumerate(combos):
        y = int(value_at(sig, (i + 0.9) * w) > vdd / 2)
        if y != int(fn(*combo)):
            bad += 1
    return float(bad)


def _levels(ckt, combos, out_net, fn, w):
    """(worst-case VOH mV, worst-case VOL mV) over the truth-table windows."""
    sig = ckt.load_signal(out_net)
    highs, lows = [], []
    for i, combo in enumerate(combos):
        y = float(value_at(sig, (i + 0.9) * w))
        (highs if fn(*combo) else lows).append(y)
    voh = min(highs) * 1e3 if highs else float("nan")
    vol = max(lows) * 1e3 if lows else float("nan")
    return voh, vol


# --- propagation-delay characterization (combinational cells) -------------------

DELAY_METRIC = "tp_ps"
_DLY_STOP = 10e-9
_DLY_RISE, _DLY_FALL = 2e-9, 6e-9      # toggled input's edges


def attach_delay_test(mod, *, in_ports, out_port, toggle, hold, name="delay"):
    """Toggle one input past a clean edge; measure input->output tp (ps).

    The toggled input pulses 0->1->0; the others are held at their
    **non-controlling** level (``hold``) so the edge propagates. tp is the mean
    of the two 50%-crossing delays (rise-driven and fall-driven).
    """
    tb = mod.create_test(name)
    mut = {"VDD": "VDD", "VSS": "0"}
    for p in in_ports:
        mut[p] = p.lower()
    mut[out_port] = out_port.lower()
    tb.add_module_under_test(mut)
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    for p in in_ports:
        if p == toggle:
            tb.add_source(sources.VoltageSource(
                p.lower(), positive=p.lower(), negative="0", dc=0,
                pulse=sources.Pulse(v1=0, v2=_VDD, td=_DLY_RISE, tr=50e-12,
                                    tf=50e-12, pw=_DLY_FALL - _DLY_RISE, per=1.0)))
        else:
            tb.add_source(sources.VoltageSource(
                p.lower(), positive=p.lower(), negative="0",
                dc=_VDD if hold[p] else 0.0))
    tb.add(elements.capacitor("cload", p=out_port.lower(), n="0",
                                      value=_C_LOAD))
    tb.add_sim_type(Transient(step=5e-12, stop=_DLY_STOP, maxstep=20e-12))
    in_probe = tb.probe(toggle.lower())
    out_probe = tb.probe(out_port.lower())
    tb.add_metric(DELAY_METRIC, _gate_tp, sig_in=in_probe, sig_out=out_probe)
    tb.add_spec(f"{mod.module_name} tp measured (< 1 ns)", DELAY_METRIC, "<", 1000.0)
    return tb


def _gate_tp(sig_in, sig_out):
    """Mean of the rise- and fall-triggered propagation delays at VDD/2, in ps."""
    half = _VDD / 2
    tp_r = propagation_delay(sig_in, sig_out, in_threshold=half, out_threshold=half,
                             in_edge='rising', out_edge='any')
    tp_f = propagation_delay(sig_in, sig_out, in_threshold=half, out_threshold=half,
                             in_edge='falling', out_edge='any')
    return float((tp_r + tp_f) / 2 * 1e12)


# --- flip-flop timing characterization ------------------------------------------

_CLKQ_RISE_METRIC = "clk_q_rise_ps"
_CLKQ_FALL_METRIC = "clk_q_fall_ps"
_CAP_ERR_METRIC = "capture_errors"
_CLKQ_MAX_PS = 1000.0
_TIM_EDGES = (4e-9, 12e-9, 20e-9)
_TIM_D_FALL = 9e-9
_TIM_D_RISE = 17e-9
_TIM_STOP = 25e-9


def attach_dff_timing_test(mod):
    """Measure clock-to-Q and check capture in one transient.

    Clear, release, then capture 1, 0 and 1 with D stable well before each edge."""
    tb = mod.create_test("timing")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "D": "d", "CLK": "clk",
                              "CLRB": "clrb", "Q": "q", "QB": "qb"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    tb.add_source(sources.VoltageSource(
        "clk", positive="clk", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=_VDD, td=_TIM_EDGES[0], tr=50e-12,
                            tf=50e-12, pw=4e-9, per=8e-9)))
    tb.add_source(sources.VoltageSource(
        "clrb", positive="clrb", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(3e-9, 0).add(3.1e-9, _VDD)))
    tb.add_source(sources.VoltageSource(
        "d", positive="d", negative="0", dc=0,
        pwl=sources.PWL().add(0, _VDD).add(_TIM_D_FALL, _VDD)
        .add(_TIM_D_FALL + 50e-12, 0).add(_TIM_D_RISE, 0)
        .add(_TIM_D_RISE + 50e-12, _VDD).add(_TIM_STOP, _VDD)))
    tb.add_sim_type(Transient(step=20e-12, stop=_TIM_STOP, maxstep=50e-12))
    tb.add_metric(_CLKQ_RISE_METRIC, _clk_q_rise)
    tb.add_metric(_CLKQ_FALL_METRIC, _clk_q_fall)
    tb.add_metric(_CAP_ERR_METRIC, _capture_errors)
    tb.add_spec(f"clk-Q rise < {_CLKQ_MAX_PS:.0f} ps", _CLKQ_RISE_METRIC, "<",
                _CLKQ_MAX_PS)
    tb.add_spec(f"clk-Q fall < {_CLKQ_MAX_PS:.0f} ps", _CLKQ_FALL_METRIC, "<",
                _CLKQ_MAX_PS)
    tb.add_spec("captures D on every edge (0 errors)", _CAP_ERR_METRIC, "==", 0.0)
    return tb


def _clk_q(ckt, edge_time, rising):
    half = _VDD / 2
    return propagation_delay(
        ckt.load_signal("clk"), ckt.load_signal("q"),
        in_threshold=half, out_threshold=half, in_edge='rising',
        out_edge='rising' if rising else 'falling',
        start=edge_time - 1e-9) * 1e12


def _clk_q_rise(ckt):
    return _clk_q(ckt, _TIM_EDGES[0], True)


def _clk_q_fall(ckt):
    return _clk_q(ckt, _TIM_EDGES[1], False)


def _capture_errors(ckt):
    q_sig = ckt.load_signal("q")
    bad = 0
    for e, exp in zip(_TIM_EDGES, (1, 0, 1)):
        q = int(value_at(q_sig, e + 3e-9) > _VDD / 2)
        if q != exp:
            bad += 1
    return float(bad)


# --- setup-time sweep -----------------------------------------------------------

_SETUP_SPACINGS_NS = (2.0, 1.5, 1.0, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3,
                      0.25, 0.2, 0.15, 0.1, 0.05, 0.0, -0.1, -0.2, -0.4)
_SETUP_TCLK = 5e-9


def _attach_setup_probe(mod, *, spacing, t_clk=_SETUP_TCLK):
    """One clock edge; D rises ``spacing`` seconds before it. Q should capture 1.

    Below the setup time the master latch cannot acquire D in time: clk-Q first
    stretches, then capture fails. Reports ``captured`` and ``clk_q_ps``.
    """
    tb = mod.create_test("setup")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "D": "d", "CLK": "clk",
                              "CLRB": "clrb", "Q": "q", "QB": "qb"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    tb.add_source(sources.VoltageSource(
        "clk", positive="clk", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=_VDD, td=t_clk, tr=50e-12, tf=50e-12,
                            pw=3e-9, per=100e-9)))
    tb.add_source(sources.VoltageSource(
        "clrb", positive="clrb", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(1e-9, 0).add(1.05e-9, _VDD)))
    t_d = t_clk - spacing
    tb.add_source(sources.VoltageSource(
        "d", positive="d", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(max(t_d, 1.1e-9), 0)
        .add(max(t_d, 1.1e-9) + 30e-12, _VDD).add(t_clk + 4e-9, _VDD)))
    tb.add_sim_type(Transient(step=10e-12, stop=t_clk + 4e-9, maxstep=30e-12))
    tb.add_metric("captured", lambda ckt, tk=t_clk: float(
        value_at(ckt.load_signal("q"), tk + 3e-9) > _VDD / 2))
    tb.add_metric("clk_q_ps", lambda ckt, tk=t_clk: _clk_q(ckt, tk, True))
    return tb


def sweep_setup_time(spacings_ns=_SETUP_SPACINGS_NS, corner="tt"):
    """Run the setup probe across data-to-clock spacings. Returns rows
    ``(spacing_ns, captured, clk_q_ps)``."""
    rows = []
    for s in spacings_ns:
        m = build_dff(with_tests=False)
        tb = _attach_setup_probe(m, spacing=s * 1e-9)
        r = tb.run(corner=corner)
        rows.append((float(s), float(r.metrics["captured"]),
                     float(r.metrics["clk_q_ps"])))
    return rows


def setup_time_ps(rows):
    """(nominal_clkq_ps, t_setup_10pct_ns, t_fail_boundary_ns) from sweep rows."""
    good = [(s, q) for s, c, q in rows if c > 0.5 and np.isfinite(q)]
    if not good:
        raise RuntimeError("setup sweep: no spacing captured — widen the range")
    nominal = max(good, key=lambda r: r[0])[1]
    t10 = min((s for s, q in good if q <= 1.1 * nominal), default=float("nan"))
    boundary = min((s for s, c, q in rows if c > 0.5), default=float("nan"))
    return nominal, t10, boundary


# --- behavioral truth cross-check -----------------------------------------------

def behavioral_truth_mismatches(cell, in_ports, out_port, fn, *, w=_TT_WIN):
    """Count behavioral output disagreements with ``fn`` over all input words.

    Drive staircase inputs and sample the delayed output near each window's
    end. This check uses numpy and requires no SPICE simulator."""
    combos = list(itertools.product((0, 1), repeat=len(in_ports)))
    top = Module("PDK_skywater130", example_output_dir(), "behtop.cir", pdk_details=pdk,
                 module_name="behtop")
    nets = {"VDD": "vdd", "VSS": "0", out_port: "out"}
    for p in in_ports:
        nets[p] = p.lower()
    top.add_module(cell, "dut", nets=nets)
    top.add_source(sources.VoltageSource("vdd", positive="vdd", negative="0",
                                         dc=_VDD))
    for i, p in enumerate(in_ports):
        levels = [c[i] for c in combos]
        top.add_source(sources.VoltageSource(
            p.lower(), positive=p.lower(), negative="0",
            waveform=sources.BitPattern(levels, ui=w, v1=0, v2=_VDD,
                                        tr=100e-12, tf=100e-12)))
    t_stop = len(combos) * w
    top.simulate_behavioral(t_step=w / 60, t_stop=t_stop)
    out = top.load_signal("out")
    t = np.asarray(out.x, dtype=float)
    y = np.asarray(out.y, dtype=float)
    bad = 0
    for i, combo in enumerate(combos):
        yi = int(np.interp((i + 0.95) * w, t, y) > _VDD / 2)
        if yi != int(fn(*combo)):
            bad += 1
    return bad


# --- drivers --------------------------------------------------------------------

def run_characterization(mod, corner="tt"):
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def transistor_count(mod):
    """Count leaf devices, recursing through subcircuit instances."""
    n = 0
    for entry in mod.elem_dict.values():
        dev = entry.get("device")
        if isinstance(dev, Module):
            n += transistor_count(dev)
        else:
            n += 1
    return n


def save_plot(setup_rows, dff_timing_ckt, xor_ckts, nor_ckt, path):
    """2x2: DFF setup sweep, DFF capture transient, the three-way XOR level
    trade, and a NOR2 truth-table walk. ``xor_ckts`` maps variant -> sim_results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ((ax_su, ax_tim), (ax_xor, ax_tt)) = plt.subplots(
        2, 2, figsize=(13, 9))

    # (0,0) setup sweep.
    nominal, t10, boundary = setup_time_ps(setup_rows)
    for s, c, q in setup_rows:
        if c > 0.5 and np.isfinite(q):
            ax_su.plot(s, q, "o", color="#0072B2", ms=6)
        else:
            ax_su.plot(s, nominal * 1.6, "x", color="#D55E00", ms=8)
    ax_su.axhline(nominal, color="#5a5a6e", ls="--", lw=1,
                  label=f"nominal clk-Q = {nominal:.0f} ps")
    ax_su.axhline(1.1 * nominal, color="#009E73", ls=":", lw=1,
                  label="1.1x nominal (setup def.)")
    ax_su.axvline(boundary, color="#D55E00", ls="-", lw=1.2,
                  label=f"capture cliff = {boundary*1e3:.0f} ps")
    ax_su.set_xlabel("data-to-clock spacing (ns)  —  + = data before clock")
    ax_su.set_ylabel("clk-to-Q (ps)")
    ax_su.set_title("DFF setup-time sweep (x = capture failed)")
    ax_su.legend(frameon=False, fontsize=8, loc="upper right")
    ax_su.invert_xaxis()

    # (0,1) DFF capture transient.
    t = np.asarray(dff_timing_ckt.load_signal("clk").x, dtype=float) * 1e9
    for sig, off, color, lab in (("clk", 4.0, "#9fb4c7", "CLK"),
                                 ("d", 2.0, "#5a5a6e", "D"),
                                 ("q", 0.0, "#0072B2", "Q")):
        y = np.asarray(dff_timing_ckt.load_signal(sig).y, dtype=float) / _VDD
        ax_tim.plot(t, y + off, color=color, lw=1.5, label=lab)
    ax_tim.set_yticks([0.5, 2.5, 4.5], ["Q", "D", "CLK"])
    ax_tim.set_ylim(-0.3, 5.7)
    ax_tim.set_xlabel("time (ns)")
    ax_tim.set_title("DFF capture: clear, then 1 / 0 / 1 on successive edges")
    ax_tim.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)

    # (1,0) the three XOR variants over the truth walk (level degradation).
    combos = list(itertools.product((0, 1), repeat=2))
    for var, color, lab in (("nand", "#0072B2", "nand (16T, full)"),
                            ("static", "#009E73", "static (12T, full)"),
                            ("ptl", "#D55E00", "ptl (4T, degraded)")):
        ckt = xor_ckts[var]
        tt = np.asarray(ckt.load_signal("y").x, dtype=float) * 1e9
        yy = np.asarray(ckt.load_signal("y").y, dtype=float)
        ax_xor.plot(tt, yy, color=color, lw=1.6, label=lab)
    ax_xor.axhline(_VDD / 2, color="#5a5a6e", ls="--", lw=0.8, label="VDD/2 decision")
    for i, c in enumerate(combos):
        ax_xor.text((i + 0.5) * _TT_WIN * 1e9, _VDD + 0.08,
                    f"{c[0]}^{c[1]}={c[0]^c[1]}", ha="center", fontsize=8,
                    color="#5a5a6e")
    ax_xor.set_ylim(-0.15, _VDD + 0.35)
    ax_xor.set_xlabel("time (ns)")
    ax_xor.set_ylabel("Y (V)")
    ax_xor.set_title("XOR2 output: nand full-swing vs ptl degraded rails")
    ax_xor.legend(frameon=False, fontsize=8, loc="center right")

    # (1,1) NOR2 truth table.
    combos2 = list(itertools.product((0, 1), repeat=2))
    tt = np.asarray(nor_ckt.load_signal("y").x, dtype=float) * 1e9
    for p, off in (("a", 4.0), ("b", 2.0)):
        y = np.asarray(nor_ckt.load_signal(p).y, dtype=float) / _VDD
        ax_tt.plot(tt, y + off, color="#5a5a6e", lw=1.4, label=p.upper())
    yout = np.asarray(nor_ckt.load_signal("y").y, dtype=float) / _VDD
    ax_tt.plot(tt, yout, color="#0072B2", lw=1.7, label="Y")
    for i, c in enumerate(combos2):
        ax_tt.text((i + 0.5) * _TT_WIN * 1e9, 3.5, "".join(str(b) for b in c),
                   ha="center", fontsize=8, color="#5a5a6e")
    ax_tt.set_yticks([0.5, 2.5, 4.5], ["Y", "B", "A"])
    ax_tt.set_ylim(-0.3, 6.0)
    ax_tt.set_xlabel("time (ns)")
    ax_tt.set_title("NOR2 truth table — Y = NOT(A OR B)")
    ax_tt.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)

    for ax in (ax_su, ax_tim, ax_xor, ax_tt):
        ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("stdcells — characterized standard-cell library (sky130, x1)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# (cell, builder, build-kwargs, behavioral in_ports, fn, tp-key) for __main__.
def _catalog_spec():
    return [
        ("inverter", build_inverter, {}, ["A"], inverter, "inverter"),
        ("buf", build_buf, {}, ["A"], buf, "buf"),
        ("nand2", build_nand2, {}, ["A", "B"], nand2, "nand2"),
        ("nand3", build_nand3, {}, ["A", "B", "C"],
         nand3, "nand3"),
        ("nor2", build_nor2, {}, ["A", "B"], nor2, "nor2"),
        ("nor3", build_nor3, {}, ["A", "B", "C"],
         nor3, "nor3"),
        ("mux2", build_mux2, {}, ["A", "B", "S"],
         mux2, "mux2"),
        ("xor2", build_xor2, {"variant": "nand"}, ["A", "B"],
         xor2, "xor2_nand"),
        ("xor2", build_xor2, {"variant": "static"}, ["A", "B"],
         xor2, "xor2_static"),
        ("xor2", build_xor2, {"variant": "ptl"}, ["A", "B"],
         xor2, "xor2_ptl"),
    ]


if __name__ == "__main__":
    OUT = example_output_dir()

    catalog = []       # (cell, variant, devices, ports, timing, beh_delay, extra)
    ckts = {}          # sim_results we keep for the plot

    for cell, builder, kw, in_ports, fn, tp_key in _catalog_spec():
        variant = kw.get("variant", "x1")
        print("=" * 72)
        print(f"cell: {cell}  (variant {variant})")
        print("=" * 72)
        mod = builder(**kw)
        n_dev = transistor_count(mod)
        n_port = len(mod.pin_list)
        results, recon = run_characterization(mod)
        print(recon.summary())

        key = f"{mod.module_name}.truth"
        tp_meas = results[f"{mod.module_name}.delay"].metrics[DELAY_METRIC]
        # Calibrate the behavioral twin's delay from the measured tp, then
        # check the behavioral output against the boolean truth function.
        nominal = _TP_NOM_PS[tp_key]
        mod.calibrate_behavioral("tp_s", tp_meas * 1e-12)
        beh_bad = behavioral_truth_mismatches(mod, in_ports, "Y", fn)
        print(f"  tp measured = {tp_meas:.1f} ps  (nominal {nominal:.1f}; "
              f"drift {tp_meas-nominal:+.1f})")
        print(f"  behavioral truth vs logic: {beh_bad} mismatches "
              f"(twin delay calibrated to {tp_meas:.1f} ps)")

        extra = ""
        if variant in ("nand", "static", "ptl"):
            voh = results[key].metrics["voh_mv"]
            vol = results[key].metrics["vol_mv"]
            extra = f"VOH/VOL {voh:.0f}/{vol:.0f} mV"
            print(f"  levels: VOH={voh:.0f} mV  VOL={vol:.0f} mV")
            ckts[f"xor_{variant}"] = results[key].sim_results
        if cell == "nor2":
            ckts["nor"] = results[key].sim_results
        catalog.append((cell, variant, n_dev, n_port, f"{tp_meas:.0f} ps",
                        f"{tp_meas:.0f} ps", extra))

    # Characterize clock-to-Q and sweep setup spacing.
    print("=" * 72)
    print("cell: dff  (variant x1)")
    print("=" * 72)
    dff = build_dff()
    dff_res, dff_recon = run_characterization(dff)
    print(dff_recon.summary())
    tim = dff_res["dff.timing"]
    ckts["dff_timing"] = tim.sim_results
    cq_r = tim.metrics[_CLKQ_RISE_METRIC]
    cq_f = tim.metrics[_CLKQ_FALL_METRIC]
    cq_avg = (cq_r + cq_f) / 2
    dff.calibrate_behavioral("tcq_s", cq_avg * 1e-12)
    print("\n  setup-time sweep (data-to-clock spacing -> capture):")
    rows = sweep_setup_time()
    nominal, t10, boundary = setup_time_ps(rows)
    for s, c, q in rows:
        tag = f"{q:6.0f} ps" if (c > 0.5 and np.isfinite(q)) else "  FAIL "
        print(f"    spacing {s:+5.2f} ns -> {tag}"
              f"{'  (captured)' if c > 0.5 else '  (wrong value)'}")
    print(f"\n  clk-Q rise/fall = {cq_r:.0f} / {cq_f:.0f} ps  (avg {cq_avg:.0f})")
    print(f"  setup (10% clk-Q) = {t10*1e3:.0f} ps   cliff = {boundary*1e3:.0f} ps")
    print(f"  behavioral twin clk-Q calibrated to {cq_avg:.0f} ps "
          f"(nominal {_CLKQ_NOM_PS:.0f}; drift {cq_avg-_CLKQ_NOM_PS:+.0f})")
    catalog.append(("dff", "x1", transistor_count(dff), len(dff.pin_list),
                    f"clk-Q {cq_r:.0f}/{cq_f:.0f} ps",
                    f"{cq_avg:.0f} ps", f"setup {t10*1e3:.0f} ps"))

    # --- catalog table ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("cell catalog")
    print("=" * 72)
    print(f"  {'cell':<9}{'variant':<8}{'dev':>4}{'ports':>6}  "
          f"{'timing':<18}{'beh delay':<11}extra")
    for cell, variant, dev, ports, timing, beh, extra in catalog:
        print(f"  {cell:<9}{variant:<8}{dev:>4}{ports:>6}  "
              f"{timing:<18}{beh:<11}{extra}")

    png = save_plot(rows, ckts["dff_timing"],
                    {"nand": ckts["xor_nand"], "static": ckts["xor_static"],
                     "ptl": ckts["xor_ptl"]},
                    ckts["nor"], os.path.join(OUT, "stdcells.png"))
    print(f"\n  saved {png}")
