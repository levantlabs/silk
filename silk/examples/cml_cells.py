"""Differential current-mode-logic cells using sky130 transistors.

The cells include a buffer, latch, flip-flop, mux, CMOS converters and output
drivers. A shared diode-connected reference biases the tail devices.
The default reference is 200 uA with 2 kohm loads, giving a nominal 0.4 V
single-ended swing. CMOS switching transfers charge proportional to C*V;
CML steers a standing current between its output legs.

``build_cmos2cml`` uses a differential pair driven by complementary CMOS
signals. ``build_cml_driver`` has a source-degenerated main pair and two
independent auxiliary sink branches. Those branches do not form a shared-tail
current-steering tap. ``build_cml_ffe_driver`` uses shared-tail slices with
mirror ratios setting four tap weights and one predriver per tap.

Characterization measures swings, propagation delays, clock-to-Q, driver
frequency response and fitted FFE tap ratios. These checks use ideal load
resistors and one corner; they do not characterize noise or process spread.

Run with ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.cml_cells``.
Requires ngspice and sky130 models. Prints the measurements and device catalog,
and saves ``cml_cells.png`` in the configured example output directory.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port, PortDir, PortRole
from silk.circuit.modules import Module
from silk.elements import elements
from silk import load_pdk
from silk import sources
from silk.metrics import sample, crossing, dc_gain_dB, peak_gain_dB, propagation_delay
from silk.simulation import Transient, AC

from silk.examples.stdcells import build_inverter
from silk.examples import example_output_dir

pdk = load_pdk('sky130')


_VDD = 1.8
_IREF = 200e-6
_RL = 2000.0
_WD, _LD = 4.0, 0.15
_WCK = 8.0              # Wider clock devices help select a single latch path.
_WH = 6.0               # Wider hold devices increase regenerative loop gain.
_WT, _LT = 5.0, 0.5
_CM = 1.6
_SWING = _IREF * _RL


# Limit output capacitance by scaling signal widths more slowly than tail current.
def _sig_mul(imul):
    return max(1.0, round(imul ** 0.5))


def build_cml_bias(*, imul=1.0) -> Module:
    """Build a diode-connected NMOS reference driven through IBIAS."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_bias.cir", pdk_details=pdk,
               module_name="cml_bias", variants=True)
    m.define_module_ports({"VSS": Port.ground(), "IBIAS": Port.analog_input()})
    with m:
        pdk.nmos1p8("MBIAS", d="IBIAS", g="IBIAS", s="VSS", b="VSS",
                    w=_WT * imul, l=_LT)
    return m


def _tail(m, tail_net, imul=1.0):
    """The mirror tail NMOS shared by every steering cell (widens with current)."""
    pdk.nmos1p8("MT", d=tail_net, g="IBIAS", s="VSS", b="VSS", w=_WT * imul, l=_LT)


def _loads(m, outp, outn, rl=_RL):
    elements.resistor("RLP", p="VDD", n=outp, value=rl)
    elements.resistor("RLN", p="VDD", n=outn, value=rl)


def build_cml_buffer(*, rl=_RL) -> Module:
    """Build a differential pair with a shared tail and resistor loads.

    OUTP follows INP; the opposite drain carries the conducting branch."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_buf.cir", pdk_details=pdk,
               module_name="cml_buffer", variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "INP": Port.analog_input(), "INM": Port.analog_input(),
        "OUTP": Port.analog_output(), "OUTN": Port.analog_output(),
    })
    with m:
        _tail(m, "tail")
        pdk.nmos1p8("M1", d="OUTN", g="INP", s="tail", b="VSS", w=_WD, l=_LD)
        pdk.nmos1p8("M2", d="OUTP", g="INM", s="tail", b="VSS", w=_WD, l=_LD)
        _loads(m, "OUTP", "OUTN", rl)
    return m


def build_cml_latch(*, rl=_RL, imul=1.0) -> Module:
    """Build a CML latch that tracks data when CLKP exceeds CLKN.

    A clock pair steers the tail between the data pair and a cross-coupled
    hold pair. Both use the same loads. Scale rl inversely with imul to
    preserve the nominal swing when increasing tail current."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_latch.cir", pdk_details=pdk,
               module_name="cml_latch", variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "CLKP": Port.analog_input(), "CLKN": Port.analog_input(),
        "DP": Port.analog_input(), "DN": Port.analog_input(),
        "OUTP": Port.analog_output(), "OUTN": Port.analog_output(),
    })
    sm = _sig_mul(imul)
    wd, wck, wh = _WD * sm, _WCK * sm, _WH * sm
    with m:
        _tail(m, "tail", imul)


        pdk.nmos1p8("MCKS", d="samp", g="CLKP", s="tail", b="VSS", w=wck, l=_LD)
        pdk.nmos1p8("MCKH", d="hold", g="CLKN", s="tail", b="VSS", w=wck, l=_LD)

        pdk.nmos1p8("MSA", d="OUTN", g="DP", s="samp", b="VSS", w=wd, l=_LD)
        pdk.nmos1p8("MSB", d="OUTP", g="DN", s="samp", b="VSS", w=wd, l=_LD)

        pdk.nmos1p8("MHA", d="OUTN", g="OUTP", s="hold", b="VSS", w=wh, l=_LD)
        pdk.nmos1p8("MHB", d="OUTP", g="OUTN", s="hold", b="VSS", w=wh, l=_LD)
        _loads(m, "OUTP", "OUTN", rl)
    return m


def build_cml_dff(*, rl=_RL, imul=1.0) -> Module:
    """Build a rising-edge CML flip-flop from two latches.

    The master tracks while CLKP is low; the slave tracks while it is high.
    imul scales both latch currents and device sizes."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_dff.cir", pdk_details=pdk,
               module_name="cml_dff", variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "CLKP": Port.analog_input(), "CLKN": Port.analog_input(),
        "DP": Port.analog_input(), "DN": Port.analog_input(),
        "OUTP": Port.analog_output(), "OUTN": Port.analog_output(),
    })
    latch = build_cml_latch(rl=rl, imul=imul)

    m.add_module(latch, "master", nets={
        "VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
        "CLKP": "CLKN", "CLKN": "CLKP", "DP": "DP", "DN": "DN",
        "OUTP": "mp", "OUTN": "mn"})

    m.add_module(latch, "slave", nets={
        "VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
        "CLKP": "CLKP", "CLKN": "CLKN", "DP": "mp", "DN": "mn",
        "OUTP": "OUTP", "OUTN": "OUTN"})
    return m


def build_cml_mux2(*, rl=_RL, imul=1.0) -> Module:
    """Build a differential mux selecting A when SELN is high, B when SELP is high.

    Drive the select legs between about 0.6 and 1.8 V so the select pair
    steers the current into one data pair. Data legs use CML levels.
    Scale rl inversely with imul to preserve nominal swing."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_mux2.cir", pdk_details=pdk,
               module_name="cml_mux2", variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "AP": Port.analog_input(), "AN": Port.analog_input(),
        "BP": Port.analog_input(), "BN": Port.analog_input(),
        "SELP": Port.analog_input(), "SELN": Port.analog_input(),
        "OUTP": Port.analog_output(), "OUTN": Port.analog_output(),
    })
    sm = _sig_mul(imul)
    wd, wck = _WD * sm, _WCK * sm
    with m:
        _tail(m, "tail", imul)


        pdk.nmos1p8("MSA", d="taila", g="SELN", s="tail", b="VSS", w=wck, l=_LD)
        pdk.nmos1p8("MSB", d="tailb", g="SELP", s="tail", b="VSS", w=wck, l=_LD)
        pdk.nmos1p8("MAA", d="OUTN", g="AP", s="taila", b="VSS", w=wd, l=_LD)
        pdk.nmos1p8("MAB", d="OUTP", g="AN", s="taila", b="VSS", w=wd, l=_LD)
        pdk.nmos1p8("MBA", d="OUTN", g="BP", s="tailb", b="VSS", w=wd, l=_LD)
        pdk.nmos1p8("MBB", d="OUTP", g="BN", s="tailb", b="VSS", w=wd, l=_LD)
        _loads(m, "OUTP", "OUTN", rl)
    return m


def build_cmos2cml(*, rl=_RL) -> Module:
    """Convert CMOS IN to differential OUTP/OUTN using complementary inputs.

    An inverter supplies the complement. Complementary rail inputs maintain
    tail-current headroom across both logic states. OUTP follows IN."""
    m = Module("PDK_skywater130", example_output_dir(), "cmos2cml.cir", pdk_details=pdk,
               module_name="cmos2cml", variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "IN": Port.analog_input(),
        "OUTP": Port.analog_output(), "OUTN": Port.analog_output(),
    })
    inv = build_inverter(with_tests=False)
    m.add_module(inv, "ginv", nets={"VDD": "VDD", "VSS": "VSS",
                                     "A": "IN", "Y": "inbar"})
    with m:
        _tail(m, "tail")
        pdk.nmos1p8("M1", d="OUTN", g="IN", s="tail", b="VSS", w=_WD, l=_LD)
        pdk.nmos1p8("M2", d="OUTP", g="inbar", s="tail", b="VSS", w=_WD, l=_LD)
        _loads(m, "OUTP", "OUTN", rl)
    return m


def build_cml2cmos(*, imul=1.0) -> Module:
    """Convert differential INP/INM to CMOS output Y.

    A differential pair and PMOS mirror load produce a single-ended signal.
    Two inverters restore the output rails. Y follows INP; imul scales sizing."""
    m = Module("PDK_skywater130", example_output_dir(), "cml2cmos.cir", pdk_details=pdk,
               module_name="cml2cmos", variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "INP": Port.analog_input(), "INM": Port.analog_input(),
        "Y": Port.analog_output(),
    })
    sm = _sig_mul(imul)
    wd = _WD * sm
    with m:
        _tail(m, "tail", imul)
        pdk.nmos1p8("MA", d="nd", g="INP", s="tail", b="VSS", w=wd, l=_LD)
        pdk.nmos1p8("MB", d="se", g="INM", s="tail", b="VSS", w=wd, l=_LD)

        pdk.pmos1p8("MPL", d="nd", g="nd", s="VDD", b="VDD", w=2 * wd, l=_LD)
        pdk.pmos1p8("MPR", d="se", g="nd", s="VDD", b="VDD", w=2 * wd, l=_LD)
    inv = build_inverter(with_tests=False)
    m.add_module(inv, "b1", nets={"VDD": "VDD", "VSS": "VSS", "A": "se", "Y": "ybar"})
    m.add_module(inv, "b2", nets={"VDD": "VDD", "VSS": "VSS", "A": "ybar", "Y": "Y"})
    return m


_DRV_RL = 50.0
_DRV_MAIN_MUL = 40
_DRV_TAP_MUL = 13
_RDEG = 60.0
_CDEG = 2.0e-12


def build_cml_driver(*, rl=_DRV_RL, main_mul=_DRV_MAIN_MUL, tap_mul=_DRV_TAP_MUL,
                     rdeg=_RDEG, cdeg=_CDEG, degen=True) -> Module:
    """Build a source-degenerated main pair with auxiliary output sinks.

    R_deg || C_deg connects the main source nodes for frequency peaking.
    degen=False connects them through a small resistance. The auxiliary
    FFEP/FFEN branches have separate tails and no path between their source
    nodes; crossed drains alone do not form a current-steering FFE tap.
    Use build_cml_ffe_driver for shared-tail weighted FFE slices."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_driver.cir", pdk_details=pdk,
               module_name="cml_driver" + ("" if degen else "_flat"),
               variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "INP": Port.analog_input(), "INM": Port.analog_input(),
        "FFEP": Port.analog_input(), "FFEN": Port.analog_input(),
        "OUTP": Port.analog_output(), "OUTN": Port.analog_output(),
    })
    w_main = _WD * round(main_mul ** 0.5)
    w_mtail = _WT * (main_mul / 2)
    w_tap = _WD * round(tap_mul ** 0.5)
    w_ttail = _WT * (tap_mul / 2)
    with m:
        elements.resistor("RLP", p="VDD", n="OUTP", value=rl)
        elements.resistor("RLN", p="VDD", n="OUTN", value=rl)

        pdk.nmos1p8("MMA", d="OUTN", g="INP", s="sa", b="VSS", w=w_main, l=_LD)
        pdk.nmos1p8("MMB", d="OUTP", g="INM", s="sb", b="VSS", w=w_main, l=_LD)
        pdk.nmos1p8("MTA", d="sa", g="IBIAS", s="VSS", b="VSS", w=w_mtail, l=_LT)
        pdk.nmos1p8("MTB", d="sb", g="IBIAS", s="VSS", b="VSS", w=w_mtail, l=_LT)
        if degen:
            elements.resistor("RDEG", p="sa", n="sb", value=rdeg)
            elements.capacitor("CDEG", p="sa", n="sb", value=cdeg)
        else:
            elements.resistor("RDEG", p="sa", n="sb", value=0.001)

        pdk.nmos1p8("MFA", d="OUTP", g="FFEP", s="ta", b="VSS", w=w_tap, l=_LD)
        pdk.nmos1p8("MFB", d="OUTN", g="FFEN", s="tb", b="VSS", w=w_tap, l=_LD)
        pdk.nmos1p8("MTFA", d="ta", g="IBIAS", s="VSS", b="VSS", w=w_ttail, l=_LT)
        pdk.nmos1p8("MTFB", d="tb", g="IBIAS", s="VSS", b="VSS", w=w_ttail, l=_LT)
    return m


_PD_IMUL = 4
_PD_RL = 875.0


def build_cml_predriver(*, imul=_PD_IMUL, rl=_PD_RL) -> Module:
    """Build a shared-tail gain stage to increase the output-pair drive swing.

    imul scales the tail current; rl sets the resistive loads."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_predriver.cir", pdk_details=pdk,
               module_name="cml_predriver", variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "INP": Port.analog_input(), "INM": Port.analog_input(),
        "OUTP": Port.analog_output(), "OUTN": Port.analog_output(),
    })
    sm = _sig_mul(imul)
    with m:
        _tail(m, "tail", imul)
        pdk.nmos1p8("M1", d="OUTN", g="INP", s="tail", b="VSS", w=_WD * sm, l=_LD)
        pdk.nmos1p8("M2", d="OUTP", g="INM", s="tail", b="VSS", w=_WD * sm, l=_LD)
        _loads(m, "OUTP", "OUTN", rl)
    return m


_FFE_TAPS = (-0.385, 1.0, -0.610, 0.106)
_FFE_MAIN_MUL = 40
_FFE_PD_IMUL = 16
_FFE_PD_RL = 200.0
_FFE_NAMES = ("pre", "main", "post1", "post2")
_FFE_SHIFT = {"pre": +1, "main": 0, "post1": -1, "post2": -2}
_FFE_TBIT = 1.0 / 600e6
_FFE_T0 = 2e-9
_FFE_HI, _FFE_LO = _CM + _SWING / 2, _CM - _SWING / 2


def _sum_node():
    # The taps sink current while the shared load determines output voltage.
    return Port(dir=PortDir.INOUT, role=PortRole.ANALOG)


def _ffe_tap_model(ports, t, params):
    """Ideal fully steered slice at the nominal reference current.

    Includes the predriver's logical polarity, but no delay, bandwidth,
    finite output resistance or bias-voltage dependence. Equal inputs split
    the tail equally; positive pin current flows into the tap.
    """
    steering = np.sign(ports.INP.v - ports.INM.v)
    tail = params.mult * params.iref
    ports.OUTP.i = tail * (1 - steering) / 2
    ports.OUTN.i = tail * (1 + steering) / 2


def _ffe_load_model(ports, t, params):
    """Convert the summed sink currents to voltage using the shared loads."""
    ports.OUTP.v = ports.VDD.v + params.rl * ports.OUTP.i
    ports.OUTN.v = ports.VDD.v + params.rl * ports.OUTN.i


def build_cml_ffe_tap(*, mult, pd_imul=_FFE_PD_IMUL, pd_rl=_FFE_PD_RL) -> Module:
    """Build a weighted shared-tail slice with a predriver.

    mult scales the tail current. INP high pulls OUTN low, giving a positive
    differential weight. Cross OUTP and OUTN when placing a negative tap.
    The tap sinks current; build_cml_ffe_load supplies the resistive loads.
    Its behavioral model assumes full steering at the nominal reference
    current and omits delay, bandwidth and bias-voltage dependence."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_ffe_tap.cir", pdk_details=pdk,
               module_name="cml_ffe_tap", variants=True)
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
        "INP": Port.analog_input(), "INM": Port.analog_input(),
        "OUTP": _sum_node(), "OUTN": _sum_node(),
    })
    m.add_module(build_cml_predriver(imul=pd_imul, rl=pd_rl), "pd",
                 nets={"VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
                       "INP": "INP", "INM": "INM", "OUTP": "gp", "OUTN": "gn"})
    with m:
        pdk.nmos1p8("MA", d="OUTN", g="gp", s="tail", b="VSS", w=_WD, l=_LD, m=mult)
        pdk.nmos1p8("MB", d="OUTP", g="gn", s="tail", b="VSS", w=_WD, l=_LD, m=mult)
        pdk.nmos1p8("MT", d="tail", g="IBIAS", s="VSS", b="VSS", w=_WT, l=_LT, m=mult)
    m.set_behavioral_model(_ffe_tap_model, params={'mult': mult, 'iref': _IREF},
                           execution='vector')
    return m


def build_cml_ffe_load(*, rl=_DRV_RL) -> Module:
    """The FFE driver's shared load: ``RLP``/``RLN`` from ``VDD`` to the outputs,
    where the taps' summed current becomes the launch voltage."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_ffe_load.cir", pdk_details=pdk,
               module_name="cml_ffe_load", variants=True)
    m.define_module_ports({"VDD": Port.supply(),
                           "OUTP": Port.analog_output(), "OUTN": Port.analog_output()})
    with m:
        _loads(m, "OUTP", "OUTN", rl)
    m.set_behavioral_model(_ffe_load_model, params={'rl': rl}, execution='vector')
    return m


def build_cml_ffe_driver(*, taps=_FFE_TAPS, main_mul=_FFE_MAIN_MUL, rl=_DRV_RL,
                         pd_imul=_FFE_PD_IMUL, pd_rl=_FFE_PD_RL) -> Module:
    """Build four current-steered FFE taps connected to shared resistive loads.

    Mirror ratios set the weights, and each tap has a predriver. Crossed
    output connections implement negative weights. Inputs supply the true
    and complementary bits at delays -1, 0, 1 and 2 UI.

    For behavioral runs, register data sources before VDD and place the taps
    before the load so their currents are available when the load evaluates.
    The ideal tap models omit delays, bandwidth and bias-voltage dependence."""
    m = Module("PDK_skywater130", example_output_dir(), "cml_ffe_drv.cir", pdk_details=pdk,
               module_name="cml_ffe_driver", variants=True)
    ports = {"VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
             "OUTP": Port.analog_output(), "OUTN": Port.analog_output()}
    for nm in _FFE_NAMES:
        ports[f"{nm.upper()}P"] = Port.analog_input()
        ports[f"{nm.upper()}N"] = Port.analog_input()
    m.define_module_ports(ports)
    for nm, c in zip(_FFE_NAMES, taps):
        mm = max(1, round(abs(c) * main_mul))
        outp, outn = ("OUTP", "OUTN") if c > 0 else ("OUTN", "OUTP")
        m.add_module(build_cml_ffe_tap(mult=mm, pd_imul=pd_imul, pd_rl=pd_rl), f"tap_{nm}",
                     nets={"VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
                           "INP": f"{nm.upper()}P", "INM": f"{nm.upper()}N",
                           "OUTP": outp, "OUTN": outn})
    m.add_module(build_cml_ffe_load(rl=rl), "load",
                 nets={"VDD": "VDD", "OUTP": "OUTP", "OUTN": "OUTN"})
    return m


def _bias_top(name):
    """A top-level Module with VDD + the IREF-forced bias diode on net ``ibias``."""
    top = Module("PDK_skywater130", example_output_dir(), f"{name}.cir", pdk_details=pdk)
    top.add_module(build_cml_bias(), "bias", nets={"VSS": "0", "IBIAS": "ibias"})
    with top:
        sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD)
        sources.CurrentSource("iref", positive="VDD", negative="ibias", dc=_IREF)
    return top


def _diff_pulse(top, p, n, *, td, per, lo=_CM - _SWING / 2, hi=_CM + _SWING / 2):
    """A complementary differential pulse pair on nets p/n around the common mode."""
    with top:
        sources.VoltageSource(f"v{p}", positive=p, negative="0",
            pulse=sources.Pulse(v1=lo, v2=hi, td=td, tr=80e-12, tf=80e-12,
                                pw=per / 2, per=per))
        sources.VoltageSource(f"v{n}", positive=n, negative="0",
            pulse=sources.Pulse(v1=hi, v2=lo, td=td, tr=80e-12, tf=80e-12,
                                pw=per / 2, per=per))


def _diff_dc(top, p, n, high):
    """A static differential pair on p/n (p high if ``high`` else n high)."""
    lo, hi = _CM - _SWING / 2, _CM + _SWING / 2
    with top:
        sources.VoltageSource(f"v{p}", positive=p, negative="0", dc=hi if high else lo)
        sources.VoltageSource(f"v{n}", positive=n, negative="0", dc=lo if high else hi)


def _se(sig):
    y = np.asarray(sig.y, dtype=float)
    return y


def _swing(y):
    """Return the 2nd-to-98th percentile span in mV."""
    y = np.asarray(y, dtype=float)
    return float(np.percentile(y, 98) - np.percentile(y, 2)) * 1e3


def characterize_buffer():
    """Differential input toggling; measure single-ended swing and INP->OUTP tp."""
    top = _bias_top("cml_buf_tb")
    top.add_module(build_cml_buffer(), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
        "INP": "inp", "INM": "inm", "OUTP": "outp", "OUTN": "outn"})
    _diff_pulse(top, "inp", "inm", td=2e-9, per=4e-9)
    top.add_sim_type(Transient(step=20e-12, stop=10e-9, maxstep=50e-12))
    top.simulate(corner="tt")
    outp = top.load_signal("outp")
    swing = _swing(_se(outp))
    tp = _cml_tp(top.load_signal("inp"), outp)
    return {"swing_mv": swing, "tp_ps": tp}


def characterize_mux():
    """A and B held to opposite differential values; toggle SEL; OUT must follow
    the selected input."""
    top = _bias_top("cml_mux_tb")
    top.add_module(build_cml_mux2(), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
        "AP": "ap", "AN": "an", "BP": "bp", "BN": "bn",
        "SELP": "selp", "SELN": "seln", "OUTP": "outp", "OUTN": "outn"})
    _diff_dc(top, "ap", "an", True)
    _diff_dc(top, "bp", "bn", False)
    _diff_pulse(top, "selp", "seln", td=2e-9, per=4e-9)
    top.add_sim_type(Transient(step=20e-12, stop=10e-9, maxstep=50e-12))
    top.simulate(corner="tt")
    t = np.asarray(top.load_signal("outp").x, dtype=float)
    op = _se(top.load_signal("outp")) - _se(top.load_signal("outn"))
    sp = _se(top.load_signal("selp")) - _se(top.load_signal("seln"))

    a_win = op[sp < -0.1]
    b_win = op[sp > 0.1]
    return {"sel_A_diff_mv": float(np.median(a_win)) * 1e3 if a_win.size else 0.0,
            "sel_B_diff_mv": float(np.median(b_win)) * 1e3 if b_win.size else 0.0,
            "swing_mv": _swing(op) / 2}


def characterize_dff(cell="dff"):
    """Measure single-ended swing and differential clock-to-output delay.

    Delay uses a rising differential clock zero crossing and the next output
    crossing of its record-mean level. The same rig can drive a latch.
    """
    build = build_cml_dff if cell == "dff" else build_cml_latch
    top = _bias_top(f"cml_{cell}_tb")
    top.add_module(build(), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
        "CLKP": "clkp", "CLKN": "clkn", "DP": "dp", "DN": "dn",
        "OUTP": "outp", "OUTN": "outn"})
    tclk = 2e-9
    _diff_pulse(top, "clkp", "clkn", td=1e-9, per=tclk)
    _diff_pulse(top, "dp", "dn", td=1.5e-9, per=4 * tclk)
    top.add_sim_type(Transient(step=10e-12, stop=20e-9, maxstep=30e-12))
    top.simulate(corner="tt")
    outd = top.load_signal("outp") - top.load_signal("outn")
    clkd = top.load_signal("clkp") - top.load_signal("clkn")
    swing = _swing(_se(outd)) / 2
    tcq = _clk_to_q(clkd, outd)
    return {"clk_q_ps": tcq, "swing_mv": swing}


def characterize_cmos2cml():
    top = _bias_top("cmos2cml_tb")
    top.add_module(build_cmos2cml(), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
        "IN": "cin", "OUTP": "outp", "OUTN": "outn"})
    with top:
        sources.VoltageSource("vin", positive="cin", negative="0", dc=0,
            pulse=sources.Pulse(v1=0, v2=_VDD, td=2e-9, tr=80e-12, tf=80e-12,
                                pw=2e-9, per=4e-9))
    top.add_sim_type(Transient(step=20e-12, stop=10e-9, maxstep=50e-12))
    top.simulate(corner="tt")
    op = _se(top.load_signal("outp"))
    return {"out_swing_mv": _swing(op)}


def characterize_driver(*, degen=True):
    """Output driver: launch swing (transient into 50 ohm) + the degeneration
    peaking (AC: LF gain, HF-peak gain, boost, and the zero). Returns a dict."""

    top = _bias_top("drv_sw")
    top.add_module(build_cml_driver(degen=degen), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias", "INP": "inp", "INM": "inm",
        "FFEP": "ffep", "FFEN": "ffen", "OUTP": "outp", "OUTN": "outn"})
    _diff_pulse(top, "inp", "inm", td=2e-9, per=4e-9)
    with top:
        sources.VoltageSource("vffep", positive="ffep", negative="0", dc=_CM)
        sources.VoltageSource("vffen", positive="ffen", negative="0", dc=_CM)
    top.add_sim_type(Transient(step=10e-12, stop=8e-9, maxstep=30e-12))
    top.simulate(corner="tt")
    swing = _swing(_se(top.load_signal("outp")))


    top = _bias_top("drv_ac")
    top.add_module(build_cml_driver(degen=degen), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias", "INP": "inp", "INM": "inm",
        "FFEP": "ffep", "FFEN": "ffen", "OUTP": "outp", "OUTN": "outn"})
    with top:
        sources.VoltageSource("vip", positive="inp", negative="0", dc=_CM, ac=0.5)
        sources.VoltageSource("vim", positive="inm", negative="0", dc=_CM, ac=-0.5)
        sources.VoltageSource("vfp", positive="ffep", negative="0", dc=_CM)
        sources.VoltageSource("vfn", positive="ffen", negative="0", dc=_CM)
    top.add_sim_type(AC(variation="dec", points=10, fstart=1e6, fstop=50e9))
    top.simulate(corner="tt")
    op, on = top.load_signal("outp"), top.load_signal("outn")
    h = op - on
    h_mag = h.mag()
    lf_db = dc_gain_dB(h)
    peak_db = peak_gain_dB(h)
    boost_db = peak_db - lf_db


    level = 10 ** (lf_db / 20) * 1.41
    fz = (crossing(h_mag, level, edge="rising")
          if peak_db > 20 * np.log10(level) else float("nan"))
    return {"swing_mv": swing, "lf_db": lf_db, "peak_db": peak_db,
            "boost_db": boost_db, "zero_ghz": fz / 1e9}


def characterize_predriven_driver(*, degen=True):
    """Measure launch swing with a predriver feeding the output stage.

    The stimulus and degeneration match the bare-driver characterization."""
    top = _bias_top("drv_pd")
    top.add_module(build_cml_predriver(), "pd", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias", "INP": "inp", "INM": "inm",
        "OUTP": "pdp", "OUTN": "pdn"})
    top.add_module(build_cml_driver(degen=degen), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias", "INP": "pdp", "INM": "pdn",
        "FFEP": "ffep", "FFEN": "ffen", "OUTP": "outp", "OUTN": "outn"})
    _diff_pulse(top, "inp", "inm", td=2e-9, per=4e-9)
    with top:
        sources.VoltageSource("vffep", positive="ffep", negative="0", dc=_CM)
        sources.VoltageSource("vffen", positive="ffen", negative="0", dc=_CM)
    top.add_sim_type(Transient(step=10e-12, stop=8e-9, maxstep=30e-12))
    top.simulate(corner="tt")
    op = _se(top.load_signal("outp"))
    on = _se(top.load_signal("outn"))
    return {"swing_mv": _swing(op), "diff_mv": _swing(op - on),
            "pd_swing_mv": _swing(_se(top.load_signal("pdp")))}


def _prbs9(n):
    """Generate all-ones, XOR, pre-shift PRBS9 through the public generator."""
    if n <= 0:
        return []
    if n == 1:
        return [1]
    # Complementing XNOR output gives XOR bits; the leading bit precedes the shift.
    return [1] + [1 - bit for bit in sources.prbs_bits(9, n - 1, seed=0)]


def _run_ffe_driver(bits, *, tbit=_FFE_TBIT, t0=_FFE_T0, simulator='ngspice', **drv_kw):
    """Drive the four tap inputs with delayed PRBS waveforms.

    Return time and differential launch voltage. The behavioral option uses
    ideal steering at the nominal reference current. It omits predriver
    delay, settling and finite mirror/output resistance. Register data
    sources before VDD and place taps before the load so their currents
    are available when the load evaluates."""
    N = len(bits) - 4
    if simulator == 'behavioral':
        top = Module('PDK_skywater130', example_output_dir(), 'cml_ffe_behavioral.cir',
                     pdk_details=pdk)
        top.simulator = simulator
    else:
        top = _bias_top("cml_ffe")
        top.simulator = simulator
    nets = {"VDD": "VDD", "VSS": "0", "IBIAS": "ibias", "OUTP": "outp", "OUTN": "outn"}
    for nm in _FFE_NAMES:
        nets[f"{nm.upper()}P"] = f"{nm}p"; nets[f"{nm.upper()}N"] = f"{nm}n"
    top.add_module(build_cml_ffe_driver(**drv_kw), "drv", nets=nets)

    def pwl(sh, hi, lo):
        p = sources.PWL().add(0, hi if bits[0] else lo)
        for k in range(1, N + 4):
            pv, idx = k - 1 + sh, k + sh
            prev = bits[pv] if 0 <= pv < len(bits) else bits[0]
            cur = bits[idx] if 0 <= idx < len(bits) else bits[-1]
            tk = t0 + k * tbit
            p.add(tk - 0.4 * tbit, hi if prev else lo).add(tk - 0.35 * tbit, hi if cur else lo)
        p.add((N + 5) * tbit + t0, lo)
        return p
    with top:
        for nm in _FFE_NAMES:
            s = _FFE_SHIFT[nm]
            sources.VoltageSource(f"v{nm}p", positive=f"{nm}p", negative="0",  pwl=pwl(s, _FFE_HI, _FFE_LO))
            sources.VoltageSource(f"v{nm}n", positive=f"{nm}n", negative="0",  pwl=pwl(s, _FFE_LO, _FFE_HI))
        if simulator == 'behavioral':
            # Data must be available before the current-summing load evaluates.
            sources.VoltageSource('vdd', positive='VDD', negative='0', dc=_VDD)
    top.add_sim_type(Transient(step=tbit / 60, stop=t0 + (N + 4) * tbit, maxstep=tbit / 30))
    top.simulate(corner="tt")
    p, n = top.load_signal("outp"), top.load_signal("outn")
    t = np.asarray(p.x, float)
    launch = np.asarray(p.y, float) - sample(n, x_new=t).y
    return t, launch


def fit_ffe_taps(t, launch, bits, *, tbit=_FFE_TBIT, t0=_FFE_T0):
    """Fit the driver's four-tap FIR by least squares from its launch: launch[k] ~
    sum_j c_j * b[k-delay_j]. Returns (taps normalized to main, main_volts, R^2)."""
    N = len(bits) - 4
    nrz = np.array([2 * x - 1 for x in bits], float)
    lm = launch - launch.mean()
    best = None
    for L in range(0, 6):
        for ph in np.linspace(0.4, 0.95, 12):
            ks = np.arange(14, N - 4)
            y = np.interp(t0 + (ks + L + ph) * tbit, t, lm)
            X = np.column_stack([nrz[ks - d] for d in (-1, 0, 1, 2)] + [np.ones(len(ks))])
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - np.var(y - X @ c) / np.var(y)
            if best is None or r2 > best[0]:
                best = (r2, c)
    r2, c = best
    return tuple(c[:4] / c[1]), float(c[1]), r2


def characterize_cml_ffe_driver(bits=None):
    """Fit tap ratios and measure differential launch p2-p98 swing over PRBS9.

    This amplitude measurement does not validate a channel or a full TX chain."""
    if bits is None:
        bits = _prbs9(160)
    t, launch = _run_ffe_driver(bits)
    taps, main_v, r2 = fit_ffe_taps(t, launch, bits)
    return {"taps": taps, "main_mv": main_v * 1e3, "pkpk_mv": _swing(launch), "r2": r2}


def characterize_cml2cmos():
    top = _bias_top("cml2cmos_tb")
    top.add_module(build_cml2cmos(), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
        "INP": "inp", "INM": "inm", "Y": "y"})
    _diff_pulse(top, "inp", "inm", td=2e-9, per=4e-9)
    top.add_sim_type(Transient(step=20e-12, stop=10e-9, maxstep=50e-12))
    top.simulate(corner="tt")
    y = _se(top.load_signal("y"))
    return {"y_swing_mv": _swing(y), "y_hi": float(np.percentile(y,98)), "y_lo": float(np.percentile(y,2))}


def _cml_tp(inp, outp):
    """Average rising and falling INP-to-OUTP propagation delays.

    Use the input common mode and the output mid-swing as thresholds."""
    tplh = propagation_delay(inp, outp, in_threshold=_CM, in_edge="rising")
    tphl = propagation_delay(inp, outp, in_threshold=_CM, in_edge="falling")
    return float(np.mean([tplh, tphl]) * 1e12)


def _clk_to_q(clkd, outd):
    """clk-to-Q: a mid-run CLK rising zero-cross -> the next OUT crossing of
    its time-mean level, either direction."""
    return float(propagation_delay(
        clkd, outd, in_threshold=0.0, out_threshold=float(np.mean(outd.y)),
        in_edge="rising", out_edge="any", start=6e-9) * 1e12)


def save_plot(buf, dff, c2c, path):
    """Plot buffer switching, flip-flop capture and CMOS-to-CML conversion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    for ax, d, title in ((ax1, buf, "CML buffer: differential output (~0.4 Vpp SE)"),
                         (ax2, dff, "CML DFF: OUTP / OUTN captured on CLK edges"),
                         (ax3, c2c, "cmos2cml: CMOS IN -> CML differential")):
        t = np.asarray(d["t"], dtype=float) * 1e9
        ax.plot(t, d["op"], color="#0072B2", lw=1.4, label="OUTP")
        ax.plot(t, d["on"], color="#D55E00", lw=1.4, label="OUTN")
        if "drv" in d:
            ax.plot(t, d["drv"], color="#9fb4c7", lw=1.0, label=d["drvlab"])
        ax.set_title(title)
        ax.set_ylabel("V")
        ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    ax3.set_xlabel("time (ns)")
    fig.suptitle("cml_cells — differential current-mode-logic library (sky130)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _waveforms_buffer():
    top = _bias_top("cml_buf_wf")
    top.add_module(build_cml_buffer(), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
        "INP": "inp", "INM": "inm", "OUTP": "outp", "OUTN": "outn"})
    _diff_pulse(top, "inp", "inm", td=2e-9, per=4e-9)
    top.add_sim_type(Transient(step=20e-12, stop=10e-9, maxstep=50e-12))
    top.simulate(corner="tt")
    return {"t": top.load_signal("outp").x, "op": _se(top.load_signal("outp")),
            "on": _se(top.load_signal("outn")),
            "drv": _se(top.load_signal("inp")), "drvlab": "INP"}


def _waveforms_dff():
    top = _bias_top("cml_dff_wf")
    top.add_module(build_cml_dff(), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
        "CLKP": "clkp", "CLKN": "clkn", "DP": "dp", "DN": "dn",
        "OUTP": "outp", "OUTN": "outn"})
    _diff_pulse(top, "clkp", "clkn", td=1e-9, per=2e-9)
    _diff_pulse(top, "dp", "dn", td=1.5e-9, per=8e-9)
    top.add_sim_type(Transient(step=10e-12, stop=20e-9, maxstep=30e-12))
    top.simulate(corner="tt")
    return {"t": top.load_signal("outp").x, "op": _se(top.load_signal("outp")),
            "on": _se(top.load_signal("outn"))}


def _waveforms_cmos2cml():
    top = _bias_top("cmos2cml_wf")
    top.add_module(build_cmos2cml(), "dut", nets={
        "VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
        "IN": "cin", "OUTP": "outp", "OUTN": "outn"})
    with top:
        sources.VoltageSource("vin", positive="cin", negative="0", dc=0,
            pulse=sources.Pulse(v1=0, v2=_VDD, td=2e-9, tr=80e-12, tf=80e-12,
                                pw=2e-9, per=4e-9))
    top.add_sim_type(Transient(step=20e-12, stop=10e-9, maxstep=50e-12))
    top.simulate(corner="tt")
    return {"t": top.load_signal("outp").x, "op": _se(top.load_signal("outp")),
            "on": _se(top.load_signal("outn")),
            "drv": _se(top.load_signal("cin")), "drvlab": "CMOS IN"}


_DEVCOUNT = {"cml_bias": 1, "cml_buffer": 3, "cml_latch": 7, "cml_dff": 14,
             "cml_mux2": 7, "cmos2cml": 3 + 2, "cml2cmos": 5 + 4, "cml_driver": 8,
             "cml_predriver": 3, "cml_ffe_driver": 4 * (3 + 3)}


def _zero_text(drv):
    """Describe the +3 dB crossing, or report that the boost does not reach it."""
    if np.isnan(drv["zero_ghz"]):
        return "no zero found (boost below +3 dB)"
    return f"zero ~{drv['zero_ghz']:.1f} GHz"


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 72)
    print(f"cml_cells — {_IREF*1e6:.0f} uA tails, {_RL:.0f} ohm loads, "
          f"~{_SWING*1e3:.0f} mV SE swing target")
    print("=" * 72)

    buf = characterize_buffer()
    mux = characterize_mux()
    dff = characterize_dff("dff")
    lat = characterize_dff("latch")
    c2c = characterize_cmos2cml()
    c2m = characterize_cml2cmos()

    print(f"  cml_buffer : swing {buf['swing_mv']:.0f} mV, tp {buf['tp_ps']:.0f} ps")
    print(f"  cml_mux2   : SEL->A {mux['sel_A_diff_mv']:+.0f} mV, "
          f"SEL->B {mux['sel_B_diff_mv']:+.0f} mV (differential; opposite signs = ok)")
    print(f"  cml_latch  : swing {lat['swing_mv']:.0f} mV "
          f"(transparent latch — clk-Q is a DFF metric, below)")
    print(f"  cml_dff    : clk-Q {dff['clk_q_ps']:.0f} ps, swing {dff['swing_mv']:.0f} mV")
    print(f"  cmos2cml   : out swing {c2c['out_swing_mv']:.0f} mV")
    print(f"  cml2cmos   : Y {c2m['y_lo']:.2f}..{c2m['y_hi']:.2f} V "
          f"(swing {c2m['y_swing_mv']:.0f} mV)")
    drv = characterize_driver(degen=True)
    drv_flat = characterize_driver(degen=False)
    print(f"  cml_driver : launch {drv['swing_mv']:.0f} mV into 50 ohm; "
          f"degeneration peaking {drv['boost_db']:.1f} dB (vs {drv_flat['boost_db']:.1f} "
          f"dB flat), {_zero_text(drv)}")
    pdr = characterize_predriven_driver(degen=True)
    print(f"  + predriver: launch {drv['swing_mv']:.0f} -> {pdr['swing_mv']:.0f} mV SE "
          f"({drv['swing_mv']*2:.0f} -> {pdr['diff_mv']:.0f} mV diff) — the "
          f"{pdr['pd_swing_mv']:.0f} mV overdrive commits the degenerated pair, "
          f"main-driver peaking +{drv['boost_db']:.1f} dB, +{_PD_IMUL*_IREF*1e3:.1f} mA")
    ffe = characterize_cml_ffe_driver()
    ft = ffe["taps"]
    print(f"  cml_ffe_drv: 4-tap current FFE realizes taps "
          f"[{ft[0]:+.2f},{ft[1]:+.2f},{ft[2]:+.2f},{ft[3]:+.2f}] "
          f"(design [{_FFE_TAPS[0]:+.2f},{_FFE_TAPS[1]:+.2f},{_FFE_TAPS[2]:+.2f},{_FFE_TAPS[3]:+.2f}], "
          f"R2={ffe['r2']:.3f}) — launch {ffe['pkpk_mv']:.0f} mV pk-pk")

    print("\n" + "=" * 72)
    print("cml cell catalog")
    print("=" * 72)
    print(f"  {'cell':<12}{'devices':>8}   note")
    for name, dev in _DEVCOUNT.items():
        print(f"  {name:<12}{dev:>8}")

    png = save_plot(_waveforms_buffer(), _waveforms_dff(), _waveforms_cmos2cml(),
                    os.path.join(OUT, "cml_cells.png"))
    print(f"\n  saved {png}")
