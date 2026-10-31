"""The PLL's digital blocks: a PFD and a /24 feedback divider.

NAND gates form positive-edge D flip-flops with active-low clear. Two
flip-flops and a NAND reset form the PFD. A ripple /8 followed by a
synchronous /3 provides the /24 feedback path.

The transfer tests compare net UP-minus-DN duty at 1 ns and 4 ns offsets.
They enforce lower bounds; the printed ratio has no pass/fail limit.
The frequency-detection test drives FB slower than REF. The zero-offset
test measures the simultaneous reset pulses. Divider tests measure fin/fout
and duty for /8, /3 and /24 at their loop input rates.
The behavioral PLL imports the stored nominal reset width.

Run from the repository root with ngspice and sky130 models installed:
    MPLBACKEND=Agg .venv/bin/python -m silk.examples.pfd_example

The script prints spec results, transfer duties, their ratio, reset width
and divider measurements, and saves pulse and divider plots.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.simulation import Transient
from silk.metrics import average_frequency, average_duty

from silk.examples.ring_oscillator_example import build_nand2, build_inverter, _WN, _WP, _L
from silk.examples import example_output_dir

pdk = load_pdk('sky130')

_VDD = 1.8
F_REF = 25e6
_T_PER = 1.0 / F_REF


GAIN1_METRIC = "net_duty_1ns"
GAIN4_METRIC = "net_duty_4ns"

FREQDET_METRIC = "freqdet_net"
PULSE_METRIC = "reset_pulse_ps"
DIV_METRIC = "div_ratio"
DUTY_METRIC = "div_duty_pct"


# Run pll_calibrate.py to compare measurements with this nominal reset width.
T_RESET_NOM_PS = 294.0


def build_nand3(*, wn=_WN, wp=_WP, l=_L) -> Module:
    """Build a three-input NAND with parallel PMOS and series NMOS devices."""
    m = Module("PDK_skywater130", example_output_dir(), "nand3.cir", pdk_details=pdk,
               module_name="nand3")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "B": Port.analog_input(),
        "C": Port.analog_input(), "Y": Port.analog_output(),
    })
    with m:
        pdk.pmos1p8("MPA", d="Y", g="A", s="VDD", b="VDD", w=wp, l=l)
        pdk.pmos1p8("MPB", d="Y", g="B", s="VDD", b="VDD", w=wp, l=l)
        pdk.pmos1p8("MPC", d="Y", g="C", s="VDD", b="VDD", w=wp, l=l)
        pdk.nmos1p8("MNA", d="Y", g="A", s="nx1", b="VSS", w=3 * wn, l=l)
        pdk.nmos1p8("MNB", d="nx1", g="B", s="nx2", b="VSS", w=3 * wn, l=l)
        pdk.nmos1p8("MNC", d="nx2", g="C", s="VSS", b="VSS", w=3 * wn, l=l)
    return m


def build_dff() -> Module:
    """Build a positive-edge D flip-flop with active-low clear.

    Six NAND gates implement the latch and clear paths. CLRB=0 forces
    Q=0 and QB=1 asynchronously.
    """
    m = Module("PDK_skywater130", example_output_dir(), "dff.cir", pdk_details=pdk,
               module_name="dff")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "D": Port.analog_input(), "CLK": Port.analog_input(),
        "CLRB": Port.analog_input(),
        "Q": Port.analog_output(), "QB": Port.analog_output(),
    })
    nand2 = build_nand2()
    nand3 = build_nand3()
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
    return m


def build_pfd(*, with_tests=True) -> Module:
    """Build two D-high flip-flops with a shared NAND reset.

    REF sets UP, FB sets DN, and both high clears the flip-flops.
    Reset-path delay produces pulses at zero phase offset.
    """
    mod = Module("PDK_skywater130", example_output_dir(), "pfd.cir", pdk_details=pdk,
                 module_name="pfd")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "REF": Port.analog_input(), "FB": Port.analog_input(),
        "UP": Port.analog_output(), "DN": Port.analog_output(),
    })
    dff = build_dff()
    nand2 = build_nand2()

    mod.add_module(dff, "ff_up", nets={
        "VDD": "VDD", "VSS": "VSS", "D": "VDD", "CLK": "REF", "CLRB": "rstb",
        "Q": "UP", "QB": "up_b"})
    mod.add_module(dff, "ff_dn", nets={
        "VDD": "VDD", "VSS": "VSS", "D": "VDD", "CLK": "FB", "CLRB": "rstb",
        "Q": "DN", "QB": "dn_b"})
    mod.add_module(nand2, "g_rst", nets={
        "VDD": "VDD", "VSS": "VSS", "A": "UP", "B": "DN", "Y": "rstb"})

    if with_tests:
        attach_transfer_tests(mod)
        attach_freqdet_test(mod)
        attach_deadzone_test(mod)
    return mod


def build_ripple_divider(*, stages=3, with_tests=True) -> Module:
    """Build a /2^stages chain of toggle flip-flops.

    Each Q clocks the next stage; D is tied to QB. Pulse RSTB low
    at startup to establish the initial state.
    """
    mod = Module("PDK_skywater130", example_output_dir(), "divider.cir", pdk_details=pdk,
                 module_name=f"div{2**stages}")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "CLK": Port.analog_input(), "RSTB": Port.analog_input(),
        "OUT": Port.analog_output(),
    })
    dff = build_dff()
    clk = "CLK"
    for i in range(stages):
        q = "OUT" if i == stages - 1 else f"q{i}"
        qb = f"qb{i}"
        mod.add_module(dff, f"ff{i}", nets={
            "VDD": "VDD", "VSS": "VSS", "D": qb, "CLK": clk, "CLRB": "RSTB",
            "Q": q, "QB": qb})
        clk = q
    if with_tests:
        attach_divider_test(mod, ratio=2 ** stages)
    return mod


def build_div3(*, with_tests=True) -> Module:
    """Build a synchronous /3 counter with two flip-flops.

    D0=QB1*QB0 and D1=Q0 give the sequence 00, 01, 10. OUT=Q1 is
    high for one clock in three. Pulse RSTB low at startup.
    """
    mod = Module("PDK_skywater130", example_output_dir(), "div3.cir", pdk_details=pdk,
                 module_name="div3")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "CLK": Port.analog_input(), "RSTB": Port.analog_input(),
        "OUT": Port.analog_output(),
    })
    dff = build_dff()
    mod.add_module(dff, "ff0", nets={
        "VDD": "VDD", "VSS": "VSS", "D": "d0", "CLK": "CLK", "CLRB": "RSTB",
        "Q": "q0", "QB": "qb0"})
    mod.add_module(dff, "ff1", nets={
        "VDD": "VDD", "VSS": "VSS", "D": "q0", "CLK": "CLK", "CLRB": "RSTB",
        "Q": "OUT", "QB": "qb1"})
    mod.add_module(build_nand2(), "g_d0", nets={
        "VDD": "VDD", "VSS": "VSS", "A": "qb1", "B": "qb0", "Y": "d0_b"})
    mod.add_module(build_inverter(), "g_inv", nets={
        "VDD": "VDD", "VSS": "VSS", "A": "d0_b", "Y": "d0"})
    if with_tests:
        attach_divider_test(mod, ratio=3, f_in=75e6, duty_bounds=(25.0, 45.0),
                            stop=600e-9, settle=100e-9, maxstep=50e-12)
    return mod


def build_divider24(*, with_tests=True) -> Module:
    """Cascade a ripple /8 and synchronous /3 with a shared reset.

    At 600 MHz input, the /3 next-state logic runs at 75 MHz.
    """
    mod = Module("PDK_skywater130", example_output_dir(), "div24.cir", pdk_details=pdk,
                 module_name="div24")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "CLK": Port.analog_input(), "RSTB": Port.analog_input(),
        "OUT": Port.analog_output(),
    })
    mod.add_module(build_ripple_divider(with_tests=False), "u_div8", nets={
        "VDD": "VDD", "VSS": "VSS", "CLK": "CLK", "RSTB": "RSTB",
        "OUT": "clk75"})
    mod.add_module(build_div3(with_tests=False), "u_div3", nets={
        "VDD": "VDD", "VSS": "VSS", "CLK": "clk75", "RSTB": "RSTB",
        "OUT": "OUT"})
    if with_tests:
        attach_divider_test(mod, ratio=24, f_in=600e6,
                            duty_bounds=(25.0, 45.0),
                            stop=600e-9, settle=100e-9, maxstep=20e-12)
    return mod


def _net_duty(tb, name, *, settle=200e-9):
    """Register UP-minus-DN duty after settling; return the UP duty handle."""
    up = tb.add_metric(f"{name}_up_duty", average_duty, sig=tb.probe("up"),
                       threshold=_VDD / 2, start=settle)
    dn = tb.add_metric(f"{name}_dn_duty", average_duty, sig=tb.probe("dn"),
                       threshold=_VDD / 2, start=settle)
    tb.add_metric(name, up - dn)
    return up


def _pfd_rig(tb, *, fb_delay_ns=0.0, fb_period=_T_PER):
    """Drive REF at 25 MHz and FB with a selected delay and period."""
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "REF": "ref", "FB": "fb",
                              "UP": "up", "DN": "dn"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    tb.add_source(sources.VoltageSource(
        "ref", positive="ref", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=_VDD, td=1e-9, tr=100e-12, tf=100e-12,
                            pw=_T_PER / 2, per=_T_PER)))
    tb.add_source(sources.VoltageSource(
        "fb", positive="fb", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=_VDD, td=1e-9 + fb_delay_ns * 1e-9,
                            tr=100e-12, tf=100e-12,
                            pw=fb_period / 2, per=fb_period)))
    tb.add_sim_type(Transient(step=20e-12, stop=2e-6, maxstep=50e-12))


def attach_transfer_tests(mod):
    """Compare net output duty at two phase offsets.

    Check a lower bound at each offset. The script prints the ratio
    separately without a pass/fail limit.
    """
    for name, delay, metric in (("xfer_1ns", 1.0, GAIN1_METRIC),
                                ("xfer_4ns", 4.0, GAIN4_METRIC)):
        tb = mod.create_test(name)
        _pfd_rig(tb, fb_delay_ns=delay)
        _net_duty(tb, metric)
        expected = delay * 1e-9 / _T_PER
        tb.add_spec(f"net duty at {delay:.0f} ns > {expected*0.8*100:.1f}%",
                    metric, ">", expected * 0.8)
    return mod


def attach_freqdet_test(mod):
    """Check positive net duty with FB at 20 MHz and REF at 25 MHz."""
    tb = mod.create_test("freqdet")
    _pfd_rig(tb, fb_period=1.0 / 20e6)
    _net_duty(tb, FREQDET_METRIC)
    tb.add_spec("frequency detection: net > 0.1", FREQDET_METRIC, ">", 0.1)
    return mod


def attach_deadzone_test(mod):
    """Measure reset-pulse width at zero input phase offset.

    The PLL behavioral model uses the nominal reset width stored above.
    """
    tb = mod.create_test("deadzone")
    _pfd_rig(tb, fb_delay_ns=0.0)
    up_duty = tb.add_metric("deadzone_up_duty", average_duty,
                            sig=tb.probe("up"), threshold=_VDD / 2,
                            start=200e-9)
    tb.add_metric(PULSE_METRIC, up_duty * _T_PER * 1e12)
    tb.add_spec("reset pulse > 50 ps (no dead zone)", PULSE_METRIC, ">", 50.0)
    return mod


def attach_divider_test(mod, *, ratio, f_in=600e6, duty_bounds=(45.0, None),
                        stop=300e-9, settle=50e-9, maxstep=10e-12):
    """Measure fin/fout and output duty after an initial reset.

    Hold RSTB low for 5 ns. duty_bounds gives lower and optional upper
    duty limits in percent.
    """
    per = 1.0 / f_in
    tb = mod.create_test("divide")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "CLK": "clk",
                              "RSTB": "rstb", "OUT": "out"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    tb.add_source(sources.VoltageSource(
        "clk", positive="clk", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=_VDD, td=1e-9, tr=50e-12, tf=50e-12,
                            pw=per / 2, per=per)))
    tb.add_source(sources.VoltageSource(
        "rstb", positive="rstb", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(5e-9, 0).add(5.2e-9, _VDD)))
    tb.add_sim_type(Transient(step=maxstep / 2, stop=stop, maxstep=maxstep))

    f_out = tb.add_metric("f_out", average_frequency, sig=tb.probe("out"),
                          threshold=_VDD / 2, start=settle)
    tb.add_metric(DIV_METRIC, f_in / f_out)
    out_duty = tb.add_metric("div_duty", average_duty, sig=tb.probe("out"),
                             threshold=_VDD / 2, start=settle)
    tb.add_metric(DUTY_METRIC, out_duty * 100)
    tb.add_spec(f"divides by {ratio} (low)", DIV_METRIC, ">", ratio * 0.99)
    tb.add_spec(f"divides by {ratio} (high)", DIV_METRIC, "<", ratio * 1.01)
    lo, hi = duty_bounds
    tb.add_spec(f"output duty > {lo:.0f}%", DUTY_METRIC, ">", lo)
    if hi is not None:
        tb.add_spec(f"output duty < {hi:.0f}%", DUTY_METRIC, "<", hi)
    return mod


def run_characterization(mod, corner="tt"):
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def save_plot(pfd_res, d8_res, d3_res, d24_res, path):
    """Plot measured reset pulses and the three divider waveforms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

    a = ax[0][0]
    ckt = pfd_res["pfd.deadzone"].sim_results
    for sig, color in (("up", "#0072B2"), ("dn", "#D55E00")):
        s = ckt.load_signal(sig)
        t = np.asarray(s.x, dtype=float)
        w = (t > 240.8e-9) & (t < 242.3e-9)
        a.plot((t[w] - 241e-9) * 1e9, np.asarray(s.y, dtype=float)[w],
               color=color, lw=1.4, label=sig.upper())
    pulse = pfd_res["pfd.deadzone"].metrics[PULSE_METRIC]
    a.set_title(f"zero offset: reset pulse {pulse:.0f} ps")
    a.set_xlabel("time (ns)")
    a.set_ylabel("V")
    a.legend(frameon=False, fontsize=8)

    for a, res, key, span, title in (
            (ax[0][1], d8_res, "div8.divide", 60e-9,
             "/8 ripple at 600 MHz (out duty regenerated ~50%)"),
            (ax[1][0], d3_res, "div3.divide", 160e-9,
             "/3 synchronous at 75 MHz (one-in-three duty)"),
            (ax[1][1], d24_res, "div24.divide", 130e-9,
             "/24 = /8 x /3 at 600 MHz -> 25 MHz = f_ref")):
        ckt = res[key].sim_results
        for sig, color, lw, alpha in (("clk", "#9fb4c7", 0.7, 0.8),
                                      ("out", "#0072B2", 1.5, 1.0)):
            s = ckt.load_signal(sig)
            t = np.asarray(s.x, dtype=float)
            w = (t > 200e-9) & (t < 200e-9 + span)
            a.plot((t[w] - 200e-9) * 1e9, np.asarray(s.y, dtype=float)[w],
                   color=color, lw=lw, alpha=alpha,
                   label="CLK in" if sig == "clk" else "OUT")
        a.set_title(title)
        a.set_xlabel("time (ns)")
        a.set_ylabel("V")
        a.legend(frameon=False, fontsize=8, loc="upper right")

    for row in ax:
        for a in row:
            a.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
            a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 70)
    print("PFD (6-NAND DFFs + reset) — the PLL's phase-frequency detector")
    print("=" * 70)
    pfd = build_pfd()
    results, recon = run_characterization(pfd)
    print(recon.summary())
    g1 = results["pfd.xfer_1ns"].metrics[GAIN1_METRIC]
    g4 = results["pfd.xfer_4ns"].metrics[GAIN4_METRIC]
    print(f"  transfer: 1 ns -> {g1*100:5.2f}%  (ideal 2.50%)")
    print(f"            4 ns -> {g4*100:5.2f}%  (ideal 10.00%)")
    print(f"            4 ns / 1 ns duty ratio = {g4/g1:.2f}  (ideal 4.00)")
    print(f"  freq detect (20 vs 25 MHz): net = "
          f"{results['pfd.freqdet'].metrics[FREQDET_METRIC]:+.3f}")
    pulse = results["pfd.deadzone"].metrics[PULSE_METRIC]
    print(f"  anti-dead-zone pulse = {pulse:.0f} ps  (in-code store: "
          f"T_RESET_NOM_PS = {T_RESET_NOM_PS:.0f}; refresh via pll_calibrate.py)")

    print()
    print("=" * 70)
    print("Ripple divider (/8) at the PLL's 600 MHz")
    print("=" * 70)
    div = build_ripple_divider()
    dres, drecon = run_characterization(div)
    print(drecon.summary())
    print(f"  measured ratio = {dres['div8.divide'].metrics[DIV_METRIC]:.3f}")
    print(f"  output duty    = {dres['div8.divide'].metrics[DUTY_METRIC]:.1f}%")

    print()
    print("=" * 70)
    print("Synchronous /3 at 75 MHz (its rate behind the /8)")
    print("=" * 70)
    d3 = build_div3()
    r3, rec3 = run_characterization(d3)
    print(rec3.summary())
    print(f"  measured ratio = {r3['div3.divide'].metrics[DIV_METRIC]:.3f}")
    print(f"  output duty    = {r3['div3.divide'].metrics[DUTY_METRIC]:.1f}%  "
          f"(one-in-three -> 33.3%)")

    print()
    print("=" * 70)
    print("/24 = /8 x /3 at 600 MHz — the PLL's feedback path, end to end")
    print("=" * 70)
    d24 = build_divider24()
    r24, rec24 = run_characterization(d24)
    print(rec24.summary())
    ratio24 = r24["div24.divide"].metrics[DIV_METRIC]
    print(f"  measured ratio = {ratio24:.3f}")
    print(f"  600 MHz in -> {600e6 / ratio24 / 1e6:.3f} MHz out  "
          f"(F_REF = {F_REF/1e6:.0f} MHz)")
    print(f"\n  saved "
          f"{save_plot(results, dres, r3, r24, os.path.join(OUT, 'pfd_dividers.png'))}")
