"""Transistor PLL with a precharge switch for a near-lock release.

The loop combines the LDO, ring VCO, /24 divider, PFD, charge pump and
computed R/C filter. The testbench precharges VCTRL, enables the ring at
20 ns, releases divider reset at 60 ns, starts REF at 100 ns and releases
the precharge switch at 300 ns. It measures output and feedback frequencies,
VCTRL and the difference between two tail-window means. The mean-drift check
supports the frequency checks; it cannot establish lock by itself.

Two inverters buffer the ring output into the 1.8 V divider. Their input
swing can cause static contention. The precharge gate is driven to 3.3 V
to turn the switch off. This uses 1.8 V devices outside their nominal rail;
consult the PDK terminal-voltage limits before reusing the circuit.
The test measures release near lock, not cold-start acquisition.

Run from the repository root with ngspice and sky130 models installed:
    MPLBACKEND=Agg .venv/bin/python -m silk.examples.pll_spice_example

The script prints spec results, tail frequencies, VCTRL, mean drift and
REF-to-FB edge offset, and saves a transient plot.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.elements import elements
from silk.metrics import average_frequency, crossings, average
from silk.simulation import Transient

from silk.examples.supply_regulated_vco_example import build_supply_regulated_vco
from silk.examples.ring_oscillator_example import build_inverter
from silk.examples.pfd_example import build_pfd, build_divider24
from silk.examples.charge_pump_example import build_charge_pump, I_CP
from silk.examples.ldo_regulator_example import _VDD as _RAIL, _I_AMP
from silk.examples.pll_example import design_loop_filter, F_REF, F_TARGET, N_DIV, _RING_P
from silk.examples import example_output_dir

pdk = load_pdk('sky130')

_VDD_DIG = 1.8


V_LOCK_PRED = (F_TARGET / 1e6 - _RING_P["f0_mhz"]) / _RING_P["kvco_mhz_per_v"]


_T_EN = 20e-9
_T_RSTB = 60e-9
_T_REF0 = 100e-9
_T_RELEASE = 300e-9

_T_STOP = 3.0e-6
_T_MAXSTEP = 20e-12
# Measure the tail after the release excursion.
_T_TAIL = 2.2e-6


FOUT_METRIC = "f_out_mhz"
FFB_METRIC = "f_fb_mhz"
VLOCK_METRIC = "vctrl_lock_v"
DRIFT_METRIC = "vctrl_drift_mv"
VCTRL_FIRST_METRIC = "vctrl_tail_first_v"
VCTRL_SECOND_METRIC = "vctrl_tail_second_v"


def build_pll_spice(*, filt=None, with_tests=True) -> Module:
    """Build the transistor PLL with VFORCE/PRECHB precharge controls."""
    filt = filt or design_loop_filter(verbose=False)
    mod = Module("PDK_skywater130", example_output_dir(), "pll_spice.cir", pdk_details=pdk,
                 module_name="pll_spice")
    mod.define_module_ports({
        "VDD33": Port.supply(), "VDD18": Port.supply(), "VSS": Port.ground(),
        "REF": Port.analog_input(), "EN": Port.analog_input(),
        "RSTB": Port.analog_input(),
        "IBIAS_AMP": Port.analog_input(), "IBIAS_CP": Port.analog_input(),
        "VFORCE": Port.analog_input(), "PRECHB": Port.analog_input(),
        "OUT": Port.analog_output(), "FB": Port.analog_output(),
        "VCTRL": Port.analog_output(),
    })

    mod.add_module(build_supply_regulated_vco(with_tests=False), "u_vco", nets={
        "VDD": "VDD33", "VSS": "VSS", "VCTRL": "VCTRL", "EN": "EN",
        "IBIAS_AMP": "IBIAS_AMP", "OUT": "OUT"})

    mod.add_module(build_inverter(), "u_buf1", nets={
        "VDD": "VDD18", "VSS": "VSS", "A": "OUT", "Y": "oscb"})
    mod.add_module(build_inverter(), "u_buf2", nets={
        "VDD": "VDD18", "VSS": "VSS", "A": "oscb", "Y": "clkd"})
    mod.add_module(build_divider24(with_tests=False), "u_div", nets={
        "VDD": "VDD18", "VSS": "VSS", "CLK": "clkd", "RSTB": "RSTB",
        "OUT": "FB"})
    mod.add_module(build_pfd(with_tests=False), "u_pfd", nets={
        "VDD": "VDD18", "VSS": "VSS", "REF": "REF", "FB": "FB",
        "UP": "up", "DN": "dn"})
    mod.add_module(build_charge_pump(with_tests=False), "u_cp", nets={
        "VDD": "VDD18", "VSS": "VSS", "IBIAS": "IBIAS_CP",
        "UP": "up", "DN": "dn", "OUT": "VCTRL"})

    with mod:

        elements.resistor("RF", p="VCTRL", n="nc1",
                          value=filt["r_ohm"])
        elements.capacitor("C1F", p="nc1", n="VSS",
                           value=filt["c1_f"])
        elements.capacitor("C2F", p="VCTRL", n="VSS",
                           value=filt["c2_f"])

        # Tie bulk to the high rail to avoid forward-biasing the body diode.
        pdk.pmos1p8("MPRE", d="VCTRL", g="PRECHB", s="VFORCE", b="VDD33",
                    w=5.0, l=0.5, m=4)

    if with_tests:
        attach_lock_test(mod)
    return mod


def attach_lock_test(mod):
    """Register a near-lock release transient with frequency and mean-drift checks."""
    tb = mod.create_test("lock")
    tb.add_module_under_test({
        "VDD33": "vdd33", "VDD18": "vdd18", "VSS": "0",
        "REF": "ref", "EN": "en", "RSTB": "rstb",
        "IBIAS_AMP": "ibias_amp", "IBIAS_CP": "ibias_cp",
        "VFORCE": "vforce", "PRECHB": "prechb",
        "OUT": "out", "FB": "fb", "VCTRL": "vctrl"})
    tb.add_source(sources.VoltageSource("vdd33", positive="vdd33", negative="0",
                                        dc=_RAIL))
    tb.add_source(sources.VoltageSource("vdd18", positive="vdd18", negative="0",
                                        dc=_VDD_DIG))
    tb.add_source(sources.CurrentSource("iamp", positive="ibias_amp",
                                        negative="0", dc=_I_AMP))
    tb.add_source(sources.CurrentSource("icp", positive="vdd18",
                                        negative="ibias_cp", dc=I_CP))
    tb.add_source(sources.VoltageSource("vforce", positive="vforce",
                                        negative="0", dc=V_LOCK_PRED))
    tb.add_source(sources.VoltageSource(
        "en", positive="en", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(_T_EN, 0).add(_T_EN + 100e-12, 1.8)))
    tb.add_source(sources.VoltageSource(
        "rstb", positive="rstb", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(_T_RSTB, 0).add(_T_RSTB + 200e-12, 1.8)))
    tb.add_source(sources.VoltageSource(
        "ref", positive="ref", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=_VDD_DIG, td=_T_REF0, tr=100e-12,
                            tf=100e-12, pw=0.5 / F_REF, per=1.0 / F_REF)))

    # Raise PRECHB to the 3.3 V rail to release the filter.
    tb.add_source(sources.VoltageSource(
        "prechb", positive="prechb", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(_T_RELEASE, 0)
                         .add(_T_RELEASE + 1e-9, _RAIL)))
    tb.add_sim_type(Transient(step=_T_MAXSTEP / 2, stop=_T_STOP,
                              maxstep=_T_MAXSTEP))

    out_sig = tb.probe("out")
    fb_sig = tb.probe("fb")
    out_sig_freq = tb.add_metric("out_sig_hz", average_frequency,
                                  sig=out_sig, threshold=0.7, start=_T_TAIL)
    tb.add_metric(FOUT_METRIC, out_sig_freq / 1e6)
    fb_sig_freq = tb.add_metric("fb_sig_hz", average_frequency,
                                  sig=fb_sig, threshold=0.9, start=_T_TAIL)
    tb.add_metric(FFB_METRIC, fb_sig_freq / 1e6)

    # Compare the two half-window means alongside the frequency checks.
    vctrl = tb.probe("vctrl")
    t_mid = _T_TAIL + (_T_STOP - _T_TAIL) / 2
    first = tb.add_metric(VCTRL_FIRST_METRIC, average,
                          sig=vctrl.window(_T_TAIL, t_mid))
    second = tb.add_metric(VCTRL_SECOND_METRIC, average,
                           sig=vctrl.window(t_mid, _T_STOP))
    tb.add_metric(VLOCK_METRIC, (first + second) / 2)
    tb.add_metric(DRIFT_METRIC, abs(second - first) * 1e3)

    f_t = F_TARGET / 1e6
    tb.add_spec(f"f_out within 1% of {f_t:.0f} MHz (low)", FOUT_METRIC,
                ">", f_t * 0.99)
    tb.add_spec(f"f_out within 1% of {f_t:.0f} MHz (high)", FOUT_METRIC,
                "<", f_t * 1.01)
    tb.add_spec("FB returns f_ref (locked divider chain)", FFB_METRIC,
                ">", F_REF / 1e6 * 0.99)
    tb.add_spec("VCTRL half-window mean difference < 10 mV", DRIFT_METRIC,
                "<", 10.0)
    return mod


def phase_offset_ps(ckt):
    """Return the median nearest REF-to-FB rising-edge offset in the tail."""
    sr = ckt.load_signal("ref")
    sf = ckt.load_signal("fb")
    er = crossings(sr, threshold=_VDD_DIG / 2, edge="rising", start=_T_TAIL)
    ef = crossings(sf, threshold=_VDD_DIG / 2, edge="rising", start=_T_TAIL)
    offs = [float(ef[np.argmin(np.abs(ef - e))] - e) for e in er]
    return float(np.median(offs) * 1e12)


def save_plot(ckt, path):
    """Plot VCTRL through release and the tail REF, FB and OUT waveforms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vc = ckt.load_signal("vctrl")
    t = np.asarray(vc.x, dtype=float)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5),
                           gridspec_kw={"height_ratios": [1.0, 1.0]})
    ax[0].plot(t * 1e6, np.asarray(vc.y, dtype=float), color="#0072B2", lw=1.2)
    ax[0].axhline(V_LOCK_PRED, color="#5a5a6e", ls="--", lw=0.8,
                  label=f"behavioral lock {V_LOCK_PRED:.4f} V")
    ax[0].axvline(_T_RELEASE * 1e6, color="#009E73", ls=":", lw=1.2)
    ax[0].text(_T_RELEASE * 1e6, ax[0].get_ylim()[0], "  precharge released",
               color="#009E73", fontsize=9, va="bottom")
    ax[0].set_ylabel("VCTRL (V)")
    ax[0].set_xlabel("time (µs)")
    ax[0].set_title("PLL closed in SPICE — hand-over and hold")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    t0, t1 = _T_STOP - 130e-9, _T_STOP - 10e-9
    for sig, color, label in (("ref", "#5a5a6e", "REF"),
                              ("fb", "#D55E00", "FB (÷24)"),
                              ("out", "#0072B2", "ring OUT")):
        s = ckt.load_signal(sig)
        ts = np.asarray(s.x, dtype=float)
        w = (ts >= t0) & (ts <= t1)
        ax[1].plot(ts[w] * 1e6, np.asarray(s.y, dtype=float)[w],
                   color=color, lw=1.0, label=label,
                   alpha=0.6 if sig == "out" else 1.0)
    ax[1].set_ylabel("V")
    ax[1].set_xlabel("time (µs)")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right", ncol=3)
    for a in ax:
        a.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 72)
    print(f"PLL closed in SPICE — {F_REF/1e6:.0f} MHz x {N_DIV} = "
          f"{F_TARGET/1e6:.0f} MHz, released near lock at t = "
          f"{_T_RELEASE*1e9:.0f} ns")
    print("=" * 72)
    print(f"  precharge target = {V_LOCK_PRED:.4f} V (the behavioral "
          f"prediction under test)")

    mod = build_pll_spice()
    results = mod.run_all_tests(corner="tt")
    recon = mod.reconcile_specs()
    print(recon.summary())

    res = results["pll_spice.lock"]
    f_out = res.metrics[FOUT_METRIC]
    v_lock = res.metrics[VLOCK_METRIC]
    print(f"  f_out      = {f_out:9.2f} MHz "
          f"({(f_out - F_TARGET/1e6) / (F_TARGET/1e6) * 100:+.3f}% of target)")
    print(f"  FB         = {res.metrics[FFB_METRIC]:9.3f} MHz "
          f"(f_ref = {F_REF/1e6:.0f})")
    print(f"  VCTRL      = {v_lock:9.4f} V  "
          f"(behavioral said {V_LOCK_PRED:.4f}; twin error = "
          f"{(v_lock - V_LOCK_PRED)*1e3:+.1f} mV)")
    print(f"  drift      = {res.metrics[DRIFT_METRIC]:9.2f} mV between tail "
          f"halves (mean-drift check)")
    print(f"  REF->FB    = {phase_offset_ps(res.sim_results):+9.0f} ps median "
          f"edge offset (includes gate-path asymmetry; CP mismatch alone "
          f"predicts ~+10)")
    print(f"\n  saved "
          f"{save_plot(res.sim_results, os.path.join(OUT, 'pll_spice_lock.png'))}")
