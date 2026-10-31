"""Five-stage CMOS ring oscillator with a NAND enable and an output buffer.

The ring contains one two-input NAND and four inverters, with a capacitor on
each stage. A separate inverter buffers the output. With EN low, the NAND holds
its output high and the ring settles to fixed logic levels. EN rises after 1 ns;
the NAND then acts as an inverter and closes the oscillating loop. The SPICE
testbenches start from the operating point without .ic or uic directives.

The default ``osc`` test measures frequency, duty cycle and period stability
after 5 ns. Its supply is constant and its transient contains no device noise.
The measured period variation describes settling and numerical effects; it is
not a measurement of physical oscillator jitter.

``attach_supply_noise_test`` adds the ``jitter`` test, which applies random noise
to the supply and measures frequency, RMS period jitter and RMS cycle-to-cycle
jitter after 5 ns. Noise amplitude and update interval are explicit parameters;
the jitter estimate depends on both and varies between runs. This measures
supply-induced timing variation, excluding intrinsic transistor noise.

Run with ``MPLBACKEND=Agg .venv/bin/python -m
silk.examples.ring_oscillator_example`` from the repository root. The program
characterizes both tests, saves a startup and waveform plot,
and repeats ``osc`` across the PDK's process corners. It prints each corner's
frequency, duty cycle and spec result, followed by the frequency spread.

``attach_tuning_sweep`` separately runs a transient at each requested supply
and returns (supply, frequency in MHz) rows, with None for an unmeasurable
frequency. ``fit_tuning`` fits measurable rows within the default linear
supply range for the supply-controlled behavioral model. The tuning sweep is
not run by this file's main program.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk import analysis
from silk.simulation import Transient
from silk.metrics import (period_jitter, cycle_to_cycle_jitter,
                          average_duty, average_frequency)
from silk.examples import example_output_dir

pdk = load_pdk('sky130')

# --- design knobs --------------------------------------------------------------
STAGES = 5             # odd inversion count for oscillation when enabled
_WN, _WP = 1.0, 2.0    # inverter NMOS/PMOS widths (µm)
_L = 0.15              # channel length (µm)
_C_STAGE = 20e-15      # added load on each stage and OUT (F)
_VDD = 1.8

# Metric keys referenced by testbench specs.
F_METRIC, DUTY_METRIC, JIT_METRIC = "f_osc_mhz", "duty_pct", "period_jit_ps"
C2C_METRIC = "c2c_jit_ps"
F_LO_MHZ, F_HI_MHZ = 100.0, 2000.0   # acceptance band for f_osc
DUTY_LO, DUTY_HI = 40.0, 60.0        # symmetric ring -> near 50%
JIT_MAX_PS = 1.0                     # noiseless period stability
_VN_RMS = 20e-3                      # supply noise amplitude (20 mV rms)
_VN_TAU = 10e-12                     # supply-noise update interval (s)
_TUNE_VDDS = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8)   # tuning sweep supplies
TUNING_FIT_MIN_VDD = 1.0             # lower supply bound of the linear fit (V)

# Default behavioral parameters for the default ring geometry, fitted from
# a tt tuning sweep within the linear supply range. Re-run ring_calibrate.py
# after changing stage count, transistor sizes or stage capacitance; these
# defaults do not automatically track circuit changes.
_BEH_F0_NOM = -952.8     # MHz     (intercept of the f vs VDD fit)
_BEH_KVCO_NOM = 1105.0   # MHz/V   (slope: the ring's supply sensitivity)
_BEH_VDDMIN_NOM = 1.0    # V       (lowest fitted tuning supply)

JIT_NOISE_MAX_PS = 20.0              # supply-induced RMS period-jitter limit (ps)


def build_inverter(*, wn=_WN, wp=_WP, l=_L) -> Module:
    """Build a CMOS inverter with supply, input and output ports."""
    m = Module("PDK_skywater130", example_output_dir(), "inv.cir", pdk_details=pdk,
               module_name="inv")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "Y": Port.analog_output(),
    })
    # The context registers each constructed device on this module.
    with m:
        pdk.pmos1p8("MP", d="Y", g="A", s="VDD", b="VDD", w=wp, l=l)
        pdk.nmos1p8("MN", d="Y", g="A", s="VSS", b="VSS", w=wn, l=l)
    return m


def build_nand2(*, wn=_WN, wp=_WP, l=_L) -> Module:
    """Build the two-input CMOS NAND used as the enable stage.

    PMOS devices connect in parallel and NMOS devices connect in series.
    Each NMOS has twice the inverter NMOS width to reduce the added series
    resistance. With B wired to EN, EN low holds Y high; EN high makes Y
    the inverse of A."""
    m = Module("PDK_skywater130", example_output_dir(), "nand2.cir", pdk_details=pdk,
               module_name="nand2")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "A": Port.analog_input(), "B": Port.analog_input(),
        "Y": Port.analog_output(),
    })
    with m:
        # PMOS in parallel: either low input pulls Y high.
        pdk.pmos1p8("MPA", d="Y", g="A", s="VDD", b="VDD", w=wp, l=l)
        pdk.pmos1p8("MPB", d="Y", g="B", s="VDD", b="VDD", w=wp, l=l)
        # NMOS in series: Y -> nx -> VSS, so both inputs must be high to pull down.
        # Doubled width offsets the stacked resistance (see docstring).
        pdk.nmos1p8("MNA", d="Y", g="A", s="nx", b="VSS", w=2 * wn, l=l)
        pdk.nmos1p8("MNB", d="nx", g="B", s="VSS", b="VSS", w=2 * wn, l=l)
    return m


def build_ring_oscillator(*, stages=STAGES, wn=_WN, wp=_WP, l=_L,
                          c_stage=_C_STAGE, with_tests=True) -> Module:
    """Build a NAND and an even number of inverters in a closed ring.

    The total stage count must be odd. Stage 0 takes the last stage's output
    and EN as its inputs. Each later stage inverts the preceding stage.
    A separate inverter buffers the last ring node to OUT. Each ring node
    and OUT has a capacitor of ``c_stage`` farads to VSS.

    ``wn``, ``wp`` and ``l`` set transistor dimensions in microns.
    The returned module includes a behavioral model and, when ``with_tests``
    is true, the ``osc`` testbench."""
    if stages % 2 == 0:
        raise ValueError(f"ring needs an ODD number of stages (an even ring is a "
                         f"latch and has a stable DC state); got {stages}")

    from silk.elements import elements

    mod = Module("PDK_skywater130", example_output_dir(), "ring_oscillator.cir",
                 pdk_details=pdk, module_name="ring_oscillator")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "EN": Port.analog_input(), "OUT": Port.analog_output(),
    })

    inv = build_inverter(wn=wn, wp=wp, l=l)
    nand = build_nand2(wn=wn, wp=wp, l=l)

    # Stage 0: the NAND. A = the ring's last node, B = EN.
    mod.add_module(nand, "stage0", nets={
        "VDD": "VDD", "VSS": "VSS",
        "A": f"n{stages - 1}", "B": "EN", "Y": "n0"})
    # Stages 1..N-1: plain inverters.
    for i in range(1, stages):
        mod.add_module(inv, f"stage{i}", nets={
            "VDD": "VDD", "VSS": "VSS", "A": f"n{i - 1}", "Y": f"n{i}"})

    # The buffer isolates the OUT load from the last ring node.
    mod.add_module(inv, "buf", nets={
        "VDD": "VDD", "VSS": "VSS", "A": f"n{stages - 1}", "Y": "OUT"})

    with mod:
        for i in range(stages):
            elements.capacitor(f"C{i}", p=f"n{i}", n="VSS", value=c_stage)
        elements.capacitor("COUT", p="OUT", n="VSS", value=c_stage)

    # Recalibrate these defaults after changing the circuit geometry.
    mod.set_behavioral_model(_RingModel(), params={
        "f0_mhz": _BEH_F0_NOM, "kvco_mhz_per_v": _BEH_KVCO_NOM,
        "vdd_min": _BEH_VDDMIN_NOM,
    }, execution='sample')

    if with_tests:
        attach_osc_test(mod)
    return mod


# --- testbench -----------------------------------------------------------------

_T_STEP = 1e-12     # must resolve ~100 ps edges on a GHz-class ring
_T_EN = 1e-9        # EN rises here — the ring is parked before this
_T_SETTLE = 5e-9    # ignore everything before this: oscillation is still growing
_T_STOP = 55e-9     # ~50 ns of steady oscillation to average over
_T_JITTER_STOP = 490e-9  # longer record for the random jitter estimate


def attach_osc_test(mod, *, t_stop=_T_STOP, vdd=_VDD):
    """Register a transient test for frequency, duty and period stability.

    EN rises from zero to the bench supply after ``_T_EN``. Measurements
    begin at ``_T_SETTLE`` and use half the bench supply as the threshold.
    The supply is constant, and the test uses neither .ic nor uic."""
    tb = mod.create_test("osc")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "EN": "en", "OUT": "out"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=vdd))
    tb.add_source(sources.VoltageSource(
        "en", positive="en", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=vdd, td=_T_EN, tr=100e-12, tf=100e-12,
                            pw=t_stop, per=2 * t_stop)))
    # Step must resolve the edges: a GHz-class ring has ~100 ps transitions.
    tb.add_sim_type(Transient(step=_T_STEP, stop=t_stop, maxstep=2 * _T_STEP))

    frequency = tb.add_metric("frequency_hz", average_frequency,
                              sig=tb.probe("out"), threshold=vdd / 2,
                              start=_T_SETTLE)
    tb.add_metric(F_METRIC, frequency / 1e6)
    duty = tb.add_metric("duty", average_duty, sig=tb.probe("out"),
                         threshold=vdd / 2, start=_T_SETTLE)
    tb.add_metric(DUTY_METRIC, duty * 100)
    jitter = tb.add_metric("period_jitter_s", period_jitter,
                           clock=tb.probe("out"), threshold=vdd / 2,
                           start=_T_SETTLE)
    tb.add_metric(JIT_METRIC, jitter * 1e12)
    tb.add_spec(f"f_osc > {F_LO_MHZ:.0f} MHz", F_METRIC, ">", F_LO_MHZ)
    tb.add_spec(f"f_osc < {F_HI_MHZ:.0f} MHz", F_METRIC, "<", F_HI_MHZ)
    tb.add_spec(f"duty > {DUTY_LO:.0f}%", DUTY_METRIC, ">", DUTY_LO)
    tb.add_spec(f"duty < {DUTY_HI:.0f}%", DUTY_METRIC, "<", DUTY_HI)
    tb.add_spec(f"period stability < {JIT_MAX_PS:.1f} ps", JIT_METRIC, "<", JIT_MAX_PS)
    return tb


def attach_supply_noise_test(mod, *, vdd=_VDD, noise_rms=_VN_RMS,
                             noise_tau=_VN_TAU, t_stop=_T_JITTER_STOP):
    """Register a transient test for supply-induced jitter.

    The supply source adds white noise with RMS amplitude ``noise_rms``
    and update interval ``noise_tau``. Both parameters affect the measured
    jitter. This source models supply noise, excluding transistor noise.

    After ``_T_SETTLE``, the test measures frequency, RMS period jitter and
    RMS cycle-to-cycle jitter at half the bench supply. The default record
    ends at 490 ns. Each run draws fresh noise, so estimates vary between
    runs. The test checks period jitter against ``JIT_NOISE_MAX_PS``."""
    tb = mod.create_test("jitter")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "EN": "en", "OUT": "out"})
    tb.add_source(sources.VoltageSource(
        "vdd", positive="VDD", negative="0", dc=vdd,
        trnoise=sources.TransientNoise(na=noise_rms, nt=noise_tau)))
    tb.add_source(sources.VoltageSource(
        "en", positive="en", negative="0", dc=0,
        pulse=sources.Pulse(v1=0, v2=vdd, td=_T_EN, tr=100e-12, tf=100e-12,
                            pw=t_stop, per=2 * t_stop)))
    tb.add_sim_type(Transient(step=_T_STEP, stop=t_stop, maxstep=2 * _T_STEP))
    # The longer jitter record needs only OUT for all three metrics.
    tb.netlist.set_save_list(["out"])

    jitter = tb.add_metric("period_jitter_s", period_jitter,
                           clock=tb.probe("out"), threshold=vdd / 2,
                           start=_T_SETTLE)
    tb.add_metric(JIT_METRIC, jitter * 1e12)
    c2c = tb.add_metric("c2c_jitter_s", cycle_to_cycle_jitter,
                        clock=tb.probe("out"), threshold=vdd / 2,
                        start=_T_SETTLE)
    tb.add_metric(C2C_METRIC, c2c * 1e12)
    frequency = tb.add_metric("frequency_hz", average_frequency,
                              sig=tb.probe("out"), threshold=vdd / 2,
                              start=_T_SETTLE)
    tb.add_metric(F_METRIC, frequency / 1e6)
    tb.add_spec(f"supply-induced jitter < {JIT_NOISE_MAX_PS:.0f} ps "
                f"({noise_rms*1e3:.0f} mV rms, {1/(2*noise_tau)/1e9:.0f} GHz)",
                JIT_METRIC, "<", JIT_NOISE_MAX_PS)
    return tb


def run_corner_sweep(mod, corners=None):
    """Run ``osc`` at the requested process corners.

    With no corner list, use the PDK manifest's process corners. Return
    ``CornerResults``, mapping each corner to its ``SpecReport``; ``passed``
    is true only when every spec passes at every corner."""
    tb = mod.get_test("osc")
    return tb.run_corners(corners)


# --- behavioral model ---------------------------------------------------------

class _RingModel:
    """Model a supply-controlled oscillator by integrating phase.

    Frequency is ``(f0_mhz + kvco_mhz_per_v * VDD) * 1e6`` Hz, clamped
    to zero below ``vdd_min`` or when the expression is negative. The
    frequency coefficients are fitted by ``ring_calibrate.py``.

    Phase starts at zero and integrates frequency over time. With EN above
    half the supply, OUT is VDD during the positive half of the phase's
    sine and zero otherwise. EN gates OUT without resetting phase.
    A DC analysis returns zero."""

    def __call__(self, ports, t, params):

        if analysis() == "dc":
            ports.OUT.v = 0.0
            return
        vdd = float(ports.VDD.v)
        en = float(ports.EN.v)
        f = max(0.0, (float(params.f0_mhz)
                      + float(params.kvco_mhz_per_v) * vdd) * 1e6)
        if vdd < float(params.vdd_min):
            f = 0.0
        if t == 0:
            self.phase = 0.0
        else:
            self.phase += 2.0 * np.pi * 0.5 * (f + self.frequency) * (t - self.time)
        self.frequency = f
        self.time = float(t)
        enabled = en > max(vdd, 1e-9) / 2.0
        ports.OUT.v = vdd if enabled and np.sin(self.phase) > 0 else 0.0


def attach_tuning_sweep(*, vdds=_TUNE_VDDS, stages=STAGES):
    """Return (supply in V, frequency in MHz or None) rows from transients.

    Build a fresh ring with ``stages`` stages for each supply. A non-finite
    frequency means the record has no measurable period at half that supply.
    Simulation and configuration errors propagate to the caller.
    """
    out = []
    for vdd in vdds:
        m = build_ring_oscillator(stages=stages, with_tests=False)
        tb = attach_osc_test(m, vdd=vdd)
        r = tb.run()
        frequency = float(r.metrics[F_METRIC])
        if not np.isfinite(frequency):
            print(f"no measurable frequency at {vdd:g} V")
            frequency = None
        out.append((vdd, frequency))
    return out


def fit_tuning(rows):
    """Return the fitted frequency intercept, supply slope and minimum supply.

    Use measured rows from ``TUNING_FIT_MIN_VDD`` through the highest default
    sweep supply. The minimum supply is the lowest fitted supply, rather
    than an interpolated startup threshold. At least two fitted rows are
    required. Rows outside the fit range remain available to the caller."""
    fit_max = max(_TUNE_VDDS)
    live = [(v, f) for v, f in rows
            if f is not None and TUNING_FIT_MIN_VDD <= v <= fit_max]
    if len(live) < 2:
        raise RuntimeError(
            f"tuning: fewer than two measured supplies in the fitted range "
            f"{TUNING_FIT_MIN_VDD:g}..{fit_max:g} V")
    v = np.array([x for x, _ in live], dtype=float)
    f = np.array([y for _, y in live], dtype=float)
    kvco, f0 = np.polyfit(v, f, 1)
    return float(f0), float(kvco), float(v.min())


def run_characterization(mod, corner="tt"):
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def save_plot(ckt, path, *, f_mhz=None, duty=None):
    """Save EN, the full output waveform and a zoom of resolved cycles.

    The zoom shows individual edges that overlap at the full record's scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = ckt.load_signal("out")
    en = ckt.load_signal("en")
    t = np.asarray(out.x, dtype=float)
    y = np.asarray(out.y, dtype=float)
    tns = t * 1e9

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7.4),
                                        gridspec_kw={"height_ratios": [0.7, 1.4, 1.4]})
    ax1.plot(tns, np.asarray(en.y, dtype=float), color="#009E73", lw=1.5)
    ax1.set_ylabel("EN (V)")
    ax1.set_xlabel("time (ns)")
    title = "5-stage ring oscillator — parked by EN=0, starts on the EN edge (no .ic)"
    if f_mhz:
        title += f"\nf_osc = {f_mhz:.0f} MHz, duty = {duty:.1f}%"
    ax1.set_title(title, fontsize=11)

    ax2.plot(tns, y, color="#0072B2", lw=0.7)
    ax2.axvline(_T_EN * 1e9, color="#009E73", ls=":", lw=1.2)
    ax2.axvspan(tns[0], _T_SETTLE * 1e9, color="#D55E00", alpha=0.07, lw=0)
    ax2.text(_T_SETTLE * 1e9, 1.9, " measurement window ->", fontsize=8,
             color="#5a5a6e", va="top")
    ax2.set_ylabel("OUT (V)")
    ax2.set_xlabel("time (ns)  —  full run (cycles alias at this scale)")

    zlo, zhi = 20e-9, 25e-9
    zm = (t >= zlo) & (t <= zhi)
    ax3.plot(t[zm] * 1e9, y[zm], color="#0072B2", lw=1.6)
    ax3.axhline(_VDD / 2, color="#5a5a6e", ls="--", lw=1, label="VDD/2 threshold")
    ax3.set_ylabel("OUT (V)")
    ax3.set_xlabel("time (ns)  —  zoom: cycles resolved")
    ax3.legend(frameon=False, fontsize=8)

    for ax in (ax1, ax2, ax3):
        ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    mod = build_ring_oscillator()
    attach_supply_noise_test(mod)
    print("=" * 70)
    print(f"{mod.module_name}: {STAGES} stages, NAND enable, C_stage = "
          f"{_C_STAGE*1e15:.0f} fF")
    print("=" * 70)

    results, recon = run_characterization(mod)
    print(recon.summary())
    r = results[f"{mod.module_name}.osc"]
    print(f"  f_osc  = {r.metrics[F_METRIC]:.1f} MHz")
    print(f"  duty   = {r.metrics[DUTY_METRIC]:.1f} %")
    print(f"  period stability = {r.metrics[JIT_METRIC]:.3f} ps  "
          f"(noiseless -> numerical, not physical)")
    print(f"  saved {save_plot(r.sim_results, os.path.join(OUT, 'ring_oscillator.png'), f_mhz=r.metrics[F_METRIC], duty=r.metrics[DUTY_METRIC])}")

    j = results[f"{mod.module_name}.jitter"]
    print(f"\n  supply-induced jitter ({_VN_RMS*1e3:.0f} mV rms, "
          f"{1/(2*_VN_TAU)/1e9:.0f} GHz):")
    print(f"    f_osc          = {j.metrics[F_METRIC]:.1f} MHz")
    print(f"    period jitter  = {j.metrics[JIT_METRIC]:.1f} ps")
    print(f"    cycle-to-cycle = {j.metrics[C2C_METRIC]:.1f} ps  "
          f"(RMS difference between consecutive periods)")
    print("    Supply noise is stochastic; each run draws a fresh realization.")
    print("    Intrinsic transistor noise is excluded.")

    # --- process corners -------------------------------------------------------
    print("\n" + "=" * 70)
    print("Process corners (osc testbench)")
    print("=" * 70)
    cr = run_corner_sweep(mod)
    print(f"  {'corner':>7} {'f_osc (MHz)':>12} {'duty (%)':>10}   specs")
    fs = []
    for corner, rep in cr.items():
        f = rep.metrics[F_METRIC]
        fs.append(f)
        n_fail = sum(not result.passed for result in rep.results)
        print(f"  {corner:>7} {f:12.1f} {rep.metrics[DUTY_METRIC]:10.1f}   "
              f"{'PASS' if rep.passed else f'FAIL ({n_fail})'}")
    lo, hi = min(fs), max(fs)
    print(f"\n  f_osc spread: {lo:.0f} .. {hi:.0f} MHz  "
          f"({(hi-lo)/np.mean(fs)*100:.0f}% of mean)")
    print("  Frequency varies with process corner.")
    print(f"\n  all corners pass: {cr.passed}")
