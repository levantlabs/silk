"""Differential track-and-hold built from two bootstrapped samplers.

Two instances of make_abo_th share clocks and supplies and sample
complementary sine inputs. The sfdr bench measures the output difference
vhp-vhm near the end of track and hold. Subtraction suppresses matched
even-order distortion; mismatch and odd-order distortion remain.

Run from the repository root with MPLBACKEND=Agg:
    .venv/bin/python -m silk.examples.differential_th_example
It prints embedded tests, design specs, and characterization results, then
saves a hold-end spectrum and circuit reports under example_output_dir().
The SPICE run requires ngspice and sky130 models.
"""
import os

from silk import Port, PortDir, PortRole
from silk.circuit.modules import Module
from silk.sources.classes import VoltageSource
from silk.sources.waveforms import Sine, Pulse
from silk.simulation import Transient
from silk.metrics.spectral import sfdr_db, fft as metric_fft

# Reuse the single-ended bootstrapped T/H subckt and its shared timing/PDK setup.
from silk.examples.abo_th_example import (
                                          make_abo_th, VDD, CLK_PER, CLK_PW, CLK_TR, N_FFT, F_SIG,
                                          V_OFFSET, FS, T_STEP, T_STOP, TSTART_TRACK_END, TSTART_HOLD_END,
                                          )
from silk import load_pdk
from silk.examples import example_output_dir


# Complementary drive doubles the differential input swing.
DIFF_AMPL = 0.6
W_SW = 2.0          # sampling-switch width (um)
C_HOLD = 250e-15
SFDR_TARGET_DB = 60.0


def _attach_sfdr_test(mod, *, diff_ampl=DIFF_AMPL):
    """Register complementary sine drive and track-/hold-end SFDR specs."""
    tb = mod.create_test("sfdr")
    tb.add_module_under_test({
        "vinp": "vinp", "vinm": "vinm", "clk": "clk", "clkb": "clkb",
        "vdd": "vdd", "vss": "0", "vhp": "vhp", "vhm": "vhm",
    })
    tb.add_source(VoltageSource("vdd_src", positive="vdd", negative="0", dc=VDD))
    tb.add_source(VoltageSource(
        "vinp_src", positive="vinp", negative="0",
        sine=Sine(voffset=V_OFFSET, vampl=+diff_ampl, freq=F_SIG)))
    tb.add_source(VoltageSource(
        "vinm_src", positive="vinm", negative="0",
        sine=Sine(voffset=V_OFFSET, vampl=-diff_ampl, freq=F_SIG)))
    tb.add_source(VoltageSource(
        "clk_src", positive="clk", negative="0",
        pulse=Pulse(v1=0, v2=VDD, td=0, tr=CLK_TR, tf=CLK_TR, pw=CLK_PW, per=CLK_PER)))
    tb.add_source(VoltageSource(
        "clkb_src", positive="clkb", negative="0",
        pulse=Pulse(v1=VDD, v2=0, td=0, tr=CLK_TR, tf=CLK_TR, pw=CLK_PW, per=CLK_PER)))
    tb.add_sim_type(Transient(step=T_STEP, stop=T_STOP))
    # Subtract the held outputs on their shared time grid.
    vdiff = tb.probe("vhp") - tb.probe("vhm")
    tb.add_metric("sfdr_track_db", sfdr_db, sig=vdiff,
                  tclk=CLK_PER, tstart=TSTART_TRACK_END, n_samples=N_FFT)
    tb.add_metric("sfdr_hold_db", sfdr_db, sig=vdiff,
                  tclk=CLK_PER, tstart=TSTART_HOLD_END, n_samples=N_FFT)
    tb.add_spec("diff SFDR track-end >= 60 dB", "sfdr_track_db", ">=", SFDR_TARGET_DB)
    tb.add_spec("diff SFDR hold-end >= 60 dB", "sfdr_hold_db", ">=", SFDR_TARGET_DB)
    return tb


def build_differential_th(out_dir=None, *, w_sw=W_SW, c_hold=C_HOLD,
                          diff_ampl=DIFF_AMPL) -> Module:
    """Build two placements of one sampler definition and register the SFDR test.

    w_sw and c_hold size both branches. diff_ampl sets each input's peak
    sine amplitude about V_OFFSET. The branches share clock and supply.
    """
    out_dir = out_dir or example_output_dir()
    pdk = load_pdk('sky130')
    # one .subckt definition, instantiated twice; with_tests=False since the
    # differential module owns its own (SFDR) characterization.
    th = make_abo_th(pdk, out_dir, w_sw=w_sw, c_hold=c_hold, with_tests=False)

    diff = Module("PDK_skywater130", out_dir, "abo_th_diff.cir",
                  pdk_details=pdk, module_name="abo_th_diff")
    diff.define_module_ports({
        "vinp": Port.analog_input(), "vinm": Port.analog_input(),
        "clk": (PortDir.IN, PortRole.CLOCK), "clkb": (PortDir.IN, PortRole.CLOCK),
        "vdd": Port.supply(), "vss": Port.ground(),
        "vhp": Port.analog_output(), "vhm": Port.analog_output(),
    })
    diff.add_module(th, "thp", nets={
        "vin": "vinp", "clk": "clk", "clkb": "clkb",
        "vdd": "vdd", "vss": "vss", "vh": "vhp"})
    diff.add_module(th, "thm", nets={
        "vin": "vinm", "clk": "clk", "clkb": "clkb",
        "vdd": "vdd", "vss": "vss", "vh": "vhm"})
    _attach_sfdr_test(diff, diff_ampl=diff_ampl)
    return diff


def run_characterization(mod, corner="tt"):
    """Run the registered test and return results and spec reconciliation."""
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def plot_diff_spectrum(report, png_path):
    """Save the hold-end differential spectrum relative to its strongest non-DC bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ckt = report.sim_results
    diff = ckt.load_signal("vhp") - ckt.load_signal("vhm")
    spec = metric_fft(diff, tclk=CLK_PER, tstart=TSTART_HOLD_END, n_samples=N_FFT)
    P = np.asarray(spec.y)
    db = 10.0 * np.log10(np.maximum(P / np.max(P[1:]), 1e-14))
    freqs = np.asarray(spec.x) / 1e6
    sfdr_hold = report.metrics["sfdr_hold_db"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stem(freqs, db, linefmt="teal", markerfmt=" ", basefmt=" ")
    ax.set_xlim(0, FS / 2 / 1e6)
    ax.set_ylim(-120, 5)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power (dBc)")
    ax.set_title(f"Differential T/H — hold-end spectrum (SFDR {sfdr_hold:.1f} dB, "
                 f"relative to strongest non-DC bin)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    return fig


mod = build_differential_th()   # module-level export — instances, test, specs


if __name__ == "__main__":
    m = mod
    print(f"tests embedded : {m.list_tests()}")
    print(f"design specs   : {[s.name for s in m._specs]}")
    print(f"sub-instances  : composes two 'abo_th' subckts (thp, thm)\n")

    results, recon = run_characterization(m)
    print(recon.summary())
    out_dir = example_output_dir()
    png_path = os.path.join(out_dir, "differential_th_spectrum.png")
    fig = plot_diff_spectrum(results["abo_th_diff.sfdr"], png_path)
    print(f"\nDifferential spectrum plot saved -> {png_path}")

    from silk.circuit.report import generate_report
    report = generate_report(m)
    report.add_figure(fig, title="Differential T/H — hold-end spectrum")
    report.save(os.path.join(out_dir, "differential_th_report.md"))
    report.save_json(os.path.join(out_dir, "differential_th_report.json"))
    report.save_html(os.path.join(out_dir, "differential_th_report.html"))
    print(f"Circuit report saved -> {out_dir}/differential_th_report.{{md,json,html}}")

    import matplotlib.pyplot as plt
    plt.close(fig)
