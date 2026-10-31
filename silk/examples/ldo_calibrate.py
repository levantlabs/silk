"""Calibrate the LDO behavioral twin from SPICE and compare step responses.

The _LDOModel twin has six calibrated parameters. DC tracking supplies gain
and vreg_max; vstep_large supplies slew_v_per_us; vstep_small supplies zeta
and bw_hz; the AC supply test supplies psrr_db. CalibrationBinding writes
these spec measurements to model parameters during reconcile_specs().
vdd_nom and initial_output are not fitted.

Extract damping and natural frequency from the same small step. Broken-loop
unity-gain crossover is a separate stability measurement. vreg_max clips the
target at the characterized output range; transient overshoot and an explicit
initial_output can exceed it. The model does not describe dropout outside
that characterized range.

Behavioral models reduce the cost of long PLL lock transients. This script
runs SPICE to update the twin, prints parameter drift and replacement
constants, then compares a small-step response on both backends. It prints
RMS and maximum error and saves an overlay and error plot.

Run from the repository root with MPLBACKEND=Agg:
    .venv/bin/python -m silk.examples.ldo_calibrate
"""
import os
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from silk.specs import CalibrationBinding
from silk.metrics import error_norm, sample
from silk.signals.base import Signal
from silk.simulation import Transient
from silk import sources

from silk.examples.ldo_regulator_example import (
                                                 build_ldo, run_characterization, attach_loop_test,
                                                 BEH_GAIN_METRIC, BEH_VREGMAX_METRIC, PSRR_METRIC, FC_METRIC, SLEW_METRIC,
                                                 ZETA_METRIC, FN_METRIC,
                                                 _VDD, _I_LOAD, _add_rig,
                                                 )
from silk.examples import example_output_dir


# Which SPICE metric feeds which behavioral parameter.
_BINDINGS = {
    BEH_GAIN_METRIC: "gain",
    BEH_VREGMAX_METRIC: "vreg_max",
    PSRR_METRIC: "psrr_db",
    SLEW_METRIC: "slew_v_per_us",
    ZETA_METRIC: "zeta",
    FN_METRIC: "bw_hz",
}


def calibrate(mod):
    """Bind the six SPICE spec measurements to twin parameters and characterize.

    reconcile_specs() applies each CalibrationBinding to its target parameter.
    """
    for spec in mod._specs:
        if spec.metric in _BINDINGS:
            spec.calibration = CalibrationBinding(
                target_module=mod, param_name=_BINDINGS[spec.metric])
    return run_characterization(mod)


def _step_test(mod, v_lo, v_hi, t_step, t_stop, t_edge):
    """One registered VCTRL step rig for SPICE and the behavioral twin.

    This bench declares no metrics: it supplies the waveforms for the
    comparison plot. ``load_signal('vreg')`` reads each backend's result;
    the SPICE arrays are copied before the behavioral run replaces it.
    """
    tb = mod.create_test("beh_check")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "VREG": "vreg", "IBIAS_AMP": "ibias_amp"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    tb.add_source(sources.VoltageSource(
        "vctrl", positive="vctrl", negative="0",
        pulse=sources.Pulse(v1=v_lo, v2=v_hi, td=t_edge, tr=1e-9, tf=1e-9,
                            pw=t_stop, per=2 * t_stop)))
    tb.add_source(sources.CurrentSource("iload", positive="vreg", negative="0",
                                        dc=_I_LOAD))
    _add_rig(tb)
    tb.add_sim_type(Transient(step=t_step, stop=t_stop))
    return tb


def _step_responses(mod, v_lo, v_hi, t_step, t_stop, t_edge):
    """Run the same registered rig on both backends, retaining each waveform."""
    tb = _step_test(mod, v_lo, v_hi, t_step, t_stop, t_edge)
    tb.netlist.simulator = "ngspice"
    tb.run()
    spice = tb.netlist.load_signal("vreg")
    ts, vs = np.array(spice.x), np.array(spice.y)
    tb.netlist.simulator = "behavioral"
    # Preserve the comparison script's output: the model run has no metrics.
    with redirect_stdout(StringIO()):
        tb.run()
    behavioral = tb.netlist.load_signal("vreg")
    return (ts, vs), (np.array(behavioral.x), np.array(behavioral.y))


if __name__ == "__main__":
    OUT = example_output_dir()

    mod = build_ldo()
    before = dict(mod._behavioral_params)

    print("=" * 70)
    print("LDO behavioral model — calibrated from SPICE")
    print("=" * 70)

    results, recon = calibrate(mod)

    # Measure broken-loop crossover separately from the step-based calibration.
    lp = build_ldo(with_tests=False, break_loop=True)
    attach_loop_test(lp)
    lres, _ = run_characterization(lp)
    fc = lres[f"{lp.module_name}.loop"].metrics[FC_METRIC]

    after = dict(mod._behavioral_params)
    print(f"\n  {'param':<14} {'in-code':>12} {'measured':>12} {'drift':>8}")
    worst_drift = 0.0
    for k in ("gain", "bw_hz", "zeta", "psrr_db", "vreg_max", "slew_v_per_us"):
        b, a = before[k], after[k]
        d = (a - b) / abs(b) * 100.0
        worst_drift = max(worst_drift, abs(d))
        print(f"  {k:<14} {b:>12.4g} {a:>12.4g} {d:>+7.1f}%")
    print(f"\n  bw_hz is the closed-loop natural frequency "
          f"({after['bw_hz']/1e6:.2f} MHz), not the loop's AC")
    print(f"  crossover ({fc/1e6:.2f} MHz) -- different quantities. "
          f"damping zeta = {after['zeta']:.2f}.")

    print("\n  paste-ready for ldo_regulator_example.py (update if drift warrants):")
    print(f"    _BEH_GAIN_NOM = {after['gain']:.6g}")
    print(f"    _BEH_BW_NOM = {after['bw_hz']:.4g}")
    print(f"    _BEH_PSRR_NOM = {after['psrr_db']:.4g}")
    print(f"    _BEH_VREGMAX_NOM = {after['vreg_max']:.5g}")
    print(f"    _BEH_SLEW_NOM = {after['slew_v_per_us']:.4g}")
    print(f"    _BEH_ZETA_NOM = {after['zeta']:.4g}")
    if worst_drift > 2.0:
        print(f"\n  !! drift up to {worst_drift:.1f}% vs the in-code values -- the design")
        print("     has moved since the last calibration. Paste the block above into")
        print("     ldo_regulator_example.py to refresh it.")
    else:
        print(f"\n  drift <= {worst_drift:.1f}%: the in-code calibration is current.")

    # Compare the calibrated twin and SPICE on the same small step.
    V_LO, V_HI = 1.2, 1.3      # small-signal comparison
    T_STEP, T_STOP, T_EDGE = 20e-9, 20e-6, 5e-6
    (ts, vs), (tb_, vb) = _step_responses(
        mod, V_LO, V_HI, T_STEP, T_STOP, T_EDGE)

    spice = Signal("vreg", "voltage", "time", "V", ts, vs)
    behavioral = Signal("vreg", "voltage", "time", "V", tb_, vb)
    vb_on_ts = sample(behavioral, x_new=ts).y
    err = vb_on_ts - vs
    rms = error_norm(spice, behavioral, kind="rms")
    worst = error_norm(spice, behavioral, kind="max")
    print(f"\n  behavioral vs SPICE, {V_LO} -> {V_HI} V step:")
    print(f"    RMS error   = {rms*1e3:.2f} mV")
    print(f"    worst error = {worst*1e3:.2f} mV")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(ts * 1e6, vs, color="#0072B2", lw=2, label="SPICE (transistor-level)")
    a1.plot(tb_ * 1e6, vb, "--", color="#D55E00", lw=1.8,
            label="behavioral (calibrated from SPICE)")
    a1.set_ylabel("VREG (V)")
    a1.set_title(f"LDO behavioral twin vs SPICE — {V_LO} to {V_HI} V step "
                 f"(RMS error {rms*1e3:.2f} mV)")
    a1.legend(frameon=False, fontsize=9)
    a2.plot(ts * 1e6, err * 1e3, color="#5a5a6e", lw=1.2)
    a2.axhline(0, color="#d5d5dd", lw=1)
    a2.set_ylabel("error (mV)")
    a2.set_xlabel("time (µs)")
    for a in (a1, a2):
        a.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    png = os.path.join(OUT, "ldo_behavioral_vs_spice.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"\n  saved {png}")
