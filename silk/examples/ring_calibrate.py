"""Calibrate the default ring oscillator's behavioral frequency model.

The behavioral model integrates phase. This script measures the default ring's
SPICE tuning curve and fits frequency and supply sensitivity within the default
linear supply range. The lowest fitted supply becomes the model's cutoff.
SPICE rows below the fitted range remain in the table, showing oscillation
outside the model's range. The cutoff is not a physical startup boundary.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.ring_calibrate`` from the
repository root with ngspice and sky130 models installed. It prints SPICE and
behavioral frequencies, calibration drift, replacement parameter values and the
runtime of a one-microsecond behavioral run. It saves ``ring_vco_tuning.png``
in the configured examples output directory.
"""
import os

import numpy as np

from silk.metrics import average_frequency
from silk import sources
from silk.circuit.modules import Module

from silk.examples.ring_oscillator_example import (
    build_ring_oscillator, attach_tuning_sweep, fit_tuning, pdk,
    TUNING_FIT_MIN_VDD,
)
from silk.examples import example_output_dir



def calibrate(mod, rows=None):
    """Fit the default ring's tuning curve and update its behavioral parameters.

    ``mod`` must be an unmodified default ``build_ring_oscillator()`` result.
    With no ``rows``, the sweep builds default rings at each supply. Supplied
    rows must describe that same design. This function does not characterize
    custom stage counts, device sizing or capacitance.

    ``fit_tuning`` applies the default linear supply range. All measured
    rows are returned, including those outside the fit. The fit spans several
    SPICE runs, so its parameters are written with ``calibrate_behavioral``.
    """
    rows = rows if rows is not None else attach_tuning_sweep()
    f0, kvco, vdd_min = fit_tuning(rows)
    mod.calibrate_behavioral("f0_mhz", f0)
    mod.calibrate_behavioral("kvco_mhz_per_v", kvco)
    mod.calibrate_behavioral("vdd_min", vdd_min)
    return rows, (f0, kvco, vdd_min)


def _behavioral_freq(mod, vdd, t_stop=200e-9, t_step=0.2e-9):
    """Return behavioral frequency in MHz from midpoint threshold crossings."""
    top = Module("PDK_skywater130", example_output_dir(), "ring_beh_top.cir", pdk_details=pdk)
    top.add_module(mod, "u_ring", nets={
        "VDD": "vdd", "VSS": "0", "EN": "en", "OUT": "out"})
    top.add_source(sources.VoltageSource("vdd", positive="vdd", negative="0", dc=vdd))
    top.add_source(sources.VoltageSource("en", positive="en", negative="0", dc=vdd))
    top.simulate_behavioral(t_step=t_step, t_stop=t_stop)

    return average_frequency(top.load_signal("out"), threshold=vdd / 2) / 1e6


if __name__ == "__main__":
    OUT = example_output_dir()

    mod = build_ring_oscillator()
    before = dict(mod._behavioral_params)

    print("=" * 72)
    print("Ring oscillator — behavioral VCO calibrated from SPICE")
    print("=" * 72)
    print("\n  measuring the tuning curve (one SPICE transient per supply)...")

    rows, (f0, kvco, vdd_min) = calibrate(mod)
    after = dict(mod._behavioral_params)

    print(f"\n  {'VDD (V)':>8} {'SPICE f_osc':>13} {'behavioral':>12} {'error':>9}")
    behavioral_freqs = {}
    for vdd, f_spice in rows:
        f_beh = _behavioral_freq(mod, vdd)
        behavioral_freqs[vdd] = f_beh
        if f_spice is None:
            ok = "OK" if not np.isfinite(f_beh) else "model oscillates without a detected SPICE oscillation"
            behavior = f"{f_beh:.1f}" if np.isfinite(f_beh) else f"no oscillation at {vdd:g} V"
            print(f"  {vdd:8.2f} {'not detected':>13} {behavior:>12}  {ok}")
        elif vdd < TUNING_FIT_MIN_VDD:
            print(f"  {vdd:8.2f} {f_spice:12.1f} {'not fitted':>12}  "
                  f"below the fitted range")
        elif not np.isfinite(f_beh):
            print(f"  no oscillation at {vdd:g} V (behavioral; SPICE {f_spice:.1f} MHz)")
        else:
            err = (f_beh - f_spice) / f_spice * 100
            print(f"  {vdd:8.2f} {f_spice:12.1f}  {f_beh:11.1f} {err:+8.1f}%")

    print(f"\n  {'param':<16} {'in-code':>12} {'measured':>12} {'drift':>8}")
    worst_drift = 0.0
    for k in ("f0_mhz", "kvco_mhz_per_v", "vdd_min"):
        b, a = before[k], after[k]
        d = (a - b) / abs(b) * 100.0
        worst_drift = max(worst_drift, abs(d))
        print(f"  {k:<16} {b:>12.4g} {a:>12.4g} {d:>+7.1f}%")

    print(f"\n  K_vco = {kvco:.0f} MHz/V: fitted supply sensitivity.")
    print(f"  vdd_min = {vdd_min:.2f} V: lowest fitted supply;")
    print("  this is the behavioral cutoff, not a measured startup boundary.")

    print("\n  paste-ready for ring_oscillator_example.py (update if drift warrants):")
    print(f"    _BEH_F0_NOM = {after['f0_mhz']:.6g}")
    print(f"    _BEH_KVCO_NOM = {after['kvco_mhz_per_v']:.6g}")
    print(f"    _BEH_VDDMIN_NOM = {after['vdd_min']:.4g}")
    if worst_drift > 2.0:
        print(f"\n  !! drift up to {worst_drift:.1f}% vs the in-code values -- the ring")
        print("     differs from the stored calibration. Paste the block above into")
        print("     ring_oscillator_example.py to refresh it.")
    else:
        print(f"\n  maximum drift {worst_drift:.1f}% from the stored calibration.")

    # Measure the runtime of a one-microsecond behavioral run.
    import time
    t0 = time.perf_counter()
    _behavioral_freq(mod, 1.8, t_stop=1e-6, t_step=0.2e-9)
    t_beh = time.perf_counter() - t0
    print(f"\n  1 us behavioral simulation: {t_beh*1e3:.0f} ms")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    live = [(v, f) for v, f in rows if f is not None]
    vs = np.array([v for v, _ in live]); fs = np.array([f for _, f in live])
    fb = np.array([behavioral_freqs[v] for v in vs])
    dead = [v for v, f in rows if f is None]

    fig, ax = plt.subplots(figsize=(8, 5))
    for v in dead:
        ax.axvspan(v - 0.06, v + 0.06, color="#D55E00", alpha=0.10, lw=0)
    if dead:
        ax.text(min(dead), max(fs) * 0.5, " no SPICE oscillation\n detected at tested supplies",
                fontsize=8.5, color="#D55E00")
    ax.plot(vs, fs, "o-", color="#0072B2", lw=2, ms=7, label="SPICE (transistor-level)")
    ax.plot(vs, fb, "s--", color="#D55E00", lw=1.6, ms=6,
            label="behavioral twin (calibrated)")
    ax.set_xlabel("supply / V_REG (V)")
    ax.set_ylabel("f_osc (MHz)")
    ax.set_title(f"Ring VCO tuning curve — K_vco = {kvco:.0f} MHz/V "
                 f"(behavioral cutoff {vdd_min:.1f} V)")
    ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    png = os.path.join(OUT, "ring_vco_tuning.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print(f"\n  saved {png}")
