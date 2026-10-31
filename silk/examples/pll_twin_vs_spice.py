"""Compare near-lock release on the behavioral and transistor PLLs.

SPICE precharges the filter and releases it at 300 ns. The behavioral loop
starts from the precharge voltage at time zero; its trace is shifted by
300 ns for the overlay. Initial reference and divider phases can differ,
so the release excursions need not match sample by sample.

Each run prints tail-voltage and frequency differences. The voltage means
use different windows: SPICE uses 2.50–3.00 us and the shifted behavioral
trace uses 2.46–3.00 us. SPICE frequency uses 2.20–3.00 us; behavioral
frequency uses 2.46–3.00 us. These averages do not prove final settling.
The plot shows both control voltages and frequency averaged over 24 cycles.

Run from the repository root with ngspice and sky130 models installed:
    MPLBACKEND=Agg .venv/bin/python -m silk.examples.pll_twin_vs_spice
"""
import os

import numpy as np

from silk.metrics import average, average_frequency, crossings
from silk.signals.spice import NgspiceSignal
from silk.examples.pll_spice_example import build_pll_spice, V_LOCK_PRED, _T_RELEASE, _T_STOP, _T_TAIL
from silk.examples.pll_example import build_pll, run_lock, F_TARGET
from silk.examples import example_output_dir


def _f_inst(t_edges, w=24):
    """Frequency averaged over nonoverlapping w-cycle windows, in MHz."""
    tt, ff = [], []
    for k in range(0, len(t_edges) - w, w):
        span = t_edges[k + w] - t_edges[k]
        tt.append(t_edges[k + w // 2] * 1e6)
        ff.append(w / span / 1e6)
    return np.asarray(tt), np.asarray(ff)


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 72)
    print("Twin vs transistors — the same near-lock release, overlaid")
    print("=" * 72)

    print("\n  transistor loop (ngspice)...")
    smod = build_pll_spice()
    sres = smod.run_all_tests(corner="tt")["pll_spice.lock"]
    ckt = sres.sim_results
    sv = ckt.load_signal("vctrl")
    st = np.asarray(sv.x, dtype=float)
    svc = np.asarray(sv.y, dtype=float)
    so = ckt.load_signal("out")
    s_edges = crossings(so, threshold=0.7, edge="rising")

    print("  behavioral twin (same release, v_init = precharge target)...")
    bmod = build_pll(v_init=V_LOCK_PRED)
    bres = run_lock(bmod, t_stop=_T_STOP - _T_RELEASE, t_step=0.2e-9)
    bt = bres["t"] + _T_RELEASE
    bvc = bres["vctrl"]
    b_th = 0.5 * float(bres["out"].max())
    b_out = NgspiceSignal(name="out", type="voltage", xlabel="time", ylabel="V",
                          xvalues=bres["t"], yvalues=bres["out"])
    b_edges = crossings(b_out, threshold=b_th, edge="rising") + _T_RELEASE

    s_lock = average(sv, start=_T_STOP - 0.5e-6)
    b_lock = bres["v_ctrl_locked"]
    print("\n  tail means (SPICE 2.50-3.00 us; twin 2.46-3.00 us): "
          f"SPICE {s_lock:.4f} V   twin {b_lock:.4f} V   "
          f"delta {(s_lock - b_lock)*1e3:+.1f} mV")
    print(f"  f tail : SPICE {average_frequency(so, threshold=0.7, start=_T_TAIL) / 1e6:.2f} MHz   "
          f"twin {bres['f_locked']/1e6:.2f} MHz   (SPICE 2.20-3.00 us; twin 2.46-3.00 us)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax[0].plot(st * 1e6, svc, color="#0072B2", lw=1.2, label="SPICE (transistors)")
    ax[0].plot(bt * 1e6, bvc, color="#D55E00", lw=1.4, ls="--",
               label="behavioral twin (shifted to release)")
    ax[0].axhline(V_LOCK_PRED, color="#5a5a6e", ls=":", lw=0.8)
    ax[0].axvline(_T_RELEASE * 1e6, color="#009E73", ls=":", lw=1.0)
    ax[0].set_ylabel("VCTRL (V)")
    ax[0].set_title(f"Near-lock release — tail means "
                    f"{(s_lock - b_lock)*1e3:+.1f} mV apart "
                    f"(twin {b_lock:.4f} V, SPICE {s_lock:.4f} V)")
    ax[0].legend(frameon=False, fontsize=8, loc="lower right")

    for edges, color, ls, label in ((s_edges, "#0072B2", "-", "SPICE"),
                                    (b_edges, "#D55E00", "--", "twin")):
        tt, ff = _f_inst(edges)
        ax[1].plot(tt, ff, color=color, lw=1.2, ls=ls, label=label)
    ax[1].axhline(F_TARGET / 1e6, color="#5a5a6e", ls=":", lw=0.8,
                  label="600 MHz target")
    ax[1].set_ylabel("f_VCO (MHz)")
    ax[1].set_xlabel("time (µs)")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    for a in ax:
        a.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(OUT, "pll_twin_vs_spice.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n  saved {path}")
