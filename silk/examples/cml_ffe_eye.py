"""Compare a transistor CML FFE driver with a calibrated behavioral twin.

A PRBS9 stimulus drives four transistor tap inputs in SPICE. A least-squares
fit measures the launch tap ratios. The twin uses those ratios and matches
the measured launch swing. Both launches pass through the same behavioral
channel with two poles and an early reflection. This compares driver models;
it does not validate the channel or the twin's assumed, grid-quantized edge.

Run with ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.cml_ffe_eye``.
Requires ngspice and sky130 models. Prints tap ratios, fit quality,
differential launch p2-p98 swing and geometric eye height and width.
Saves ``cml_twin_vs_spice_eye.png`` in the configured example output directory.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from silk.examples import cml_cells as cml
from silk.examples import serdes_tx_example as tx
from silk.examples import example_output_dir
from silk.signals import Signal
from silk.metrics import sample
from silk.sources import prbs_bits
OUT = example_output_dir(create=False)

_N_BITS = 200
_PRBS_ORDER = 9


def _draw_eye(ax, t, rx, bits, title, color):
    """Draw the catalog eye's original-sample traces and return mV / ps."""
    sig = Signal("rx", "voltage", "time", "V", t, rx)
    eye = tx._rx_eye(sig, bits, tx._UI, n_settle=14, max_lag=8)
    h, w = eye.height * 1e3, eye.width * 1e12
    for trace in eye.traces:
        Signal("eye", "voltage", "time within UI (ps)", ax.get_ylabel(),
                   (trace.x - 0.5 * tx._UI) * 1e12, trace.y * 1e3).plot(ax=ax,
                color=color, lw=0.4, alpha=0.25)
    for threshold in eye.levels:
        ax.axhline(threshold * 1e3, color="#009E73", ls=":", lw=1.0)
    ax.set_title(f"{title}\neye = {h:.1f} mV / {w:.0f} ps" +
                 "  (geometric opening)", fontsize=10)
    ax.set_xlabel("time within UI (ps)", fontsize=8)
    ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    return h, w


def run():
    bits = prbs_bits(_PRBS_ORDER, _N_BITS + 6, seed=0)
    serial = list(bits[:_N_BITS])


    t, launch = cml._run_ffe_driver(bits)
    mtaps, main_v, r2 = cml.fit_ffe_taps(t, launch, bits)

    dt = tx._UI / tx._N_OS
    tg = np.arange(cml._FFE_T0, float(t[-1]), dt)
    lg = sample(Signal("launch", "voltage", "time", "V", t, launch),
                x_new=tg).y
    lg = lg - lg.mean()
    spice_rx = tx._channel(lg, dt)


    nser = np.array([2 * b - 1 for b in serial], float)
    fir = tx._apply_ffe(nser, mtaps, tx._FFE_DELAYS) / np.sum(np.abs(mtaps))
    t_tw, tw_launch = tx._nrz(fir, tx._UI, dt, tx._TEDGE_NOM)
    scale = ((np.percentile(lg, 98) - np.percentile(lg, 2)) /
             (np.percentile(tw_launch, 98) - np.percentile(tw_launch, 2)))
    twin_rx = tx._channel(tw_launch * scale, dt)

    print("=" * 72)
    print("CML TX eye — SPICE netlist (multi-tap current FFE) vs calibrated twin")
    print("=" * 72)
    print(f"  driver realized taps = "
          f"[{mtaps[0]:+.3f}, {mtaps[1]:+.3f}, {mtaps[2]:+.3f}, {mtaps[3]:+.3f}]  "
          f"(design ZF [{cml._FFE_TAPS[0]:+.3f}, {cml._FFE_TAPS[1]:+.3f}, "
          f"{cml._FFE_TAPS[2]:+.3f}, {cml._FFE_TAPS[3]:+.3f}], R^2={r2:.3f})")
    print(f"  launch pk-pk = {(np.percentile(lg,98)-np.percentile(lg,2))*1e3:.0f} mV "
          f"(differential p2-p98 on the resampled launch)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    hs, ws = _draw_eye(a1, tg - cml._FFE_T0, spice_rx, serial,
                       "SPICE netlist\n(multi-tap current FFE driver)", "#0072B2")
    ht, wt = _draw_eye(a2, t_tw, twin_rx, serial,
                       "behavioral twin\n(calibrated to measured taps)", "#D55E00")
    a1.set_ylabel("RX differential (mV)")
    fig.suptitle(f"CML TX eye — SPICE vs behavioral twin  (PRBS{_PRBS_ORDER}, {_N_BITS} bits, "
                 f"600 Mb/s over the 2-pole + reflection channel)\n"
                 f"measured-tap twin and SPICE launch through the same behavioral channel",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(OUT, "cml_twin_vs_spice_eye.png")
    fig.savefig(path, dpi=125)
    plt.close(fig)

    print(f"\n  SPICE eye  = {hs:+.1f} mV / {ws:.0f} ps  (geometric opening)")
    print(f"  twin  eye  = {ht:+.1f} mV / {wt:.0f} ps  (geometric opening)")
    print(f"  twin - SPICE = {ht - hs:+.2f} mV height / {wt - ws:+.2f} ps width "
          "(shared behavioral channel; assumed twin edge)")
    print(f"\n  saved {path}")
    return hs, ht


if __name__ == "__main__":
    example_output_dir()
    run()
