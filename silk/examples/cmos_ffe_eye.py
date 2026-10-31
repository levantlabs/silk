"""Compare a unit-weighted CMOS FFE driver with a calibrated behavioral twin.

A PRBS9 stimulus drives four transistor tap inputs in SPICE. Each CMOS leg
operates between 0 and 1.8 V. A least-squares fit measures launch tap ratios;
the twin uses these ratios and matches the differential launch p2-p98 swing.
Both launches pass through the same behavioral channel with two poles and
an early reflection. This compares driver models; it does not validate the
channel or the twin's assumed, grid-quantized edge.

Run with ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.cmos_ffe_eye``.
Requires ngspice and sky130 models. Prints tap ratios, fit quality, launch
swing and geometric eye measurements. Saves ``cmos_twin_vs_spice_eye.png``
and ``cmos_ffe_eye.html`` in the configured example output directory.
"""
import base64
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


    t, launch = tx._run_cmos_ffe_driver(bits)
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
    print("CMOS TX eye — SPICE netlist (unit-weighted SST FFE) vs calibrated twin")
    print("=" * 72)
    print(f"  driver realized taps = "
          f"[{mtaps[0]:+.3f}, {mtaps[1]:+.3f}, {mtaps[2]:+.3f}, {mtaps[3]:+.3f}]  "
          f"(design ZF [{tx._FFE_TAPS_NOM[0]:+.3f}, {tx._FFE_TAPS_NOM[1]:+.3f}, "
          f"{tx._FFE_TAPS_NOM[2]:+.3f}, {tx._FFE_TAPS_NOM[3]:+.3f}], R^2={r2:.3f})")
    print(f"  launch pk-pk = {(np.percentile(lg,98)-np.percentile(lg,2))*1e3:.0f} mV "
          f"(differential p2-p98 on the resampled launch)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    hs, ws = _draw_eye(a1, tg - cml._FFE_T0, spice_rx, serial,
                       "SPICE netlist\n(unit-weighted SST FFE driver)", "#0072B2")
    ht, wt = _draw_eye(a2, t_tw, twin_rx, serial,
                       "behavioral twin\n(calibrated to measured taps)", "#D55E00")
    a1.set_ylabel("RX differential (mV)")
    fig.suptitle(f"CMOS TX eye — SPICE vs behavioral twin  (PRBS{_PRBS_ORDER}, {_N_BITS} bits, "
                 f"600 Mb/s over the 2-pole + reflection channel)\n"
                 f"twin calibrated to measured taps; crossing-centred eyes differ",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(OUT, "cmos_twin_vs_spice_eye.png")
    fig.savefig(path, dpi=125)
    plt.close(fig)

    print(f"\n  SPICE eye  = {hs:+.1f} mV / {ws:.0f} ps  (geometric opening)")
    print(f"  twin  eye  = {ht:+.1f} mV / {wt:.0f} ps  (geometric opening)")
    print(f"  difference = {abs(hs - ht):.2f} mV — crossing-centred twin vs netlist")
    print(f"\n  saved {path}")
    html = _emit_html(path, mtaps, r2, hs, ws, ht, wt)
    print(f"  saved {html}")
    return hs, ht


def _emit_html(png_path, mtaps, r2, hs, ws, ht, wt):
    """Save the eye plot and measured tap table in a self-contained HTML file."""
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    zf = tx._FFE_TAPS_NOM
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>CMOS TX eye — SPICE vs behavioral twin</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;
      margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fafafa}}
 h1{{font-size:1.35rem;margin:.2rem 0}} .sub{{color:#666;margin:.1rem 0 1.2rem}}
 img{{width:100%;border:1px solid #ddd;border-radius:6px;background:#fff}}
 table{{border-collapse:collapse;margin:1rem 0;font-size:14px}}
 td,th{{border:1px solid #ddd;padding:.35rem .7rem;text-align:left}}
 th{{background:#f0f0f4}} code{{background:#eee;padding:.05rem .3rem;border-radius:3px}}
 .ok{{color:#0a7d32;font-weight:600}}
</style></head><body>
<h1>CMOS TX eye — SPICE netlist vs behavioral twin</h1>
<p class="sub">Unit-weighted segmented SST FFE driver (<code>build_cmos_ffe_driver</code>),
PRBS9, 600&nbsp;Mb/s over the 2-pole&nbsp;+&nbsp;reflection channel. The twin is
calibrated to the driver's <em>measured</em> taps — so this compares real transistors
with a twin fitted to the launch. Both use the same behavioral channel. The twin uses an assumed, grid-quantized edge.</p>
<img src="data:image/png;base64,{b64}" alt="CMOS twin-vs-SPICE eye">
<table>
<tr><th></th><th>eye height</th><th>eye width</th><th></th></tr>
<tr><td>SPICE netlist</td><td>{hs:+.1f} mV</td><td>{ws:.0f} ps</td>
    <td class="ok">geometric opening</td></tr>
<tr><td>behavioral twin</td><td>{ht:+.1f} mV</td><td>{wt:.0f} ps</td>
    <td class="ok">geometric opening</td></tr>
<tr><td>difference</td><td colspan="3">{abs(hs-ht):.2f} mV — crossing-centred twin vs netlist</td></tr>
</table>
<p>Driver realized taps
<code>[{mtaps[0]:+.3f}, {mtaps[1]:+.3f}, {mtaps[2]:+.3f}, {mtaps[3]:+.3f}]</code>
vs zero-forcing design
<code>[{zf[0]:+.3f}, {zf[1]:+.3f}, {zf[2]:+.3f}, {zf[3]:+.3f}]</code>,
R&sup2;&nbsp;=&nbsp;{r2:.3f}. Scaling inverter multiplicity and series resistance
together preserves the intended conductance ratios. Nominal unit counts are 4/10/6/1.</p>
</body></html>"""
    out = os.path.join(OUT, "cmos_ffe_eye.html")
    with open(out, "w") as f:
        f.write(doc)
    return out


if __name__ == "__main__":
    example_output_dir()
    run()
