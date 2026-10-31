"""Sweep SAR resolution with capacitor mismatch and comparator noise.

The behavioral sweep measures ENOB with each effect separately and together,
plus endpoint-normalized static DNL and INL. It reports the maximum ENOB over
the swept range and the first resolution more than 0.5 bit below nominal.
That shortfall does not establish where ENOB stops improving.

An optional nominal transistor comparison digitizes short sine records. It
uses ideal capacitors and no declared mismatch or noise, so it does not
validate the behavioral mismatch or noise limit.

Run .venv/bin/python -m silk.examples.sar_resolution_sweep from the repository
root for the behavioral sweep. Add quick for 16-conversion SPICE runs at four
and six bits, or full for 32-conversion runs at four, six and eight bits.
SPICE requires ngspice and sky130 models. Prints metric tables and saves PNG
and HTML reports in the configured example output directory.
"""
import base64
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from silk.examples import sar_adc_example as sar
from silk.examples import sar_adc_spice as spice
from silk.metrics import sndr, enob
from silk.signals import Signal
from silk import load_pdk
from silk.examples import example_output_dir
OUT = example_output_dir(create=False)

# --- sweep configuration --------------------------------------------------------
BEHAV_NS = list(range(4, 14))       # 4..13 bits (behavioral is cheap → sweep wide)
SPICE_NS = [4, 6, 8]                # optional nominal transistor resolutions
SINE_TCLK = 14e-9                  # Clock period shared by SPICE and its nominal reference.
SINE_MCONV = 32                     # conversions per SPICE sine run (short coherent FFT)
SINE_PERIODS = 7                    # signal cycles over the run (coprime to MCONV → 1 bin)
MISMATCH_SEED = 0                   # fixed cap-mismatch draw so the sweep is reproducible


# --- behavioral resolution sweep -------------------------

def _enob(n, *, sigma, comp_sigma, ampl=None):
    """ENOB of the behavioral twin at resolution ``n`` with the given cap-mismatch
    sigma and comparator-noise sigma. One coherent-sine SNDR run (4096 samples)."""
    w, wtot = sar.unit_cap_array(n, sigma=sigma, seed=MISMATCH_SEED)
    dyn = sar.dynamic_sndr(n, w, wtot, comp_sigma=comp_sigma, comp_offset=0.0, ampl=ampl)
    return dyn["enob"], dyn["sndr_db"]


def sweep_behavioral(ns=BEHAV_NS):
    """Sweep resolution and measure ENOB four ways — ideal (quantization only),
    capacitor mismatch alone, comparator noise alone, and both together — plus the
    static DNL/INL peak vs n. Returns a dict of numpy arrays indexed by ``ns``."""
    ns = np.asarray(ns)
    enob_ideal = np.empty(len(ns))      # sigma=0, noise=0  → should recover ENOB≈n
    enob_mm = np.empty(len(ns))         # cap mismatch only
    enob_noise = np.empty(len(ns))      # comparator noise only
    enob_both = np.empty(len(ns))       # both (the realized ADC)
    dnl_peak = np.empty(len(ns))
    inl_peak = np.empty(len(ns))
    for i, n in enumerate(ns):
        n = int(n)
        enob_ideal[i], _ = _enob(n, sigma=0.0, comp_sigma=0.0)
        enob_mm[i], _ = _enob(n, sigma=sar.CAP_SIGMA_UNIT, comp_sigma=0.0)
        enob_noise[i], _ = _enob(n, sigma=0.0, comp_sigma=sar.COMP_SIGMA)
        enob_both[i], _ = _enob(n, sigma=sar.CAP_SIGMA_UNIT, comp_sigma=sar.COMP_SIGMA)
        w, wtot = sar.unit_cap_array(n, sigma=sar.CAP_SIGMA_UNIT, seed=MISMATCH_SEED)
        dnl, inl, _ = sar.static_dnl_inl(n, w, wtot)
        dnl_peak[i] = np.abs(dnl).max()
        inl_peak[i] = np.abs(inl).max()
    return {"ns": ns, "enob_ideal": enob_ideal, "enob_mm": enob_mm,
            "enob_noise": enob_noise, "enob_both": enob_both,
            "dnl_peak": dnl_peak, "inl_peak": inl_peak}


def _saturation_bits(ns, enob_both, tol=0.5):
    """Maximum swept ENOB and first resolution more than tol bits below nominal."""
    ns = np.asarray(ns)
    short = (ns - enob_both) > tol
    knee = int(ns[short][0]) if short.any() else None
    return float(np.max(enob_both)), knee


# --- nominal SPICE comparison: digitize a sine at a few resolutions --------------------

def _enob_from_codes(codes):
    """SNDR/ENOB from a short coherent record of conversion codes (metric used by the
    SPICE sine and nominal short-record twin)."""
    sig = Signal('codes', 'voltage', 'time', 'LSB',
                 np.arange(len(codes), dtype=float), np.asarray(codes, float))
    return (sndr(sig, tclk=1.0, n_samples=len(codes)),
            enob(sig, tclk=1.0, n_samples=len(codes)))


def _matched_record_enob(n, *, m_conv=SINE_MCONV, periods=SINE_PERIODS, ampl=None):
    """Compare a nominal twin using the short sine record and dummy terms.

    Sampling uses the end of the sample-control falling edge. Finite switch
    aperture and comparator behavior can still differ from this approximation.
    """
    ampl = ampl if ampl is not None else 0.8 * spice.VREFP
    c = np.arange(m_conv)
    theta = 2 * np.pi * periods * (c + (1.0 - 0.1e-9 / SINE_TCLK) / (n + 2)) / m_conv
    vin = ampl * np.sin(theta)
    w, wtot = sar.unit_cap_array(n, sigma=0.0)          # nominal caps, as build_cap_dac
    codes = sar.sar_codes(vin, w, wtot, vref=spice.VREFP, comp_sigma=0.0,
                          dac_gain=2**n/(2**n-1),
                          comp_offset=-spice.VREFP/(2**n-1))
    return _enob_from_codes(codes)[1]


def _sine_enob_spice(n, pdk, od, *, m_conv=SINE_MCONV, periods=SINE_PERIODS):
    """Run a transistor sine record and return its metrics and nominal reference."""
    t, resid, vindiff, st, codes, Tconv = spice.convert_sar_sine(
        n, m_conv, pdk, od, periods=periods, tclk=SINE_TCLK)
    lsb = 2 * spice.VREFP / 2 ** n
    recon = (codes + 0.5) * lsb - spice.VREFP
    sndr, enob = _enob_from_codes(codes)
    enob_matched = _matched_record_enob(n, m_conv=m_conv, periods=periods)
    # analog input sampled at each conversion's sample instant (for the plot)
    vin_samp = np.interp(st, t, vindiff)
    return {"n": n, "sndr_db": sndr, "enob": enob, "enob_matched": enob_matched,
            "sample_t": st, "vin_samp": vin_samp, "recon": recon, "codes": codes,
            "vindiff_t": (t, vindiff)}


def sweep_spice_sine(ns=SPICE_NS, *, m_conv=SINE_MCONV, periods=SINE_PERIODS):
    """Digitize a sine in real transistors at each resolution in ``ns``. Slow (one big
    continuous transient per resolution). Returns a list of per-n result dicts."""
    pdk = load_pdk('sky130')
    od = example_output_dir()
    out = []
    for n in ns:
        print(f"  SPICE sine: {n}-bit, {m_conv} conversions in one transient "
              f"(~{(n+2)*m_conv} clock cycles)...")
        r = _sine_enob_spice(int(n), pdk, od, m_conv=m_conv, periods=periods)
        print(f"    -> SPICE ENOB {r['enob']:.2f} b  (SNDR {r['sndr_db']:.1f} dB);  "
              f"twin @ same {m_conv}-record {r['enob_matched']:.2f} b;  "
              f"ENOB difference {r['enob_matched']-r['enob']:+.2f} b")
        out.append(r)
    return out


# --- plots + HTML ----------------------------------------------------------------

def plot_saturation(bh, spice_res, png_path):
    """ENOB vs resolution — ideal / mismatch-wall / noise-wall / both, with the SPICE
    points overlaid, and a static-linearity (DNL/INL) panel."""
    ns = bh["ns"]
    ceiling, knee = _saturation_bits(ns, bh["enob_both"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # --- panel 1: ENOB vs n ---
    a1.plot(ns, ns, ls="--", color="#999", lw=1.2, label="ideal (ENOB = n)")
    a1.plot(ns, bh["enob_ideal"], ":", color="#bbb", lw=1.0,
            label="twin, no non-idealities")
    a1.plot(ns, bh["enob_mm"], "o-", color="#009E73", ms=4, lw=1.2,
            label=f"cap mismatch only ({sar.CAP_SIGMA_UNIT*100:.1f}%/unit)")
    a1.plot(ns, bh["enob_noise"], "s-", color="#E69F00", ms=4, lw=1.2,
            label=f"comparator noise only ({sar.COMP_SIGMA*1e3:.0f} mV rms)")
    a1.plot(ns, bh["enob_both"], "D-", color="#0072B2", ms=5, lw=2.0,
            label="both (realized SAR)")
    if spice_res:
        sx = [r["n"] for r in spice_res]
        sy = [r["enob"] for r in spice_res]
        sm = [r["enob_matched"] for r in spice_res]
        # Join each nominal twin result to the corresponding SPICE result.
        for xi, ylo, yhi in zip(sx, sy, sm):
            a1.plot([xi, xi], [ylo, yhi], color="#D55E00", lw=0.8, alpha=0.4, zorder=4)
        a1.plot(sx, sm, "x", color="#555", ms=9, mew=1.6, zorder=5,
                label=f"twin @ matched {len(spice_res[0]['codes'])}-conv record")
        a1.plot(sx, sy, "*", color="#D55E00", ms=15, zorder=6,
                label="real transistor SAR (SPICE sine)")
    if knee is not None:
        a1.axvline(knee, color="#D55E00", lw=0.8, ls=":", alpha=0.7)
        a1.annotate(f"first >0.5-bit shortfall\n(maximum {ceiling:.1f} bits)",
                    xy=(knee, bh["enob_both"][np.flatnonzero(ns == knee)[0]]),
                    xytext=(knee + 0.3, ceiling - 2.2),
                    fontsize=8, color="#D55E00",
                    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=0.8))
    a1.set_title("ENOB vs nominal resolution", fontsize=10)
    a1.set_xlabel("nominal resolution n (bits)"); a1.set_ylabel("ENOB (effective bits)")
    a1.legend(fontsize=7.5, loc="upper left"); a1.grid(True, alpha=0.4)
    a1.set_xticks(ns)

    # --- panel 2: static linearity wall ---
    a2.plot(ns, bh["dnl_peak"], "o-", color="#009E73", ms=4, lw=1.2, label="peak |DNL|")
    a2.plot(ns, bh["inl_peak"], "s-", color="#D55E00", ms=4, lw=1.2, label="peak |INL|")
    a2.axhline(1.0, color="#999", lw=0.8, ls="--")
    a2.text(ns[0], 1.05, "1-LSB error reference", fontsize=7.5, color="#666")
    a2.set_title(f"static linearity from the {sar.CAP_SIGMA_UNIT*100:.1f}%/unit arrays",
                 fontsize=10)
    a2.set_xlabel("nominal resolution n (bits)"); a2.set_ylabel("peak error (LSB)")
    a2.legend(fontsize=8, loc="upper left"); a2.grid(True, alpha=0.4)
    a2.set_xticks(ns)

    fig.suptitle("SAR ADC resolution sweep — ENOB with cap mismatch + "
                 "comparator noise (behavioral sweep)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=125)
    plt.close(fig)
    return png_path


def plot_spice_sines(spice_res, png_path):
    """Small-multiples: the real transistor SAR digitizing a sine at each resolution —
    showing the measured codes at each resolution."""
    if not spice_res:
        return None
    k = len(spice_res)
    fig, axes = plt.subplots(k, 1, figsize=(11, 2.3 * k), sharex=False)
    if k == 1:
        axes = [axes]
    for ax, r in zip(axes, spice_res):
        t, vindiff = r["vindiff_t"]
        ax.plot(t * 1e9, vindiff * 1e3, color="#009E73", lw=1.2, label="input sine")
        ax.plot(r["sample_t"] * 1e9, r["recon"] * 1e3, "o", color="#0072B2", ms=5,
                label="SAR output (code→V)")
        ax.step(r["sample_t"] * 1e9, r["recon"] * 1e3, where="mid", color="#0072B2",
                lw=0.8, alpha=0.5)
        ax.set_title(f"{r['n']}-bit real SAR — measured ENOB {r['enob']:.2f} b "
                     f"(SNDR {r['sndr_db']:.1f} dB, {len(r['codes'])} conversions)",
                     fontsize=9)
        ax.set_ylabel("diff (mV)"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper right")
    axes[-1].set_xlabel("time (ns)")
    fig.suptitle("Real transistor SAR digitizing a sine in one transient (on-chip FSM, "
                 "no Python) — reconstruction refines with resolution", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=125)
    plt.close(fig)
    return png_path


def emit_html(bh, spice_res, sweep_png, sine_png):
    ceiling, knee = _saturation_bits(bh["ns"], bh["enob_both"])
    with open(sweep_png, "rb") as f:
        b64_sweep = base64.b64encode(f.read()).decode("ascii")
    sine_img = ""
    if sine_png:
        with open(sine_png, "rb") as f:
            b64_sine = base64.b64encode(f.read()).decode("ascii")
        sine_img = (f'<h2>Nominal transistor comparison — sampled sine</h2>'
                    f'<img src="data:image/png;base64,{b64_sine}" '
                    f'alt="SPICE sine digitization">')
    # behavioral table
    rows = "\n".join(
        f"<tr><td>{int(n)}</td><td>{ei:.2f}</td><td>{em:.2f}</td>"
        f"<td>{en:.2f}</td><td>{eb:.2f}</td><td>{dn:.2f}</td><td>{il:.2f}</td></tr>"
        for n, ei, em, en, eb, dn, il in zip(
            bh["ns"], bh["enob_ideal"], bh["enob_mm"], bh["enob_noise"],
            bh["enob_both"], bh["dnl_peak"], bh["inl_peak"]))
    spice_rows = ""
    if spice_res:
        srows = "\n".join(
            f"<tr><td>{r['n']}</td><td>{r['sndr_db']:.1f}</td><td>{r['enob']:.2f}</td>"
            f"<td>{r['enob_matched']:.2f}</td><td>{r['enob_matched']-r['enob']:+.2f}</td>"
            f"<td>{len(r['codes'])}</td></tr>" for r in spice_res)
        spice_rows = (
            "<h2>SPICE sine (real transistors) vs the twin at the same short record</h2>"
            "<table><tr><th>n</th><th>SNDR (dB)</th><th>SPICE ENOB</th>"
            "<th>twin @ matched record</th><th>ENOB difference</th>"
            f"<th>conversions</th></tr>{srows}</table>"
            "<p class='sub'>This compares short-record SPICE and behavioral ENOB. "
            "Differences can include sampling, model and circuit effects.</p>")

    knee_txt = (f"<b>n = {knee}</b>"
                if knee else "beyond the swept range")
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SAR ADC resolution sweep — ENOB vs resolution</title>
<style>
 body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
      margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fafafa}}
 h1{{font-size:1.35rem;margin:.2rem 0}} h2{{font-size:1.1rem;margin:1.6rem 0 .4rem}}
 .sub{{color:#666;margin:.1rem 0 1.2rem}}
 img{{width:100%;border:1px solid #ddd;border-radius:6px;background:#fff;margin:.4rem 0}}
 table{{border-collapse:collapse;margin:1rem 0;font-size:14px}}
 td,th{{border:1px solid #ddd;padding:.3rem .7rem;text-align:right}} th{{background:#f0f0f4}}
 code{{background:#eee;padding:.05rem .3rem;border-radius:3px}}
</style></head><body>
<h1>SAR ADC resolution sweep — ENOB and static error vs resolution</h1>
<p class="sub">Differential charge-redistribution SAR. ENOB is swept vs nominal
resolution with cap mismatch ({sar.CAP_SIGMA_UNIT*100:.1f}%/unit) and comparator noise
({sar.COMP_SIGMA*1e3:.0f}&nbsp;mV rms) turned on one at a time and together, optionally accompanied by a nominal transistor comparison.</p>
<img src="data:image/png;base64,{b64_sweep}" alt="ENOB vs resolution">
<p>The first resolution more than 0.5 bit below nominal is {knee_txt}.
The maximum ENOB over this swept range is {ceiling:.1f} bits.</p>
{sine_img}
{spice_rows}
<h2>Behavioral sweep data</h2>
<table>
<tr><th>n</th><th>ENOB ideal</th><th>mismatch only</th><th>noise only</th>
    <th>both</th><th>peak DNL</th><th>peak INL</th></tr>
{rows}
</table>
<p class="sub">ENOB ideal recovers ≈ n (validates the SNDR measurement). "Mismatch only"
is the static-linearity wall; "noise only" is the dynamic floor; "both" is the realized
converter under both modeled effects.</p>
</body></html>"""
    out = os.path.join(OUT, "sar_resolution_sweep.html")
    with open(out, "w") as f:
        f.write(doc)
    return out


def run(spice_ns=None, *, m_conv=SINE_MCONV):
    """Behavioral sweep (always) + optional nominal SPICE sine comparison at ``spice_ns``."""
    example_output_dir()
    print("=" * 74)
    print("SAR ADC resolution sweep — ENOB with cap mismatch + comparator noise")
    print("=" * 74)
    print(f"  behavioral: n = {BEHAV_NS[0]}..{BEHAV_NS[-1]} bits, "
          f"cap {sar.CAP_SIGMA_UNIT*100:.1f}%/unit, comp {sar.COMP_SIGMA*1e3:.0f} mV rms")
    bh = sweep_behavioral()
    ceiling, knee = _saturation_bits(bh["ns"], bh["enob_both"])
    print(f"\n  {'n':>3}{'ideal':>8}{'mism':>8}{'noise':>8}{'both':>8}"
          f"{'DNLpk':>8}{'INLpk':>8}")
    print("  " + "-" * 48)
    for i, n in enumerate(bh["ns"]):
        print(f"  {int(n):>3}{bh['enob_ideal'][i]:>8.2f}{bh['enob_mm'][i]:>8.2f}"
              f"{bh['enob_noise'][i]:>8.2f}{bh['enob_both'][i]:>8.2f}"
              f"{bh['dnl_peak'][i]:>8.2f}{bh['inl_peak'][i]:>8.2f}")
    print(f"\n  Maximum swept ENOB {ceiling:.1f} bits; first >0.5-bit shortfall at n={knee}.")

    spice_res = []
    if spice_ns:
        print(f"\n  comparing nominal SPICE at n = {spice_ns} "
              f"({m_conv} conversions each, one transient per resolution)...")
        spice_res = sweep_spice_sine(spice_ns, m_conv=m_conv)

    sweep_png = plot_saturation(bh, spice_res, os.path.join(OUT, "sar_resolution_sweep.png"))
    sine_png = (plot_spice_sines(spice_res, os.path.join(OUT, "sar_resolution_sines.png"))
                if spice_res else None)
    html = emit_html(bh, spice_res, sweep_png, sine_png)
    print(f"\n  saved {sweep_png}")
    if sine_png:
        print(f"  saved {sine_png}")
    print(f"  saved {html}")
    return bh, spice_res


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "full":
        run(SPICE_NS)                       # behavioral + SPICE sine at 4,6,8 (slow)
    elif arg == "quick":
        run([4, 6], m_conv=16)              # behavioral + a fast 2-point SPICE check
    else:
        run()                               # behavioral sweep only (fast)
