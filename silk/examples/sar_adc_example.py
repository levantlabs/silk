"""Differential charge-redistribution SAR ADC behavioral model.

The binary capacitor array includes independent unit-cap mismatch. Each bit
comparison can include Gaussian noise and a fixed input offset. The input
range is -VREF to +VREF, with nominal LSB = 2*VREF / 2**n.

A noise-free ramp measures endpoint-normalized DNL and endpoint INL using
silk.metrics. A coherent sine measures SNDR, ENOB and SFDR.
Run .venv/bin/python -m silk.examples.sar_adc_example from the repository root.
It prints the metrics and saves a PNG and HTML in the configured example
output directory. See sar_adc_spice.py for transistor-level comparisons.
"""
import base64
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from silk.metrics import sndr as metric_sndr, sfdr as metric_sfdr, enob as metric_enob, fft, extract_code_transitions, dnl as metric_dnl, inl as metric_inl
from silk.signals import Signal
from silk.examples import example_output_dir

OUT = example_output_dir(create=False)

# --- design constants -----------------------------------------------------------
N_BITS = 8                      # default resolution (everything is parameterized by n)
VREF = 0.9                      # differential reference (V); FS = 2*VREF = 1.8 Vppd
CAP_SIGMA_UNIT = 0.005          # unit-cap relative mismatch (0.5% unit-cap model)
COMP_SIGMA = 1.0e-3             # comparator input-referred noise, rms (V)
COMP_OFFSET = 0.0               # comparator input-referred offset (V)

# coherent-sine spectrum (power-of-two FFT, prime bin -> no window needed)
SINE_N_FFT = 4096
SINE_M_COH = 1193               # prime, coprime to 4096 -> fundamental in one bin


def lsb_volts(n=N_BITS, vref=VREF):
    """One LSB in volts (differential full-scale = 2*vref)."""
    return 2.0 * vref / (2 ** n)


# --- the capacitor DAC ----------------------------------------------------------

def unit_cap_array(n=N_BITS, sigma=CAP_SIGMA_UNIT, seed=0):
    """Build a binary-weighted array from ``2**n`` unit caps, each ~ N(1, sigma).

    Returns ``(w, wtot)`` where ``w[k]`` is the total capacitance (in unit-cap units)
    assigned to bit k (nominally ``2**k``, built from that many perturbed units) and
    ``wtot`` is the whole array (the ``2**n``-th unit is the terminating dummy). The
    per-bit relative sigma is therefore ``sigma/sqrt(2**k)`` by construction."""
    rng = np.random.default_rng(seed)
    units = 1.0 + sigma * rng.standard_normal(2 ** n)   # every unit cap, perturbed
    w = np.empty(n)
    idx = 0
    for k in range(n):                                  # bit k gets 2**k units
        m = 2 ** k
        w[k] = units[idx:idx + m].sum()
        idx += m
    wtot = units.sum()                                  # includes the 1-unit dummy
    return w, wtot


def dac_diff(code, w, wtot, vref=VREF):
    """Differential DAC output for integer ``code`` (array-ok): the set bits' caps
    to +ref, the rest to -ref -> ``v = FS * S(code)/wtot - vref``, S = sum of set
    bits' weights. Maps code 0 -> -vref, full -> ~+vref (mismatch bends the line)."""
    code = np.asarray(code)
    n = len(w)
    bits = ((code[..., None] >> np.arange(n)) & 1).astype(float)   # (..., n)
    s = bits @ w
    return 2.0 * vref * s / wtot - vref


# --- the SAR conversion (binary search over the mismatched DAC) ------------------

def sar_codes(vin, w, wtot, *, vref=VREF, comp_sigma=0.0, comp_offset=0.0, seed=1,
              dac_gain=1.0):
    """Convert differential inputs ``vin`` (array) to codes by successive
    approximation over the mismatched cap-DAC. MSB->LSB: at each step add
    the trial bit, form its DAC level, and keep the bit iff the (noisy, offset)
    input is at or above it. Vectorized over all inputs at once.

    ``comp_sigma`` adds fresh per-decision Gaussian noise (the comparator re-samples
    noise every bit); ``comp_offset`` is a fixed input-referred shift.
    ``dac_gain`` scales every DAC level the decision compares against: a circuit
    whose decisions trip at ``dac_gain * v_dac`` instead of ``v_dac``. The default
    1.0 is the ideal model; ``sar_adc_spice.LOOP_DAC_GAIN`` is the schematic
    eight-bit dummy-capacitor factor 256/255."""
    vin = np.asarray(vin, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(w)
    code = np.zeros(vin.shape, dtype=np.int64)
    s_cur = np.zeros(vin.shape)                          # running sum of kept weights
    vshift = vin + comp_offset
    for k in range(n - 1, -1, -1):
        v_dac = dac_gain * (2.0 * vref * (s_cur + w[k]) / wtot - vref)
        noise = comp_sigma * rng.standard_normal(vin.shape) if comp_sigma else 0.0
        keep = (vshift + noise) >= v_dac
        code[keep] |= (1 << k)
        s_cur[keep] += w[k]
    return code


# --- Static linearity from a noise-free ramp ---

def static_dnl_inl(n=N_BITS, w=None, wtot=None, *, vref=VREF, oversample=64,
                   comp_offset=0.0):
    """Return endpoint DNL, endpoint INL and transition voltages.

    DNL covers interior codes 1..2**n-2. INL uses the same code axis;
    trans[0] is unavailable because code zero has no measured transition.
    Unobserved transitions remain NaN rather than implying a missing code.
    """
    if w is None:
        w, wtot = unit_cap_array(n)
    npts = oversample * (2 ** n)
    vin = np.linspace(-vref, vref, npts)
    codes = sar_codes(vin, w, wtot, vref=vref, comp_sigma=0.0, comp_offset=comp_offset)
    axis = np.arange(npts, dtype=float)
    transitions = extract_code_transitions(
        Signal('input', 'voltage', 'time', 'V', axis, vin),
        Signal('codes', 'voltage', 'time', 'LSB', axis, codes))
    trans = np.full(2 ** n, np.nan)
    trans[transitions.x.astype(int)] = transitions.y
    table = Signal('transitions', 'voltage', 'code', 'V',
                   np.arange(1, 2 ** n, dtype=float), trans[1:])
    return metric_dnl(table).dnl_lsb, metric_inl(table).inl_lsb[:-1], trans


# --- dynamic: coherently-sampled sine -> SNDR / ENOB / SFDR ----------------------

def dynamic_sndr(n=N_BITS, w=None, wtot=None, *, vref=VREF, comp_sigma=COMP_SIGMA,
                 comp_offset=COMP_OFFSET, ampl=None, n_fft=SINE_N_FFT,
                 m_coh=SINE_M_COH, seed=2):
    """Feed a coherently-sampled full-scale sine, one conversion per sample, and
    return a dict with sndr_db, enob, sfdr_db, spectrum, codes. ENOB from the ideal
    quantization law ENOB = (SNDR - 1.76)/6.02."""
    if w is None:
        w, wtot = unit_cap_array(n)
    if ampl is None:
        ampl = 0.98 * vref                              # near full scale, no clip
    k = np.arange(n_fft)
    vin = ampl * np.sin(2.0 * np.pi * m_coh * k / n_fft)
    codes = sar_codes(vin, w, wtot, vref=vref, comp_sigma=comp_sigma,
                      comp_offset=comp_offset, seed=seed)
    sig = Signal('codes', 'voltage', 'time', 'LSB', k.astype(float), codes)
    sndr = metric_sndr(sig, tclk=1.0, n_samples=n_fft)
    sfdr = metric_sfdr(sig, tclk=1.0, n_samples=n_fft).sfdr_db
    enob = metric_enob(sig, tclk=1.0, n_samples=n_fft)
    spectrum = fft(sig, tclk=1.0, n_samples=n_fft)
    P = spectrum.y
    return {"sndr_db": sndr, "enob": enob, "sfdr_db": sfdr, "spectrum": P,
            "frequency": spectrum.x,
            "codes": codes, "ampl": ampl}


# --- report + plots -------------------------------------------------------------

def characterize(n=N_BITS, *, sigma=CAP_SIGMA_UNIT, comp_sigma=COMP_SIGMA,
                 comp_offset=COMP_OFFSET, seed=0):
    """Full behavioral characterization of an n-bit differential SAR: build one
    mismatched array, measure DNL/INL and SNDR/ENOB/SFDR. Returns a dict."""
    w, wtot = unit_cap_array(n, sigma, seed)
    dnl, inl, trans = static_dnl_inl(n, w, wtot, comp_offset=comp_offset)
    dyn = dynamic_sndr(n, w, wtot, comp_sigma=comp_sigma, comp_offset=comp_offset)
    return {"n": n, "w": w, "wtot": wtot, "dnl": dnl, "inl": inl, "trans": trans,
            "lsb_mv": lsb_volts(n) * 1e3, "sigma": sigma,
            "comp_sigma": comp_sigma, "comp_offset": comp_offset, **dyn}


def save_plots(res, png_path):
    """Transfer/DNL/INL + output spectrum -> one PNG. Returns the path."""
    n = res["n"]
    codes_x = np.arange(1, 2 ** n - 1)                   # DNL/INL code axis
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    # transfer curve (measured transitions vs ideal line)
    trans = res["trans"]
    D = np.arange(2 ** n)
    ideal = -VREF + D * lsb_volts(n)
    ax[0, 0].plot(D, trans * 1e3, color="#0072B2", lw=1.0, label="measured")
    ax[0, 0].plot(D, ideal * 1e3, color="#999999", lw=0.8, ls="--", label="ideal")
    ax[0, 0].set_title(f"transfer — code transition level ({n}-bit)", fontsize=10)
    ax[0, 0].set_xlabel("output code"); ax[0, 0].set_ylabel("input (mV, diff)")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True, alpha=0.4)

    ax[0, 1].plot(codes_x, res["dnl"], color="#009E73", lw=0.8)
    ax[0, 1].axhline(0, color="#999", lw=0.6)
    ax[0, 1].set_title(f"DNL — peak {np.abs(res['dnl']).max():.2f} LSB", fontsize=10)
    ax[0, 1].set_xlabel("code"); ax[0, 1].set_ylabel("DNL (LSB)")
    ax[0, 1].grid(True, alpha=0.4)

    ax[1, 0].plot(codes_x, res["inl"], color="#D55E00", lw=0.8)
    ax[1, 0].axhline(0, color="#999", lw=0.6)
    ax[1, 0].set_title(f"INL — peak {np.abs(res['inl']).max():.2f} LSB", fontsize=10)
    ax[1, 0].set_xlabel("code"); ax[1, 0].set_ylabel("INL (LSB)")
    ax[1, 0].grid(True, alpha=0.4)

    # Spectrum relative to its measured peak.
    P = res["spectrum"]
    Pn = P / P.max()
    f = res["frequency"]                            # normalized to Fs
    with np.errstate(divide="ignore"):
        dbfs = 10 * np.log10(np.maximum(Pn, 1e-20))
    ax[1, 1].plot(f, dbfs, color="#0072B2", lw=0.6)
    ax[1, 1].set_ylim(-120, 5)
    ax[1, 1].set_title(f"output spectrum — SNDR {res['sndr_db']:.1f} dB / "
                       f"ENOB {res['enob']:.2f} b", fontsize=10)
    ax[1, 1].set_xlabel("f / Fs"); ax[1, 1].set_ylabel("dB relative to peak")
    ax[1, 1].grid(True, alpha=0.4)

    fig.suptitle(f"{n}-bit differential SAR ADC — behavioral twin  "
                 f"(cap mismatch {res['sigma']*100:.1f}%/unit, "
                 f"comp noise {res['comp_sigma']*1e3:.1f} mV rms)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_path, dpi=125)
    plt.close(fig)
    return png_path


def emit_html(res, png_path):
    """Save an HTML report with the PNG embedded."""
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    n = res["n"]
    ideal_sndr = 6.02 * n + 1.76
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{n}-bit differential SAR ADC — behavioral twin</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
      margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fafafa}}
 h1{{font-size:1.35rem;margin:.2rem 0}} .sub{{color:#666;margin:.1rem 0 1.2rem}}
 img{{width:100%;border:1px solid #ddd;border-radius:6px;background:#fff}}
 table{{border-collapse:collapse;margin:1rem 0;font-size:14px}}
 td,th{{border:1px solid #ddd;padding:.35rem .7rem;text-align:left}}
 th{{background:#f0f0f4}} code{{background:#eee;padding:.05rem .3rem;border-radius:3px}}
</style></head><body>
<h1>{n}-bit differential SAR ADC — behavioral twin</h1>
<p class="sub">Charge-redistribution SAR, {2**n}-unit binary cap array with
{res['sigma']*100:.1f}%/unit mismatch, comparator noise {res['comp_sigma']*1e3:.1f}&nbsp;mV rms.
Static from a DC ramp, dynamic from a coherent sine + FFT.</p>
<img src="data:image/png;base64,{b64}" alt="SAR ADC characterization">
<table>
<tr><th>metric</th><th>value</th><th>reference</th></tr>
<tr><td>SNDR</td><td>{res['sndr_db']:.1f} dB</td>
    <td>ideal {ideal_sndr:.1f} dB (6.02·N+1.76)</td></tr>
<tr><td>ENOB</td><td>{res['enob']:.2f} bits</td><td>of {n} nominal</td></tr>
<tr><td>SFDR</td><td>{res['sfdr_db']:.1f} dB</td><td></td></tr>
<tr><td>DNL peak</td><td>{np.abs(res['dnl']).max():.2f} LSB</td><td>Missing codes require signed DNL &le; -1</td></tr>
<tr><td>INL peak</td><td>{np.abs(res['inl']).max():.2f} LSB</td><td></td></tr>
<tr><td>LSB</td><td>{res['lsb_mv']:.2f} mV</td><td>FS = 2·VREF = {2*VREF:.1f} Vppd</td></tr>
</table>
<p>See <code>sar_adc_spice.py</code> for transistor-level comparisons.</p>
</body></html>"""
    out = os.path.join(OUT, f"sar_adc_{n}bit.html")
    with open(out, "w") as f:
        f.write(doc)
    return out


def run(n=N_BITS):
    res = characterize(n)
    print("=" * 72)
    print(f"{n}-bit differential SAR ADC — behavioral twin")
    print("=" * 72)
    print(f"  full scale   = {2*VREF:.1f} Vppd differential,  LSB = {res['lsb_mv']:.2f} mV")
    print(f"  cap mismatch = {CAP_SIGMA_UNIT*100:.1f}%/unit ({2**n} unit caps)")
    if not np.isfinite(res['dnl']).all():
        code_status = 'unavailable widths'
    else:
        code_status = 'no missing codes' if res['dnl'].min() > -1 else 'missing codes'
    print(f"  DNL peak     = {np.abs(res['dnl']).max():+.2f} LSB   "
          f"INL peak = {np.abs(res['inl']).max():+.2f} LSB   "
          f"({code_status})")
    print(f"  SNDR         = {res['sndr_db']:.1f} dB   "
          f"ENOB = {res['enob']:.2f} b   SFDR = {res['sfdr_db']:.1f} dB   "
          f"(ideal SNDR {6.02*n+1.76:.1f} dB)")
    png = save_plots(res, os.path.join(OUT, f"sar_adc_{n}bit.png"))
    html = emit_html(res, png)
    print(f"\n  saved {png}")
    print(f"  saved {html}")
    return res


if __name__ == "__main__":
    example_output_dir()
    run()
