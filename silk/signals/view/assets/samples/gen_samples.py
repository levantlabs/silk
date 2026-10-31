#!/usr/bin/env python3
"""Generate the five hand-authored sample payloads for the waveform-explorer
frontend, using the typed-array payload format consumed by the viewer.

These are *not* produced by the real emitter (silk/signals/view/emit.py,
Lane A's territory) -- they are independently hand-built from the same pinned
schema and the same pinned decimation algorithm, so
that Lane B can develop and test view.js without any dependency on Lane A's
code. Structurally they must be interchangeable with the real emitter's
output.

Usage:
    python3 gen_samples.py

Requires silk and numpy. Uses the shared JSON codec and writes sample_*.js next to this script.
"""
from __future__ import annotations

import base64
import json
import math
import os

import numpy as np

from silk._json_codec import encode_json

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Decimation pyramid -- pinned exactly per design doc section 4.8.
# ---------------------------------------------------------------------------

DECIMATE_MIN_POINTS = 8000
DECIMATE_BASE = 8
DECIMATE_MIN_BUCKET = 1000
DECIMATE_MAX_LEVELS = 8


def b64(arr, dtype):
    """base64-encode *arr* cast to little-endian *dtype*."""
    return base64.b64encode(np.asarray(arr, dtype=dtype).tobytes()).decode("ascii")


def choose_y_dtype(y):
    """Float32 range escape hatch."""
    finite = y[np.isfinite(y)]
    if finite.size:
        mag = np.abs(finite)
        nonzero = mag[mag > 0]
        if np.any(mag > 3.0e38) or (nonzero.size and np.any(nonzero < 1e-37)):
            return "<f8", "f64"
    return "<f4", "f32"


def minmax_level(x, y, stride, nbuckets):
    """Pinned minmax_level algorithm."""
    n = len(y)
    xs = []
    ys = []
    for b in range(nbuckets):
        lo = b * stride
        hi = min(lo + stride, n)
        seg = y[lo:hi]
        if np.all(np.isnan(seg)):
            xs.append(x[lo])
            ys.append(np.nan)
            continue
        imin = lo + int(np.nanargmin(seg))
        imax = lo + int(np.nanargmax(seg))
        if imin <= imax:
            xs.append(x[imin]); ys.append(y[imin])
            xs.append(x[imax]); ys.append(y[imax])
        else:
            xs.append(x[imax]); ys.append(y[imax])
            xs.append(x[imin]); ys.append(y[imin])
    return np.asarray(xs, dtype="<f8"), np.asarray(ys, dtype="<f8")


def build_levels(x, y):
    """Return the `levels` array for a data Signal (level 0 carries y only;
    levels[k>=1] carry both x and y)."""
    n = len(y)
    np_dtype, dt_tag = choose_y_dtype(y)
    levels = [{
        "stride": 1,
        "n": n,
        "dt": dt_tag,
        "y": b64(y, np_dtype),
    }]
    if n > DECIMATE_MIN_POINTS:
        stride = 1
        while len(levels) < DECIMATE_MAX_LEVELS:
            stride *= DECIMATE_BASE
            nbuckets = math.ceil(n / stride)
            if nbuckets < DECIMATE_MIN_BUCKET:
                break
            lx, ly = minmax_level(x, y, stride, nbuckets)
            levels.append({
                "stride": stride,
                "n": len(ly),
                "dt": dt_tag,
                "x": b64(lx, "<f8"),
                "y": b64(ly, np_dtype),
            })
    return levels


def make_x_signal(name, type_, unit, x):
    return {
        "name": name,
        "type": type_,
        "unit": unit,
        "part": None,
        "group": None,
        "nonfinite": 0,
        "levels": [{
            "stride": 1,
            "n": len(x),
            "dt": "f64",
            "x": b64(np.asarray(x, dtype="<f8"), "<f8"),
        }],
    }


def make_signal(name, type_, unit, y, x_full, part=None, group=None, renamed=False):
    y = np.asarray(y, dtype="<f8")
    nonfinite = int(np.sum(~np.isfinite(y)))
    sig = {
        "name": name,
        "type": type_,
        "unit": unit,
        "part": part,
        "group": group,
        "nonfinite": nonfinite,
        "levels": build_levels(np.asarray(x_full, dtype="<f8"), y),
    }
    if renamed:
        sig["renamed"] = True
    return sig


def make_complex_pair(orig_name, type_, unit, z, x_full):
    """A complex vector -> (mag, phase) Signal pair."""
    mag = np.abs(z)
    phase_deg = np.degrees(np.unwrap(np.angle(z)))
    mag_sig = make_signal(f"{orig_name} (mag)", type_, unit, mag, x_full,
                           part="mag", group=orig_name)
    phase_sig = make_signal(f"{orig_name} (phase)", type_, "deg", phase_deg, x_full,
                             part="phase", group=orig_name)
    return mag_sig, phase_sig


def make_plot(name, kind, complex_, x_signal, signals, x_log_default=False, date=""):
    npoints = x_signal["levels"][0]["n"]
    return {
        "name": name,
        "kind": kind,
        "complex": complex_,
        "npoints": npoints,
        "x": x_signal,
        "signals": signals,
        "x_log_default": x_log_default,
        "date": date,
    }


def make_payload(run_id, state_key, title, source, plots, default_plot, annotations=None):
    return {
        "format": 1,
        "endian": "le",
        "generator": "silk.signals.view",
        "generator_version": "0.2.0",
        "run": {
            "id": run_id,
            "state_key": state_key,
            "emitted_at": "2026-08-10T00:00:00Z",
            "source": source,
            "title": title,
        },
        "plots": plots,
        "default_plot": default_plot,
        "annotations": annotations or {},
    }


def write_sample(filename, payload):
    """JSONP-style handshake, byte for byte."""
    body = json.dumps(encode_json(payload, finite_only=True), allow_nan=False, separators=(",", ":"))
    js = (
        "(function(){var p=" + body + ";\n"
        "if(typeof window.__silkViewData==='function'){window.__silkViewData(p);}"
        "else{window.SILK_VIEW_DATA=p;}})();\n"
    )
    path = os.path.join(HERE, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(js)
    print(f"wrote {filename}  ({os.path.getsize(path):,} bytes)")


# ---------------------------------------------------------------------------
# sample_tran.js -- 1 plot, kind=tran, 6 signals, ~200 points, level 0 only.
# ---------------------------------------------------------------------------

def gen_sample_tran():
    n = 200
    t = np.linspace(0, 200e-9, n)
    vin = np.where(t < 20e-9, 0.0, 1.2) + 0.01 * np.sin(2 * np.pi * 5e7 * t)
    vout = 1.2 * (1 - np.exp(-(t - 20e-9) / 15e-9)) * (t >= 20e-9)
    vbias = 0.6 + 0.005 * np.sin(2 * np.pi * 2e7 * t)
    ivdd = -2.5e-3 * np.exp(-t / 40e-9) - 0.1e-3
    vclk = 0.6 + 0.6 * np.sign(np.sin(2 * np.pi * 1e8 * t))
    xstate = (np.sin(2 * np.pi * 3e7 * t) > 0).astype(float)

    x_sig = make_x_signal("time", "time", "s", t)
    signals = [
        make_signal("v(in)", "voltage", "V", vin, t),
        make_signal("v(out)", "voltage", "V", vout, t),
        make_signal("v(bias)", "voltage", "V", vbias, t),
        make_signal("i(vdd)", "current", "A", ivdd, t),
        make_signal("v(clk)", "voltage", "V", vclk, t),
        make_signal("x(state)", "notype", "", xstate, t),
    ]
    plot = make_plot("Transient Analysis", "tran", False, x_sig, signals,
                      date="Mon Aug 10 00:00:00 2026")
    payload = make_payload(
        "a1a1a1a1a1a1a1a1", "a1a1a1a1", "sample_tran — inverter step response",
        "samples/sample_tran.raw", [plot], 0,
    )
    write_sample("sample_tran.js", payload)


# ---------------------------------------------------------------------------
# sample_tran_big.js -- 1 plot, 2 signals, 1e6 points, 3-sample spike, 4 levels.
# ---------------------------------------------------------------------------

def gen_sample_tran_big():
    n = 1_000_000
    t = np.linspace(0, 1e-3, n)
    sig_a = 0.5 * np.sin(2 * np.pi * 1e3 * t) + 0.02 * np.sin(2 * np.pi * 5e5 * t)
    spike_idx = n // 2
    sig_a[spike_idx:spike_idx + 3] = 50.0  # deliberate 3-sample spike, must survive every level
    sig_b = 0.8 * np.exp(-t / 3e-4) * np.cos(2 * np.pi * 2e4 * t)

    x_sig = make_x_signal("time", "time", "s", t)
    signals = [
        make_signal("v(node_a)", "voltage", "V", sig_a, t),
        make_signal("v(node_b)", "voltage", "V", sig_b, t),
    ]
    plot = make_plot("Transient Analysis (big)", "tran", False, x_sig, signals,
                      date="Mon Aug 10 00:00:01 2026")
    expected_strides = [lvl["stride"] for lvl in signals[0]["levels"]]
    assert expected_strides == [1, 8, 64, 512], expected_strides
    payload = make_payload(
        "b2b2b2b2b2b2b2b2", "b2b2b2b2", "sample_tran_big — 1e6-point pyramid stress",
        "samples/sample_tran_big.raw", [plot], 0,
    )
    write_sample("sample_tran_big.js", payload)


# ---------------------------------------------------------------------------
# sample_ac.js -- kind=ac, complex source expanded to mag/phase, x_log_default.
# ---------------------------------------------------------------------------

def gen_sample_ac():
    freq = np.logspace(0, 9, 61)  # 1 Hz .. 1 GHz, 61 points (matches c_ac_bin.raw shape)
    s = 1j * freq
    fp1 = 1e6
    h_out = 1.0 / (1 + s / (2 * np.pi * fp1))
    a0 = 8000.0
    fp2, fp3 = 2e4, 5e6
    h_loop = a0 / ((1 + s / (2 * np.pi * fp2)) * (1 + s / (2 * np.pi * fp3)))

    x_sig = make_x_signal("frequency", "frequency", "Hz", freq)
    out_mag, out_phase = make_complex_pair("v(out)", "voltage", "V", h_out, freq)
    loop_mag, loop_phase = make_complex_pair("v(loop)", "voltage", "V", h_loop, freq)
    signals = [out_mag, out_phase, loop_mag, loop_phase]
    plot = make_plot("AC Analysis", "ac", True, x_sig, signals,
                      x_log_default=True, date="Mon Aug 10 00:00:02 2026")
    payload = make_payload(
        "c3c3c3c3c3c3c3c3", "c3c3c3c3", "sample_ac — loop gain + output pole",
        "samples/sample_ac.raw", [plot], 0,
    )
    write_sample("sample_ac.js", payload)


# ---------------------------------------------------------------------------
# sample_multi.js -- 3 plots: ac, tran, trailing constants; default_plot=1.
# ---------------------------------------------------------------------------

def gen_sample_multi():
    # Plot 0: AC
    freq = np.logspace(1, 8, 41)
    s = 1j * freq
    h = 1.0 / (1 + s / (2 * np.pi * 1e5))
    x_ac = make_x_signal("frequency", "frequency", "Hz", freq)
    mag, phase = make_complex_pair("v(out)", "voltage", "V", h, freq)
    plot_ac = make_plot("AC Analysis", "ac", True, x_ac, [mag, phase],
                         x_log_default=True, date="Mon Aug 10 00:00:03 2026")

    # Plot 1: transient (the default)
    n = 150
    t = np.linspace(0, 100e-9, n)
    vout = 1.2 * (1 - np.exp(-t / 10e-9))
    vin = np.full(n, 1.2)
    iload = -1e-3 * np.exp(-t / 10e-9)
    x_tr = make_x_signal("time", "time", "s", t)
    signals_tr = [
        make_signal("v(in)", "voltage", "V", vin, t),
        make_signal("v(out)", "voltage", "V", vout, t),
        make_signal("i(load)", "current", "A", iload, t),
    ]
    plot_tran = make_plot("Transient Analysis", "tran", False, x_tr, signals_tr,
                           date="Mon Aug 10 00:00:04 2026")

    # Plot 2: trailing constants block (e.g. an aborted .op left behind)
    x_const = make_x_signal("op", "notype", "", [0.0])
    signals_const = [
        make_signal("v(out)", "voltage", "V", [1.1987], [0.0]),
        make_signal("i(vdd)", "current", "A", [-1.02e-3], [0.0]),
        make_signal("gain", "notype", "", [42.7], [0.0]),
    ]
    plot_const = make_plot("Constants", "constants", False, x_const, signals_const,
                            date="Mon Aug 10 00:00:05 2026")

    payload = make_payload(
        "d4d4d4d4d4d4d4d4", "d4d4d4d4", "sample_multi — ac + tran + trailing constants",
        "samples/sample_multi.raw", [plot_ac, plot_tran, plot_const], 1,
    )
    write_sample("sample_multi.js", payload)


# ---------------------------------------------------------------------------
# sample_edge.js -- zero-point plot, NaN/Inf signal, f64-promoted signal,
# duplicate-name #2 signal, and a populated annotations block.
# ---------------------------------------------------------------------------

def gen_sample_edge():
    # Plot 0: a zero-point plot (e.g. an aborted run before the first point).
    x_zero = make_x_signal("time", "time", "s", np.array([], dtype="<f8"))
    sig_zero = make_signal("v(out)", "voltage", "V", np.array([], dtype="<f8"),
                            np.array([], dtype="<f8"))
    plot_zero = make_plot("Transient Analysis (aborted)", "tran", False,
                           x_zero, [sig_zero], date="Mon Aug 10 00:00:06 2026")

    # Plot 1: the real data, carrying every edge case.
    n = 120
    t = np.linspace(0, 60e-9, n)
    v_clk = 0.6 + 0.6 * np.sign(np.sin(2 * np.pi * 5e7 * t))

    v_fault = 0.9 + 0.1 * np.sin(2 * np.pi * 2e7 * t)
    v_fault[10:14] = np.nan
    v_fault[40] = np.inf
    v_fault[41] = -np.inf

    i_leak = 3e-40 * (1 + 0.1 * np.sin(2 * np.pi * 1e7 * t))  # forces f64 promotion

    v_out_a = 1.2 * (1 - np.exp(-t / 8e-9))
    v_out_b = 1.15 * (1 - np.exp(-t / 9e-9))  # duplicate name "v(out)" -> "v(out) #2"

    x_sig = make_x_signal("time", "time", "s", t)
    sig_clk = make_signal("v(clk)", "voltage", "V", v_clk, t)
    sig_fault = make_signal("v(fault)", "voltage", "V", v_fault, t)
    sig_leak = make_signal("i(leak)", "current", "A", i_leak, t)
    assert sig_leak["levels"][0]["dt"] == "f64"
    sig_out_1 = make_signal("v(out)", "voltage", "V", v_out_a, t)
    sig_out_2 = make_signal("v(out) #2", "voltage", "V", v_out_b, t, renamed=True)

    signals = [sig_clk, sig_fault, sig_leak, sig_out_1, sig_out_2]
    plot_main = make_plot("Transient Analysis", "tran", False, x_sig, signals,
                           date="Mon Aug 10 00:00:07 2026")

    annotations = {
        "markers": [
            {"axis": "x", "value": 30e-9, "label": "t_settle", "signal": None, "color": None},
            {"axis": "y", "value": 1.2, "label": "VDD", "signal": "v(out)", "color": "#ff8800"},
        ],
        "bands": [
            {"axis": "y", "lo": 1.08, "hi": 1.32, "label": "spec ±10%", "signal": "v(out)"},
        ],
        "notes": [
            {"signal": "v(fault)", "text": "contains NaN + Inf samples", "severity": "error"},
            {"signal": "i(leak)", "text": "promoted to f64 (sub-1e-37 magnitude)", "severity": "info"},
        ],
    }

    payload = make_payload(
        "e5e5e5e5e5e5e5e5", "e5e5e5e5", "sample_edge — zero-point / NaN-Inf / f64 / dup-name / annotations",
        "samples/sample_edge.raw", [plot_zero, plot_main], 1, annotations,
    )
    write_sample("sample_edge.js", payload)


# ---------------------------------------------------------------------------
# sample_hier.js -- v1.3 Feature A fixture (docs/superpowers/specs/
# 2026-08-11-waveform-explorer-v13-design.md §2.1). Two plots, names taken
# verbatim from the design doc's corpus; the *data* behind them is
# synthetic (the noise rawfile that produced the real names is not
# committed -- this sample is the durable artifact, per §7.2).
#
# Plot 0: the 72-signal ngspice `.noise` motivating case -- inoise_spectrum
# + onoise_spectrum (2 bare top-level leaves) plus 5 xmm groups
# (xmm1/xmm2 = nfet_01v8, xmm3/xmm4/xmm5 = pfet_01v8), each a 14-way
# fan-out: 1 bare "interior-node-is-also-a-signal" leaf
# (onoise.m.xdut.xmmN.msky130_fd_pr__{n,p}fet_01v8) that is ALSO the parent
# of 13 further leaves. The first 7 suffixes (.1overf/.id/.igb/.rbdb/.rd/
# .rg/.rs) are verbatim from the design doc; the doc elides the rest with
# "(...same 14-way fan-out)" so the remaining 6 are reconstructed following
# the same ngspice MOSFET-noise-generator naming pattern (not verbatim, but
# plausible): .ibs/.ibd/.igs/.igd/.rbpb/.rbsb. 2 + 5*14 = 72.
#
# Plot 1: the ngspice `.tran` device-internals/ammeter-wrapper set, verbatim
# from the design doc, MINUS the "v(m.xclkdel1.xmmn_inv1...#sbody)" line --
# dropped deliberately, not by oversight: keeping it would give the "m" root
# two children (xdut, xclkdel1), which blocks path compression from firing
# at "m" at all and defeats the specific compression case §7.2 asks to be
# exercised. Every other line is kept, including the ammeter (v.xdut...),
# `@m[...]`, `@i[...]`, `@r[...]` forms and the #body/#dbody suffixes.
# ---------------------------------------------------------------------------

NOISE_SUFFIXES = [
    ".1overf", ".id", ".igb", ".rbdb", ".rd", ".rg", ".rs",  # verbatim, §2.1
    ".ibs", ".ibd", ".igs", ".igd", ".rbpb", ".rbsb",         # reconstructed, same pattern
]
assert len(NOISE_SUFFIXES) == 13  # + 1 bare = 14-way fan-out per xmm, per §2.1


def gen_sample_hier():
    # ---- plot 0: the 72-signal noise plot ----
    n_noise = 40
    freq = np.logspace(1, 8, n_noise)
    x_noise = make_x_signal("frequency", "frequency", "Hz", freq)

    def noise_curve(scale, corner):
        return scale / (1.0 + (freq / corner))

    signals = [
        make_signal("inoise_spectrum", "notype", "", noise_curve(2e-9, 1e5), freq),
        make_signal("onoise_spectrum", "notype", "", noise_curve(4e-6, 2e5), freq),
    ]
    for xmm in range(1, 6):
        device = "msky130_fd_pr__nfet_01v8" if xmm <= 2 else "msky130_fd_pr__pfet_01v8"
        base = f"onoise.m.xdut.xmm{xmm}.{device}"
        signals.append(make_signal(base, "notype", "", noise_curve(1e-7 * xmm, 3e4 * xmm), freq))
        for j, suf in enumerate(NOISE_SUFFIXES):
            signals.append(make_signal(base + suf, "notype", "",
                                        noise_curve(1e-8 * (j + 1), 1e4 * (j + 2)), freq))
    assert len(signals) == 72, len(signals)
    plot_noise = make_plot("Noise Analysis", "noise", False, x_noise, signals,
                            date="Tue Aug 11 00:00:00 2026")

    # ---- plot 1: the tran / device-internals / ammeter-wrapper set ----
    n_tran = 30
    t = np.linspace(0, 50e-9, n_tran)

    def ramp(a, b, tau):
        return a + (b - a) * (1 - np.exp(-t / tau))

    x_tran = make_x_signal("time", "time", "s", t)
    tran_signals = [
        make_signal("v(xdut.vbias)", "voltage", "V", ramp(0, 0.6, 8e-9), t),
        make_signal("v(xdut.out)", "voltage", "V", ramp(0, 1.2, 6e-9), t),
        make_signal("v(vdd)", "voltage", "V", np.full(n_tran, 1.8), t),
        make_signal("v(xdut.xmmref.d_i)", "voltage", "V", ramp(0.2, 0.9, 5e-9), t),
        make_signal("v(xdut.xmmref.g_i)", "voltage", "V", ramp(0, 0.6, 4e-9), t),
        make_signal("v(xdut.xmmref.s_i)", "voltage", "V", np.zeros(n_tran), t),
        make_signal("v(xdut.xmmref.b_i)", "voltage", "V", np.zeros(n_tran), t),
        make_signal("v(m.xdut.xmmout.xmdev.msky130_fd_pr__nfet_01v8#dbody)", "voltage", "V",
                     ramp(0, 0.05, 3e-9), t),
        make_signal("v(m.xdut.xmmout.xmdev.msky130_fd_pr__nfet_01v8#body)", "voltage", "V",
                     np.zeros(n_tran), t),
        make_signal("i(vvdd)", "current", "A", -1.5e-3 * np.exp(-t / 12e-9), t),
        make_signal("i(v.xdut.xmmout.vamm_d)", "current", "A", 1.2e-3 * np.exp(-t / 10e-9), t),
        make_signal("i(v.xdut.xmmout.vamm_s)", "current", "A", -1.2e-3 * np.exp(-t / 10e-9), t),
        make_signal("i(@m.xdut.xmmref.xmdev.msky130_fd_pr__nfet_01v8[id])", "current", "A",
                     8e-4 * ramp(0, 1, 6e-9), t),
        make_signal("i(@m.xdut.xmmout.xmdev.msky130_fd_pr__nfet_01v8[ib])", "current", "A",
                     3e-6 * ramp(0, 1, 6e-9), t),
        make_signal("i(@i.xdut.iiref[current])", "current", "A", np.full(n_tran, 5e-6), t),
        make_signal("i(@r.xdut.rrload[i])", "current", "A", 6e-4 * ramp(0, 1, 7e-9), t),
    ]
    plot_tran = make_plot("Transient Analysis", "tran", False, x_tran, tran_signals,
                           date="Tue Aug 11 00:00:01 2026")

    payload = make_payload(
        "f6f6f6f6f6f6f6f6", "f6f6f6f6",
        "sample_hier — v1.3 hierarchical picker (72-signal noise + device-internal tran)",
        "samples/sample_hier.raw", [plot_noise, plot_tran], 0,
    )
    write_sample("sample_hier.js", payload)


def main():
    gen_sample_tran()
    gen_sample_tran_big()
    gen_sample_ac()
    gen_sample_multi()
    gen_sample_edge()
    gen_sample_hier()


if __name__ == "__main__":
    main()
