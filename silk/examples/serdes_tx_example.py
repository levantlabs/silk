"""SerDes transmitter: serializer, four-tap FFE, driver and channel.

Reference PRBS9 bits drive the serializer's parallel lanes in the short CMOS
and CML transistor runs. Each chain includes three retiming flip-flops,
a differential driver and a lumped RC channel. Ideal sources supply the clocks.
These runs exercise the composed circuits; they do not compare their eyes
with the behavioral model.

The longer behavioral tests apply a four-tap FIR to the reference bits.
They compare no FFE, one post-cursor tap and four taps through a channel with
two poles and an early reflection. Separate ideal-channel tests bypass the
channel. Metrics report geometric eye height and width, plus errors from a
known-bit slicer. The compared bit count excludes startup and alignment tails.
These finite, noiseless checks do not establish a link error rate.

The CMOS driver quantizes nominal tap magnitudes to unit counts 4/10/6/1.
Scaling inverter multiplicity and series resistance together preserves the
intended conductance ratios. CML mirror current ratios set the tap weights.
The CML twin uses a measured launch amplitude with nominal taps. Both twins
use an assumed edge rounded to the simulation grid and a behavioral channel.
The driver characterization fits realized taps and reports launch swing.

Run with ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.serdes_tx_example``.
Requires ngspice and sky130 models. Prints tap fits, spec results, eye metrics,
compared bit counts and short transistor-run swings. Saves eye plots,
capstone rawfiles and composition diagrams in the configured example output
directory. The behavioral testbenches can also run without a simulator.
"""
import os


import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.simulation import Transient
from silk.metrics import eye_diagram, peak_to_peak
from silk.signals import Signal

from silk.sources import prbs_bits
from silk.examples import cml_cells as cml
from silk.examples import example_output_dir
pdk = load_pdk('sky130')

_VDD = 1.8
F_BIT = 600e6
_UI = 1.0 / F_BIT


_SWING_CMOS_NOM = 1.8          # Differential launch span (V).
_SWING_CML_NOM = 1.602         # Differential launch p2-p98 span (V).
_TEDGE_NOM = 60e-12            # Assumed edge duration (s), rounded to the grid.


_FFE_DELAYS = (-1, 0, 1, 2)    # Precursor, main, post1 and post2 (UI).
_FFE_TAPS_NOM = (-0.385, 1.0, -0.610, 0.106)
_R_TERM = 100.0


_FP1_NOM = 60e6
_FP2_NOM = 220e6
_REFL_NOM = 0.35
_REFL_UI_NOM = 1.0            # Reflection arrives this many UI before the main path.

_PRBS_ORDER = 9
_N_OS = 40


def _nrz(levels, ui, dt, tedge):
    """Hold each level for one UI with an edge rounded to the sample grid.

    The ramp uses round(tedge/dt) samples, with a minimum of one. A one-sample
    ramp contains only the previous level, so the transition is a step."""
    n = len(levels)
    t = np.arange(n * int(round(ui / dt))) * dt
    y = np.empty_like(t)
    spu = int(round(ui / dt))
    # A sub-grid edge becomes a one-sample step.
    ramp = max(1, int(round(tedge / dt)))
    for k in range(n):
        lo = k * spu
        y[lo:lo + spu] = levels[k]
        if k > 0 and levels[k] != levels[k - 1] and lo - ramp >= 0:
            y[lo - ramp:lo] = np.linspace(levels[k - 1], levels[k], ramp)
    return t, y


def _one_pole(x, dt, fp):
    """First-order RC low-pass — the channel's building block."""
    a = dt / (1.0 / (2 * np.pi * fp) + dt)
    y = np.empty_like(x)
    v = x[0]
    for i in range(len(x)):
        v += a * (x[i] - v)
        y[i] = v
    return y


def _channel(x, dt, *, fp1=_FP1_NOM, fp2=_FP2_NOM, refl=_REFL_NOM,
             refl_ui=_REFL_UI_NOM):
    """Apply two low-pass poles and an early copy of their output.

    Set either pole to None to bypass it, and refl=0 to omit the reflection."""
    y = np.array(x, dtype=float)
    if fp1 is not None:
        y = _one_pole(y, dt, fp1)
    if fp2 is not None:
        y = _one_pole(y, dt, fp2)
    if refl:
        sh = int(round(refl_ui * _UI / dt))
        early = np.zeros_like(y)
        early[:-sh] = y[sh:]
        y = y + refl * early
    return y


_IDEAL_CHANNEL = dict(fp1=None, fp2=None, refl=0.0)


def _apply_ffe(nrz, taps, delays):
    """Apply out[n] = sum(taps[i] * nrz[n - delays[i]]).

    Negative delays select future bits; samples outside the record are zero."""
    out = np.zeros_like(nrz, dtype=float)
    n = len(nrz)
    for i in range(n):
        v = 0.0
        for c, d in zip(taps, delays):
            j = i - d
            if 0 <= j < n:
                v += c * nrz[j]
        out[i] = v
    return out


def channel_cursors(n_span=(-3, 4)):
    """Return isolated-pulse channel samples indexed relative to the main cursor."""
    dt = _UI / _N_OS
    pre, post = 8, 16
    levels = np.array([0.0] * pre + [1.0] + [0.0] * post)
    t, x = _nrz(levels, _UI, dt, _TEDGE_NOM)
    p = _channel(x, dt)
    best = (-1e30, pre, 0.5)
    for kk in range(pre - 1, pre + 4):
        for phi in np.linspace(0.05, 0.95, 60):
            v = float(np.interp((kk + phi) * _UI, t, p))
            if v > best[0]:
                best = (v, kk, phi)
    _, base, phase = best
    return {k: float(np.interp((base + k + phase) * _UI, t, p))
            for k in range(n_span[0], n_span[1])}


def solve_ffe_taps(delays, cur=None):
    """Closed-form zero-forcing TX-FFE. Main tap c0 = 1 (fixed); solve the other
    taps so the equalized neighbour cursors g[m] = sum_i c_i h[m - D_i] are nulled
    for the controlled positions. Returns the tap vector aligned to ``delays``."""
    if cur is None:
        cur = channel_cursors()
    other = [d for d in delays if d != 0]
    A = np.array([[cur.get(m - d, 0.0) for d in other] for m in other])
    rhs = -np.array([cur.get(m, 0.0) for m in other])
    sol = np.linalg.solve(A, rhs)
    it = iter(sol)
    return np.array([1.0 if d == 0 else next(it) for d in delays])


_CUR_NOM = channel_cursors()
_MODE_TAPS = {
    "off": (0.0, 1.0, 0.0, 0.0),
    "1tap": (0.0, 1.0, float(solve_ffe_taps((0, 1), _CUR_NOM)[1]), 0.0),
    "full": _FFE_TAPS_NOM,
}

_C0C1C2 = solve_ffe_taps((0, 1, 2), _CUR_NOM)
_MODE_TAPS_3TAP = (0.0, 1.0, float(_C0C1C2[1]), float(_C0C1C2[2]))


def tx_rx(bits, *, variant, ffe="full", ideal=False):
    """The behavioral TX chain -> RX waveform. ``ffe`` in {off, 1tap, full}.
    ``ideal=True`` bypasses the channel (poles + reflection) to a pure wire — the
    chain-sanity baseline. Returns (t, rx, ui)."""
    dt = _UI / _N_OS
    taps = _MODE_TAPS[ffe] if isinstance(ffe, str) else tuple(ffe)
    nrz = np.array([2.0 * b - 1.0 for b in bits])
    fir = _apply_ffe(nrz, taps, _FFE_DELAYS)
    swing = _SWING_CMOS_NOM if variant == "cmos" else _SWING_CML_NOM

    fir = fir / np.sum(np.abs(taps)) * (swing / 2.0)
    t, drv = _nrz(fir, _UI, dt, _TEDGE_NOM)


    rx = _channel(drv, dt, **(_IDEAL_CHANNEL if ideal else {}))
    return t, rx, _UI


def expected_launch_mv(variant, ffe):
    """Return the ideal-channel held-run differential level span in mV.

    Compute abs(sum(taps)) / sum(abs(taps)) times the launch swing."""
    swing = _SWING_CMOS_NOM if variant == "cmos" else _SWING_CML_NOM
    taps = _MODE_TAPS[ffe]
    return abs(sum(taps)) / np.sum(np.abs(taps)) * swing * 1e3


_FFE_MAIN_UNITS = 10
_R_U1 = 1000.0
_INV_WN_U, _INV_WP_U = 20.0, 40.0


def _ffe_unit_counts(taps=_FFE_TAPS_NOM, main_units=_FFE_MAIN_UNITS):
    """Quantize |taps| to integer unit-slice counts (main tap = ``main_units``)."""
    cmain = abs(taps[1])
    return [0 if c == 0 else max(1, int(round(abs(c) / cmain * main_units)))
            for c in taps]


def build_cmos_ffe_driver(*, taps=_FFE_TAPS_NOM, main_units=_FFE_MAIN_UNITS,
                          r_u1=_R_U1, wn=_INV_WN_U, wp=_INV_WP_U, l=0.15) -> Module:
    """Build a differential CMOS FFE driver with integer unit counts.

    Each tap uses inverter multiplicity N and series resistance r_u1/N.
    Scaling both preserves the intended conductance ratios. Inputs carry
    true bits on P and complements on N; crossed output legs set tap signs.
    Zero-weight taps have no output slices."""
    from silk.examples.stdcells import build_inverter
    from silk.elements import elements
    names = ("pre", "main", "post1", "post2")
    N = _ffe_unit_counts(taps, main_units)
    mod = Module("PDK_skywater130", example_output_dir(), "cmos_ffe.cir",
                 pdk_details=pdk,
                 module_name="cmos_ffe_driver")
    ports = {"VDD": Port.supply(), "VSS": Port.ground(),
             "OUTP": Port.analog_output(), "OUTN": Port.analog_output()}
    for nm in names:
        ports[f"{nm.upper()}P"] = Port.analog_input()
        ports[f"{nm.upper()}N"] = Port.analog_input()
    mod.define_module_ports(ports)
    for nm, c, ni in zip(names, taps, N):
        if ni == 0:
            continue


        # Separate cell names keep taps with different multiplicities distinct.
        seg = build_inverter(wn=wn, wp=wp, l=l, m=ni, name=f"cmos_seg_{nm}",
                             with_tests=False)
        mod.add_module(seg, f"seg_{nm}_p", nets={"VDD": "VDD", "VSS": "VSS",
                        "A": f"{nm.upper()}P", "Y": f"d{nm}p"})
        mod.add_module(seg, f"seg_{nm}_n", nets={"VDD": "VDD", "VSS": "VSS",
                        "A": f"{nm.upper()}N", "Y": f"d{nm}n"})
    with mod:
        for nm, c, ni in zip(names, taps, N):
            if ni == 0:
                continue
            r = r_u1 / ni


            # Crossing the output legs supplies the sign of a negative tap.
            hi_leg, lo_leg = ((f"d{nm}n", f"d{nm}p") if c >= 0
                              else (f"d{nm}p", f"d{nm}n"))
            elements.resistor(f"R_{nm}_p", p=hi_leg, n="OUTP", value=r)
            elements.resistor(f"R_{nm}_n", p=lo_leg, n="OUTN", value=r)
        elements.capacitor("C_padp", p="OUTP", n="VSS", value=50e-15)
        elements.capacitor("C_padn", p="OUTN", n="VSS", value=50e-15)
    return mod


def build_serdes_tx(*, variant="cmos") -> Module:
    """Build a serializer, retiming chain, differential FFE driver and RC channel."""
    from silk.examples import serializer_example as ser
    from silk.elements import elements
    mod = Module("PDK_skywater130", example_output_dir(), f"serdes_tx_{variant}.cir",
                 pdk_details=pdk, module_name=f"serdes_tx_{variant}")
    if variant == "cml":


        mod.define_module_ports({
            "VDD": Port.supply(), "VSS": Port.ground(), "IBIAS": Port.analog_input(),
            "CLKP": Port.analog_input(), "CLKN": Port.analog_input(),
            "LOADP": Port.analog_input(), "LOADN": Port.analog_input(),
            "PADP": Port.analog_output(), "PADN": Port.analog_output(),
            "RXP": Port.analog_output(), "RXN": Port.analog_output(),
            **{f"P{i}P": Port.analog_input() for i in range(ser.RATIO)},
            **{f"P{i}N": Port.analog_input() for i in range(ser.RATIO)}})
        smod = ser.build_serializer(variant="cml", ratio=ser.RATIO)


        snets = {"VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS", "CLKP": "CLKP",
                 "CLKN": "CLKN", "LOADP": "LOADP", "LOADN": "LOADN",
                 "SOUT": "sobs", "SOUTP": "soutp", "SOUTN": "soutn"}
        for i in range(ser.RATIO):
            snets[f"P{i}P"] = f"P{i}P"
            snets[f"P{i}N"] = f"P{i}N"
        mod.add_module(smod, "ser", nets=snets)


        # The unretimed output is one UI ahead of the main tap at d1.
        rt = cml.build_cml_dff()
        mod.add_module(rt, "rt1", nets={"VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
            "CLKP": "CLKP", "CLKN": "CLKN", "DP": "soutp", "DN": "soutn",
            "OUTP": "d1p", "OUTN": "d1n"})
        mod.add_module(rt, "rt2", nets={"VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
            "CLKP": "CLKP", "CLKN": "CLKN", "DP": "d1p", "DN": "d1n",
            "OUTP": "d2p", "OUTN": "d2n"})
        mod.add_module(rt, "rt3", nets={"VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
            "CLKP": "CLKP", "CLKN": "CLKN", "DP": "d2p", "DN": "d2n",
            "OUTP": "d3p", "OUTN": "d3n"})


        mod.add_module(cml.build_cml_ffe_driver(), "driver", nets={
            "VDD": "VDD", "VSS": "VSS", "IBIAS": "IBIAS",
            "PREP": "soutp", "PREN": "soutn", "MAINP": "d1p", "MAINN": "d1n",
            "POST1P": "d2p", "POST1N": "d2n", "POST2P": "d3p", "POST2N": "d3n",
            "OUTP": "PADP", "OUTN": "PADN"})
        with mod:


            elements.resistor("R_chp", p="PADP", n="RXP", value=50.0)
            elements.resistor("R_chn", p="PADN", n="RXN", value=50.0)
            elements.capacitor("C_chp", p="RXP", n="VSS", value=2e-12)
            elements.capacitor("C_chn", p="RXN", n="VSS", value=2e-12)
        return mod
    if variant == "cmos":
        mod.define_module_ports({
            "VDD": Port.supply(), "VSS": Port.ground(),
            "CLK": Port.analog_input(), "LOAD": Port.analog_input(),
            "PADP": Port.analog_output(), "PADN": Port.analog_output(),
            "RXP": Port.analog_output(), "RXN": Port.analog_output(),
            **{f"P{i}": Port.analog_input() for i in range(ser.RATIO)}})
        smod = ser.build_serializer(variant="cmos", ratio=ser.RATIO)
        snets = {"VDD": "VDD", "VSS": "VSS", "CLK": "CLK", "LOAD": "LOAD",
                 "SOUT": "sout"}
        for i in range(ser.RATIO):
            snets[f"P{i}"] = f"P{i}"
        mod.add_module(smod, "ser", nets=snets)


        from silk.examples.stdcells import build_dff, build_inverter
        dff = build_dff(with_tests=False)
        mod.add_module(dff, "rt1", nets={"VDD": "VDD", "VSS": "VSS", "D": "sout",
                        "CLK": "CLK", "CLRB": "VDD", "Q": "d1", "QB": "d1b"})
        mod.add_module(dff, "rt2", nets={"VDD": "VDD", "VSS": "VSS", "D": "d1",
                        "CLK": "CLK", "CLRB": "VDD", "Q": "d2", "QB": "d2b"})
        mod.add_module(dff, "rt3", nets={"VDD": "VDD", "VSS": "VSS", "D": "d2",
                        "CLK": "CLK", "CLRB": "VDD", "Q": "d3", "QB": "d3b"})
        inv_la = build_inverter(wn=20, wp=40, l=0.15, name="inv_la",
                                with_tests=False)
        mod.add_module(inv_la, "inv_soutb", nets={"VDD": "VDD", "VSS": "VSS",
                        "A": "sout", "Y": "soutb"})


        mod.add_module(build_cmos_ffe_driver(), "driver", nets={
            "VDD": "VDD", "VSS": "VSS",
            "MAINP": "d1", "MAINN": "d1b", "PREP": "sout", "PREN": "soutb",
            "POST1P": "d2", "POST1N": "d2b", "POST2P": "d3", "POST2N": "d3b",
            "OUTP": "PADP", "OUTN": "PADN"})
        with mod:


            elements.resistor("R_chp", p="PADP", n="RXP", value=50.0)
            elements.resistor("R_chn", p="PADN", n="RXN", value=50.0)
            elements.capacitor("C_chp", p="RXP", n="VSS", value=2e-12)
            elements.capacitor("C_chn", p="RXN", n="VSS", value=2e-12)
            elements.resistor("R_term", p="RXP", n="RXN", value=_R_TERM)
    else:
        raise ValueError(f"unknown serdes_tx variant {variant!r}")
    return mod


def _slicer_settings(t, rx, bits, ui, *, n_settle=20, n_phase=161, max_lag=4):
    """Find phase, threshold, polarity and latency for a known-bit slicer.

    Symbol separation selects the settings; eye_diagram supplies the
    separately reported geometric eye measurements."""
    phases = np.linspace(0.02, 0.98, n_phase)
    best = (-np.inf, 0.5, 0.0, True, 0)
    for lag in range(max_lag + 1):
        ks = np.arange(n_settle, len(bits) - 1 - lag)
        b = np.array([bits[k] for k in ks])
        mid = np.interp((ks + lag) * ui + 0.5 * ui, t, rx)
        upper_is_1 = mid[b == 1].mean() >= mid[b == 0].mean()
        for phase in phases:
            ys = np.interp((ks + lag) * ui + phase * ui, t, rx)
            g1, g0 = ys[b == 1], ys[b == 0]
            up, lo = (g1, g0) if upper_is_1 else (g0, g1)
            upper, lower = float(up.min()), float(lo.max())
            if upper - lower > best[0]:
                best = (upper - lower, phase, 0.5 * (upper + lower), upper_is_1, lag)
    return best[1:]


def _rx_eye(sig, bits, ui, *, n_settle=20, max_lag=4):
    """Measure the geometric eye over the record at the slicer threshold.

    Small positive gaps or NaN do not establish bit recovery."""
    _, threshold, _, _ = _slicer_settings(
        sig.x, sig.y, bits, ui, n_settle=n_settle, max_lag=max_lag)
    return eye_diagram(sig, period=ui, auto_center=True, threshold=threshold)


def _bit_errors(t, rx, bits, ui, phase, thr, upper_is_1, lag, *, n_settle=20):
    """Count the fewest errors over residual alignment lags after startup.

    Compare len(bits) - n_settle - lag - 12 bits for each alignment."""
    n = len(bits)
    ys = np.interp((np.arange(n) + lag) * ui + phase * ui, t, rx)
    rec = (ys > thr).astype(int)
    if not upper_is_1:
        rec = 1 - rec
    ref = np.array(bits)
    # Leave room for all residual alignment lags within the saved waveform.
    m = n - n_settle - lag - 12
    best = m + 1
    for extra in range(0, 8):
        e = int(np.sum(rec[n_settle + extra:n_settle + extra + m] !=
                       ref[n_settle:n_settle + m]))
        best = min(best, e)
    return best


def _attach_tx_test(mod, name, *, variant, ffe, bits, ideal=False):
    tb = mod.create_test(name)
    tb.add_module_under_test({"rx": "rx", "ffe_sel": "ffe_sel", "ch_sel": "ch_sel"})
    tb.add_source(sources.VoltageSource(
        "ffe_sel", positive="ffe_sel", negative="0", dc=("off", "1tap", "full").index(ffe)))
    tb.add_source(sources.VoltageSource(
        "ch_sel", positive="ch_sel", negative="0", dc=int(ideal)))
    dt = _UI / _N_OS
    tb.add_sim_type(Transient(step=dt, stop=(len(bits) * _N_OS - 1) * dt))
    rx = tb.probe("rx")

    def bit_errors(sig):
        return float(_bit_errors(sig.x, sig.y, bits, _UI,
                                 *_slicer_settings(sig.x, sig.y, bits, _UI)))

    def height_mv(sig):
        return _rx_eye(sig, bits, _UI).height * 1e3

    def width_ps(sig):
        return _rx_eye(sig, bits, _UI).width * 1e12

    tb.add_metric("bit_errors", bit_errors, sig=rx)
    tb.add_metric("compared_bits", lambda sig: len(bits) - 20 - 12 -
                  _slicer_settings(sig.x, sig.y, bits, _UI)[-1], sig=rx)
    tb.add_metric("eye_h_mv", height_mv, sig=rx)
    tb.add_metric("eye_w_ps", width_ps, sig=rx)
    return tb


_EYE_H_MIN = {"cmos": {"1tap": 40.0, "full": 100.0},
              "cml": {"1tap": 31.0, "full": 62.0}}
_EYE_W_MIN = {"cmos": {"1tap": 350.0, "full": 700.0},
              "cml": {"1tap": 300.0, "full": 650.0}}


def build_serdes_tx_behavioral(*, variant, bits) -> Module:
    """Build a behavioral TX with ideal-channel and lossy-channel testbenches.

    run_all_tests runs the twin; reconcile_specs checks its measurements."""
    mod = Module("PDK_skywater130", example_output_dir(), f"tx_beh_{variant}.cir",
                 pdk_details=pdk, module_name=f"tx_beh_{variant}")
    mod.define_module_ports({"rx": Port.analog_output(),
                             "ffe_sel": Port.analog_input(),
                             "ch_sel": Port.analog_input()})

    def model(ports, t, params):
        ffe = ("off", "1tap", "full")[int(round(np.asarray(ports.ffe_sel.v)[0]))]
        ideal = bool(round(np.asarray(ports.ch_sel.v)[0]))
        time, rx, _ = tx_rx(bits, variant=variant, ffe=ffe, ideal=ideal)
        ports.rx.v = np.interp(t, time, rx)

    mod.set_behavioral_model(model, execution='vector')
    mod.simulator = "behavioral"


    ui_ps = _UI * 1e12
    for ffe, name in (("off", "ideal_off"), ("full", "ideal_full")):
        tb = _attach_tx_test(mod, name, variant=variant, ffe=ffe, bits=bits, ideal=True)
        if ffe == "off":
            exp = expected_launch_mv(variant, "off")
            tb.add_spec(f"{variant} ideal-off: 0 bit errors (clean chain)",
                        "bit_errors", "==", 0.0)
            tb.add_spec(f"{variant} ideal-off: eye height > 90% launch "
                        f"({0.9*exp:.0f} mV)", "eye_h_mv", ">", 0.9 * exp)
            tb.add_spec(f"{variant} ideal-off: eye height < 110% launch "
                        f"({1.1*exp:.0f} mV)", "eye_h_mv", "<", 1.1 * exp)
            tb.add_spec(f"{variant} ideal-off: eye width > 0.9 UI "
                        f"({0.9*ui_ps:.0f} ps)", "eye_w_ps", ">", 0.9 * ui_ps)
        else:
            tb.add_spec(f"{variant} ideal-full: over-equalization errors > 0",
                        "bit_errors", ">", 0.0)


    for ffe in ("off", "1tap", "full"):
        tb = _attach_tx_test(mod, ffe, variant=variant, ffe=ffe, bits=bits)
        if ffe == "full":
            tb.add_spec(f"{variant} full: 0 bit errors",
                        "bit_errors", "==", 0.0)
            tb.add_spec(f"{variant} full: eye height > {_EYE_H_MIN[variant]['full']:.0f} mV",
                        "eye_h_mv", ">", _EYE_H_MIN[variant]["full"])
            tb.add_spec(f"{variant} full: eye width > {_EYE_W_MIN[variant]['full']:.0f} ps",
                        "eye_w_ps", ">", _EYE_W_MIN[variant]["full"])
        elif ffe == "1tap":

            tb.add_spec(f"{variant} 1tap: eye opens (>{_EYE_H_MIN[variant]['1tap']:.0f} mV)",
                        "eye_h_mv", ">", _EYE_H_MIN[variant]["1tap"])
            tb.add_spec(f"{variant} 1tap: eye width > {_EYE_W_MIN[variant]['1tap']:.0f} ps",
                        "eye_w_ps", ">", _EYE_W_MIN[variant]["1tap"])
        else:
            tb.add_spec(f"{variant} off: slicer errors > 0", "bit_errors", ">", 0.0)
    return mod


def ffe_tap_report():
    """Print channel cursors, solved taps and CMOS eye dimensions by tap set.

    Return the cursor, tap and eye data used by the plot."""
    cur = _CUR_NOM
    opt4 = solve_ffe_taps(_FFE_DELAYS, cur)
    print("  channel cursors h[k] (isolated-pulse response, main = h[0]):")
    print("   ", "  ".join(f"h[{k:+d}]={cur[k]:+.3f}" for k in range(-2, 4)))
    print(f"    precursor h[-1] = {100*cur[-1]/cur[0]:.0f}% of main "
          f"(the reflection's non-minimum-phase cursor)")
    print("\n  zero-forcing taps (main c0 = 1, closed-form):")
    print(f"    optimal 4-tap [c-1,c0,c1,c2] = "
          f"[{opt4[0]:+.3f}, {opt4[1]:+.3f}, {opt4[2]:+.3f}, {opt4[3]:+.3f}]")
    print(f"    shipped 4-tap (ZF)           = "
          f"[{_FFE_TAPS_NOM[0]:+.3f}, {_FFE_TAPS_NOM[1]:+.3f}, "
          f"{_FFE_TAPS_NOM[2]:+.3f}, {_FFE_TAPS_NOM[3]:+.3f}]  "
          f"(CMOS unit counts {tuple(_ffe_unit_counts())}, realized ~ ZF)")

    bits = prbs_bits(_PRBS_ORDER, 2 ** _PRBS_ORDER - 1, seed=0)
    rows = {}
    print("\n  CMOS geometric eye vs tap set:")
    print(f"    {'config':<22}{'eye_h(mV)':>10}{'eye_w(ps)':>11}{'area':>9}")
    for lab, taps in (("off (no FFE)", _MODE_TAPS["off"]),
                      ("1-tap (post1)", _MODE_TAPS["1tap"]),
                      ("+c2 (post1+post2)", _MODE_TAPS_3TAP),
                      ("full [c-1..c2]", _MODE_TAPS["full"])):
        t, rx, ui = tx_rx(bits, variant="cmos", ffe=taps)
        eye = _rx_eye(Signal("rx", "voltage", "time", "V", t, rx), bits, ui)
        h, w = eye.height * 1e3, eye.width * 1e12
        rows[lab] = (h, w)
        print(f"    {lab:<22}{h:>10.0f}{w:>11.0f}{h*w/1e3:>9.0f}")
    print("    Post-cursors open the eye; geometric width decreases from +c2 to full. "
          "Crossing-centred geometry does not establish bit recovery.")
    return {"cursors": cur, "opt4": opt4, "rows": rows}


def _run_cmos_ffe_driver(bits, *, tbit=None, t0=None):
    """Drive build_cmos_ffe_driver's 4 differential tap inputs with the PRBS at the
    _FFE_SHIFT delays (each tap: true bit on P, complement on N); return
    (t, launch = V(OUTP)-V(OUTN))."""
    from silk.examples import cml_cells as cml
    tbit = cml._FFE_TBIT if tbit is None else tbit
    t0 = cml._FFE_T0 if t0 is None else t0
    names = ("pre", "main", "post1", "post2")
    N = len(bits) - 4
    top = Module("PDK_skywater130", example_output_dir(), "cmos_ffe_tb.cir", pdk_details=pdk)
    nets = {"VDD": "VDD", "VSS": "0", "OUTP": "outp", "OUTN": "outn"}
    for nm in names:
        nets[f"{nm.upper()}P"] = f"{nm}p"
        nets[f"{nm.upper()}N"] = f"{nm}n"
    top.add_module(build_cmos_ffe_driver(), "drv", nets=nets)

    def lvl(b, invert):
        return (0.0 if b else _VDD) if invert else (_VDD if b else 0.0)

    def pwl(sh, invert):
        b0 = bits[sh] if 0 <= sh < len(bits) else 0
        p = sources.PWL().add(0, lvl(b0, invert))
        for k in range(1, N + 4):
            pv, idx = k - 1 + sh, k + sh
            prev = bits[pv] if 0 <= pv < len(bits) else 0
            cur = bits[idx] if 0 <= idx < len(bits) else 0
            tk = t0 + k * tbit
            p.add(tk - 0.4 * tbit, lvl(prev, invert)).add(tk - 0.35 * tbit, lvl(cur, invert))
        p.add((N + 5) * tbit + t0, lvl(0, invert))
        return p
    with top:
        sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD)
        for nm in names:
            sh = cml._FFE_SHIFT[nm]
            sources.VoltageSource(f"v{nm}p", positive=f"{nm}p", negative="0",
                                  pwl=pwl(sh, False))
            sources.VoltageSource(f"v{nm}n", positive=f"{nm}n", negative="0",
                                  pwl=pwl(sh, True))
    top.add_sim_type(Transient(step=tbit / 60, stop=t0 + (N + 4) * tbit,
                               maxstep=tbit / 30))
    top.simulate(corner="tt")
    p, n = top.load_signal("outp"), top.load_signal("outn")
    t = np.asarray(p.x, float)
    launch = (np.asarray(p.y, float) -
              np.interp(t, np.asarray(n.x, float), np.asarray(n.y, float)))
    return t, launch


def characterize_cmos_ffe_driver(bits=None):
    """Fit tap ratios and measure differential launch p2-p98 swing over PRBS9."""
    from silk.examples import cml_cells as cml
    if bits is None:
        bits = cml._prbs9(160)
    t, launch = _run_cmos_ffe_driver(bits)
    taps, main_v, r2 = cml.fit_ffe_taps(t, launch, bits)
    pkpk = float(np.percentile(launch, 98) - np.percentile(launch, 2))
    return {"taps": taps, "main_mv": main_v * 1e3, "pkpk_mv": pkpk * 1e3, "r2": r2}


def spice_capstone_cmos(*, n_words=3):
    """Exercise the composed transistor CMOS TX through the capstone RC channel.

    Return time and V(RXP)-V(RXN) over a few words."""
    from silk.examples import serializer_example as ser
    tx = build_serdes_tx(variant="cmos")
    top = Module("PDK_skywater130", example_output_dir(), "tx_cap.cir", pdk_details=pdk)
    nets = {"VDD": "VDD", "VSS": "0", "CLK": "clk", "LOAD": "load",
            "PADP": "padp", "PADN": "padn", "RXP": "rxp", "RXN": "rxn"}
    for i in range(ser.RATIO):
        nets[f"P{i}"] = f"p{i}"
    top.add_module(tx, "tx", nets=nets)
    tb = ser._T_BIT
    n_clocks = ser.RATIO * n_words
    bits = prbs_bits(ser._PRBS_ORDER, n_clocks + ser.RATIO, seed=0)
    lanes = ser._lane_bits(bits, ser.RATIO, n_words)
    with top:
        sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD)
        sources.VoltageSource("clk", positive="clk", negative="0", dc=0,
            pulse=sources.Pulse(v1=0, v2=_VDD, td=ser._T_START, tr=50e-12,
                                tf=50e-12, pw=tb / 2, per=tb))
        sources.VoltageSource("load", positive="load", negative="0", dc=0,
                              pulse=sources.Pulse(v1=0, v2=_VDD,
                                  td=ser._T_START - 0.3 * tb,
                                  tr=0.05 * tb, tf=0.05 * tb, pw=0.95 * tb,
                                  per=ser.RATIO * tb, np=n_words))
        for i in range(ser.RATIO):
            sources.VoltageSource(f"vp{i}", positive=f"p{i}", negative="0",
                                  waveform=sources.BitPattern(lanes[i], v1=0, v2=_VDD,
                                      ui=ser.RATIO * tb,
                                      td=ser._T_START - 0.4 * tb,
                                      tr=0.05 * tb, tf=0.05 * tb))
    stop = ser._T_START + (n_clocks + 4) * tb
    top.add_sim_type(Transient(step=tb / 60, stop=stop, maxstep=tb / 30))
    top.simulate(corner="tt")
    rxp = top.load_signal("rxp")
    rxn = top.load_signal("rxn")
    diff = rxp - rxn
    return np.asarray(diff.x, dtype=float), np.asarray(diff.y, dtype=float)


def spice_capstone_cml(*, n_words=3):
    """Exercise the composed transistor CML TX through the capstone RC channel.

    Return time and V(RXP)-V(RXN). Ideal sources drive the parallel lanes,
    clock and LOAD; one reference biases the CML chain."""
    from silk.examples import serializer_example as ser
    tx = build_serdes_tx(variant="cml")
    top = Module("PDK_skywater130", example_output_dir(), "tx_cml_cap.cir",
                 pdk_details=pdk)
    top.add_module(cml.build_cml_bias(imul=ser._CML_IMUL), "bias",
                    nets={"VSS": "0", "IBIAS": "ibias"})
    nets = {"VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
            "CLKP": "clkp", "CLKN": "clkn", "LOADP": "loadp", "LOADN": "loadn",
            "PADP": "padp", "PADN": "padn", "RXP": "rxp", "RXN": "rxn"}
    for i in range(ser.RATIO):
        nets[f"P{i}P"] = f"p{i}p"
        nets[f"P{i}N"] = f"p{i}n"
    top.add_module(tx, "tx", nets=nets)

    tb = ser._T_BIT
    n_clocks = ser.RATIO * n_words
    bits = prbs_bits(_PRBS_ORDER, n_clocks + ser.RATIO, seed=0)
    lanes = ser._lane_bits(bits, ser.RATIO, n_words)

    with top:
        sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD)
        sources.CurrentSource("iref", positive="VDD", negative="ibias", dc=ser._CML_IREF)
        _te = min(50e-12, tb / 10)
        sources.VoltageSource("clkp", positive="clkp", negative="0",
            pulse=sources.Pulse(v1=ser._CML_LO, v2=ser._CML_HI, td=ser._T_START,
                                tr=_te, tf=_te, pw=tb / 2, per=tb))
        sources.VoltageSource("clkn", positive="clkn", negative="0",
            pulse=sources.Pulse(v1=ser._CML_HI, v2=ser._CML_LO, td=ser._T_START,
                                tr=_te, tf=_te, pw=tb / 2, per=tb))
        sources.VoltageSource("loadp", positive="loadp", negative="0",
                              pulse=sources.Pulse(v1=ser._SEL_LO, v2=ser._SEL_HI,
                                  td=ser._T_START - 0.3 * tb,
                                  tr=0.05 * tb, tf=0.05 * tb, pw=0.95 * tb,
                                  per=ser.RATIO * tb, np=n_words))
        sources.VoltageSource("loadn", positive="loadn", negative="0",
                              pulse=sources.Pulse(v1=ser._SEL_HI, v2=ser._SEL_LO,
                                  td=ser._T_START - 0.3 * tb,
                                  tr=0.05 * tb, tf=0.05 * tb, pw=0.95 * tb,
                                  per=ser.RATIO * tb, np=n_words))
        for i in range(ser.RATIO):
            sources.VoltageSource(f"vp{i}p", positive=f"p{i}p", negative="0",
                 waveform=sources.BitPattern(lanes[i], v1=ser._CML_LO, v2=ser._CML_HI,
                     ui=ser.RATIO * tb, td=ser._T_START - 0.4 * tb,
                     tr=0.05 * tb, tf=0.05 * tb))
            sources.VoltageSource(f"vp{i}n", positive=f"p{i}n", negative="0",
                 waveform=sources.BitPattern(lanes[i], v1=ser._CML_HI, v2=ser._CML_LO,
                     ui=ser.RATIO * tb, td=ser._T_START - 0.4 * tb,
                     tr=0.05 * tb, tf=0.05 * tb))
    stop = ser._T_START + (n_clocks + 4) * tb
    top.add_sim_type(Transient(step=tb / 60, stop=stop, maxstep=tb / 30))
    top.simulate(corner="tt")
    rxp = top.load_signal("rxp")
    rxn = top.load_signal("rxn")
    diff = rxp - rxn
    return np.asarray(diff.x, dtype=float), np.asarray(diff.y, dtype=float)


def composition_diagrams(out_dir):
    """Save composition PNGs for both variants via silk.circuit.composition."""
    from silk.circuit.composition import module_graph, save_graph_png
    paths = {}
    for variant in ("cmos", "cml"):
        tx = build_serdes_tx(variant=variant)
        g = module_graph(tx, depth=1)
        p = os.path.join(out_dir, f"composition_serdes_tx_{variant}.png")
        save_graph_png(g, p)
        paths[variant] = p
    return paths


def save_eye_plot(results, tap_data, path, *, bits):
    """Plot eyes for both variants with no FFE, one tap and four taps.

    Additional panels show channel cursors, solved taps and eye dimensions.
    Passing eyes carry markers at the slicer phase and threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95], hspace=0.42,
                          wspace=0.28)
    modes = ["off", "1tap", "full"]
    mode_lab = {"off": "no FFE", "1tap": "1-tap", "full": "full 4-tap"}
    row_ax = {}
    for r, var in enumerate(("cmos", "cml")):
        for c, mode in enumerate(modes):
            ax = fig.add_subplot(gs[r, c], sharey=row_ax.get(var))
            row_ax.setdefault(var, ax)
            res = results[(var, mode)]
            sig = res.sim_results.load_signal("rx")
            t, rx, ui = np.asarray(sig.x), np.asarray(sig.y), _UI
            phase, thr, _, _ = _slicer_settings(t, rx, bits, ui)
            height = res.metrics["eye_h_mv"]
            width = res.metrics["eye_w_ps"]
            errors = int(res.metrics["bit_errors"])
            for k in range(20, len(bits) - 1):
                m = (t >= (k - 0.5) * ui) & (t <= (k + 0.5) * ui)
                if np.any(m):
                    Signal("eye", "voltage", ax.get_xlabel(), ax.get_ylabel(),
                           (t[m] - k * ui) * 1e9, rx[m]).plot(ax=ax,
                            color=("#0072B2" if bits[k] else "#D55E00"),
                            lw=0.3, alpha=0.3)
            if errors == 0 and height > 0:
                ax.axvline(((phase + 0.5) % 1.0 - 0.5) * ui * 1e9, color="#009E73", ls=":",
                           lw=1.4, label="sample")
                ax.axhline(thr, color="#5a5a6e", ls="--", lw=1.0,
                           label="threshold")
                ax.legend(frameon=False, fontsize=7, loc="upper right")
            ax.set_title(f"{var.upper()} — {mode_lab[mode]}: "
                         f"{height:.0f} mV / {width:.0f} ps, "
                         f"{errors} err" +
                         ("  (closed)" if height <= 0 else ""), fontsize=9)
            ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
            ax.set_axisbelow(True)
            if r == 1:
                ax.set_xlabel("time within UI (ns)", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{var.upper()} RX diff (V)", fontsize=8)


    axc = fig.add_subplot(gs[2, 0])
    cur = tap_data["cursors"]
    ks = list(range(-2, 4))
    axc.stem([k - 0.08 for k in ks], [cur[k] for k in ks], linefmt="#888",
             markerfmt="o", basefmt=" ", label="channel cursor h[k]")
    opt4 = tap_data["opt4"]
    tap_at = {d: opt4[i] for i, d in enumerate(_FFE_DELAYS)}
    axc.stem([k + 0.08 for k in ks], [tap_at.get(k, 0.0) for k in ks],
             linefmt="#D55E00", markerfmt="s", basefmt=" ", label="ZF tap c[k]")
    axc.axhline(0, color="#333", lw=0.7)
    axc.set_title("channel cursors vs ZF taps (c-1 cancels the precursor)",
                  fontsize=9)
    axc.set_xlabel("UI index k (0 = main)", fontsize=8)
    axc.legend(frameon=False, fontsize=7)
    axc.grid(True, axis="y", color="#d5d5dd", lw=0.6, alpha=0.6)


    rows = tap_data["rows"]
    labs = list(rows.keys())
    axh = fig.add_subplot(gs[2, 1])
    axh.bar(range(len(labs)), [max(0.0, rows[l][0]) for l in labs],
            color="#0072B2")
    axh.set_xticks(range(len(labs)))
    axh.set_xticklabels(["off", "1tap", "+c2", "full"], fontsize=8)
    axh.set_title("CMOS eye HEIGHT per tap set (mV)", fontsize=9)
    axh.grid(True, axis="y", color="#d5d5dd", lw=0.6, alpha=0.6)
    axh.set_axisbelow(True)
    axw = fig.add_subplot(gs[2, 2])
    axw.bar(range(len(labs)), [rows[l][1] for l in labs], color="#009E73")
    axw.set_xticks(range(len(labs)))
    axw.set_xticklabels(["off", "1tap", "+c2", "full"], fontsize=8)
    axw.set_title("CMOS geometric eye WIDTH per tap set (ps)",
                  fontsize=9)
    axw.grid(True, axis="y", color="#d5d5dd", lw=0.6, alpha=0.6)
    axw.set_axisbelow(True)

    fig.suptitle("SerDes TX differential eye at RX (V(RXP)-V(RXN)) — 4-tap FFE "
                 "over a 2-pole + reflection channel (CML launch calibrated to 1.602 V)", fontsize=12)
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return path


def save_ideal_eye_plot(results, path, *, bits):
    """Plot ideal-channel eyes with and without the four-tap FFE.

    Draw slicer settings on passing eyes; each panel has its own voltage axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    cols = [("ideal_off", "no FFE"), ("ideal_full", "full 4-tap")]
    for r, var in enumerate(("cmos", "cml")):
        for c, (mode, mlab) in enumerate(cols):
            ax = axes[r, c]
            res = results[(var, mode)]
            sig = res.sim_results.load_signal("rx")
            t, rx, ui = np.asarray(sig.x), np.asarray(sig.y), _UI
            phase, thr, _, _ = _slicer_settings(t, rx, bits, ui)
            height = res.metrics["eye_h_mv"]
            width = res.metrics["eye_w_ps"]
            errors = int(res.metrics["bit_errors"])
            for k in range(20, len(bits) - 1):
                m = (t >= (k - 0.5) * ui) & (t <= (k + 0.5) * ui)
                if np.any(m):
                    Signal("eye", "voltage", ax.get_xlabel(), ax.get_ylabel(),
                           (t[m] - k * ui) * 1e9, rx[m]).plot(ax=ax,
                            color=("#0072B2" if bits[k] else "#D55E00"),
                            lw=0.3, alpha=0.3)
            if errors == 0 and height > 0:
                ax.axvline(((phase + 0.5) % 1.0 - 0.5) * ui * 1e9, color="#009E73", ls=":",
                           lw=1.4, label="sample")
                ax.axhline(thr, color="#5a5a6e", ls="--", lw=1.0,
                           label="threshold")
                ax.legend(frameon=False, fontsize=7, loc="upper right")
            if mode == "ideal_off":
                ttl = (f"{var.upper()} — ideal ch, {mlab}: {height:.0f} mV "
                       f"(= full launch) / {width:.0f} ps, {errors} err")
            else:
                ttl = (f"{var.upper()} — ideal ch, {mlab}: over-equalized, "
                       f"{height:.0f} mV, {errors} err  (closed)")
            ax.set_title(ttl, fontsize=9)
            ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
            ax.set_axisbelow(True)
            if r == 1:
                ax.set_xlabel("time within UI (ns)", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{var.upper()} RX diff (V)", fontsize=8)
    fig.suptitle("SerDes TX — Ideal-channel baseline eyes (pure wire): "
                 "CML launch budget 1.602 V; off = baseline, full = over-equalized",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()
    bits = prbs_bits(_PRBS_ORDER, 2 ** _PRBS_ORDER - 1, seed=0)

    print("=" * 72)
    print(f"SerDes TX — full chain, behavioral twins, PRBS{_PRBS_ORDER} "
          f"({len(bits)} bits) @ {F_BIT/1e6:.0f} Mb/s")
    print("=" * 72)


    print("-" * 72)
    print("FFE tap optimization (zero-forcing against the 2-pole+reflection channel)")
    print("-" * 72)
    tap_data = ffe_tap_report()


    results = {}
    for var in ("cmos", "cml"):
        print("-" * 72)
        print(f"behavioral TX functional tests — {var.upper()} (slicer bit-recovery)")
        print("-" * 72)
        mod = build_serdes_tx_behavioral(variant=var, bits=bits)
        res_dict = mod.run_all_tests()
        print(mod.reconcile_specs().summary())
        for mode in ("ideal_off", "ideal_full", "off", "1tap", "full"):
            results[(var, mode)] = res_dict[f"tx_beh_{var}.{mode}"]


    print("\n  Ideal-channel baseline (pure wire):")
    print(f"    {'variant':<7}{'FFE':<6}{'eye_h(mV)':>10}{'eye_w(ps)':>11}"
          f"{'bit err':>9}{'compared':>10}   {'expected'}")
    for var in ("cmos", "cml"):
        for mode, lab, exp in (("ideal_off", "off",
                                f"~{expected_launch_mv(var, 'off'):.0f} mV diff swing"),
                               ("ideal_full", "full",
                                "over-equalized")):
            r = results[(var, mode)]
            print(f"    {var.upper():<7}{lab:<6}{r.metrics['eye_h_mv']:>10.0f}"
                  f"{r.metrics['eye_w_ps']:>11.0f}{int(r.metrics['bit_errors']):>9d}{int(r.metrics['compared_bits']):>10d}   {exp}")

    cm1, c0, c1, c2 = _FFE_TAPS_NOM
    nonmain = abs(cm1) + abs(c1) + abs(c2)
    worst = c0 - nonmain
    print(f"    why FFE-full closes on a clean wire (shipped taps): "
          f"|c-1|+|c1|+|c2| = {abs(cm1):.3f}+{abs(c1):.3f}+{abs(c2):.3f} = "
          f"{nonmain:.2f}  >  main c0 = {c0:.2f}")
    print(f"      -> worst-case bit (pattern 0111): c0 - (|c-1|+|c1|+|c2|) = "
          f"{c0:.2f} - {nonmain:.2f} = {worst:+.2f} (normalized) -> sign reversal, "
          f"so the eye closes.")
    print("    FFE off passes the sampled bit check. Full FFE over-equalizes "
          "the ideal channel; its taps were chosen for the lossy channel.")

    print("\n  Lossy channel (2 poles + reflection) — eye and bit errors "
          "(PRBS%d, %d bits):" % (_PRBS_ORDER, len(bits)))
    print(f"    {'variant':<7}{'FFE':<7}{'eye_h(mV)':>10}{'eye_w(ps)':>11}"
          f"{'bit err':>9}{'compared':>10}")
    for var in ("cmos", "cml"):
        for mode, lab in (("off", "off"), ("1tap", "1-tap"), ("full", "full")):
            r = results[(var, mode)]
            print(f"    {var.upper():<7}{lab:<7}{r.metrics['eye_h_mv']:>10.0f}"
                  f"{r.metrics['eye_w_ps']:>11.0f}{int(r.metrics['bit_errors']):>9d}{int(r.metrics['compared_bits']):>10d}")
    print("\n  Geometric eye heights use the differential RX waveform. "
          "Bit errors are counted over the listed comparison windows. "
          f"The CML twin uses a {_SWING_CML_NOM*1e3:.0f} mV launch budget "
          "with nominal taps. Channel and grid-quantized edges are behavioral assumptions.")

    eye_path = save_eye_plot(results, tap_data, os.path.join(OUT, "serdes_tx_eye.png"),
                             bits=bits)
    ideal_path = save_ideal_eye_plot(results, os.path.join(OUT, "serdes_tx_ideal_eye.png"),
                                     bits=bits)
    print(f"\n  saved {eye_path}")
    print(f"  saved {ideal_path}")


    print("\n  CMOS FFE driver — realized taps measured from the netlist "
          "(unit-weighted SST DAC):")
    cd = characterize_cmos_ffe_driver()
    ct = cd["taps"]
    print(f"    realized [{ct[0]:+.3f}, {ct[1]:+.3f}, {ct[2]:+.3f}, {ct[3]:+.3f}]"
          f"  vs ZF [{_FFE_TAPS_NOM[0]:+.3f}, {_FFE_TAPS_NOM[1]:+.3f}, "
          f"{_FFE_TAPS_NOM[2]:+.3f}, {_FFE_TAPS_NOM[3]:+.3f}]  "
          f"(R²={cd['r2']:.3f}, unit counts {tuple(_ffe_unit_counts())})")
    print("    Scaling inverter multiplicity and series resistance together "
          "preserves the intended conductance ratios.")


    print("\n  SPICE capstones (full transistor chains, 3 words each):")
    tcap, rxcap = spice_capstone_cmos()
    diff = Signal("rx", "voltage", "time", "V", tcap, rxcap)
    rng = peak_to_peak(diff, start=tcap[int(0.35 * len(rxcap))]) * 1e3
    print(f"    CMOS: RX {rng:.0f} mV pk-pk differential over the selected short-run "
          "window. This exercises the composed TX through the capstone RC channel.")
    tcml, rxcml = spice_capstone_cml()
    scml = rxcml[int(0.45 * len(rxcml)):]
    rngc = (np.percentile(scml, 98) - np.percentile(scml, 2)) * 1e3
    print(f"    CML: RX {rngc:.0f} mV differential p2-p98 over the selected "
          "short-run window, including LOAD and pipeline-fill transients. "
          "This exercises the composed TX through the capstone RC channel.")


    cpaths = composition_diagrams(OUT)
    print(f"    composition PNGs: {os.path.basename(cpaths['cmos'])}, "
          f"{os.path.basename(cpaths['cml'])}")
