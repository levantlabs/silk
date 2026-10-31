"""Differential CMOS driver with a one-tap feed-forward equalizer (FFE).

Each output leg combines a main inverter and a tap inverter through series
resistors. The tap input is the delayed, inverted bitstream. During repeated
bits its output opposes the main slice; at a transition the outputs agree.
This reduces the held level relative to the transition level.

Stimulus supplies PRBS7 bits and a delayed, inverted copy. Serializer and
clock recovery are not modeled. Ideal R/C sets the channel's dominant pole;
sky130 transistors implement the driver. The channel is a lumped low-pass
without a far-end termination, not a transmission-line model.

The edge test measures differential driver swing and single-ended PAD rise
and fall times using an isolated pulse. The two link tests use the same
hardware and bits. With FFE off, both slices carry the same bit. With FFE on,
the tap carries the delayed, inverted bit. Each test measures geometric eye
height and width at the differential receiver. An unavailable eye measurement
does not establish bit recovery or link operation.

Run with ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.serdes_tx_driver_example``.
Requires ngspice and sky130 models. Prints spec results, swing, edge times,
channel pole and eye measurements. Saves ``serdes_tx_driver_eye.png`` in the
configured example output directory.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.metrics import (eye_diagram, eye_height, eye_width, fall_time,
                          rise_time, value_at)
from silk.simulation import Transient
from silk.signals import Signal

from silk.examples.ring_oscillator_example import build_inverter
from silk.examples import example_output_dir

pdk = load_pdk('sky130')


_VDD = 1.8
DATA_RATE = 2e9
_UI = 1.0 / DATA_RATE
N_BITS = 127
_PRBS_SEED = 0x7F
_T_EDGE = 30e-12


_WN_DRV, _WP_DRV, _L_DRV = 20.0, 40.0, 0.15


# Keep the summing resistance below R_ch so the dominant filtering occurs at RX.
_R_MAIN = 50.0
_R_TAP = 150.0
_C_PAD = 50e-15


_R_CH = 600.0
_C_CH = 1.3e-12


_T_STEP = 2e-12

SWING_METRIC = "swing_mv"
TRISE_METRIC = "trise_ps"
TFALL_METRIC = "tfall_ps"
EYE_METRIC = "eye_mv"

SWING_MIN_MV = 2500.0

EDGE_MAX_PS = 0.5 * _UI * 1e12

EYE_MIN_MV = 250.0


def prbs7(n_bits, seed=_PRBS_SEED):
    """Generate XOR, pre-shift PRBS7 through the public XNOR generator.

    Complement the seed and output, and prepend the seed's high bit to
    preserve the bit convention used by the driver stimulus.
    """
    state = seed & 0x7F
    if state == 0:
        raise ValueError("PRBS seed must be nonzero (the all-zero state is a fixed point)")
    if n_bits <= 0:
        return []
    first = [(state >> 6) & 1]
    if n_bits == 1:
        return first
    return first + [1 - bit for bit in sources.prbs_bits(7, n_bits - 1,
                                                        seed=state ^ 0x7F)]


def _tap_bits_ffe(bits):
    """Return the inverted previous bit, wrapping at the pattern boundary."""
    # Cyclic delay keeps a repeated PRBS pattern continuous.
    return [1 - bits[(k - 1) % len(bits)] for k in range(len(bits))]


def _invert(bits):
    """Return the complementary bits for the other driver leg."""
    return [1 - b for b in bits]


def build_serdes_tx(*, data_rate=DATA_RATE, n_bits=N_BITS,
                    wn=_WN_DRV, wp=_WP_DRV, l=_L_DRV,
                    r_main=_R_MAIN, r_tap=_R_TAP, c_pad=_C_PAD,
                    r_ch=_R_CH, c_ch=_C_CH, with_tests=True) -> Module:
    """Build a differential two-slice inverter driver and lumped RC channel.

    Main and delayed tap inputs arrive as external stimulus. Series resistors
    set their nominal relative conductance; inverter resistance also contributes."""
    ui = 1.0 / data_rate
    mod = Module("PDK_skywater130", example_output_dir(), "serdes_tx.cir",
                 pdk_details=pdk,
                 module_name="serdes_tx")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "DIN": Port.analog_input(), "DINB": Port.analog_input(),
        "DTAP": Port.analog_input(), "DTAPB": Port.analog_input(),
        "PADP": Port.analog_output(), "PADN": Port.analog_output(),
        "RXP": Port.analog_output(), "RXN": Port.analog_output(),
    })


    inv = build_inverter(wn=wn, wp=wp, l=l)
    mod.add_module(inv, "seg_main_p", nets={
        "VDD": "VDD", "VSS": "VSS", "A": "DIN", "Y": "drv_main_p"})
    mod.add_module(inv, "seg_main_n", nets={
        "VDD": "VDD", "VSS": "VSS", "A": "DINB", "Y": "drv_main_n"})
    mod.add_module(inv, "seg_tap_p", nets={
        "VDD": "VDD", "VSS": "VSS", "A": "DTAP", "Y": "drv_tap_p"})
    mod.add_module(inv, "seg_tap_n", nets={
        "VDD": "VDD", "VSS": "VSS", "A": "DTAPB", "Y": "drv_tap_n"})

    from silk.elements import elements
    with mod:
        for leg in ("p", "n"):
            PAD, RX = f"PAD{leg.upper()}", f"RX{leg.upper()}"
            elements.resistor(f"R_main_{leg}", p=f"drv_main_{leg}", n=PAD,
                              value=r_main)
            elements.resistor(f"R_tap_{leg}", p=f"drv_tap_{leg}", n=PAD,
                              value=r_tap)
            elements.capacitor(f"C_pad{leg}", p=PAD, n="VSS", value=c_pad)
            elements.resistor(f"R_ch_{leg}", p=PAD, n=RX, value=r_ch)
            elements.capacitor(f"C_ch{leg}", p=RX, n="VSS", value=c_ch)

    if with_tests:
        bits = prbs7(n_bits)
        attach_edge_test(mod, ui=ui)
        attach_link_test(mod, "ffe_off", din_bits=bits, dtap_bits=bits, ui=ui)
        attach_link_test(mod, "ffe_on", din_bits=bits,
                         dtap_bits=_tap_bits_ffe(bits), ui=ui)


    return mod


_EDGE_TD = 3.0
_EDGE_PW = 6.0


def attach_edge_test(mod, *, ui=_UI):
    """Measure driver swing and PAD edge times with an isolated pulse.

    Both slices carry the same pulse. Holding each level lets it settle
    before the next edge; PAD is measured before the RC channel."""
    tb = mod.create_test("edge")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "DIN": "din", "DINB": "dinb",
        "DTAP": "dtap", "DTAPB": "dtapb",
        "PADP": "padp", "PADN": "padn", "RXP": "rxp", "RXN": "rxn"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    pulse = sources.Pulse(v1=0, v2=_VDD, td=3 * ui, tr=_T_EDGE, tf=_T_EDGE,
                          pw=6 * ui, per=100 * ui)
    pulseb = sources.Pulse(v1=_VDD, v2=0, td=3 * ui, tr=_T_EDGE, tf=_T_EDGE,
                           pw=6 * ui, per=100 * ui)

    tb.add_source(sources.VoltageSource("din", positive="din", negative="0", pulse=pulse))
    tb.add_source(sources.VoltageSource("dinb", positive="dinb", negative="0", pulse=pulseb))
    tb.add_source(sources.VoltageSource("dtap", positive="dtap", negative="0", pulse=pulse))
    tb.add_source(sources.VoltageSource("dtapb", positive="dtapb", negative="0", pulse=pulseb))
    tb.add_sim_type(Transient(step=_T_STEP, stop=14 * ui, maxstep=_T_STEP))


    t_fall = _EDGE_TD * ui
    t_rise = (_EDGE_TD + _EDGE_PW) * ui
    pad_p = tb.probe("padp")
    pad_diff = pad_p - tb.probe("padn")

    hi = tb.add_metric("pad_hi_v", value_at, sig=pad_diff, x=t_fall - 0.1 * ui)
    lo = tb.add_metric("pad_lo_v", value_at, sig=pad_diff, x=t_rise - 0.1 * ui)
    tb.add_metric(SWING_METRIC, (hi - lo) * 1e3)


    fall_s = tb.add_metric("t_fall_s", fall_time,
                           sig=pad_p.window(t_fall - 0.2 * ui, t_fall + 5 * ui),
                           initial=_VDD, final=0.0)
    rise_s = tb.add_metric("t_rise_s", rise_time,
                           sig=pad_p.window(t_rise - 0.2 * ui, t_rise + 4 * ui),
                           initial=0.0, final=_VDD)
    tb.add_metric(TRISE_METRIC, rise_s * 1e12)
    tb.add_metric(TFALL_METRIC, fall_s * 1e12)
    tb.add_spec(f"driver swing > {SWING_MIN_MV:.0f} mV", SWING_METRIC, ">", SWING_MIN_MV)
    tb.add_spec(f"rise < {EDGE_MAX_PS:.0f} ps", TRISE_METRIC, "<", EDGE_MAX_PS)
    tb.add_spec(f"fall < {EDGE_MAX_PS:.0f} ps", TFALL_METRIC, "<", EDGE_MAX_PS)
    return tb


def attach_link_test(mod, name, *, din_bits, dtap_bits, ui):
    """One transient over the PRBS7: main slice = ``din_bits``, tap slice =
    ``dtap_bits``. Same rig for both tests; only the tap sequence changes.
    """
    tb = mod.create_test(name)
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "DIN": "din", "DINB": "dinb",
        "DTAP": "dtap", "DTAPB": "dtapb",
        "PADP": "padp", "PADN": "padn", "RXP": "rxp", "RXN": "rxn"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    tb.add_source(sources.VoltageSource(
        "din", positive="din", negative="0",
        waveform=sources.BitPattern(din_bits, ui=ui, v1=0, v2=_VDD,
                                    tr=_T_EDGE, tf=_T_EDGE)))
    tb.add_source(sources.VoltageSource(
        "dinb", positive="dinb", negative="0",
        waveform=sources.BitPattern(_invert(din_bits), ui=ui, v1=0, v2=_VDD,
                                    tr=_T_EDGE, tf=_T_EDGE)))
    tb.add_source(sources.VoltageSource(
        "dtap", positive="dtap", negative="0",
        waveform=sources.BitPattern(dtap_bits, ui=ui, v1=0, v2=_VDD,
                                    tr=_T_EDGE, tf=_T_EDGE)))
    tb.add_source(sources.VoltageSource(
        "dtapb", positive="dtapb", negative="0",
        waveform=sources.BitPattern(_invert(dtap_bits), ui=ui, v1=0, v2=_VDD,
                                    tr=_T_EDGE, tf=_T_EDGE)))
    tb.add_sim_type(Transient(step=_T_STEP, stop=len(din_bits) * ui, maxstep=_T_STEP))

    rx = tb.probe("rxp") - tb.probe("rxn")
    height = tb.add_metric("eye_v", eye_height, sig=rx, period=ui, threshold=0.0)
    width = tb.add_metric("eye_s", eye_width, sig=rx, period=ui, threshold=0.0)
    tb.add_metric(EYE_METRIC, height * 1e3)
    tb.add_metric("eye_width_ps", width * 1e12)

    if name == "ffe_on":
        tb.add_spec(f"eye (FFE on) > {EYE_MIN_MV:.0f} mV", EYE_METRIC, ">", EYE_MIN_MV)
    return tb


def run_characterization(mod, corner="tt"):
    """Run the edge and link testbenches, then reconcile their specs."""
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def save_plot(results, path, *, ui=_UI):
    """Plot the two geometric eyes and the FFE-on driver and receiver waveforms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 4.8), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
    ax_off = fig.add_subplot(gs[0, 0])
    ax_on = fig.add_subplot(gs[0, 1], sharey=ax_off)
    ax_wf = fig.add_subplot(gs[0, 2])

    def _eye(ax, test, title):
        run = results[f"serdes_tx.{test}"].sim_results
        sig = run.load_signal("rxp") - run.load_signal("rxn")
        eye = eye_diagram(sig, period=ui, auto_center=True, threshold=0.0)
        eye_mv = eye.height * 1e3
        for trace in eye.traces:
            Signal("eye", "voltage", "time within UI (ps)", ax.get_ylabel(),
                   (trace.x - 0.5 * ui) * 1e12, trace.y).plot(ax=ax,
                    color="#0072B2", lw=0.4, alpha=0.35)
        ax.axvline(0.0, color="#009E73", ls=":", lw=1.4)
        ax.set_title(f"{title}\neye = {eye_mv:.0f} mV" +
                     ("  (unmeasurable)" if not np.isfinite(eye_mv) else ""), fontsize=10)
        ax.set_xlabel("time about eye centre (ps)")
        ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        return eye_mv

    ax_off.set_ylabel("RX diff (V)")
    _eye(ax_off, "ffe_off", "channel out, FFE off")
    _eye(ax_on, "ffe_on", "channel out, FFE on")


    on = results["serdes_tx.ffe_on"].sim_results
    pad = on.load_signal("padp") - on.load_signal("padn")
    rx = on.load_signal("rxp") - on.load_signal("rxn")
    tp = np.asarray(pad.x, dtype=float)
    lo, hi = 20 * ui, 34 * ui
    mp = (tp >= lo) & (tp <= hi)
    tr = np.asarray(rx.x, dtype=float)
    mr = (tr >= lo) & (tr <= hi)
    ax_wf.plot((tp[mp] - lo) * 1e9, np.asarray(pad.y, dtype=float)[mp],
               color="#5a5a6e", lw=1.0, label="PAD diff (driver out, pre-emphasized)")
    ax_wf.plot((tr[mr] - lo) * 1e9, np.asarray(rx.y, dtype=float)[mr],
               color="#0072B2", lw=1.6, label="RX diff (channel out)")
    ax_wf.set_title("FFE on: the pre-emphasis kick\nand the equalized RX", fontsize=10)
    ax_wf.set_xlabel("time (ns)")
    ax_wf.set_ylabel("diff (V)")
    ax_wf.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
    ax_wf.set_axisbelow(True)
    ax_wf.legend(frameon=False, fontsize=7.5, loc="upper right")

    fig.suptitle("SerDes TX 1-tap FFE (differential) — same PRBS7, same channel, "
                 "tap slice off vs on", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 72)
    print(f"SerDes TX driver + 1-tap FFE — {DATA_RATE/1e9:.1f} Gb/s, PRBS7, "
          f"1.8 V CMOS")
    print("=" * 72)

    mod = build_serdes_tx()
    results, recon = run_characterization(mod)
    print(recon.summary())

    edge = results[f"{mod.module_name}.edge"]
    off = results[f"{mod.module_name}.ffe_off"]
    on = results[f"{mod.module_name}.ffe_on"]
    eye_off, eye_on = off.metrics[EYE_METRIC], on.metrics[EYE_METRIC]

    print(f"  driver swing (diff) = {edge.metrics[SWING_METRIC]:.0f} mV")
    print(f"  driver rise / fall  = {edge.metrics[TRISE_METRIC]:.1f} / "
          f"{edge.metrics[TFALL_METRIC]:.1f} ps (10-90%, at PAD)")
    print(f"  channel pole        ~ "
          f"{1/(2*np.pi*((_R_MAIN*_R_TAP/(_R_MAIN+_R_TAP))+_R_CH)*_C_CH)/1e6:.0f} MHz "
          f"(tau ~ {((_R_MAIN*_R_TAP/(_R_MAIN+_R_TAP))+_R_CH)*_C_CH*1e12:.0f} ps "
          f"vs UI {_UI*1e12:.0f} ps)")
    print(f"  eye @ RX, FFE off   = {eye_off:7.0f} mV / "
          f"{off.metrics['eye_width_ps']:.0f} ps (geometric opening)")
    print(f"  eye @ RX, FFE on    = {eye_on:7.0f} mV / "
          f"{on.metrics['eye_width_ps']:.0f} ps (geometric opening)")
    print(f"  FFE eye improvement = {eye_on - eye_off:.0f} mV "
          "(unavailable if either geometric eye is NaN)")
    print(f"  saved {save_plot(results, os.path.join(OUT, 'serdes_tx_driver_eye.png'))}")
