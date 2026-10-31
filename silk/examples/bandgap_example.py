"""Hierarchical Banba low-voltage bandgap reference.

The reference sums a PTAT current derived from two PNP emitter voltages
with a CTAT current derived from VBE. An output resistor converts their
sum to VREF. The circuit composes amplifier, current-bank, startup, and
BJT/resistor blocks.

Variant A uses a PMOS-input two-stage OTA to sense the branch nodes directly.
Variant B uses an NMOS-input five-transistor OTA and two PMOS source followers
to raise the sense voltages. Their temperature sweeps compare the effects of
these input arrangements.

Each variant has a transient supply-ramp bench to exercise startup and a DC
temperature sweep to measure box temperature coefficient (TC) and mean VREF.

Run from the repository root with MPLBACKEND=Agg:
    .venv/bin/python -m silk.examples.bandgap_example
It prints spec results, settled VREF, and TC for both variants, and saves
VREF-versus-temperature plots.
"""
import os

import numpy as np


from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.metrics import average, final_value, peak_to_peak
from silk.simulation import DC, Transient

from silk.examples.two_stage_miller_ota_example import build_two_stage_ota
from silk.examples.source_follower_example import build_source_follower
from silk.examples import example_output_dir

# The PDK manifest resolves the model YAML and the ngspice lib itself —
# no paths in example code (silk/pdk/data/sky130.pdk.yaml).
pdk = load_pdk('sky130')

# --- design knobs --------------------------------------------------------------
N_RATIO = 8           # Q2 emitter-area multiple (ΔV_be = V_T·ln(8))
_R1 = 50e3            # sets the PTAT current  I_ptat = ΔV_be / R1
# R0/R1 sets the PTAT-to-CTAT current balance and temperature coefficient.
_R0 = 395e3          # CTAT injection resistors (R0/R1 = 7.9)
# R_OUT converts the summed current to the reference voltage.
_R_OUT = 409e3       # V_ref = R_out*(I_ptat + I_ctat)
_WP = 10.0            # PMOS current-source width (µm), both variants
# Size each current bank to keep VG within its amplifier's output range.
# Variant B uses L=0.5 um and m=32, giving W/L*m=640.
_BANK = {"pmos": dict(l=1.0, m=1), "nmos": dict(l=0.5, m=32)}
_I_AMP = 10e-6       # op-amp bias reference current
# Recalculate loop compensation when loop gain changes. Variant A has
# internal Miller compensation; Variant B uses the larger VG capacitor.
_CCOMP_PMOS = 1e-12
_CCOMP_NMOS = 10e-12

VREF_LO, VREF_HI = 1.0, 1.4   # acceptance band for the settled V_ref

# Check both output level and temperature dependence.
TC_T_START, TC_T_STOP, TC_T_STEP = -40, 125, 5   # .dc temp -40 125 5
TC_TARGET_PPM = 100.0                            # box TC target, ppm/C


def build_5t_ota(*, input_type="nmos", w_in=10.0, l_in=0.15,
                 w_load=40.0, l_load=0.5, m_load=4,
                 w_tail=20.0, l_tail=0.15, w_bias=20.0, l_bias=0.15) -> Module:
    """Build the NMOS-input five-transistor error amplifier and bias diode.

    M1 drains into the mirror diode node; M2 drains into VOUT. VINP is
    non-inverting. Parallel load fingers increase W/L while keeping each
    PMOS width within the catalog model bins.
    """
    m = Module("PDK_skywater130", example_output_dir("bandgap_example"),
               f"ota_5t_{input_type}.cir",
               pdk_details=pdk, module_name=f"ota_5t_{input_type}")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "VINP": Port.analog_input(), "VINM": Port.analog_input(),
        "VOUT": Port.analog_output(), "IBIAS": Port.analog_input(),
    })
    if input_type == "nmos":
        with m:
            # input pair: VINP -> the diode node n1 (non-inverting);
            #             VINM -> VOUT, the mirror-output node (inverting).
            pdk.nmos1p8("M1", d="n1", g="VINP", s="tail", b="VSS", w=w_in, l=l_in, m=1)
            pdk.nmos1p8("M2", d="VOUT", g="VINM", s="tail", b="VSS", w=w_in, l=l_in, m=1)
            # PMOS mirror load: M3 is the diode, M4 the output leg.
            pdk.pmos1p8("M3", d="n1", g="n1", s="VDD", b="VDD", w=w_load, l=l_load, m=m_load)
            pdk.pmos1p8("M4", d="VOUT", g="n1", s="VDD", b="VDD", w=w_load, l=l_load, m=m_load)
            # tail current source + its bias diode.
            pdk.nmos1p8("M5", d="tail", g="IBIAS", s="VSS", b="VSS", w=w_tail, l=l_tail, m=1)
            pdk.nmos1p8("MBIAS", d="IBIAS", g="IBIAS", s="VSS", b="VSS", w=w_bias, l=l_bias, m=1)
    else:
        raise ValueError(f"5T OTA: only 'nmos' implemented, got {input_type!r}")
    return m


def build_pmos_current_bank(*, kind="simple", w=_WP, l=1.0, m=1) -> Module:
    """Build three PMOS current sources sharing gate VG and supply VDD.

    OUT1/OUT2 feed the reference branches; OUT3 feeds the output resistor.
    The multiplier m raises effective W/L without shortening the devices.
    """
    mod = Module("PDK_skywater130", example_output_dir("bandgap_example"),
                 f"pmos_ibank_{kind}.cir",
                 pdk_details=pdk, module_name=f"pmos_ibank_{kind}")
    mod.define_module_ports({
        "VDD": Port.supply(), "VG": Port.analog_input(),
        "OUT1": Port.analog_output(), "OUT2": Port.analog_output(),
        "OUT3": Port.analog_output(),
    })
    if kind == "simple":
        with mod:      # auto-registration stays active across the loop
            for name, out in (("MP1", "OUT1"), ("MP2", "OUT2"), ("MP3", "OUT3")):
                pdk.pmos1p8(name, d=out, g="VG", s="VDD", b="VDD", w=w, l=l, m=m)
    else:
        raise ValueError(f"unknown current-bank kind {kind!r}")
    return mod


def build_startup(*, kind="emitter_sense") -> Module:
    """Build an emitter-sensing startup circuit.

    RPU raises nstart when SENSE is low. MINJ pulls VG down to start the
    PMOS sources and turns off as the core starts conducting and MDET
    pulls nstart low. Connect SENSE to PNP emitter node A.
    """
    from silk.elements import elements
    m = Module("PDK_skywater130", example_output_dir("bandgap_example"),
               f"startup_{kind}.cir",
               pdk_details=pdk, module_name=f"startup_{kind}")
    m.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "SENSE": Port.analog_input(), "VG": Port.analog_output(),
    })
    if kind == "emitter_sense":
        # One `with` block covers devices and passives alike.
        with m:
            # MDET pulls nstart low when the PNP emitter voltage rises.
            pdk.nmos1p8("MDET", d="nstart", g="SENSE", s="VSS", b="VSS", w=2.0, l=0.5, m=1)
            # MINJ pulls VG down until the emitter-sense detector conducts.
            pdk.nmos1p8("MINJ", d="VG", g="nstart", s="VSS", b="VSS", w=1.0, l=0.25, m=1)
            elements.resistor("RPU", p="VDD", n="nstart", value=1e6)
    else:
        raise ValueError(f"unknown startup kind {kind!r}")
    return m


def build_resistor(resistance, *, module_name) -> Module:
    """Build a two-terminal resistor block with ports P and N.

    The two CTAT branches reuse one resistor definition to express equal
    nominal values. Physical matching requires layout work.
    """
    from silk.elements import elements
    m = Module("PDK_skywater130", example_output_dir("bandgap_example"),
               f"{module_name}.cir",
               pdk_details=pdk, module_name=module_name)
    m.define_module_ports({"P": Port.analog_input(), "N": Port.analog_input()})
    with m:
        elements.resistor("R", p="P", n="N", value=resistance)
    return m


def build_banba_branches(*, n_ratio=N_RATIO, r1=_R1, r0=_R0, r_out=_R_OUT) -> Module:
    """The BJT + resistor branches as their own module.

    Diode-connected PNPs Q1 (×1) / Q2 (×N) generate ΔV_be; R1 sets the PTAT
    current, R0A/R0B inject the CTAT current, ROUT converts the summed current
    to V_ref. The CTAT pair R0A/R0B are two instances of one shared resistor
    module (matched by construction). Ports ``A``/``B`` are the current-injection
    + sense nodes, ``VREF`` the output, ``VSS`` the ground; ``C`` is internal.
    """
    m = Module("PDK_skywater130", example_output_dir("bandgap_example"),
               "banba_branches.cir",
               pdk_details=pdk, module_name="banba_branches")
    m.define_module_ports({
        "VSS": Port.ground(), "A": Port.analog_input(),
        "B": Port.analog_input(), "VREF": Port.analog_output(),
    })
    with m:
        # The three-terminal PNP ties its substrate internally to its collector.
        pdk.pnp_small("Q1", e="A", b="VSS", c="VSS", m=1)
        pdk.pnp_small("Q2", e="C", b="VSS", c="VSS", m=n_ratio)

    r_ptat = build_resistor(r1, module_name="res_ptat")
    r_ctat = build_resistor(r0, module_name="res_ctat")   # matched pair share one module
    r_outp = build_resistor(r_out, module_name="res_out")
    m.add_module(r_ptat, "R1", nets={"P": "B", "N": "C"})       # PTAT
    m.add_module(r_ctat, "R0A", nets={"P": "A", "N": "VSS"})    # CTAT branch A
    m.add_module(r_ctat, "R0B", nets={"P": "B", "N": "VSS"})    # CTAT branch B (matched)
    m.add_module(r_outp, "ROUT", nets={"P": "VREF", "N": "VSS"})
    return m


def _add_banba_core(mod, *, n_ratio=N_RATIO, r1=_R1, r0=_R0, r_out=_R_OUT,
                    wp=_WP, lp=1.0, mp=1, current_bank="simple",
                    startup="emitter_sense", ccomp=_CCOMP_PMOS):
    """Place the current bank, BJT/resistor branches, and startup blocks.

    add_module connects A/B as sense nodes, VG as the shared PMOS gate,
    and VREF as the reference output.
    """

    bank = build_pmos_current_bank(kind=current_bank, w=wp, l=lp, m=mp)
    mod.add_module(bank, "ibank", nets={
        "VDD": "VDD", "VG": "VG", "OUT1": "A", "OUT2": "B", "OUT3": "VREF"})

    branches = build_banba_branches(n_ratio=n_ratio, r1=r1, r0=r0, r_out=r_out)
    mod.add_module(branches, "branches", nets={
        "VSS": "VSS", "A": "A", "B": "B", "VREF": "VREF"})

    su = build_startup(kind=startup)
    mod.add_module(su, "startup", nets={
        "VDD": "VDD", "VSS": "VSS", "SENSE": "A", "VG": "VG"})

    # Compensate the high-impedance amplifier output at VG.
    from silk.elements import elements
    with mod:
        elements.capacitor("CCOMP", p="VG", n="VSS", value=ccomp)


def build_bandgap(input_type="pmos", *, n_ratio=N_RATIO, r1=_R1, r0=_R0,
                  r_out=_R_OUT, current_bank="simple", startup="emitter_sense",
                  ccomp=None) -> Module:
    """Build a Banba bandgap composing the OTA error amplifier.

    ``input_type='pmos'`` (Variant A): a PMOS-input OTA senses A/B (~V_be)
    directly. ``input_type='nmos'`` (Variant B): an NMOS-input OTA senses
    level-shifted copies A'/B' produced by two PMOS source followers.

    ``current_bank`` and ``startup`` select the sub-module structures
    (`build_pmos_current_bank` / `build_startup`), so an alternative current
    source or startup cell can be dropped in without changing the core.
    """
    ports = {
        "VDD": Port.supply(), "VSS": Port.ground(),
        "VREF": Port.analog_output(), "IBIAS_AMP": Port.analog_input(),
    }
    if input_type == "nmos":
        ports["IBIAS_SF"] = Port.analog_input()   # bias for the level shifters
    if input_type not in _BANK:
        raise ValueError(f"input_type must be 'pmos' or 'nmos', got {input_type!r}")
    if ccomp is None:
        ccomp = _CCOMP_NMOS if input_type == "nmos" else _CCOMP_PMOS
    bank = _BANK[input_type]
    mod = Module("PDK_skywater130", example_output_dir("bandgap_example"),
                 f"bandgap_{input_type}.cir",
                 pdk_details=pdk, module_name=f"bandgap_{input_type}")
    mod.define_module_ports(ports)
    _add_banba_core(mod, n_ratio=n_ratio, r1=r1, r0=r0, r_out=r_out,
                    lp=bank["l"], mp=bank["m"],
                    current_bank=current_bank, startup=startup, ccomp=ccomp)

    if input_type == "pmos":
        # Variant A: PMOS-input OTA senses the ~V_be nodes A/B directly.
        amp = build_two_stage_ota(input_type="pmos", with_tests=False)
        # Polarity for negative feedback: A>B must decrease VG (more PMOS
        # current), so A drives the inverting input. VINP is non-inverting in
        # build_two_stage_ota, hence VINP<-B, VINM<-A.
        mod.add_module(amp, "amp", nets={
            "VDD": "VDD", "VSS": "VSS", "VINP": "B", "VINM": "A",
            "VOUT": "VG", "IBIAS": "IBIAS_AMP"})
    elif input_type == "nmos":
        # Variant B: two matched PMOS source followers lift A/B by |Vgs| into
        # the NMOS-input OTA's common-mode range; the OTA (with a widened load
        # to raise its CM ceiling) then senses the lifted copies Ap/Bp. Matched
        # shifters mean forcing Ap=Bp is equivalent to forcing A=B.
        # Minimum-length followers reduce the level shift into the NMOS pair.
        # Shared bias: one diode-connected PMOS on IBIAS_SF sets the gate
        # voltage; each shifter keeps its own M2 mirroring that single diode
        # (internal_bias=False drops each follower's private diode), so both run
        # the full reference current instead of splitting it across two diodes.
        with mod:
            pdk.pmos1p8("MBIAS_SF", d="IBIAS_SF", g="IBIAS_SF", s="VDD", b="VDD", w=6.0, l=1.0, m=1)
        # Tie each follower bulk to its source to reduce body effect.
        sf = build_source_follower(device="pmos", with_tests=False, internal_bias=False,
                                   bulk_to_source=True,
                                   w_follower=40.0, l_follower=0.15)
        mod.add_module(sf, "sf_a", nets={
            "VDD": "VDD", "VSS": "VSS", "VIN": "A", "VOUT": "Ap", "IBIAS": "IBIAS_SF"})
        mod.add_module(sf, "sf_b", nets={
            "VDD": "VDD", "VSS": "VSS", "VIN": "B", "VOUT": "Bp", "IBIAS": "IBIAS_SF"})
        amp = build_5t_ota(input_type="nmos")   # single-stage 5T (sizing from its defaults)
        # Polarity: B is the *resistor* branch (B = V_be2 + I*R1), so its voltage
        # climbs linearly with branch current while A = V_be1 climbs only
        # logarithmically. Too much current therefore shows up as B >> A, and the
        # loop must raise VG to throttle the PMOS bank. VG must rise with B, so
        # the B side lands on the non-inverting input: VINP<-Bp, VINM<-Ap. (Same
        # sense as Variant A's VINP<-B / VINM<-A.)
        mod.add_module(amp, "amp", nets={
            "VDD": "VDD", "VSS": "VSS", "VINP": "Bp", "VINM": "Ap",
            "VOUT": "VG", "IBIAS": "IBIAS_AMP"})
    else:
        raise ValueError(f"input_type must be 'pmos' or 'nmos', got {input_type!r}")
    return mod


def _add_bias_currents(tb, i_amp, input_type):
    """The reference currents — shared by the settle and TC testbenches."""
    if input_type == "pmos":
        # PMOS-input OTA's PMOS diode sinks the reference to ground.
        tb.add_source(sources.CurrentSource("iamp", positive="ibias_amp", negative="0", dc=i_amp))
    else:
        # NMOS-input OTA's NMOS diode sources its reference from VDD.
        tb.add_source(sources.CurrentSource("iamp", positive="VDD", negative="ibias_amp", dc=i_amp))
        # Level-shifter bias: PMOS source followers' PMOS diode sinks to ground.
        tb.add_source(sources.CurrentSource("isf", positive="ibias_sf", negative="0", dc=i_amp))


def _mut_map(input_type):
    mut = {"VDD": "VDD", "VSS": "0", "VREF": "vref", "IBIAS_AMP": "ibias_amp"}
    if input_type == "nmos":
        mut["IBIAS_SF"] = "ibias_sf"
    return mut


def attach_settle_test(mod, *, i_amp=_I_AMP, input_type="pmos", stop=20e-6):
    """Ramp VDD from 0 to 1.8 V over 2 us and measure final VREF.

    The transient exercises the startup circuit at nominal temperature.
    The separate DC temperature sweep measures steady-state VREF.
    """
    tb = mod.create_test("settle")
    tb.add_module_under_test(_mut_map(input_type))
    tb.add_source(sources.VoltageSource(
        "vdd", positive="VDD", negative="0",
        pwl=sources.PWL().add(0, 0).add(2e-6, 1.8).add(stop, 1.8)))
    _add_bias_currents(tb, i_amp, input_type)
    tb.add_sim_type(Transient(step=20e-9, stop=stop))
    tb.add_metric("vref", final_value, sig=tb.probe("vref"))  # settled V_ref
    tb.add_spec("Vref in [1.0,1.4] (lower)", "vref", ">", VREF_LO)
    tb.add_spec("Vref in [1.0,1.4] (upper)", "vref", "<", VREF_HI)
    return tb


def _tc_ppm_per_C(ckt):
    """Measure box TC as peak-to-peak VREF divided by mean VREF and span.

    The result is in ppm/C. average integrates over the temperature axis.
    Near-zero output raises RuntimeError because the core did not start.
    """
    sig = ckt.load_signal("vref")
    T = np.asarray(sig.x, dtype=float)
    v = np.asarray(sig.y, dtype=float)
    if v.min() < 0.1:
        # Reject near-zero outputs: failed startup can appear to have low TC.
        dead = T[v < 0.1]
        raise RuntimeError(f"core did not start at {dead.tolist()} C (V_ref ~ 0) — "
                           f"a TC over a dead core is meaningless")
    return float(peak_to_peak(sig) / average(sig) / (T.max() - T.min()) * 1e6)


def attach_tc_test(mod, *, i_amp=_I_AMP, input_type="pmos",
                   t_start=TC_T_START, t_stop=TC_T_STOP, t_step=TC_T_STEP):
    """Sweep temperature and measure box TC and mean output voltage.

    DC(source='temp', ...) returns temperature on the signal x-axis.
    Mean-level specs reject a flat output outside the acceptance band.
    The near-zero guard in _tc_ppm_per_C rejects a failed core.

    A DC sweep does not exercise startup dynamics; attach_settle_test
    uses a transient supply ramp for that measurement.
    """
    tb = mod.create_test("tc")
    tb.add_module_under_test(_mut_map(input_type))
    # Plain DC supply: no ramp, because .dc solves the operating point directly.
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=1.8))
    _add_bias_currents(tb, i_amp, input_type)
    tb.add_sim_type(DC(source="temp", start=t_start, stop=t_stop, step=t_step))
    tb.add_metric("tc_ppm_per_C", _tc_ppm_per_C)
    tb.add_spec(f"TC < {TC_TARGET_PPM:.0f} ppm/C", "tc_ppm_per_C", "<", TC_TARGET_PPM)
    # A flat output can still have the wrong level; check its mean as well.
    tb.add_metric("vref_mean", average, sig=tb.probe("vref"))
    tb.add_spec(f"Vref mean in [{VREF_LO:.1f},{VREF_HI:.1f}] (lower)", "vref_mean", ">", VREF_LO)
    tb.add_spec(f"Vref mean in [{VREF_LO:.1f},{VREF_HI:.1f}] (upper)", "vref_mean", "<", VREF_HI)
    return tb


def save_tc_plot(ckt, path, title):
    """Save VREF versus temperature from the DC sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = ckt.load_signal("vref")
    T = np.asarray(sig.x, dtype=float)
    v = np.asarray(sig.y, dtype=float)
    tc_ppm = _tc_ppm_per_C(ckt)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T, v * 1e3, "-o", color="#0072B2", lw=2, ms=4)
    ax.axhline(average(sig) * 1e3, color="#5a5a6e", ls="--", lw=1, label="mean")
    ax.set_xlabel("temperature (°C)")
    ax.set_ylabel("V_ref (mV)")
    ax.set_title(f"{title} — TC = {tc_ppm:.0f} ppm/°C, "
                 f"{(v.max()-v.min())*1e3:.1f} mV over {T.max()-T.min():.0f} °C")
    ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_characterization(mod, corner="tt"):
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


if __name__ == "__main__":
    # Compare startup and temperature dependence for both input arrangements.
    OUT = example_output_dir()

    for variant, note in (("pmos", "PMOS-input OTA senses A/B directly"),
                          ("nmos", "NMOS-input OTA + PMOS source-follower level shift")):
        label = "A" if variant == "pmos" else "B"
        m = build_bandgap(variant)
        attach_settle_test(m, input_type=variant)
        attach_tc_test(m, input_type=variant)
        print("=" * 64)
        print(f"Variant {label}: {m.module_name}")
        print(f"  {note}")
        print("=" * 64)
        try:
            results, recon = run_characterization(m)
            print(recon.summary())
            settle = results[f"{m.module_name}.settle"]
            tc = results[f"{m.module_name}.tc"]
            print(f"  V_ref (settled, 27 C) = {settle.metrics['vref']*1e3:.1f} mV")
            print(f"  TC = {tc.metrics['tc_ppm_per_C']:.0f} ppm/C")
            png = save_tc_plot(tc.sim_results,   # the netlist obj; load_signal works on it
                               os.path.join(OUT, f"bandgap_{variant}_tc.png"),
                               f"Bandgap Variant {label} ({m.module_name})")
            print(f"  saved {png}\n")
        except Exception as e:
            print(f"SPICE characterization skipped: {e}\n")
