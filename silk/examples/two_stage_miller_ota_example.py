"""Characterize a two-stage Miller-compensated OTA.

A differential pair with a mirror load drives a common-source second stage.
A capacitor and series nulling resistor connect the output to the first
stage's output. input_type selects complementary NMOS or PMOS input cores.
The ac bench measures open-loop gain, unity-gain frequency and phase margin.
The tran bench closes unity-gain feedback and measures step slew rate.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.two_stage_miller_ota_example``.
It prints the netlist, test names and spec results, then saves a Bode plot
and circuit reports under example_output_dir(). Needs ngspice and sky130
models.
"""
import os

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.simulation import AC, Transient
from silk.metrics import dc_gain_dB, phase_margin, unity_gain_freq
from silk.metrics.timing import slew_rate
from silk.examples import example_output_dir


pdk = load_pdk('sky130')


_W_IN, _L_IN = 10.0, 1.0
_W_LOAD, _L_LOAD = 10.0, 1.0
_W_TAIL, _L_TAIL = 20.0, 1.0
_W_CS, _L_CS = 20.0, 1.0
_W_SINK, _L_SINK = 20.0, 1.0
_CC = 4.0e-12
_RZ = 2.0e3
_CLOAD = 1.0e-12

_W_BIAS, _L_BIAS = 20.0, 1.0
_I_REF = 30e-6


GAIN_METRIC = "gain_dB"
GAIN_TARGET_DB = 60.0
PM_METRIC = "pm_deg"
PM_TARGET_DEG = 45.0
SLEW_METRIC = "slew_v_per_us"
SLEW_TARGET = 1.0

_STEP_T = 1e-7


def _add_devices(mod, *, input_type, w_in, l_in, w_load, l_load, w_tail, l_tail,
                 w_cs, l_cs, w_sink, l_sink, w_bias, l_bias, cc, rz, c_load):
    """Wire the selected complementary core, Miller branch and load capacitor."""
    from silk.elements import elements

    sizes = dict(w_in=w_in, l_in=l_in, w_load=w_load, l_load=l_load,
                 w_tail=w_tail, l_tail=l_tail, w_cs=w_cs, l_cs=l_cs,
                 w_sink=w_sink, l_sink=l_sink, w_bias=w_bias, l_bias=l_bias)
    if input_type == "nmos":
        _add_core_nmos(mod, **sizes)
    elif input_type == "pmos":
        _add_core_pmos(mod, **sizes)
    else:
        raise ValueError(f"input_type must be 'nmos' or 'pmos', got {input_type!r}")

    with mod:
        elements.resistor("RZ", p="VOUT", n="ncomp", value=rz)
        elements.capacitor("CC", p="ncomp", n="n2", value=cc)
        elements.capacitor("CL", p="VOUT", n="VSS", value=c_load)


def _add_core_nmos(mod, *, w_in, l_in, w_load, l_load, w_tail, l_tail,
                   w_cs, l_cs, w_sink, l_sink, w_bias, l_bias):
    """Wire an NMOS input pair and PMOS second stage.

    VINP drives n2 so an increase at VINP raises VOUT. Connecting VINM to
    VOUT therefore closes negative feedback.
    """
    with mod:
        pdk.nmos1p8("M1", d="n2", g="VINP", s="tail", b="VSS", w=w_in, l=l_in, m=1)
        pdk.nmos1p8("M2", d="n1", g="VINM", s="tail", b="VSS", w=w_in, l=l_in, m=1)
        pdk.pmos1p8("M3", d="n1", g="n1", s="VDD", b="VDD", w=w_load, l=l_load, m=1)
        pdk.pmos1p8("M4", d="n2", g="n1", s="VDD", b="VDD", w=w_load, l=l_load, m=1)
        pdk.nmos1p8("M5", d="tail", g="IBIAS", s="VSS", b="VSS", w=w_tail, l=l_tail, m=1)
        pdk.pmos1p8("M6", d="VOUT", g="n2", s="VDD", b="VDD", w=w_cs, l=l_cs, m=1)
        pdk.nmos1p8("M7", d="VOUT", g="IBIAS", s="VSS", b="VSS", w=w_sink, l=l_sink, m=1)

        pdk.nmos1p8("MBIAS", d="IBIAS", g="IBIAS", s="VSS", b="VSS", w=w_bias, l=l_bias, m=1)


def _add_core_pmos(mod, *, w_in, l_in, w_load, l_load, w_tail, l_tail,
                   w_cs, l_cs, w_sink, l_sink, w_bias, l_bias):
    """Wire the complementary PMOS input core with VINP as non-inverting input."""
    with mod:
        pdk.pmos1p8("M1", d="n2", g="VINP", s="tail", b="VDD", w=w_in, l=l_in, m=1)
        pdk.pmos1p8("M2", d="n1", g="VINM", s="tail", b="VDD", w=w_in, l=l_in, m=1)
        pdk.nmos1p8("M3", d="n1", g="n1", s="VSS", b="VSS", w=w_load, l=l_load, m=1)
        pdk.nmos1p8("M4", d="n2", g="n1", s="VSS", b="VSS", w=w_load, l=l_load, m=1)
        pdk.pmos1p8("M5", d="tail", g="IBIAS", s="VDD", b="VDD", w=w_tail, l=l_tail, m=1)
        pdk.nmos1p8("M6", d="VOUT", g="n2", s="VSS", b="VSS", w=w_cs, l=l_cs, m=1)
        pdk.pmos1p8("M7", d="VOUT", g="IBIAS", s="VDD", b="VDD", w=w_sink, l=l_sink, m=1)

    with mod:
        pdk.pmos1p8("MBIAS", d="IBIAS", g="IBIAS", s="VDD", b="VDD", w=w_bias, l=l_bias, m=1)


def _add_supply_and_bias(tb, i_ref, input_type):
    """Shared rig: supply + reference current, added to any testbench.

    Each ``create_test`` builds its own independent netlist, so this is called
    once per testbench (not shared mutable state) — it just removes the
    duplicated supply/bias wiring (cf. ``add_strongarm_rig`` in strongarm_tb.py).
    The reference current direction follows the bias diode: pushed *into* IBIAS
    for the NMOS diode, pulled *out* for the PMOS diode.
    """
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=1.8))
    if input_type == "nmos":
        tb.add_source(sources.CurrentSource("iref", positive="VDD", negative="ibias", dc=i_ref))
    else:
        tb.add_source(sources.CurrentSource("iref", positive="ibias", negative="0", dc=i_ref))


def _attach_ac_test(mod, v_cm, i_ref, input_type):
    """Measure open-loop gain, phase margin and unity-gain frequency.

    LFB closes DC feedback to establish bias and opens at AC. CIN holds
    VINM at AC ground while VINP supplies the AC drive.
    """
    from silk.elements import elements

    tb = mod.create_test("ac")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "VINP": "vinp",
        "VINM": "vinm", "VOUT": "vout", "IBIAS": "ibias",
    })
    _add_supply_and_bias(tb, i_ref, input_type)
    tb.add_source(sources.VoltageSource("vinp", positive="vinp", negative="0", dc=v_cm, ac=1.0))

    # Close DC feedback for bias while keeping the AC response open-loop.
    tb.add(elements.inductor("LFB", p="vout", n="vinm", value=1e6))
    tb.add(elements.capacitor("CIN", p="vinm", n="0", value=1.0))
    tb.add_sim_type(AC(variation="dec", points=40, fstart=1, fstop=1e9))

    # Bind all AC metrics to the output probe.
    vout = tb.probe("vout")
    tb.add_metric(GAIN_METRIC, dc_gain_dB, sig=vout)
    tb.add_metric(PM_METRIC, phase_margin, sig=vout)
    tb.add_metric("ugf_hz", unity_gain_freq, sig=vout)
    tb.add_spec("DC gain > 60 dB", GAIN_METRIC, ">", GAIN_TARGET_DB)
    tb.add_spec("phase margin > 45 deg", PM_METRIC, ">", PM_TARGET_DEG)
    return tb


def _slew(ckt):
    """Average slew rate of VOUT in V/us over the 10%-90% step edge."""
    sig = ckt.load_signal("vout")
    return slew_rate(sig) / 1e6


def _attach_transient_test(mod, v_cm, i_ref, input_type):
    """Unity-gain follower large-step slew rate.

    VINM is tied to VOUT (mapping both ports to the same net) to close the
    loop; a ~0.4 V step on VINP (centered on the input common mode) drives the
    output through its slew-limited edge before settling. ``slew_rate`` returns
    V/s; ``_slew`` converts to V/us.
    """
    step_lo, step_hi = v_cm - 0.2, v_cm + 0.2
    tb = mod.create_test("tran")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "VINP": "vinp",
        "VINM": "vout", "VOUT": "vout", "IBIAS": "ibias",
    })
    _add_supply_and_bias(tb, i_ref, input_type)
    tb.add_source(sources.VoltageSource(
        "vinp", positive="vinp", negative="0",
        pulse=sources.Pulse(v1=step_lo, v2=step_hi, td=_STEP_T,
                            tr=1e-9, tf=1e-9, pw=1e-6, per=2e-6)))
    tb.add_sim_type(Transient(step=1e-9, stop=1e-6))

    tb.add_metric(SLEW_METRIC, _slew)
    tb.add_spec("slew > 1 V/us", SLEW_METRIC, ">", SLEW_TARGET)
    return tb


def build_two_stage_ota(*, input_type="nmos", with_tests=True,
                        w_in=_W_IN, l_in=_L_IN, w_load=_W_LOAD, l_load=_L_LOAD,
                        w_tail=_W_TAIL, l_tail=_L_TAIL, w_cs=_W_CS, l_cs=_L_CS,
                        w_sink=_W_SINK, l_sink=_L_SINK, w_bias=_W_BIAS, l_bias=_L_BIAS,
                        cc=_CC, rz=_RZ, c_load=_CLOAD, v_cm=None, i_ref=_I_REF) -> Module:
    """Build the two-stage Miller OTA with its characterization suite embedded.

    All design knobs are keyword arguments (defaulting to the module-level
    constants) so the block can be reused or swept without editing the file.
    ``input_type`` selects the diff-pair flavor:

      * ``'nmos'`` — NMOS input pair. The default.
      * ``'pmos'`` — PMOS input pair; senses a **low** input CM (near VSS), as
        needed when the OTA is a bandgap error amplifier sensing ~V_be.

    The returned module owns an open-loop AC testbench (``mod.get_test('ac')``,
    gain/PM/UGF) and a closed-loop transient testbench (``mod.get_test('tran')``,
    slew rate), plus the design Specs they declare. Pass it to
    ``run_characterization()``.

    Parameters
    ----------
    input_type : {'nmos', 'pmos'}   Diff-pair flavor (sets the input CM range).
    w_in, l_in : float        Input pair M1/M2 W/L (µm).
    w_load, l_load : float    Mirror load M3/M4 W/L (µm).
    w_tail, l_tail : float    Tail M5 W/L (µm).
    w_cs, l_cs : float        Second-stage M6 W/L (µm).
    w_sink, l_sink : float    Sink/source load M7 W/L (µm).
    w_bias, l_bias : float    Diode-connected bias device W/L (µm); mirrors to M5/M7.
    cc : float                Miller capacitance (F).
    rz : float                Nulling resistance (Ω).
    c_load : float            Output load capacitance (F).
    v_cm : float, optional    Input common-mode bias (V). Defaults to 0.9 for
                              'nmos', 0.6 for 'pmos'.
    i_ref : float             Reference current into the bias diode (A).
    """
    if v_cm is None:
        v_cm = 0.9 if input_type == "nmos" else 0.6
    mod = Module("PDK_skywater130", example_output_dir("two_stage_miller_ota_example"),
                 f"two_stage_ota_{input_type}.cir",
                 pdk_details=pdk, module_name=f"two_stage_ota_{input_type}")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "VINP": Port.analog_input(), "VINM": Port.analog_input(),
        "VOUT": Port.analog_output(), "IBIAS": Port.analog_input(),
    })
    _add_devices(mod, input_type=input_type, w_in=w_in, l_in=l_in,
                 w_load=w_load, l_load=l_load, w_tail=w_tail, l_tail=l_tail,
                 w_cs=w_cs, l_cs=l_cs, w_sink=w_sink, l_sink=l_sink,
                 w_bias=w_bias, l_bias=l_bias, cc=cc, rz=rz, c_load=c_load)
    if with_tests:
        _attach_ac_test(mod, v_cm, i_ref, input_type)
        _attach_transient_test(mod, v_cm, i_ref, input_type)
    return mod


def run_characterization(mod, corner="tt"):
    """Drive the embedded loop; return ``(results, reconcile_report)``."""
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def plot_bode(tb, report, png_path):
    """Plot the AC bench's current vout response and annotate gain, unity-gain
    frequency and phase margin.

    Pass results[f'{mod.module_name}.ac'] as report. Save the PNG at
    png_path and return the open Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = tb.probe("vout")
    gain_db = report.metrics[GAIN_METRIC]
    pm = report.metrics[PM_METRIC]
    ugf = report.metrics["ugf_hz"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sig.plot(ax=ax, color="darkorange", lw=1.6, label="|vout/vin|")
    ax.axvline(ugf, color="crimson", ls="--", lw=0.8,
               label=f"UGF = {ugf / 1e6:.1f} MHz")
    ax.set_title(f"Two-stage Miller OTA — gain {gain_db:.1f} dB, PM {pm:.0f}°")
    ax.grid(True, which="both", alpha=0.3)

    axp = ax.twinx()
    sig.plot(ax=axp, mode="phase", color="slategray", lw=1.0, alpha=0.7)
    axp.set_ylabel("Phase (deg)", color="slategray")

    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    return fig


mod = build_two_stage_ota()


if __name__ == "__main__":
    m = build_two_stage_ota()
    print(m.generate_netlist())
    print(f"tests embedded : {m.list_tests()}")
    print(f"design specs   : {[o.spec.name for o in m.reconcile_specs().outcomes]}")

    results, recon = run_characterization(m)
    print()
    print(recon.summary())
    out_dir = example_output_dir()
    png_path = os.path.join(out_dir, "two_stage_ota_bode.png")
    fig = plot_bode(m.get_test("ac"), results[f"{m.module_name}.ac"], png_path)
    print(f"\nBode plot saved -> {png_path}")

    from silk.circuit.report import generate_report
    report = generate_report(m)
    report.add_figure(fig, title="Two-stage Miller OTA — open-loop Bode")
    report.save(os.path.join(out_dir, "two_stage_ota_report.md"))
    report.save_json(os.path.join(out_dir, "two_stage_ota_report.json"))
    report.save_html(os.path.join(out_dir, "two_stage_ota_report.html"))
    print(f"Circuit report saved -> {out_dir}/two_stage_ota_report.{{md,json,html}}")

    import matplotlib.pyplot as plt
    plt.close(fig)
