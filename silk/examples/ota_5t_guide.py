"""5T OTA — the silk agent-guide worked example.

NMOS input pair (M1/M2), PMOS current-mirror load (M3/M4, M3 diode-
connected), NMOS tail source (M5, gate = IBIAS).

Builds a five-transistor OTA with AC, transient and noise tests.
``add_metric`` registers measurements and ``add_spec`` records their
requirements. ``run_characterization()`` runs the tests and reconciles
measurements against the design specs.
"""
import os

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.simulation import AC, Transient, Noise
from silk.metrics import dc_gain_dB, bandwidth_3dB, integrated_noise
from silk.metrics.timing import settling_time
from silk.examples import example_output_dir


# Device dimensions are in microns; tail gate bias is in volts.
PAIR_W, PAIR_L = 4.0, 0.5
MIRROR_W, MIRROR_L = 8.0, 0.5
TAIL_W, TAIL_L = 6.0, 1.0
TAIL_BIAS = 0.9

_PDK = None


def get_pdk():
    """Load and cache the sky130 catalog on first use."""
    global _PDK
    if _PDK is None:
        _PDK = load_pdk('sky130')
    return _PDK


def __getattr__(name):
    """Keep ``pdk`` and ``mod`` importable without building them at import time.

    PEP 562. Both attributes stay available to importers — they are simply
    resolved when asked for rather than when the module loads. Note this hook
    is *not* consulted for global lookups inside this module, which is why the
    functions below call ``get_pdk()`` explicitly.
    """
    if name == 'pdk':
        return get_pdk()
    if name == 'mod':
        return build_5t_ota()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


# Metric names are shared by the testbench metric and its spec, so they all
# refer to one thing. tb.add_spec promotes each to a module-level design Spec.
GAIN_METRIC = "gain_dB"
GAIN_TARGET_DB = 20.0
BW_METRIC = "bw_hz"
BW_TARGET_HZ = 10e6      # 10 MHz; measured ~178 MHz at tt (no explicit load cap)
SETTLE_METRIC = "t_settle_1pct"
SETTLE_TARGET_S = 100e-9  # 100 ns; measured ~1 ns (single dominant pole, fast)
NOISE_METRIC = "vn_in_uV"
NOISE_TARGET_UV = 2000.0   # 2 mV; measured ~581 uV input-referred at tt
_STEP_T = 1e-6            # transient input-step edge time
_STEP_LO, _STEP_HI = 0.9, 0.95   # 50 mV input step about the 0.9 V common mode


def _add_devices(mod, *, pair_w, pair_l, mirror_w, mirror_l, tail_w, tail_l):
    """Wire the two internal nets and five transistors onto ``mod``.

    Devices are constructed inside a ``with mod:`` block, which auto-registers
    whatever is created in it — devices, sources and passives alike. Each PDK
    device is a class, so pins and parameters are plain keyword arguments: a
    mis-typed pin fails right here at construction, rather than surviving inside
    a dict to become a mystery later.
    """

    pdk = get_pdk()
    with mod:
        # Input pair. VINP drains into the mirror diode (vint), VINM into VOUT.
        pdk.nmos1p8("M1", d="vint", g="VINP", s="tail", b="VSS", w=pair_w, l=pair_l)
        pdk.nmos1p8("M2", d="VOUT", g="VINM", s="tail", b="VSS", w=pair_w, l=pair_l)
        # PMOS mirror load: M3 is the diode, M4 the output leg.
        pdk.pmos1p8("M3", d="vint", g="vint", s="VDD", b="VDD", w=mirror_w, l=mirror_l)
        pdk.pmos1p8("M4", d="VOUT", g="vint", s="VDD", b="VDD", w=mirror_w, l=mirror_l)
        # Tail current source, biased from IBIAS.
        pdk.nmos1p8("M5", d="tail", g="IBIAS", s="VSS", b="VSS", w=tail_w, l=tail_l)


def _attach_ac_test(mod, *, tail_bias):
    """Register AC gain and bandwidth metrics and their requirements.

    The test uses 1 V AC on VINP with VINM AC-grounded. Declare the vout
    probe before running; it resolves against the bench's current results.
    ``add_spec`` records each requirement on the owning module.
    """
    tb = mod.create_test("ac")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "VINP": "vinp",
        "VINM": "vinm", "VOUT": "vout", "IBIAS": "ibias",
    })
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=1.8))
    tb.add_source(sources.VoltageSource("vbias", positive="ibias", negative="0", dc=tail_bias))
    # AC stimulus: 1 V AC on VINP; VINM at the same DC bias, AC-grounded.
    tb.add_source(sources.VoltageSource("vinp", positive="vinp", negative="0", dc=0.9, ac=1.0))
    tb.add_source(sources.VoltageSource("vinm", positive="vinm", negative="0", dc=0.9))
    tb.add_sim_type(AC(variation="dec", points=20, fstart=1, fstop=1e9))
    vout = tb.probe("vout")
    tb.add_metric(GAIN_METRIC, dc_gain_dB, sig=vout)
    tb.add_metric(BW_METRIC, bandwidth_3dB, sig=vout)
    tb.add_spec("DC gain > 20 dB", GAIN_METRIC, ">", GAIN_TARGET_DB)
    tb.add_spec("BW > 10 MHz", BW_METRIC, ">", BW_TARGET_HZ)
    return tb


def _attach_transient_test(mod, *, tail_bias):
    """Register a closed-loop step-response testbench *on* the module.

    Open-loop, a ~50x OTA would just rail on any real input step, so the
    testbench wires the OTA as a **unity-gain follower** — VINM is tied to
    VOUT (negative feedback) by mapping both ports to the same ``vout`` net.
    A 50 mV step on VINP then produces a settling step at VOUT, which the
    ``t_settle_1pct`` metric measures. Same embedded pattern as the AC test:
    ``tb.add_spec`` records the requirement on the owning module.
    """
    tb = mod.create_test("tran")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "VINP": "vinp",
        "VINM": "vout", "VOUT": "vout", "IBIAS": "ibias",  # VINM↔VOUT feedback
    })
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=1.8))
    tb.add_source(sources.VoltageSource("vbias", positive="ibias", negative="0", dc=tail_bias))
    tb.add_source(sources.VoltageSource(
        "vinp", positive="vinp", negative="0",
        pulse=sources.Pulse(v1=_STEP_LO, v2=_STEP_HI, td=_STEP_T,
                            tr=1e-9, tf=1e-9, pw=5e-6, per=10e-6)))
    tb.add_sim_type(Transient(step=1e-9, stop=2e-6))
    # target=None uses the output's final value; tol is relative to the
    # swing from initial to that value.
    tb.add_metric(SETTLE_METRIC, settling_time, sig=tb.probe("vout"),
                  initial=_STEP_LO, tol=0.01, start=_STEP_T)
    tb.add_spec("settles < 100 ns", SETTLE_METRIC, "<", SETTLE_TARGET_S)
    return tb


def _attach_noise_test(mod, *, tail_bias):
    """Register a noise testbench *on* the module.

    Open-loop AC config (VINM AC-grounded at bias), with VINP carrying the
    AC=1 reference the .noise analysis needs. ngspice integrates the per-
    device noise to onoise/inoise spectra; we report the input-referred
    integrated noise over 1 Hz - 1 GHz.
    """
    tb = mod.create_test("noise")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "VINP": "vinp",
        "VINM": "vinm", "VOUT": "vout", "IBIAS": "ibias",
    })
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=1.8))
    tb.add_source(sources.VoltageSource("vbias", positive="ibias", negative="0", dc=tail_bias))
    tb.add_source(sources.VoltageSource("vinp", positive="vinp", negative="0", dc=0.9, ac=1.0))
    tb.add_source(sources.VoltageSource("vinm", positive="vinm", negative="0", dc=0.9))
    tb.add_sim_type(Noise(output="vout", ref="0", source="vinp",
                          variation="dec", points=10, fstart=1, fstop=1e9))
    # Python-callable metric style, kept deliberately; the probe form is demonstrated
    # above for GAIN_METRIC/BW_METRIC (tb.probe(...) + add_metric(fn, sig=...)).
    tb.add_metric(NOISE_METRIC,
                  lambda ckt: integrated_noise(ckt.load_signal("noise_in"), 1, 1e9) * 1e6)
    tb.add_spec("input-referred noise < 2 mV", NOISE_METRIC, "<", NOISE_TARGET_UV)
    return tb


def build_5t_ota(*, pair_w=PAIR_W, pair_l=PAIR_L,
                 mirror_w=MIRROR_W, mirror_l=MIRROR_L,
                 tail_w=TAIL_W, tail_l=TAIL_L, tail_bias=TAIL_BIAS) -> Module:
    """Build the 5T OTA with its characterization suite embedded.

    The returned module knows how to characterize itself: it owns an AC
    testbench (``mod.get_test('ac')``, open-loop gain/bandwidth), a transient
    testbench (``mod.get_test('tran')``, unity-gain step settling), and a noise
    testbench (``mod.get_test('noise')``, input-referred noise spectral density),
    and the design Specs this factory declares for those testbenches to check.
    ``run_characterization()`` runs the tests and reconciles their measurements
    against those Specs.

    Sizes are in microns; ``tail_bias`` is the M5 gate voltage in volts,
    applied by every embedded bench independently of its input stimulus.
    Rebuild with keyword arguments to characterize a changed design::

        wider = build_5t_ota(pair_w=8.0)
        report = wider.get_test('ac').run(corner='tt')
        print(report.metrics['gain_dB'])

    With sky130 and ngspice-42 at tt, this wider input pair raises the
    measured DC gain from 34.850731 dB (defaults) to 34.956581 dB.
    """
    mod = Module("PDK_skywater130", example_output_dir("ota_5t_guide"), "ota5t.cir",
                 pdk_details=get_pdk(), module_name="ota_5t")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "VINP": Port.analog_input(), "VINM": Port.analog_input(),
        "VOUT": Port.analog_output(), "IBIAS": Port.analog_input(),
    })
    _add_devices(mod, pair_w=pair_w, pair_l=pair_l,
                 mirror_w=mirror_w, mirror_l=mirror_l,
                 tail_w=tail_w, tail_l=tail_l)
    _attach_ac_test(mod, tail_bias=tail_bias)
    _attach_transient_test(mod, tail_bias=tail_bias)
    _attach_noise_test(mod, tail_bias=tail_bias)
    return mod


def run_characterization(mod, corner="tt"):
    """Drive the embedded loop; return ``(results, reconcile_report)``.

    Runs every testbench the module owns (each run records its spec's
    measurement as a side effect), then reconciles the module's whole
    spec tree. Requires ngspice + a real sky130 lib.
    """
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def plot_transfer_function(tb, report, png_path):
    """Save a Bode magnitude plot of the OTA's open-loop transfer function.

    ``tb`` is the AC testbench (``mod.get_test('ac')``) the metrics were
    declared on; ``report`` is its ``SpecReport`` — e.g.
    ``results['ota_5t.ac']`` from ``run_characterization``, or a direct
    ``tb.run()``.

    The curve is the bench's own ``vout`` probe, which is the last step of
    the taught program: probe -> add_metric -> run -> plot. A probe holds no
    samples, only the name and the bench, so one declared before the run
    reads the bench's current results. Pass the report from the same run as
    the bench's current AC results. The AC ``vout`` signal is complex, so
    ``Signal.plot()`` renders it as magnitude in dB on a log-frequency axis
    automatically — we just annotate the DC gain and 3 dB bandwidth on top.
    """
    import matplotlib
    matplotlib.use("Agg")          # headless-safe; no display needed
    import matplotlib.pyplot as plt

    sig = tb.probe("vout")
    gain_db = report.metrics[GAIN_METRIC]
    bw_hz = report.metrics[BW_METRIC]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sig.plot(ax=ax, color="steelblue", lw=1.6, label="|vout/vin|")  # complex → dB vs log-f
    ax.axhline(gain_db, color="gray", ls="--", lw=0.8,
               label=f"DC gain = {gain_db:.1f} dB")
    ax.axvline(bw_hz, color="crimson", ls="--", lw=0.8,
               label=f"f_3dB = {bw_hz / 1e6:.1f} MHz")
    ax.set_title("5T OTA — open-loop AC transfer function")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    m = build_5t_ota()
    print(m.generate_netlist())
    print(f"tests embedded : {m.list_tests()}")
    print(f"design specs   : {[s.name for s in m._specs]}")

    # Full SPICE characterization requires ngspice and installed sky130 models.
    results, recon = run_characterization(m)
    print()
    print(recon.summary())
    out_dir = example_output_dir()
    ac_tb = m.get_test("ac")
    png = plot_transfer_function(ac_tb, results["ota_5t.ac"],
                                 os.path.join(out_dir, "ota_5t_transfer.png"))
    print(f"\nTransfer-function plot saved -> {png}")

    # The plot half, with no drawing code at all -- and no metric either.
    # A probe is just a name plus the bench it belongs to, so one minted
    # here reads the run that already finished, and tb.plot() picks the
    # panel layout and the axis treatment off the signal itself. The
    # annotated Bode above is what the same idea looks like when you want
    # to draw on top of it.
    import matplotlib.pyplot as plt
    tran_tb = m.get_test("tran")
    step_png = os.path.join(out_dir, "ota_5t_step_probe.png")
    fig = tran_tb.plot(tran_tb.probe("vout"), show=False)
    fig.savefig(step_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Step-response probe plot     -> {step_png}")

    # Open the AC test's saved rawfile in the interactive waveform explorer.
    from silk.signals.view import emit_view
    view = emit_view(results["ota_5t.ac"].sim_results.results_file,
                     os.path.join(out_dir, "ota_5t_view.html"),
                     title="5T OTA — AC characterization")
    print(f"Waveform explorer saved       -> {view.html_path}")
