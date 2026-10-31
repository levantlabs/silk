"""Characterize a two-NMOS current mirror.

M1 is diode-connected and carries the reference current. M2 shares its gate
and sinks the output current. The match bench measures absolute percentage copy error from signed current
and checks steadiness over the final half of a transient. SPICE starts at
its DC operating point; this is a bias check, not a power-on startup test.

The zout bench measures 1/Re(-I/V) at 0.1 Hz and checks its flatness through
10 Hz. The output voltage source supplies the branch-current measurement.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.current_mirror_example``.
It prints the netlist, test names and spec results, then saves an impedance
plot and circuit reports under example_output_dir(). Needs ngspice and
sky130 models.
"""
import os

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.metrics import average, peak_to_peak, value_at
from silk.simulation import AC, Transient
from silk.examples import example_output_dir


pdk = load_pdk('sky130')

ERR_METRIC = "gain_error_pct"
ERR_TARGET_PCT = 5.0
ROUT_METRIC = "rout_ohm"
ROUT_TARGET_OHM = 50e3


I_REF = 10e-6
_VOUT_DC = 0.9
_W_DEV, _L_DEV = 4.0, 1.0


def _add_devices(mod, w, l):
    """Diode-connected M1 (reference) and mirror M2 sharing its gate."""
    with mod:
        pdk.nmos1p8("M1", d="REF", g="REF", s="VSS", b="VSS", w=w, l=l, m=1)
        pdk.nmos1p8("M2", d="OUT", g="REF", s="VSS", b="VSS", w=w, l=l, m=1)


def _add_supply_and_ref(tb, i_ref):
    """Add the supply and reference current to an independent testbench netlist."""
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=1.8))
    tb.add_source(sources.CurrentSource("iref", positive="VDD", negative="ref", dc=i_ref))


def _attach_match_test(mod, i_ref, v_out):
    """Measure absolute percentage copy error and current steadiness at one output bias."""
    tb = mod.create_test("match")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "REF": "ref", "OUT": "out",
    })
    _add_supply_and_ref(tb, i_ref)
    tb.add_source(sources.VoltageSource("vout", positive="out", negative="0", dc=v_out))
    tb.add_sim_type(Transient(step=1e-9, stop=1e-6))

    # Negate the output voltage source's current to measure the NMOS sink current.
    sink = -tb.probe("vvout", "I")
    i_out = tb.add_metric("i_out", average, sig=sink, start=0.5e-6, end=1e-6)
    ripple = tb.add_metric("i_out_pp", peak_to_peak, sig=sink,
                           start=0.5e-6, end=1e-6)
    tb.add_metric("settled_ripple_pct", 100.0 * ripple / i_ref)
    tb.add_metric(ERR_METRIC, abs(i_out / i_ref - 1.0) * 100.0)
    tb.add_spec("copy error < 5%", ERR_METRIC, "<", ERR_TARGET_PCT)
    tb.add_spec("settled ripple < 0.1%", "settled_ripple_pct", "<", 0.1)
    return tb


def _attach_zout_test(mod, i_ref, v_out):
    """Measure reciprocal real output admittance near DC and check flatness."""
    tb = mod.create_test("zout")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "REF": "ref", "OUT": "out",
    })
    _add_supply_and_ref(tb, i_ref)
    tb.add_source(sources.VoltageSource("vout", positive="out", negative="0",
                                        dc=v_out, ac=1.0))
    tb.add_sim_type(AC(variation="dec", points=20, fstart=0.1, fstop=1e9))

    # Normalize complex phasors before taking the real admittance.
    conductance = (-tb.probe("vvout", "I") / tb.probe("out")).real()
    resistance = 1.0 / conductance
    rout = tb.add_metric(ROUT_METRIC, value_at, sig=resistance, x=0.1)
    spread = tb.add_metric("rout_low_band_pp", peak_to_peak, sig=resistance,
                           start=0.1, end=10.0)
    tb.add_metric("rout_flatness_pct", 100.0 * spread / abs(rout))
    tb.add_spec("Rout flatness < 1%", "rout_flatness_pct", "<", 1.0)
    tb.add_spec("Rout > 50 kOhm", ROUT_METRIC, ">", ROUT_TARGET_OHM)
    return tb


def build_current_mirror(*, w=_W_DEV, l=_L_DEV, i_ref=I_REF,
                         v_out=_VOUT_DC) -> Module:
    """Build the NMOS current mirror with its characterization suite embedded.

    All design knobs are keyword arguments (defaulting to the module-level
    constants) so the block can be reused or swept without editing the file —
    e.g. ``build_current_mirror(i_ref=50e-6, w=8)``. The returned module owns a
    copy-accuracy testbench (``mod.get_test('match')``) and an output-resistance
    testbench (``mod.get_test('zout')``), plus the design Specs they declare
    (accessible through ``mod.reconcile_specs().outcomes``). Pass it to ``run_characterization()``.

    Parameters
    ----------
    w, l : float
        Mirror device width/length in µm (M1 = M2 for a 1:1 copy).
    i_ref : float
        Reference current forced into the diode leg, in amperes.
    v_out : float
        Output-node DC bias in volts; choose a value that keeps M2 in saturation.
    """
    mod = Module("PDK_skywater130", example_output_dir("current_mirror_example"),
                 "current_mirror.cir",
                 pdk_details=pdk, module_name="current_mirror")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "REF": Port.analog_input(), "OUT": Port.analog_output(),
    })
    _add_devices(mod, w, l)
    _attach_match_test(mod, i_ref, v_out)
    _attach_zout_test(mod, i_ref, v_out)
    return mod


def run_characterization(mod, corner="tt"):
    """Run the embedded SPICE tests and return results and spec reconciliation."""
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def plot_output_impedance(report, png_path):
    """Save a magnitude plot of the mirror's output impedance Zout(f).

    ``report`` is the AC ``SpecReport`` (e.g. ``results['current_mirror.zout']``).
    The plotted quantity is |v(out)/i(vvout)| — the output impedance the AC current
    measurement sees. Saves the PNG and returns the open Figure for embedding.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    isig = report.sim_results.load_signal("vvout", "I")
    freq = np.asarray(isig.x, dtype=float)
    vsig = report.sim_results.load_signal("out")
    zout = np.abs(np.asarray(vsig.y) / np.asarray(isig.y))
    rout = report.metrics[ROUT_METRIC]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(freq, zout, color="indigo", lw=1.6, label="|Zout|")
    ax.axhline(rout, color="crimson", ls="--", lw=0.8,
               label=f"Rout = {rout / 1e3:.0f} kOhm")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Output impedance (Ohm)")
    ax.set_title("Current mirror — output impedance")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    return fig


mod = build_current_mirror()


if __name__ == "__main__":
    m = build_current_mirror()
    print(m.generate_netlist())
    print(f"tests embedded : {m.list_tests()}")
    print(f"design specs   : {[o.spec.name for o in m.reconcile_specs().outcomes]}")

    try:
        results, recon = run_characterization(m)
        print()
        print(recon.summary())
        out_dir = example_output_dir()
        png_path = os.path.join(out_dir, "current_mirror_zout.png")
        fig = plot_output_impedance(results["current_mirror.zout"], png_path)
        print(f"\nOutput-impedance plot saved -> {png_path}")

        from silk.circuit.report import generate_report
        report = generate_report(m)
        report.add_figure(fig, title="Current mirror — output impedance")
        report.save(os.path.join(out_dir, "current_mirror_report.md"))
        report.save_json(os.path.join(out_dir, "current_mirror_report.json"))
        report.save_html(os.path.join(out_dir, "current_mirror_report.html"))
        print(f"Circuit report saved -> {out_dir}/current_mirror_report.{{md,json,html}}")

        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        print(f"\nCurrent-mirror example failed: {e}")
        print("(requires ngspice and sky130 models)")
