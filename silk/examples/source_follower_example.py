"""Characterize an NMOS or PMOS source follower.

The follower drives a capacitive load and a mirrored bias device sets its
current. The ac bench measures voltage gain and -3 dB bandwidth. The zout
bench measures output impedance magnitude at 1 Hz by injecting AC current.
The factory registers these two testbenches and their specs by default.
A first-order behavioral model supports transient simulation and calibration.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.source_follower_example``.
It prints the netlist, test names and spec results, then saves an AC plot
and circuit reports under example_output_dir(). Needs ngspice and sky130
models for the characterization.
"""
import os

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.simulation import AC
from silk.metrics import gain_at_freq, bandwidth_3dB, value_at
from silk.examples import example_output_dir


pdk = load_pdk('sky130')


GAIN_METRIC = "gain_vv"
GAIN_TARGET_VV = 0.70
BW_METRIC = "bw_hz"
BW_TARGET_HZ = 10e6
ROUT_METRIC = "rout_ohm"
ROUT_TARGET_OHM = 5000.0


_W_FOLLOWER, _L_FOLLOWER = 10.0, 0.5
_W_SINK, _L_SINK = 6.0, 1.0
_W_BIAS, _L_BIAS = 6.0, 1.0
_VIN_DC = 1.2
_I_REF = 20e-6
_CLOAD = 0.5e-12


_BEH_GAIN_NOM = 0.85
_BEH_OFFSET_NOM = 0.55
_BEH_BW_NOM = 1.0e8


def _sf_behavioral_model(ports, t, params):
    """Filter gain*vin - offset with a first-order low-pass response.

    Parameters are gain (V/V), offset (V), and bw_hz (-3 dB frequency).
    Each time step uses an exponential update with tau = 1/(2*pi*bw_hz).
    """
    import numpy as np
    vin = np.asarray(ports.VIN.v, dtype=float)
    gain, offset, bw = params.gain, params.offset, params.bw_hz
    tau = 1.0 / (2.0 * np.pi * bw)
    vout = np.empty_like(t, dtype=float)
    y = gain * float(vin[0]) - offset
    vout[0] = y
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        alpha = 1.0 - np.exp(-dt / tau)
        y += alpha * (gain * float(vin[i]) - offset - y)
        vout[i] = y
    ports.VOUT.v = vout


def _add_devices(mod, device, w_follower, l_follower, w_sink, l_sink,
                 w_bias, l_bias, c_load, internal_bias=True, bulk_to_source=False):
    """Wire the follower, mirrored bias device, optional bias diode and load.

    NMOS shifts the input down; PMOS shifts it up. With internal_bias=False,
    IBIAS is a gate-voltage input for composition with an external bias.
    For PMOS, bulk_to_source ties its isolated nwell to its source. With the
    bulk at VDD, reverse source-to-body bias increases |Vth|; tying the well
    to the source removes that body bias.
    """
    from silk.elements import elements

    if device == "nmos":
        fet = pdk.nmos1p8

        with mod:
            fet("M1", d="VDD", g="VIN", s="VOUT", b="VSS",
                w=w_follower, l=l_follower, m=1)
            fet("M2", d="VOUT", g="IBIAS", s="VSS", b="VSS",
                w=w_sink, l=l_sink, m=1)
            if internal_bias:
                fet("MBIAS", d="IBIAS", g="IBIAS", s="VSS", b="VSS",
                    w=w_bias, l=l_bias, m=1)
    elif device == "pmos":
        fet = pdk.pmos1p8

        m1_bulk = "VOUT" if bulk_to_source else "VDD"
        with mod:
            fet("M1", d="VSS", g="VIN", s="VOUT", b=m1_bulk, w=w_follower, l=l_follower, m=1)
            fet("M2", d="VOUT", g="IBIAS", s="VDD", b="VDD", w=w_sink, l=l_sink, m=1)
            if internal_bias:
                fet("MBIAS", d="IBIAS", g="IBIAS", s="VDD", b="VDD", w=w_bias, l=l_bias, m=1)
    else:
        raise ValueError(f"device must be 'nmos' or 'pmos', got {device!r}")
    with mod:
        elements.capacitor("CL", p="VOUT", n="VSS", value=c_load)


def _add_supply_and_bias(tb, i_ref, device):
    """Add supply and reference current to an independent testbench netlist.

    Current enters the NMOS bias diode and leaves the PMOS bias diode.
    """
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=1.8))
    if device == "nmos":
        tb.add_source(sources.CurrentSource("iref", positive="VDD", negative="ibias", dc=i_ref))
    else:
        tb.add_source(sources.CurrentSource("iref", positive="ibias", negative="0", dc=i_ref))


def _attach_ac_test(mod, v_in, i_ref, device, specs):
    """Register AC gain at 1 Hz and -3 dB bandwidth using the VOUT probe."""
    tb = mod.create_test("ac")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "VIN": "vin",
        "VOUT": "vout", "IBIAS": "ibias",
    })
    _add_supply_and_bias(tb, i_ref, device)
    tb.add_source(sources.VoltageSource("vin", positive="vin", negative="0", dc=v_in, ac=1.0))
    tb.add_sim_type(AC(variation="dec", points=20, fstart=1, fstop=1e9))
    tb.add_metric(GAIN_METRIC, gain_at_freq, sig=tb.probe("vout"), freq=1.0)
    tb.add_metric(BW_METRIC, bandwidth_3dB, sig=tb.probe("vout"))
    specs.append(tb.add_spec("gain > 0.7 V/V", GAIN_METRIC, ">", GAIN_TARGET_VV))
    specs.append(tb.add_spec("BW > 10 MHz", BW_METRIC, ">", BW_TARGET_HZ))
    return tb


def _attach_zout_test(mod, v_in, i_ref, device, specs):
    """Register output impedance magnitude at 1 Hz with 1 A AC injection."""
    tb = mod.create_test("zout")
    tb.add_module_under_test({
        "VDD": "VDD", "VSS": "0", "VIN": "vin",
        "VOUT": "vout", "IBIAS": "ibias",
    })
    _add_supply_and_bias(tb, i_ref, device)
    tb.add_source(sources.VoltageSource("vin", positive="vin", negative="0", dc=v_in))

    tb.add_source(sources.CurrentSource("iout", positive="0", negative="vout", ac=1.0))
    tb.add_sim_type(AC(variation="dec", points=20, fstart=1, fstop=1e9))

    tb.add_metric(ROUT_METRIC, value_at, sig=tb.probe("vout").mag(), x=1.0)
    specs.append(tb.add_spec("Rout < 5 kOhm", ROUT_METRIC, "<", ROUT_TARGET_OHM))
    return tb


def build_source_follower(*, device="nmos", with_tests=True, internal_bias=True,
                          bulk_to_source=False,
                          w_follower=None, l_follower=_L_FOLLOWER,
                          w_sink=_W_SINK, l_sink=_L_SINK,
                          w_bias=_W_BIAS, l_bias=_L_BIAS,
                          c_load=_CLOAD, v_in=None, i_ref=None, db_path=None,
                          model_params=None, specs=None) -> Module:
    """Build the source follower with its characterization suite embedded.

    All design knobs are keyword arguments (defaulting to the module-level
    constants) so the block can be reused or swept without editing the file —
    e.g. ``build_source_follower(c_load=2e-12, i_ref=40e-6)``. ``device``
    selects the flavor:

      * ``'nmos'`` — NMOS common-drain, VOUT = VIN - Vgs (level shifts **down**).
        The default.
      * ``'pmos'`` — PMOS common-drain, VOUT = VIN + |Vgs| (level shifts **up**);
        used as a bandgap level shifter that lifts ~V_be into an NMOS-input
        op-amp's common-mode range.

    The returned module owns an AC testbench (``mod.get_test('ac')``, gain and
    bandwidth) and an output-resistance testbench (``mod.get_test('zout')``), plus
    the design Specs they declare. Pass it to ``run_characterization()``.

    With internal_bias=True, IBIAS is a reference-current input: an on-chip diode-connected
    Mbias turns it into M2's gate bias (a current mirror), so the bias current
    is set by ``i_ref`` and the M2/Mbias size ratio — not by an ideal voltage.

    Set internal_bias=False only with with_tests=False and supply an external
    voltage bias on IBIAS in the parent circuit.

    Parameters
    ----------
    device : {'nmos', 'pmos'}   Follower flavor (sets the level-shift direction).
    w_follower, l_follower : float   Follower M1 width/length in µm.
    w_sink, l_sink : float           Bias current-source M2 width/length in µm.
    w_bias, l_bias : float           Diode-connected bias device width/length in µm.
    c_load : float                   Output load capacitance in F (sets bandwidth).
    model_params : dict, optional    Receives the live behavioral parameter values.
    specs : list, optional           Receives the authored Spec handles.
    v_in : float, optional           Input bias in V. Defaults to 1.2 for 'nmos'
                                     and 0.5 for 'pmos'.
    i_ref : float                    Reference current into the bias diode in A.
    db_path : str, optional         Database for explicitly stored characterization.
    """

    if with_tests and not internal_bias:
        raise ValueError("internal_bias=False requires with_tests=False; "
                         "provide an external voltage bias on IBIAS")
    # PMOS defaults use a wider follower and lower current for bias headroom.
    if v_in is None:
        v_in = _VIN_DC if device == "nmos" else 0.5
    if w_follower is None:
        w_follower = _W_FOLLOWER if device == "nmos" else 40.0
    if i_ref is None:
        i_ref = _I_REF if device == "nmos" else 10e-6

    # PMOS shifts the input upward, so its behavioral offset is negative.
    offset_nom = _BEH_OFFSET_NOM if device == "nmos" else -_BEH_OFFSET_NOM
    mod = Module("PDK_skywater130", example_output_dir("source_follower_example"),
                 f"source_follower_{device}.cir",
                 pdk_details=pdk, module_name=f"source_follower_{device}", db_path=db_path)
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "VIN": Port.analog_input(), "VOUT": Port.analog_output(),
        "IBIAS": Port.analog_input(),
    })
    _add_devices(mod, device, w_follower, l_follower, w_sink, l_sink,
                 w_bias, l_bias, c_load, internal_bias=internal_bias,
                 bulk_to_source=bulk_to_source)

    if model_params is None:
        model_params = {}
    model_params.update(gain=_BEH_GAIN_NOM, offset=offset_nom, bw_hz=_BEH_BW_NOM)
    mod.set_behavioral_model(_sf_behavioral_model, execution="vector", params=model_params)
    if specs is None:
        specs = []
    if with_tests:
        _attach_ac_test(mod, v_in, i_ref, device, specs)
        _attach_zout_test(mod, v_in, i_ref, device, specs)
    return mod


def run_characterization(mod, corner="tt"):
    """Run the embedded SPICE tests and return results and spec reconciliation."""
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def plot_transfer_function(tb, report, png_path):
    """Plot the AC bench's current VOUT response and mark its bandwidth.

    Pass results[f'{mod.module_name}.ac'] as report. Save the PNG at
    png_path and return the open Figure for embedding in a circuit report.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = tb.probe("vout")
    gain_vv = report.metrics[GAIN_METRIC]
    bw_hz = report.metrics[BW_METRIC]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sig.plot(ax=ax, color="seagreen", lw=1.6, label="|vout/vin|")
    ax.axvline(bw_hz, color="crimson", ls="--", lw=0.8,
               label=f"f_3dB = {bw_hz / 1e6:.1f} MHz")
    ax.set_title(f"Source follower — AC transfer (gain = {gain_vv:.3f} V/V)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    return fig


mod = build_source_follower()


if __name__ == "__main__":
    specs = []
    m = build_source_follower(specs=specs)
    print(m.generate_netlist())
    print(f"tests embedded : {m.list_tests()}")
    print(f"design specs   : {[s.name for s in specs]}")

    results, recon = run_characterization(m)
    print()
    print(recon.summary())
    out_dir = example_output_dir()
    png_path = os.path.join(out_dir, "source_follower_ac.png")
    fig = plot_transfer_function(m.get_test("ac"),
                                 results[f"{m.module_name}.ac"], png_path)
    print(f"\nTransfer-function plot saved -> {png_path}")

    from silk.circuit.report import generate_report
    report = generate_report(m)
    report.add_figure(fig, title="Source follower — AC transfer function")
    report.save(os.path.join(out_dir, "source_follower_report.md"))
    report.save_json(os.path.join(out_dir, "source_follower_report.json"))
    report.save_html(os.path.join(out_dir, "source_follower_report.html"))
    print(f"Circuit report saved -> {out_dir}/source_follower_report.{{md,json,html}}")

    import matplotlib.pyplot as plt
    plt.close(fig)
