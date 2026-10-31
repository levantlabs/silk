"""Charge pump for the PLL loop filter.

A reference current drives NMOS and PMOS mirrors. Series switches source
current into OUT for UP and sink it for DN; an inverter drives the PMOS
switch. The pump uses a 1.8 V supply.

The match transient measures source and sink currents at mid-rail.
Two DC sweeps find the output-voltage range where each current remains
within 10% of its mid-rail value. The leakage test measures current with
both switches off. The behavioral PLL imports stored nominal values;
pll_calibrate.py compares them with fresh measurements.

Run from the repository root with ngspice and sky130 models installed:
    MPLBACKEND=Agg .venv/bin/python -m silk.examples.charge_pump_example

The script prints spec results, currents, mismatch, compliance and leakage,
and saves the compliance curves.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk.simulation import DC, Transient
from silk.metrics import average

from silk.examples.ring_oscillator_example import build_inverter
from silk.examples import example_output_dir

pdk = load_pdk('sky130')

_VDD = 1.8
I_CP = 10e-6


_W_N, _W_P, _L_MIR = 4.0, 8.0, 1.0
_W_SW, _L_SW = 2.0, 0.15

MATCH_METRIC = "mismatch_pct"
IUP_METRIC = "i_up_ua"
IDN_METRIC = "i_dn_ua"
UPHI_METRIC = "up_compliance_hi_v"
DNLO_METRIC = "dn_compliance_lo_v"
LEAK_METRIC = "leak_na"

MATCH_MAX_PCT = 10.0
LEAK_MAX_NA = 1.0


# Run pll_calibrate.py to compare measurements with these nominal constants.
I_UP_NOM_UA = 9.78
I_DN_NOM_UA = 10.12
LEAK_NOM_NA = 0.042
VCOMP_LO_NOM_V = 0.19
VCOMP_HI_NOM_V = 1.62


def build_charge_pump(*, i_cp=I_CP, with_tests=True) -> Module:
    """Build a reference, source/sink mirrors, series switches and an inverter."""
    mod = Module("PDK_skywater130", example_output_dir(), "charge_pump.cir", pdk_details=pdk,
                 module_name="charge_pump")
    mod.define_module_ports({
        "VDD": Port.supply(), "VSS": Port.ground(),
        "IBIAS": Port.analog_input(),
        "UP": Port.analog_input(), "DN": Port.analog_input(),
        "OUT": Port.analog_output(),
    })
    with mod:

        pdk.nmos1p8("MBN", d="IBIAS", g="IBIAS", s="VSS", b="VSS",
                    w=_W_N, l=_L_MIR)
        pdk.nmos1p8("MN_SINK", d="nds", g="IBIAS", s="VSS", b="VSS",
                    w=_W_N, l=_L_MIR)

        pdk.nmos1p8("MN2", d="pdio", g="IBIAS", s="VSS", b="VSS",
                    w=_W_N, l=_L_MIR)
        pdk.pmos1p8("MBP", d="pdio", g="pdio", s="VDD", b="VDD",
                    w=_W_P, l=_L_MIR)
        pdk.pmos1p8("MP_SRC", d="nus", g="pdio", s="VDD", b="VDD",
                    w=_W_P, l=_L_MIR)

        pdk.pmos1p8("MSW_UP", d="OUT", g="upb", s="nus", b="VDD",
                    w=_W_SW, l=_L_SW)
        pdk.nmos1p8("MSW_DN", d="OUT", g="DN", s="nds", b="VSS",
                    w=_W_SW, l=_L_SW)
    inv = build_inverter()
    mod.add_module(inv, "inv_up", nets={"VDD": "VDD", "VSS": "VSS",
                                         "A": "UP", "Y": "upb"})
    if with_tests:
        attach_match_test(mod, i_cp=i_cp)
        attach_compliance_tests(mod, i_cp=i_cp)
        attach_leakage_test(mod, i_cp=i_cp)
    return mod


def _cp_rig(tb, *, i_cp, v_out, up="0", dn="0"):
    """Bias the pump and hold OUT with a voltage-source ammeter."""
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
                              "UP": up, "DN": dn, "OUT": "vout"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    tb.add_source(sources.CurrentSource("iref", positive="VDD", negative="ibias",
                                        dc=i_cp))

    # The source holds OUT and measures its branch current.
    tb.add_source(sources.VoltageSource("vout", positive="vout", negative="0",
                                        dc=v_out))


def _i_out_ua(ckt, *, settle=None):
    """Pump current into the OUT ammeter, in µA (sign: + = sourcing into OUT)."""
    sig = ckt.load_signal("vvout", "I")
    if settle is not None:
        sig = sig.window(settle, sig.x[-1])
    return average(sig) * 1e6


def attach_match_test(mod, *, i_cp, v_out=0.9):
    """Measure source/sink current magnitudes at mid-rail and their mismatch."""
    tb = mod.create_test("match")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "IBIAS": "ibias",
                              "UP": "up", "DN": "dn", "OUT": "vout"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD))
    tb.add_source(sources.CurrentSource("iref", positive="VDD", negative="ibias",
                                        dc=i_cp))
    # The source holds OUT and measures its branch current.
    tb.add_source(sources.VoltageSource("vout", positive="vout", negative="0",
                                        dc=v_out))
    tb.add_source(sources.VoltageSource(
        "up", positive="up", negative="0", dc=0,
        pwl=sources.PWL().add(0, _VDD).add(0.99e-6, _VDD).add(1.01e-6, 0)))
    tb.add_source(sources.VoltageSource(
        "dn", positive="dn", negative="0", dc=0,
        pwl=sources.PWL().add(0, 0).add(0.99e-6, 0).add(1.01e-6, _VDD)))
    tb.add_sim_type(Transient(step=1e-9, stop=2e-6))

    current_ua = tb.probe("vvout", "I") * 1e6
    i_up = tb.add_metric(IUP_METRIC, average,
                         sig=current_ua.window(0.3e-6, 0.9e-6).mag())
    i_dn = tb.add_metric(IDN_METRIC, average,
                         sig=current_ua.window(1.3e-6, 1.9e-6).mag())
    tb.add_metric(MATCH_METRIC, abs(i_up - i_dn) / (0.5 * (i_up + i_dn)) * 100)
    tb.add_spec(f"up/down mismatch < {MATCH_MAX_PCT:.0f}%", MATCH_METRIC, "<",
                MATCH_MAX_PCT)
    return tb


def attach_compliance_tests(mod, *, i_cp):
    """Sweep OUT and find each mirror's compliance limit.

    Starting at mid-rail, find the contiguous range within 10% of the
    mid-rail current. Sweep toward the rail for UP and toward ground for DN.
    """
    for name, up, dn, metric, spec_op, spec_val, desc in (
            ("compl_up", "VDD", "0", UPHI_METRIC, ">", 1.5,
             "UP compliance to >1.5 V"),
            ("compl_dn", "0", "VDD", DNLO_METRIC, "<", 0.3,
             "DN compliance to <0.3 V")):
        tb = mod.create_test(name)
        _cp_rig(tb, i_cp=i_cp, v_out=0.9, up=up, dn=dn)
        tb.add_sim_type(DC(source="vvout", start=0.05, stop=_VDD - 0.05,
                           step=0.01))
        hi = metric == UPHI_METRIC

        def _edge(ckt, _hi=hi):
            sig = ckt.load_signal("vvout", "I")
            v = np.asarray(sig.x, dtype=float)
            i = np.abs(np.asarray(sig.y, dtype=float))
            nom = float(np.interp(0.9, v, i))
            ok = np.abs(i - nom) < 0.10 * nom
            mid = int(np.argmin(np.abs(v - 0.9)))
            j = mid
            if _hi:
                while j + 1 < len(v) and ok[j + 1]:
                    j += 1
            else:
                while j - 1 >= 0 and ok[j - 1]:
                    j -= 1
            return float(v[j])

        tb.add_metric(metric, _edge)
        tb.add_spec(desc, metric, spec_op, spec_val)
    return mod


def attach_leakage_test(mod, *, i_cp, v_out=0.9):
    """Measure off-state current magnitude at mid-rail."""
    tb = mod.create_test("leakage")
    _cp_rig(tb, i_cp=i_cp, v_out=v_out, up="0", dn="0")
    tb.add_sim_type(Transient(step=10e-9, stop=5e-6))
    tb.add_metric(LEAK_METRIC,
                  lambda ckt: abs(_i_out_ua(ckt, settle=1e-6)) * 1e3)
    tb.add_spec(f"off-state leakage < {LEAK_MAX_NA:.0f} nA", LEAK_METRIC, "<",
                LEAK_MAX_NA)
    return tb


def run_characterization(mod, corner="tt"):
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def save_plot(results, path):
    """Plot source and sink compliance curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for test, color, label in (("compl_up", "#0072B2", "UP (sourcing)"),
                               ("compl_dn", "#D55E00", "DN (sinking)")):
        sig = results[f"charge_pump.{test}"].sim_results.load_signal("vvout", "I")
        v = np.asarray(sig.x, dtype=float)
        i = np.abs(np.asarray(sig.y, dtype=float)) * 1e6
        ax.plot(v, i, color=color, lw=1.8, label=label)
    ax.axhline(I_CP * 1e6, color="#5a5a6e", ls="--", lw=0.8,
               label=f"nominal {I_CP*1e6:.0f} µA")
    ax.axvline(1.405, color="#009E73", ls=":", lw=1.2,
               label="illustrative VCTRL = 1.405 V")
    ax.set_xlabel("V_OUT (V)")
    ax.set_ylabel("|I_pump| (µA)")
    ax.set_title("Charge-pump compliance curves")
    ax.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    print("=" * 70)
    print(f"Charge pump — I_cp = {I_CP*1e6:.0f} µA (the PLL's pump)")
    print("=" * 70)
    mod = build_charge_pump()
    results, recon = run_characterization(mod)
    print(recon.summary())
    r = lambda t, m: results[f"{mod.module_name}.{t}"].metrics[m]
    print(f"  I_up  = {r('match', IUP_METRIC):6.2f} µA   "
          f"I_dn = {r('match', IDN_METRIC):6.2f} µA   "
          f"mismatch = {r('match', MATCH_METRIC):.2f}%")
    print(f"  compliance: DN good down to {r('compl_dn', DNLO_METRIC):.2f} V, "
          f"UP good up to {r('compl_up', UPHI_METRIC):.2f} V")
    print(f"  off-state leakage = {r('leakage', LEAK_METRIC):.3f} nA")
    print("  (in-code *_NOM store feeds pll_example; refresh via pll_calibrate.py)")
    print(f"  saved {save_plot(results, os.path.join(OUT, 'charge_pump.png'))}")
