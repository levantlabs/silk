"""Transistor-level differential SAR ADC and behavioral comparisons.

The capacitor DAC uses binary-weighted capacitors, a unit dummy capacitor,
NMOS bottom-plate switches and top-plate reset switches. High-reference
switches are 28 um wide; the other switches are 2 um wide, all at L=0.15 um.
A StrongArm comparator drives either a Python bit sequencer or a controller
built from standard cells. The capacitor model uses charge conservation.

The ideal conversion residue common mode is
VCM + (Cs/Ctot)*((vrefp+vrefn)/2 - VIN_CM).
VIN_CM is the reference midpoint so the residue stays near VDD/2.
The dummy capacitor gives loop comparison terms 2**n/(2**n-1) for gain
and VREFP/(2**n-1) for trip offset. These are schematic terms, not a fit.
See docs/examples.md for SAR accuracy results and limitations.

Run .venv/bin/python -m silk.examples.sar_adc_spice from the repository root
for the DAC transfer sweep. Arguments loop, conv, self, transfer and sine
select loop comparisons, residue replay, self-clocked conversion, transfer
and sine tests. Requires ngspice and sky130 models. Each run prints its
measurements and saves plots and HTML in the configured example output directory.
"""
import base64
import os
import sys


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from silk import PortDir, PortRole, Port, VoltageDomain
from silk import load_pdk
from silk.circuit.modules import Module
from silk.simulation import Transient
from silk.sources.classes import VoltageSource
from silk.sources.waveforms import PWL, Pulse, Sine
from silk.elements import elements as el

from silk.examples import sar_adc_example as sar
from silk.metrics import value_at, sample, sndr as metric_sndr, enob as metric_enob
from silk.signals import Signal, codes_from_bits
from silk.examples import example_output_dir
OUT = example_output_dir(create=False)

# --- cap-DAC design constants ---------------------------------------------------
VDD = 1.8
AVDD = VoltageDomain("avdd", nominal=VDD, tolerance=0.1)
CUNIT = 4e-15                    # unit capacitor (F); bit k = 2**k * CUNIT
VREFP, VREFN = 0.9, 0.0         # reference rails (NMOS-friendly; span = 0.9 V)
# Top-plate reset level at mid-supply so the differential top plates swing symmetrically
# about VDD/2 and neither leaves [0, VDD].
VCM = 0.9
# Reference-midpoint sampling keeps the conversion residue near mid-supply.
VIN_CM = 0.5 * (VREFP + VREFN)
SW_W, SW_L = 2.0, 0.15          # bottom-plate / reset NMOS switch sizing (µm)
SW_REFP_W = 28.0               # compensate the lower VGS/body effect at vrefp

# per-conversion transient: reset window (top plates -> vcm) then convert window
T_RESET = 4e-9
T_CONV = 16e-9
T_RISE = 0.2e-9


def _pdk():
    return load_pdk("sky130")


def _cap_dac_voltages(n, cunit, sample_input, t, *, smp, vcm, vrefp, vrefn,
                      bits, vinp=None, vinn=None):
    """Return cap-DAC top-plate arrays from charge conservation on the given grid.

    Inputs are arrays on ``t``; ``bits[k]`` drives binary-weighted capacitor k.
    While ``smp`` is high the top plates track vcm and the bottoms hold the sample
    value (input in SAR mode, vrefn/vrefp in DAC mode). At sample-to-convert the
    top-plate charge latches; each set bit then selects vrefp (P) / vrefn (N).
    """
    ck = [cunit * (2 ** k) for k in range(n)]
    csum = sum(ck)
    ctot = csum + cunit                                   # + 1-unit dummy
    vrp, vrn = vrefp, vrefn
    bk = bits
    toutp = np.empty(len(t))
    toutn = np.empty(len(t))
    qp = qn = 0.0
    latched = False
    for i in range(len(t)):
        if smp[i] > 0.9 * VDD:                        # sample / reset
            toutp[i] = toutn[i] = vcm[i]
            latched = False
        else:                                         # convert (top floats)
            if not latched:
                sbp = vinp[i] if sample_input else vrn[i]
                sbn = vinn[i] if sample_input else vrp[i]
                qp = ctot * vcm[i] - (csum * sbp + cunit * vrn[i])
                qn = ctot * vcm[i] - (csum * sbn + cunit * vrp[i])
                latched = True
            sump = sum(ck[k] * (vrp[i] if bk[k][i] > 0.9 * VDD else vrn[i])
                       for k in range(n)) + cunit * vrn[i]
            sumn = sum(ck[k] * (vrn[i] if bk[k][i] > 0.9 * VDD else vrp[i])
                       for k in range(n)) + cunit * vrp[i]
            toutp[i] = (qp + sump) / ctot
            toutn[i] = (qn + sumn) / ctot
    return toutp, toutn


def _cap_dac_behavioral(n, cunit, sample_input):
    """Ports wrapper sharing the array equations used by the SPICE overlays."""
    def model(ports, t, params):
        toutp, toutn = _cap_dac_voltages(
            n, cunit, sample_input, t,
            smp=np.asarray(ports.smp.v, float),
            vcm=np.asarray(ports.vcm.v, float),
            vrefp=np.asarray(ports.vrefp.v, float),
            vrefn=np.asarray(ports.vrefn.v, float),
            vinp=np.asarray(ports.vinp.v, float) if sample_input else None,
            vinn=np.asarray(ports.vinn.v, float) if sample_input else None,
            bits=[np.asarray(getattr(ports, f"b{k}").v, float) for k in range(n)])
        ports.toutp.v = toutp
        ports.toutn.v = toutn
    return model


def build_cap_dac(n, pdk, out_dir, *, cunit=CUNIT, sample_input=False,
                  sw_refp_w=SW_REFP_W):
    """Differential binary-weighted charge-redistribution cap-DAC (n-bit).

    Bit k has a ``2**k * cunit`` cap on each side; a 1-unit dummy per side makes the
    total ``2**n`` units. Bottom plates switch between vrefp/vrefn via NMOS gated by
    the code bit ``b{k}`` (and its complement ``b{k}b``); set bit -> P bottom to
    vrefp, N bottom to vrefn (differential). ``smp`` closes the top-plate reset
    switches to vcm. Top plates ``toutp``/``toutn`` are the outputs.

    ``sw_refp_w`` sizes both arrays' high-reference switches (default 28 µm).
    Their lower gate drive and body effect require extra width at vrefp.

    ``sample_input=True`` adds the SAR sampling front end: extra ``vinp``/``vinn``
    ports and NMOS switches that connect the bottom plates to the input during
    ``smp`` (the caller holds ``b=b{k}b=0`` in the sample phase so the convert
    switches are off). This makes the array sample the input and then binary-search
    against the references — a full charge-redistribution SAR — instead of the
    code-to-voltage DAC used by the default sweep."""
    m = Module("PDK_skywater130", out_dir, f"cap_dac_{n}b.cir",
               pdk_details=pdk, module_name=f"cap_dac_{n}b")
    for k in range(n):
        ck = cunit * (2 ** k)
        m.add(el.capacitor(f"cp{k}", p="toutp", n=f"pbot{k}", value=ck))
        m.add(el.capacitor(f"cn{k}", p="toutn", n=f"nbot{k}", value=ck))
        # P bottom: -> vrefp when b{k}=1, -> vrefn when b{k}=0
        m.add(pdk.nmos1p8(f"swpp{k}", d="vrefp", g=f"b{k}",  s=f"pbot{k}", b="vss",
                          w=sw_refp_w, l=SW_L))
        m.add(pdk.nmos1p8(f"swpn{k}", d="vrefn", g=f"b{k}b", s=f"pbot{k}", b="vss",
                          w=SW_W, l=SW_L))
        # N bottom: complementary (-> vrefn when b{k}=1, -> vrefp when b{k}=0)
        m.add(pdk.nmos1p8(f"swnn{k}", d="vrefn", g=f"b{k}",  s=f"nbot{k}", b="vss",
                          w=SW_W, l=SW_L))
        m.add(pdk.nmos1p8(f"swnp{k}", d="vrefp", g=f"b{k}b", s=f"nbot{k}", b="vss",
                          w=sw_refp_w, l=SW_L))
        if sample_input:
            # sample switch: bottom plate -> input during smp (b/b{k}b held low then)
            m.add(pdk.nmos1p8(f"smpp{k}", d="vinp", g="smp", s=f"pbot{k}", b="vss",
                              w=SW_W, l=SW_L))
            m.add(pdk.nmos1p8(f"smpn{k}", d="vinn", g="smp", s=f"nbot{k}", b="vss",
                              w=SW_W, l=SW_L))
    # 1-unit terminating dummy per side (bottom fixed at the reset reference)
    m.add(el.capacitor("cdp", p="toutp", n="vrefn", value=cunit))
    m.add(el.capacitor("cdn", p="toutn", n="vrefp", value=cunit))
    # top-plate reset switches to vcm (close during sample)
    m.add(pdk.nmos1p8("rstp", d="vcm", g="smp", s="toutp", b="vss", w=SW_W, l=SW_L))
    m.add(pdk.nmos1p8("rstn", d="vcm", g="smp", s="toutn", b="vss", w=SW_W, l=SW_L))
    # equalization switch: short toutp<->toutn during reset so the DIFFERENTIAL residue
    # starts at exactly 0 each conversion regardless of the previous conversion's state
    # (the vcm resets alone leave a history-dependent differential on big inter-
    # conversion swings; equalizing the two nodes is far more robust).
    m.add(pdk.nmos1p8("rsteq", d="toutp", g="smp", s="toutn", b="vss", w=SW_W, l=SW_L))

    ports = {
        "vss":   Port(dir=PortDir.IN,  role=PortRole.GROUND, domain=AVDD),
        "vrefp": Port(dir=PortDir.IN,  role=PortRole.ANALOG, domain=AVDD),
        "vrefn": Port(dir=PortDir.IN,  role=PortRole.ANALOG, domain=AVDD),
        "vcm":   Port(dir=PortDir.IN,  role=PortRole.ANALOG, domain=AVDD),
        "smp":   Port(dir=PortDir.IN,  role=PortRole.CLOCK,  domain=AVDD),
        "toutp": Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=AVDD),
        "toutn": Port(dir=PortDir.OUT, role=PortRole.ANALOG, domain=AVDD),
    }
    if sample_input:
        ports["vinp"] = Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD)
        ports["vinn"] = Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD)
    for k in range(n):
        ports[f"b{k}"] = Port(dir=PortDir.IN, role=PortRole.DIGITAL, domain=AVDD)
        ports[f"b{k}b"] = Port(dir=PortDir.IN, role=PortRole.DIGITAL, domain=AVDD)
    m.define_module_ports(ports)
    m.set_behavioral_model(_cap_dac_behavioral(n, cunit, sample_input),
                           execution="vector")
    return m


def _bit(code, k):
    return (code >> k) & 1


def _convert_one(n, code, pdk, out_dir):
    """Run one reset-to-convert transient for a single code; return the settled
    differential top-plate voltage V(toutp)-V(toutn). Each conversion starts from a
    clean DC reset (top plates at vcm), which is why per-code sims are robust where a
    back-to-back schedule leaves residual charge from the previous code."""
    top = Module("PDK_skywater130", out_dir, f"sar_dac_tb_{n}b.cir", pdk_details=pdk)
    tconv = T_RESET
    tend = T_RESET + T_CONV
    # Reset holds until conversion; reuse the clean event-edge convention below.
    smp = _step_pwl([(0.0, VDD), (tconv, 0.0)], edge_length=T_RISE)
    b, bb = {}, {}
    for k in range(n):
        bit = float(_bit(code, k)) * VDD
        b[k] = _step_pwl([(0.0, 0.0), (tconv, bit)], edge_length=T_RISE)
        bb[k] = _step_pwl([(0.0, VDD), (tconv, VDD - bit)], edge_length=T_RISE)

    top.add_source(VoltageSource("vrp", positive="vrefp", negative="gnd", dc=VREFP))
    top.add_source(VoltageSource("vrn", positive="vrefn", negative="gnd", dc=VREFN))
    top.add_source(VoltageSource("vcm", positive="vcm", negative="gnd", dc=VCM))
    top.add_source(VoltageSource("vsmp", positive="smp", negative="gnd", pwl=smp))
    for k in range(n):
        top.add_source(VoltageSource(f"vb{k}", positive=f"b{k}", negative="gnd", pwl=b[k]))
        top.add_source(VoltageSource(f"vb{k}b", positive=f"b{k}b", negative="gnd", pwl=bb[k]))

    nets = {"vss": "gnd", "vrefp": "vrefp", "vrefn": "vrefn", "vcm": "vcm",
            "smp": "smp", "toutp": "toutp", "toutn": "toutn"}
    for k in range(n):
        nets[f"b{k}"] = f"b{k}"
        nets[f"b{k}b"] = f"b{k}b"
    top.add_module(build_cap_dac(n, pdk, out_dir), "dac", nets=nets)
    top.add_sim_type(Transient(step=50e-12, stop=tend, maxstep=100e-12))
    top.simulate(corner="tt")

    return value_at(top.load_signal("toutp") - top.load_signal("toutn"),
                    tend - 1e-9)


def dac_transfer_spice(n=8, codes=None):
    """Convert a set of codes (one clean sim each) and return (codes, vdiff_spice,
    vdiff_ideal) — the settled differential top-plate voltage per code, measured and
    predicted by charge redistribution on nominal caps."""
    if codes is None:
        codes = sorted(set(list(range(0, 2 ** n, max(1, 2 ** n // 16))) + [2 ** n - 1]))
    pdk = _pdk()
    out_dir = example_output_dir()
    vdiff = np.array([_convert_one(n, int(c), pdk, out_dir) for c in codes])
    ctot = (2 ** n) * CUNIT
    ideal = np.array([2.0 * (VREFP - VREFN) * c * CUNIT / ctot for c in codes])
    return np.array(codes), vdiff, ideal


# --- Stage 2: comparator-in-the-loop conversion (real cap-DAC + real StrongARM) --
# One SPICE sim per bit trial: sample the input onto the array, apply the trial code,
# let the DAC residue settle, then clock the StrongARM to decide the bit. Each trial
# starts with a fresh sample; the continuous bench instead holds charge across bits.
# Ideal charge redistribution motivates the comparison; the schematic dummy terms
# (LOOP_DAC_GAIN / LOOP_TRIP_OFFSET) are what the dummy-aware twin adds. The Python loop
# is only the SAR sequencer.
# settle long enough that the DAC residue is resolved before the comparator fires —
# otherwise the offset picks up an under-settling artifact (grows as T_SET shrinks).
T_SMP = 4e-9
T_SET = 10e-9
T_EVAL = 6e-9
# Default 8-bit trip terms come from the schematic: the 1-unit dummy holds no
# input charge, weighting the sampled input by 255/256 against the DAC.
# Schematic dummy-capacitor terms for the default eight-bit array.
LOOP_DAC_GAIN = 256 / 255
LOOP_TRIP_OFFSET = VREFP / 255


def _sar_decision(n, vinp, vinn, trial_bits, pdk, out_dir, *, ideal_comp=False):
    """One bit-trial: sample (vinp,vinn), apply ``trial_bits``, settle, evaluate the
    StrongARM. Return True if vout is HIGH (toutp > toutn). ``ideal_comp=True`` skips
    the comparator and returns sign(toutp-toutn) from the DAC directly — a diagnostic
    to separate cap-DAC/sampling offset from comparator offset."""
    from silk.examples.strongarm_tb import make_comparator_strongarm
    tconv, teval, tend = T_SMP, T_SMP + T_SET, T_SMP + T_SET + T_EVAL
    top = Module("PDK_skywater130", out_dir, f"sar_loop_{n}b.cir", pdk_details=pdk)
    smp = _step_pwl([(0.0, VDD), (tconv, 0.0)], edge_length=T_RISE)
    clkb = _step_pwl([(0.0, 0.0), (teval, VDD)], edge_length=T_RISE)
    b, bb = {}, {}
    for k in range(n):
        bit = float((trial_bits >> k) & 1) * VDD
        # Both convert switches remain off throughout sampling.
        b[k] = _step_pwl([(0.0, 0.0), (tconv, bit)], edge_length=T_RISE)
        bb[k] = _step_pwl([(0.0, 0.0), (tconv, VDD - bit)], edge_length=T_RISE)

    top.add_source(VoltageSource("vdd", positive="vdd", negative="gnd", dc=VDD))
    top.add_source(VoltageSource("vrp", positive="vrefp", negative="gnd", dc=VREFP))
    top.add_source(VoltageSource("vrn", positive="vrefn", negative="gnd", dc=VREFN))
    top.add_source(VoltageSource("vcm", positive="vcm", negative="gnd", dc=VCM))
    top.add_source(VoltageSource("vinp", positive="vinp", negative="gnd", dc=vinp))
    top.add_source(VoltageSource("vinn", positive="vinn", negative="gnd", dc=vinn))
    top.add_source(VoltageSource("vsmp", positive="smp", negative="gnd", pwl=smp))
    top.add_source(VoltageSource("vclkb", positive="clkb", negative="gnd", pwl=clkb))
    for k in range(n):
        top.add_source(VoltageSource(f"vb{k}", positive=f"b{k}", negative="gnd", pwl=b[k]))
        top.add_source(VoltageSource(f"vb{k}b", positive=f"b{k}b", negative="gnd", pwl=bb[k]))

    dnets = {"vss": "gnd", "vrefp": "vrefp", "vrefn": "vrefn", "vcm": "vcm",
             "smp": "smp", "toutp": "toutp", "toutn": "toutn",
             "vinp": "vinp", "vinn": "vinn"}
    for k in range(n):
        dnets[f"b{k}"] = f"b{k}"
        dnets[f"b{k}b"] = f"b{k}b"
    top.add_module(build_cap_dac(n, pdk, out_dir, sample_input=True), "dac", nets=dnets)
    if not ideal_comp:
        top.add_module(make_comparator_strongarm(pdk, out_dir), "cmp", nets={
            "vp": "toutp", "vm": "toutn", "clkb": "clkb", "vdd": "vdd", "vss": "gnd",
            "vout": "vout"})
    top.add_sim_type(Transient(step=50e-12, stop=tend, maxstep=100e-12))
    top.simulate(corner="tt")
    if ideal_comp:
        return value_at(top.load_signal("toutp") - top.load_signal("toutn"),
                        teval - 0.5e-9) > 0.0
    vo = top.load_signal("vout")
    v_end = value_at(vo, tend - 0.5e-9)
    return v_end > 0.5 * VDD


def measure_trip_points(n=8, codes=(13, 64, 128, 192, 243), *, ideal_comp=False,
                        span=60e-3, tol=0.5e-3):
    """Measure, per trial code, the differential input at which the loop's decision
    flips (one ``_sar_decision`` simulation per bisection step). Returns
    ``(codes, v_dac_twin, v_trip)``: the twin's DAC level for each code and the
    measured trip input, both in volts. ``ideal_comp=True`` decides on the residue
    sign, so it measures the sampled cap-DAC alone; the default uses the StrongARM.
    Each trip is bracketed in ``v_dac_twin +- span`` and bisected to ``tol``."""
    w, wtot = sar.unit_cap_array(n, sigma=0.0)
    codes = np.asarray(codes, dtype=int)
    v_dac = sar.dac_diff(codes, w, wtot, vref=VREFP)
    pdk = _pdk()
    out_dir = example_output_dir()

    def dac_above(vin_diff, code):          # True: the DAC level exceeds the input
        return _sar_decision(n, VIN_CM + vin_diff / 2.0, VIN_CM - vin_diff / 2.0,
                             int(code), pdk, out_dir, ideal_comp=ideal_comp)

    trips = []
    for code, vd in zip(codes, v_dac):
        lo, hi = vd - span, vd + span
        if not dac_above(lo, code) or dac_above(hi, code):
            raise RuntimeError(f"code {code}: decision trip not inside "
                               f"{lo*1e3:+.1f}..{hi*1e3:+.1f} mV; widen span")
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if dac_above(mid, code):
                lo = mid
            else:
                hi = mid
        trips.append(0.5 * (lo + hi))
    return codes, v_dac, np.array(trips)


def convert_spice(n, vin_diff, pdk, out_dir, *, keep_if_vout_low=True, ideal_comp=False):
    """Digitize one differential input (V) via a real SPICE-in-the-loop binary search:
    MSB->LSB, each bit set by a real StrongARM decision on the real cap-DAC residue.
    vout HIGH => DAC exceeded the input => clear the trial bit (keep_if_vout_low)."""
    vinp, vinn = VIN_CM + vin_diff / 2.0, VIN_CM - vin_diff / 2.0
    code = 0
    for k in range(n - 1, -1, -1):
        trial = code | (1 << k)
        vout_high = _sar_decision(n, vinp, vinn, trial, pdk, out_dir, ideal_comp=ideal_comp)
        keep = (not vout_high) if keep_if_vout_low else vout_high
        if keep:
            code = trial
    return code


def sar_transfer_spice(n=8, vin_diffs=None, *, with_ideal=False,
                       dac_gain=None, trip_offset=None):
    """Digitize a set of differential inputs in SPICE and return (vin_diffs,
    codes_strongarm, codes_ideal, codes_twin). ``codes_strongarm`` uses the real
    StrongARM in the loop; ``codes_ideal`` decides each bit on the DAC residue sign
    (includes the sampling path but skips the comparator); ``codes_twin`` is the behavioral
    model on the same nominal array, calibrated with the schematic dummy gain and
    trip offset (``dac_gain=1.0, trip_offset=0.0`` gives the ideal twin)."""
    if dac_gain is None:
        dac_gain = 2 ** n / (2 ** n - 1)
    if trip_offset is None:
        trip_offset = VREFP / (2 ** n - 1)
    from silk.examples import sar_adc_example as sar
    if vin_diffs is None:
        vin_diffs = np.linspace(-0.9 * VREFP, 0.9 * VREFP, 7)
    vin_diffs = np.asarray(vin_diffs)
    pdk = _pdk()
    out_dir = example_output_dir()
    codes_sa = np.array([convert_spice(n, float(v), pdk, out_dir) for v in vin_diffs])
    codes_id = (np.array([convert_spice(n, float(v), pdk, out_dir, ideal_comp=True)
                          for v in vin_diffs]) if with_ideal else None)
    w, wtot = sar.unit_cap_array(n, sigma=0.0)       # nominal array, matches SPICE caps
    codes_twin = np.asarray(sar.sar_codes(vin_diffs, w, wtot, vref=VREFP,
                                          comp_sigma=0.0, comp_offset=-trip_offset,
                                          dac_gain=dac_gain))
    return vin_diffs, codes_sa, codes_id, codes_twin


def save_plot(codes, vspice, videal, n, png_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(codes, videal * 1e3, "-", color="#999", lw=1.2, label="charge-redist (ideal)")
    a1.plot(codes, vspice * 1e3, "o", color="#0072B2", ms=4, label="SPICE cap-DAC")
    a1.set_title(f"{n}-bit differential cap-DAC transfer", fontsize=10)
    a1.set_xlabel("input code"); a1.set_ylabel("V(toutp)-V(toutn)  (mV)")
    a1.legend(fontsize=8); a1.grid(True, alpha=0.4)

    err_lsb = (vspice - videal) / (2 * (VREFP - VREFN) / 2 ** n)
    a2.plot(codes, err_lsb, "o-", color="#D55E00", ms=3, lw=0.8)
    a2.axhline(0, color="#999", lw=0.6)
    a2.set_title(f"SPICE - ideal  (peak {np.abs(err_lsb).max():.3f} LSB)", fontsize=10)
    a2.set_xlabel("input code"); a2.set_ylabel("settling error (LSB)")
    a2.grid(True, alpha=0.4)

    fig.suptitle(f"SAR cap-DAC — real transistors vs charge-redistribution model  "
                 f"({2**n} unit caps of {CUNIT*1e15:.0f} fF, NMOS switches)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=125)
    plt.close(fig)
    return png_path


def emit_html(codes, vspice, videal, n, png_path):
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    err_lsb = (vspice - videal) / (2 * (VREFP - VREFN) / 2 ** n)
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SAR cap-DAC — SPICE vs charge-redistribution</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
      margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fafafa}}
 h1{{font-size:1.3rem;margin:.2rem 0}} .sub{{color:#666;margin:.1rem 0 1.2rem}}
 img{{width:100%;border:1px solid #ddd;border-radius:6px;background:#fff}}
 table{{border-collapse:collapse;margin:1rem 0;font-size:14px}}
 td,th{{border:1px solid #ddd;padding:.35rem .7rem}} th{{background:#f0f0f4}}
 code{{background:#eee;padding:.05rem .3rem;border-radius:3px}}
</style></head><body>
<h1>SAR cap-DAC — real transistors vs the twin's charge-redistribution model</h1>
<p class="sub">{n}-bit differential binary-weighted array, {2**n} unit caps of
{CUNIT*1e15:.0f}&nbsp;fF, NMOS bottom-plate + reset switches. Reset the top plates to
vcm, drive the bottom plates to a code, read the settled differential top-plate
voltage.</p>
<img src="data:image/png;base64,{b64}" alt="cap-DAC transfer">
<p>Peak settling error <b>{np.abs(err_lsb).max():.3f} LSB</b> vs the ideal law
<code>V = 2·(vrefp−vrefn)·S(code)/Ctot</code> — the real caps + switches reproduce the
charge redistribution the behavioral twin (<code>sar_adc_example.dac_diff</code>)
assumes, to the measured error above. This DAC-only bench does not exercise input
sampling or the comparator. The loop comparison includes the schematic
dummy-capacitor terms; see docs/examples.md for accuracy results.</p>
</body></html>"""
    out = os.path.join(OUT, f"sar_cap_dac_{n}bit.html")
    with open(out, "w") as f:
        f.write(doc)
    return out


# --- twin trial replay in one transient (top plate floats across all bits) --------
# Unlike convert_spice (a fresh sample per bit), this samples once and replays the
# twin's predetermined trial codes in one continuous transient. The comparator is
# clocked, but its outputs do not select the next trial. Overlaying the two residue
# trajectories exposes differences; the returned code is the twin's prediction,
# not a measured SPICE conversion. It uses the ideal twin's trials.

def _trial_sequence(n, vin_diff):
    """The bottom-plate code applied at each MSB->LSB step + the resolved code."""
    from silk.examples import sar_adc_example as sar
    w, wtot = sar.unit_cap_array(n, sigma=0.0)
    code, trials = 0, []
    for k in range(n - 1, -1, -1):
        trial = code | (1 << k)
        trials.append(trial)
        if vin_diff >= sar.dac_diff(trial, w, wtot, vref=VREFP):
            code = trial
    return trials, code


def _sched_arrays(t, trials, tsmp, step, vin_diff):
    """Sample the conversion schedule (smp, b{k}, inputs) onto time grid ``t`` — the
    array arguments for the cap-DAC equations."""
    n = len(trials)
    t = np.asarray(t, float)
    j = np.clip(np.floor((t - tsmp) / step).astype(int), 0, n - 1)
    sampling = t < tsmp
    inp = {"smp": np.where(sampling, VDD, 0.0), "vcm": np.full(len(t), VCM),
           "vrefp": np.full(len(t), VREFP), "vrefn": np.full(len(t), VREFN),
           "vinp": np.full(len(t), VIN_CM + vin_diff / 2),
           "vinn": np.full(len(t), VIN_CM - vin_diff / 2), "bits": []}
    for k in range(n):
        bits = np.array([(trials[jj] >> k) & 1 for jj in range(n)], float) * VDD
        inp["bits"].append(np.where(sampling, 0.0, bits[j]))
    return inp


def convert_spice_continuous(n, vin_diff, pdk, out_dir, *, step=8e-9):
    """Replay twin trials in a single transient; return (t, resid_spice, trials,
    code, tsmp, step). ``code`` is the twin prediction, not a SPICE decision.
    resid = V(toutp)-V(toutn), the comparator's differential input."""
    from silk.examples.strongarm_tb import make_comparator_strongarm
    trials, code = _trial_sequence(n, vin_diff)
    tsmp = T_SMP
    tend = tsmp + n * step
    top = Module("PDK_skywater130", out_dir, f"sar_conv_{n}b.cir", pdk_details=pdk)
    smp = _step_pwl([(0.0, VDD), (tsmp, 0.0)], edge_length=T_RISE)  # top floats for the whole search
    b, bb = {}, {}
    for k in range(n):
        events = [(tsmp + j * step, ((tc >> k) & 1) * VDD)
                  for j, tc in enumerate(trials)]
        b[k] = _step_pwl([(0.0, 0.0)] + events, edge_length=T_RISE)
        bb[k] = _step_pwl([(0.0, 0.0)] + [(t, VDD - bit) for t, bit in events], edge_length=T_RISE)

    top.add_source(VoltageSource("vdd", positive="vdd", negative="gnd", dc=VDD))
    top.add_source(VoltageSource("vrp", positive="vrefp", negative="gnd", dc=VREFP))
    top.add_source(VoltageSource("vrn", positive="vrefn", negative="gnd", dc=VREFN))
    top.add_source(VoltageSource("vcm", positive="vcm", negative="gnd", dc=VCM))
    top.add_source(VoltageSource("vinp", positive="vinp", negative="gnd", dc=VIN_CM + vin_diff / 2))
    top.add_source(VoltageSource("vinn", positive="vinn", negative="gnd", dc=VIN_CM - vin_diff / 2))
    top.add_source(VoltageSource("vsmp", positive="smp", negative="gnd", pwl=smp))
    top.add_source(VoltageSource("vclkb", positive="clkb", negative="gnd",
                   pulse=Pulse(v1=0, v2=VDD, td=tsmp + 0.6 * step, tr=0.1e-9,
                               tf=0.1e-9, pw=0.35 * step, per=step)))
    for k in range(n):
        top.add_source(VoltageSource(f"vb{k}", positive=f"b{k}", negative="gnd", pwl=b[k]))
        top.add_source(VoltageSource(f"vb{k}b", positive=f"b{k}b", negative="gnd", pwl=bb[k]))
    dnets = {"vss": "gnd", "vrefp": "vrefp", "vrefn": "vrefn", "vcm": "vcm",
             "smp": "smp", "toutp": "toutp", "toutn": "toutn",
             "vinp": "vinp", "vinn": "vinn"}
    for k in range(n):
        dnets[f"b{k}"] = f"b{k}"
        dnets[f"b{k}b"] = f"b{k}b"
    top.add_module(build_cap_dac(n, pdk, out_dir, sample_input=True), "dac", nets=dnets)
    top.add_module(make_comparator_strongarm(pdk, out_dir), "cmp", nets={
        "vp": "toutp", "vm": "toutn", "clkb": "clkb", "vdd": "vdd", "vss": "gnd",
        "vout": "vout"})
    top.add_sim_type(Transient(step=40e-12, stop=tend, maxstep=80e-12))
    top.simulate(corner="tt")
    residue = top.load_signal("toutp") - top.load_signal("toutn")
    t, resid = residue.x, residue.y
    return t, resid, trials, code, tsmp, step


def conversion_overlay(n=8, vin_diff=0.3, out_dir=None):
    """Overlay SPICE and behavioral residues for a replay of the twin's trials.
    The displayed code is the twin prediction. Saves PNG + self-contained HTML."""
    pdk = _pdk()
    out_dir = out_dir or example_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    t, resid, trials, code, tsmp, step = convert_spice_continuous(n, vin_diff, pdk, out_dir)
    # behavioral twin on the same schedule, on the SPICE time grid
    inp = _sched_arrays(t, trials, tsmp, step, vin_diff)
    toutp, toutn = _cap_dac_voltages(n, CUNIT, True, t, **inp)
    resid_beh = toutp - toutn

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.plot(t * 1e9, resid_beh * 1e3, color="#D55E00", lw=2.2, alpha=0.55,
            label="behavioral twin")
    ax.plot(t * 1e9, resid * 1e3, color="#0072B2", lw=1.1, label="SPICE (full transient)")
    for jstep in range(n):
        tx = (tsmp + (jstep + 0.6) * step) * 1e9
        ax.axvline(tx, color="#e5e5ea", lw=0.8, zorder=0)
        ax.text((tsmp + (jstep + 0.5) * step) * 1e9, ax.get_ylim()[1],
                f"b{n-1-jstep}", fontsize=7, ha="center", va="top", color="#888")
    ax.axvspan(0, tsmp * 1e9, color="#f0f0f5", zorder=0)
    ax.text(tsmp * 1e9 / 2, 0, "sample", fontsize=8, ha="center", color="#888")
    ax.set_title(f"{n}-bit SAR twin-trial replay — comparator differential residue "
                 f"(vin={vin_diff*1e3:.0f} mV; twin code {code})  |  SPICE vs behavioral twin",
                 fontsize=10)
    ax.set_xlabel("time (ns)"); ax.set_ylabel("V(toutp)−V(toutn)  (mV)")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(out_dir, f"sar_conversion_{n}b.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)

    with open(png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    html = os.path.join(out_dir, f"sar_conversion_{n}b.html")
    with open(html, "w") as f:
        f.write(f"""<!doctype html><html><head><meta charset="utf-8">
<title>SAR twin-trial replay — SPICE vs behavioral</title>
<style>body{{font:15px/1.5 -apple-system,sans-serif;max-width:1000px;margin:2rem auto;
padding:0 1rem;background:#fafafa}}img{{width:100%;border:1px solid #ddd;border-radius:6px}}</style>
</head><body><h1>{n}-bit SAR twin-trial replay — SPICE vs behavioral twin</h1>
<p>A single continuous transient: sample once, then replay the twin's MSB→LSB trial
sequence while the top plate floats. The StrongARM is clocked but its outputs do not
choose the trials. Both residue trajectories use that schedule; differences expose
analog behavior missing from the twin. vin={vin_diff*1e3:.0f} mV, twin code {code};
this code is the ideal twin's prediction, not a measured SPICE conversion. The
cap-DAC residue twin here has no comparator. The per-bit loop separately checks the
StrongARM decision against the schematic dummy-capacitor terms.</p>
<img src="data:image/png;base64,{b64}"></body></html>""")
    return png, html


def build_sar_fsm(n, pdk, out_dir):
    """A SAR controller from stdcells — no Python in the loop. Given clk, rstb
    (clear), start (1-cycle token load), smp (sample phase), and comp (comparator
    output), it drives the cap-DAC's b{k}/b{k}b directly.

    Per bit k:
      * seq{k} — a one-hot shift-register DFF; the token loads at the MSB (start) and
        shifts toward the LSB one bit per clock, marking whose turn it is.
      * q{k}   — the code-register DFF; on bit k's turn it captures NOT(comp) (keep the
        bit unless the comparator says the DAC overshot), otherwise it holds.
      * trial  — b{k} = (q{k} OR seq{k}) AND conv, b{k}b = NOR(q{k},seq{k}) AND conv,
        where conv = NOT(smp): during sampling both are forced low so the cap-DAC's
        input-sample switches win; during convert b{k} raises the trial bit."""
    from silk.examples.stdcells import (build_dff, build_mux2, build_nor2, build_nand2,
                                        build_inverter)
    m = Module("PDK_skywater130", out_dir, f"sar_fsm_{n}b.cir", pdk_details=pdk,
               module_name=f"sar_fsm_{n}b")
    dff = build_dff(with_tests=False)
    mux = build_mux2(with_tests=False)
    nor2 = build_nor2(with_tests=False)
    nand2 = build_nand2(with_tests=False)
    inv = build_inverter(name="fsm_inv", with_tests=False)

    def add(cell, inst, **nets):
        m.add_module(cell, inst, nets={"VDD": "vdd", "VSS": "vss", **nets})

    add(inv, "inv_notcomp", A="comp", Y="notcomp")          # ~comp
    add(inv, "inv_conv", A="smp", Y="conv")                 # convert enable = ~smp
    for k in range(n):
        seqin = "start" if k == n - 1 else f"seq{k+1}"
        add(dff, f"seqff{k}", D=seqin, CLK="clk", CLRB="rstb",
            Q=f"seq{k}", QB=f"seqb{k}")                      # sequencer shift register
        add(mux, f"qmux{k}", A=f"q{k}", B="notcomp", S=f"seq{k}", Y=f"qd{k}")
        add(dff, f"qff{k}", D=f"qd{k}", CLK="clk", CLRB="rstb",
            Q=f"q{k}", QB=f"qb{k}")                          # code-register bit
        add(nor2, f"nqs{k}", A=f"q{k}", B=f"seq{k}", Y=f"nqs{k}")   # NOT(q OR seq)
        add(inv, f"tinv{k}", A=f"nqs{k}", Y=f"t{k}")                # trial = q OR seq
        add(nand2, f"bn{k}", A=f"t{k}", B="conv", Y=f"bnn{k}")
        add(inv, f"binv{k}", A=f"bnn{k}", Y=f"b{k}")               # b{k} = trial AND conv
        add(nand2, f"bbn{k}", A=f"nqs{k}", B="conv", Y=f"bbnn{k}")
        add(inv, f"bbinv{k}", A=f"bbnn{k}", Y=f"b{k}b")            # b{k}b = ~trial AND conv

    ports = {"vdd": Port(dir=PortDir.IN, role=PortRole.POWER, domain=AVDD),
             "vss": Port(dir=PortDir.IN, role=PortRole.GROUND, domain=AVDD),
             "clk": Port(dir=PortDir.IN, role=PortRole.CLOCK, domain=AVDD),
             "rstb": Port(dir=PortDir.IN, role=PortRole.DIGITAL, domain=AVDD),
             "start": Port(dir=PortDir.IN, role=PortRole.DIGITAL, domain=AVDD),
             "smp": Port(dir=PortDir.IN, role=PortRole.CLOCK, domain=AVDD),
             "comp": Port(dir=PortDir.IN, role=PortRole.DIGITAL, domain=AVDD)}
    for k in range(n):
        ports[f"b{k}"] = Port(dir=PortDir.OUT, role=PortRole.DIGITAL, domain=AVDD)
        ports[f"b{k}b"] = Port(dir=PortDir.OUT, role=PortRole.DIGITAL, domain=AVDD)
        ports[f"q{k}"] = Port(dir=PortDir.OUT, role=PortRole.DIGITAL, domain=AVDD)
    m.define_module_ports(ports)
    return m


def fsm_logic_check(comp_level, n=8):
    """Check the FSM sequencer and capture: hold comp constant and read
    the code register. comp=0 -> keep every bit -> code 2**n-1; comp=1 -> code 0."""
    pdk = _pdk()
    od = example_output_dir()
    Tclk = 4e-9
    ncyc = n + 3
    top = Module("PDK_skywater130", od, f"fsm_tb_{n}b.cir", pdk_details=pdk)
    top.add_source(VoltageSource("vdd", positive="vdd", negative="gnd", dc=VDD))
    top.add_source(VoltageSource("vsmp", positive="smp", negative="gnd", dc=0.0))  # convert
    top.add_source(VoltageSource("vcomp", positive="comp", negative="gnd", dc=comp_level * VDD))
    # clk: square, rising edges at Tclk, 2Tclk, ...
    top.add_source(VoltageSource("vclk", positive="clk", negative="gnd",
                   pulse=Pulse(v1=0, v2=VDD, td=Tclk, tr=0.1e-9, tf=0.1e-9,
                               pw=Tclk / 2, per=Tclk)))
    # rstb low for the first ~half cycle, then high
    rstb = PWL().add(0, 0).add(0.5e-9, 0).add(0.6e-9, VDD)
    top.add_source(VoltageSource("vrstb", positive="rstb", negative="gnd", pwl=rstb))
    # start high across the first active edge (t=Tclk), then low
    start = (PWL().add(0, VDD).add(1.5 * Tclk, VDD).add(1.5 * Tclk + 0.1e-9, 0))
    top.add_source(VoltageSource("vstart", positive="start", negative="gnd", pwl=start))
    nets = {"vdd": "vdd", "vss": "gnd", "clk": "clk", "rstb": "rstb",
            "start": "start", "smp": "smp", "comp": "comp"}
    for k in range(n):
        nets[f"b{k}"] = f"b{k}"; nets[f"b{k}b"] = f"b{k}b"; nets[f"q{k}"] = f"q{k}"
    top.add_module(build_sar_fsm(n, pdk, od), "fsm", nets=nets)
    top.add_sim_type(Transient(step=40e-12, stop=ncyc * Tclk, maxstep=80e-12))
    top.simulate(corner="tt")
    codes = codes_from_bits([top.load_signal(f"q{k}") for k in range(n)],
                            weighting='binary', msb_first=False, threshold=VDD / 2)
    return int(round(value_at(codes, (ncyc - 0.2) * Tclk)))


def convert_sar_selfclocked(n, vin_diff, pdk, out_dir, *, tclk=14e-9, capture=True):
    """One full conversion, self-sequenced in SPICE: cap-DAC + StrongARM +
    build_sar_fsm in a single transient. Sample once (top plate then floats, charge
    held), then N clocked bit-decisions driven by the on-chip FSM using the controller. Returns (code, t, resid, top, tsmp, tclk). Clock phasing per bit cycle: CLK rising edge sets
    the trial + captures the prior decision; the comparator strobe clkb evaluates late
    in the cycle and stays high across the next CLK edge so its decision is captured."""
    from silk.examples.strongarm_tb import make_comparator_strongarm
    tsmp = T_SMP
    tend = tsmp + (n + 1) * tclk
    top = Module("PDK_skywater130", out_dir, f"sar_self_{n}b.cir", pdk_details=pdk)
    top.add_source(VoltageSource("vdd", positive="vdd", negative="gnd", dc=VDD))
    top.add_source(VoltageSource("vrp", positive="vrefp", negative="gnd", dc=VREFP))
    top.add_source(VoltageSource("vrn", positive="vrefn", negative="gnd", dc=VREFN))
    top.add_source(VoltageSource("vcm", positive="vcm", negative="gnd", dc=VCM))
    top.add_source(VoltageSource("vinp", positive="vinp", negative="gnd", dc=VIN_CM + vin_diff / 2))
    top.add_source(VoltageSource("vinn", positive="vinn", negative="gnd", dc=VIN_CM - vin_diff / 2))
    # sample high then low (0.2 ns before the first CLK edge so sampling fully ends)
    smp = PWL().add(0, VDD).add(tsmp - 0.3e-9, VDD).add(tsmp - 0.1e-9, 0)
    top.add_source(VoltageSource("vsmp", positive="smp", negative="gnd", pwl=smp))
    # clear during sample; token load pulse spanning the first CLK edge at tsmp
    rstb = PWL().add(0, 0).add(tsmp - 0.6e-9, 0).add(tsmp - 0.5e-9, VDD)
    top.add_source(VoltageSource("vrstb", positive="rstb", negative="gnd", pwl=rstb))
    start = PWL().add(0, VDD).add(tsmp + 0.5 * tclk, VDD).add(tsmp + 0.5 * tclk + 0.1e-9, 0)
    top.add_source(VoltageSource("vstart", positive="start", negative="gnd", pwl=start))
    top.add_source(VoltageSource("vclk", positive="clk", negative="gnd",
                   pulse=Pulse(v1=0, v2=VDD, td=tsmp, tr=0.1e-9, tf=0.1e-9,
                               pw=tclk / 2, per=tclk)))
    top.add_source(VoltageSource("vclkb", positive="clkb", negative="gnd",
                   pulse=Pulse(v1=0, v2=VDD, td=tsmp + 0.6 * tclk, tr=0.1e-9,
                               tf=0.1e-9, pw=0.5 * tclk, per=tclk)))
    dn = {"vss": "gnd", "vrefp": "vrefp", "vrefn": "vrefn", "vcm": "vcm", "smp": "smp",
          "toutp": "toutp", "toutn": "toutn", "vinp": "vinp", "vinn": "vinn"}
    fn = {"vdd": "vdd", "vss": "gnd", "clk": "clk", "rstb": "rstb", "start": "start",
          "smp": "smp", "comp": "vout"}
    for k in range(n):
        dn[f"b{k}"] = f"b{k}"; dn[f"b{k}b"] = f"b{k}b"
        fn[f"b{k}"] = f"b{k}"; fn[f"b{k}b"] = f"b{k}b"; fn[f"q{k}"] = f"q{k}"
    top.add_module(build_cap_dac(n, pdk, out_dir, sample_input=True), "dac", nets=dn)
    top.add_module(make_comparator_strongarm(pdk, out_dir), "cmp", nets={
        "vp": "toutp", "vm": "toutn", "clkb": "clkb", "vdd": "vdd", "vss": "gnd",
        "vout": "vout"})
    top.add_module(build_sar_fsm(n, pdk, out_dir), "fsm", nets=fn)
    top.add_sim_type(Transient(step=40e-12, stop=tend, maxstep=80e-12))
    top.simulate(corner="tt")
    residue = top.load_signal("toutp") - top.load_signal("toutn")
    t, resid = residue.x, residue.y
    code = None
    if capture:
        codes = codes_from_bits([top.load_signal(f"q{k}") for k in range(n)],
                                weighting='binary', msb_first=False, threshold=VDD / 2)
        code = int(round(value_at(codes, tend - 0.3 * tclk)))
    return code, t, resid, top, tsmp, tclk


def selfclocked_plot(n=8, vin_diff=0.3, out_dir=None):
    """Run one fully self-clocked SAR conversion and plot it: the analog residue
    (top) and the code bits resolving MSB->LSB under the on-chip FSM (bottom). Saves
    PNG + self-contained HTML."""
    from silk.examples import sar_adc_example as sar
    pdk = _pdk()
    out_dir = out_dir or example_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    code, t, resid, top, tsmp, tclk = convert_sar_selfclocked(n, vin_diff, pdk, out_dir)
    w, wtot = sar.unit_cap_array(n, sigma=0.0)
    twin = int(sar.sar_codes(np.array([vin_diff]), w, wtot, vref=VREFP)[0])

    # behavioral cap-DAC residue, driven by the SAME FSM-produced b{k} schedule
    smp_s = top.load_signal("smp")
    inp = {"smp": sample(smp_s, x_new=t).y,
           "vcm": np.full(len(t), VCM), "vrefp": np.full(len(t), VREFP),
           "vrefn": np.full(len(t), VREFN),
           "vinp": np.full(len(t), VIN_CM + vin_diff / 2),
           "vinn": np.full(len(t), VIN_CM - vin_diff / 2), "bits": []}
    for k in range(n):
        bk = top.load_signal(f"b{k}")
        inp["bits"].append(sample(bk, x_new=t).y)
    toutp, toutn = _cap_dac_voltages(n, CUNIT, True, t, **inp)
    resid_beh = toutp - toutn

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 3]})
    a1.axhline(0, color="#bbb", lw=0.8)
    a1.plot(t * 1e9, resid_beh * 1e3, color="#D55E00", lw=2.4, alpha=0.5,
            label="behavioral twin")
    a1.plot(t * 1e9, resid * 1e3, color="#0072B2", lw=1.1, label="SPICE (full transient)")
    a1.axvspan(0, tsmp * 1e9, color="#f0f0f5"); a1.set_ylabel("residue (mV)")
    a1.set_title(f"Self-clocked {n}-bit SAR — on-chip FSM, one transient, no Python  "
                 f"(vin={vin_diff*1e3:.0f} mV → code {code}, twin {twin})", fontsize=10)
    a1.legend(fontsize=8, loc="upper right"); a1.grid(True, alpha=0.3)
    for k in range(n):                                   # the code bits, MSB at top
        q = top.load_signal(f"q{k}")
        y = np.asarray(q.y, float) / VDD
        a2.plot(np.asarray(q.x, float) * 1e9, y * 0.8 + (n - 1 - k),
                color="#0072B2", lw=1.0)
        a2.text(-1, (n - 1 - k) + 0.35, f"q{k}", fontsize=8, ha="right", va="center")
    a2.axvspan(0, tsmp * 1e9, color="#f0f0f5")
    a2.set_yticks([]); a2.set_xlabel("time (ns)"); a2.set_ylabel("code register (MSB bottom)")
    a2.set_title("code register resolving MSB→LSB under the shift-register sequencer",
                 fontsize=9)
    a2.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    png = os.path.join(out_dir, f"sar_selfclocked_{n}b.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    with open(png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    html = os.path.join(out_dir, f"sar_selfclocked_{n}b.html")
    with open(html, "w") as f:
        f.write(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Self-clocked SAR conversion</title><style>body{{font:15px/1.5 sans-serif;
max-width:1000px;margin:2rem auto;padding:0 1rem;background:#fafafa}}
img{{width:100%;border:1px solid #ddd;border-radius:6px}}</style></head><body>
<h1>Self-clocked {n}-bit SAR — one transient, no Python in the loop</h1>
<p>The whole conversion runs in a transistor simulation: sample once (top plate then floats, charge
held), then the stdcell FSM (shift-register sequencer + code register, build_sar_fsm)
walks MSB→LSB, the StrongARM decides each bit, and its decision clocks into the
register that drives the cap-DAC. vin={vin_diff*1e3:.0f} mV → code {code} (twin {twin}).
Top: the analog residue converging. Bottom: the code bits resolving.</p>
<img src="data:image/png;base64,{b64}"></body></html>""")
    return png, html, code, twin


def _step_pwl(events, *, edge_length=1e-12):
    """Hold levels between events, ending each ``edge_length`` transition at its event."""
    p = PWL()
    prev = None
    for t, v in events:
        if prev is not None:
            p.add(t - edge_length, prev)
        p.add(t, v)
        prev = v
    return p


def convert_sar_sine(n, m_conv, pdk, out_dir, *, ampl=None, tclk=14e-9, periods=1.5):
    """Digitize a sine in one continuous transient: the FSM runs m_conv back-to-back
    conversions while a differential sine drives the input; each conversion samples the
    sine at its sample instant. Per conversion = N+2 clock cycles (0=sample, 1..N=decide,
    N+1=hold/read). Returns (t, resid, vindiff, sample_t, codes, Tconv)."""
    from silk.examples.strongarm_tb import make_comparator_strongarm
    if ampl is None:
        ampl = 0.8 * VREFP
    Tconv = (n + 2) * tclk
    fsig = periods / (m_conv * Tconv)
    tend = m_conv * Tconv
    top = Module("PDK_skywater130", out_dir, f"sar_sine_{n}b.cir", pdk_details=pdk)
    top.add_source(VoltageSource("vdd", positive="vdd", negative="gnd", dc=VDD))
    top.add_source(VoltageSource("vrp", positive="vrefp", negative="gnd", dc=VREFP))
    top.add_source(VoltageSource("vrn", positive="vrefn", negative="gnd", dc=VREFN))
    top.add_source(VoltageSource("vcm", positive="vcm", negative="gnd", dc=VCM))
    top.add_source(VoltageSource("vinp", positive="vinp", negative="gnd",
                   sine=Sine(voffset=VIN_CM, vampl=ampl / 2, freq=fsig)))
    top.add_source(VoltageSource("vinn", positive="vinn", negative="gnd",
                   sine=Sine(voffset=VIN_CM, vampl=ampl / 2, freq=fsig, phase=180.0)))
    top.add_source(VoltageSource("vclk", positive="clk", negative="gnd",
                   pulse=Pulse(v1=0, v2=VDD, td=tclk, tr=0.1e-9, tf=0.1e-9,
                               pw=tclk / 2, per=tclk)))
    smp_e, rstb_e, start_e, clkb_e = [], [(0.0, 0.0)], [(0.0, 0.0)], [(0.0, 0.0)]
    sample_t = []
    for c in range(m_conv):
        T0 = c * Tconv
        smp_e += [(T0, VDD), (T0 + tclk - 0.1e-9, 0.0)]         # cycle 0 = sample
        rstb_e += [(T0, 0.0), (T0 + 0.5 * tclk, VDD)]           # clear at conv start
        start_e += [(T0 + 0.5 * tclk, VDD), (T0 + 1.5 * tclk, 0.0)]  # token load
        # comparator strobe during decision cycles 1..N (excluding sample/hold —
        # a clkb pulse overlapping the sample-end kicks back onto the floating top plate)
        for j in range(1, n + 1):
            clkb_e += [(T0 + (j + 0.55) * tclk, VDD), (T0 + (j + 1.02) * tclk, 0.0)]
        sample_t.append(T0 + tclk - 0.1e-9)                              # End of the nominal sample-control falling edge.
    top.add_source(VoltageSource("vclkb", positive="clkb", negative="gnd", pwl=_step_pwl(clkb_e)))
    top.add_source(VoltageSource("vsmp", positive="smp", negative="gnd", pwl=_step_pwl(smp_e)))
    top.add_source(VoltageSource("vrstb", positive="rstb", negative="gnd", pwl=_step_pwl(rstb_e)))
    top.add_source(VoltageSource("vstart", positive="start", negative="gnd", pwl=_step_pwl(start_e)))
    dn = {"vss": "gnd", "vrefp": "vrefp", "vrefn": "vrefn", "vcm": "vcm", "smp": "smp",
          "toutp": "toutp", "toutn": "toutn", "vinp": "vinp", "vinn": "vinn"}
    fn = {"vdd": "vdd", "vss": "gnd", "clk": "clk", "rstb": "rstb", "start": "start",
          "smp": "smp", "comp": "vout"}
    for k in range(n):
        dn[f"b{k}"] = f"b{k}"; dn[f"b{k}b"] = f"b{k}b"
        fn[f"b{k}"] = f"b{k}"; fn[f"b{k}b"] = f"b{k}b"; fn[f"q{k}"] = f"q{k}"
    top.add_module(build_cap_dac(n, pdk, out_dir, sample_input=True), "dac", nets=dn)
    top.add_module(make_comparator_strongarm(pdk, out_dir), "cmp", nets={
        "vp": "toutp", "vm": "toutn", "clkb": "clkb", "vdd": "vdd", "vss": "gnd",
        "vout": "vout"})
    top.add_module(build_sar_fsm(n, pdk, out_dir), "fsm", nets=fn)
    top.add_sim_type(Transient(step=60e-12, stop=tend, maxstep=120e-12))
    top.simulate(corner="tt")
    residue = top.load_signal("toutp") - top.load_signal("toutn")
    t, resid = residue.x, residue.y
    vindiff = sample(top.load_signal("vinp") - top.load_signal("vinn"), x_new=t).y
    code_trace = codes_from_bits([top.load_signal(f"q{k}") for k in range(n)],
                                 weighting='binary', msb_first=False, threshold=VDD / 2)
    codes = []
    for c in range(m_conv):
        tread = c * Tconv + (n + 1.5) * tclk  # in the hold cycle
        codes.append(int(round(value_at(code_trace, tread))))
    return t, resid, vindiff, np.array(sample_t), np.array(codes), Tconv


def sar_sine_plot(n=8, m_conv=16, periods=3, out_dir=None):
    """Digitize a sine in one transient (self-clocked SAR, no Python) and plot the
    input sine vs the reconstructed digital output; report SNDR/ENOB from the codes."""
    pdk = _pdk()
    out_dir = out_dir or example_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    t, resid, vindiff, st, codes, Tconv = convert_sar_sine(n, m_conv, pdk, out_dir, periods=periods)
    lsb = 2 * VREFP / 2 ** n
    recon = (codes + 0.5) * lsb - VREFP
    sig = Signal('codes', 'voltage', 'time', 'LSB',
                 np.arange(m_conv) * Tconv, codes.astype(float))
    sndr = metric_sndr(sig, tclk=Tconv, n_samples=m_conv)
    enob = metric_enob(sig, tclk=Tconv, n_samples=m_conv)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(t * 1e9, vindiff * 1e3, color="#009E73", lw=1.4, label="input sine (analog)")
    a1.step(st * 1e9, recon * 1e3, where="mid", color="#0072B2", lw=1.0, alpha=0.6)
    a1.plot(st * 1e9, recon * 1e3, "o", color="#0072B2", ms=6,
            label="SAR digital output (code→V)")
    a1.set_ylabel("differential (mV)")
    a1.set_title(f"Self-clocked {n}-bit SAR digitizing a sine — one transient, on-chip FSM  "
                 f"({m_conv} conversions, SNDR {sndr:.1f} dB / ENOB {enob:.2f} b)", fontsize=10)
    a1.legend(fontsize=8, loc="upper right"); a1.grid(True, alpha=0.3)
    a2.axhline(0, color="#bbb", lw=0.8)
    a2.plot(t * 1e9, resid * 1e3, color="#0072B2", lw=0.8)
    for c in range(m_conv):
        a2.axvspan(c * Tconv * 1e9, (c * Tconv + tclk_default() ) * 1e9, color="#f0f0f5")
    a2.set_ylabel("residue (mV)"); a2.set_xlabel("time (ns)"); a2.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(out_dir, f"sar_sine_{n}b.png")
    fig.savefig(png, dpi=125)
    plt.close(fig)
    with open(png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    html = os.path.join(out_dir, f"sar_sine_{n}b.html")
    with open(html, "w") as f:
        f.write(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Self-clocked SAR digitizing a sine</title><style>body{{font:15px/1.5 sans-serif;
max-width:1000px;margin:2rem auto;padding:0 1rem;background:#fafafa}}
img{{width:100%;border:1px solid #ddd;border-radius:6px}}</style></head><body>
<h1>Self-clocked {n}-bit SAR digitizing a sine — one transient, no Python</h1>
<p>A differential sine drives the input while the on-chip FSM runs {m_conv} back-to-back
conversions in a single transient, each sampling the sine at its sample instant (the
equalization switch shorts the top plates at reset so conversions carry no history).
Top: input sine (green) vs the SAR's digital output (blue). Bottom: the analog residue.
SNDR {sndr:.1f} dB, ENOB {enob:.2f} bits from {m_conv} samples.</p>
<img src="data:image/png;base64,{b64}"></body></html>""")
    return t, vindiff, st, recon, codes, sndr, enob, png, html


def tclk_default():
    return 14e-9


def selfclocked_transfer(n=8, npts=21, out_dir=None):
    """Complete characterization of the self-clocked SAR: run the full on-chip
    FSM conversion over a sweep of inputs and plot recovered code vs the twin. Every
    point is one continuous transient (cap-DAC + StrongARM + stdcell FSM, no Python)."""
    from silk.examples import sar_adc_example as sar
    pdk = _pdk()
    out_dir = out_dir or example_output_dir()
    os.makedirs(out_dir, exist_ok=True)
    vin = np.linspace(-0.9 * VREFP, 0.9 * VREFP, npts)
    codes = np.array([convert_sar_selfclocked(n, float(v), pdk, out_dir)[0] for v in vin])
    w, wtot = sar.unit_cap_array(n, sigma=0.0)
    twin = np.asarray(sar.sar_codes(vin, w, wtot, vref=VREFP))
    resid = codes - twin

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(vin * 1e3, twin, "-", color="#999", lw=1.4, label="behavioral twin")
    a1.plot(vin * 1e3, codes, "o", color="#0072B2", ms=4,
            label="self-clocked SPICE (on-chip FSM)")
    a1.set_title(f"{n}-bit SELF-CLOCKED SAR transfer — full FSM, no Python", fontsize=10)
    a1.set_xlabel("differential input (mV)"); a1.set_ylabel("recovered code")
    a1.legend(fontsize=8); a1.grid(True, alpha=0.4)
    a2.plot(vin * 1e3, resid, "o-", color="#D55E00", ms=4, lw=0.8)
    a2.axhline(0, color="#999", lw=0.6); a2.set_ylim(-8, 8)
    a2.set_title(f"self-clocked − twin  (max |Δ| = {int(np.abs(resid).max())} LSB)", fontsize=10)
    a2.set_xlabel("differential input (mV)"); a2.set_ylabel("code difference")
    a2.grid(True, alpha=0.4)
    fig.suptitle("Self-sequencing SAR ADC — every point a full one-transient conversion "
                 "(cap-DAC + StrongARM + stdcell FSM in the loop)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(out_dir, f"sar_selfclocked_transfer_{n}b.png")
    fig.savefig(png, dpi=125)
    plt.close(fig)
    with open(png, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    html = os.path.join(out_dir, f"sar_selfclocked_transfer_{n}b.html")
    with open(html, "w") as f:
        f.write(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Self-clocked SAR transfer</title><style>body{{font:15px/1.5 sans-serif;
max-width:1000px;margin:2rem auto;padding:0 1rem;background:#fafafa}}
img{{width:100%;border:1px solid #ddd;border-radius:6px}}</style></head><body>
<h1>Self-sequencing {n}-bit SAR — full transfer, no Python in the loop</h1>
<p>{npts} inputs, each a complete conversion in one transient: the stdcell FSM
sequences MSB→LSB, the StrongARM decides, the code register drives the cap-DAC, charge
held on the floating top plate. Recovered code tracks the behavioral twin within
{int(np.abs(resid).max())} LSB.</p>
<img src="data:image/png;base64,{b64}"></body></html>""")
    return vin, codes, twin, png, html


def build_sar_top(n=8, *, pdk=None, out_dir=None):
    """Structural SAR ADC: sampling cap-DAC <-> StrongARM comparator <-> the real
    stdcell FSM (build_sar_fsm), wired as the feedback loop. The analog residue
    toutp/toutn goes to the comparator; its decision (comp) is captured by the FSM,
    which sequences MSB->LSB and drives the cap-DAC bottom plates b{k}. This is the
    self-sequencing SAR that convert_sar_selfclocked runs."""
    from silk.examples.strongarm_tb import make_comparator_strongarm
    pdk = pdk or _pdk()
    out_dir = out_dir or example_output_dir()
    m = Module("PDK_skywater130", out_dir, f"sar_adc_{n}b.cir", pdk_details=pdk,
               module_name=f"sar_adc_{n}b")
    dnets = {"vss": "vss", "vrefp": "vrefp", "vrefn": "vrefn", "vcm": "vcm",
             "smp": "smp", "toutp": "toutp", "toutn": "toutn",
             "vinp": "vinp", "vinn": "vinn"}
    fnets = {"vdd": "vdd", "vss": "vss", "clk": "clk", "rstb": "rstb",
             "start": "start", "smp": "smp", "comp": "comp"}
    for k in range(n):
        dnets[f"b{k}"] = f"b{k}"
        dnets[f"b{k}b"] = f"b{k}b"
        fnets[f"b{k}"] = f"b{k}"
        fnets[f"b{k}b"] = f"b{k}b"
        fnets[f"q{k}"] = f"q{k}"
    m.add_module(build_cap_dac(n, pdk, out_dir, sample_input=True), "capdac", nets=dnets)
    m.add_module(make_comparator_strongarm(pdk, out_dir), "comparator", nets={
        "vp": "toutp", "vm": "toutn", "clkb": "clkb", "vdd": "vdd", "vss": "vss",
        "vout": "comp"})
    m.add_module(build_sar_fsm(n, pdk, out_dir), "sar_fsm", nets=fnets)
    ports = {
        "vinp": Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD),
        "vinn": Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD),
        "vrefp": Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD),
        "vrefn": Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD),
        "vcm": Port(dir=PortDir.IN, role=PortRole.ANALOG, domain=AVDD),
        "smp": Port(dir=PortDir.IN, role=PortRole.CLOCK, domain=AVDD),
        "clk": Port(dir=PortDir.IN, role=PortRole.CLOCK, domain=AVDD),
        "clkb": Port(dir=PortDir.IN, role=PortRole.CLOCK, domain=AVDD),
        "rstb": Port(dir=PortDir.IN, role=PortRole.DIGITAL, domain=AVDD),
        "start": Port(dir=PortDir.IN, role=PortRole.DIGITAL, domain=AVDD),
        "vdd": Port(dir=PortDir.IN, role=PortRole.POWER, domain=AVDD),
        "vss": Port(dir=PortDir.IN, role=PortRole.GROUND, domain=AVDD)}
    for k in range(n):
        ports[f"q{k}"] = Port(dir=PortDir.OUT, role=PortRole.DIGITAL, domain=AVDD)
    m.define_module_ports(ports)
    return m


def sar_composition_diagram(n=8, out_dir=None):
    """Render the SAR block diagram (PNG + SVG) via silk.circuit.composition."""
    from silk.circuit.composition import module_graph, save_graph_png
    out_dir = out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    g = module_graph(build_sar_top(n), depth=1)
    png = save_graph_png(g, os.path.join(out_dir, f"sar_adc_{n}b_blocks.png"))
    svg = save_graph_png(g, os.path.join(out_dir, f"sar_adc_{n}b_blocks.svg"))
    return png, svg


def save_sar_transfer_plot(vin_diffs, codes_sa, codes_id, codes_twin, n, png_path, *,
                           codes_twin_ideal=None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(vin_diffs * 1e3, codes_twin, "-", color="#999", lw=1.4,
            label="calibrated twin (schematic dummy terms)")
    if codes_id is not None:
        a1.plot(vin_diffs * 1e3, codes_id, "s", color="#009E73", ms=5, mfc="none",
                label="SPICE cap-DAC (ideal comp)")
    a1.plot(vin_diffs * 1e3, codes_sa, "o", color="#0072B2", ms=5,
            label="SPICE + real StrongARM")
    a1.set_title(f"{n}-bit SAR transfer — real cap-DAC + real StrongARM", fontsize=10)
    a1.set_xlabel("differential input (mV)"); a1.set_ylabel("recovered code")
    a1.legend(fontsize=8); a1.grid(True, alpha=0.4)

    a2.plot(vin_diffs * 1e3, codes_sa - codes_twin, "o-", color="#0072B2", ms=4, lw=0.8,
            label=f"StrongARM − calibrated twin (max |Δ| = {int(np.abs(codes_sa-codes_twin).max())})")
    if codes_id is not None:
        a2.plot(vin_diffs * 1e3, codes_id - codes_twin, "s-", color="#009E73", ms=4,
                lw=0.8, mfc="none", label="cap-DAC − twin (ideal comp)")
    if codes_twin_ideal is not None:
        a2.plot(vin_diffs * 1e3, codes_sa - codes_twin_ideal, "^:", color="#999", ms=4,
                lw=0.8, label="StrongARM − ideal twin "
                f"(max |Δ| = {int(np.abs(codes_sa-codes_twin_ideal).max())})")
    a2.axhline(0, color="#999", lw=0.6)
    a2.set_ylim(-8, 8)
    a2.set_title("code error vs calibrated twin", fontsize=10)
    a2.set_xlabel("differential input (mV)"); a2.set_ylabel("code difference")
    a2.legend(fontsize=8); a2.grid(True, alpha=0.4)

    fig.suptitle("SAR ADC — SPICE-in-the-loop conversion vs behavioral twin  "
                 f"(twin dac_gain {2**n/(2**n-1)}, trip offset {VREFP/(2**n-1)*1e3:.2f} mV)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=125)
    plt.close(fig)
    return png_path


def emit_sar_html(vin_diffs, codes_sa, codes_id, codes_twin, n, png_path, *,
                  codes_twin_ideal=None):
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    resid = codes_sa - codes_twin
    if codes_id is None:
        codes_id = [None] * len(codes_sa)
    if codes_twin_ideal is None:
        codes_twin_ideal = [None] * len(codes_sa)
    rows = "\n".join(
        f"<tr><td>{v*1e3:+.0f}</td><td>{cs}</td>"
        f"<td>{'' if ci is None else ci}</td><td>{'' if cti is None else cti}</td>"
        f"<td>{ct}</td><td>{'' if cti is None else f'{cs-cti:+d}'}</td>"
        f"<td>{cs-ct:+d}</td></tr>"
        for v, cs, ci, cti, ct in zip(vin_diffs, codes_sa, codes_id, codes_twin_ideal,
                                      codes_twin))
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SAR ADC — SPICE-in-the-loop vs twin</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;
      margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fafafa}}
 h1{{font-size:1.3rem;margin:.2rem 0}} .sub{{color:#666;margin:.1rem 0 1.2rem}}
 img{{width:100%;border:1px solid #ddd;border-radius:6px;background:#fff}}
 table{{border-collapse:collapse;margin:1rem 0;font-size:14px}}
 td,th{{border:1px solid #ddd;padding:.3rem .7rem;text-align:right}} th{{background:#f0f0f4}}
 code{{background:#eee;padding:.05rem .3rem;border-radius:3px}}
</style></head><body>
<h1>SAR ADC — SPICE-in-the-loop conversion vs the behavioral twin</h1>
<p class="sub">{n}-bit differential SAR. Every bit is decided by the real
<code>make_comparator_strongarm</code> StrongARM latch evaluating the real
<code>build_cap_dac</code> residue; the Python loop is only the SAR sequencer.
Recovered codes vs <code>sar_adc_example.sar_codes</code> on the same nominal array,
ideal and calibrated with the schematic dummy-capacitor terms.</p>
<img src="data:image/png;base64,{b64}" alt="SAR transfer SPICE vs twin">
<p>Maximum code difference from the calibrated twin in this run:
<b>{int(np.abs(resid).max())} LSB</b>. The calibrated twin compares against
<code>(2**n/(2**n−1))·v_dac + VREFP/(2**n−1)</code> = {2**n/(2**n-1)}·v_dac +
{VREFP/(2**n-1)*1e3:.2f}&nbsp;mV, the schematic dummy-capacitor terms. High-reference switches use 28 µm width to compensate for reduced gate drive
and body effect at the high reference.</p>
<p>The comparator operates near mid-supply: the ideal convert-phase residue common
mode is <code>VCM + (Cs/Ctot)·((vrefp+vrefn)/2 − VIN_CM)</code>, so sampling the input
at Vref/2 places it near VDD/2. That operating point alone does not establish code
agreement with the twin.</p>
<table><tr><th>input (mV)</th><th>StrongARM</th><th>ideal comp</th><th>ideal twin</th><th>calibrated twin</th><th>Δ(SA−ideal)</th><th>Δ(SA−calibrated)</th></tr>
{rows}</table>
</body></html>"""
    out = os.path.join(OUT, f"sar_loop_{n}bit.html")
    with open(out, "w") as f:
        f.write(doc)
    return out


def run_loop(n=8, vin_diffs=None):
    """Run the comparator-in-the-loop transfer comparison."""
    vd, csa, cid, ct = sar_transfer_spice(n, vin_diffs)
    w, wtot = sar.unit_cap_array(n, sigma=0.0)
    cti = np.asarray(sar.sar_codes(vd, w, wtot, vref=VREFP))      # ideal twin
    print("=" * 72)
    print(f"SAR ADC ({n}-bit) — SPICE-in-the-loop conversion vs behavioral twin")
    print("=" * 72)
    print("  every bit = a real StrongARM decision on the real cap-DAC residue")
    print(f"  calibrated twin: dac_gain {2**n/(2**n-1)}, trip offset "
          f"{VREFP/(2**n-1)*1e3:.2f} mV (schematic dummy-capacitor terms)")
    print(f"  {'vin_diff(mV)':>13}{'StrongARM':>11}{'idealComp':>11}{'idealTwin':>11}"
          f"{'calTwin':>9}{'sa−ideal':>10}{'sa−cal':>9}")
    for i, v in enumerate(vd):
        ci = f"{'':>11}" if cid is None else f"{cid[i]:>11d}"
        print(f"  {v*1e3:>13.0f}{csa[i]:>11d}{ci}{cti[i]:>11d}{ct[i]:>9d}"
              f"{csa[i]-cti[i]:>10d}{csa[i]-ct[i]:>9d}")
    print(f"  maximum StrongARM − calibrated twin code difference = "
          f"{int(np.abs(csa-ct).max())} (nominal seven-input comparison)")
    print(f"  maximum StrongARM − ideal twin code difference = {int(np.abs(csa-cti).max())}")
    os.makedirs(OUT, exist_ok=True)
    png = save_sar_transfer_plot(vd, csa, cid, ct, n, os.path.join(OUT, f"sar_loop_{n}bit.png"),
                                 codes_twin_ideal=cti)
    html = emit_sar_html(vd, csa, cid, ct, n, png, codes_twin_ideal=cti)
    print(f"\n  saved {png}\n  saved {html}")
    return vd, csa, cid, ct


def run(n=8):
    codes, vspice, videal = dac_transfer_spice(n)
    err_lsb = (vspice - videal) / (2 * (VREFP - VREFN) / 2 ** n)
    print("=" * 72)
    print(f"SAR cap-DAC ({n}-bit differential) — SPICE vs charge-redistribution")
    print("=" * 72)
    print(f"  {2**n} unit caps of {CUNIT*1e15:.0f} fF, NMOS switches, "
          f"vrefp/vrefn/vcm = {VREFP}/{VREFN}/{VCM} V")
    print(f"  codes tested : {list(codes)}")
    print(f"  peak settling error = {np.abs(err_lsb).max():.3f} LSB   "
          f"(rms {np.sqrt(np.mean(err_lsb**2)):.3f} LSB)")
    os.makedirs(OUT, exist_ok=True)
    png = save_plot(codes, vspice, videal, n, os.path.join(OUT, f"sar_cap_dac_{n}bit.png"))
    html = emit_html(codes, vspice, videal, n, png)
    print(f"\n  saved {png}")
    print(f"  saved {html}")
    return codes, vspice, videal


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "trip":
        codes, v_dac, v_trip = measure_trip_points()   # ~50 sims, one per bisection step
        gain, offset = np.polyfit(v_dac, v_trip, 1)
        for c, vd, vt in zip(codes, v_dac, v_trip):
            print(f"code {c:>3}: twin DAC {vd*1e3:+8.2f} mV, StrongARM trip {vt*1e3:+8.2f} mV")
        print(f"trip line: gain {gain:.4f}, offset {offset*1e3:+.2f} mV  "
              f"(LOOP_DAC_GAIN {LOOP_DAC_GAIN}, LOOP_TRIP_OFFSET {LOOP_TRIP_OFFSET*1e3:.2f} mV)")
    elif arg == "loop":
        run_loop()                 # Stage 2: comparator-in-the-loop (slow, ~112 sims)
    elif arg == "diagram":
        png, svg = sar_composition_diagram()
        print(f"saved {png}\nsaved {svg}")
    elif arg == "conv":
        vd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
        png, html = conversion_overlay(8, vd)   # one full conversion, SPICE vs twin
        print(f"saved {png}\nsaved {html}")
    elif arg == "self":
        vd = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
        png, html, code, twin = selfclocked_plot(8, vd)   # on-chip FSM, no Python
        print(f"code {code} (twin {twin})\nsaved {png}\nsaved {html}")
    elif arg == "transfer":
        npts = int(sys.argv[2]) if len(sys.argv) > 2 else 21
        vin, codes, twin, png, html = selfclocked_transfer(8, npts)  # full self-clocked sweep
        print(f"max |Δ| = {int(np.abs(codes-twin).max())} LSB\nsaved {png}\nsaved {html}")
    elif arg == "sine":
        m = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        *_, sndr, enob, png, html = sar_sine_plot(8, m)   # digitize a sine, one transient
        print(f"SNDR {sndr:.1f} dB / ENOB {enob:.2f} b\nsaved {png}\nsaved {html}")
    else:
        run()                      # Stage 1: cap-DAC transfer vs charge-redistribution
