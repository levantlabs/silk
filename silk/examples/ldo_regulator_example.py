"""NMOS source-follower regulator with a PMOS-input error amplifier.

The amplifier compares VCTRL with VREG and drives the NMOS pass gate.
The block can supply the ring in supply_regulated_vco_example.py.

Seven embedded benches measure DC tracking, small-step damping and natural
frequency, large-step slew rate, load-step regulation and droop, load-current
range, supply rejection, and device stress. A separate broken-loop build
measures loop gain and phase margin. The behavioral twin tracks VCTRL with
second-order dynamics, slew limiting, supply leakage, and target clipping.
It does not model load regulation.

The circuit uses 1.8 V devices on a 3.3 V supply. The reliability bench checks
terminal voltages against the PDK catalog and is expected to fail its
zero-violation spec. This circuit is not suitable for fabrication as drawn.

Run from the repository root with MPLBACKEND=Agg:
    .venv/bin/python -m silk.examples.ldo_regulator_example
It prints spec results, regulation measurements, junction violations, loop
gain, and phase margin, and saves a tracking/load-step/PSRR plot.
"""
import os

import numpy as np

from silk.circuit.circuit_elements import Port
from silk.circuit.modules import Module
from silk import load_pdk
from silk import sources
from silk import analysis
from silk.elements import elements
from silk.metrics import (argmax_x, average, dc_gain_dB, error_norm, maximum,
                          overshoot, peak_to_peak, phase_margin, psrr_db,
                          slew_rate, unity_gain_freq)
from silk.signals.base import Signal
from silk.simulation import AC, DC, Transient

from silk.examples.two_stage_miller_ota_example import build_two_stage_ota
from silk.examples import example_output_dir

pdk = load_pdk('sky130')

# --- design knobs --------------------------------------------------------------
# The 3.3 V supply overstresses some 1.8 V devices. The reliability bench
# checks each terminal pair against its PDK catalog rating.
_VDD = 3.3
# Pass device: big, so it carries the load at a small V_gs (small dropout) and a
# high gm (low output impedance -> the output pole stays out of the way).
_W_PASS, _L_PASS, _M_PASS = 20.0, 0.15, 8
_C_REG = 10e-12        # output capacitance for load-current transients
_I_AMP = 10e-6         # error-amp bias reference
_AMP_CC = 2e-12        # error-amp Miller cap -> sets the LOOP bandwidth
_I_LOAD = 500e-6       # nominal test load (~ what the ring draws)
_I_LOAD_STEP = 1e-3    # load-step height for the regulation test
# Leave time for both load states to settle before measuring their averages.
_LOAD_TD, _LOAD_PW = 5e-6, 5e-6
_LOAD_STOP = 15e-6

# The pass gate needs VGS headroom above VREG, including body effect.
# Limit the control range to the downstream ring's nominal supply range.
_VCTRL_LO, _VCTRL_HI = 0.5, 1.8
_VCTRL_NOM = 1.2                   # bias point for the load / PSRR / loop tests

# Metric / spec names
TRACK_METRIC = "track_err_mv"      # |VREG - VCTRL| at nominal
LOADREG_METRIC = "load_reg_mv"     # STATIC: settled VREG shift across the step
DROOP_METRIC = "droop_mv"          # TRANSIENT: peak excursion at the edge
PSRR_METRIC = "psrr_db"
ILOAD_METRIC = "iload_droop_mv"   # droop across the load-current sweep
STRESS_METRIC = "junction_violations"  # STR-001 violation count, whole hierarchy
PM_METRIC = "pm_deg"
GAIN_METRIC = "loop_gain_db"
FC_METRIC = "loop_fc_hz"           # broken-loop unity-gain crossover
BEH_GAIN_METRIC = "beh_gain_vv"    # -> behavioral gain
BEH_VREGMAX_METRIC = "beh_vreg_max"  # -> behavioral target limit
SLEW_METRIC = "slew_v_per_us"      # -> behavioral slew limit
ZETA_METRIC = "zeta"               # -> behavioral damping (from overshoot)
FN_METRIC = "fn_hz"                # -> behavioral bw_hz (CLOSED-loop wn)

TRACK_MAX_MV = 20.0
LOADREG_MAX_MV = 10.0     # static: loop gain should make this tiny
DROOP_MAX_MV = 100.0      # transient: set by output impedance + C_REG
PSRR_MIN_DB = 20.0
ILOAD_DROOP_MAX_MV = 50.0
# Check regulation from light load through ten times the nominal test load.
_ILOAD_LO, _ILOAD_HI = 10e-6, 5e-3
PM_MIN_DEG = 45.0
SLEW_MIN_V_PER_US = 0.3
ZETA_MIN = 0.2
FN_MIN_HZ = 0.5e6
_VSTEP_TD, _VSTEP_STOP = 5e-6, 20e-6
_VSTEP_SMALL = (1.20, 1.30)   # linear regime  -> zeta, f_n
_VSTEP_LARGE = (0.80, 1.50)   # slew regime    -> slew rate

# Behavioral parameters stored for the default design. Re-run ldo_calibrate
# after changing geometry, capacitance, bias, or supply voltage.
_BEH_GAIN_NOM = 0.999971  # V/V   (beh_gain_vv: VREG/VCTRL slope of the dc sweep)
_BEH_BW_NOM = 7.711e6     # Hz    (fn_hz: CLOSED-loop natural freq, small step)
_BEH_PSRR_NOM = 95.46     # dB    (psrr_db, low frequency)
_BEH_VREGMAX_NOM = 1.7999  # V    (beh_vreg_max: top of the characterized range)
_BEH_SLEW_NOM = 3.994     # V/us  (slew_v_per_us: 10-90% of the large step)
_BEH_ZETA_NOM = 0.5223    # -     (zeta: from the small step's overshoot)


def build_ldo(*, w_pass=_W_PASS, l_pass=_L_PASS, m_pass=_M_PASS, c_reg=_C_REG,
              amp_cc=_AMP_CC, with_tests=True, break_loop=False, initial_output=None) -> Module:
    """Build the LDO: PMOS-input two-stage error amp + NMOS follower pass device.

    Ports: ``VDD``, ``VSS``, ``VCTRL`` (reference in), ``VREG`` (regulated out),
    ``IBIAS_AMP`` (error-amp reference current).

    ``break_loop=True`` splits the amp output from the pass gate into ``VG``
    (gate side) and ``VGA`` (amp side), both exposed as ports, so the loop gain
    can be measured by injection. Nothing else changes — see ``attach_loop_test``.

    For behavioral transients, ``initial_output`` sets VREG at t=0 with zero
    slope. None starts at the first clipped target. It has no effect on DC.

    Parameters
    ----------
    w_pass, l_pass, m_pass : float/int   Pass NMOS geometry. Wide (high gm) so the
                                         dropout and the output impedance are low.
    c_reg : float                        Output capacitance (F).
    amp_cc : float                       Error-amp Miller cap (F) — the loop-
                                         bandwidth knob (see ``_AMP_CC``).
    """
    ports = {
        "VDD": Port.supply(), "VSS": Port.ground(),
        "VCTRL": Port.analog_input(), "VREG": Port.analog_output(),
        "IBIAS_AMP": Port.analog_input(),
    }
    if break_loop:
        ports["VG"] = Port.analog_input()     # pass-gate side (inject here)
        ports["VGA"] = Port.analog_output()   # amp-output side (measure here)

    mod = Module("PDK_skywater130", example_output_dir("ldo_regulator_example"),
                 "ldo.cir", pdk_details=pdk,
                 module_name="ldo")
    mod.define_module_ports(ports)

    gate = "VG" if break_loop else "vg"
    amp_out = "VGA" if break_loop else "vg"

    # Error amp. PMOS input: senses VCTRL near VSS. Its NMOS-CS output stage
    # swings high, which the NMOS pass gate needs (VG must exceed VREG + Vth).
    # Polarity: VINP is non-inverting, so VREG on VINM gives negative feedback --
    # VREG up -> amp output down -> pass gate down -> less current -> VREG down.
    amp = build_two_stage_ota(input_type="pmos", with_tests=False, cc=amp_cc)
    mod.add_module(amp, "amp", nets={
        "VDD": "VDD", "VSS": "VSS", "VINP": "VCTRL", "VINM": "VREG",
        "VOUT": amp_out, "IBIAS": "IBIAS_AMP"})

    with mod:
        # Pass device: drain on VDD, gate from the amp, source is the regulated
        # rail. A follower, so it buffers rather than amplifies.
        pdk.nmos1p8("MPASS", d="VDD", g=gate, s="VREG", b="VSS",
                    w=w_pass, l=l_pass, m=m_pass)
        elements.capacitor("CREG", p="VREG", n="VSS", value=c_reg)

    # Use the stored parameters; ldo_calibrate measures replacements.
    mod.set_behavioral_model(_LDOModel(), execution="sample", params={
        "gain": _BEH_GAIN_NOM, "bw_hz": _BEH_BW_NOM, "psrr_db": _BEH_PSRR_NOM,
        "vreg_max": _BEH_VREGMAX_NOM, "slew_v_per_us": _BEH_SLEW_NOM,
        "zeta": _BEH_ZETA_NOM, "vdd_nom": _VDD, "initial_output": initial_output,
    })

    if with_tests:
        attach_dc_test(mod)
        # Use the small step for damping/frequency and the large step for slew.
        attach_vstep_test(mod, name="vstep_small", v_lo=_VSTEP_SMALL[0],
                          v_hi=_VSTEP_SMALL[1], small_signal=True)
        attach_vstep_test(mod, name="vstep_large", v_lo=_VSTEP_LARGE[0],
                          v_hi=_VSTEP_LARGE[1], small_signal=False)
        attach_load_test(mod)
        attach_iload_test(mod)
        attach_psrr_test(mod)
        attach_reliability_test(mod)
    return mod


# --- behavioral model ----------------------------------------------------------

class _LDOModel:
    """Second-order VCTRL tracker with six calibrated parameters.

    VREG'' + 2*zeta*wn*VREG' + wn^2*VREG = wn^2*target, with
    wn = 2*pi*bw_hz and a slew_v_per_us limit on VREG'. The target is
    min(gain*VCTRL + (VDD-vdd_nom)/10**(psrr_db/20), vreg_max).

    The six calibrated parameters are gain, bw_hz, zeta, slew_v_per_us,
    psrr_db, and vreg_max. The last clips the target at the characterized
    range; it does not cap transient overshoot or initial_output.
    Load regulation is not modeled.

    Transient state starts at initial_output with zero slope, or at the
    first clipped target when initial_output is None. DC returns the target.
    """

    def __call__(self, ports, t, params):

        gain = float(params.gain)
        wn = 2.0 * np.pi * float(params.bw_hz)
        zeta = float(params.zeta)
        psrr_lin = 10.0 ** (float(params.psrr_db) / 20.0)
        target = min(gain * float(ports.VCTRL.v)
                     + (float(ports.VDD.v) - float(params.vdd_nom)) / psrr_lin,
                     float(params.vreg_max))
        if analysis() == "dc":
            ports.VREG.v = target
            return
        if t == 0:
            initial = params.initial_output
            self.y = target if initial is None else float(initial)
            self.dy = 0.0
        else:
            dt = float(t - self.time)
            slew = float(params.slew_v_per_us) * 1e6
            ddy = wn * wn * (target - self.y) - 2.0 * zeta * wn * self.dy
            self.dy += ddy * dt
            self.dy = min(max(self.dy, -slew), slew)
            self.y += self.dy * dt
        self.time = float(t)
        ports.VREG.v = self.y


# --- shared rig ----------------------------------------------------------------

def _add_rig(tb, *, i_amp=_I_AMP):
    """Add the error-amplifier bias current to a testbench.

    The PMOS bias diode needs a current sink to ground.
    """
    tb.add_source(sources.CurrentSource("iamp", positive="ibias_amp", negative="0",
                                        dc=i_amp))


# --- testbenches ---------------------------------------------------------------

def attach_dc_test(mod, *, vctrl_lo=_VCTRL_LO, vctrl_hi=_VCTRL_HI, i_load=_I_LOAD):
    """Sweep VCTRL with DC(source='vvctrl', ...) and measure VREG.

    The result's x-axis is VCTRL. Tracking gain and the maximum output in
    this sweep supply the behavioral gain and target-limit calibration.
    """
    tb = mod.create_test("dc")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "VREG": "vreg", "IBIAS_AMP": "ibias_amp"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                        dc=vctrl_lo))
    tb.add_source(sources.CurrentSource("iload", positive="vreg", negative="0",
                                        dc=i_load))
    _add_rig(tb)
    tb.add_sim_type(DC(source="vvctrl", start=vctrl_lo, stop=vctrl_hi, step=0.02))
    tb.add_metric(TRACK_METRIC, _track_err_mv)
    # These metrics calibrate gain and the target limit through their specs.
    tb.add_metric(BEH_GAIN_METRIC, _beh_gain_vv)
    tb.add_metric(BEH_VREGMAX_METRIC, maximum, sig=tb.probe("vreg"))
    tb.add_spec(f"tracking error < {TRACK_MAX_MV:.0f} mV", TRACK_METRIC, "<",
                TRACK_MAX_MV)
    # CalibrationBinding attaches to design specs, so both metrics have specs.
    tb.add_spec("tracking gain > 0.99 V/V", BEH_GAIN_METRIC, ">", 0.99)
    tb.add_spec(f"reaches VREG >= {_VCTRL_HI:.1f} V (the ring's full supply)",
                BEH_VREGMAX_METRIC, ">=", _VCTRL_HI - 0.01)
    return tb


def _beh_gain_vv(ckt):
    """VREG/VCTRL slope over the sweep -- the behavioral model's `gain`."""
    sig = ckt.load_signal("vreg")
    vctrl = np.asarray(sig.x, dtype=float)
    vreg = np.asarray(sig.y, dtype=float)
    return float(np.polyfit(vctrl, vreg, 1)[0])


def _track_err_mv(ckt):
    """Measure worst absolute VREG-VCTRL tracking error in mV.

    The DC sweep supplies VCTRL on the signal x-axis.
    """
    sig = ckt.load_signal("vreg")
    vctrl = np.asarray(sig.x, dtype=float)
    goal = Signal("vctrl", "voltage", sig.xlabel, sig.ylabel, vctrl, vctrl)
    return error_norm(sig, goal, kind="max") * 1e3


def attach_load_test(mod, *, vctrl=_VCTRL_NOM, i_load=_I_LOAD, i_step=_I_LOAD_STEP):
    """Measure settled load regulation and transient droop separately.

    The load pulse leaves a settling interval in each state. Regulation is
    the difference between their final-window averages; droop is the largest
    excursion from the light-load baseline near the rising load edge.
    """
    tb = mod.create_test("load")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "VREG": "vreg", "IBIAS_AMP": "ibias_amp"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                        dc=vctrl))
    tb.add_source(sources.CurrentSource(
        "iload", positive="vreg", negative="0",
        pulse=sources.Pulse(v1=i_load, v2=i_load + i_step, td=_LOAD_TD,
                            tr=1e-9, tf=1e-9, pw=_LOAD_PW, per=2 * _LOAD_PW)))
    _add_rig(tb)
    tb.add_sim_type(Transient(step=200e-12, stop=_LOAD_STOP))
    vreg = tb.probe("vreg")
    lo = tb.add_metric("vreg_light_V", average, sig=vreg,
                       start=_LOAD_TD - 0.2 * _LOAD_PW, end=_LOAD_TD)
    hi = tb.add_metric("vreg_heavy_V", average, sig=vreg,
                       start=_LOAD_TD + 0.8 * _LOAD_PW, end=_LOAD_TD + _LOAD_PW)
    tb.add_metric(LOADREG_METRIC, abs(hi - lo) * 1e3)
    tb.add_metric(DROOP_METRIC, _droop_mv)
    tb.add_spec(f"static load regulation < {LOADREG_MAX_MV:.0f} mV", LOADREG_METRIC,
                "<", LOADREG_MAX_MV)
    tb.add_spec(f"transient droop < {DROOP_MAX_MV:.0f} mV", DROOP_METRIC, "<",
                DROOP_MAX_MV)
    return tb


def _droop_mv(ckt):
    """Largest load-edge excursion from the settled light-load level, in mV."""
    sig = ckt.load_signal("vreg")
    lo = average(sig, start=_LOAD_TD - 0.2 * _LOAD_PW, end=_LOAD_TD)
    # maximum includes its end; nextafter preserves the exclusive upper bound.
    return maximum((sig - lo).mag(), start=_LOAD_TD,
                   end=np.nextafter(_LOAD_TD + 0.2 * _LOAD_PW, -np.inf),
                   interpolate=False) * 1e3


def attach_psrr_test(mod, *, vctrl=_VCTRL_NOM, i_load=_I_LOAD):
    """Apply 1 V AC on VDD and measure supply rejection at VREG.

    For this drive, PSRR = -20*log10(abs(VREG_ac)).
    """
    tb = mod.create_test("psrr")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "VREG": "vreg", "IBIAS_AMP": "ibias_amp"})
    # AC rides on the supply; VCTRL is a quiet DC reference.
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0",
                                        dc=_VDD, ac=1.0))
    tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                        dc=vctrl))
    tb.add_source(sources.CurrentSource("iload", positive="vreg", negative="0",
                                        dc=i_load))
    _add_rig(tb)
    tb.add_sim_type(AC(variation="dec", points=20, fstart=1e3, fstop=1e9))
    leak_db = tb.add_metric("supply_leak_db", dc_gain_dB, sig=tb.probe("vreg"))
    tb.add_metric(PSRR_METRIC, -leak_db)
    tb.add_spec(f"PSRR > {PSRR_MIN_DB:.0f} dB (low freq)", PSRR_METRIC, ">",
                PSRR_MIN_DB)
    return tb


def attach_loop_test(mod, *, vctrl=_VCTRL_NOM, i_load=_I_LOAD):
    """Measure broken-loop gain, unity-gain crossover, and phase margin.

    Requires build_ldo(break_loop=True). An inductor closes the loop at DC;
    a capacitor couples 1 V AC to the pass gate. Loop gain is -VGA/VG.
    """
    tb = mod.create_test("loop")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "VREG": "vreg", "IBIAS_AMP": "ibias_amp",
                              "VG": "vg", "VGA": "vga"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                        dc=vctrl))
    tb.add_source(sources.CurrentSource("iload", positive="vreg", negative="0",
                                        dc=i_load))
    _add_rig(tb)
    tb.add(elements.inductor("LBREAK", p="vga", n="vg", value=1e9))
    tb.add_source(sources.VoltageSource("vinj", positive="inj", negative="0",
                                        dc=0, ac=1.0))
    tb.add(elements.capacitor("CINJ", p="inj", n="vg", value=1.0))
    tb.add_sim_type(AC(variation="dec", points=40, fstart=10, fstop=1e10))
    loop_gain = -(tb.probe("vga") / tb.probe("vg"))      # T = -VGA/VG
    tb.add_metric(GAIN_METRIC, dc_gain_dB, sig=loop_gain)
    tb.add_metric(PM_METRIC, phase_margin, sig=loop_gain)
    tb.add_metric(FC_METRIC, unity_gain_freq, sig=loop_gain)   # broken-loop unity-gain crossover
    tb.add_spec(f"phase margin > {PM_MIN_DEG:.0f} deg", PM_METRIC, ">", PM_MIN_DEG)
    return tb


# --- load-current sweep --------------------------------------------------------

def attach_iload_test(mod, *, vctrl=_VCTRL_NOM, i_lo=_ILOAD_LO, i_hi=_ILOAD_HI):
    """Sweep load current and measure the peak-to-peak VREG variation.

    DC(source='iiload', ...) returns load current on the x-axis. The sweep
    checks regulation over the requested range; it does not locate dropout.
    """
    tb = mod.create_test("iload")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "VREG": "vreg", "IBIAS_AMP": "ibias_amp"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                        dc=vctrl))
    tb.add_source(sources.CurrentSource("iload", positive="vreg", negative="0",
                                        dc=i_lo))
    _add_rig(tb)
    tb.add_sim_type(DC(source="iiload", start=i_lo, stop=i_hi, step=(i_hi - i_lo) / 40))
    tb.add_metric(ILOAD_METRIC, _iload_droop_mv)
    tb.add_spec(f"droop over {i_lo*1e6:.0f} uA-{i_hi*1e3:.0f} mA load < "
                f"{ILOAD_DROOP_MAX_MV:.0f} mV", ILOAD_METRIC, "<", ILOAD_DROOP_MAX_MV)
    return tb


def _iload_droop_mv(ckt):
    """VREG droop across the whole load-current sweep, in mV."""
    sig = ckt.load_signal("vreg")
    return peak_to_peak(sig) * 1e3


# --- reference step (large-signal) ---------------------------------------------

def attach_vstep_test(mod, *, name, v_lo, v_hi, small_signal):
    """Register a VCTRL-step bench for small- or large-signal measurements.

    The small step measures overshoot-derived damping and peak-time natural
    frequency. The large step measures 10–90% slew rate. Use the same small
    step for damping and frequency calibration.
    """
    tb = mod.create_test(name)
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "VREG": "vreg", "IBIAS_AMP": "ibias_amp"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    tb.add_source(sources.VoltageSource(
        "vctrl", positive="vctrl", negative="0",
        pulse=sources.Pulse(v1=v_lo, v2=v_hi, td=_VSTEP_TD, tr=1e-9, tf=1e-9,
                            pw=_VSTEP_STOP, per=2 * _VSTEP_STOP)))
    tb.add_source(sources.CurrentSource("iload", positive="vreg", negative="0",
                                        dc=_I_LOAD))
    _add_rig(tb)
    tb.add_sim_type(Transient(step=5e-9, stop=_VSTEP_STOP))
    # CalibrationBinding uses these design specs to update the twin.
    if small_signal:
        vreg_sig = tb.probe("vreg")
        tb.add_metric(ZETA_METRIC, _zeta_from_overshoot, sig=vreg_sig,
                      v_lo=v_lo, v_hi=v_hi)
        tb.add_metric(FN_METRIC, _fn_from_step, sig=vreg_sig,
                      v_lo=v_lo, v_hi=v_hi)
        tb.add_spec(f"damping zeta > {ZETA_MIN:.2f}", ZETA_METRIC, ">", ZETA_MIN)
        tb.add_spec(f"closed-loop f_n > {FN_MIN_HZ/1e6:.1f} MHz", FN_METRIC, ">",
                    FN_MIN_HZ)
    else:
        tb.add_metric(SLEW_METRIC, _slew_v_per_us)
        tb.add_spec(f"slew rate > {SLEW_MIN_V_PER_US:.2f} V/us", SLEW_METRIC, ">",
                    SLEW_MIN_V_PER_US)
    return tb


def _zeta_from_overshoot(sig, v_lo=0.8, v_hi=1.5):
    """Estimate damping from fractional overshoot.

    zeta = -ln(OS)/sqrt(pi**2 + ln(OS)**2). Overshoot at or below 1e-4
    returns 1 as a no-overshoot convention, not a damping measurement.
    """
    os_frac = overshoot(sig, v_lo, v_hi)
    if os_frac <= 1e-4:
        return 1.0                       # no-overshoot convention
    os_frac = min(os_frac, 0.95)         # guard the log
    ln = np.log(os_frac)
    return float(-ln / np.sqrt(np.pi ** 2 + ln ** 2))


def _fn_from_step(sig, v_lo=0.8, v_hi=1.5):
    """Estimate closed-loop natural frequency from an underdamped peak.

    f_n = 1/(2*t_peak*sqrt(1-zeta**2)), with t_peak measured from the
    step edge. Return nan when there is no measurable overshoot, no peak
    after the edge, or the trace ends at its maximum. This is distinct
    from broken-loop unity-gain crossover.
    """
    zeta = _zeta_from_overshoot(sig, v_lo, v_hi)
    if not np.isfinite(zeta) or zeta >= 1.0:
        return float("nan")
    peak_time = argmax_x(sig, start=_VSTEP_TD)
    t_peak = peak_time - _VSTEP_TD
    if t_peak <= 0 or peak_time >= sig.x[-1]:
        return float("nan")
    wd = np.sqrt(1.0 - zeta ** 2)
    wn = np.pi / (t_peak * wd)
    return float(wn / (2 * np.pi))


def _slew_v_per_us(ckt):
    """Measure sustained step slew in V/us with 10–90% crossings.

    The crossing interval avoids treating an edge-feedthrough spike as
    the sustained slew rate.
    """
    return slew_rate(ckt.load_signal("vreg"), start=_VSTEP_TD) / 1e6


# --- reliability (hierarchical device-stress check) -----------------------------

def reliability_violations(ckt):
    """Return STR-001 terminal-voltage violations from the simulated bench.

    The bench contains the DUT hierarchy and its waveforms. Initial-condition
    samples count along with all other samples.
    """
    from silk import check_junction_overrange
    return check_junction_overrange(ckt)


def count_junction_violations(ckt):
    """Metric: how many (device, terminal pair) ratings the run exceeds."""
    return float(len(reliability_violations(ckt)))


def attach_reliability_test(mod, *, vctrl=_VCTRL_LO, i_load=_I_LOAD):
    """Check hierarchical terminal voltages against PDK catalog ratings.

    The bench uses the low control voltage, which increases pass-device
    VDS. It checks the pass transistor and the amplifier hierarchy, with
    a spec of zero violations. The 3.3 V supply exceeds the ratings of
    some 1.8 V devices, so this spec is expected to fail.
    """
    tb = mod.create_test("reliability")
    tb.add_module_under_test({"VDD": "VDD", "VSS": "0", "VCTRL": "vctrl",
                              "VREG": "vreg", "IBIAS_AMP": "ibias_amp"})
    tb.add_source(sources.VoltageSource("vdd", positive="VDD", negative="0", dc=_VDD))
    tb.add_source(sources.VoltageSource("vctrl", positive="vctrl", negative="0",
                                        dc=vctrl))
    tb.add_source(sources.CurrentSource("iload", positive="vreg", negative="0",
                                        dc=i_load))
    _add_rig(tb)
    tb.add_sim_type(Transient(step=1e-9, stop=2e-6))
    tb.add_metric(STRESS_METRIC, count_junction_violations)
    tb.add_spec("no device over its rated voltage", STRESS_METRIC, "<=", 0)
    return tb


def run_characterization(mod, corner="tt"):
    results = mod.run_all_tests(corner=corner)
    return results, mod.reconcile_specs()


def save_plot(results, mod_name, path):
    """DC tracking, the load step, and PSRR vs frequency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dc = results[f"{mod_name}.dc"].sim_results.load_signal("vreg")
    ld = results[f"{mod_name}.load"].sim_results.load_signal("vreg")
    ps = results[f"{mod_name}.psrr"].sim_results.load_signal("vreg")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4))

    vctrl = np.asarray(dc.x, float); vreg = np.asarray(dc.y, float)
    a1.plot([vctrl[0], vctrl[-1]], [vctrl[0], vctrl[-1]], "--", color="#5a5a6e",
            lw=1, label="ideal VREG = VCTRL")
    a1.plot(vctrl, vreg, color="#0072B2", lw=2, label="VREG")
    a1.set_xlabel("VCTRL (V)"); a1.set_ylabel("VREG (V)")
    a1.set_title("tracks the reference\n(this is the VCO's tuning input)")
    a1.legend(frameon=False, fontsize=8)

    t = np.asarray(ld.x, float) * 1e6
    a2.plot(t, np.asarray(ld.y, float) * 1e3, color="#D55E00", lw=1.4)
    a2.set_xlabel("time (us)"); a2.set_ylabel("VREG (mV)")
    a2.set_title("load step 0.5 -> 1.5 mA\nsettles in each state")

    supply = results[f"{mod_name}.psrr"].sim_results.load_signal("vdd")
    rejection = psrr_db(supply, ps)
    a3.semilogx(rejection.x, rejection.y, color="#009E73", lw=2)
    a3.set_xlabel("frequency (Hz)"); a3.set_ylabel("PSRR (dB)")
    a3.set_title("supply rejection")

    for a in (a1, a2, a3):
        a.grid(True, color="#d5d5dd", lw=0.6, alpha=0.6); a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


if __name__ == "__main__":
    OUT = example_output_dir()

    mod = build_ldo()
    print("=" * 66)
    print(f"{mod.module_name}: NMOS source-follower pass device + PMOS-input "
          f"two-stage error amp")
    print("=" * 66)
    results, recon = run_characterization(mod)
    print(recon.summary())
    r = lambda t, m: results[f"{mod.module_name}.{t}"].metrics[m]
    print(f"  tracking error   = {r('dc', TRACK_METRIC):.1f} mV "
          f"(VCTRL {_VCTRL_LO}..{_VCTRL_HI} V)")
    print(f"  load regulation  = {r('load', LOADREG_METRIC):.1f} mV "
          f"({_I_LOAD*1e6:.0f} -> {(_I_LOAD+_I_LOAD_STEP)*1e6:.0f} uA step)")
    print(f"  transient droop  = {r('load', DROOP_METRIC):.1f} mV (at the step edge)")
    print(f"  PSRR (low freq)  = {r('psrr', PSRR_METRIC):.1f} dB")
    print(f"  load sweep droop = {r('iload', ILOAD_METRIC):.3f} mV "
          f"({_ILOAD_LO*1e6:.0f} uA -> {_ILOAD_HI*1e3:.0f} mA)")
    print(f"  saved {save_plot(results, mod.module_name, os.path.join(OUT, 'ldo_regulator.png'))}")

    # Print the device and terminal pair for each reported stress violation.
    rel_ckt = results[f"{mod.module_name}.reliability"].sim_results
    violations = reliability_violations(rel_ckt)
    print(f"\n  --- STR-001 junction check: {len(violations)} violation(s) ---")
    for v in violations:
        print(f"  {v.message}")
    print("  (expected stress violations: 1.8 V devices on a 3.3 V rail)")

    # Loop gain / phase margin needs the broken-loop build.
    lp = build_ldo(with_tests=False, break_loop=True)
    attach_loop_test(lp)
    lres, lrecon = run_characterization(lp)
    print()
    print(lrecon.summary())
    lm = lres[f"{lp.module_name}.loop"].metrics
    print(f"  DC loop gain     = {lm[GAIN_METRIC]:.1f} dB")
    print(f"  phase margin     = {lm[PM_METRIC]:.1f} deg")
