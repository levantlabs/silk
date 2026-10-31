"""Comparator offset, dynamic hysteresis, kickback and track-and-hold errors.

Rising and falling ramps measure a mismatched NMOS pair's trip points.
Two ramp rates separate static offset from output-pole lag. Reference-step
benches compare kickback and delay with resistive and ideal input drive.
An NMOS switch and hold capacitor provide sampling-error and droop measurements.
The kT/C value is a thermal-noise estimate; the transient does not simulate noise.

The example prints measured values, model comparisons and kickback versus
measurement-window width. Requires ngspice and sky130 models. Run with::

    python -m silk.examples.metrics.comparator_and_sampler
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.circuit.ports import Port
from silk.elements import elements
from silk.metrics import (hysteresis, kT_over_C, peak_to_peak, derivative, peak,
                          propagation_delay, rise_time, value_at, crossing)
from silk.signals.base import Signal
from silk.simulation import Transient

VDD = 1.8
VREF = 0.9
W_UNIT, L_PAIR = 1.0, 0.5  # µm
M_M1, M_M2 = 10, 11
MISMATCH = M_M2 / M_M1
RD_OHM = 80e3
CL_F = 100e-15
I_TAIL = 10e-6

# Use the balanced branch current to set a fixed output threshold.
VOUT_BAL = VDD - I_TAIL * RD_OHM / 2

TAU_OUT = RD_OHM * CL_F

# The tap disturbance is RS_NEAR / RS_TOTAL times the gate disturbance.
RS_NEAR, RS_FAR = 10e3, 90e3
RS_TOTAL = RS_NEAR + RS_FAR
KICK_DIVIDER = RS_NEAR / RS_TOTAL

V_RAMP_LO, V_RAMP_HI = VREF - 0.25, VREF + 0.25
T_RAMP = 2e-6
RAMP_SLOPE = (V_RAMP_HI - V_RAMP_LO) / T_RAMP
FAST = 4
RAMP_TSTEP = 500e-12

T_REF_STEP = 20e-9
V_OVERDRIVE = 0.1
REF_EDGE = 100e-12
T_DECISION_STOP = 80e-9
DECISION_TSTEP = 10e-12

W_SW, L_SW = 2.0, 0.15  # µm
C_HOLD = 100e-15
V_TRACK_LO, V_TRACK_HI = 0.89, 0.90
V_DUMP = 0.10
T_SIG_STEP = 1e-9
T_SAMPLE = 2e-9
T_DUMP = 2.2e-9
T_HOLD_START = 3e-9
T_HOLD_END = 1e-6
SH_EDGE = 50e-12
SH_TSTEP, SH_MAXSTEP = 2e-12, 200e-12

TEMP_C = 27.0
TEMP_K = TEMP_C + 273.15

# Separate transient grids need a relative tolerance for the hysteresis comparison.
HYST_ROUTE_TOL = 2e-2


class Row(NamedTuple):
    """A measurement row with an optional model value and a display scale."""
    label: str
    measured: float
    expected: Optional[float]  # None omits the expected value and relative difference.
    scale: float
    unit: str
    note: str


def build_comparator(out_dir: str, pdk) -> Module:
    """Build a two-transistor comparator subcircuit without testbenches."""
    mod = Module('PDK_skywater130', out_dir, 'comparator.cir', pdk_details=pdk,
                 module_name='comparator')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VINP': Port.analog_input(), 'VREF': Port.analog_input(),
        'VOUT': Port.analog_output(), 'VOUTB': Port.analog_output(),
        'VTAIL': Port.analog_output(),
    })
    with mod:
        # M1's inverting drain couples kickback to VINP; VOUT is non-inverting.
        pdk.nmos1p8('M1', d='VOUTB', g='VINP', s='VTAIL', b='VSS',
                    w=W_UNIT, l=L_PAIR, m=M_M1)
        pdk.nmos1p8('M2', d='VOUT', g='VREF', s='VTAIL', b='VSS',
                    w=W_UNIT, l=L_PAIR, m=M_M2)
        elements.resistor('rd1', p='VDD', n='VOUTB', value=RD_OHM)
        elements.resistor('rd2', p='VDD', n='VOUT', value=RD_OHM)
        elements.capacitor('cl1', p='VOUTB', n='VSS', value=CL_F)
        elements.capacitor('cl2', p='VOUT', n='VSS', value=CL_F)
    # The ideal tail source fixes the current used to calculate VOUT_BAL.
    mod.add_source(sources.CurrentSource('it', positive='VTAIL',
                                         negative='VSS', dc=I_TAIL))
    return mod


def build_ramp_farm(out_dir: str, pdk) -> Module:
    """Build four comparator instances for rising and falling ramps at two rates."""
    child = build_comparator(out_dir, pdk)
    top = Module('PDK_skywater130', out_dir, 'ramp_farm.cir', pdk_details=pdk,
                 module_name='ramp_farm')
    ports = {'VDD': Port.supply(), 'VSS': Port.ground(),
             'VREF': Port.analog_input()}
    for tag in ('U1', 'D1', 'U4', 'D4'):
        ports[f'VI_{tag}'] = Port.analog_input()
        ports[f'VO_{tag}'] = Port.analog_output()
    top.define_module_ports(ports)
    for tag in ('U1', 'D1', 'U4', 'D4'):
        # Separate internal nets keep unused outputs isolated between instances.
        top.add_module(child, tag, nets={
            'VDD': 'VDD', 'VSS': 'VSS', 'VREF': 'VREF',
            'VINP': f'VI_{tag}', 'VOUT': f'VO_{tag}',
            'VOUTB': f'b_{tag}', 'VTAIL': f't_{tag}'})
    return top


def build_track_hold(out_dir: str, pdk) -> Module:
    """Build an NMOS sampling switch and hold capacitor."""
    mod = Module('PDK_skywater130', out_dir, 'track_hold.cir', pdk_details=pdk,
                 module_name='track_hold')
    mod.define_module_ports({
        'VSS': Port.ground(), 'VSIG': Port.analog_input(),
        'VCLK': Port.digital_input(), 'VHOLD': Port.analog_output(),
    })
    with mod:
        pdk.nmos1p8('MSW', d='VHOLD', g='VCLK', s='VSIG', b='VSS',
                    w=W_SW, l=L_SW)
        elements.capacitor('ch', p='VHOLD', n='VSS', value=C_HOLD)
    return mod


def _dut_and_supply(tb) -> None:
    """Connect the comparator ports and DC supply to a testbench."""
    tb.add_module_under_test({
        'VDD': 'vdd', 'VSS': '0', 'VINP': 'vin', 'VREF': 'vref',
        'VOUT': 'vout', 'VOUTB': 'voutb', 'VTAIL': 'vtail'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                        dc=VDD))


def _ramp(name: str, node: str, v_from: float, v_to: float, t_ramp: float):
    """A single linear ramp from ``v_from`` to ``v_to`` over ``t_ramp``."""
    return sources.VoltageSource(
        name, positive=node, negative='0',
        pulse=sources.Pulse(v1=v_from, v2=v_to, td=0.0, tr=t_ramp, tf=t_ramp,
                            pw=2 * t_ramp, per=10 * t_ramp))


def attach_comparator_benches(cmp_: Module) -> None:
    """Attach rising and falling ramps and two reference-step benches."""
    for name, (v_from, v_to) in (('ramp_up', (V_RAMP_LO, V_RAMP_HI)),
                                 ('ramp_dn', (V_RAMP_HI, V_RAMP_LO))):
        tb = cmp_.create_test(name)
        _dut_and_supply(tb)
        tb.add_source(sources.VoltageSource('vr', positive='vref',
                                            negative='0', dc=VREF))
        tb.add_source(_ramp('vs', 'vin', v_from, v_to, T_RAMP))
        tb.add_sim_type(Transient(step=RAMP_TSTEP, stop=T_RAMP,
                                  maxstep=RAMP_TSTEP))

    tb = cmp_.create_test('decision')
    _dut_and_supply(tb)
    tb.add_source(sources.VoltageSource('vs', positive='vsrc', negative='0',
                                        dc=VREF))
    tb.add(elements.resistor('rs_near', p='vsrc', n='vmid', value=RS_NEAR))
    tb.add(elements.resistor('rs_far', p='vmid', n='vin', value=RS_FAR))
    tb.add_source(sources.VoltageSource(
        'vr', positive='vref', negative='0',
        pulse=sources.Pulse(v1=VREF - V_OVERDRIVE, v2=VREF + V_OVERDRIVE,
                            td=T_REF_STEP, tr=REF_EDGE, tf=REF_EDGE,
                            pw=10 * T_DECISION_STOP,
                            per=100 * T_DECISION_STOP)))
    tb.add_sim_type(Transient(step=DECISION_TSTEP, stop=T_DECISION_STOP,
                              maxstep=DECISION_TSTEP))

    # An ideal input source provides a delay comparison for the resistive-drive bench.
    tb = cmp_.create_test('decision_ideal')
    _dut_and_supply(tb)
    tb.add_source(sources.VoltageSource('vs', positive='vin', negative='0',
                                        dc=VREF))
    tb.add_source(sources.VoltageSource(
        'vr', positive='vref', negative='0',
        pulse=sources.Pulse(v1=VREF - V_OVERDRIVE, v2=VREF + V_OVERDRIVE,
                            td=T_REF_STEP, tr=REF_EDGE, tf=REF_EDGE,
                            pw=10 * T_DECISION_STOP,
                            per=100 * T_DECISION_STOP)))
    tb.add_sim_type(Transient(step=DECISION_TSTEP, stop=T_DECISION_STOP,
                              maxstep=DECISION_TSTEP))
    # Register testbenches before propagating their temperature.
    cmp_.set_temperature(TEMP_C)


def attach_farm_bench(farm: Module):
    """Attach four ramps and register hysteresis metrics at two rates."""
    tb = farm.create_test('hyst')
    nets = {'VDD': 'vdd', 'VSS': '0', 'VREF': 'vref'}
    for tag in ('U1', 'D1', 'U4', 'D4'):
        nets[f'VI_{tag}'] = f'vi_{tag.lower()}'
        nets[f'VO_{tag}'] = f'vo_{tag.lower()}'
    tb.add_module_under_test(nets)
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                        dc=VDD))
    tb.add_source(sources.VoltageSource('vr', positive='vref', negative='0',
                                        dc=VREF))
    for tag, (v_from, v_to), t_ramp in (
            ('u1', (V_RAMP_LO, V_RAMP_HI), T_RAMP),
            ('d1', (V_RAMP_HI, V_RAMP_LO), T_RAMP),
            ('u4', (V_RAMP_LO, V_RAMP_HI), T_RAMP / FAST),
            ('d4', (V_RAMP_HI, V_RAMP_LO), T_RAMP / FAST)):
        tb.add_source(_ramp(f'vs_{tag}', f'vi_{tag}', v_from, v_to, t_ramp))
    tb.add_sim_type(Transient(step=RAMP_TSTEP, stop=T_RAMP,
                              maxstep=RAMP_TSTEP))
    for name, tags in (('hysteresis_slow_V', ('u1', 'd1')),
                       ('hysteresis_fast_V', ('u4', 'd4'))):
        up, dn = tags
        tb.add_metric(name, type='hysteresis',
                      signals={'sig_in_rising': f'vi_{up}',
                               'sig_out_rising': f'vo_{up}',
                               'sig_in_falling': f'vi_{dn}',
                               'sig_out_falling': f'vo_{dn}'},
                      out_threshold=VOUT_BAL)
    farm.set_temperature(TEMP_C)
    return tb


def attach_track_hold_bench(th: Module) -> None:
    """Attach a transient bench with track, sample and hold intervals."""
    tb = th.create_test('sample')
    tb.add_module_under_test({'VSS': '0', 'VSIG': 'vsig', 'VCLK': 'vclk',
                              'VHOLD': 'vh'})
    tb.add_source(sources.VoltageSource('vsig', positive='vsig', negative='0',
                                        pwl=sources.PWL([
                                            (0.0, V_TRACK_LO),
                                            (T_SIG_STEP, V_TRACK_LO),
                                            (T_SIG_STEP + SH_EDGE, V_TRACK_HI),
                                            (T_DUMP, V_TRACK_HI),
                                            (T_DUMP + SH_EDGE, V_DUMP),
                                            (T_HOLD_END, V_DUMP)])))
    # Start the clock high to give the hold node a DC path at the operating point.
    tb.add_source(sources.VoltageSource(
        'vclk', positive='vclk', negative='0',
        pulse=sources.Pulse(v1=VDD, v2=0.0, td=T_SAMPLE, tr=SH_EDGE,
                            tf=SH_EDGE, pw=10 * T_HOLD_END,
                            per=100 * T_HOLD_END)))
    tb.add_sim_type(Transient(step=SH_TSTEP, stop=T_HOLD_END,
                              maxstep=SH_MAXSTEP))
    # Tight tolerances resolve the residual of the tracking step.
    tb.netlist.set_simulator_options(preset='tight')
    th.set_temperature(TEMP_C)


def settling_tau(sig: Signal, target: float, t0: float, t1: float) -> float:
    """The time constant of an exponential approach to ``target``."""
    x = np.asarray(sig.x, dtype=float)
    y = np.asarray(sig.y, dtype=float)
    mask = (x >= t0) & (x <= t1)
    err = np.abs(target - y[mask])
    keep = err > 0
    slope = np.polyfit(x[mask][keep], np.log(err[keep]), 1)[0]
    return float(-1.0 / slope)


def trip_offset(sig_in: Signal, sig_out: Signal, *, ideal_trip: float,
                out_threshold: float) -> float:
    """The input voltage at the output's first rising crossing, less the ideal trip point."""
    t_dec = crossing(sig_out, out_threshold, edge='rising', n=1)
    return value_at(sig_in, t_dec) - ideal_trip


def window_pk_pk(sig: Signal, t_centre: float, width: float) -> float:
    """Peak-to-peak of ``sig`` over ``t_centre ± width/2``."""
    return peak_to_peak(sig, start=t_centre - 0.5 * width,
                        end=t_centre + 0.5 * width)


def peak_gain(v_in: Signal, v_out: Signal) -> float:
    """Return the maximum absolute output-versus-input slope over the ramp."""
    return peak(derivative(v_out.versus(v_in)))


def transconductance(gain_vv: float) -> float:
    """``gm`` from the measured gain, with ``ro`` folded in."""
    return 2.0 * gain_vv / RD_OHM


def predicted_offset(gm_over_id: float) -> float:
    """``V_os = ln(m2/m1) / (gm/Id)``, the balance condition of a mismatched pair."""
    return math.log(MISMATCH) / gm_over_id


def predicted_hysteresis(slope_v_per_s: float) -> float:
    """Estimate ramp hysteresis as twice the output time constant times the ramp slope."""
    return 2.0 * TAU_OUT * slope_v_per_s


def predicted_prop_delay(v_start: float, v_final: float) -> float:
    """``Rd·C_L·ln((V_f − V_i)/(V_f − V_th))`` — the output pole crossing the threshold on
    its way from ``v_start`` to ``v_final``.
    """
    return TAU_OUT * math.log((v_final - v_start) / (v_final - VOUT_BAL))


def measure_decision(vref: Signal, vout_b: Signal, vin: Signal,
                     vmid: Signal, vsrc: Signal, vin_dc: float,
                     vref_ideal: Signal, vout_b_ideal: Signal,
                     ) -> tuple[list[Row], float, float]:
    """``propagation_delay`` and the kickback on the reference-step bench."""
    t_dec = crossing(vout_b, VOUT_BAL, edge='rising', n=1)
    # Use twice the output rise time to bracket the decision event.
    t_edge = rise_time(vout_b)
    window = 2.0 * t_edge

    ideal = np.asarray(vout_b_ideal.y, dtype=float)
    tpd_ideal = propagation_delay(vref_ideal, vout_b_ideal,
                                  in_threshold=vin_dc, out_threshold=VOUT_BAL,
                                  in_edge='rising')
    tpd_loaded = propagation_delay(vref, vout_b, in_threshold=vin_dc,
                                   out_threshold=VOUT_BAL, in_edge='rising')

    kick_gate = window_pk_pk(vin, t_dec, window)
    kick_tap = window_pk_pk(vmid, t_dec, window)
    kick_src = window_pk_pk(vsrc, t_dec, window)

    ns, mv = 1e-9, 1e-3
    rows = [
        Row('propagation_delay (ideal)', tpd_ideal,
            predicted_prop_delay(float(ideal[0]), float(ideal[-1])), ns, 'ns',
            'drawn output pole, measured starting and final levels'),
        Row('propagation_delay (100 kΩ)', tpd_loaded, tpd_ideal, ns, 'ns',
            'loaded input compared with the ideal-drive control'),
        Row('kickback  (gate, 100 kΩ)', kick_gate, None, mv, 'mV',
            'input disturbance while the reference steps'),
        Row('kickback  (tap, 10 kΩ)', kick_tap, KICK_DIVIDER * kick_gate, mv,
            'mV',
            'drawn resistance divider times gate disturbance'),
        Row('kickback  (source, 0 Ω)', kick_src, 0.0, mv, 'mV',
            'ideal source holds its node fixed'),
    ]
    return rows, t_dec, window


def kickback_window_sweep(vin: Signal, t_dec: float,
                          t_edge: float) -> list[tuple[float, float]]:
    """Measure peak-to-peak gate disturbance over several window widths."""
    return [(w, window_pk_pk(vin, t_dec, w))
            for w in (0.1 * t_edge, 0.25 * t_edge, 0.5 * t_edge, t_edge,
                      1.5 * t_edge, 2.0 * t_edge, 2.5 * t_edge)]


def measure_trip(v_in_up: Signal, v_out_up: Signal,
                 v_in_dn: Signal, v_out_dn: Signal,
                 offset_slow: float, offset_fast: float,
                 gm_over_id: float) -> list[Row]:
    """Measure trip offset and hysteresis, and extrapolate static offset."""
    hyst = hysteresis(v_in_up, v_out_up, v_in_dn, v_out_dn,
                      out_threshold=VOUT_BAL)
    # Extrapolate two ramp-rate offsets to cancel the term proportional to ramp slope.
    v_os = (FAST * offset_slow - offset_fast) / (FAST - 1)
    raw = trip_offset(v_in_up, v_out_up, ideal_trip=VREF,
                      out_threshold=VOUT_BAL)
    mv = 1e-3
    return [
        Row('trip offset (first rising)', raw, v_os + RAMP_SLOPE * TAU_OUT, mv,
            'mV',
            'static offset plus output-pole ramp lag'),
        Row('hysteresis  (direct)', hyst, predicted_hysteresis(RAMP_SLOPE),
            mv, 'mV',
            '2·Rd·C_L·ramp slope; dynamic lag rather than memory'),
        Row('static offset (extrapolated)', v_os, predicted_offset(gm_over_id),
            mv, 'mV',
            'mismatch prediction uses measured gm/Id'),
    ]


def measure_sampler(vh: Signal) -> tuple[list[Row], dict]:
    """``value_at``, the hold droop and the ``kT_over_C`` floor."""
    # Fit near the sampling instant because switch resistance changes during tracking.
    tau = settling_tau(vh, V_TRACK_HI, T_SAMPLE - 0.5e-9, T_SAMPLE)
    t_step_done = T_SIG_STEP + SH_EDGE
    step_v = V_TRACK_HI - V_TRACK_LO

    v_sampled = value_at(vh, T_SAMPLE)
    v_held = value_at(vh, T_HOLD_START)
    err_at_sample = abs(v_sampled - V_TRACK_HI) / abs(V_TRACK_HI)
    err_after = abs(v_held - V_TRACK_HI) / abs(V_TRACK_HI)
    pedestal = v_sampled - v_held

    # Normalize droop by the initial held voltage, which is away from zero.
    v_hold_end = value_at(vh, T_HOLD_END)
    droop_v = v_held - v_hold_end
    droop = droop_v / v_held
    i_leak = C_HOLD * droop_v / (T_HOLD_END - T_HOLD_START)
    kt_c = kT_over_C(C_HOLD, TEMP_K)

    facts = {'tau': tau, 'ron': tau / C_HOLD, 'pedestal': pedestal,
             'droop_v': droop_v, 'i_leak': i_leak, 'kt_c': kt_c,
             't_cross': (T_HOLD_END - T_HOLD_START) * kt_c / droop_v}
    us, uv = 1e-6, 1e-6
    rows = [
        Row('settling error (at sample)', err_at_sample,
            (step_v / V_TRACK_HI) * math.exp(
                -(T_SAMPLE - t_step_done) / tau), 1.0, '',
            'exponential residual fitted near the sampling instant'),
        Row('settling error (after clk)', err_after, None, 1.0, '',
            'includes the switch-opening pedestal'),
        Row('hold droop', droop, None, 1e-6, 'ppm',
            'held-node change divided by the initial held voltage'),
        Row('kT_over_C', kt_c, None, uv, 'µV',
            'thermal RMS floor; this transient itself is noiseless'),
    ]
    return rows, facts


def _report(title: str, rows: list[Row],
            sink: Optional[list] = None) -> None:
    if sink is not None:
        sink.extend(rows)
    print()
    print(title)
    print(f'  {"metric":32s} {"measured":>13s} {"expected":>13s} {"rel":>9s}'
          '   note')
    for r in rows:
        meas = f'{r.measured / r.scale:9.4f} {r.unit:<3s}'
        if r.expected is None:
            exp, rel = f'{"—":>9s} {r.unit:<3s}', f'{"—":>8s}'
        else:
            exp = f'{r.expected / r.scale:9.4f} {r.unit:<3s}'
            denom = abs(r.expected) if r.expected else 1.0
            rel = f'{abs(r.measured - r.expected) / denom:8.2e}'
        print(f'  {r.label:32s} {meas} {exp} {rel}   {r.note}')


def main() -> int:
    out_dir = example_output_dir('comparator_and_sampler')
    print('Comparator and sampler metrics (sky130)')
    print(f'  decks and rawfiles: {out_dir}')
    pdk = load_pdk('sky130')
    cmp_ = build_comparator(out_dir, pdk)
    attach_comparator_benches(cmp_)
    results = cmp_.run_all_tests()
    up = results['comparator.ramp_up'].sim_results
    dn = results['comparator.ramp_dn'].sim_results
    dec = results['comparator.decision'].sim_results
    ideal = results['comparator.decision_ideal'].sim_results
    v_in_up, v_out_up = up.load_signal('vin'), up.load_signal('vout')
    v_in_dn, v_out_dn = dn.load_signal('vin'), dn.load_signal('vout')

    farm = build_ramp_farm(out_dir, pdk)
    attach_farm_bench(farm)
    fr = farm.run_all_tests()['ramp_farm.hyst'].sim_results
    offset_slow = trip_offset(fr.load_signal('vi_u1'), fr.load_signal('vo_u1'),
                              ideal_trip=VREF, out_threshold=VOUT_BAL)
    offset_fast = trip_offset(fr.load_signal('vi_u4'), fr.load_signal('vo_u4'),
                              ideal_trip=VREF, out_threshold=VOUT_BAL)
    gain = peak_gain(v_in_up, v_out_up)
    gm_over_id = transconductance(gain) / (I_TAIL / 2)
    vin_dc = float(np.asarray(dec.load_signal('vin').y, dtype=float)[0])
    rows, t_dec, _ = measure_decision(
        dec.load_signal('vref'), dec.load_signal('voutb'),
        dec.load_signal('vin'), dec.load_signal('vmid'),
        dec.load_signal('vsrc'), vin_dc,
        ideal.load_signal('vref'), ideal.load_signal('voutb'))
    reported: list[Row] = []
    _report('Comparator decision and kickback', rows, reported)
    t_edge = rise_time(dec.load_signal('voutb'))
    print('  kickback against window width (gate, tap, source):')
    for w, k in kickback_window_sweep(dec.load_signal('vin'), t_dec, t_edge):
        kt = window_pk_pk(dec.load_signal('vmid'), t_dec, w)
        ks = window_pk_pk(dec.load_signal('vsrc'), t_dec, w)
        print(f'    {w*1e9:7.2f} ns   {k*1e3:7.4f} mV   {kt*1e3:6.4f} mV   '
              f'{ks*1e3:6.4f} mV')
    _report('Trip point and ramp-rate lag',
            measure_trip(v_in_up, v_out_up, v_in_dn, v_out_dn,
                         offset_slow, offset_fast, gm_over_id), reported)

    th = build_track_hold(out_dir, pdk)
    attach_track_hold_bench(th)
    sh = th.run_all_tests()['track_hold.sample'].sim_results
    rows, _ = measure_sampler(sh.load_signal('vh'))
    _report('Track-and-hold sampling errors', rows, reported)
    bad = [r.label for r in reported if not math.isfinite(r.measured)]
    assert not bad, f'metric(s) returned a non-finite value: {bad}'
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
