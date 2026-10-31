"""Step-response measurements and waveform transforms on a series RLC.

A delayed voltage step drives a series resistor and inductor with a shunt
capacitor. Print settling time, overshoot, post-transition undershoot and
RMS error against the analytic second-order step response. Differentiate
output voltage to estimate capacitor current and integrate it for charge.
Correlation estimates the edge-to-peak delay and ringing period; the period
search uses a window around the analytic period. Print these measurements
alongside physical reference formulas.

Requires ngspice; no installed PDK models are needed. Run from the repository root:
    .venv/bin/python -m silk.examples.metrics.rlc_step
"""
from __future__ import annotations

import math

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.elements import elements
from silk.metrics import (window, argmax_x, cross_correlation,
                          derivative, error_norm, lshift, overshoot,
                          peak, running_integral, sample, settling_time,
                          undershoot)
from silk.signals.base import Signal
from silk.simulation import Transient

L_H = 1e-6
C_F = 1e-9
F0_HZ = 1.0 / (2.0 * math.pi * math.sqrt(L_H * C_F))
Z0_OHM = math.sqrt(L_H / C_F)             # 31.6 Ω — the R that gives Q = 1
Q_STEP = Z0_OHM / 10.0                    # R = 10 Ω  -> Q = 3.162 (underdamped)
R_OHM = Z0_OHM / Q_STEP

WN = 1.0 / math.sqrt(L_H * C_F)           # rad/s, natural frequency
ZETA = 1.0 / (2.0 * Q_STEP)               # damping ratio
WD = WN * math.sqrt(1.0 - ZETA ** 2)      # damped (ringing) frequency
T_PEAK = math.pi / WD                     # time of the first overshoot peak
T_RING = 2.0 * math.pi / WD               # ring period 2π/ωd

# Delay the drive edge so settling time is measured relative to the transition.
V0 = 1.0
TD_S = 50e-9
TR_S = 1e-9                               # finite step edge
TOL = 0.02                                # ±2% settling band

STEP_S = 0.5e-9
STOP_S = TD_S + 3.0e-6                    # well past settling (closed ts ≈ 0.78 µs)


def build_rlc(out_dir: str) -> Module:
    """The series RLC from rlc_filter_ac.py with a delayed step drive."""
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, 'rlc_step.cir', pdk_details=pdk)
    mod.add_source(sources.VoltageSource(
        'vin', positive='vin', negative='0', dc=0.0,
        pulse=sources.Pulse(v1=0.0, v2=V0, td=TD_S, tr=TR_S, tf=TR_S,
                            pw=STOP_S, per=2 * STOP_S)))
    with mod:
        elements.resistor('r1', p='vin', n='nlc', value=R_OHM)
        elements.inductor('l1', p='nlc', n='vout', value=L_H)
        elements.capacitor('c1', p='vout', n='0', value=C_F)
    mod.add_sim_type(Transient(step=STEP_S, stop=STOP_S))
    return mod


def ideal_step(x: np.ndarray) -> Signal:
    """The analytic step as the reference waveform on the supplied time grid."""
    t = np.asarray(x, dtype=float) - TD_S
    y = V0 * (1.0 - (1.0 / math.sqrt(1 - ZETA ** 2))
              * np.exp(-ZETA * WN * t) * np.sin(WD * t + math.acos(ZETA)))
    y = np.where(t < 0.0, 0.0, y)
    return Signal('v_ideal', 'voltage', 'time', 'V', np.asarray(x, dtype=float), y)


def measure_core(vout: Signal, goal: Signal) -> dict:
    """Settling, overshoot, undershoot and RMS error on the simulated step."""
    return {
        'settling_time_ns': settling_time(vout, initial=0.0, target=V0, tol=TOL,
                                          start=TD_S) * 1e9,
        'overshoot': overshoot(vout, initial=0.0, target=V0),
        'undershoot': undershoot(vout, initial=0.0, target=V0),
        'error_norm_rms_mV': error_norm(vout, goal, kind='rms') * 1e3,
    }


def measure_derived(vout: Signal, vin: Signal) -> dict:
    """Current, charge, edge delay and model-guided ringing period."""
    # Capacitor current follows i = C dv/dt.
    dv = derivative(vout)
    i_c = Signal('i_loop', 'current', 'time', 'A',
                np.asarray(dv.x), C_F * np.asarray(dv.y))
    peak_current_mA = peak(i_c) * 1e3

    charge_nC = float(running_integral(i_c).y[-1]) * 1e9

    # Resample adaptive SPICE data onto matching uniform grids for correlation.
    edge = derivative(vin)
    a_win = lshift(window(edge, TD_S - 20e-9, TD_S + 400e-9), TD_S)
    b_win = lshift(window(vout, TD_S - 20e-9, TD_S + 400e-9), TD_S)
    n_pts = max(len(a_win.x), len(b_win.x))
    cc = cross_correlation(sample(a_win, n=n_pts), sample(b_win, n=n_pts))
    delay_ns = argmax_x(cc.mag()) * 1e9

    # Search for the first autocorrelation side-lobe near the analytic ring period.
    ring_win = window(vout, TD_S + 0.5 * T_PEAK, TD_S + 0.5 * T_PEAK + 2.0 * T_RING)
    ring = sample(ring_win, n=4 * len(ring_win.x))
    ring0 = Signal(ring.name, ring.type, ring.xlabel, ring.ylabel,
                   ring.x - ring.x[0], ring.y - float(np.mean(ring.y)))
    ac = cross_correlation(ring0, ring0)
    mid = len(ac.y) // 2
    dx = float(ring0.x[1] - ring0.x[0])
    expect_k = max(1, int(round(T_RING / dx)))
    lo, hi = max(1, expect_k - 15), expect_k + 15
    ring_period_ns = (argmax_x(ac, start=float(ac.x[mid + lo]),
                               end=float(ac.x[mid + hi - 1]),
                               interpolate=False) - float(ac.x[mid])) * 1e9

    return {
        'peak_loop_current_mA': peak_current_mA,
        'charge_delivered_nC': charge_nC,
        'edge_to_peak_delay_ns': abs(delay_ns),
        'ring_period_ns': ring_period_ns,
    }


_M_P = math.exp(-ZETA * math.pi / math.sqrt(1 - ZETA ** 2))
_TS_CLOSED_NS = math.log(1.0 / (TOL * math.sqrt(1 - ZETA ** 2))) / (ZETA * WN) * 1e9

_CORE_NOTES = {
    'settling_time_ns': ('closed ts = ln(1/(tol*sqrt(1-z^2)))/(z*wn) = %.4f ns '
                         '(envelope, not exact crossing)' % _TS_CLOSED_NS),
    'overshoot': 'M_p = exp(-z*pi/sqrt(1-z^2)) = %.6f' % _M_P,
    'undershoot': ('M_p^2 = %.6f (second extremum, one half-cycle later)'
                  % _M_P ** 2),
    'error_norm_rms_mV': 'RMS difference from the ideal step; includes finite-rise and solver effects.',
}
_DERIVED_NOTES = {
    'peak_loop_current_mA': ('i(t) = C dv/dt'),
    'charge_delivered_nC': ('closed C*V0 = %.4f nC (charge conservation)'
                            % (C_F * V0 * 1e9)),
    'edge_to_peak_delay_ns': 'closed t_peak = pi/wd = %.4f ns' % (T_PEAK * 1e9),
    'ring_period_ns': 'closed 2*pi/wd = %.4f ns' % (T_RING * 1e9),
}


def _report(title: str, values: dict, notes: dict) -> None:
    print()
    print(title)
    for key, val in values.items():
        print(f'  {key:26s} {val:14.6f}   {notes[key]}')


def main() -> int:
    out_dir = example_output_dir('rlc_step')
    print('=' * 92)
    print('Step-response metrics on a series RLC — no PDK device, no PDK install')
    print('=' * 92)
    print(f'  L = {L_H * 1e6:.3g} µH   C = {C_F * 1e9:.3g} nF   R = {R_OHM:.1f} Ω   '
          f'Q = {Q_STEP:.3f}   f0 = {F0_HZ / 1e6:.3f} MHz')
    print(f'  step: 0 -> {V0} V at td={TD_S * 1e9:.0f} ns, tr=tf={TR_S * 1e9:.0f} ns   '
          f'sim stop = {STOP_S * 1e6:.2f} µs')
    print(f'  decks and rawfiles: {out_dir}')

    mod = build_rlc(out_dir)
    mod.simulate()
    vout = mod.load_signal('vout')
    vin = mod.load_signal('vin')
    goal = ideal_step(vout.x)

    _report('--- core metrics: settling_time, overshoot, undershoot, error_norm '
            + '-' * 15, measure_core(vout, goal), _CORE_NOTES)
    _report('--- derived measurements via waveform transforms '
            + '-' * 40, measure_derived(vout, vin), _DERIVED_NOTES)

    print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
