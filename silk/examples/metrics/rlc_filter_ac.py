"""AC filter measurements on a series RLC low-pass circuit.

An AC voltage source drives a series resistor and inductor with a shunt
capacitor at the output. Sweep two resistor values: one gives a resonant
peak and the other a maximally flat response. Print gain, phase, Q, bandwidth,
passband variation, stopband attenuation and group delay. Compare each
measurement with the same metric applied to the analytic transfer function
on the simulation frequency grid:

    H(s) = 1 / (1 + sRC + s²LC)
    f0 = 1 / (2π√(LC)), Q = √(L/C) / R

The analytic comparison checks the simulated response, while the table notes
supply formula-based references. bandwidth_3dB and q_from_peak select sweep
points without interpolation, so their results depend on sweep density.

Requires ngspice; no installed PDK models are needed. Run from the repository root:
    .venv/bin/python -m silk.examples.metrics.rlc_filter_ac
"""
from __future__ import annotations

import math

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.elements import elements
from silk.metrics import (bandwidth_3dB, dc_gain_dB, gain_at_freq, group_delay,
                          passband_ripple_db, peak_gain_dB, phase_at_freq,
                          q_from_peak, stopband_attenuation_db, value_at)
from silk.signals.base import Signal
from silk.simulation import AC

L_H = 1e-6
C_F = 1e-9
F0_HZ = 1.0 / (2.0 * math.pi * math.sqrt(L_H * C_F))
Z0_OHM = math.sqrt(L_H / C_F)            # 31.6 Ω — the R that gives Q = 1

Q_PEAKED = Z0_OHM / 10.0                 # R = 10 Ω  -> Q = 3.162
Q_FLAT = 1.0 / math.sqrt(2.0)            # R = 44.7 Ω -> maximally flat

# Use a dense sweep because bandwidth and peak Q select grid points.
POINTS_PER_DECADE = 1000
F_START_HZ, F_STOP_HZ = 1e4, 1e9

F_PASS_HZ = 0.3 * F0_HZ                  # passband edge for the ripple read
F_STOP_BAND_HZ = 10.0 * F0_HZ            # a decade past f0: expect 40 dB


def build_rlc(out_dir: str, *, q: float, name: str) -> Module:
    """Build a series RLC low-pass with R = √(L/C) / q and an AC testbench."""
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, f'{name}.cir', pdk_details=pdk)
    # A unit AC input makes vout equal to the transfer function.
    mod.add_source(sources.VoltageSource('vin', positive='vin', negative='0',
                                         dc=0.0, ac=1.0))
    with mod:
        elements.resistor('r1', p='vin', n='nlc', value=Z0_OHM / q)
        elements.inductor('l1', p='nlc', n='vout', value=L_H)
        elements.capacitor('c1', p='vout', n='0', value=C_F)
    mod.add_sim_type(AC('dec', POINTS_PER_DECADE, F_START_HZ, F_STOP_HZ))
    return mod


def ideal_response(freqs: np.ndarray, *, q: float) -> Signal:
    """Evaluate H(jω) on the supplied frequency grid as a complex Signal."""
    x = np.asarray(freqs, dtype=float) / F0_HZ          # normalised frequency
    h = 1.0 / (1.0 - x ** 2 + 1j * x / q)
    return Signal('H_ideal', 'voltage', 'frequency', 'V',
                  np.asarray(freqs, dtype=float), h)


def _sweep(out_dir: str, *, q: float, name: str) -> Signal:
    mod = build_rlc(out_dir, q=q, name=name)
    mod.simulate()
    return mod.load_signal('vout')


def measure_peaked(sig: Signal) -> dict:
    """Measure gain, phase and peak-based Q on the resonant response."""
    return {
        'dc_gain_dB': dc_gain_dB(sig),
        'peak_gain_dB': peak_gain_dB(sig),
        'q_from_peak': q_from_peak(sig),
        'gain_at_f0_VV': gain_at_freq(sig, F0_HZ),
        'phase_at_f0_deg': phase_at_freq(sig, F0_HZ),
    }


def measure_flat(sig: Signal) -> dict:
    """Measure bandwidth, passband variation, attenuation and low-frequency delay."""
    tau = group_delay(sig)
    return {
        'bandwidth_3dB_MHz': bandwidth_3dB(sig) / 1e6,
        'passband_ripple_dB': passband_ripple_db(sig, F_PASS_HZ),
        'stopband_atten_dB': stopband_attenuation_db(sig, F_STOP_BAND_HZ),
        'group_delay_lf_ns': value_at(tau, float(tau.x[0])) * 1e9,
    }


_NOTES = {
    'dc_gain_dB': 'H(0) = 1',
    'peak_gain_dB': 'Q/√(1−1/4Q²) = %.3f dB' % (
        20 * math.log10(Q_PEAKED / math.sqrt(1 - 1 / (4 * Q_PEAKED ** 2)))),
    'q_from_peak': 'textbook Q = %.3f; grid selection and f_peak < f0 affect peak-based Q' % Q_PEAKED,
    'gain_at_f0_VV': '|H(jω0)| = Q = %.3f' % Q_PEAKED,
    'phase_at_f0_deg': '−90° at f0 for any Q',
    'bandwidth_3dB_MHz': 'at Q = 1/√2 the corner is f0 = %.3f MHz' % (F0_HZ / 1e6),
    'passband_ripple_dB': 'Butterworth passband droop to 0.3·f0',
    'stopband_atten_dB': '2nd order: 40 dB a decade past f0',
    'group_delay_lf_ns': 'τ(0) = R·C = %.3f ns' % (Z0_OHM / Q_FLAT * C_F * 1e9),
}


def _report(title: str, spice: dict, ideal: dict) -> None:
    print()
    print(title)
    print(f'  {"metric":24s} {"SPICE":>12s} {"ideal H(s)":>12s} {"rel":>9s}   note')
    for key in spice:
        a, b = spice[key], ideal[key]
        rel = abs(a - b) / abs(b) if b else abs(a - b)
        print(f'  {key:24s} {a:12.4f} {b:12.4f} {rel:8.2e}   {_NOTES[key]}')


def main() -> int:
    out_dir = example_output_dir('rlc_filter_ac')
    print('=' * 86)
    print('AC filter metrics on a series RLC — no PDK device, no PDK install')
    print('=' * 86)
    print(f'  L = {L_H * 1e6:.3g} µH   C = {C_F * 1e9:.3g} nF   '
          f'Z0 = √(L/C) = {Z0_OHM:.1f} Ω   f0 = {F0_HZ / 1e6:.3f} MHz')
    print(f'  AC sweep: dec {POINTS_PER_DECADE}, '
          f'{F_START_HZ:.0g} Hz .. {F_STOP_HZ:.0g} Hz')
    print(f'  decks and rawfiles: {out_dir}')

    sig = _sweep(out_dir, q=Q_PEAKED, name='rlc_peaked')
    _report(f'--- R = 10.0 Ω  ->  Q = Z0/R = {Q_PEAKED:.3f}  (resonant) '
            + '-' * 26,
            measure_peaked(sig), measure_peaked(ideal_response(sig.x, q=Q_PEAKED)))

    sig_flat = _sweep(out_dir, q=Q_FLAT, name='rlc_flat')
    _report(f'--- R = {Z0_OHM / Q_FLAT:.1f} Ω  ->  Q = {Q_FLAT:.3f}  '
            '(maximally flat) ' + '-' * 18,
            measure_flat(sig_flat), measure_flat(ideal_response(sig_flat.x, q=Q_FLAT)))

    print()
    print('The "rel" column compares SPICE with the analytic response on the same grid.')
    print('Both columns use the same metrics.')
    print('This comparison cannot detect shared metric errors.')
    print('The notes give formula-based references. Peak-based Q also depends on')
    print('the peak frequency and sweep grid. bandwidth_3dB and q_from_peak select')
    print('grid points without interpolation.')
    print('=' * 86)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
