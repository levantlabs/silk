"""Measure gain compression and efficiency in a common-source amplitude sweep.

A Variable sets the input sine amplitude. VariableSpace supplies ten grid
points from 2 mV to 300 mV. Each transient provides fundamental amplitudes,
supply power and gate power. The example prints a power-sweep table, the
input-referred 1 dB compression point, efficiency and power-added efficiency.

The sweep axes convert peak sine amplitudes to power using A²/(2*50 Ω).
The 50 Ω value is a reference impedance. Output efficiency uses the drain
load power A_out²/(2*Rd). Power integrals span seven whole cycles; FFT
samples use a coherent record at the same frequency.

Requires ngspice and sky130 models. Run with::

    python -m silk.examples.metrics.cs_amp_power_sweep
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.circuit.circuit_elements import Port
from silk.circuit.variable import Variable
from silk.elements import elements
from silk.metrics import (average_power, dc_to_rf_efficiency, fft, p1db,
                          pae)
from silk.optimize import VariableSpace
from silk.signals import power_sweep
from silk.signals.base import Signal
from silk.simulation import Transient

RD_OHM = 20e3
W_UM, L_UM = 4.0, 0.5  # µm
VDD_V = 1.8
VGS_V = 0.75
TEMP_C = 27.0

FS_HZ = 1e9
TCLK_S = 1.0 / FS_HZ
N_SAMPLES = 256
CYCLES = 7
F0_HZ = CYCLES * FS_HZ / N_SAMPLES

# Use a whole-cycle window for power integration and coherent FFT sampling.
T_WIN_S = N_SAMPLES * TCLK_S

# Limit the integration step to resolve waveform curvature before interpolation.
TSTEP_S = 1e-12
STOP_S = (N_SAMPLES + 20) * TCLK_S

AMPL_MIN_V, AMPL_MAX_V = 0.002, 0.300
N_POINTS = 10

Z0_REF_OHM = 50.0
MW = 1e-3

_SAMPLING = dict(tclk=TCLK_S, tstart=0.0, n_samples=N_SAMPLES)


def build_cs_amp(out_dir: str):
    """Build the stage and return its module, testbench and amplitude variable."""
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, 'cs_amp_power_sweep.cir',
                 pdk_details=pdk, module_name='cs_amp_power_sweep')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VIN': Port.analog_input(), 'VOUT': Port.analog_output(),
    })
    with mod:
        pdk.nmos1p8('M1', d='VOUT', g='VIN', s='VSS', b='VSS', w=W_UM, l=L_UM)
        elements.resistor('rd', p='VDD', n='VOUT', value=RD_OHM)

    vampl = Variable(name='vampl', value=AMPL_MIN_V,
                           bounds=(AMPL_MIN_V, AMPL_MAX_V))
    tb = _attach_test(mod, vampl)
    # Register testbenches before propagating their temperature.
    mod.set_temperature(TEMP_C)
    return mod, tb, vampl


def _attach_test(mod: Module, vampl: Variable):
    """Attach a transient bench with zero-volt sources for supply and gate currents."""
    tb = mod.create_test('tran')
    tb.add_module_under_test({'VDD': 'vdd', 'VSS': '0', 'VIN': 'vin',
                              'VOUT': 'vout'})
    tb.add_source(sources.VoltageSource('dd', positive='vdd_src',
                                        negative='0', dc=VDD_V))
    tb.add_source(sources.VoltageSource('amm_dd', positive='vdd_src',
                                        negative='vdd', dc=0.0))
    tb.add_source(sources.VoltageSource(
        'g', positive='vin_src', negative='0',
        sine=sources.Sine(voffset=VGS_V, vampl=vampl, freq=F0_HZ)))
    tb.add_source(sources.VoltageSource('amm_g', positive='vin_src',
                                        negative='vin', dc=0.0))
    tb.add_sim_type(Transient(step=TSTEP_S, stop=STOP_S, maxstep=TSTEP_S))
    return tb


@dataclass(frozen=True)
class Point:
    """Amplitudes, powers and stimulus card from one sweep point."""
    vampl_v: float
    a_in_v: float
    a_out_v: float
    p_dc_w: float
    p_rf_in_w: float
    p_rf_out_w: float
    sine_line: str


def _dbm(amplitude_v: float, z0_ohm: float = Z0_REF_OHM) -> float:
    """Convert peak sine voltage to average power in dBm using the reference impedance."""
    return 10.0 * np.log10((amplitude_v ** 2 / 2.0) / z0_ohm / MW)


def _fundamental_amplitude(sig: Signal) -> float:
    """Return the peak amplitude in the FFT bin nearest F0_HZ."""
    spec = fft(sig, scaling='amplitude', **_SAMPLING)
    return float(spec.y[int(np.argmin(np.abs(spec.x - F0_HZ)))])


def _sine_card(report) -> str:
    """The emitted stimulus line, read back out of the deck this run used."""
    path = report.run.deck_path
    with open(path) as fh:
        for line in fh:
            if 'SIN(' in line:
                return line.strip()
    return '(no SIN( ) card found)'


def run_point(mod: Module, tb, vampl_v: float) -> Point:
    """Simulate once at the amplitude currently held by the variable."""
    report = mod.run_test('tran')
    res = report.sim_results
    vin = res.load_signal('vin')
    vout = res.load_signal('vout')
    vdd = res.load_signal('vdd')
    i_sup = res.load_signal('vamm_dd', 'I')
    i_gate = res.load_signal('vamm_g', 'I')

    # Trim the traces to whole cycles before integrating average power.
    w = (0.0, T_WIN_S)
    a_out = _fundamental_amplitude(vout)
    return Point(
        vampl_v=vampl_v,
        a_in_v=_fundamental_amplitude(vin),
        a_out_v=a_out,
        p_dc_w=average_power(vdd.window(*w), i_sup.window(*w)),
        p_rf_in_w=average_power(vin.window(*w), i_gate.window(*w)),
        p_rf_out_w=a_out ** 2 / (2.0 * RD_OHM),
        sine_line=_sine_card(report),
    )


def sweep(mod: Module, tb, vampl: Variable) -> list:
    """Apply each amplitude grid point, simulate and collect its measurements."""
    space = VariableSpace([vampl])
    points = space.sample(N_POINTS, method='grid')
    results = []
    for i, point in enumerate(points, start=1):
        space.apply(point)
        print(f'  point {i:2d}/{len(points)}: vampl = '
              f'{point["vampl"] * 1e3:7.3f} mV', flush=True)
        results.append(run_point(mod, tb, point['vampl']))
    return results


def power_sweep_signal(points: list, z0_out_ohm: float = Z0_REF_OHM) -> Signal:
    """Assemble input and output powers in dBm for p1db.

    Input powers are ascending. z0_out_ohm sets the output reference impedance.
    """
    p_in = np.array([_dbm(p.a_in_v) for p in points], dtype=float)
    p_out = np.array([_dbm(p.a_out_v, z0_out_ohm) for p in points],
                     dtype=float)
    return power_sweep(p_in, p_out, name='cs_amp_power_sweep')


def measure(points: list) -> dict:
    """Measure the compression point from the sweep and efficiencies from each run."""
    sweep_50 = power_sweep_signal(points)
    return {
        'sweep': sweep_50,
        'p1db_dbm': p1db(sweep_50),
        'pae': [pae(p.p_rf_out_w, p.p_rf_in_w, p.p_dc_w) for p in points],
        'eta': [dc_to_rf_efficiency(p.p_rf_out_w, p.p_dc_w) for p in points],
    }


def report(points: list, m: dict) -> None:
    sweep_sig = m['sweep']
    p_in, p_out = sweep_sig.x, sweep_sig.y
    gain = p_out - p_in
    compression = gain[0] - gain

    print()
    print('=' * 92)
    print('cs_amp_power_sweep — p1db from a curve assembled across '
          f'{len(points)} runs')
    print('=' * 92)
    print(f'  M1 nmos1p8 W/L = {W_UM:g}/{L_UM:g} um   Rd = {RD_OHM / 1e3:.0f} kOhm'
          f'   VDD = {VDD_V} V   Vgs = {VGS_V} V')
    print(f'  fs = {FS_HZ / 1e9:.0f} GHz   n_samples = {N_SAMPLES}   '
          f'cycles = {CYCLES}   f0 = {F0_HZ / 1e6:.5f} MHz '
          f'(= {CYCLES}*fs/{N_SAMPLES}, exact, every point)')
    print(f'  vampl swept {AMPL_MIN_V * 1e3:g} mV .. {AMPL_MAX_V * 1e3:g} mV, '
          f"grid of {N_POINTS} (VariableSpace.sample(n, method='grid'))")

    print()
    print('--- the power sweep: x = p_in (dBm), y = p_out (dBm), both at '
          f'{Z0_REF_OHM:g} Ohm ' + '-' * 8)
    print(f'  {"#":>2s}  {"vampl":>9s}  {"A_in":>9s}  {"A_out":>9s}  '
          f'{"p_in":>8s}  {"p_out":>8s}  {"gain":>7s}  {"compr.":>7s}')
    print(f'  {"":>2s}  {"(mV)":>9s}  {"(mV)":>9s}  {"(mV)":>9s}  '
          f'{"(dBm)":>8s}  {"(dBm)":>8s}  {"(dB)":>7s}  {"(dB)":>7s}')
    for i, p in enumerate(points):
        print(f'  {i + 1:2d}  {p.vampl_v * 1e3:9.3f}  {p.a_in_v * 1e3:9.3f}  '
              f'{p.a_out_v * 1e3:9.3f}  {p_in[i]:8.3f}  {p_out[i]:8.3f}  '
              f'{gain[i]:7.3f}  {compression[i]:7.3f}')
    print()
    print('--- p1db ' + '-' * 82)
    print(f'  p1db = {m["p1db_dbm"]:.4f} dBm  (input-referred, '
          f'{Z0_REF_OHM:g} Ohm)')
    print(f'  at that input, A_in = {10 ** (m["p1db_dbm"] / 20.0) * np.sqrt(2 * Z0_REF_OHM * MW) * 1e3:.2f} mV')
    print()
    print('--- pae and dc_to_rf_efficiency, from measured watts ' + '-' * 39)
    print(f'  {"#":>2s}  {"p_in":>8s}  {"P_rf,out":>10s}  {"P_rf,in":>11s}  '
          f'{"P_dc":>9s}  {"eta":>8s}  {"PAE":>8s}')
    print(f'  {"":>2s}  {"(dBm)":>8s}  {"(uW)":>10s}  {"(W)":>11s}  '
          f'{"(uW)":>9s}  {"(%)":>8s}  {"(%)":>8s}')
    for i, p in enumerate(points):
        print(f'  {i + 1:2d}  {p_in[i]:8.3f}  {p.p_rf_out_w * 1e6:10.4f}  '
              f'{p.p_rf_in_w:11.3e}  {p.p_dc_w * 1e6:9.3f}  '
              f'{100 * m["eta"][i]:8.4f}  {100 * m["pae"][i]:8.4f}')
    print('  Efficiency uses the real drain-load power, not the 50 Ohm '
          'reference used for the sweep axes.')
    print('=' * 92)


def main() -> int:
    out_dir = example_output_dir('cs_amp_power_sweep')
    print(f'decks and rawfiles: {out_dir}')
    mod, tb, vampl = build_cs_amp(out_dir)
    print(f'sweeping {vampl.name} over {vampl.bounds} V, '
          f'{N_POINTS} grid points, one transient each:')
    points = sweep(mod, tb, vampl)
    report(points, measure(points))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
