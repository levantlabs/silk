"""Measure third-order intermodulation in a sky130 common-source amplifier.

Three series voltage sources sum a DC gate bias and two sine tones.
Frequencies f1 = 11*fs/256 and f2 = 13*fs/256 put the fundamentals and
third-order products on distinct FFT bins below Nyquist.

The example prints IMD3, mixing-product amplitudes and input-referred
third-order intercept estimates at two drive levels. It also compares the
fundamental and IMD3 growth with the 1:1 and 3:1 slopes used for intercept
extrapolation. Sine power uses the peak amplitude A and A²/(2*50 Ω).
The 50 Ω reference is not a physical termination.

Requires ngspice and sky130 models. Run with::

    python -m silk.examples.metrics.cs_amp_two_tone
"""
from __future__ import annotations

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.circuit.circuit_elements import Port
from silk.elements import elements
from silk.metrics import fft, imd3_db, iip3_dbm, mix_product, sample
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
DF_HZ = FS_HZ / N_SAMPLES
K1, K2 = 11, 13
F1_HZ = K1 * FS_HZ / N_SAMPLES
F2_HZ = K2 * FS_HZ / N_SAMPLES

AMPL_LO_V = 0.02
AMPL_HI_V = 0.08

# Limit the integration step to resolve waveform curvature before interpolation.
TSTEP_S = 1e-12
STOP_S = (N_SAMPLES + 20) * TCLK_S

_Z0_OHM = 50.0


def build_cs_amp(out_dir: str, ampl_v: float, label: str) -> Module:
    """One NMOS, one resistor, two series-summed tones on the gate."""
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, f'cs_amp_two_tone_{label}.cir',
                 pdk_details=pdk, module_name=f'cs_amp_two_tone_{label}')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VIN': Port.analog_input(), 'VOUT': Port.analog_output(),
    })
    with mod:
        pdk.nmos1p8('M1', d='VOUT', g='VIN', s='VSS', b='VSS', w=W_UM, l=L_UM)
        elements.resistor('rd', p='VDD', n='VOUT', value=RD_OHM)
    _attach_test(mod, ampl_v)
    mod.set_temperature(TEMP_C)
    return mod


def _attach_test(mod: Module, ampl_v: float):
    tb = mod.create_test('tran')
    tb.add_module_under_test({'VDD': 'vdd', 'VSS': '0', 'VIN': 'vin',
                              'VOUT': 'vout'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                        dc=VDD_V))
    # Series sources sum the bias and two tones at VIN.
    tb.add_source(sources.VoltageSource('bias', positive='vin_a',
                                        negative='0', dc=VGS_V))
    tb.add_source(sources.VoltageSource(
        'tone1', positive='vin_b', negative='vin_a', dc=0.0,
        sine=sources.Sine(voffset=0.0, vampl=ampl_v, freq=F1_HZ)))
    tb.add_source(sources.VoltageSource(
        'tone2', positive='vin', negative='vin_b', dc=0.0,
        sine=sources.Sine(voffset=0.0, vampl=ampl_v, freq=F2_HZ)))
    tb.add_sim_type(Transient(step=TSTEP_S, stop=STOP_S, maxstep=TSTEP_S))
    return tb


_SAMPLING = dict(tclk=TCLK_S, tstart=0.0, n_samples=N_SAMPLES)


def _amplitude_dbm(v_amplitude: float, z0: float = _Z0_OHM) -> float:
    """Convert peak sine voltage to average power in dBm using the reference impedance."""
    return float(10.0 * np.log10(v_amplitude ** 2 / 2.0 / z0 / 1e-3))


def _bin_amplitude(spectrum: Signal, freq_hz: float) -> float:
    """Return the peak-voltage amplitude in the FFT bin nearest freq_hz."""
    return float(spectrum.y[int(np.argmin(np.abs(spectrum.x - freq_hz)))])


def measure(vout: Signal, vin: Signal, ampl_v: float) -> dict:
    """Measure IMD3, mixing products and intercept from output and input waveforms."""
    out = {'ampl_v': ampl_v}

    out['imd3_db'] = imd3_db(vout, tclk=TCLK_S, f1_hz=F1_HZ, f2_hz=F2_HZ,
                             tstart=0.0, n_samples=N_SAMPLES)

    out['mp_f1'] = mix_product(vout, 1, 0, f1_hz=F1_HZ, f2_hz=F2_HZ,
                               **_SAMPLING)
    out['mp_f2'] = mix_product(vout, 0, 1, f1_hz=F1_HZ, f2_hz=F2_HZ,
                               **_SAMPLING)
    out['mp_2f1_f2'] = mix_product(vout, 2, -1, f1_hz=F1_HZ, f2_hz=F2_HZ,
                                   **_SAMPLING)
    out['mp_2f2_f1'] = mix_product(vout, -1, 2, f1_hz=F1_HZ, f2_hz=F2_HZ,
                                   **_SAMPLING)

    spec = fft(vout, scaling='amplitude', **_SAMPLING)
    out['amp_f1'] = _bin_amplitude(spec, F1_HZ)
    out['amp_f2'] = _bin_amplitude(spec, F2_HZ)
    fund_amp = max(out['amp_f1'], out['amp_f2'])
    imd_amp = max(_bin_amplitude(spec, 2 * F1_HZ - F2_HZ),
                  _bin_amplitude(spec, 2 * F2_HZ - F1_HZ))

    out['p_in_dbm'] = _amplitude_dbm(ampl_v)
    out['p_fund_dbm'] = _amplitude_dbm(fund_amp)
    out['p_imd3_dbm'] = _amplitude_dbm(imd_amp)

    # Sample both traces on the same time grid for IIP3.
    times = _SAMPLING['tstart'] + np.arange(N_SAMPLES) * TCLK_S
    out['iip3_dbm'] = iip3_dbm(
        sample(vout, x_new=times), sample(vin, x_new=times),
        f1_hz=F1_HZ, f2_hz=F2_HZ, r_ref=_Z0_OHM, **_SAMPLING)

    return out


def report_run(label: str, m: dict) -> None:
    print()
    print('-' * 88)
    print(f'  drive level: {label}   A = {m["ampl_v"] * 1e3:.0f} mV / tone')
    print('-' * 88)

    print(f'  imd3_db(vout, f1, f2)              = {m["imd3_db"]:.3f} dB '
          '(larger IMD3 product, below the larger fundamental)')

    print()
    print(f'  mix_product(1,0) |X(f1)| = {abs(m["mp_f1"]):.6e}   '
          f'mix_product(0,1) |X(f2)| = {abs(m["mp_f2"]):.6e}')

    print(f'  p_in   = {m["p_in_dbm"]:8.3f} dBm  ({m["ampl_v"] * 1e3:.0f} mV '
          f'amplitude @ {_Z0_OHM:.0f} ohm)')
    print(f'  p_fund = {m["p_fund_dbm"]:8.3f} dBm')
    print(f'  p_imd3 = {m["p_imd3_dbm"]:8.3f} dBm')
    print(f'  iip3_dbm(vout, vin) = '
          f'{m["iip3_dbm"]:8.3f} dBm')


def report_summary(m_lo: dict, m_hi: dict) -> None:
    print()
    print('=' * 88)
    print('cs_amp_two_tone -- imd3_db, mix_product, iip3_dbm, '
          'two drive levels')
    print('=' * 88)
    print(f'  M1 nmos1p8 W/L = {W_UM:g}/{L_UM:g} um   Rd = {RD_OHM / 1e3:.0f} '
          f'kOhm   VDD = {VDD_V} V   Vgs = {VGS_V} V')
    print(f'  fs = {FS_HZ / 1e9:.0f} GHz   n_samples = {N_SAMPLES}   '
          f'df = {DF_HZ / 1e6:.5f} MHz')
    print(f'  k1={K1}, k2={K2}  ->  f1 = {F1_HZ / 1e6:.5f} MHz   '
          f'f2 = {F2_HZ / 1e6:.5f} MHz')
    print(f'  bins used: f1={K1} f2={K2}  2f1-f2={2 * K1 - K2}  '
          f'2f2-f1={2 * K2 - K1}  2f1={2 * K1}  2f2={2 * K2}  '
          f'f1+f2={K1 + K2}  -- all distinct, all < Nyquist bin '
          f'{N_SAMPLES // 2}')

    report_run('LOW', m_lo)
    report_run('HIGH', m_hi)

    print()
    print('-' * 88)
    print('  3:1 slope check')
    print('-' * 88)
    gap = m_hi['iip3_dbm'] - m_lo['iip3_dbm']
    print(f'  gap = {gap:+.3f} dB')
    fund_gain_db = m_hi['p_fund_dbm'] - m_lo['p_fund_dbm']
    drive_gain_db = m_hi['p_in_dbm'] - m_lo['p_in_dbm']
    imd_gain_db = m_hi['p_imd3_dbm'] - m_lo['p_imd3_dbm']
    print(f'  drive stepped up {drive_gain_db:.3f} dB; fundamental grew '
          f'{fund_gain_db:.3f} dB (1:1 would be {drive_gain_db:.3f} dB); '
          f'IMD3 grew {imd_gain_db:.3f} dB (3:1 would be '
          f'{3 * drive_gain_db:.3f} dB)')
    print('=' * 88)


def _run_one(ampl_v: float, label: str) -> dict:
    out_dir = example_output_dir('cs_amp_two_tone', label)
    print(f'[{label}] decks and rawfiles: {out_dir}')
    mod = build_cs_amp(out_dir, ampl_v, label)
    results = mod.run_all_tests()
    ckt = results[f'cs_amp_two_tone_{label}.tran'].sim_results
    vout = ckt.load_signal('vout')
    return measure(vout, ckt.load_signal('vin'), ampl_v)


def main() -> int:
    m_lo = _run_one(AMPL_LO_V, 'lo')
    m_hi = _run_one(AMPL_HI_V, 'hi')
    report_summary(m_lo, m_hi)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
