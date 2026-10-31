"""Coherent single-tone distortion through a sky130 common-source amplifier.

An NMOS and drain resistor form a common-source stage. A biased sine drives
the gate; seven cycles fit in the spectral sample record. Print FFT and power
spectrum sizes, harmonic frequencies and levels, distortion and noise ratios,
band-integrated power, and input/output coherence. THD uses harmonic bins;
spectral_power integrates the selected band. Welch coherence uses shorter
segments that do not contain whole cycles of the tone.

Requires ngspice and sky130 models. Run from the repository root:
    .venv/bin/python -m silk.examples.metrics.cs_amp_distortion
"""
from __future__ import annotations

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.circuit.circuit_elements import Port
from silk.elements import elements
from silk.metrics import (argmax_x, coherence, fft, harmonic, harmonic_freqs,
                          harmonic_levels_db, noise_floor_dbfs, psd,
                          sfdr, sinad, sndr, snr,
                          spectral_power, thd, thd_percent, welch_psd)
from silk.signals.base import Signal
from silk.simulation import Transient

RD_OHM = 20e3        # drain load
W_UM, L_UM = 4.0, 0.5
VDD_V = 1.8
VGS_V = 0.75         # gate bias, near threshold
AMPL_V = 0.15        # large-signal distortion stimulus
TEMP_C = 27.0

FS_HZ = 1e9
TCLK_S = 1.0 / FS_HZ
N_SAMPLES = 256
CYCLES = 7
F0_HZ = CYCLES * FS_HZ / N_SAMPLES          # 27 343 750.0 Hz, exact

# Cap integration gaps at 1 ps; interpolate spectral samples at requested times.
TSTEP_S = 1e-12
STOP_S = (N_SAMPLES + 20) * TCLK_S           # 276 ns


def build_cs_amp(out_dir: str) -> Module:
    """Build an NMOS common-source stage with its sine-drive testbench."""
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, 'cs_amp_distortion.cir',
                 pdk_details=pdk, module_name='cs_amp_distortion')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VIN': Port.analog_input(), 'VOUT': Port.analog_output(),
    })
    with mod:
        pdk.nmos1p8('M1', d='VOUT', g='VIN', s='VSS', b='VSS', w=W_UM, l=L_UM)
        elements.resistor('rd', p='VDD', n='VOUT', value=RD_OHM)
    _attach_test(mod)
    # Set temperature after registering the test so it propagates to the bench.
    mod.set_temperature(TEMP_C)
    return mod


def _attach_test(mod: Module):
    tb = mod.create_test('tran')
    tb.add_module_under_test({'VDD': 'vdd', 'VSS': '0', 'VIN': 'vin',
                              'VOUT': 'vout'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                        dc=VDD_V))
    tb.add_source(sources.VoltageSource(
        'vin', positive='vin', negative='0',
        sine=sources.Sine(voffset=VGS_V, vampl=AMPL_V, freq=F0_HZ)))
    tb.add_sim_type(Transient(step=TSTEP_S, stop=STOP_S, maxstep=TSTEP_S))
    return tb


_SAMPLING = dict(tclk=TCLK_S, tstart=0.0, n_samples=N_SAMPLES)
_N_HARM_TABLE = 5          # harmonics 2..5 printed and summed for the band
# Integrate a band spanning harmonics 2 through 5 and the bins between them.
_DF_HZ = FS_HZ / N_SAMPLES                           # 3 906 250 Hz / bin
_BAND_LO_HZ = 1.5 * F0_HZ
_BAND_HI_HZ = 5.5 * F0_HZ


def measure(vin: Signal, vout: Signal) -> dict:
    """Measure spectra, harmonics, distortion, noise ratios and coherence."""
    out = {}
    out['fft'] = fft(vout, **_SAMPLING)
    out['psd'] = psd(vout, **_SAMPLING)
    out['welch_psd'] = welch_psd(vout, **_SAMPLING)
    out['harmonic'] = {k: harmonic(vout, k, **_SAMPLING)
                       for k in range(1, _N_HARM_TABLE + 1)}
    out['harmonic_freqs_true'] = harmonic_freqs(
        vout, n=_N_HARM_TABLE, observed=False, **_SAMPLING)
    out['harmonic_freqs_observed'] = harmonic_freqs(
        vout, n=_N_HARM_TABLE, observed=True, **_SAMPLING)
    out['harmonic_levels_db'] = harmonic_levels_db(
        vout, n_harmonics=_N_HARM_TABLE, **_SAMPLING)
    out['thd_db'] = thd(vout, n_harmonics=_N_HARM_TABLE, **_SAMPLING)
    out['thd_percent'] = thd_percent(vout, n_harmonics=_N_HARM_TABLE,
                                     **_SAMPLING)
    out['sndr_db'] = sndr(vout, **_SAMPLING)
    out['sinad_db'] = sinad(vout, **_SAMPLING)
    out['snr_db'] = snr(vout, n_harmonics=_N_HARM_TABLE, **_SAMPLING)
    out['sfdr'] = sfdr(vout, **_SAMPLING)
    out['noise_floor_dbfs'] = noise_floor_dbfs(vout, **_SAMPLING)
    out['spectral_power'] = spectral_power(vout, _BAND_LO_HZ, _BAND_HI_HZ,
                                           **_SAMPLING)
    out['coherence'] = coherence(vin, vout, **_SAMPLING)
    return out


def _finite(label: str, value: float) -> str:
    """Flag a non-finite measurement in the report."""
    if not np.isfinite(value):
        return f"{value!r}  <-- non-finite ({label})"
    return f'{value:.3f}'


def report(m: dict) -> None:
    print()
    print('=' * 88)
    print('cs_amp_distortion — spectral measurements with a coherent tone through '
          'a common-source stage')
    print('=' * 88)
    print(f'  M1 nmos1p8 W/L = {W_UM:g}/{L_UM:g} um   Rd = {RD_OHM / 1e3:.0f} kOhm'
          f'   VDD = {VDD_V} V   Vgs = {VGS_V} V   +/- {AMPL_V} V')
    print(f'  fs = {FS_HZ / 1e9:.0f} GHz   n_samples = {N_SAMPLES}   '
          f'cycles = {CYCLES}   f0 = {F0_HZ / 1e6:.5f} MHz  '
          f'(= {CYCLES}*fs/{N_SAMPLES}, exact)')
    print(f'  bin width df = fs/n_samples = {_DF_HZ / 1e6:.5f} MHz')

    print()
    print('--- fft / psd / welch_psd -- all return Signal, '
          'lengths ' + '-' * 5)
    print(f'  fft(vout):           {len(m["fft"].x):4d} bins, peak at '
          f'{argmax_x(m["fft"], interpolate=False) / 1e6:.5f} MHz')
    print(f'  psd(vout):           {len(m["psd"].x):4d} bins (Hann-windowed '
          f'V^2/Hz)')
    print(f'  welch_psd(vout):     {len(m["welch_psd"].x):4d} bins '
          '(8 segments, 50% overlap)')

    print()
    print('--- harmonic(vout, k) vs harmonic_freqs -- predicted freq is k*f0 '
          '-' * 3)
    print(f'  {"k":>2s}  {"|X_k|":>12s}  {"true k*f0 (MHz)":>16s}  '
          f'{"observed (MHz)":>15s}  {"level (dBc)":>12s}')
    for k in range(1, _N_HARM_TABLE + 1):
        mag = abs(m['harmonic'][k])
        f_true = m['harmonic_freqs_true'][k - 1] / 1e6
        f_obs = m['harmonic_freqs_observed'][k - 1] / 1e6
        lvl = 0.0 if k == 1 else m['harmonic_levels_db'][k - 2]
        print(f'  {k:2d}  {mag:12.6e}  {f_true:16.5f}  {f_obs:15.5f}  '
              f'{lvl:12.3f}')
    print()
    print('--- thd / thd_percent / spectral_power -- same harmonics, two '
          'different sums ' + '-' * 2)
    print(f'  thd (harmonic bins 2..{_N_HARM_TABLE} only): '
          f'{m["thd_db"]:.3f} dB   ({m["thd_percent"]:.3f} %)')
    print(f'  spectral_power over {_BAND_LO_HZ / 1e6:.3f}..'
          f'{_BAND_HI_HZ / 1e6:.3f} MHz (whole band, harmonics + floor): '
          f'{m["spectral_power"]:.6e} V^2')
    print()
    print('--- sndr / sinad / snr / sfdr / noise_floor_dbfs -- the ratio '
          'metrics ' + '-' * 6)
    print(f'  sndr:  {_finite("sndr", m["sndr_db"])} dB')
    print(f'  sinad: {_finite("sinad", m["sinad_db"])} dB')
    print(f'  snr (harmonics excluded): {_finite("snr", m["snr_db"])} dB')
    s = m['sfdr']
    print(f'  sfdr:  {_finite("sfdr", s.sfdr_db)} dB   fundamental = '
          f'{s.fundamental_hz / 1e6:.5f} MHz   spur = {s.spur_hz / 1e6:.5f} '
          f'MHz   (bin = {s.bin_hz / 1e6:.5f} MHz)')
    print(f'  noise_floor_dbfs: {_finite("noise_floor_dbfs", m["noise_floor_dbfs"])}'
          ' dB')

    print()
    print('--- coherence(vin, vout) -- Welch-segmented, coarser grid than '
          'the record FFT ' + '-' * 1)
    coh = m['coherence']
    seg_bins = len(coh.x)
    df_coh = coh.x[1] - coh.x[0] if seg_bins > 1 else float('nan')
    seg_len = round(FS_HZ / df_coh)   # back out the Welch segment length
    print(f'  {seg_bins} bins (8 segments, 50% overlap, {seg_len} samples '
          f'each) -- df = {df_coh / 1e6:.3f} MHz, vs {_DF_HZ / 1e6:.3f} MHz '
          'for the full record')
    print(f'  a {seg_len}-sample segment holds {seg_len * F0_HZ / FS_HZ:.2f} '
          'cycles of the tone; the segment does not contain whole cycles')
    fund_idx = int(np.argmin(np.abs(coh.x - F0_HZ)))
    print(f'  gamma^2 nearest the fundamental ({coh.x[fund_idx] / 1e6:.3f} '
          f'MHz, true {F0_HZ / 1e6:.3f} MHz): {coh.y[fund_idx]:.6f}')
    h2_idx = int(np.argmin(np.abs(coh.x - 2 * F0_HZ)))
    print(f'  gamma^2 nearest the 2nd harmonic ({coh.x[h2_idx] / 1e6:.3f} '
          f'MHz, true {2 * F0_HZ / 1e6:.3f} MHz): {coh.y[h2_idx]:.6f}')
    print('=' * 88)


def main() -> int:
    out_dir = example_output_dir('cs_amp_distortion')
    print(f'decks and rawfiles: {out_dir}')
    mod = build_cs_amp(out_dir)
    results = mod.run_all_tests()
    ckt = results['cs_amp_distortion.tran'].sim_results
    vin = ckt.load_signal('vin')
    vout = ckt.load_signal('vout')
    m = measure(vin, vout)
    report(m)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
