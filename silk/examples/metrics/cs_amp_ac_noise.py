"""AC gain, loop margins and noise of a sky130 common-source amplifier.

The AC bench measures the response through three loaded RC sections.
Loop margins use T(s) = -vout/vin. An ideal transconductance model fitted
to the measured low-frequency gain provides a comparison.
The noise bench measures the drain ahead of the RC ladder. It reports
integrated noise, input referral, noise figures and a flicker-noise corner.

The example prints AC model comparisons and noise tables. Requires ngspice
and sky130 models. Run with::

    python -m silk.examples.metrics.cs_amp_ac_noise
"""
from __future__ import annotations

import math

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.circuit.circuit_elements import Port
from silk.elements import elements
from silk.metrics import (bandwidth_3dB, cascade_noise_figure, dc_gain_dB,
                          gain_bw_product, gain_margin, gain_margins,
                          input_referred_noise, integrated_noise, noise_figure,
                          band_noise_figure, one_over_f_corner,
                          phase_margin, sum_uncorrelated, unity_gain_freq)
from silk.signals.base import Signal
from silk.simulation import AC, Noise

RS_OHM = 10e3
RD_OHM = 20e3
W_UM, L_UM = 4.0, 0.5  # µm
VDD_V = 1.8
VGS_V = 0.75

# The model solves the loaded ladder without assuming isolated RC sections.
RL_OHM = 200e3
CL_F = 160e-15

# A dense frequency grid limits sampling error in crossover measurements.
AC_POINTS, AC_F1, AC_F2 = 100, 1e3, 1e9
NOISE_POINTS, NOISE_F1, NOISE_F2 = 100, 1.0, 1e9

# Use SIG_BAND for noise figures and FULL_BAND to locate the white-noise floor.
SIG_BAND = (1.0, 1e6)
FULL_BAND = (NOISE_F1, NOISE_F2)

V_SIG_RMS = 1e-3
TEMP_C = 27.0
K_B = 1.380649e-23


def build_cs_amp(out_dir: str) -> Module:
    """Build the amplifier and attach AC and noise testbenches.

    VAMP is the drain before the RC ladder; VOUT is the ladder output.
    The AC bench measures VOUT and the noise bench measures VAMP.
    """
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, 'cs_amp.cir', pdk_details=pdk,
                 module_name='cs_amp')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VIN': Port.analog_input(),
        'VAMP': Port.analog_output(), 'VOUT': Port.analog_output(),
    })
    with mod:
        elements.resistor('rs', p='VIN', n='g', value=RS_OHM)
        pdk.nmos1p8('M1', d='VAMP', g='g', s='VSS', b='VSS', w=W_UM, l=L_UM)
        elements.resistor('rd', p='VDD', n='VAMP', value=RD_OHM)
        elements.resistor('r2', p='VAMP', n='n2', value=RL_OHM)
        elements.capacitor('c2', p='n2', n='VSS', value=CL_F)
        elements.resistor('r3', p='n2', n='n3', value=RL_OHM)
        elements.capacitor('c3', p='n3', n='VSS', value=CL_F)
        elements.resistor('r4', p='n3', n='VOUT', value=RL_OHM)
        elements.capacitor('c4', p='VOUT', n='VSS', value=CL_F)
    _attach_ac_test(mod)
    _attach_noise_test(mod)
    # Register testbenches before propagating their temperature.
    mod.set_temperature(TEMP_C)
    return mod


def _bias(tb):
    """Use unit AC drive; refer the noise analysis to the same input source."""
    tb.add_module_under_test({'VDD': 'vdd', 'VSS': '0', 'VIN': 'vin',
                              'VAMP': 'vamp', 'VOUT': 'vout'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                        dc=VDD_V))
    tb.add_source(sources.VoltageSource('vin', positive='vin', negative='0',
                                        dc=VGS_V, ac=1.0))


def _attach_ac_test(mod):
    """The ``.ac`` bench: the open-loop gain and its margins, measured at VOUT."""
    tb = mod.create_test('ac')
    _bias(tb)
    tb.add_sim_type(AC('dec', AC_POINTS, AC_F1, AC_F2))
    vout = tb.probe('vout')
    tb.add_metric('a0_dB', dc_gain_dB, sig=vout)
    tb.add_metric('ugf_MHz',
                  lambda c: unity_gain_freq(loop_gain(c.load_signal('vout'))) / 1e6)
    tb.add_metric('pm_deg', phase_margin, sig=loop_gain(vout))
    tb.add_metric('gm_dB', gain_margin, sig=loop_gain(vout))
    return tb


def _attach_noise_test(mod):
    """Attach a drain-noise analysis referred to the input voltage source.

    The input-referred spectrum includes the source resistor and amplifier.
    """
    tb = mod.create_test('noise')
    _bias(tb)
    tb.add_sim_type(Noise(output='vamp', ref='0', source='vin', variation='dec',
                          points=NOISE_POINTS, fstart=NOISE_F1, fstop=NOISE_F2))
    tb.add_metric('vn_in_uV',
                  lambda c: integrated_noise(c.load_signal('noise_in'),
                                             *SIG_BAND) * 1e6)
    tb.add_metric('corner_MHz',
                  lambda c: one_over_f_corner(c.load_signal('noise_in')) / 1e6)
    return tb


def loop_gain(sig: Signal) -> Signal:
    """Return T(s) = -A(s), preserving the complex frequency response."""
    return -sig


def ideal_response(freqs, gm_s: float) -> Signal:
    """Solve the loaded ladder with ideal gm; omit transistor ro and capacitances."""
    f = np.asarray(freqs, dtype=float)
    gd, gl = 1.0 / RD_OHM, 1.0 / RL_OHM
    out = np.empty(len(f), dtype=complex)
    drive = np.array([-gm_s, 0.0, 0.0, 0.0], dtype=complex)
    for i, s in enumerate(2j * np.pi * f):
        y = np.array([
            [gd + gl,      -gl,                 0.0,                0.0],
            [-gl,          2 * gl + s * CL_F,   -gl,                0.0],
            [0.0,          -gl,                 2 * gl + s * CL_F,  -gl],
            [0.0,          0.0,                 -gl,                gl + s * CL_F],
        ], dtype=complex)
        out[i] = np.linalg.solve(y, drive)[3]
    return Signal('A_ideal', 'voltage', 'frequency', 'V/V', f, out)


def measure_ac(sig: Signal) -> dict:
    """Measure gain, bandwidth and loop margins."""
    t = loop_gain(sig)
    return {
        'dc_gain_dB': dc_gain_dB(t),
        'bandwidth_3dB_MHz': bandwidth_3dB(t) / 1e6,
        'unity_gain_freq_MHz': unity_gain_freq(t) / 1e6,
        'gain_bw_product_MHz': gain_bw_product(t) / 1e6,
        'phase_margin_deg': phase_margin(t),
        'gain_margin_dB': gain_margin(t),
    }


_AC_NOTES = {
    'dc_gain_dB': 'gm·(Rd‖ro); the ladder draws no DC current',
    'bandwidth_3dB_MHz': '−3 dB bandwidth of the loaded response',
    'unity_gain_freq_MHz': 'where |T| = 1, interpolated',
    'gain_bw_product_MHz': 'A0 × f_3dB; distinct from unity-gain frequency',
    'phase_margin_deg': '180° + phase at the UGF',
    'gain_margin_dB': 'gain margin at the −180° crossing',
}


def report_ac(spice: dict, ideal: dict, crossings) -> None:
    print()
    print('--- .ac: open-loop gain and margins, measured on T(s) = −vout/vin '
          + '-' * 12)
    print(f'  {"metric":24s} {"SPICE":>12s} {"ideal gm+RC":>12s} {"rel":>9s}'
          '   note')
    for key in spice:
        a, b = spice[key], ideal[key]
        rel = abs(a - b) / abs(b) if b else abs(a - b)
        print(f'  {key:24s} {a:12.4f} {b:12.4f} {rel:8.2e}   {_AC_NOTES[key]}')
    print()
    print(f'  gain_margins(): {len(crossings)} phase crossover(s) — '
          + ', '.join(f'{f / 1e6:.3f} MHz at {gm:+.2f} dB' for f, gm in crossings))


def measure_noise(noise_in: Signal, noise_out: Signal, noise_rs: Signal,
                  a0_vv: float) -> dict:
    """Noise metrics from total and source-resistor spectra.

    *noise_in* / *noise_out* are ngspice's input- and output-referred
    densities; *noise_rs* is its per-instance output-referred contribution
    from ``Rs`` alone, used for the independent source-noise comparison.
    The band noise figure uses the gain from the paired total spectra.
    """
    f_lo, f_hi = SIG_BAND
    v_in = integrated_noise(noise_in, f_lo, f_hi)
    v_out = integrated_noise(noise_out, f_lo, f_hi)
    v_rs_out = integrated_noise(noise_rs, f_lo, f_hi)

    # Compare constant-gain input referral with the simulator spectrum.
    v_in_from_out = input_referred_noise(v_out, a0_vv)
    v_rs_in = input_referred_noise(v_rs_out, a0_vv)

    # Use the simulation temperature for the source-resistor noise reference.
    temp_k = TEMP_C + 273.15
    nf_band = band_noise_figure(noise_in, noise_out, f_lo, f_hi,
                               RS_OHM, temp_k, temp_ref_k=temp_k)

    # Compare the band noise figure with the difference between input and output SNR.
    snr_in_db = 20.0 * math.log10(V_SIG_RMS / v_rs_in)
    snr_out_db = 20.0 * math.log10(V_SIG_RMS / v_in)
    nf_from_snr = noise_figure(snr_in_db, snr_out_db)

    # Assume equal impedances when converting voltage gain to power gain for the cascade.
    gain_db = 20.0 * math.log10(a0_vv)
    nf_cascade = cascade_noise_figure([nf_band] * 2, [gain_db] * 2)

    corner_hz = one_over_f_corner(noise_in)
    v_flicker = integrated_noise(noise_in, FULL_BAND[0], corner_hz)
    v_white = integrated_noise(noise_in, corner_hz, FULL_BAND[1])
    v_total_full = integrated_noise(noise_in, *FULL_BAND)

    return {
        'v_in': v_in, 'v_out': v_out, 'v_rs_out': v_rs_out,
        'v_in_from_out': v_in_from_out, 'v_rs_in': v_rs_in,
        'nf_band': nf_band,
        'snr_in_db': snr_in_db, 'snr_out_db': snr_out_db,
        'nf_from_snr': nf_from_snr, 'gain_db': gain_db,
        'nf_cascade': nf_cascade,
        'corner_hz': corner_hz, 'v_flicker': v_flicker, 'v_white': v_white,
        'v_sum': sum_uncorrelated(v_flicker, v_white),
        'v_total_full': v_total_full,
    }


def _row(label: str, value: float, unit: str, note: str = '') -> None:
    print(f'  {label:30s}{value:10.3f} {unit:4s} {note}'.rstrip())


def report_noise(n: dict) -> None:
    lo, hi = SIG_BAND
    v_rs_txt = math.sqrt(4.0 * K_B * (TEMP_C + 273.15) * RS_OHM * (hi - lo))
    print()
    print(f'--- .noise: device noise at VAMP, integrated over {lo:.0f} Hz .. '
          f'{hi / 1e6:.0f} MHz ' + '-' * 9)
    _row('total, input-referred', n['v_in'] * 1e6, 'uV',
         'integrated_noise(noise_in)')
    _row('total, at the drain', n['v_out'] * 1e6, 'uV',
         'integrated_noise(noise_out)')
    _row('... referred back by A0', n['v_in_from_out'] * 1e6, 'uV',
         'input_referred_noise(v_out, A0)')
    _row('Rs alone, at the drain', n['v_rs_out'] * 1e6, 'uV',
         'ngspice per-instance onoise')
    _row('... referred back by A0', n['v_rs_in'] * 1e6, 'uV',
         'input_referred_noise(v_rs, A0)')
    _row('... textbook sqrt(4kT.Rs.df)', v_rs_txt * 1e6, 'uV',
         f'closed form at {TEMP_C:.0f} C')
    print()
    _row('band_noise_figure', n['nf_band'], 'dB', 'reference = simulation temperature')
    _row(f'SNR_in  at {V_SIG_RMS * 1e3:.0f} mV rms', n['snr_in_db'], 'dB',
         'signal over the Rs noise alone')
    _row(f'SNR_out at {V_SIG_RMS * 1e3:.0f} mV rms', n['snr_out_db'], 'dB',
         'signal over the total')
    _row('noise_figure(SNR_in, SNR_out)', n['nf_from_snr'], 'dB',
         'noise figure from the SNR difference')
    _row('cascade_noise_figure, 2 stages', n['nf_cascade'], 'dB',
         f'Friis, two stages of {n["gain_db"]:.1f} dB')
    print()
    _row('one_over_f_corner', n['corner_hz'] / 1e6, 'MHz',
         'floor = median of the upper half')
    print()
    _row('flicker band 1 Hz..corner', n['v_flicker'] * 1e6, 'uV')
    _row('white band corner..1 GHz', n['v_white'] * 1e6, 'uV')
    _row('sum_uncorrelated of the two', n['v_sum'] * 1e6, 'uV')
    _row('integrated over 1 Hz..1 GHz', n['v_total_full'] * 1e6, 'uV',
         f'disjoint bands: rel '
         f'{abs(n["v_sum"] - n["v_total_full"]) / n["v_total_full"]:.1e}')


def main() -> int:
    out_dir = example_output_dir('cs_amp_ac_noise')
    print('=' * 86)
    print('Open-loop margins and the .noise family — one NMOS, one load '
          'resistor, sky130')
    print('=' * 86)
    print(f'  M1 nmos1p8 W/L = {W_UM:g}/{L_UM:g} µm   Rd = {RD_OHM / 1e3:.0f} kΩ'
          f'   Rs = {RS_OHM / 1e3:.0f} kΩ   VDD = {VDD_V} V   Vgs = {VGS_V} V')
    print(f'  three-pole load: {RL_OHM / 1e3:.0f} kΩ / {CL_F * 1e15:.0f} fF, '
          'three loaded RC sections')
    print(f'  .ac    dec {AC_POINTS}, {AC_F1:.0g} Hz .. {AC_F2:.0g} Hz     '
          f'at VOUT')
    print(f'  .noise dec {NOISE_POINTS}, {NOISE_F1:.0g} Hz .. {NOISE_F2:.0g} Hz'
          f'     at VAMP, referred to vin')
    print(f'  decks and rawfiles: {out_dir}')

    mod = build_cs_amp(out_dir)
    results = mod.run_all_tests()
    ac_report = results['cs_amp.ac']
    noise_report = results['cs_amp.noise']

    sig = ac_report.sim_results.load_signal('vout')
    a0_vv = float(abs(loop_gain(sig).y[0]))
    spice = measure_ac(sig)
    # Fit the ideal transconductance to the measured low-frequency gain.
    ideal = measure_ac(ideal_response(sig.x, a0_vv / RD_OHM))
    report_ac(spice, ideal, gain_margins(loop_gain(sig)))

    ckt = noise_report.sim_results
    noise_in = ckt.load_signal('noise_in')
    noise_out = ckt.load_signal('noise_out')
    # The source-resistor noise vector is output-referred; xdut names the DUT instance.
    noise_rs = ckt.load_signal('onoise_r.xdut.rrs')
    n = measure_noise(noise_in, noise_out, noise_rs, a0_vv)
    report_noise(n)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
