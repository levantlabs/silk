"""CMRR and PSRR of a resistively loaded NMOS differential pair.

Three AC benches apply differential input, common-mode input and supply
excitation at the same DC bias. The example prints rejection versus frequency
and compares it with circuit models. The CMRR model uses measured gain and
tail impedance with a fitted body-effect factor. The PSRR model uses the
specified drain resistance and load capacitance.

A matched differential output illustrates the metric's numerical limit.
Requires ngspice and sky130 models. Run with::

    python -m silk.examples.metrics.diff_pair_rejection
"""
from __future__ import annotations

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.circuit.circuit_elements import Port
from silk.elements import elements
from silk.metrics import cmrr_dB, gain_at_freq, psrr_db, value_at
from silk.signals.base import Signal
from silk.simulation import AC

VDD_V = 1.8
VCM_V = 1.1
VBIAS_V = 0.85
RD_OHM = 60e3
CL_F = 2e-12
W_PAIR, L_PAIR = 4.0, 0.5  # µm
W_TAIL, L_TAIL = 4.0, 2.0  # µm

# The explicit tail capacitor brings the tail-impedance roll-off into the sweep.
CT_F = 1e-12

# Reporting frequencies lie on decade points of the AC sweep.
AC_POINTS, AC_F1, AC_F2 = 50, 1e3, 1e9
REPORT_HZ = (1e3, 1e5, 1e6, 1e7, 1e8, 1e9)

TEMP_C = 27.0
CMRR_ROUTE_TOL = 1e-12


def build_pair(out_dir: str, pdk) -> Module:
    """The differential pair as a subcircuit, with no tests attached."""
    mod = Module('PDK_skywater130', out_dir, 'diff_pair.cir', pdk_details=pdk,
                 module_name='diff_pair')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VINP': Port.analog_input(), 'VINN': Port.analog_input(),
        'VBIAS': Port.analog_input(),
        'VOUTP': Port.analog_output(), 'VOUTN': Port.analog_output(),
        'VTAIL': Port.analog_output(),
    })
    with mod:
        pdk.nmos1p8('M1', d='VOUTN', g='VINP', s='VTAIL', b='VSS',
                    w=W_PAIR, l=L_PAIR)
        pdk.nmos1p8('M2', d='VOUTP', g='VINN', s='VTAIL', b='VSS',
                    w=W_PAIR, l=L_PAIR)
        pdk.nmos1p8('MT', d='VTAIL', g='VBIAS', s='VSS', b='VSS',
                    w=W_TAIL, l=L_TAIL)
        elements.resistor('rd1', p='VDD', n='VOUTN', value=RD_OHM)
        elements.resistor('rd2', p='VDD', n='VOUTP', value=RD_OHM)
        elements.capacitor('cl1', p='VOUTN', n='VSS', value=CL_F)
        elements.capacitor('cl2', p='VOUTP', n='VSS', value=CL_F)
        elements.capacitor('ct', p='VTAIL', n='VSS', value=CT_F)
    return mod


def build_two_instance_pair(out_dir: str, pdk) -> Module:
    """Build two pair instances for a shared CMRR testbench."""
    child = build_pair(out_dir, pdk)
    top = Module('PDK_skywater130', out_dir, 'cmrr_pair.cir', pdk_details=pdk,
                 module_name='cmrr_pair')
    top.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VBIAS': Port.analog_input(),
        'VIP': Port.analog_input(), 'VIN': Port.analog_input(),
        'VIC': Port.analog_input(),
        'VOD': Port.analog_output(), 'VOC': Port.analog_output(),
    })
    top.add_module(child, 'A', nets={
        'VDD': 'VDD', 'VSS': 'VSS', 'VBIAS': 'VBIAS',
        'VINP': 'VIP', 'VINN': 'VIN',
        'VOUTN': 'VOD', 'VOUTP': 'nad', 'VTAIL': 'tad'})
    top.add_module(child, 'B', nets={
        'VDD': 'VDD', 'VSS': 'VSS', 'VBIAS': 'VBIAS',
        'VINP': 'VIC', 'VINN': 'VIC',
        'VOUTN': 'VOC', 'VOUTP': 'nac', 'VTAIL': 'tac'})
    return top


def _attach_bench(mod: Module, name: str, *, ac_dm: float, ac_cm: float,
                  ac_supply: float):
    """Attach an AC bench with the requested differential, common-mode and supply drive."""
    tb = mod.create_test(name)
    tb.add_module_under_test({
        'VDD': 'vdd', 'VSS': '0', 'VINP': 'vip', 'VINN': 'vin_n',
        'VBIAS': 'vb', 'VOUTP': 'voutp', 'VOUTN': 'voutn', 'VTAIL': 'vtail'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                        dc=VDD_V, ac=ac_supply))
    tb.add_source(sources.VoltageSource('vb', positive='vb', negative='0',
                                        dc=VBIAS_V))
    tb.add_source(sources.VoltageSource('vcm', positive='vmid', negative='0',
                                        dc=VCM_V, ac=ac_cm))
    tb.add_source(sources.VoltageSource('vidp', positive='vip',
                                        negative='vmid', dc=0, ac=ac_dm / 2))
    tb.add_source(sources.VoltageSource('vidn', positive='vmid',
                                        negative='vin_n', dc=0, ac=ac_dm / 2))
    tb.add_sim_type(AC('dec', AC_POINTS, AC_F1, AC_F2))
    return tb


def attach_pair_benches(mod: Module) -> None:
    """The three single-instance benches: differential, common mode, supply."""
    _attach_bench(mod, 'dm', ac_dm=1.0, ac_cm=0.0, ac_supply=0.0)
    _attach_bench(mod, 'cm', ac_dm=0.0, ac_cm=1.0, ac_supply=0.0)
    psrr_tb = _attach_bench(mod, 'psrr', ac_dm=0.0, ac_cm=0.0, ac_supply=1.0)
    # Reduce the PSRR spectrum at 1 MHz to register a scalar metric.
    psrr_tb.add_metric(
        'psrr_1MHz_dB',
        lambda ckt: value_at(psrr_db(ckt.load_signal('vdd'),
                                     ckt.load_signal('voutn')), 1e6))
    # Register testbenches before propagating their temperature.
    mod.set_temperature(TEMP_C)


def attach_declarative_bench(top: Module) -> None:
    """Attach a two-instance AC bench and register CMRR metrics."""
    tb = top.create_test('cmrr')
    tb.add_module_under_test({
        'VDD': 'vdd', 'VSS': '0', 'VBIAS': 'vb',
        'VIP': 'vip', 'VIN': 'vin_n', 'VIC': 'vic',
        'VOD': 'voutd', 'VOC': 'voutc'})
    tb.add_source(sources.VoltageSource('vdd', positive='vdd', negative='0',
                                        dc=VDD_V))
    tb.add_source(sources.VoltageSource('vb', positive='vb', negative='0',
                                        dc=VBIAS_V))
    tb.add_source(sources.VoltageSource('vcm', positive='vmid', negative='0',
                                        dc=VCM_V))
    tb.add_source(sources.VoltageSource('vidp', positive='vip',
                                        negative='vmid', dc=0, ac=0.5))
    tb.add_source(sources.VoltageSource('vidn', positive='vmid',
                                        negative='vin_n', dc=0, ac=0.5))
    tb.add_source(sources.VoltageSource('vic', positive='vic', negative='0',
                                        dc=VCM_V, ac=1.0))
    tb.add_sim_type(AC('dec', AC_POINTS, AC_F1, AC_F2))
    for freq in REPORT_HZ:
        tb.add_metric(f'cmrr_{_fname(freq)}_dB', type='cmrr_dB',
                      signals={'vdiff': 'voutd', 'vcm': 'voutc'}, freq=freq)
    top.set_temperature(TEMP_C)


def _fname(freq: float) -> str:
    """A metric-name-safe spelling of a frequency: 1e6 -> '1MHz'."""
    for scale, suffix in ((1e9, 'GHz'), (1e6, 'MHz'), (1e3, 'kHz')):
        if freq >= scale:
            return f'{freq / scale:g}{suffix}'
    return f'{freq:g}Hz'


def load_impedance(freqs) -> np.ndarray:
    """Return Z_L(f) = Rd ∥ 1/(jωCL) from the specified component values."""
    f = np.asarray(freqs, dtype=float)
    return RD_OHM / (1.0 + 2j * np.pi * f * RD_OHM * CL_F)


def tail_impedance(v_tail: Signal, v_out_cm: Signal) -> np.ndarray:
    """``Z_tail(f)``, measured, from the common-mode bench alone."""
    z_l = load_impedance(v_tail.x)
    return -z_l * np.asarray(v_tail.y) / (2.0 * np.asarray(v_out_cm.y))


def transconductance(v_out_dm: Signal) -> float:
    """``gm`` from the differential gain, with ``ro`` folded in."""
    i = int(np.argmin(np.asarray(v_out_dm.x, dtype=float)))
    z_l = load_impedance(v_out_dm.x)[i]
    return float(abs(-2.0 * np.asarray(v_out_dm.y)[i] / z_l))


def body_effect_factor(cmrr_lf_db: float, gm_s: float,
                       z_tail_lf: complex) -> float:
    """Fit χ = gmb/gm from low-frequency CMRR, gm and tail impedance."""
    cmrr_lin = 10.0 ** (cmrr_lf_db / 20.0)
    return (2.0 * cmrr_lin - 1.0) / (2.0 * gm_s * abs(z_tail_lf)) - 1.0


def predicted_cmrr_db(gm_s: float, chi: float, z_tail) -> np.ndarray:
    """``CMRR(f) = |1 + 2·gm(1+χ)·Z_tail(f)| / 2``, in dB."""
    return 20.0 * np.log10(
        np.abs(1.0 + 2.0 * gm_s * (1.0 + chi) * np.asarray(z_tail)) / 2.0)


def predicted_psrr_db(freqs) -> np.ndarray:
    """``20·log₁₀|1 + jω·Rd·CL|`` from the specified component values."""
    f = np.asarray(freqs, dtype=float)
    return 20.0 * np.log10(np.abs(1.0 + 2j * np.pi * f * RD_OHM * CL_F))


def output_resistance(psrr_lf_db: float, cmrr_lf_db: float) -> float:
    """``ro`` of a pair device, backed out of the low-frequency PSRR."""
    excess = 10.0 ** (psrr_lf_db / 20.0) - 1.0
    d_factor = 2.0 * 10.0 ** (cmrr_lf_db / 20.0)
    return RD_OHM / (excess * d_factor)


def _hz(freq: float) -> str:
    for scale, suffix in ((1e9, 'GHz'), (1e6, 'MHz'), (1e3, 'kHz')):
        if freq >= scale:
            return f'{freq / scale:g} {suffix}'
    return f'{freq:g} Hz'


def report_cmrr(v_dm: Signal, v_cm: Signal, v_tail: Signal,
                gm_s: float, chi: float) -> dict:
    """Print measured CMRR and model comparisons at the reporting frequencies."""
    f = np.asarray(v_dm.x, dtype=float)
    z_tail = tail_impedance(v_tail, v_cm)
    naive = predicted_cmrr_db(gm_s, 0.0, z_tail)
    fitted = predicted_cmrr_db(gm_s, chi, z_tail)

    print()
    print('--- cmrr_dB: differential drive against common-mode drive '
          + '-' * 24)
    print(f'  {"freq":>8s} {"|A_dm|":>8s} {"|A_cm|":>8s} {"CMRR":>9s} '
          f'{"|Z_tail|":>10s} {"chi=0":>9s} {f"chi={chi:.3f}":>11s}')
    print(f'  {"":>8s} {"V/V":>8s} {"V/V":>8s} {"dB":>9s} {"kohm":>10s} '
          f'{"dB":>9s} {"dB":>11s}')
    out = {}
    for freq in REPORT_HZ:
        i = int(np.argmin(np.abs(f - freq)))
        measured = cmrr_dB(v_dm, v_cm, freq=freq)
        out[freq] = measured
        print(f'  {_hz(freq):>8s} {gain_at_freq(v_dm, freq):8.2f} '
              f'{gain_at_freq(v_cm, freq):8.2f} {measured:9.3f} '
              f'{abs(z_tail[i]) / 1e3:10.1f} {naive[i]:9.3f} {fitted[i]:11.3f}')
    return out


def report_psrr(psrr: Signal, psrr_diff: Signal, ro_ohm: float) -> None:
    """Print measured PSRR, the RC prediction and the differential numerical limit."""
    pred = predicted_psrr_db(np.asarray(REPORT_HZ, dtype=float))
    print()
    print('--- psrr_db: supply ripple against the single-ended output '
          + '-' * 23)
    print(f'  {"freq":>8s} {"measured":>10s} {"predicted":>11s} '
          f'{"diff":>8s}   note')
    notes = {
        1e3: 'low-frequency supply coupling',
        1e5: 'below the output RC pole',
        1e6: 'near the output RC pole',
        1e7: 'load capacitance reduces supply coupling',
        1e8: '',
        1e9: '',
    }
    for freq, p in zip(REPORT_HZ, pred):
        m = value_at(psrr, freq)
        print(f'  {_hz(freq):>8s} {m:10.3f} {p:11.3f} {m - p:+8.3f}   '
              f'{notes[freq]}'.rstrip())
    print(f'  inferred device ro = {ro_ohm / 1e3:.0f} kohm')
    y = np.asarray(psrr_diff.y, dtype=float)
    cap = 20.0 * np.log10(1.0 / np.finfo(float).eps)
    print(f'  matched differential output: {y.min():.2f} .. {y.max():.2f} dB; '
          f'numerical floor {cap:.2f} dB')


def main() -> int:
    out_dir = example_output_dir('diff_pair_rejection')
    print('=' * 86)
    print('CMRR and PSRR — a resistively loaded NMOS pair with a tail current '
          'source, sky130')
    print('=' * 86)
    print(f'  M1,M2 nmos1p8 W/L = {W_PAIR:g}/{L_PAIR:g} µm   '
          f'MT nmos1p8 W/L = {W_TAIL:g}/{L_TAIL:g} µm   '
          f'Rd = {RD_OHM / 1e3:.0f} kΩ')
    print(f'  VDD = {VDD_V} V   Vcm = {VCM_V} V   Vbias = {VBIAS_V} V   '
          f'CL = {CL_F * 1e12:.0f} pF/output   Ctail = {CT_F * 1e12:.0f} pF')
    print(f'  three benches, one .ac each: dec {AC_POINTS}, '
          f'{AC_F1:.0g} Hz .. {AC_F2:.0g} Hz')
    print(f'  decks and rawfiles: {out_dir}')

    pdk = load_pdk('sky130')

    pair = build_pair(out_dir, pdk)
    attach_pair_benches(pair)
    results = pair.run_all_tests()

    dm = results['diff_pair.dm'].sim_results
    cm = results['diff_pair.cm'].sim_results
    ps = results['diff_pair.psrr'].sim_results

    v_dm = dm.load_signal('voutn')
    v_cm = cm.load_signal('voutn')
    v_tail = cm.load_signal('vtail')

    gm_s = transconductance(v_dm)
    z_tail = tail_impedance(v_tail, v_cm)
    chi = body_effect_factor(cmrr_dB(v_dm, v_cm, freq=AC_F1), gm_s, z_tail[0])
    print()
    print(f'  extracted:  gm = {gm_s * 1e6:.1f} µS (from A_dm, ro folded in)   '
          f'Z_tail(DC) = {abs(z_tail[0]) / 1e3:.1f} kΩ (measured)')
    print(f'              chi = gmb/gm = {chi:.3f} (fitted body-effect factor)   '
          f'tail pole ≈ '
          f'{1.0 / (2 * np.pi * abs(z_tail[0]) * CT_F) / 1e3:.0f} kHz')

    direct = report_cmrr(v_dm, v_cm, v_tail, gm_s, chi)

    psrr = psrr_db(ps.load_signal('vdd'), ps.load_signal('voutn'))
    psrr_diff = psrr_db(ps.load_signal('vdd'),
                        ps.load_signal('voutn') - ps.load_signal('voutp'))
    ro_ohm = output_resistance(value_at(psrr, AC_F1), direct[AC_F1])
    report_psrr(psrr, psrr_diff, ro_ohm)
    registered = results['diff_pair.psrr'].metrics['psrr_1MHz_dB']
    print(f'  bench-registered metric psrr_1MHz_dB = {registered:.3f} dB')
    assert np.all(np.isfinite(list(direct.values()))), 'non-finite CMRR'
    assert np.all(np.isfinite(psrr.y)), 'non-finite PSRR'
    assert np.isfinite(registered), 'non-finite registered PSRR'
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
