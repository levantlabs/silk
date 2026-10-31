"""Three-bit flash ADC: code transitions, linearity and converter figures of merit.

Seven two-transistor comparators share a resistor ladder with one widened
step. A ramp testbench decodes the thermometer outputs and measures DNL, INL,
monotonicity and missing codes. A coherent-sine testbench measures ENOB and
supply power. Print the transitions, linearity results, measured and modeled
ENOB, and power and timing figures of merit. The static model uses the
measured transitions.

Requires ngspice and sky130 models. Run from the repository root:
    .venv/bin/python -m silk.examples.metrics.flash_adc_codes
"""
from __future__ import annotations

import math

import numpy as np

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.circuit.ports import Port
from silk.elements import elements
from silk.metrics import (average, average_power, aperture_jitter_snr_db, dnl, enob,
                          extract_code_transitions, inl, is_monotonic,
                          missing_codes, schreier_fom, walden_fom)
from silk.signals.assemble import codes_from_bits
from silk.signals.base import Signal
from silk.simulation import Transient

VDD = 1.8                   # V
W_UNIT, L_UNIT = 1.0, 0.5   # µm — one unit device, used for both inputs
I_TAIL = 10e-6              # A   ideal, so the device count stays at two
RD_OHM = 80e3               # Ω   one per output
CL_F = 20e-15               # F   one per output

# At balance each branch carries I_TAIL/2, giving the decoding threshold.
VOUT_BAL = VDD - I_TAIL * RD_OHM / 2            # 1.4 V

TAU_OUT = RD_OHM * CL_F                         # 1.6 ns

N_BITS = 3
N_CMP = 2 ** N_BITS - 1                         # 7 comparators, 8 codes
R_UNIT = 10e3               # Ω   one ladder step
R_BOT, R_TOP = 70e3, 30e3   # Ω   the two range-setting resistors
DELTA = 0.5                 # the ladder mismatch: r4 = (1 + δ)·R_UNIT
MISMATCHED_STEP = 4         # which ladder resistor carries it
LOST_CMP, LOST_TAP = 6, 5   # converter B: comparator 6 reads comparator 5's tap

I_LADDER_NOM = VDD / (R_BOT + 8 * R_UNIT + R_TOP)
LSB_NOM = I_LADDER_NOM * R_UNIT                 # 0.1 V
V_MID_NOM = I_LADDER_NOM * (R_BOT + 4 * R_UNIT)  # 1.1 V — nominal mid-scale

# Endpoint-normalized linearity is invariant under affine voltage scaling.
U_LSB = DELTA / (6.0 + DELTA)                   # 1/13
DNL_NORMAL = -U_LSB                             # −1/13
DNL_WIDE = 5.0 * U_LSB                          # +5/13
INL_SHAPE = np.array([0., -1., -2., 3., 2., 1., 0.]) * U_LSB
DNL_SHAPE_LOST = np.array([-1., -1., 5., -1., -13., 11.]) * U_LSB
INL_SHAPE_LOST = np.array([0., -1., -2., 3., 2., -11., 0.]) * U_LSB

V_RAMP_LO, V_RAMP_HI = 0.6, 1.6   # V, span all ladder thresholds
T_RAMP = 10e-6                    # s
RAMP_TSTEP = 1e-9                 # s   print step and maximum integration step
RAMP_SLOPE = (V_RAMP_HI - V_RAMP_LO) / T_RAMP          # 100 kV/s
RAMP_DV = RAMP_SLOPE * RAMP_TSTEP                      # 100 µV per sample
# Common output-pole lag cancels in endpoint-normalized linearity.
RAMP_LAG_V = TAU_OUT * RAMP_SLOPE                      # 160 µV

FS_HZ = 20e6
TCLK_S = 1.0 / FS_HZ
N_SAMPLES = 256
CYCLES = 7
F0_HZ = CYCLES * FS_HZ / N_SAMPLES             # 546 875.0 Hz, exact
SINE_TSTEP = 1e-9                              # s   step and maxstep
SINE_STOP = (N_SAMPLES + 8) * TCLK_S           # 8 tclk of margin past the record
# Set the sine amplitude from nominal LSBs, independently of measured transitions.
V_SINE_CEN = V_MID_NOM                         # 1.1 V
V_SINE_AMPL = 4.0 * LSB_NOM                    # 0.4 V

# Assume uniform half-step timing uncertainty for the aperture-SNR estimate.
SIGMA_APERTURE_S = SINE_TSTEP / math.sqrt(12.0)

TEMP_C = 27.0

# Allow for transition quantization when comparing linearity with the ladder model.
LINEARITY_TOL_LSB = 5e-3


def build_comparator(out_dir: str, pdk) -> Module:
    """Build a two-transistor comparator subcircuit."""
    mod = Module('PDK_skywater130', out_dir, 'comparator.cir', pdk_details=pdk,
                 module_name='comparator')
    mod.define_module_ports({
        'VDD': Port.supply(), 'VSS': Port.ground(),
        'VINP': Port.analog_input(), 'VREF': Port.analog_input(),
        'VOUT': Port.analog_output(), 'VOUTB': Port.analog_output(),
        'VTAIL': Port.analog_output(),
    })
    with mod:
        # VOUT rises as VINP crosses VREF, forming a thermometer output.
        pdk.nmos1p8('M1', d='VOUTB', g='VINP', s='VTAIL', b='VSS',
                    w=W_UNIT, l=L_UNIT)
        pdk.nmos1p8('M2', d='VOUT', g='VREF', s='VTAIL', b='VSS',
                    w=W_UNIT, l=L_UNIT)
        elements.resistor('rd1', p='VDD', n='VOUTB', value=RD_OHM)
        elements.resistor('rd2', p='VDD', n='VOUT', value=RD_OHM)
        elements.capacitor('cl1', p='VOUTB', n='VSS', value=CL_F)
        elements.capacitor('cl2', p='VOUT', n='VSS', value=CL_F)
    # A constant tail current sets VOUT_BAL independently of the input voltage.
    mod.add_source(sources.CurrentSource('it', positive='VTAIL',
                                         negative='VSS', dc=I_TAIL))
    return mod


def ladder_values() -> list:
    """Return ladder resistances from bottom to top, including the widened step."""
    return [R_UNIT * (1.0 + DELTA) if k == MISMATCHED_STEP else R_UNIT
            for k in range(1, 9)]


def build_flash(out_dir: str, pdk, comparator: Module, module_name: str,
                filename: str, *, misroute: bool) -> Module:
    """A 3-bit flash: one ladder, seven instances of one comparator."""
    mod = Module('PDK_skywater130', out_dir, filename, pdk_details=pdk,
                 module_name=module_name)
    ports = {'VDD': Port.supply(), 'VSS': Port.ground(),
             'VIN': Port.analog_input()}
    for k in range(1, N_CMP + 1):
        ports[f'D{k}'] = Port.analog_output()
    mod.define_module_ports(ports)

    with mod:
        elements.resistor('rbot', p='VSS', n='lad0', value=R_BOT)
        for k, value in enumerate(ladder_values(), start=1):
            elements.resistor(f'r{k}', p=f'lad{k - 1}', n=f'lad{k}', value=value)
        elements.resistor('rtop', p='lad8', n='VDD', value=R_TOP)

    for k in range(1, N_CMP + 1):
        tap = LOST_TAP if (misroute and k == LOST_CMP) else k
        # Unused comparator ports connect to separate internal nets.
        mod.add_module(comparator, f'X{k}', nets={
            'VDD': 'VDD', 'VSS': 'VSS', 'VINP': 'VIN',
            'VREF': f'lad{tap}', 'VOUT': f'D{k}',
            'VOUTB': f'b{k}', 'VTAIL': f't{k}'})
    return mod


def _dut_and_supply(tb) -> None:
    """The wiring both benches share: the DUT, the rail, and the ammeter."""
    tb.add_module_under_test({
        'VDD': 'vdd', 'VSS': '0', 'VIN': 'vin',
        **{f'D{k}': f'd{k}' for k in range(1, N_CMP + 1)}})
    tb.add_source(sources.VoltageSource('sup', positive='vsup', negative='0',
                                        dc=VDD))
    tb.add_source(sources.VoltageSource('amm', positive='vsup',
                                        negative='vdd', dc=0.0))


def attach_ramp_bench(flash: Module) -> None:
    """The slow ramp the whole static transfer curve is read from."""
    tb = flash.create_test('ramp')
    _dut_and_supply(tb)
    tb.add_source(sources.VoltageSource(
        'in', positive='vin', negative='0',
        pulse=sources.Pulse(v1=V_RAMP_LO, v2=V_RAMP_HI, td=0.0, tr=T_RAMP,
                            tf=T_RAMP, pw=2 * T_RAMP, per=10 * T_RAMP)))
    # Cap integration gaps; transition resolution follows local sample spacing.
    tb.add_sim_type(Transient(step=RAMP_TSTEP, stop=T_RAMP,
                              maxstep=RAMP_TSTEP))


def attach_sine_bench(flash: Module) -> None:
    """The coherent tone ENOB is read from."""
    tb = flash.create_test('sine')
    _dut_and_supply(tb)
    tb.add_source(sources.VoltageSource(
        'in', positive='vin', negative='0',
        sine=sources.Sine(voffset=V_SINE_CEN, vampl=V_SINE_AMPL, freq=F0_HZ)))
    tb.add_sim_type(Transient(step=SINE_TSTEP, stop=SINE_STOP,
                              maxstep=SINE_TSTEP))


def decode(ckt) -> Signal:
    """Seven comparator outputs → one decimal-code transient."""
    bits = [ckt.load_signal(f'd{k}') for k in range(1, N_CMP + 1)]
    return codes_from_bits(bits, weighting='thermometer',
                           threshold=VOUT_BAL, name='code')


def designed_taps() -> np.ndarray:
    """Tap voltages ``lad1 … lad7`` from the drawn resistors alone."""
    values = ladder_values()
    total = R_BOT + sum(values) + R_TOP
    current = VDD / total
    cumulative = R_BOT + np.cumsum(values)
    return current * cumulative[:N_CMP]


def designed_transitions(misroute: bool) -> np.ndarray:
    """What the ramp should read: the tap voltages, sorted, plus the lag."""
    taps = designed_taps()
    if misroute:
        taps = taps.copy()
        taps[LOST_CMP - 1] = taps[LOST_TAP - 1]
    return np.sort(taps) + RAMP_LAG_V


def measure_static(vin: Signal, code: Signal) -> dict:
    """Extract ramp transitions, then measure linearity and code coverage."""
    transitions = extract_code_transitions(vin, code)
    return {
        'transitions': transitions,
        'inl': inl(transitions),
        'dnl': dnl(transitions),
        'is_monotonic': is_monotonic(transitions),
        'missing_codes': missing_codes(transitions),
        'lsb_v': float((transitions.y[-1] - transitions.y[0])
                       / (len(transitions.y) - 1)),
        'codes_seen': sorted({int(c) for c in np.rint(code.y)}),
    }


def static_model_code(vin: Signal, transitions: Signal) -> Signal:
    """Apply the measured transition levels to the input waveform."""
    levels = np.asarray(transitions.y, dtype=float)
    codes = np.searchsorted(levels, np.asarray(vin.y, dtype=float),
                            side='right').astype(float)
    if levels.size:
        codes += float(transitions.x[0]) - 1
    return Signal('code_static_model', 'count', 'time', 'code',
                  np.asarray(vin.x, dtype=float), codes)


def measure_dynamic(ckt, transitions: Signal) -> dict:
    """Measure sine-record ENOB, supply power and converter figures of merit."""
    vin = ckt.load_signal('vin')
    code = decode(ckt)
    sampling = dict(tclk=TCLK_S, tstart=0.0, n_samples=N_SAMPLES)
    enob_bits = enob(code, **sampling)
    # Recover SNDR from ENOB using the metric's defining relation.
    sndr_db = 6.02 * enob_bits + 1.76
    power_w = average_power(ckt.load_signal('vdd'),
                            ckt.load_signal('vamm', 'I'))
    return {
        'enob': enob_bits,
        'enob_static_model': enob(static_model_code(vin, transitions),
                                  **sampling),
        'sndr_db': sndr_db,
        'power_w': power_w,
        'i_supply_a': average(ckt.load_signal('vamm', 'I')),
        'codes_seen': sorted({int(c) for c in np.rint(decode(ckt).y)}),
        'walden_fom': walden_fom(power_w, FS_HZ, enob_bits),
        'schreier_fom': schreier_fom(sndr_db, FS_HZ / 2.0, power_w),
        'aperture_snr_db': aperture_jitter_snr_db(F0_HZ, SIGMA_APERTURE_S),
    }


def main() -> int:
    out_dir = example_output_dir('flash_adc_codes')
    print('Flash ADC code metrics (sky130): decoding → transitions → linearity → ENOB')
    print(f'  decks and rawfiles: {out_dir}')
    pdk = load_pdk('sky130')
    comparator = build_comparator(out_dir, pdk)
    flash = build_flash(out_dir, pdk, comparator, 'flash3', 'flash3.cir',
                        misroute=False)
    attach_ramp_bench(flash)
    attach_sine_bench(flash)
    # Set temperature after registering benches so it propagates to them.
    flash.set_temperature(TEMP_C)
    results = flash.run_all_tests()
    ramp = results['flash3.ramp'].sim_results
    sine = results['flash3.sine'].sim_results

    static = measure_static(ramp.load_signal('vin'), decode(ramp))
    transitions = static['transitions']
    assert len(transitions.y) >= 2, 'not enough ADC transitions'
    assert np.all(np.isfinite(transitions.y)), 'non-finite ADC transitions'
    assert np.all(np.isfinite(static['dnl'].dnl_lsb)), 'non-finite DNL'
    assert np.all(np.isfinite(static['inl'].inl_lsb)), 'non-finite INL'
    print('  code   transition (V)')
    for code, level in zip(transitions.x, transitions.y):
        print(f'  {int(code):4d}   {level:.6f}')
    print(f"  dnl.max_abs_lsb  {static['dnl'].max_abs_lsb:.4f} LSB")
    print(f"  inl.max_abs_lsb  {static['inl'].max_abs_lsb:.4f} LSB")
    print(f"  is_monotonic    {static['is_monotonic']}")
    print(f"  missing_codes   {list(static['missing_codes'])}")
    print(f"  codes exercised by the ramp: {static['codes_seen']}")

    dynamic = measure_dynamic(sine, transitions)
    scalar_names = ('enob', 'enob_static_model', 'sndr_db', 'power_w',
                    'i_supply_a', 'walden_fom', 'schreier_fom', 'aperture_snr_db')
    assert all(math.isfinite(dynamic[key]) for key in scalar_names), (
        'non-finite dynamic ADC metric')
    print(f"  enob  measured {dynamic['enob']:.4f} bits   static model "
          f"{dynamic['enob_static_model']:.4f} bits")
    print(f"  sndr  = 6.02·ENOB + 1.76 = {dynamic['sndr_db']:.3f} dB")
    print(f"  average_power   {dynamic['power_w'] / 1e-6:.4f} µW")
    print(f"  walden_fom      {dynamic['walden_fom'] * 1e12:.3f} pJ/conv-step")
    print(f"  schreier_fom    {dynamic['schreier_fom']:.2f} dB")
    print(f"  aperture_jitter_snr_db   {dynamic['aperture_snr_db']:.2f} dB")
    print('  Aperture SNR uses a supplied uniform timing-uncertainty assumption; '
          'it is not a spectral-sampling ceiling.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
