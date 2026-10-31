"""Transient timing, levels and supply power on a sky130 CMOS inverter.

A pulse drives a loaded inverter for five periods. Two 0 V ammeters measure
supply and pull-down currents. Print timing, output statistics and supply
power alongside charge and switched-capacitance estimates. Current extrema
use the second period. A final comparison shows how propagation_delay pairs
input and output edges; t_pHL uses a rising input and falling output.

Requires ngspice and sky130 models. Run from the repository root:
    .venv/bin/python -m silk.examples.metrics.inverter_timing
"""
from __future__ import annotations

from typing import NamedTuple, Optional

from silk import Module, load_pdk, sources
from silk.examples import example_output_dir
from silk.elements import elements
from silk.metrics import (argmax_x, argmin_x, average, average_power,
                          average_period, average_duty,
                          dynamic_energy_per_cycle,
                          fall_time, integral, off_time, on_time, peak,
                          peak_to_peak, propagation_delay, pulse_width, rise_time,
                          rms, slew_rate, stddev, maximum, minimum,
                          crossing)
from silk.signals.base import Signal
from silk.simulation import Transient

VDD = 1.8                      # V
PER = 10e-9                    # s   input period
TEDGE = 100e-12                # s   input rise = fall
PW = PER / 2 - TEDGE           # s   so the input's HIGH time is exactly PER/2
N_CYCLES = 5
STOP = N_CYCLES * PER          # an exact whole number of periods (see above)
TSTEP = 5e-12                  # s   print step and maximum integration step

T_IN_RISE_MID = TEDGE / 2                    # 50 ps
T_IN_FALL_MID = TEDGE + PW + TEDGE / 2       # 5.05 ns
T_IN_FALL_END = TEDGE + PW + TEDGE           # 5.10 ns: input fully at 0

W_N, W_P, L_CH = 1.0, 2.0, 0.15      # µm — PDK units (sky130 W/L in µm)
C_LOAD = 50e-15                      # F

I_WIN = (PER, 2 * PER)


class Row(NamedTuple):
    """One printed line: what was measured, what was expected, and why."""
    label: str
    measured: float
    expected: Optional[float]   # None for measurements without a model estimate
    scale: float
    unit: str
    note: str


def build_inverter(out_dir: str) -> Module:
    """One CMOS inverter, a load cap, a pulse drive and two 0 V ammeters."""
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, 'inverter.cir', pdk_details=pdk)

    mod.add_source(sources.VoltageSource('sup', positive='vdd_src',
                                         negative='0', dc=VDD))
    mod.add_source(sources.VoltageSource('amm_p', positive='vdd_src',
                                         negative='vdd', dc=0.0))
    mod.add_source(sources.VoltageSource('amm_n', positive='nsrc',
                                         negative='0', dc=0.0))
    mod.add_source(sources.VoltageSource('in', positive='vin', negative='0',
                                         dc=0.0,
                                         pulse=sources.Pulse(
                                             v1=0.0, v2=VDD, td=0.0,
                                             tr=TEDGE, tf=TEDGE,
                                             pw=PW, per=PER)))
    with mod:
        pdk.nmos1p8('mn', d='vout', g='vin', s='nsrc', b='0', w=W_N, l=L_CH)
        pdk.pmos1p8('mp', d='vout', g='vin', s='vdd', b='vdd', w=W_P, l=L_CH)
        elements.capacitor('cl', p='vout', n='0', value=C_LOAD)

    # Limit integration gaps to resolve switching edges.
    mod.add_sim_type(Transient(step=TSTEP, stop=STOP, maxstep=TSTEP))
    return mod


def simulate(out_dir: str) -> dict:
    """Run the transient and hand back the five signals the metrics read."""
    mod = build_inverter(out_dir)
    mod.simulate()
    return {
        'vin': mod.load_signal('vin'),
        'vout': mod.load_signal('vout'),
        'vdd': mod.load_signal('vdd'),
        'i_sup': mod.load_signal('vamm_p', 'I'),    # everything VDD delivers
        'i_pd': mod.load_signal('vamm_n', 'I'),     # the NMOS pull-down leg
    }


def measure_timing(vin: Signal, vout: Signal, i_sup: Signal, i_pd: Signal,
                   c_eff: float) -> list[Row]:
    """Timing measurements and charge-balance edge estimates."""
    mid = VDD / 2
    swing = peak_to_peak(vout)
    lo = minimum(vout)
    v10, v90 = lo + 0.1 * swing, lo + 0.9 * swing
    t_r_pred = c_eff * 0.8 * swing / average(
        i_sup, start=crossing(vout, v10, edge='rising', n=2),
        end=crossing(vout, v90, edge='rising', n=2))
    t_f_pred = c_eff * 0.8 * swing / average(
        i_pd, start=crossing(vout, v90, edge='falling', n=2),
        end=crossing(vout, v10, edge='falling', n=2))
    return [
        Row('rise_time', rise_time(vout), t_r_pred, 1e-12, 'ps', 'charge / mean supply current'),
        Row('fall_time', fall_time(vout), t_f_pred, 1e-12, 'ps', 'charge / mean pull-down current'),
        Row('slew_rate', slew_rate(vout), None, 1e9, 'GV/s', '10–90 % rising slew'),
        Row('propagation_delay', propagation_delay(vin, vout, in_edge='rising',
                                                   out_edge='falling'),
            None, 1e-12, 'ps', 't_pHL: input rising, output falling'),
        Row('pulse_width', pulse_width(vout), None, 1e-9, 'ns', 'output HIGH width'),
        Row('average_duty', average_duty(vout), None, 1.0, '', 'output HIGH fraction'),
        Row('on_time', on_time(vout, mid), None, 1e-9, 'ns', 'above VDD/2 over the record'),
        Row('off_time', off_time(vout, mid), None, 1e-9, 'ns', 'below VDD/2 over the record'),
        Row('average_period', average_period(vout, threshold=mid, edge='rising'),
            PER, 1e-9, 'ns', 'rising-to-rising spacing'),
        Row('crossing', crossing(vout, mid, edge='falling', n=3), None,
            1e-9, 'ns', 'third falling crossing of VDD/2'),
    ]


def measure_levels(vout: Signal) -> list[Row]:
    """Levels and time-weighted waveform statistics."""
    return [
        Row('peak', peak(vout), VDD, 1.0, 'V', 'absolute peak; rail coupling can overshoot'),
        Row('peak_to_peak', peak_to_peak(vout), VDD, 1.0, 'V', 'full observed swing'),
        Row('average', average(vout), None, 1.0, 'V', 'time-weighted mean'),
        Row('rms', rms(vout), None, 1.0, 'V', 'root mean square'),
        Row('stddev', stddev(vout), None, 1.0, 'V', 'time-weighted standard deviation'),
        Row('integral', integral(vout), None, 1e-9, 'nV·s', 'area over the complete record'),
    ]


def measure_power(vdd: Signal, i_sup: Signal) -> list[Row]:
    """Windowed current extrema and supply power from the ammeter."""
    t0, t1 = I_WIN
    return [
        Row('maximum', maximum(i_sup, start=t0, end=t1), None, 1e-6, 'µA', 'peak charging current'),
        Row('minimum', minimum(i_sup, start=t0, end=t1), None, 1e-6, 'µA', 'charge coupled back into the rail'),
        Row('argmax_x', argmax_x(i_sup, start=t0, end=t1), None, 1e-9, 'ns', 'time of maximum current'),
        Row('argmin_x', argmin_x(i_sup, start=t0, end=t1), None, 1e-9, 'ns', 'time of minimum current'),
        Row('average_power', average_power(vdd, i_sup), C_LOAD * VDD ** 2 / PER,
            1e-6, 'µW', 'C_L VDD²/T excludes device parasitics'),
        Row('dynamic_energy_per_cycle', dynamic_energy_per_cycle(vdd, i_sup, PER),
            C_LOAD * VDD ** 2, 1e-15, 'fJ', 'C_L VDD² excludes device parasitics'),
    ]


def _report(title: str, rows: list[Row]) -> None:
    print()
    print(title)
    print(f'  {"metric":24s} {"measured":>14s} {"expected":>14s} {"rel":>9s}'
          '   note')
    for r in rows:
        meas = f'{r.measured / r.scale:10.4f} {r.unit:<3s}'
        if r.expected is None:
            exp, rel = f'{"—":>10s} {r.unit:<3s}', f'{"—":>8s}'
        else:
            exp = f'{r.expected / r.scale:10.4f} {r.unit:<3s}'
            denom = abs(r.expected) if r.expected else 1.0
            rel = f'{abs(r.measured - r.expected) / denom:8.2e}'
        print(f'  {r.label:24s} {meas} {exp} {rel}   {r.note}')


def main() -> int:
    out_dir = example_output_dir('inverter_timing')
    print('=' * 110)
    print('Transient timing and primitive metrics on one CMOS inverter '
          '(sky130)')
    print('=' * 110)
    print(f'  MN W/L = {W_N}/{L_CH} µm   MP W/L = {W_P}/{L_CH} µm   '
          f'C_L = {C_LOAD * 1e15:.0f} fF   VDD = {VDD} V')
    print(f'  input pulse: 0 -> {VDD} V, {TEDGE * 1e12:.0f} ps edges, '
          f'period {PER * 1e9:.0f} ns, duty 0.5000')
    print(f'  .tran: step = maxstep = {TSTEP * 1e12:.0f} ps, stop = '
          f'{STOP * 1e9:.0f} ns = {N_CYCLES} whole periods')
    print(f'  decks and rawfiles: {out_dir}')

    sig = simulate(out_dir)
    vin, vout = sig['vin'], sig['vout']
    vdd, i_sup, i_pd = sig['vdd'], sig['i_sup'], sig['i_pd']

    energy = dynamic_energy_per_cycle(vdd, i_sup, PER)
    c_eff = energy / VDD ** 2
    print(f'  C_eff from the energy metric: {c_eff * 1e15:.2f} fF '
          f'against a load C_L of {C_LOAD * 1e15:.0f} fF; '
          f'difference {(c_eff - C_LOAD) * 1e15:.2f} fF')

    _report('--- timing, on vin and vout ' + '-' * 82,
            measure_timing(vin, vout, i_sup, i_pd, c_eff))
    _report('--- levels and statistics, on vout ' + '-' * 75,
            measure_levels(vout))
    _report('--- supply current and power, on i(vamm_p) and v(vdd) ' + '-' * 56,
            measure_power(vdd, i_sup))

    print()
    # Pair a rising input with a falling output when measuring t_pHL.
    print('--- propagation_delay edge selection ' + '-' * 74)
    print(f"  propagation_delay(vin, vout, in_edge='any')   = "
          f"{propagation_delay(vin, vout, in_edge='any') * 1e12:8.2f} ps   <- t_pHL")
    print(f"  propagation_delay(vin, vout)  [in_edge='rising'] = "
          f"{propagation_delay(vin, vout) * 1e9:8.4f} ns   "
          f"<- includes half an input period")
    print(f"  propagation_delay(vin, vout, in_edge='rising',")
    print(f"                    out_edge='falling')         = "
          f"{propagation_delay(vin, vout, in_edge='rising', out_edge='falling') * 1e12:8.2f} ps   "
          f"<- t_pHL, named explicitly")
    print("  Name the output edge explicitly on an inverting stage.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
