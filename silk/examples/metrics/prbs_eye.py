"""PRBS eye folding and baseband EVM through two sky130 inverter/RC lanes.

Two shifted PRBS sequences drive inverter/RC channels at three resistances.
Compare simulated traces with an RC convolution model. Print settled rail swing,
step time constants, eye openings, known-bit decision margins and EVM at
transmitter and recovered phases. EVM uses both nearest constellation points
and the transmitted bits as references.

Requires ngspice and sky130 models. Run from the repository root:
    .venv/bin/python -m silk.examples.metrics.prbs_eye
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.examples import example_output_dir
from silk.metrics import (average, window, convolve, crossings, evm_constellation,
                          error_norm, eye_diagram, peak_to_peak, rise_time, sample, crossing)
from silk.metrics.eye import EyeResult
from silk.signals.base import Signal
from silk.simulation import Transient

VDD = 1.8                       # V
UI = 200e-12                    # s   unit interval -> 5 Gb/s
TEDGE = 20e-12                  # s   PWL rise = fall at the inverter gate
DT = UI / 256                   # s, requested step and maxstep

PRBS_ORDER = 7                  # period: 127 bits
N_PERIODS = 2                   # number of PRBS periods
N_BITS = N_PERIODS * (2 ** PRBS_ORDER - 1)
Q_SHIFT = 63                    # bits, cyclic shift of the Q lane
N_LEAD = 12                     # UI at 0, then N_LEAD at 1, before the data

# The lead-in holds each rail for twelve time constants of the slowest channel.
T_DATA = 2 * N_LEAD * UI

W_N, W_P, L_CH = 4.0, 8.0, 0.15     # µm — PDK units (sky130 W/L in µm)
C_CH = 100e-15                      # F   the receiver end of the channel

# Vary only channel resistance to compare RC time constants.
R_CHANNELS = (200.0, 1200.0, 2000.0)        # Ω -> τ = 0.10, 0.60, 1.00 UI

MIN_TRACES = 200                # minimum number of eye windows
MIN_CROSSINGS = 40              # rising mid-rail crossings in the data window
MIN_SWING = 1.70                # V, minimum lead-in swing


class Fold(NamedTuple):
    """One fold phase, and everything measured at it."""
    label: str
    origin: float           # s, relative to the start of the data window
    why: str
    eye: EyeResult          # on the simulated trace
    model: EyeResult        # on the closed-form channel model, same fold
    t_start: float          # s, absolute — window() snaps to the sample grid
    n_traces: int
    centre_samples: np.ndarray  # V — original waveform at the fold centres
    bit_aware: float        # V — min(ones) − max(zeros) at the same samples
    model_bit_aware: float  # V — the same margin on the RC model's folds


class EvmRow(NamedTuple):
    """EVM against nearest points and known bits, with decision checks."""
    label: str
    evm_percent: float      # what the metric returns
    evm_true: float         # EVM against transmitted symbols
    n_wrong: int            # decisions the nearest-point snap got wrong
    n_decisions: int
    identity_dv: float      # V — |eye centre samples − evm.received.real|


def prbs_lanes() -> tuple[list[int], list[int], float]:
    """Return I/Q symbols (lead-in plus PRBS data) and lane correlation."""
    bits = list(sources.prbs_bits(PRBS_ORDER, N_BITS))
    shifted = bits[Q_SHIFT:] + bits[:Q_SHIFT]
    a = 2 * np.asarray(bits, dtype=float) - 1.0
    b = 2 * np.asarray(shifted, dtype=float) - 1.0
    rho = float(np.dot(a, b) / len(a))
    lead = [0] * N_LEAD + [1] * N_LEAD
    return lead + bits, lead + shifted, rho


def pwl_points(symbols: list[int]) -> list[tuple[float, float]]:
    """Complementary NRZ knots, with each edge centred on its symbol boundary."""
    drive = [1 - s for s in symbols]        # the inverter output is the symbol
    pts = [(0.0, VDD * drive[0])]
    for k in range(1, len(drive)):
        if drive[k] != drive[k - 1]:
            t = k * UI
            pts.append((t - TEDGE / 2, VDD * drive[k - 1]))
            pts.append((t + TEDGE / 2, VDD * drive[k]))
    pts.append((len(drive) * UI, VDD * drive[-1]))
    return pts


def build_link(out_dir: str, r_ch: float,
               sym_i: list[int], sym_q: list[int]) -> Module:
    """Two inverter/RC lanes sharing one simulation time axis for I/Q EVM."""
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, 'prbs_eye.cir', pdk_details=pdk)

    mod.add_source(sources.VoltageSource('sup', positive='vdd', negative='0',
                                         dc=VDD))
    mod.add_source(sources.VoltageSource('ini', positive='vin_i', negative='0',
                                         dc=0.0,
                                         pwl=sources.PWL(pwl_points(sym_i))))
    mod.add_source(sources.VoltageSource('inq', positive='vin_q', negative='0',
                                         dc=0.0,
                                         pwl=sources.PWL(pwl_points(sym_q))))
    with mod:
        for lane in ('i', 'q'):
            pdk.nmos1p8(f'mn_{lane}', d=f'vtx_{lane}', g=f'vin_{lane}',
                        s='0', b='0', w=W_N, l=L_CH)
            pdk.pmos1p8(f'mp_{lane}', d=f'vtx_{lane}', g=f'vin_{lane}',
                        s='vdd', b='vdd', w=W_P, l=L_CH)
            elements.resistor(f'rch_{lane}', p=f'vtx_{lane}', n=f'vrx_{lane}', value=r_ch)
            elements.capacitor(f'cch_{lane}', p=f'vrx_{lane}', n='0', value=C_CH)

    # Limit integration gaps to resolve gate edges at decision instants.
    stop = len(sym_i) * UI
    mod.add_sim_type(Transient(step=DT, stop=stop, maxstep=DT))
    return mod


def simulate(out_dir: str, r_ch: float,
             sym_i: list[int], sym_q: list[int]) -> dict:
    """One transient per channel setting — one analysis, four signals."""
    mod = build_link(out_dir, r_ch, sym_i, sym_q)
    mod.simulate()
    return {
        'vtx_i': mod.load_signal('vtx_i'),   # the launch: what the channel sees
        'vrx_i': mod.load_signal('vrx_i'),   # the eye, and the I baseband
        'vtx_q': mod.load_signal('vtx_q'),
        'vrx_q': mod.load_signal('vrx_q'),   # the Q baseband
    }


def measure_levels(rx: Signal) -> tuple[float, float, float, float]:
    """Average the settled lead-in rails; reject insufficient swing."""
    lo = average(rx, start=(N_LEAD - 2) * UI, end=(N_LEAD - 0.05) * UI)
    hi = average(rx, start=(2 * N_LEAD - 2) * UI,
                 end=(2 * N_LEAD - 0.05) * UI)
    swing = hi - lo
    if swing < MIN_SWING:
        raise RuntimeError(
            f'lead-in swing {swing:.4f} V < {MIN_SWING} V; '
            f'increase the {N_LEAD} UI rail holds before measuring levels'
        )
    return lo, hi, swing, lo + swing / 2


def measure_tau(rx: Signal, lo: float, swing: float) -> tuple[float, float, float]:
    """Estimate RC tau from t90-t50, t50-t10 and rise time of the lead-in."""
    step = window(rx, (N_LEAD - 0.5) * UI, 2 * N_LEAD * UI)
    t10 = crossing(step, lo + 0.1 * swing, edge='rising', n=1)
    t50 = crossing(step, lo + 0.5 * swing, edge='rising', n=1)
    t90 = crossing(step, lo + 0.9 * swing, edge='rising', n=1)
    return ((t90 - t50) / math.log(5.0),
            (t50 - t10) / math.log(1.8),
            rise_time(step) / math.log(9.0))


def channel_model(vtx: Signal, tau: float) -> Signal:
    """Convolve the measured launch with an RC kernel of unit sampled area."""
    grid = np.arange(0.0, float(vtx.x[-1]), DT)
    launch = sample(vtx, x_new=grid)
    k_x = np.arange(0.0, 18.0 * tau, DT)          # e^-18 ≈ 1.5e-8 of truncation
    k_y = np.exp(-k_x / tau)
    k_y = k_y / (k_y.sum() * DT)
    kernel = Signal(f'h_rc(tau={tau:.3e})', 'V', 'time', '1/s', k_x, k_y)
    return window(convolve(launch, kernel), 0.0, float(vtx.x[-1]))


def recover_phase(rx: Signal, mid: float,
                  stop: float) -> tuple[float, float, int]:
    """Circular mean of rising-crossing phases modulo the UI.

    Concentration near zero means the crossings do not define a useful phase."""
    data = window(rx, T_DATA, stop)
    cs = np.asarray(crossings(data, threshold=mid, edge='rising'), dtype=float)
    if len(cs) < MIN_CROSSINGS:
        raise RuntimeError(
            f'{len(cs)} rising mid-rail crossings in the data window, '
            f'need >= {MIN_CROSSINGS} to estimate phase; increase the record length'
        )
    phases = (cs - float(data.x[0])) % UI
    z = np.mean(np.exp(2j * np.pi * phases / UI))
    return float((np.angle(z) / (2 * np.pi) * UI) % UI), float(abs(z)), len(cs)


def fold_at(rx: Signal, model: Signal, origin: float, stop: float,
            label: str, why: str, symbols: list[int]) -> Fold:
    """Fold the measured trace and RC model and sample known-bit margins.

    window retains recorded samples. Use its first sample as the fold origin
    and interpolate the original trace at each decision centre."""
    seg = window(rx, T_DATA + origin, stop)
    n_traces = int((float(seg.x[-1]) - float(seg.x[0])) / UI)
    if n_traces < MIN_TRACES:
        raise RuntimeError(
            f'{label} fold gets {n_traces} UI, need >= {MIN_TRACES}: '
            f'{N_PERIODS} PRBS{PRBS_ORDER} periods should give '
            f'{N_BITS - 1}. The record or the clip window is wrong.'
        )
    eye = eye_diagram(seg, period=UI, n_traces=n_traces)

    seg_m = window(model, T_DATA + origin, stop)
    n_model = int((float(seg_m.x[-1]) - float(seg_m.x[0])) / UI)
    model_eye = eye_diagram(seg_m, period=UI, n_traces=min(n_traces, n_model))

    centre = np.asarray(sample(rx, x_new=float(seg.x[0])
                               + (np.arange(len(eye.traces)) + 0.5) * UI).y)
    model_centre = np.asarray(sample(model, x_new=float(seg_m.x[0])
                                     + (np.arange(len(model_eye.traces)) + 0.5) * UI).y)
    n_traces = len(eye.traces)
    bits = np.asarray(symbols[2 * N_LEAD:2 * N_LEAD + n_traces])
    model_bits = bits[:len(model_eye.traces)]
    bit_aware = float(centre[bits == 1].min() - centre[bits == 0].max())
    model_bit_aware = float(model_centre[model_bits == 1].min()
                           - model_centre[model_bits == 0].max())

    return Fold(label, origin, why, eye, model_eye, float(seg.x[0]),
                n_traces, centre, bit_aware, model_bit_aware)


def measure_evm(rx_i: Signal, rx_q: Signal, mid: float, swing: float,
                fold: Fold, sym_i: list[int], sym_q: list[int]) -> EvmRow:
    """Measure EVM against nearest points and known bits at the fold centres.

    Subtract mid-rail and use the settled swing for constellation amplitudes.
    identity_dv compares eye-centre samples with the EVM input samples."""
    i_bb, q_bb = rx_i - mid, rx_q - mid
    half = swing / 2
    ideals = [(1 + 1j) * half, (1 - 1j) * half,
              (-1 + 1j) * half, (-1 - 1j) * half]
    res = evm_constellation(i_bb, q_bb, symbol_period=UI, ideal_points=ideals,
                            tstart=fold.t_start, n_symbols=fold.n_traces)

    n = fold.n_traces
    want_i = np.asarray([2 * sym_i[2 * N_LEAD + k] - 1
                         for k in range(n)], dtype=float) * half
    want_q = np.asarray([2 * sym_q[2 * N_LEAD + k] - 1
                         for k in range(n)], dtype=float) * half
    known = evm_constellation(
        i_bb, q_bb, symbol_period=UI, ideal_points=ideals,
        tstart=fold.t_start, n_symbols=n, reference=want_i + 1j * want_q)
    n_wrong = int(np.sum(np.sign(res.assigned_ideals.real) != np.sign(want_i))
                  + np.sum(np.sign(res.assigned_ideals.imag) != np.sign(want_q)))

    centre = fold.centre_samples - mid
    identity = float(np.max(np.abs(centre - res.received.real)))
    return EvmRow(fold.label, res.evm_percent, known.evm_percent,
                  n_wrong, 2 * n, identity)


PS, MV = 1e12, 1e3


def report_channel(r_ch: float, tau: float, swing: float, taus: tuple,
                   phi: float, conc: float, n_cross: int, p2p: float,
                   resid: tuple[float, float]) -> None:
    print()
    print(f'--- R_ch = {r_ch:.0f} Ω, C_ch = {C_CH * 1e15:.0f} fF   '
          f'τ = {tau * PS:.0f} ps = {tau / UI:.2f} UI ' + '-' * 40)
    print(f'  settled rails: swing {swing * MV:.1f} mV; peak_to_peak over the '
          f'data window {p2p * MV:.1f} mV, {p2p / swing - 1:+.1%} relative to settled swing')
    print(f'  τ from the lead-in step, three ways: (t90−t50)/ln5 = '
          f'{taus[0] * PS:6.1f} ps   (t50−t10)/ln1.8 = {taus[1] * PS:6.1f} ps'
          f'   t_r/ln9 = {taus[2] * PS:6.1f} ps')
    spread = (max(taus) - min(taus)) / min(taus)
    print(f'     spread {spread:.1%} against a channel τ of {tau * PS:.0f} ps: '
          + ('step estimates agree within 10%'
             if spread < 0.10 else
             'step estimates differ by at least 10%; driver dynamics can affect the edge'))
    print(f'  channel model (convolve of the measured launch, no fitted '
          f'parameter): residual over the data window '
          f'{resid[0] * MV:.3f} mV rms, {resid[1] * MV:.3f} mV max')
    print(f'  recovered symbol phase {phi * PS:6.1f} ps of {UI * PS:.0f} ps '
          f'from {n_cross} rising crossings, concentration {conc:.3f}'
          + ('' if conc > 0.30 else
             '  <- below 0.30: crossing phases are dispersed'))


def report_folds(folds: list[Fold]) -> None:
    print(f'  {"fold phase":14s} {"windows":>8s} {"model windows":>14s} '
          f'{"bit-aware (mV)":>15s} {"model (mV)":>12s}')
    for f in folds:
        print(f'  {f.label:14s} {len(f.eye.traces):8d} '
              f'{len(f.model.traces):14d} {f.bit_aware * MV:15.2f} '
              f'{f.model_bit_aware * MV:12.2f}')


def report_openings(measured: EyeResult, model: EyeResult) -> None:
    print('  Auto-centred geometric openings (each waveform aligns independently):')
    for label, eye in (('measured', measured), ('RC model', model)):
        if not (np.isfinite(eye.height) and np.isfinite(eye.width)):
            print(f'    {label}: geometric measurements unavailable; '
                  'see eye_diagram diagnostics')
            continue
        print(f'    {label}: min height {eye.height * MV:.2f} mV, '
              f'min width {eye.width * PS:.2f} ps')
        print(f'      {"opening":>7s} {"level (mV)":>12s} '
              f'{"height (mV)":>12s} {"width (ps)":>12s}')
        for k, (level, height, width) in enumerate(
                zip(eye.levels, eye.heights, eye.widths), start=1):
            print(f'      {k:7d} {level * MV:12.2f} '
                  f'{height * MV:12.2f} {width * PS:12.2f}')
    print('    Level is the threshold the opening is measured at (mid-swing), '
          'not a symbol level. Minima do not certify that all expected eyes '
          'exist or are open.')


def report_evm(rows: list[EvmRow]) -> None:
    print(f'  {"EVM at":14s} {"metric":>9s} {"vs known bits":>15s} '
          f'{"wrong decisions":>17s}   decision check')
    for r in rows:
        agree = 'all sampled bits match' if r.n_wrong == 0 else (
            f'nearest-point EVM is {r.evm_true - r.evm_percent:.1f} percentage points lower')
        print(f'  {r.label:14s} {r.evm_percent:8.3f}% {r.evm_true:14.3f}% '
              f'{r.n_wrong:8d}/{r.n_decisions:<8d} {agree}')


def main() -> int:
    sym_i, sym_q, rho = prbs_lanes()
    stop = len(sym_i) * UI

    print('=' * 112)
    print('eye_diagram and evm_constellation on a PRBS through a lossy RC '
          'channel (sky130)')
    print('=' * 112)
    print(f'  PRBS{PRBS_ORDER}, {N_PERIODS} periods = {N_BITS} bits at '
          f'{1 / UI / 1e9:.1f} Gb/s (UI = {UI * PS:.0f} ps), '
          f'{TEDGE * PS:.0f} ps gate edges, {N_LEAD} + {N_LEAD} UI of lead-in')
    print(f'  I and Q are the same sequence {Q_SHIFT} bits apart; measured '
          f'cross-correlation {rho:+.5f} against the −1/{2 ** PRBS_ORDER - 1} '
          f'= {-1 / (2 ** PRBS_ORDER - 1):+.5f} expected for a nonzero cyclic shift')
    print(f'  driver: MN W/L = {W_N}/{L_CH} µm, MP W/L = {W_P}/{L_CH} µm, '
          f'VDD = {VDD} V   channel: R_ch into C_ch = {C_CH * 1e15:.0f} fF')
    print(f'  .tran: step = maxstep = {DT * PS:.3f} ps, stop = '
          f'{stop * 1e9:.1f} ns; eye_diagram preserves original samples')
    print('  Auto-centering frames the eye; geometric openings and known-bit '
          'decision margins are distinct measurements.')

    summary = []
    for r_ch in R_CHANNELS:
        out_dir = example_output_dir('metrics', 'prbs_eye', f'r_{r_ch:g}')
        sig = simulate(out_dir, r_ch, sym_i, sym_q)
        rx, tx, rx_q = sig['vrx_i'], sig['vtx_i'], sig['vrx_q']
        tau = r_ch * C_CH

        lo, hi, swing, mid = measure_levels(rx)
        taus = measure_tau(rx, lo, swing)
        model = channel_model(tx, tau)
        phi, conc, n_cross = recover_phase(rx, mid, stop)

        resid = (error_norm(model, rx, kind='rms', start=T_DATA, end=stop - DT),
                 error_norm(model, rx, kind='max', start=T_DATA, end=stop - DT))
        p2p = peak_to_peak(rx, start=T_DATA, end=stop)

        report_channel(r_ch, tau, swing, taus, phi, conc, n_cross, p2p, resid)

        folds = [
            fold_at(rx, model, 0.0, stop, 'mid-slot',
                    'transmitter symbol boundary; decision samples at '
                    'the middle of the transmitted bit', sym_i),
            fold_at(rx, model, phi, stop, 'recovered',
                    'mean rising-crossing phase; decision samples half '
                    'a UI later', sym_i),
        ]
        report_folds(folds)

        mid_slot, recovered = folds
        eye = eye_diagram(window(rx, T_DATA, stop), period=UI, auto_center=True, threshold=mid)
        model_eye = eye_diagram(window(model, T_DATA, stop), period=UI, auto_center=True, threshold=mid)
        report_openings(eye, model_eye)
        closed_form = swing * (1 - 2 * math.exp(-UI / (2 * tau)))
        print(f'    closed form at the mid-slot phase, A·(1−2e^(−T/2τ)) = '
              f'{closed_form * MV:+8.2f} mV against a bit-aware '
              f'{mid_slot.bit_aware * MV:+8.2f} mV: the channel alone, '
              f'with an ideal driver and instantaneous edges')
        evm_rows = [measure_evm(rx, rx_q, mid, swing, f, sym_i, sym_q)
                    for f in (recovered, mid_slot)]
        report_evm(evm_rows)

        summary.append((tau, eye, model_eye, evm_rows[0]))
        print(f'  decks and rawfiles: {out_dir}')

    print()
    print('--- auto-centred geometric minima and recovered-phase EVM ' + '-' * 45)
    have_openings = all(np.isfinite(v) for _, measured, model_eye, _ in summary
                        for eye in (measured, model_eye)
                        for v in (eye.height, eye.width))
    columns = (f'{"height (mV)":>12s} {"model (mV)":>12s} '
               f'{"width (ps)":>12s} {"model (ps)":>12s} ' if have_openings else '')
    print(f'  {"τ/UI":>6s} {"windows":>8s} {columns}{"EVM":>8s}')
    for tau, eye, model_eye, evm in summary:
        values = (f'{eye.height * MV:12.2f} {model_eye.height * MV:12.2f} '
                  f'{eye.width * PS:12.2f} {model_eye.width * PS:12.2f} '
                  if have_openings else '')
        print(f'  {tau / UI:6.2f} {len(eye.traces):8d} '
              f'{values}{evm.evm_percent:7.2f}%')
    if not have_openings:
        print('  Unavailable geometric columns omitted; see per-channel diagnostics.')
    print('  Bit-aware margins use known bits; geometric openings do not certify decisions.')
    print('=' * 112)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
