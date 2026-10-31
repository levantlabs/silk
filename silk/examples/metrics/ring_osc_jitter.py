"""ring_osc_jitter.py — jitter and stability curves on a supply-modulated ring.

A NAND gate enables a three-stage sky130 ring oscillator. Two constant-supply
runs estimate the period sensitivity to supply voltage and the numerical
jitter floor. A third run applies deterministic multi-tone supply modulation.
Measure the clock after startup and compare its jitter with a first-order
supply model. Print scalar jitter, cycle frequency, time interval error (TIE)
and Allan deviation alongside model estimates.
The long record supplies the edges needed for jitter and stability curves.

Requires ngspice and sky130 models. Simulation files use temporary directories.
Run from the repository root:
    .venv/bin/python -m silk.examples.metrics.ring_osc_jitter
"""
from __future__ import annotations

import math
import tempfile
from typing import NamedTuple, Optional

import numpy as np

from silk import Module, load_pdk, sources
from silk.elements import elements
from silk.metrics import (allan_deviation, average, average_period, crossings,
                          cycle_frequency, cycle_to_cycle_jitter,
                          long_term_jitter, peak, peak_to_peak, period_jitter, stddev,
                          tie, window)
from silk.signals.base import Signal
from silk.simulation import Transient

VDD = 1.80                     # V   nominal supply
VDD_HI = 1.86                  # V   the second point of dT/dV (see below)
W_N, W_P, L_CH = 1.0, 2.0, 0.15   # µm — PDK units (sky130 W/L in µm)
C_STAGE = 20e-15               # F   per-stage load: the knob that sets f_osc

T_STEP = 1e-12                 # s, requested step; maxstep is twice this
T_EN = 1e-9                    # s   EN rises here; the ring is parked before it
T_SETTLE = 6e-9                # s, exclude startup before measuring jitter
STOP_SHORT = 25e-9             # s, constant-supply runs
STOP_LONG = 200e-9             # s, modulated-supply run

N_TONES = 12
F_TONE_LO, F_TONE_HI = 100e6, 600e6    # Hz, modulation band
V_NOISE_RMS = 20e-3                    # V RMS, ~1.1 % of the 1.8 V rail

#: Log-spaced frequencies distribute the modulation tones across octaves.
TONE_F = np.geomspace(F_TONE_LO, F_TONE_HI, N_TONES)

#: Set per-tone amplitude from the long-time RMS of distinct sinusoids.
TONE_A = V_NOISE_RMS / math.sqrt(N_TONES / 2.0)

#: Fixed golden-angle phases make the modulation reproducible.
TONE_PHI = np.array([(i * 2.399963229728653) % (2 * math.pi)
                     for i in range(N_TONES)])

MIN_TAU_POINTS = 5
MIN_CURVE_POINTS = 100


class Row(NamedTuple):
    """One printed line: measured, predicted, and the unit they share."""
    label: str
    measured: float
    predicted: Optional[float]   # None where no independent prediction exists
    scale: float
    unit: str
    note: str


def noise_expression() -> str:
    """The multi-tone supply noise as one behavioural-source expression."""
    return ' + '.join(
        f'{TONE_A:.6e}*sin({2 * math.pi * f:.6e}*time + {phi:.6f})'
        for f, phi in zip(TONE_F, TONE_PHI)
    )


def _inverter(pdk, tag: str, a: str, y: str) -> None:
    """One CMOS inverter, instantiated into whatever module is open."""
    pdk.pmos1p8(f'mp_{tag}', d=y, g=a, s='vdd', b='vdd', w=W_P, l=L_CH)
    pdk.nmos1p8(f'mn_{tag}', d=y, g=a, s='0', b='0', w=W_N, l=L_CH)


def build_ring(out_dir: str, *, vdd: float, noise: str, stop: float) -> Module:
    """Build a three-stage ring with DC supply and additive modulation."""
    pdk = load_pdk('sky130')
    mod = Module('PDK_skywater130', out_dir, 'ring_osc.cir', pdk_details=pdk)

    mod.add_source(sources.VoltageSource('sup', positive='vdd_dc',
                                         negative='0', dc=vdd))
    mod.add_source(sources.BehavioralVoltage('noise', positive='vdd',
                                             negative='vdd_dc',
                                             expression=noise))
    mod.add_source(sources.VoltageSource(
        'en', positive='en', negative='0', dc=0.0,
        pulse=sources.Pulse(v1=0.0, v2=vdd, td=T_EN, tr=100e-12, tf=100e-12,
                            pw=stop, per=2 * stop)))

    with mod:
        pdk.pmos1p8('mnand_pa', d='n0', g='n2', s='vdd', b='vdd',
                    w=W_P, l=L_CH)
        pdk.pmos1p8('mnand_pb', d='n0', g='en', s='vdd', b='vdd',
                    w=W_P, l=L_CH)
        pdk.nmos1p8('mnand_na', d='n0', g='n2', s='nx', b='0',
                    w=2 * W_N, l=L_CH)
        pdk.nmos1p8('mnand_nb', d='nx', g='en', s='0', b='0',
                    w=2 * W_N, l=L_CH)
        _inverter(pdk, 'inv1', 'n0', 'n1')
        _inverter(pdk, 'inv2', 'n1', 'n2')
        # Buffer the measurement output to isolate its load from the ring.
        _inverter(pdk, 'buf', 'n2', 'out')
        for i, net in enumerate(('n0', 'n1', 'n2', 'out')):
            elements.capacitor(f'c{i}', p=net, n='0', value=C_STAGE)

    mod.add_sim_type(Transient(step=T_STEP, stop=stop, maxstep=2 * T_STEP))
    return mod


def steady_clock(mod: Module) -> Signal:
    """``out`` with the start-up window dropped."""
    return _after_settle(mod.load_signal('out'))


def _after_settle(sig: Signal) -> Signal:
    """``sig`` from ``T_SETTLE`` to the end of the record (``window``)."""
    return window(sig, T_SETTLE, float(sig.x[-1]))


def simulate(*, vdd: float, noise: str, stop: float) -> dict:
    """Run one transient and return the clock, rising edges and rail statistics."""
    with tempfile.TemporaryDirectory() as out_dir:
        mod = build_ring(out_dir, vdd=vdd, noise=noise, stop=stop)
        mod.simulate()
        clk = steady_clock(mod)
        vdd_sig = mod.load_signal('vdd')
        rail = _after_settle(vdd_sig)
    swing = peak_to_peak(clk)
    if swing < 0.95 * vdd:
        raise RuntimeError(
            f'clock swing {swing:.3f} V < 95% of {vdd:.3f} V after '
            f'T_SETTLE={T_SETTLE * 1e9:.1f} ns; check startup and settling'
        )
    return {
        'clock': clk,
        'edges': crossings(clk, edge='rising'),
        'rail_rms': stddev(rail),
        'rail_mean': average(rail),
    }


class SupplyModel(NamedTuple):
    """The ring's supply sensitivity, measured rather than assumed."""
    t_nom: float       # s   mean period at VDD, clean
    dT_dV: float       # s/V
    per_tone_dT: np.ndarray   # s — amplitude of the period modulation, per tone

    @property
    def frac_sensitivity(self) -> float:
        """``-(1/T)·dT/dV`` — the fractional frequency pull, per volt."""
        return -self.dT_dV / self.t_nom


def build_supply_model(t_nom: float, t_hi: float) -> SupplyModel:
    """Turn the two clean periods into the per-tone period modulation."""
    dT_dV = (t_hi - t_nom) / (VDD_HI - VDD)
    per_tone = abs(dT_dV) * TONE_A * np.sinc(TONE_F * t_nom)
    return SupplyModel(t_nom=t_nom, dT_dV=dT_dV, per_tone_dT=per_tone)


def predict_period_jitter(model: SupplyModel) -> float:
    """σ_T for a sum of incommensurate sinusoids: RMS of their amplitudes."""
    return float(math.sqrt(np.sum(model.per_tone_dT ** 2) / 2.0))


def predict_long_term_jitter(model: SupplyModel) -> float:
    """Estimate RMS TIE from accumulated sinusoidal period modulation."""
    amps = model.per_tone_dT / (2 * math.pi * TONE_F * model.t_nom)
    return float(math.sqrt(np.sum(amps ** 2) / 2.0))


def predict_allan(model: SupplyModel, taus: np.ndarray) -> np.ndarray:
    """σ_y(τ) in closed form for sinusoidal frequency modulation."""
    eps = model.per_tone_dT / model.t_nom
    out = np.empty_like(taus)
    for i, tau in enumerate(taus):
        out[i] = math.sqrt(float(np.sum(
            eps ** 2 * np.sinc(TONE_F * tau) ** 2
            * np.sin(math.pi * TONE_F * tau) ** 2)))
    return out


def predict_cycle_frequency(model: SupplyModel, t_mid: np.ndarray) -> np.ndarray:
    """Estimate cycle frequency from the supply waveform and measured sensitivity."""
    weighted = TONE_A * np.sinc(TONE_F * model.t_nom)
    dv = np.array([float(np.sum(weighted * np.sin(2 * math.pi * TONE_F * t
                                                  + TONE_PHI)))
                   for t in t_mid])
    return 1.0 / (model.t_nom + model.dT_dV * dv)


def predict_tie(model: SupplyModel, edges: np.ndarray) -> np.ndarray:
    """The whole TIE curve, integrated from the known supply waveform."""
    w = 2 * math.pi * TONE_F
    t0 = float(edges[0])
    raw = np.array([float(np.sum(TONE_A * (np.cos(w * t0 + TONE_PHI)
                                           - np.cos(w * t + TONE_PHI)) / w))
                    for t in edges]) * (model.dT_dV / model.t_nom)
    k = np.arange(len(edges), dtype=float)
    slope, intercept = np.polyfit(k, raw, 1)
    return raw - (intercept + slope * k)


def _print_rows(title: str, rows: list[Row]) -> None:
    print(f'\n{title}')
    print(f"  {'quantity':<34} {'measured':>13} {'predicted':>13} "
          f"{'rel':>8}  note")
    print('  ' + '-' * 104)
    for r in rows:
        meas = f'{r.measured / r.scale:.4f} {r.unit}'
        if r.predicted is None:
            pred, rel = '—', '—'
        else:
            pred = f'{r.predicted / r.scale:.4f} {r.unit}'
            rel = ('—' if r.predicted == 0
                   else f'{(r.measured - r.predicted) / abs(r.predicted):+.1%}')
        print(f'  {r.label:<34} {meas:>13} {pred:>13} {rel:>8}  {r.note}')


def report_setup(clean: dict, hi: dict, noisy: dict,
                 model: SupplyModel) -> None:
    """Supply sensitivity and the record used for the measurements."""
    print(f'  clean supply {VDD:.2f} V: period {model.t_nom * 1e12:.3f} ps, '
          f'f_osc {1 / model.t_nom / 1e9:.4f} GHz')
    print(f'  clean supply {VDD_HI:.2f} V: period '
          f'{average_period(hi["clock"]) * 1e12:.3f} ps')
    print(f'  dT/dV = {model.dT_dV * 1e12:.3f} ps/V')
    print(f'  noisy rail: mean {noisy["rail_mean"]:.4f} V, '
          f'{noisy["rail_rms"] * 1e3:.3f} mV RMS; '
          f'{len(noisy["edges"])} rising edges')


def report_floor(clean_jitter: float, noisy_jitter: float,
                 t_nom: float) -> None:
    """Print constant-supply and modulated-supply jitter for comparison."""
    print(f'  period_jitter, clean supply {clean_jitter * 1e12:.5f} ps')
    print(f'  period_jitter, noisy supply {noisy_jitter * 1e12:.5f} ps')
    print(f'  noisy/clean jitter ratio {noisy_jitter / clean_jitter:.0f}x')


def report_scalars(rows: list[Row]) -> None:
    _print_rows('Scalar jitter: first-order supply-model estimates', rows)


def report_cycle_frequency(sig: Signal, pred: np.ndarray,
                          model: SupplyModel) -> None:
    print(f'\ncycle_frequency: {len(sig.y)} points; time (ns), measured / model (GHz)')
    indices = np.linspace(0, len(sig.y) - 1, 8).astype(int)
    for i in indices[[0, 3, 7]]:
        print(f'  {sig.x[i] * 1e9:10.3f} {sig.y[i] / 1e9:16.5f} {pred[i] / 1e9:17.5f}')
    # Frequency spread uses unweighted cycle samples.
    print(f'  sample std(f) = {np.std(sig.y, ddof=1) / 1e6:.3f} MHz')


def report_tie(sig: Signal, pred: np.ndarray, ltj: float) -> None:
    print(f'\nTIE: {len(sig.y)} points; time (ns), measured / model (ps)')
    indices = np.linspace(0, len(sig.y) - 1, 9).astype(int)
    for i in indices[[0, 4, 8]]:
        print(f'  {sig.x[i] * 1e9:10.2f} {sig.y[i] * 1e12:15.3f} {pred[i] * 1e12:16.3f}')
    print(f'  sample standard deviation = {ltj * 1e12:.4f} ps (long_term_jitter)')
    print(f'  peak |TIE| = {peak(sig) * 1e12:.3f} ps; {peak(sig) / ltj:.2f}x sample standard deviation')


def report_allan(sig: Signal, pred: np.ndarray, model: SupplyModel,
                 n_periods: int, c2c: float) -> None:
    print(f'\nAllan deviation: {len(sig.y)} points; tau (ns), measured / model')
    for i in range(len(sig.y)):
        print(f'  {sig.x[i] * 1e9:10.3f} {sig.y[i]:12.4e} {pred[i]:12.4e}')


def main() -> int:
    print(__doc__.split('\n')[0])

    clean = simulate(vdd=VDD, noise='0', stop=STOP_SHORT)
    hi = simulate(vdd=VDD_HI, noise='0', stop=STOP_SHORT)
    t_nom = average_period(clean['clock'])
    t_hi = average_period(hi['clock'])
    model = build_supply_model(t_nom, t_hi)

    noisy = simulate(vdd=VDD, noise=noise_expression(), stop=STOP_LONG)
    clk = noisy['clock']
    n_periods = len(noisy['edges']) - 1

    report_setup(clean, hi, noisy, model)

    t_meas = average_period(clk)
    pj = period_jitter(clk)
    c2c = cycle_to_cycle_jitter(clk)
    ltj = long_term_jitter(clk)
    report_floor(period_jitter(clean['clock']), pj, t_nom)

    report_scalars([
        Row('average_period', t_meas, t_nom, 1e-12, 'ps',
            'mean rising-to-rising period; predicted = the clean-supply '
            'period (zero-mean noise)'),
        Row('period_jitter', pj, predict_period_jitter(model), 1e-12, 'ps',
            'RMS period deviation; model = per-tone modulation'),
        Row('cycle_to_cycle_jitter', c2c, None, 1e-12, 'ps',
            'RMS difference between adjacent period deviations'),
        Row('long_term_jitter', ltj, predict_long_term_jitter(model),
            1e-12, 'ps',
            'sample standard deviation of fitted TIE'),
    ])
    freq = cycle_frequency(clk)
    if len(freq.y) < MIN_CURVE_POINTS:
        raise RuntimeError(
            f'cycle_frequency returned {len(freq.y)} points; the record is too '
            f'short to be a curve (see the record-length note in the module '
            f'docstring)')
    report_cycle_frequency(freq, predict_cycle_frequency(model, freq.x), model)

    tie_sig = tie(clk)
    if len(tie_sig.y) < MIN_CURVE_POINTS:
        raise RuntimeError(
            f'tie returned {len(tie_sig.y)} points; the record is too short '
            f'to be a curve')
    report_tie(tie_sig, predict_tie(model, np.asarray(tie_sig.x, dtype=float)),
               ltj)

    allan = allan_deviation(clk)
    if len(allan.y) < MIN_TAU_POINTS or not np.all(np.isfinite(allan.y)):
        raise RuntimeError(
            f'allan_deviation returned {len(allan.y)} τ points '
            f'({int(np.sum(~np.isfinite(allan.y)))} of them non-finite); the '
            f'record is too short for the default τ ladder')
    report_allan(allan, predict_allan(model, np.asarray(allan.x, dtype=float)),
                 model, n_periods, c2c)

    for name, value in (('average_period', t_meas), ('period_jitter', pj),
                        ('cycle_to_cycle_jitter', c2c),
                        ('long_term_jitter', ltj)):
        if not math.isfinite(value):
            raise RuntimeError(f'{name} returned {value}')
    print('\nAll seven metrics returned finite values.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
