"""REVIEW-METRICS (TASK-34) — edge-case matrix, metric × degenerate input.

Rows: every public Signal-taking metric. Columns: the degenerate inputs
the Signal contract must survive — ``empty`` (0 samples), ``single``
(1 sample), ``constant`` (flat trace), ``nan`` (one NaN sample in an
otherwise valid trace), ``wrongaxis`` (the other xlabel), plus a
``window`` column for metrics that take a band/window: the window is
placed entirely outside the data.

Each cell pins the OBSERVED behaviour class today:

``ValueError``   loud and typed (the contract)
``<Other>Error`` loud but untyped (a numpy accident). None remain:
                 TASK-77 gave every metric a typed, metric-naming error
                 on empty input; TASK-80 added the one metric-specific
                 minimum-length guard needed for ``derivative``.
``nan``/``inf``  honest "unknown" (acceptable when documented) — for a
                 Signal / array / report result, ``+nan`` means the
                 payload (a headline scalar field, for a report) is nan
``value`` / ``empty`` / ``Signal`` / ``report`` / ``array`` / ``bool`` /
``complex``      a SILENT answer — every such cell on a degenerate input
                 must be listed in ``BY_DESIGN`` (with the reason) or in
                 ``FILED`` (with the task that owns the fix — and that
                 task must still be OPEN, TASK-78);
                 ``test_every_silent_cell_is_accounted_for`` enforces it.

The table was generated from the metrics as shipped on 2026-09-02 and
is asserted verbatim: a behavioural fix changes a cell and must update
the row (and drop the FILED entry).
"""
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from silk.signals.base import Signal
import silk.metrics as M


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

T = 1e-9


def _sig(x, y, xl):
    return Signal('s', 'voltage', xl, 'V', np.asarray(x, dtype=float), np.asarray(y))


def _clock(xl='time'):
    t = np.linspace(0, 64 * T, 64 * 50 + 1)
    return _sig(t, 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / T + 1e-12)), xl)


def _sine(xl='time'):
    n = 1024
    return _sig(np.arange(n) * T, np.sin(2 * np.pi * 11 * np.arange(n) / n), xl)


def _ac(xl='frequency'):
    f = np.logspace(1, 7, 121)
    return _sig(f, 100.0 / (1 + 1j * f / 1e4), xl)


def _noise_spectrum(xl='frequency'):
    """A real V/sqrt(Hz) density on the same grid ``_ac`` uses.

    ADDED by TASK-197. The two noise metrics below (``integrated_noise``,
    ``one_over_f_corner``) were pinned against
    ``_ac``, whose ``y`` is COMPLEX — an ``.ac`` transfer function handed
    to a metric that declares a ``noise_density`` input. That is now a typed
    refusal (design §3.4), so with ``_ac`` the
    ``nan`` and ``window`` columns would stop measuring NaN propagation and
    out-of-band windows and start measuring the kind check instead — a cell
    reading ``ValueError`` for a reason that has nothing to do with its
    column. The refusal itself is pinned where it belongs
    (``test_noise.py::TestTask197CrossKindGuards``,
    ``test_rawfile_realism.py``, ``test_complex_y_coverage.py``); this
    fixture gives the rows back the input their metrics actually take, and
    every one of the four cells lands on the value it was pinned to in 2026-09.
    (A third metric, ``phase_noise_at_offset``, used to be pinned here too;
    TASK-210 retired it along with the ``phase_noise_sideband`` kind.)
    """
    f = np.logspace(1, 7, 121)
    return _sig(f, 1e-9 * np.sqrt(1.0 + 1e3 / f), xl)


def _codes(xl='code'):
    return _sig(np.arange(8), np.arange(8) * 0.125, xl)


def _psweep(xl='power'):
    """A compressing power sweep: 20 dB gain, falling 0.25 dB/dB past
    −5 dBm, so P1dB = −1 dBm (TASK-180)."""
    p_in = np.linspace(-30, 10, 41)
    return _sig(p_in, p_in + 20.0 - 0.25 * np.clip(p_in + 5.0, 0, None), xl)


def _with_nan(s):
    y = np.array(s.y, copy=True)
    y[len(y) // 5] = np.nan
    return Signal(s.name, s.type, s.xlabel, s.ylabel, s.x, y)


def _as_current(s):
    """The same trace, re-typed as a current — what the ``sig_i`` slot of
    the two power metrics takes.

    ``_sig`` builds every fixture as a ``'voltage'``, and since TASK-201
    ``average_power`` / ``dynamic_energy_per_cycle`` refuse a RECOGNISED
    voltage in ``sig_i``. Re-typing the copy keeps this matrix about the
    DEGENERATE INPUT it pins (empty / single / constant / nan / wrongaxis)
    rather than about the type guard, which has its own tests in
    ``tests/metrics/test_transient.py::TestPowerSlotTypes``. Every other
    observable — x, y, xlabel — is carried over untouched, so each cell
    below is the same measurement it was before the guard existed.
    """
    return Signal(s.name, 'current', s.xlabel, 'A', s.x, s.y)


def _other_axis(xl):
    return 'frequency' if xl == 'time' else 'time'


def _edge_inputs(base, xl):
    """The five degenerate variants of ``base(xl)``."""
    const_x = np.logspace(1, 7, 101) if xl == 'frequency' else np.arange(2048) * T
    return {
        'empty':     _sig([], [], xl),
        'single':    _sig([0.0 if xl != 'frequency' else 10.0], [1.0], xl),
        'constant':  _sig(const_x, np.ones(len(const_x)), xl),
        'nan':       _with_nan(base(xl)),
        'wrongaxis': base(_other_axis(xl)),
    }


# ---------------------------------------------------------------------------
# Registry: metric name -> (base builder, xlabel, call(sig), window-call or None)
# ---------------------------------------------------------------------------

K = dict(tclk=T, tstart=0.0, n_samples=1024)
FAR = 1e3          # a window / band far outside every fixture's x-range

REG = {
    'sfdr_db': (_sine, 'time', lambda s: M.sfdr_db(s, **K), None),
    'harmonic_level_dbc': (_sine, 'time', lambda s: M.harmonic_level_dbc(s, n=2, **K), None),
    'eye_height': (_clock, 'time', lambda s: M.eye_height(s, period=T, threshold=.5), None),
    'eye_width': (_clock, 'time', lambda s: M.eye_width(s, period=T, threshold=.5), None),
    'inl_max_lsb': (_codes, 'code', lambda s: M.inl_max_lsb(s), None),
    'dnl_max_lsb': (_codes, 'code', lambda s: M.dnl_max_lsb(s), None),
    'offset_error_lsb': (_codes, 'code', lambda s: M.offset_error_lsb(s, full_scale=1.0, n_bits=3), None),
    'gain_error_lsb': (_codes, 'code', lambda s: M.gain_error_lsb(s, full_scale=1.0, n_bits=3), None),
    'full_scale_error_lsb': (_codes, 'code', lambda s: M.full_scale_error_lsb(s, full_scale=1.0, n_bits=3), None),
    'missing_code_count': (_codes, 'code', lambda s: M.missing_code_count(s), None),
    'non_monotonic_steps': (_codes, 'code', lambda s: M.non_monotonic_steps(s), None),
    # transient ------------------------------------------------------
    'crossings':               (_clock, 'time', lambda s: M.crossings(s), None),
    'rise_time':               (_clock, 'time', lambda s: M.rise_time(s), None),
    'fall_time':               (_clock, 'time', lambda s: M.fall_time(s), None),
    'slew_rate':               (_clock, 'time', lambda s: M.slew_rate(s), None),
    'rise_times':              (_clock, 'time', lambda s: M.rise_times(s), None),
    'fall_times':              (_clock, 'time', lambda s: M.fall_times(s), None),
    'slew_rates':              (_clock, 'time', lambda s: M.slew_rates(s), None),
    'settling_time':           (_clock, 'time', lambda s: M.settling_time(s, initial=0.0, target=1.0),
                                lambda s: M.settling_time(s, initial=0.0, target=1.0, start=FAR)),
    'propagation_delay':       (_clock, 'time', lambda s: M.propagation_delay(s, s),
                                lambda s: M.propagation_delay(s, s, start=FAR)),
    'propagation_delays':      (_clock, 'time', lambda s: M.propagation_delays(s, s),
                                lambda s: M.propagation_delays(s, s, start=FAR)),
    'overshoot':               (_clock, 'time', lambda s: M.overshoot(s, initial=0.0, target=0.5), None),
    'undershoot':              (_clock, 'time', lambda s: M.undershoot(s, initial=0.0, target=0.5), None),
    'average_duty':            (_clock, 'time', lambda s: M.average_duty(s), None),
    'cycle_duty':              (_clock, 'time', lambda s: M.cycle_duty(s), None),
    'average_period':          (_clock, 'time', lambda s: M.average_period(s), None),
    'average_frequency':       (_clock, 'time', lambda s: M.average_frequency(s), None),
    'pulse_width':             (_clock, 'time', lambda s: M.pulse_width(s), None),
    'pulse_widths':            (_clock, 'time', lambda s: M.pulse_widths(s), None),
    'period_jitter':           (_clock, 'time', lambda s: M.period_jitter(s),
                                lambda s: M.period_jitter(s, start=FAR)),
    'period_jitter_pp':        (_clock, 'time', lambda s: M.period_jitter_pp(s),
                                lambda s: M.period_jitter_pp(s, start=FAR)),
    'cycle_to_cycle_jitter':   (_clock, 'time', lambda s: M.cycle_to_cycle_jitter(s),
                                lambda s: M.cycle_to_cycle_jitter(s, start=FAR)),
    'cycle_to_cycle_jitter_pp': (_clock, 'time', lambda s: M.cycle_to_cycle_jitter_pp(s),
                                lambda s: M.cycle_to_cycle_jitter_pp(s, start=FAR)),
    'period_deviation':        (_clock, 'time', lambda s: M.period_deviation(s),
                                lambda s: M.period_deviation(s, start=FAR)),
    'cycle_to_cycle_difference': (_clock, 'time', lambda s: M.cycle_to_cycle_difference(s),
                                lambda s: M.cycle_to_cycle_difference(s, start=FAR)),
    'crossing_jitter_rms':     (_clock, 'time', lambda s: M.crossing_jitter_rms(s, period=T),
                                lambda s: M.crossing_jitter_rms(s, period=T, start=FAR)),
    'average_power':           (_clock, 'time',
                                lambda s: M.average_power(s, _as_current(s)), None),
    'dynamic_energy_per_cycle': (_clock, 'time',
                                lambda s: M.dynamic_energy_per_cycle(s, _as_current(s), T), None),
    'hysteresis':              (_clock, 'time', lambda s: M.hysteresis(s, s, s, s), None),
    'average':                 (_clock, 'time', lambda s: M.average(s),
                                lambda s: M.average(s, start=FAR, end=2 * FAR)),
    'rms':                     (_clock, 'time', lambda s: M.rms(s),
                                lambda s: M.rms(s, start=FAR, end=2 * FAR)),
    'stddev':                  (_clock, 'time', lambda s: M.stddev(s),
                                lambda s: M.stddev(s, start=FAR, end=2 * FAR)),
    'variance':                (_clock, 'time', lambda s: M.variance(s),
                                lambda s: M.variance(s, start=FAR, end=2 * FAR)),
    'peak':                    (_clock, 'time', lambda s: M.peak(s),
                                lambda s: M.peak(s, start=FAR, end=2 * FAR)),
    'peak_to_peak':            (_clock, 'time', lambda s: M.peak_to_peak(s),
                                lambda s: M.peak_to_peak(s, start=FAR, end=2 * FAR)),
    'minimum':            (_clock, 'time', lambda s: M.minimum(s),
                                lambda s: M.minimum(s, start=FAR, end=2 * FAR)),
    'maximum':            (_clock, 'time', lambda s: M.maximum(s),
                                lambda s: M.maximum(s, start=FAR, end=2 * FAR)),
    'argmin_x':                (_clock, 'time', lambda s: M.argmin_x(s),
                                lambda s: M.argmin_x(s, start=FAR, end=2 * FAR)),
    'argmax_x':                (_clock, 'time', lambda s: M.argmax_x(s),
                                lambda s: M.argmax_x(s, start=FAR, end=2 * FAR)),
    'integral':                (_clock, 'time', lambda s: M.integral(s),
                                lambda s: M.integral(s, start=FAR, end=2 * FAR)),
    'running_integral':        (_clock, 'time', lambda s: M.running_integral(s), None),
    'moving_average':          (_clock, 'time', lambda s: M.moving_average(s, window=1e-9), None),
    'derivative':              (_clock, 'time', lambda s: M.derivative(s), None),
    'value_at':                (_clock, 'time', lambda s: M.value_at(s, 5 * T),
                                lambda s: M.value_at(s, FAR)),
    'final_value':             (_clock, 'time', lambda s: M.final_value(s), None),
    'crossing':                  (_clock, 'time', lambda s: M.crossing(s, 0.5, edge='any'), None),
    'cycle_period':            (_clock, 'time', lambda s: M.cycle_period(s), None),
    'cycle_frequency':         (_clock, 'time', lambda s: M.cycle_frequency(s), None),
    'on_time':                 (_clock, 'time', lambda s: M.on_time(s, 0.5), None),
    'off_time':                (_clock, 'time', lambda s: M.off_time(s, 0.5), None),
    'eye_diagram':             (_clock, 'time', lambda s: M.eye_diagram(s, period=T), None),
    'histogram':               (_clock, 'time', lambda s: M.histogram(s, bins=4),
                                lambda s: M.histogram(s, bins=4, start=FAR, end=2 * FAR)),
    'error_norm':              (_clock, 'time', lambda s: M.error_norm(s, s), None),
    'relative_error_norm':     (_clock, 'time', lambda s: M.relative_error_norm(s, s, kind='rms', floor=0.0), None),
    'convolve':                (_clock, 'time', lambda s: M.convolve(s, s), None),
    'lshift':                  (_clock, 'time', lambda s: M.lshift(s, T), None),
    'rshift':                  (_clock, 'time', lambda s: M.rshift(s, T), None),
    'clip':                    (_clock, 'time', lambda s: M.clip(s, low=0.2, high=0.8), None),
    'window':                  (_clock, 'time', lambda s: M.window(s, 0, 5 * T),
                                lambda s: M.window(s, FAR, 2 * FAR)),
    'flip':                    (_clock, 'time', lambda s: M.flip(s), None),
    'sample':                  (_clock, 'time', lambda s: M.sample(s, n=16), None),
    'tie':                     (_clock, 'time', lambda s: M.tie(s), None),
    'long_term_jitter':        (_clock, 'time', lambda s: M.long_term_jitter(s), None),
    'allan_deviation':         (_clock, 'time', lambda s: M.allan_deviation(s), None),
    'cross_correlation':       (_clock, 'time', lambda s: M.cross_correlation(s, s), None),
    # spectral (window column = sample window beyond the trace) ---------
    'fft':                     (_sine, 'time', lambda s: M.fft(s, **K),
                                lambda s: M.fft(s, tclk=T, tstart=FAR)),
    'sndr':                    (_sine, 'time', lambda s: M.sndr(s, **K),
                                lambda s: M.sndr(s, tclk=T, n_samples=4096)),
    'sinad':                   (_sine, 'time', lambda s: M.sinad(s, **K), None),
    'sfdr':                    (_sine, 'time', lambda s: M.sfdr(s, **K), None),
    'enob':                    (_sine, 'time', lambda s: M.enob(s, **K), None),
    'thd':                     (_sine, 'time', lambda s: M.thd(s, **K), None),
    'thd_percent':             (_sine, 'time', lambda s: M.thd_percent(s, **K), None),
    'noise_floor_dbfs':        (_sine, 'time', lambda s: M.noise_floor_dbfs(s, **K), None),
    'harmonic_levels_db':      (_sine, 'time', lambda s: M.harmonic_levels_db(s, **K), None),
    'imd3_db':                 (_sine, 'time', lambda s: M.imd3_db(s, f1_hz=11 / (1024 * T), f2_hz=13 / (1024 * T), **K), None),
    'oip3_dbm':                (_sine, 'time', lambda s: M.oip3_dbm(s, f1_hz=11 / (1024 * T), f2_hz=13 / (1024 * T), **K), None),
    'iip3_dbm':                (_sine, 'time', lambda s: M.iip3_dbm(s, s, f1_hz=11 / (1024 * T), f2_hz=13 / (1024 * T), **K), None),
    'psd':                     (_sine, 'time', lambda s: M.psd(s, **K), None),
    'spectral_power':          (_sine, 'time', lambda s: M.spectral_power(s, 0.0, 0.5 / T, **K),
                                lambda s: M.spectral_power(s, 1e12, 2e12, **K)),
    'harmonic':                (_sine, 'time', lambda s: M.harmonic(s, 1, **K), None),
    'mix_product':             (_sine, 'time', lambda s: M.mix_product(s, 1, 0, f1_hz=11 / (1024 * T), f2_hz=13 / (1024 * T), **K), None),
    'snr':                     (_sine, 'time', lambda s: M.snr(s, **K), None),
    'evm_percent':             (_sine, 'time', lambda s: M.evm_percent(s, s, symbol_period=10 * T, modulation='qpsk'),
                                lambda s: M.evm_percent(s, s, symbol_period=10 * T, modulation='qpsk', tstart=FAR)),
    'evm_constellation':       (_sine, 'time', lambda s: M.evm_constellation(s, s, symbol_period=10 * T, ideal_points=[1 + 1j, -1 - 1j]),
                                lambda s: M.evm_constellation(s, s, symbol_period=10 * T, ideal_points=[1 + 1j], tstart=FAR)),
    'harmonic_freqs':          (_sine, 'time', lambda s: M.harmonic_freqs(s, **K), None),
    'welch_psd':               (_sine, 'time', lambda s: M.welch_psd(s, **K), None),
    'coherence':               (_sine, 'time', lambda s: M.coherence(s, s, **K), None),
    # ac ------------------------------------------------------------------
    'passband_ripple_db':      (_ac, 'frequency', lambda s: M.passband_ripple_db(s, 1e3),
                                lambda s: M.passband_ripple_db(s, 1e-3)),
    'stopband_attenuation_db': (_ac, 'frequency', lambda s: M.stopband_attenuation_db(s, 1e6),
                                lambda s: M.stopband_attenuation_db(s, 1e12)),
    'q_from_peak':             (_ac, 'frequency', lambda s: M.q_from_peak(s), None),
    'group_delay':             (_ac, 'frequency', lambda s: M.group_delay(s), None),
    'gain_bw_product':         (_ac, 'frequency', lambda s: M.gain_bw_product(s), None),
    'psrr_db':                 (_ac, 'frequency', lambda s: M.psrr_db(s, s), None),
    'one_over_f_corner':       (_noise_spectrum, 'frequency', lambda s: M.one_over_f_corner(s), None),
    'dc_gain_dB':              (_ac, 'frequency', lambda s: M.dc_gain_dB(s), None),
    'peak_gain_dB':            (_ac, 'frequency', lambda s: M.peak_gain_dB(s), None),
    'gain_at_freq':            (_ac, 'frequency', lambda s: M.gain_at_freq(s, 1e3), None),
    'bandwidth_3dB':           (_ac, 'frequency', lambda s: M.bandwidth_3dB(s), None),
    'unity_gain_freq':         (_ac, 'frequency', lambda s: M.unity_gain_freq(s), None),
    'phase_at_freq':           (_ac, 'frequency', lambda s: M.phase_at_freq(s, 1e3), None),
    'phase_margin':            (_ac, 'frequency', lambda s: M.phase_margin(s), None),
    'gain_margins':            (_ac, 'frequency', lambda s: M.gain_margins(s), None),
    'gain_margin':             (_ac, 'frequency', lambda s: M.gain_margin(s), None),
    'cmrr_dB':                 (_ac, 'frequency', lambda s: M.cmrr_dB(s, s), None),
    'integrated_noise':        (_noise_spectrum, 'frequency', lambda s: M.integrated_noise(s, 1e2, 1e5),
                                lambda s: M.integrated_noise(s, 1e12, 1e13)),
    'noise_density_at_freq':   (_noise_spectrum, 'frequency', lambda s: M.noise_density_at_freq(s, 1e3),
                                lambda s: M.noise_density_at_freq(s, 1e12)),
    # TASK-414: rs = 10 ohm puts the 1 nV/sqrt(Hz) fixture at a finite NF.
    'noise_figure_at_freq':    (_noise_spectrum, 'frequency', lambda s: M.noise_figure_at_freq(s, 1e3, 10.0, 290.0),
                                lambda s: M.noise_figure_at_freq(s, 1e12, 10.0, 290.0)),
    'band_noise_figure':       (_noise_spectrum, 'frequency', lambda s: M.band_noise_figure(s, s, 1e2, 1e5, 10.0, 290.0),
                                lambda s: M.band_noise_figure(s, s, 1e12, 1e13, 10.0, 290.0)),
    'noise_figure_vs_freq':    (_noise_spectrum, 'frequency', lambda s: M.noise_figure_vs_freq(s, 10.0, 290.0), None),
    # dc ------------------------------------------------------------------
    'inl':                     (_codes, 'code', lambda s: M.inl(s), None),
    'dnl':                     (_codes, 'code', lambda s: M.dnl(s), None),
    'static_errors':           (_codes, 'code', lambda s: M.static_errors(s, full_scale=1.0, n_bits=3), None),
    'is_monotonic':            (_codes, 'code', lambda s: M.is_monotonic(s), None),
    'missing_codes':           (_codes, 'code', lambda s: M.missing_codes(s), None),
    'p1db':                    (_psweep, 'power', lambda s: M.p1db(s), None),
    'extract_code_transitions': (
        lambda xl='time': _sig(np.linspace(0, 1, 401), np.floor(np.linspace(0, 1, 401) * 4).clip(0, 3), xl),
        'time',
        lambda s: M.extract_code_transitions(Signal('v', 'voltage', s.xlabel, 'V', s.x, s.x), s),
        None),
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(fn):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            r = fn()
        except ValueError:
            return 'ValueError'
        except Exception as e:                      # noqa: BLE001 — we WANT the class
            return type(e).__name__
    if isinstance(r, Signal):
        if len(r.y) == 0:
            return 'empty'
        return 'Signal+nan' if np.any(np.isnan(np.asarray(r.y, dtype=float))) else 'Signal'
    if isinstance(r, np.ndarray):
        if r.size == 0:
            return 'empty'
        return 'array+nan' if np.any(np.isnan(r.astype(float))) else 'array'
    if isinstance(r, (bool, np.bool_)):
        return 'bool'
    if isinstance(r, list):
        if not r:
            return 'empty'
        # every element so far is a tuple of floats (gain_margins' (freq, gm)
        # pairs) — flatten to a plain float array the same way 'array+nan' does.
        flat = np.asarray(r, dtype=float).ravel()
        return 'list+nan' if np.any(np.isnan(flat)) else 'list'
    if hasattr(r, '__dataclass_fields__'):
        # A report is honest when a HEADLINE (scalar float) field is nan —
        # arrays are deliberately not inspected: eye_diagram's ``traces``
        # carried the NaN all along while height/width read a finite 0.
        scalars = [getattr(r, f) for f in r.__dataclass_fields__]
        if any(isinstance(v, (float, np.floating)) and math.isnan(v) for v in scalars):
            return 'report+nan'
        return 'report'
    if isinstance(r, complex):
        return 'nan' if math.isnan(r.real) else 'complex'
    f = float(r)
    if math.isnan(f):
        return 'nan'
    if math.isinf(f):
        return 'inf' if f > 0 else '-inf'
    return 'value'


EDGES = ('empty', 'single', 'constant', 'nan', 'wrongaxis', 'window')
LOUD = {'ValueError'}
HONEST = {'nan', 'inf', '-inf', 'Signal+nan', 'array+nan', 'report+nan', 'list+nan'}


def _cell(name, edge):
    base, xl, call, wcall = REG[name]
    if edge == 'window':
        if wcall is None:
            return '-'
        return classify(lambda: wcall(base(xl)))
    return classify(lambda: call(_edge_inputs(base, xl)[edge]))


# ---------------------------------------------------------------------------
# The pinned matrix (generated 2026-09-02 from the shipped code; '-' = n/a)
# ---------------------------------------------------------------------------

MATRIX = """
sfdr_db | ValueError | ValueError | value | nan | ValueError | -
harmonic_level_dbc | ValueError | ValueError | value | nan | ValueError | -
eye_height | ValueError | ValueError | nan | nan | ValueError | -
eye_width | ValueError | ValueError | nan | nan | ValueError | -
inl_max_lsb | ValueError | ValueError | ValueError | nan | ValueError | -
dnl_max_lsb | ValueError | ValueError | ValueError | nan | ValueError | -
offset_error_lsb | ValueError | ValueError | ValueError | nan | ValueError | -
gain_error_lsb | ValueError | ValueError | ValueError | nan | ValueError | -
full_scale_error_lsb | ValueError | ValueError | ValueError | nan | ValueError | -
missing_code_count | ValueError | ValueError | ValueError | nan | ValueError | -
non_monotonic_steps | ValueError | ValueError | ValueError | nan | ValueError | -
crossings | ValueError | empty | empty | array+nan | array | -
rise_time | ValueError | nan | nan | nan | ValueError | -
fall_time | ValueError | nan | nan | nan | ValueError | -
slew_rate | ValueError | nan | nan | nan | ValueError | -
rise_times | ValueError | empty | empty | empty | ValueError | -
fall_times | ValueError | empty | empty | empty | ValueError | -
slew_rates | ValueError | empty | empty | empty | ValueError | -
settling_time | ValueError | value | value | nan | ValueError | ValueError
propagation_delay | ValueError | nan | nan | nan | ValueError | ValueError
propagation_delays | ValueError | empty | empty | array+nan | ValueError | ValueError
overshoot | ValueError | value | value | nan | ValueError | -
undershoot | ValueError | value | value | nan | ValueError | -
average_duty | ValueError | nan | nan | nan | ValueError | -
cycle_duty | ValueError | empty | empty | Signal+nan | ValueError | -
average_period | ValueError | nan | nan | nan | ValueError | -
average_frequency | ValueError | nan | nan | nan | ValueError | -
pulse_width | ValueError | nan | nan | nan | ValueError | -
pulse_widths | ValueError | empty | empty | array+nan | ValueError | -
period_jitter | ValueError | nan | nan | nan | ValueError | ValueError
period_jitter_pp | ValueError | nan | nan | nan | ValueError | ValueError
cycle_to_cycle_jitter | ValueError | nan | nan | nan | ValueError | ValueError
cycle_to_cycle_jitter_pp | ValueError | nan | nan | nan | ValueError | ValueError
period_deviation | ValueError | empty | empty | Signal+nan | ValueError | ValueError
cycle_to_cycle_difference | ValueError | empty | empty | Signal+nan | ValueError | ValueError
crossing_jitter_rms | ValueError | nan | nan | nan | ValueError | ValueError
average_power | ValueError | ValueError | value | nan | ValueError | -
dynamic_energy_per_cycle | ValueError | ValueError | value | nan | ValueError | -
hysteresis | ValueError | nan | nan | nan | ValueError | -
average | ValueError | ValueError | value | nan | value | ValueError
rms | ValueError | ValueError | value | nan | value | ValueError
stddev | ValueError | ValueError | value | nan | value | ValueError
variance | ValueError | ValueError | value | nan | value | ValueError
peak | ValueError | ValueError | value | nan | value | ValueError
peak_to_peak | ValueError | ValueError | value | nan | value | ValueError
minimum | ValueError | ValueError | value | nan | value | ValueError
maximum | ValueError | ValueError | value | nan | value | ValueError
argmin_x | ValueError | ValueError | value | nan | value | ValueError
argmax_x | ValueError | ValueError | value | nan | value | ValueError
integral | ValueError | ValueError | value | nan | value | ValueError
running_integral | ValueError | Signal | Signal | Signal+nan | Signal | -
moving_average | ValueError | Signal+nan | Signal+nan | Signal+nan | Signal+nan | -
derivative | ValueError | ValueError | Signal | Signal+nan | Signal | -
value_at | ValueError | nan | value | value | value | nan
final_value | ValueError | value | value | value | value | -
crossing | ValueError | nan | nan | nan | value | -
cycle_period | ValueError | empty | empty | Signal+nan | ValueError | -
cycle_frequency | ValueError | empty | empty | Signal+nan | ValueError | -
on_time | ValueError | value | value | nan | ValueError | -
off_time | ValueError | value | value | nan | ValueError | -
eye_diagram | ValueError | ValueError | report+nan | report+nan | ValueError | -
histogram | ValueError | ValueError | report | report | ValueError | ValueError
error_norm | ValueError | value | value | nan | value | -
relative_error_norm | ValueError | value | value | nan | value | -
convolve | ValueError | ValueError | Signal | Signal+nan | Signal | -
lshift | ValueError | Signal | Signal | Signal+nan | Signal | -
rshift | ValueError | Signal | Signal | Signal+nan | Signal | -
clip | ValueError | Signal | Signal | Signal+nan | Signal | -
window | ValueError | Signal | Signal | Signal | Signal | empty
flip | ValueError | Signal | Signal | Signal+nan | Signal | -
sample | ValueError | Signal | Signal | Signal+nan | Signal | -
tie | ValueError | empty | empty | Signal+nan | ValueError | -
long_term_jitter | ValueError | nan | nan | nan | ValueError | -
allan_deviation | ValueError | empty | empty | Signal+nan | ValueError | -
cross_correlation | ValueError | ValueError | Signal | Signal+nan | Signal | -
fft | ValueError | ValueError | Signal | Signal+nan | ValueError | ValueError
sndr | ValueError | ValueError | value | nan | ValueError | ValueError
sinad | ValueError | ValueError | value | nan | ValueError | -
sfdr | ValueError | ValueError | report | report+nan | ValueError | -
enob | ValueError | ValueError | value | nan | ValueError | -
thd | ValueError | ValueError | value | nan | ValueError | -
thd_percent | ValueError | ValueError | value | nan | ValueError | -
noise_floor_dbfs | ValueError | ValueError | value | nan | ValueError | -
harmonic_levels_db | ValueError | ValueError | array | array+nan | ValueError | -
imd3_db | ValueError | ValueError | nan | nan | ValueError | -
oip3_dbm | ValueError | ValueError | nan | nan | ValueError | -
iip3_dbm | ValueError | ValueError | nan | nan | ValueError | -
psd | ValueError | ValueError | Signal | Signal+nan | ValueError | -
spectral_power | ValueError | ValueError | value | nan | ValueError | ValueError
harmonic | ValueError | ValueError | complex | nan | ValueError | -
mix_product | ValueError | ValueError | complex | nan | ValueError | -
snr | ValueError | ValueError | value | nan | ValueError | -
evm_constellation | ValueError | ValueError | report | report+nan | ValueError | ValueError
evm_percent | ValueError | ValueError | value | nan | ValueError | ValueError
harmonic_freqs | ValueError | ValueError | array | array+nan | ValueError | -
welch_psd | ValueError | ValueError | Signal | Signal+nan | ValueError | -
coherence | ValueError | ValueError | Signal+nan | Signal+nan | ValueError | -
passband_ripple_db | ValueError | ValueError | value | nan | ValueError | ValueError
stopband_attenuation_db | ValueError | ValueError | value | nan | ValueError | ValueError
q_from_peak | ValueError | ValueError | nan | nan | ValueError | -
group_delay | ValueError | ValueError | ValueError | Signal+nan | ValueError | -
gain_bw_product | ValueError | ValueError | nan | nan | ValueError | -
psrr_db | ValueError | ValueError | Signal | Signal+nan | ValueError | -
one_over_f_corner | ValueError | nan | nan | nan | ValueError | -
dc_gain_dB | ValueError | ValueError | value | value | ValueError | -
peak_gain_dB | ValueError | ValueError | value | nan | ValueError | -
gain_at_freq | ValueError | ValueError | value | value | ValueError | -
bandwidth_3dB | ValueError | ValueError | nan | nan | ValueError | -
unity_gain_freq | ValueError | ValueError | value | nan | ValueError | -
phase_at_freq | ValueError | ValueError | value | value | ValueError | -
phase_margin | ValueError | ValueError | value | nan | ValueError | -
gain_margins | ValueError | ValueError | empty | list+nan | ValueError | -
gain_margin | ValueError | ValueError | nan | nan | ValueError | -
cmrr_dB | ValueError | ValueError | value | value | ValueError | -
integrated_noise | ValueError | ValueError | value | nan | ValueError | ValueError
noise_density_at_freq | ValueError | ValueError | value | value | ValueError | nan
noise_figure_at_freq | ValueError | ValueError | value | value | ValueError | nan
band_noise_figure | ValueError | ValueError | value | nan | ValueError | ValueError
noise_figure_vs_freq | ValueError | ValueError | Signal | Signal+nan | ValueError | -
inl | ValueError | ValueError | ValueError | report+nan | ValueError | -
dnl | ValueError | ValueError | ValueError | report+nan | ValueError | -
static_errors | ValueError | ValueError | ValueError | report+nan | ValueError | -
is_monotonic | ValueError | ValueError | ValueError | bool | ValueError | -
missing_codes | ValueError | ValueError | ValueError | array+nan | ValueError | -
extract_code_transitions | ValueError | empty | empty | Signal+nan | ValueError | -
p1db | ValueError | ValueError | nan | nan | ValueError | -
"""

# eye_diagram's row is back with the replacement (TASK-226.12), on the same
# tokens it used to carry -- but for a different reason in the two middle
# columns: the fold now MEASURES NOTHING, so every scalar field of the
# returned EyeResult is its nan placeholder whatever the trace does. The
# cells become a statement about a measurement again only when a later task
# restores one.

# dnl / missing_codes 'constant' moved from report+nan to ValueError: DNL is
# defined per code and needs every code present (decision-1 caller error).
# The shared 'constant' fixture's code axis is np.arange(2048)*T -- not
# integers, let alone consecutive ones -- so it is now correctly refused
# rather than silently read through a per-sample LSB that happened to
# collapse to zero. missing_codes calls dnl and inherits the same refusal.
#
# inl / inl_max_lsb / static_errors / is_monotonic / non_monotonic_steps
# 'constant' moved to ValueError too (TASK-441): the shared code check now
# requires strictly ascending integer codes for the whole dc family, and the
# fixture axis is neither.


def _parse(text):
    rows = {}
    for line in text.strip().splitlines():
        cells = [c.strip() for c in line.split('|')]
        rows[cells[0]] = dict(zip(EDGES, cells[1:]))
    return rows


PINNED = _parse(MATRIX)

# Silent cells that are the mathematically right answer for that input.
NOFUND = 'no fundamental: eps-floored spectrum (TASK-52) — '
NAN_POINT = ('point read: consumes only the sample it is asked for and propagates '
             'nan when that sample is NaN (TestTask78NanInConsumedSamples); the '
             "fixture's NaN lies elsewhere in the trace")

BY_DESIGN = {
    ('sfdr_db', 'constant'): NOFUND + '0 dB SFDR floor inherited from sfdr',
    ('harmonic_level_dbc', 'constant'): NOFUND + '0 dBc harmonic floor inherited from harmonic_levels_db',
    ('allan_deviation', 'constant'): 'no edges in a flat trace: documented empty result',
    ('allan_deviation', 'single'): 'no edges in a one-sample trace: documented empty result',
    ('argmax_x', 'constant'): 'well-defined on a flat trace',
    ('argmax_x', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('argmin_x', 'constant'): 'well-defined on a flat trace',
    ('argmin_x', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('average', 'constant'): 'well-defined on a flat trace',
    ('average', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('average_power', 'constant'): 'v*i constant: exact',
    ('clip', 'constant'): 'well-defined on a flat trace',
    ('clip', 'single'): 'one-sample Signal in, one-sample Signal out',
    ('clip', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('window', 'constant'): 'well-defined on a flat trace',
    ('window', 'nan'): 'slicing primitive: samples pass through unchanged, a NaN inside the window is kept (Signal+nan), one outside it is not consumed (TestTask78NanInConsumedSamples)',
    ('clip', 'nan'): 'y limiter: NaN passes through unchanged (Signal+nan)',
    ('window', 'single'): 'one-sample Signal in, one-sample Signal out',
    ('window', 'window'): 'window semantics: an empty window returns an empty Signal; every downstream metric then raises',
    ('window', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('cmrr_dB', 'constant'): 'exact',
    ('cmrr_dB', 'nan'): NAN_POINT,
    ('convolve', 'constant'): 'well-defined on a flat trace',
    ('convolve', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('cross_correlation', 'constant'): 'well-defined on a flat trace',
    ('cross_correlation', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('crossings', 'constant'): 'no edges in a flat trace: documented empty result',
    ('crossings', 'single'): 'no edges in a one-sample trace: documented empty result',
    ('rise_times', 'constant'): 'no edge in a flat trace: documented empty array, as the singular\'s nan',
    ('fall_times', 'constant'): 'no edge in a flat trace: documented empty array, as the singular\'s nan',
    ('slew_rates', 'constant'): 'no edge in a flat trace: documented empty array, as the singular\'s nan',
    ('rise_times', 'nan'): 'consumed NaN: empty array with the singular\'s nan_waveform MetricWarning (the matrix cannot see warnings)',
    ('fall_times', 'nan'): 'consumed NaN: empty array with the singular\'s nan_waveform MetricWarning (the matrix cannot see warnings)',
    ('slew_rates', 'nan'): 'consumed NaN: empty array with the singular\'s nan_waveform MetricWarning (the matrix cannot see warnings)',
    ('rise_times', 'single'): 'no edge in a one-sample trace: documented empty array plus the singular\'s warning',
    ('fall_times', 'single'): 'no edge in a one-sample trace: documented empty array plus the singular\'s warning',
    ('slew_rates', 'single'): 'no edge in a one-sample trace: documented empty array plus the singular\'s warning',
    ('propagation_delays', 'constant'): 'no trigger edge in a flat trace: documented empty result, as crossings',
    ('propagation_delays', 'single'): 'no trigger edge in a one-sample trace: documented empty result, as crossings',
    ('dc_gain_dB', 'constant'): 'gain at the lowest frequency: exact',
    ('dc_gain_dB', 'nan'): NAN_POINT,
    ('derivative', 'constant'): 'well-defined on a flat trace',
    ('derivative', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('dynamic_energy_per_cycle', 'constant'): 'constant power over whole cycles: exact',
    ('enob', 'constant'): NOFUND + 'negative ENOB from the failing sndr floor',
    ('error_norm', 'constant'): 'identical signals: 0',
    ('error_norm', 'single'): 'identical one-sample signals: 0',
    ('error_norm', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('relative_error_norm', 'constant'): 'identical signals: 0',
    ('relative_error_norm', 'single'): 'identical one-sample signals: 0',
    ('relative_error_norm', 'wrongaxis'): 'axis-agnostic primitive by design (like error_norm)',
    ('evm_constellation', 'constant'): 'well-defined on a flat trace',
    ('evm_percent', 'constant'): 'a constant I/Q is one QPSK point after the gain fit: 0 % is exact',
    ('extract_code_transitions', 'single'): 'one sample = one code = the only (and lowest) observed code: IEEE 1241 transition levels start at lowest+1, so no row exists (TASK-53 fix)',
    ('extract_code_transitions', 'constant'): 'a constant code trace has one observed code = the lowest: IEEE 1241 transition levels start at lowest+1, so no row exists (TASK-53 fix)',
    # eye_diagram needs no BY_DESIGN entry: 'report+nan' is an HONEST cell,
    # and under TASK-226.12 it is honest by construction -- the fold returns
    # windows and computes no scalar at all. The withdrawn implementation's
    # reason for the same token, kept because it is what the cell will mean
    # again once the measurement comes back: 'a flat trace never crosses the
    # threshold, so there is no cluster, no gap and no instant:
    # height/width all nan (TASK-226.8).
    # It used to read height 0 / width 0 / jitter 0, and a 0.0 jitter out of
    # a trace in which nothing was measured PASSES a jitter spec'.
    ('fft', 'constant'): 'well-defined on a flat trace',
    ('flip', 'constant'): 'well-defined on a flat trace',
    ('flip', 'single'): 'one-sample Signal in, one-sample Signal out',
    ('flip', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('gain_at_freq', 'constant'): 'exact',
    ('gain_at_freq', 'nan'): NAN_POINT,
    ('gain_margins', 'constant'): 'no phase crossover: documented empty list',
    ('harmonic', 'constant'): NOFUND + 'the elected bin holds 0+0j, an honest zero',
    ('harmonic_freqs', 'constant'): NOFUND + 'bin 1 is elected; its finite harmonic grid is reported, the ratio metrics fail',
    ('harmonic_levels_db', 'constant'): NOFUND + 'every harmonic reads 0 dBc (eps/eps): a failing level',
    ('histogram', 'constant'): 'a constant window spends all its time in one bin: exact',
    ('histogram', 'nan'): 'fractions are all nan with a nan_waveform warning; the classifier does not inspect arrays (tests/metrics/test_histogram_382.py)',
    ('cycle_duty', 'constant'): 'no edges in a flat trace: documented empty result, as cycle_period',
    ('cycle_duty', 'single'): 'no edges in a one-sample trace: documented empty result, as cycle_period',
    ('pulse_widths', 'constant'): 'no pulse in a flat trace: documented empty result, as crossings',
    ('pulse_widths', 'single'): 'no pulse in a one-sample trace: documented empty result, as crossings',
    ('cycle_period', 'constant'): 'no edges in a flat trace: documented empty result',
    ('period_deviation', 'constant'): 'no edges in a flat trace: documented empty result, as cycle_period',
    ('period_deviation', 'single'): 'no edges in a one-sample trace: documented empty result, as cycle_period',
    ('cycle_to_cycle_difference', 'constant'): 'no edges in a flat trace: documented empty result, as cycle_period',
    ('cycle_to_cycle_difference', 'single'): 'no edges in a one-sample trace: documented empty result, as cycle_period',
    ('cycle_period', 'single'): 'no edges in a one-sample trace: documented empty result',
    ('cycle_frequency', 'constant'): 'no edges in a flat trace: documented empty result',
    ('cycle_frequency', 'single'): 'no edges in a one-sample trace: documented empty result',
    ('integral', 'constant'): 'well-defined on a flat trace',
    ('integral', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('integrated_noise', 'constant'): 'white PSD integrated over the band: exact',
    ('is_monotonic', 'nan'): 'a NaN cannot be ordered so the curve is not provably increasing: False, the always-failing direction (decision-1 §1; a bool has no nan) — docstring says so (TASK-78)',
    ('lshift', 'constant'): 'well-defined on a flat trace',
    ('lshift', 'single'): 'one-sample Signal in, one-sample Signal out',
    ('lshift', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('rshift', 'constant'): 'well-defined on a flat trace',
    ('rshift', 'single'): 'one-sample Signal in, one-sample Signal out',
    ('rshift', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('mix_product', 'constant'): 'well-defined on a flat trace',
    ('noise_density_at_freq', 'constant'): 'exact',
    ('noise_density_at_freq', 'nan'): NAN_POINT,
    ('noise_figure_at_freq', 'constant'): 'a flat density is a white-noise amplifier: exact spot NF',
    ('noise_figure_at_freq', 'nan'): NAN_POINT,
    ('band_noise_figure', 'constant'): 'a flat density at unity gain is a white-noise amplifier: exact band NF',
    ('noise_figure_vs_freq', 'constant'): 'a flat density is a white-noise amplifier: exact NF at every point',
    ('noise_floor_dbfs', 'constant'): NOFUND + '0 dBFS (noise == fundamental == eps): a failing floor',
    ('off_time', 'constant'): 'always above level: 0 s',
    ('off_time', 'single'): 'zero-duration trace: 0 s',
    ('on_time', 'constant'): 'always above level: whole trace',
    ('on_time', 'single'): 'zero-duration trace: 0 s',
    ('overshoot', 'constant'): 'flat trace: peak = level',
    ('overshoot', 'single'): 'the one sample is the peak',
    ('passband_ripple_db', 'constant'): 'flat passband: 0 dB ripple is correct',
    ('peak', 'constant'): 'well-defined on a flat trace',
    ('peak', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('peak_gain_dB', 'constant'): 'exact',
    ('peak_to_peak', 'constant'): 'well-defined on a flat trace',
    ('peak_to_peak', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('phase_at_freq', 'constant'): 'exact',
    ('phase_at_freq', 'nan'): NAN_POINT,
    ('phase_margin', 'constant'): 'phase 0 at UGF sits on the (-180, 180] branch cut and reads -180 (doc nit, ledger)',
    ('psd', 'constant'): 'well-defined on a flat trace',
    ('psrr_db', 'constant'): 'well-defined on a flat trace',
    ('rms', 'constant'): 'well-defined on a flat trace',
    ('rms', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('running_integral', 'constant'): 'well-defined on a flat trace',
    ('running_integral', 'single'): 'one-sample Signal in, one-sample Signal out',
    ('running_integral', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('sample', 'constant'): 'well-defined on a flat trace',
    ('sample', 'single'): 'one-sample Signal in, one-sample Signal out',
    ('sample', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('settling_time', 'constant'): 'flat trace equal to the target: settled at t=0',
    ('settling_time', 'single'): 'one sample equal to the target: settled at t=0',
    ('sfdr', 'constant'): NOFUND + 'SFDR 0 dB (spur == fundamental == eps): a failing value',
    ('sinad', 'constant'): NOFUND + 'same failing floor as sndr',
    ('sndr', 'constant'): NOFUND + '-10*log10(N/2-1) dB, N-dependent, always failing',
    ('snr', 'constant'): NOFUND + '-10*log10(N/2-10) dB (harmonic bins excluded), always failing',
    ('spectral_power', 'constant'): 'mean-removed flat trace has zero band power: 0 is correct',
    ('stddev', 'constant'): 'well-defined on a flat trace',
    ('stddev', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('variance', 'constant'): 'well-defined on a flat trace',
    ('variance', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('stopband_attenuation_db', 'constant'): 'flat response: 0 dB attenuation is correct',
    ('thd', 'constant'): NOFUND + '+6 dB (4 harmonic bins of eps over eps): a failing value',
    ('thd_percent', 'constant'): NOFUND + '200 % (sqrt(4 eps / eps)): a failing value',
    ('tie', 'constant'): 'no edges in a flat trace: documented empty result',
    ('tie', 'single'): 'no edges in a one-sample trace: documented empty result',
    ('undershoot', 'constant'): 'flat trace: trough = level',
    ('undershoot', 'single'): 'the one sample is the trough',
    ('unity_gain_freq', 'constant'): '|H| == 1 everywhere: every f is a unity-gain point; returns the first',
    ('final_value', 'constant'): 'well-defined on a flat trace',
    ('final_value', 'nan'): NAN_POINT.replace('point read', 'last-sample read'),
    ('final_value', 'single'): 'the one sample is the final value',
    ('final_value', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('value_at', 'constant'): 'well-defined on a flat trace',
    ('value_at', 'nan'): NAN_POINT,
    ('value_at', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('welch_psd', 'constant'): 'well-defined on a flat trace',
    ('maximum', 'constant'): 'well-defined on a flat trace',
    ('maximum', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('minimum', 'constant'): 'well-defined on a flat trace',
    ('minimum', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('crossing', 'wrongaxis'): 'axis-agnostic primitive by design (module docstring)',
    ('crossings', 'wrongaxis'): 'acquisition-order primitive on any monotonic axis (TASK-513 approval)',
}

# Silent cells that are wrong or misleading — each owned by a filed task,
# which must still be OPEN (test_every_silent_cell_is_accounted_for reads
# the owner's backlog status; a closed or never-filed owner fails like a
# missing one — TASK-78).
#
# History. TASK-59 (2026-09-04) fixed its ten ruled NaN outliers (crossings,
# average_duty, argmin_x, argmax_x, on_time, off_time, crossing, settling_time,
# bandwidth_3dB, harmonic_freqs). TASK-77 (2026-09-04) closed the whole
# 'empty' column (typed ValueError from _check_nonempty). TASK-78
# (2026-09-04) triaged the 25 'nan' cells left pointing at the closed
# TASK-59: 11 returned a finite/empty wrong answer once the NaN sat in the
# samples they consume and now propagate nan (tie, allan_deviation,
# cycle_frequency -> one-sample nan Signal; missing_codes -> array([nan]);
# extract_code_transitions -> observed code axis, all-nan voltages; sfdr,
# eye_diagram -> nan headline fields; passband_ripple_db, unity_gain_freq,
# gain_bw_product, one_over_f_corner -> nan); 3 already carried the nan in
# their report and became visible once the classifier looked at headline
# fields (dnl, inl, evm_constellation -> 'report+nan'); 11 are point/window
# reads whose fixture NaN was never consumed and moved to BY_DESIGN with a
# proof test (TestTask78NanInConsumedSamples). Nothing is filed today.
FILED = {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_matrix_covers_every_public_signal_metric():
    scalar_only = {'noise_figure', 'noise_figure_from_factor', 'input_referred_noise',
                   'sum_uncorrelated', 'kT_over_C',
                   'aperture_jitter_snr_db', 'cascade_noise_figure',
                   'walden_fom', 'schreier_fom', 'iip3_extrapolation', 'pae',
                   'dc_to_rf_efficiency', 'modulation_grid'}
    classes = {'SFDR', 'INL', 'DNL', 'StaticErrors', 'EyeResult', 'EVMResult', 'HistogramResult'}
    submodules = {'waveform', 'statistics', 'timing', 'jitter', 'eye', 'power',
                            'spectral', 'rf', 'converter', 'fom', 'ac', 'noise'}
    public = set(M.__all__) - scalar_only - classes - submodules
    assert public == set(REG), (sorted(public - set(REG)), sorted(set(REG) - public))
    assert set(PINNED) == set(REG)


@pytest.mark.parametrize('name,edge', [(n, e) for n in REG for e in EDGES])
def test_cell(name, edge):
    # Every cell is pinned to ONE token. The matrix was regenerated under
    # numpy 1.26.4 / 2.2.6 / 2.3.4 / 2.4.4 / 2.5.2 (TASK-65) and is
    # identical on all of them — a cell that moves with the numpy version
    # is a metric leaning on numpy's incidental empty-input behaviour, and
    # the metric (not this table) is what gets fixed.
    assert _cell(name, edge) == PINNED[name][edge]


# ---------------------------------------------------------------------------
# FILED owners must be OPEN tasks (TASK-78)
#
# A FILED entry is a promise that somebody still owns the fix. The original
# guard only checked that an owner STRING existed, so the moment TASK-59
# closed, 32 cells kept citing it and the suite stayed green — the inventory
# stopped self-policing exactly when it mattered. Now the owner's backlog
# task file is read and its status must not be terminal; a closed or
# never-filed owner fails as loudly as a missing one.
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_BACKLOG = _REPO / 'backlog'


def _terminal_statuses():
    """Terminal statuses per backlog/config.yml: the LAST configured status
    (Backlog.md's convention — 'Done' here)."""
    cfg = yaml.safe_load((_BACKLOG / 'config.yml').read_text())
    return {cfg['statuses'][-1]}


def _owner_status(task_id):
    """Status of ``task_id`` from its task file's front matter, or ``None``
    if no file for it exists under ``backlog/tasks`` (never filed, or moved
    to completed/archive — either way nobody active owns the cell)."""
    files = list((_BACKLOG / 'tasks').glob(f'{task_id.lower()} - *.md'))
    if not files:
        return None
    fm = re.match(r'---\n(.*?)\n---', files[0].read_text(), re.DOTALL)
    m = re.search(r'^status:\s*(.+?)\s*$', fm.group(1), re.MULTILINE) if fm else None
    return m.group(1).strip('\'"') if m else None


def _assert_owner_open(key, owner):
    """Fail unless ``owner`` names a backlog task that is still open."""
    name, edge = key
    assert owner.startswith('TASK-'), f'{name} × {edge}: owner {owner!r} is not a task id'
    if not _BACKLOG.is_dir():
        pytest.skip('backlog/ not present: FILED ownership cannot be verified '
                    'outside the source repository')
    status = _owner_status(owner)
    assert status is not None, (
        f'{name} × {edge} is FILED under {owner}, but no such task exists in '
        f'backlog/tasks — the cell has no owner. Re-point it at an open task '
        f'or move it to BY_DESIGN with a reason')
    assert status not in _terminal_statuses(), (
        f'{name} × {edge} is FILED under {owner}, which is {status!r} — a closed '
        f'task does not own a fix. Re-point it at an open task or move it to '
        f'BY_DESIGN with a reason')


@pytest.mark.parametrize('name', sorted(REG))
def test_every_silent_cell_is_accounted_for(name):
    for edge in EDGES:
        cls = PINNED[name][edge]
        if cls == '-' or cls in LOUD or cls in HONEST or cls.endswith('Error'):
            continue                      # untyped-loud cells: see the test below
        key = (name, edge)
        assert key in BY_DESIGN or key in FILED, (
            f'{name} × {edge} silently returns {cls!r}: add it to BY_DESIGN '
            f'with a reason or FILE it')
        if key in FILED:
            _assert_owner_open(key, FILED[key])


def test_filed_owner_in_terminal_status_fails_as_loudly_as_a_missing_owner():
    """TASK-78: the guard hole. TASK-59 is a KNOWN CLOSED task (it is the one
    that orphaned 32 cells); a cell pinned to it must fail, and so must a cell
    pinned to a task that was never filed. Both failures name the cell and
    the owner so the reader knows what to re-point."""
    if not _BACKLOG.is_dir():
        pytest.skip('backlog/ not present')
    assert _owner_status('TASK-59') in _terminal_statuses(), (
        'this test relies on TASK-59 being closed; if it was reopened, pick '
        'another closed task id')
    with pytest.raises(AssertionError, match=r"x × nan is FILED under TASK-59, which is 'Done'"):
        _assert_owner_open(('x', 'nan'), 'TASK-59')
    with pytest.raises(AssertionError, match=r'x × nan is FILED under TASK-999999, but no such task'):
        _assert_owner_open(('x', 'nan'), 'TASK-999999')
    # an OPEN owner passes — this task itself, while it is in progress
    if _owner_status('TASK-78') not in _terminal_statuses():
        _assert_owner_open(('x', 'nan'), 'TASK-78')


def test_no_untyped_loud_cells_remain():
    """Every loud failure is a typed, actionable metric error.

    TASK-77 cleared the whole ``empty`` column. TASK-80 then closed the
    sole survivor, ``derivative × single``, with a derivative-local guard;
    other metrics may still legitimately accept one sample.
    """
    untyped = sorted((n, e) for n in REG for e in EDGES
                     if PINNED[n][e].endswith('Error') and PINNED[n][e] not in LOUD)
    assert untyped == [], untyped


if __name__ == '__main__':                   # regenerate the table
    for n in REG:
        print(f'{n} | ' + ' | '.join(_cell(n, e) for e in EDGES))
