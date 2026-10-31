"""crossing_jitter_rms: data-signal crossing jitter against an ideal grid
(TASK-226.20).

Every waveform here is piecewise linear with each transition a linear
ramp centred on its intended crossing time, so :func:`crossings` is exact
(its interpolation floor is zero) and the expected jitter is analytic.
"""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal

UI = 100e-12
RAMP = 20e-12


def prbs7(n):
    """n bits of PRBS7 (x^7 + x^6 + 1) — runs of up to seven equal bits."""
    state = 0x7F
    out = []
    for _ in range(n):
        bit = ((state >> 6) ^ (state >> 5)) & 1
        state = ((state << 1) | bit) & 0x7F
        out.append(bit)
    return out


def nrz(bits, *, offsets=None, falling_shift=0.0, ui=UI, t0=0.0):
    """PWL NRZ: transition i at ``t0 + i*ui + offsets[i]`` (+ ``falling_shift``
    on falling edges). Returns ``(Signal, rising_times, falling_times)``."""
    offsets = np.zeros(len(bits)) if offsets is None else offsets
    x, y = [t0 - ui], [float(bits[0])]
    rise, fall = [], []
    for i in range(1, len(bits)):
        if bits[i] == bits[i - 1]:
            continue
        c = t0 + i * ui + offsets[i]
        if bits[i] < bits[i - 1]:
            c += falling_shift
            fall.append(c)
        else:
            rise.append(c)
        x += [c - RAMP / 2, c + RAMP / 2]
        y += [float(bits[i - 1]), float(bits[i])]
    x.append(t0 + (len(bits) + 1) * ui)
    y.append(float(bits[-1]))
    sig = Signal('data', 'voltage', 'time', 'V', np.array(x), np.array(y))
    return sig, np.array(rise), np.array(fall)


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('error', MetricWarning)
        return fn(*args, **kwargs)


# --- registration ----------------------------------------------------------

def test_registered_in_catalog():
    from silk.signals.schema import export_metric_schema
    entry = export_metric_schema()['crossing_jitter_rms']
    assert entry['domain'] == 'tran'
    assert entry['unit'] == 's'
    assert [s['name'] for s in entry['signal_inputs']] == ['sig']
    params = {p['name']: p for p in entry['parameters']}
    assert params['period']['required'] is True
    assert params['period']['unit'] == 's'
    assert params['edge']['default'] == 'rising'
    assert params['edge']['choices'] == ['rising', 'falling', 'any']
    assert params['start']['unit'] == 's' and params['end']['unit'] == 's'


# --- analytic behaviour ----------------------------------------------------

@pytest.mark.parametrize('edge', ['rising', 'falling', 'any'])
def test_zero_jitter_nrz_reads_zero(edge):
    sig, _, _ = nrz(prbs7(2000))
    assert quiet(M.crossing_jitter_rms, sig, period=UI, edge=edge) < 1e-9 * UI


@pytest.mark.parametrize('edge', ['rising', 'falling', 'any'])
def test_gaussian_edge_jitter_recovered(edge):
    sigma = 2e-12
    bits = prbs7(8000)
    offsets = np.random.default_rng(226_20).normal(0.0, sigma, len(bits))
    sig, rise, fall = nrz(bits, offsets=offsets)
    got = quiet(M.crossing_jitter_rms, sig, period=UI, edge=edge)
    # The metric equals the sample std of the injected offsets of the
    # selected edges exactly (the grid is fixed, the edges are exact) ...
    sel = {'rising': rise, 'falling': fall,
           'any': np.concatenate([rise, fall])}[edge]
    injected = np.mod(sel + UI / 2, UI) - UI / 2
    assert got == pytest.approx(np.std(injected, ddof=1), rel=1e-9)
    # ... and that recovers sigma within its sampling uncertainty
    # (relative SE 1/sqrt(2(n-1)) is below 3 % for n >= 600 edges).
    assert got == pytest.approx(sigma, rel=0.1)


def test_runs_of_equal_bits_do_not_inflate():
    sigma = 2e-12
    n = 4000
    offsets = np.random.default_rng(7).normal(0.0, sigma, n)
    data, _, _ = nrz(prbs7(n), offsets=offsets)
    alternating, _, _ = nrz([i % 2 for i in range(n)], offsets=offsets)
    on_data = quiet(M.crossing_jitter_rms, data, period=UI, edge='any')
    on_clock = quiet(M.crossing_jitter_rms, alternating, period=UI, edge='any')
    assert on_data == pytest.approx(sigma, rel=0.1)
    assert on_clock == pytest.approx(sigma, rel=0.1)
    # The contrast: long_term_jitter assumes one rising edge per period,
    # which the data pattern breaks, and reads many UIs of "jitter".
    assert M.long_term_jitter(data, ideal_period=UI) > 100 * sigma


def test_edge_selections_differ_by_duty_cycle_distortion():
    delta = 6e-12
    sig, rise, fall = nrz(prbs7(2000), falling_shift=delta)
    assert quiet(M.crossing_jitter_rms, sig, period=UI, edge='rising') < 1e-9 * UI
    assert quiet(M.crossing_jitter_rms, sig, period=UI, edge='falling') < 1e-9 * UI
    pooled = np.concatenate([np.zeros(len(rise)), np.full(len(fall), delta)])
    got = quiet(M.crossing_jitter_rms, sig, period=UI, edge='any')
    assert got == pytest.approx(np.std(pooled, ddof=1), rel=1e-6)
    assert got == pytest.approx(delta / 2, rel=0.01)


def test_default_edge_is_rising():
    sig, _, _ = nrz(prbs7(2000), falling_shift=6e-12)
    assert M.crossing_jitter_rms(sig, period=UI) == \
        M.crossing_jitter_rms(sig, period=UI, edge='rising')


@pytest.mark.parametrize('grid_phase', [0.0, 0.25, 0.5, 0.75, 0.999])
def test_wrap_does_not_split_the_cluster(grid_phase):
    """The same jitter reads the same wherever the grid sits in [0, UI)."""
    bits = prbs7(3000)
    offsets = np.random.default_rng(3).normal(0.0, 3e-12, len(bits))
    ref, _, _ = nrz(bits, offsets=offsets)
    shifted, _, _ = nrz(bits, offsets=offsets, t0=grid_phase * UI)
    a = quiet(M.crossing_jitter_rms, ref, period=UI, edge='any')
    b = quiet(M.crossing_jitter_rms, shifted, period=UI, edge='any')
    assert b == pytest.approx(a, rel=1e-6)


def test_window_excludes_startup():
    bits = prbs7(1000)
    offsets = np.zeros(len(bits))
    offsets[:40] = 30e-12          # a start-up transient on the first 40 UI
    sig, _, _ = nrz(bits, offsets=offsets)
    assert quiet(M.crossing_jitter_rms, sig, period=UI, edge='any') > 5e-12
    windowed = quiet(M.crossing_jitter_rms, sig, period=UI, edge='any',
                     start=50 * UI)
    assert windowed < 1e-9 * UI
    head_only = quiet(M.crossing_jitter_rms, sig, period=UI, edge='any',
                      end=500 * UI)
    assert head_only > 5e-12


def test_frequency_offset_counts_as_jitter():
    """period is the caller's UI and is never fitted."""
    sig, _, _ = nrz(prbs7(200))
    assert quiet(M.crossing_jitter_rms, sig, period=UI * (1 + 1e-4),
                 edge='any') > 1e-13


# --- decision-1: degenerate data returns nan, caller error raises -----------

@pytest.mark.parametrize('bits', [[0] * 10, [0] * 5 + [1] * 5])
def test_too_few_crossings_nan_with_warning(bits):
    sig, _, _ = nrz(bits)
    with pytest.warns(MetricWarning) as rec:
        got = M.crossing_jitter_rms(sig, period=UI)
    assert math.isnan(got)
    codes = [w.message.code for w in rec if isinstance(w.message, MetricWarning)]
    assert codes == ['insufficient_edges']
    assert rec[0].message.metric == 'crossing_jitter_rms'


def test_nan_sample_returns_nan_with_one_warning():
    sig, _, _ = nrz(prbs7(200))
    y = sig.y.copy()
    y[len(y) // 2] = np.nan
    sig = Signal('data', 'voltage', 'time', 'V', sig.x, y)
    with pytest.warns(MetricWarning) as rec:
        got = M.crossing_jitter_rms(sig, period=UI)
    assert math.isnan(got)
    assert len([w for w in rec if issubclass(w.category, MetricWarning)]) == 1


@pytest.mark.parametrize('edge', ['both', 'rise', None, ''])
def test_bad_edge_raises(edge):
    sig, _, _ = nrz(prbs7(200))
    with pytest.raises(ValueError, match='crossing_jitter_rms: edge'):
        M.crossing_jitter_rms(sig, period=UI, edge=edge)


@pytest.mark.parametrize('period', [0.0, -UI, float('nan'), float('inf'),
                                    True, '100p', None])
def test_bad_period_raises(period):
    sig, _, _ = nrz(prbs7(200))
    with pytest.raises(ValueError, match='crossing_jitter_rms: period'):
        M.crossing_jitter_rms(sig, period=period)


def test_period_is_required():
    sig, _, _ = nrz(prbs7(200))
    with pytest.raises(TypeError):
        M.crossing_jitter_rms(sig)


def test_window_outside_trace_raises():
    sig, _, _ = nrz(prbs7(200))
    with pytest.raises(ValueError, match='does not overlap'):
        M.crossing_jitter_rms(sig, period=UI, start=1.0)


def test_non_transient_raises():
    sig = Signal('g', 'voltage', 'frequency', 'V',
                 np.arange(10.0), np.arange(10.0) % 2)
    with pytest.raises(ValueError):
        M.crossing_jitter_rms(sig, period=1.0)
