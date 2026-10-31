"""Approved spectral input fixes, with numeric and locality controls."""
import math
import warnings

import numpy as np
import pytest

from silk.circuit.spec_report import SpecReport
import silk.metrics as M
from silk.metrics import rf
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def trace(values=None):
    t = np.arange(65, dtype=float)
    if values is None:
        values = np.cos(2 * np.pi * .125 * t)
    return Signal('trace', 'voltage', 'time', 'V', t, np.asarray(values))


IMD = dict(tclk=1, n_samples=64, f1_hz=.125, f2_hz=.1875)


def test_contaminated_imd_cannot_pass_upper_bound():
    # The audit's false PASS: max(0, nan) used to hide the missing product.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        value = rf.imd3_db(trace(np.full(65, np.nan)), **IMD)
    report = SpecReport('tran', 'two_tone', {'imd3': value})
    assert not report.check('distortion', 'imd3', '<', -40).passed
    assert type(value) is float and math.isnan(value)
    for op in ('<=', '>', '>=', '=='):
        separate = SpecReport('tran', 'two_tone', {'imd3': value})
        assert not separate.check(op, 'imd3', op, -40).passed


@pytest.mark.parametrize('bad_bin', [4, 8, 12, 16])
def test_each_consumed_imd_bin_is_checked_before_max(monkeypatch, bad_bin):
    # Isolate bins: a time-domain NaN contaminates every FFT coefficient.
    spectrum = np.zeros(33, dtype=complex)
    spectrum[[8, 12]] = [32, 16]
    spectrum[bad_bin] = np.nan
    monkeypatch.setattr(np.fft, 'rfft', lambda samples: spectrum)
    with pytest.warns(MetricWarning) as caught:
        value = rf.imd3_db(trace(), **IMD)
    assert math.isnan(value)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == 'imd3_db'
    assert diagnostic.code == 'nan_consumed_bins'
    assert 'contains NaNs' in diagnostic.message
    assert 'source result' in diagnostic.suggestion
    assert diagnostic.details['nan_bin_count'] == 1
    assert diagnostic.details['bin_hz'] == 1 / 64
    assert diagnostic.details['fft_bin_count'] == 33
    assert all(type(v) in (str, float, int, type(None))
               for v in diagnostic.details.values())


def test_imd_zero_products_with_real_reference_remain_minus_inf(monkeypatch):
    spectrum = np.zeros(33, dtype=complex)
    spectrum[[8, 12]] = [32, 16]
    spectrum[20] = np.nan  # This bin is not consumed by IMD3.
    monkeypatch.setattr(np.fft, 'rfft', lambda samples: spectrum)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        value = rf.imd3_db(trace(), **IMD)
    assert value == -math.inf
    assert not caught


def test_imd_nan_outside_sample_window_is_not_consumed():
    clean = trace()
    dirty = trace()
    dirty.y[-1] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        assert rf.imd3_db(dirty, **IMD) == rf.imd3_db(clean, **IMD)
    assert not caught


@pytest.mark.parametrize('tones,fragments', [
    ((1, 2), ['f1_hz=1', 'bin=64']),
    ((.125, 2), ['f2_hz=2', 'bin=128']),
    ((.125, .501), ['f2_hz=0.501', 'bin=32']),
    ((.001, .1875), ['bin=0']),
    ((.125, .126), ['degenerate', '8']),
])
def test_imd_invalid_bins_report_fft_coverage(tones, fragments):
    with pytest.raises(ValueError) as caught:
        rf.imd3_db(trace(), tclk=1, n_samples=64,
                  f1_hz=tones[0], f2_hz=tones[1])
    message = str(caught.value)
    for fragment in ['imd3_db:', 'df=', 'n=33', '0.5', *fragments]:
        assert fragment in message


def test_imd_in_range_nearest_bin_behavior_is_preserved():
    assert rf.imd3_db(trace(), **dict(IMD, f1_hz=.126, f2_hz=.188)) == (
        rf.imd3_db(trace(), **IMD))


@pytest.mark.parametrize('policy', ['ignore', 'error'])
def test_imd_warning_policy(policy):
    with warnings.catch_warnings():
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning):
                rf.imd3_db(trace(np.full(65, np.nan)), **IMD)
        else:
            assert math.isnan(rf.imd3_db(trace(np.full(65, np.nan)), **IMD))


def test_evm_rejects_symbol_center_before_trace():
    with pytest.raises(ValueError) as caught:
        rf.evm_constellation(trace(), trace(), symbol_period=8,
                            ideal_points=[-1, 1], tstart=-8)
    message = str(caught.value)
    for fragment in ['evm_constellation:', '-4', 'before', '0.0', '64.0']:
        assert fragment in message


@pytest.mark.parametrize('tstart,n_symbols', [(-4, 9), (0, 8)])
def test_evm_in_coverage_centers_and_result_fields(tstart, n_symbols):
    i = trace(np.ones(65))
    q = trace(np.zeros(65))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        result = rf.evm_constellation(i, q, symbol_period=8,
                                    ideal_points=[-1, 1], tstart=tstart,
                                    n_symbols=n_symbols)
    assert isinstance(result, rf.EVMResult)
    assert result.evm_percent == 0.0
    for values in (result.received, result.errors, result.assigned_ideals):
        assert values.shape == (n_symbols,)
        assert np.iscomplexobj(values)
    np.testing.assert_array_equal(result.errors, 0)
    assert not caught


def test_evm_nan_outside_interpolation_support_is_not_consumed():
    i = trace(np.ones(65))
    q = trace(np.zeros(65))
    i.y[1] = np.nan
    q.y[-1] = np.nan
    result = rf.evm_constellation(i, q, symbol_period=8, ideal_points=[-1, 1])
    assert result.evm_percent == 0.0
    np.testing.assert_array_equal(result.received, np.ones(8, dtype=complex))


@pytest.mark.parametrize('ideals', [[0], [0j, 1 + 1j], [-1, 0, 1]])
def test_evm_rejects_each_zero_magnitude_ideal(ideals):
    with pytest.raises(ValueError) as caught:
        rf.evm_constellation(trace(), trace(), symbol_period=8,
                            ideal_points=ideals)
    message = str(caught.value)
    for fragment in ['evm_constellation:', 'ideal_points', 'zero', 'index']:
        assert fragment in message


@pytest.mark.parametrize('n,fund_bin,k', [(8, 4, 2), (9, 3, 4)])
def test_unattributable_harmonic_level_is_nan(n, fund_bin, k):
    t = np.arange(n, dtype=float)
    sig = Signal('harmonics', 'voltage', 'time', 'V', t,
                 np.cos(2 * np.pi * fund_bin * t / n))
    with pytest.warns(MetricWarning) as caught:
        levels = M.harmonic_levels_db(sig, tclk=1, n_samples=n, n_harmonics=k)
    assert levels.shape == (k - 1,)
    assert math.isnan(levels[k - 2])
    diagnostic = caught[-1].message
    assert diagnostic.metric == 'harmonic_levels_db'
    assert diagnostic.code == 'unattributable_harmonic'
    assert diagnostic.details['harmonic'] == k
    assert diagnostic.details['fundamental_bin'] == fund_bin
    assert diagnostic.details['folded_bin'] in (0, fund_bin)
    assert diagnostic.details['bin_hz'] == 1 / n
    assert math.isnan(M.harmonic(sig, k, tclk=1, n_samples=n).real)
    assert math.isnan(M.harmonic_freqs(sig, tclk=1, n_samples=n,
                                     n=k, observed=True)[-1])


@pytest.mark.parametrize('name,kwargs,fragments', [
    ('fft', {'tclk': 0}, ['tclk', '0', '64']),
    ('sinad', {'tclk': 0}, ['tclk', '0', '64']),
    ('enob', {'tclk': 0}, ['tclk', '0', '64']),
    ('fft', {'tstart': -1}, ['tstart', '-1', '64']),
    ('fft', {'tstart': 64}, ['too short', '64']),
    ('fft', {'n_samples': 2}, ['n_samples', '2', '4']),
    ('fft', {'n_samples': 100}, ['window', '99', '64']),
    ('psd', {'window': 'wrong'}, ['window', 'wrong', 'hann', '64']),
    ('spectral_power', {'f1_hz': .1, 'f2_hz': .3, 'window': 'wrong'},
     ['window', 'wrong', '64']),
    ('thd', {'n_harmonics': 1}, ['n_harmonics', '1', '2']),
    ('thd_percent', {'n_harmonics': 1}, ['n_harmonics', '1', '2']),
    ('harmonic_levels_db', {'n_harmonics': 1}, ['n_harmonics', '1', '2']),
    ('snr', {'n_harmonics': 1}, ['n_harmonics', '1', '2']),
    ('harmonic', {'k': 0}, ['k', '0', '1']),
    ('harmonic_freqs', {'n': 0}, ['n', '0', '1']),
    ('welch_psd', {'segments': 0}, ['segments', '0', '64']),
    ('welch_psd', {'overlap': 1}, ['overlap', '1', '64']),
    ('welch_psd', {'segments': 100}, ['segments', '100', '64', 'seg_len']),
])
def test_local_validation_names_caller_request_and_available(name, kwargs, fragments):
    with pytest.raises(ValueError) as caught:
        getattr(M, name)(trace(), **(dict(tclk=1, n_samples=64) | kwargs))
    message = str(caught.value)
    assert message.startswith(name + ':')
    for fragment in fragments:
        assert fragment in message


@pytest.mark.parametrize('kwargs,fragments', [
    ({'segments': 0}, ['segments', '0', '64']),
    ({'segments': 1}, ['segments', '1', '64']),
    ({'overlap': 1}, ['overlap', '1', '64']),
    ({'window': 'wrong'}, ['window', 'wrong', 'hann']),
])
def test_coherence_validation_context(kwargs, fragments):
    with pytest.raises(ValueError) as caught:
        M.coherence(trace(), trace(), **(dict(tclk=1, n_samples=64) | kwargs))
    message = str(caught.value)
    assert message.startswith('coherence:')
    for fragment in fragments:
        assert fragment in message


@pytest.mark.parametrize('kwargs,fragments', [
    ({'symbol_period': 0}, ['symbol_period', '0', '64']),
    ({'tstart': 64}, ['0 symbols', '64']),
    ({'n_symbols': 0}, ['n_symbols', '0', '8']),
    ({'n_symbols': 9}, ['9 symbols', '8', '64']),
    ({'tstart': 60.000001, 'n_symbols': 1}, ['window', '64']),
])
def test_evm_window_validation_context(kwargs, fragments):
    with pytest.raises(ValueError) as caught:
        rf.evm_constellation(trace(), trace(), **(
            dict(symbol_period=8, ideal_points=[-1, 1]) | kwargs))
    message = str(caught.value)
    assert message.startswith('evm_constellation:')
    for fragment in fragments:
        assert fragment in message


@pytest.mark.parametrize('name', ['evm_constellation', 'coherence'])
def test_pair_axis_length_mismatch_has_context(name):
    short = Signal('short', 'voltage', 'time', 'V',
                   np.arange(5, dtype=float), np.ones(5))
    kwargs = (dict(symbol_period=8, ideal_points=[-1, 1])
              if name == 'evm_constellation' else dict(tclk=1))
    with pytest.raises(ValueError) as caught:
        getattr(M, name)(trace(), short, **kwargs)
    message = str(caught.value)
    for fragment in [name + ':', 'same time axis', '65 points', '5 points']:
        assert fragment in message


def test_evm_consumed_nan_preserves_deferred_assignment_policy():
    i = trace(np.ones(65))
    i.y[4] = np.nan
    result = rf.evm_constellation(i, trace(np.zeros(65)), symbol_period=8,
                                ideal_points=[-1, 1])
    assert math.isnan(result.evm_percent)
    assert np.isnan(result.received[0]) and np.isnan(result.errors[0])
    assert result.assigned_ideals[0] == -1
    np.testing.assert_array_equal(result.assigned_ideals[1:], 1)
    assert result.received.shape == result.errors.shape == (8,)


@pytest.mark.parametrize('policy', ['ignore', 'error'])
def test_harmonic_warning_policy(policy):
    t = np.arange(8, dtype=float)
    sig = Signal('nyquist', 'voltage', 'time', 'V', t, (-1.) ** t)
    with warnings.catch_warnings():
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning, match='unattributable_harmonic'):
                M.harmonic_levels_db(sig, tclk=1, n_samples=8, n_harmonics=2)
        else:
            levels = M.harmonic_levels_db(sig, tclk=1, n_samples=8, n_harmonics=2)
            assert levels.shape == (1,) and np.isnan(levels[0])


def test_evm_before_trace_guard_also_checks_q_support():
    i = trace(np.ones(65))
    q = Signal('q', 'voltage', 'time', 'V', i.x + 1e-9, np.zeros(65))
    # allclose accepts these axes, but interpolation must not clamp the first
    # Q symbol: its requested center is zero and Q starts later.
    with pytest.raises(ValueError, match='evm_constellation:.*before.*q_sig'):
        rf.evm_constellation(i, q, symbol_period=8, ideal_points=[-1, 1],
                            tstart=-4, n_symbols=1)
