"""Warnings explain spectral refusals without changing numeric contracts."""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics import rf
from silk.metrics.diagnostics import MetricWarning
from silk.metrics.spectral import _sample_at_clock
from silk.signals.base import Signal


NAMES = [
    'fft', 'sndr', 'sinad', 'enob', 'snr', 'sfdr', 'thd', 'thd_percent',
    'noise_floor_dbfs', 'harmonic_levels_db', 'imd3_db', 'psd',
    'spectral_power', 'harmonic', 'mix_product', 'harmonic_freqs',
    'welch_psd', 'coherence', 'evm_constellation',
]


def trace(values=None):
    t = np.arange(65 if values is None else len(values), dtype=float)
    if values is None:
        values = (np.cos(2 * np.pi * 3 * t / 64)
                  + .01 * np.random.default_rng(270).normal(size=len(t)))
    return Signal('selected', 'voltage', 'time', 'V', t, np.asarray(values))


def measure(name, sig, **overrides):
    kwargs = dict(tclk=1, n_samples=64)
    args = [sig]
    if name == 'imd3_db':
        kwargs.update(f1_hz=3/64, f2_hz=5/64)
    elif name == 'spectral_power':
        kwargs.update(f1_hz=0, f2_hz=.5)
    elif name == 'harmonic':
        kwargs.update(k=2)
    elif name == 'mix_product':
        kwargs.update(m=2, n=-1, f1_hz=3/64, f2_hz=5/64)
    elif name in ('welch_psd', 'coherence'):
        kwargs.update(segments=4)
        if name == 'coherence':
            args.append(trace())
    elif name == 'evm_constellation':
        args.append(trace(np.zeros(65)))
        kwargs = dict(symbol_period=8, ideal_points=[-1, 1])
    kwargs.update(overrides)
    return getattr(M, name)(*args, **kwargs)


def result_values(result):
    if isinstance(result, Signal):
        return result.y
    if isinstance(result, M.SFDR):
        return [result.sfdr_db, result.fundamental_hz, result.spur_hz]
    if isinstance(result, rf.EVMResult):
        return np.r_[result.evm_percent, result.received, result.errors]
    return np.asarray(result)


@pytest.mark.parametrize('name', NAMES)
def test_consumed_nan_explained_once_with_origin(name):
    sig = trace()
    sig.y[12] = np.nan  # Also an EVM symbol centre and a used Welch sample.
    with pytest.warns(MetricWarning) as caught:
        result = measure(name, sig)
    assert len(caught) == 1
    warning = caught[0].message
    origin = {'sinad': 'sndr', 'enob': 'sndr', 'spectral_power': 'psd'}.get(name, name)
    assert warning.metric == origin
    assert 'NaNs' in warning.message
    assert 'source result' in warning.suggestion
    assert np.any(np.isnan(result_values(result)))
    assert all(type(v) in (int, float, str, type(None)) for v in warning.details.values())
    if name == 'imd3_db':
        # TASK-260's existing explanation is the only explanation here.
        assert warning.code == 'nan_consumed_bins'
    else:
        prefix = 'a_' if name == 'coherence' else 'i_' if name == 'evm_constellation' else ''
        details = warning.details
        assert details[prefix + 'nan_sample_count'] == 1
        assert details[prefix + 'sample_unit'] == 'V'
        expected_count = 61 if name in ('coherence', 'welch_psd') else 8 if name == 'evm_constellation' else 64
        assert details[prefix + 'sample_count'] == expected_count
        assert details[prefix + 'sample_start_s'] == (4 if name == 'evm_constellation' else 0)
        assert details[prefix + 'sample_end_s'] == (60 if expected_count in (8, 61) else 63)
    if name == 'sfdr':
        assert result.bin_hz == 1/64
    elif name == 'harmonic_freqs':
        assert result.shape == (1,)
    elif name == 'harmonic_levels_db':
        assert result.shape == (9,)
    elif name == 'evm_constellation':
        assert result.assigned_ideals[1] == -1  # Deferred assignment policy.
        assert result.received.shape == result.errors.shape == (8,)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('unused_nan', [False, True])
def test_success_and_unconsumed_nan_are_quiet(name, unused_nan):
    clean = trace()
    sig = trace()
    if unused_nan:
        sig.y[-1] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = measure(name, sig)
        expected = measure(name, clean)
    assert not caught
    np.testing.assert_array_equal(result_values(actual), result_values(expected))
    assert np.all(np.isfinite(result_values(actual)))


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('policy', ['ignore', 'error'])
def test_warning_policy(name, policy):
    sig = trace()
    sig.y[12] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning):
                measure(name, sig)
        else:
            assert np.any(np.isnan(result_values(measure(name, sig))))


@pytest.mark.parametrize('name', ['thd', 'thd_percent'])
def test_empty_harmonic_selection(name):
    sig = trace((-1.) ** np.arange(65))
    with pytest.warns(MetricWarning) as caught:
        result = measure(name, sig)
    assert len(caught) == 1 and math.isnan(result)
    warning = caught[0].message
    assert warning.metric == name and warning.code == 'no_harmonic_bins'
    assert warning.details['selected_bin_count'] == 0
    assert warning.details['fundamental_bin'] == 32
    assert warning.details['bin_hz'] == 1/64


def test_empty_noise_selection():
    with pytest.warns(MetricWarning) as caught:
        result = measure('snr', trace([0., 1., 0., -1., 0.]), n_samples=4)
    assert len(caught) == 1 and math.isnan(result)
    assert caught[0].message.code == 'no_residual_bins'
    assert caught[0].message.details['selected_bin_count'] == 0


def test_zero_imd_reference():
    with pytest.warns(MetricWarning) as caught:
        result = measure('imd3_db', trace(np.zeros(65)))
    assert len(caught) == 1 and math.isnan(result)
    warning = caught[0].message
    assert warning.code == 'zero_reference_power'
    assert warning.details['reference_power'] == 0
    assert warning.details['sample_count'] == 64
    assert warning.details['f1_hz'] == 3/64


@pytest.mark.parametrize('name', ['harmonic', 'harmonic_freqs', 'harmonic_levels_db'])
def test_unavailable_harmonic_location(name):
    sig = trace((-1.) ** np.arange(65))
    kwargs = {'harmonic': dict(k=2), 'harmonic_freqs': dict(n=2, observed=True),
              'harmonic_levels_db': dict(n_harmonics=2)}[name]
    with pytest.warns(MetricWarning) as caught:
        result = measure(name, sig, **kwargs)
    assert len(caught) == 1
    warning = caught[0].message
    assert warning.metric == name and warning.code == 'unattributable_harmonic'
    assert warning.details['folded_bin'] == 0
    if name == 'harmonic':
        assert math.isnan(result.real) and result.imag == 0
    else:
        assert result.shape == ((2,) if name == 'harmonic_freqs' else (1,))
        assert math.isnan(result[-1])


@pytest.mark.parametrize('name', ['welch_psd', 'coherence'])
@pytest.mark.parametrize('index', [60, 61, 62, 63])
def test_welch_uses_only_segment_support(name, index):
    sig = trace()
    sig.y[index] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        actual = measure(name, sig)
    if index == 60:
        assert len(caught) == 1
        assert np.all(np.isnan(actual.y))
    else:
        assert not caught
        np.testing.assert_array_equal(actual.y, measure(name, trace()).y)


def test_coherence_zero_segment_power():
    zero = trace(np.zeros(65))
    with pytest.warns(MetricWarning) as caught:
        result = M.coherence(zero, zero, tclk=1, n_samples=64, segments=4)
    assert len(caught) == 1 and np.all(np.isnan(result.y))
    warning = caught[0].message
    assert warning.code == 'zero_segment_power'
    assert warning.details['zero_power_bin_count'] == 13
    assert warning.details['a_sample_count'] == warning.details['b_sample_count'] == 61
    assert warning.details['segment_count'] == 4


@pytest.mark.parametrize('name', ['sndr', 'sinad', 'enob', 'snr', 'sfdr', 'fft', 'psd', 'welch_psd', 'coherence'])
def test_finite_sample_overflow_is_not_reported_as_nan_input(name):
    sig = trace()
    sig.y *= 1e200
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.warns(MetricWarning) as caught:
            result = measure(name, sig)
    assert len(caught) == 1
    warning = caught[0].message
    assert warning.code == 'nonfinite_spectrum'
    assert 'NaNs' not in warning.message
    assert 'overflow' in warning.suggestion
    assert not np.all(np.isfinite(result_values(result)))
    prefix = 'a_' if name == 'coherence' else ''
    assert warning.details[prefix + 'nan_sample_count'] == 0
    assert warning.details[prefix + 'inf_sample_count'] == 0


@pytest.mark.parametrize('name,kwargs', [
    ('psd', {'window': 'bad'}), ('welch_psd', {'segments': 0}), ('coherence', {'segments': 1}),
    ('harmonic', {'k': 0}), ('harmonic_freqs', {'n': 0}),
    ('thd', {'n_harmonics': 1}), ('snr', {'n_samples': 2}),
    ('imd3_db', {'f1_hz': 2}),
    ('evm_constellation', {'ideal_points': [0, 1]}),
])
def test_bad_request_with_contaminated_data_retains_caller_error(name, kwargs):
    sig = trace()
    sig.y[12] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match=name + ':'):
            measure(name, sig, **kwargs)
    assert not caught


@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_contaminated_psd_precedes_spectral_power_band_validation(policy):
    sig = trace()
    sig.y[12] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        error = MetricWarning if policy == 'error' else ValueError
        with pytest.raises(error, match='psd:' if policy == 'error' else '0 of 33 FFT bins'):
            measure('spectral_power', sig, f1_hz=1, f2_hz=2)
    assert len(caught) == (1 if policy == 'always' else 0)


def test_sample_at_clock_does_not_inspect_skipped_simulator_points():
    sig = trace()
    sig.y[1::2] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = M.sndr(sig, tclk=2, n_samples=32)
    assert math.isfinite(value) and not caught


@pytest.mark.parametrize('index,contaminates', [(11, True), (12, True), (10, False), (13, False)])
def test_evm_checks_only_interpolated_symbols(index, contaminates):
    sig = trace(np.ones(65))
    sig.y[index] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = measure('evm_constellation', sig, tstart=7.5, n_symbols=1)
    assert len(caught) == int(contaminates)
    assert math.isnan(result.evm_percent) if contaminates else result.evm_percent == 0
    if contaminates:
        warning = caught[0].message
        assert warning.details['i_sample_start_s'] == 11.5
        assert warning.details['i_sample_count'] == 1


def test_dead_waveform_metadata_and_zero_coefficients_remain_quiet():
    sig = trace(np.zeros(65))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = measure('sfdr', sig)
        assert result.sfdr_db == 0
        assert result.fundamental_hz == 1/64 and result.spur_hz == 2/64
        assert measure('harmonic', sig) == 0j
        np.testing.assert_array_equal(measure('harmonic_freqs', sig), np.arange(1, 11)/64)
    assert not caught


@pytest.mark.parametrize('name', ['thd', 'thd_percent', 'noise_floor_dbfs',
                                 'harmonic_levels_db', 'imd3_db'])
def test_overflowing_ratio_operands_get_one_factual_warning(name):
    sig = trace()
    sig.y *= 1e200
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.warns(MetricWarning) as caught:
            result = measure(name, sig)
    assert len(caught) == 1
    assert caught[0].message.code == 'nonfinite_spectrum'
    assert 'consumed samples are finite' in caught[0].message.message
    assert np.any(np.isnan(result_values(result)))


@pytest.mark.parametrize('name', ['sndr', 'fft', 'harmonic_freqs'])
def test_all_nan_sample_range_is_explicit(name):
    with pytest.warns(MetricWarning) as caught:
        measure(name, trace(np.full(65, np.nan)))
    assert len(caught) == 1
    details = caught[0].message.details
    assert details['sample_min'] is details['sample_max'] is None
    assert details['nan_sample_count'] == details['sample_count'] == 64


def test_harmonic_fundamental_nan_coefficient():
    sig = trace()
    sig.y[12] = np.nan
    with pytest.warns(MetricWarning) as caught:
        result = measure('harmonic', sig, k=1)
    assert len(caught) == 1 and math.isnan(result.real) and math.isnan(result.imag)
    assert caught[0].message.metric == 'harmonic'


@pytest.mark.parametrize('name', ['coherence', 'evm_constellation'])
@pytest.mark.parametrize('both', [False, True])
def test_pair_contamination_in_second_or_both_inputs_is_explained_once(name, both):
    first, second = trace(), trace()
    second.y[12] = np.nan
    if both:
        first.y[12] = np.nan
    with pytest.warns(MetricWarning) as caught:
        if name == 'coherence':
            result = M.coherence(first, second, tclk=1, n_samples=64, segments=4)
        else:
            result = rf.evm_constellation(first, second, symbol_period=8,
                                         ideal_points=[-1, 1])
    assert len(caught) == 1 and np.any(np.isnan(result_values(result)))
    prefix = 'b_' if name == 'coherence' else 'q_'
    assert caught[0].message.details[prefix + 'nan_sample_count'] == 1


def test_consumed_infinity_is_not_described_as_nan_input():
    sig = trace()
    sig.y[12] = np.inf
    with np.errstate(invalid='ignore'):
        with pytest.warns(MetricWarning) as caught:
            result = measure('sndr', sig)
    assert len(caught) == 1 and math.isnan(result)
    warning = caught[0].message
    assert warning.code == 'nonfinite_spectrum'
    assert warning.details['nan_sample_count'] == 0
    assert warning.details['inf_sample_count'] == 1
    assert 'contains NaNs' not in warning.message


def test_exact_zero_imd_product_keeps_minus_infinity_without_warning():
    # These bit-exact periodic values give power at bins 8 and 24, while the
    # requested product at bin 16 is exactly zero.
    sig = trace(np.resize([1., 0., -1., 0., -1., 0., 1., 0.], 65))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = measure('imd3_db', sig, f1_hz=8/64, f2_hz=12/64)
    assert value == -math.inf and not caught


def test_interpolation_summary_counts_and_absolute_distance():
    sig = Signal('ramp', 'voltage', 'time', 'V', np.arange(6.), np.arange(6.))
    with pytest.warns(MetricWarning) as caught:
        M.fft(sig, tclk=1.25, tstart=.25, n_samples=4)
    diagnostic, = [w.message for w in caught]
    assert diagnostic.metric == 'fft'
    assert diagnostic.code == 'interpolated_samples'
    assert diagnostic.details == dict(signal_name='ramp', sample_count=4,
        interpolated_sample_count=3, max_nearest_time_distance_s=.5,
        endpoint_clamped_sample_count=0)
    assert 'resolution' in diagnostic.suggestion
    assert 'roundoff' in diagnostic.suggestion.lower()


@pytest.mark.parametrize('policy', ['ignore', 'error'])
def test_interpolation_warning_policy(policy):
    with warnings.catch_warnings():
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning, match='interpolated_samples'):
                M.fft(trace(), tclk=1, tstart=.25, n_samples=32)
        else:
            assert len(M.fft(trace(), tclk=1, tstart=.25, n_samples=32).y) == 17


CLOCKED_NAMES = [name for name in NAMES if name != 'evm_constellation'] + [
    'sfdr_db', 'harmonic_level_dbc', 'oip3_dbm', 'iip3_dbm']


@pytest.mark.parametrize('name', CLOCKED_NAMES)
def test_all_clocked_consumers_interpolate_before_their_formulas(name):
    x = np.arange(66.)
    y = (np.cos(2*np.pi*7*x/64) + .2*np.cos(2*np.pi*11*x/64)
         + .03*np.cos(2*np.pi*3*x/64) + .02*np.cos(2*np.pi*15*x/64)
         + .01*np.cos(2*np.pi*14*x/64))
    raw = Signal('out', 'voltage', 'time', 'V', x, y)
    # Quarter-step interpolation, independently evaluated by weighted endpoints.
    expected = Signal('out', 'voltage', 'time', 'V', x[:-1]+.25,
                      .75*y[:-1] + .25*y[1:])
    raw_in = Signal('in', 'voltage', 'time', 'V', x, y/2)
    expected_in = Signal('in', 'voltage', 'time', 'V', x[:-1]+.25,
                         (.75*y[:-1] + .25*y[1:])/2)
    kw = dict(tclk=1, tstart=.25, n_samples=64)
    if name in ('imd3_db', 'oip3_dbm', 'iip3_dbm', 'mix_product'):
        kw.update(f1_hz=7/64, f2_hz=11/64)
    if name == 'mix_product':
        kw.update(m=2, n=-1)
    if name == 'spectral_power':
        kw.update(f1_hz=0, f2_hz=.5)
    if name == 'harmonic':
        kw['k'] = 2
    if name == 'harmonic_level_dbc':
        kw['n'] = 2
    if name in ('coherence', 'welch_psd'):
        kw['segments'] = 4
    args = [raw, raw_in] if name in ('coherence', 'iip3_dbm') else [raw]
    expected_args = ([expected, expected_in] if len(args) == 2 else [expected])
    with pytest.warns(MetricWarning) as caught:
        actual = getattr(M, name)(*args, **kw)
    summaries = [w.message for w in caught if w.message.code == 'interpolated_samples']
    assert len(summaries) == len(args)
    assert [w.details['signal_name'] for w in summaries] == [s.name for s in args]
    for w in summaries:
        assert w.details['sample_count'] == w.details['interpolated_sample_count'] == 64
        assert w.details['max_nearest_time_distance_s'] == .25
    with warnings.catch_warnings(record=True) as exact_warnings:
        warnings.simplefilter('always', MetricWarning)
        want = getattr(M, name)(*expected_args, **kw)
    assert not any(w.message.code == 'interpolated_samples' for w in exact_warnings)
    np.testing.assert_allclose(result_values(actual), result_values(want), rtol=1e-12,
                               atol=1e-12, equal_nan=True)
    if isinstance(actual, Signal):
        np.testing.assert_array_equal(actual.x, want.x)
        assert (actual.xlabel, actual.ylabel) == (want.xlabel, want.ylabel)


def test_endpoint_roundoff_is_reported_as_clamping_not_interpolation():
    end = np.nextafter(3., 0.)
    sig = Signal('boundary', 'voltage', 'time', 'V', [0., 1., 2., end], [0., 1., 2., 3.])
    with pytest.warns(MetricWarning) as caught:
        got = _sample_at_clock(sig, metric='fft', tclk=1, n_samples=4)
    np.testing.assert_array_equal(got, [0, 1, 2, 3])
    w, = [w.message for w in caught]
    assert w.details['interpolated_sample_count'] == 0
    assert w.details['endpoint_clamped_sample_count'] == 1
    assert w.details['max_nearest_time_distance_s'] == 3. - end
    assert 'boundary' in w.message
    with pytest.raises(ValueError, match='exceeds trace end'):
        _sample_at_clock(sig, metric='fft', tclk=1.0001, n_samples=4)


@pytest.mark.parametrize('bad', [np.nan, complex(np.inf, np.nan)])
@pytest.mark.parametrize('bad_index,consumed', [(0, False), (1, True), (2, True), (6, False)])
def test_interpolation_nan_support_remains_honest(bad, bad_index, consumed):
    y = np.arange(7., dtype=complex if np.iscomplexobj(bad) else float)
    y[bad_index] = bad
    sig = Signal('support', 'voltage', 'time', 'V', np.arange(7.), y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual = M.fft(sig, tclk=1, tstart=1.25, n_samples=4)
    codes = [w.message.code for w in caught if isinstance(w.message, MetricWarning)]
    assert codes.count('interpolated_samples') == 1
    assert ('nan_samples' in codes) == consumed
    assert bool(np.isnan(actual.y).any()) == consumed
