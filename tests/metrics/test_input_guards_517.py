"""Raw-component contamination and scale-aware paired grids (TASK-517)."""
import dataclasses
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics import _guards as G
from silk.metrics import spectral as S
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


BAD = [complex(np.inf, np.nan), complex(np.nan, np.inf)]


def signal(y, x=None, axis='time', kind='voltage', unit='V'):
    y = np.asarray(y)
    return Signal('s', kind, axis, unit,
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x), y)


def captured(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = call()
    return result, [w.message for w in caught if isinstance(w.message, MetricWarning)]


def assert_same_result(actual, expected):
    """Compare public result fields, including NaN sentinels, without recomputing."""
    if isinstance(expected, Signal):
        np.testing.assert_equal(actual.x, expected.x)
        np.testing.assert_equal(actual.y, expected.y)
    elif dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            assert_same_result(getattr(actual, field.name), getattr(expected, field.name))
    else:
        np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize('bad', BAD + [np.nan])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('interpolate', [False, True])
def test_rms_raw_complex_nan_warns(bad, reverse, interpolate):
    x, y = np.arange(3.), np.array([1, bad, 1])
    if reverse:
        x, y = x[::-1], y[::-1]
    result, diagnostics = captured(lambda: M.rms(signal(y, x), interpolate=interpolate))
    assert np.isnan(result)
    assert [w.code for w in diagnostics] == ['nan_waveform']
    assert diagnostics[0].details['nan_sample_count'] == 1


@pytest.mark.parametrize('interpolate', [False, True])
def test_rms_nan_window_support(interpolate):
    s = signal([1, 1, BAD[0], 1])
    value, diag = captured(lambda: M.rms(s, start=0, end=1, interpolate=interpolate))
    assert value == 1 and not diag
    value, diag = captured(lambda: M.rms(s, start=0, end=1.5, interpolate=interpolate))
    assert np.isnan(value) if interpolate else value == 1
    assert bool(diag) == interpolate


def test_rms_magnitude_before_window_interpolation():
    # Unit magnitudes stay one; Cartesian interpolation would shrink the edge.
    assert M.rms(signal([1+0j, 1j, -1+0j]), start=.5, end=1.5) == 1
    value, diag = captured(lambda: M.rms(signal([1+0j, complex(np.inf, 0), 1])))
    assert value == np.inf and not diag


@pytest.mark.parametrize('name', ['propagation_delay', 'propagation_delays', 'hysteresis'])
@pytest.mark.parametrize('bad', BAD)
def test_delay_raw_complex_nan(name, bad):
    s = signal([0, bad, 1, 0, 1])
    call = (lambda: M.hysteresis(s, s, s, s, out_threshold=.5)) if name == 'hysteresis' else (
        lambda: getattr(M, name)(s, s, in_threshold=.5, out_threshold=.5))
    result, diag = captured(call)
    assert np.isnan(result).all()
    assert any(w.code == 'nan_waveform' for w in diag)


@pytest.mark.parametrize('bad', BAD)
def test_delay_completed_before_nan_tail(bad):
    s = signal([0, 1, 0, bad])
    value, diag = captured(lambda: M.propagation_delay(s, s, in_threshold=.5, out_threshold=.5))
    assert value == 0 and not diag


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('bad', BAD)
def test_q_peak_checks_entire_raw_sweep(reverse, bad):
    x, y = np.arange(1., 5.), np.array([1, bad, complex(np.inf, 0), 1])
    if reverse:
        x, y = x[::-1], y[::-1]
    value, diag = captured(lambda: M.q_from_peak(signal(y, x, axis='frequency')))
    assert np.isnan(value)
    assert [w.code for w in diag] == ['nan_magnitude']


TIMING = ('crossings average_period average_frequency cycle_period cycle_frequency '
          'period_jitter period_jitter_pp cycle_to_cycle_jitter cycle_to_cycle_jitter_pp '
          'period_deviation cycle_to_cycle_difference tie long_term_jitter allan_deviation '
          'average_duty cycle_duty pulse_width pulse_widths on_time off_time histogram').split()


@pytest.mark.parametrize('name', TIMING)
def test_magnitude_consumers_keep_raw_nan(name):
    s = signal([0, BAD[0], 1, 0, 1, 0, 1, 0, 1])
    kwargs = {'bins': 3} if name == 'histogram' else {'threshold': .5}
    value, diag = captured(lambda: getattr(M, name)(s, **kwargs))
    values = value.y if isinstance(value, Signal) else (
        value.fractions if name == 'histogram' else value)
    assert np.isnan(values).all()
    assert any(w.code == 'nan_waveform' for w in diag)


@pytest.mark.parametrize('name', ['stopband_attenuation_db', 'psrr_db'])
@pytest.mark.parametrize('bad', BAD)
def test_ac_magnitude_consumers_keep_raw_nan(name, bad):
    s = signal([1, bad, 1], [1, 2, 3], 'frequency')
    call = (lambda: M.psrr_db(s, signal([.1]*3, s.x, 'frequency'))) if name == 'psrr_db' else (
        lambda: M.stopband_attenuation_db(s, 2))
    result, diag = captured(call)
    assert np.isnan(result.y[1] if isinstance(result, Signal) else result)
    assert [w.code for w in diag] == ['nan_magnitude']


SPECTRAL = ('fft sndr sinad sfdr sfdr_db enob thd thd_percent noise_floor_dbfs '
            'harmonic_levels_db harmonic_level_dbc imd3_db oip3_dbm iip3_dbm psd '
            'spectral_power harmonic mix_product snr harmonic_freqs welch_psd coherence').split()


def spectral_call(name, s):
    kw = dict(tclk=1, n_samples=64)
    args = [s]
    if name in ('imd3_db', 'oip3_dbm', 'iip3_dbm', 'mix_product'):
        kw.update(f1_hz=.125, f2_hz=.1875)
    if name in ('iip3_dbm', 'coherence'):
        args.append(signal(np.cos(np.arange(65)*np.pi/4)))
    if name == 'harmonic_level_dbc':
        kw['n'] = 2
    if name == 'harmonic':
        args.append(2)
    if name == 'mix_product':
        args.extend([2, -1])
    if name == 'spectral_power':
        args.extend([0, .5])
    return getattr(M, name)(*args, **kw)


@pytest.mark.parametrize('name', SPECTRAL)
def test_spectral_sampler_preserves_raw_nan(name):
    real = np.cos(np.arange(65)*np.pi/4)
    dirty = real.astype(complex)
    real[8], dirty[8] = np.nan, complex(1, np.nan)
    expected, warnings_real = captured(lambda: spectral_call(name, signal(real)))
    result, warnings_complex = captured(lambda: spectral_call(name, signal(dirty)))
    assert_same_result(result, expected)
    assert [w.code for w in warnings_complex] == [w.code for w in warnings_real]


def test_spectral_sampler_unconsumed_and_finite_complex():
    s = signal([1+2j]*5)
    s.y[-1] = BAD[0]
    result, diag = captured(lambda: S._sample_at_clock(s, metric='fft', tclk=1, n_samples=4))
    np.testing.assert_equal(result, [1]*4)
    assert not diag


def test_evm_explicit_ideals_preserves_nan_error_norm():
    i = signal([1+0j, complex(np.inf, np.nan), 1+0j])
    q = signal([0., 0., 0.])
    result, diag = captured(lambda: M.evm_constellation(
        i, q, symbol_period=2, n_symbols=1, ideal_points=[-1, 1]))
    assert np.isnan(result.evm_percent)
    assert result.received.shape == result.errors.shape == result.assigned_ideals.shape == (1,)
    assert np.isnan(result.received[0]) and np.isnan(result.errors[0])
    assert [w.code for w in diag] == ['nan_symbols']


DC = ('inl dnl static_errors inl_max_lsb dnl_max_lsb offset_error_lsb '
      'gain_error_lsb full_scale_error_lsb missing_codes missing_code_count '
      'is_monotonic non_monotonic_steps p1db').split()


def dc_call(name, y):
    kw = dict(full_scale=4, n_bits=2) if name in (
        'static_errors', 'offset_error_lsb', 'gain_error_lsb', 'full_scale_error_lsb') else {}
    return getattr(M, name)(signal(y, axis='power' if name == 'p1db' else 'code'), **kw)


@pytest.mark.parametrize('name', DC)
def test_dc_real_casts_preserve_raw_nan_and_warning_conventions(name):
    expected, real_diag = captured(lambda: dc_call(name, [0., np.nan, 2., 3.]))
    result, complex_diag = captured(lambda: dc_call(name, [0., complex(1, np.nan), 2., 3.]))
    assert_same_result(result, expected)
    assert [w.code for w in complex_diag] == [w.code for w in real_diag]


def test_dc_code_nan_transition_table_remains_quiet():
    result, diag = captured(lambda: M.extract_code_transitions(
        signal([0., 1., 2.]), signal([0., complex(1, np.nan), 2.])))
    np.testing.assert_equal(result.y, [np.nan, np.nan])
    assert np.isnan(result.code_reversals)
    assert not diag


@pytest.mark.parametrize('bad', BAD + [np.nan, complex(1, np.nan)])
def test_adc_ramp_selected_raw_nan(bad):
    result, diag = captured(lambda: M.extract_code_transitions(
        signal([0, bad, 2]), signal([0, 1, 2])))
    np.testing.assert_equal(result.y, [np.nan, 2])
    assert [w.code for w in diag] == ['nan_transition_voltage']
    assert diag[0].details['sample_index'] == 1


def test_adc_ramp_unselected_nan_is_not_consumed():
    result, diag = captured(lambda: M.extract_code_transitions(
        signal([BAD[0], 1, 2]), signal([0, 1, 2])))
    np.testing.assert_equal(result.y, [1, 2])
    assert not diag


def test_adc_ramp_finite_complex_control():
    result, diag = captured(lambda: M.extract_code_transitions(
        signal([0+3j, 1+3j, 2+3j]), signal([0, 1, 2])))
    np.testing.assert_equal(result.y, [1, 2])
    assert not diag


def hysteresis_inputs(rising, falling, exact=False):
    up, down = ([0, .5, 1, 1], [1, .5, 0, 0]) if exact else ([0, 1, 1, 1], [1, 0, 0, 0])
    return M.hysteresis(signal(rising), signal(up), signal(falling), signal(down), out_threshold=.5)


@pytest.mark.parametrize('role', ['rising', 'falling', 'both'])
@pytest.mark.parametrize('exact,bad_index', [(True, 1), (False, 0), (False, 1)])
@pytest.mark.parametrize('bad', [np.nan, complex(1, np.nan)])
def test_hysteresis_input_raw_nan_support(role, exact, bad_index, bad):
    r, f = np.arange(4, dtype=complex), np.arange(4, dtype=complex)
    if role in ('rising', 'both'):
        r[bad_index] = bad
    if role in ('falling', 'both'):
        f[bad_index] = bad
    value, diag = captured(lambda: hysteresis_inputs(r, f, exact))
    assert np.isnan(value)
    assert {w.details['signal_role'] for w in diag} == (
        {'sig_in_rising', 'sig_in_falling'} if role == 'both' else {f'sig_in_{role}'})
    assert all(w.details['purpose'] == 'decision_interpolation' for w in diag)


@pytest.mark.parametrize('exact,bad_index', [(False, 3), (True, 0), (True, 2), (True, 3)])
def test_hysteresis_input_unconsumed_nan(exact, bad_index):
    r = np.arange(4, dtype=complex)
    r[bad_index] = complex(1, np.nan)
    value, diag = captured(lambda: hysteresis_inputs(r, r.copy(), exact))
    assert value == 0 and not diag


def test_hysteresis_input_finite_complex_control():
    r = np.arange(4)+2j
    value, diag = captured(lambda: hysteresis_inputs(r, r+1))
    assert value == -1 and not diag


PAIRED = ['extract_code_transitions', 'psrr_db', 'evm_constellation', 'evm_percent',
          'coherence', 'average_power', 'dynamic_energy_per_cycle', 'hysteresis_rising', 'hysteresis_falling']


def paired_call(name, a, b):
    n = len(a)
    if name == 'extract_code_transitions':
        return M.extract_code_transitions(signal(np.arange(n), a), signal(np.arange(n), b))
    if name == 'psrr_db':
        return M.psrr_db(signal(np.ones(n), a, 'frequency'), signal(np.ones(n), b, 'frequency'))
    if name.startswith('evm'):
        return getattr(M, name)(signal(np.ones(n), a), signal(np.zeros(n), b),
                               symbol_period=2*(a[1]-a[0]), tstart=a[0], n_symbols=1, modulation='qpsk')
    if name == 'coherence':
        return M.coherence(signal(np.arange(n), a), signal(np.arange(n), b),
                           tclk=a[1]-a[0], tstart=a[0], n_samples=6, segments=2)
    if name in ('average_power', 'dynamic_energy_per_cycle'):
        kw = {'period': 2*(a[1]-a[0])} if name == 'dynamic_energy_per_cycle' else {}
        return getattr(M, name)(signal(np.ones(n), a), signal(np.ones(n), b, kind='current', unit='A'), **kw)
    up, down = np.r_[0., np.ones(n-1)], np.r_[1., np.zeros(n-1)]
    return M.hysteresis(signal(np.arange(n), a), signal(up, b if name.endswith('rising') else a),
                        signal(np.arange(n), a), signal(down, b if name.endswith('falling') else a), out_threshold=.5)


@pytest.mark.parametrize('name', PAIRED)
@pytest.mark.parametrize('scale,origin', [(1e-9, 0), (1., 0), (1., 1e9)])
def test_paired_grid_mismatch(name, scale, origin):
    a = origin+np.arange(6.)*scale
    b = origin+np.array([0, 2, 3, 4, 4.5, 5])*scale
    with pytest.raises(ValueError, match='share.*axis'):
        paired_call(name, a, b)


@pytest.mark.parametrize('name', PAIRED)
@pytest.mark.parametrize('offset', [0., 1e-6])
def test_paired_same_capture_passes(name, offset):
    a = np.arange(6.)*1e-9
    b = a.copy()
    b[2] += offset*1e-9
    result, _ = captured(lambda: paired_call(name, a, b))
    assert result is not None


@pytest.mark.parametrize('reverse', [False, True])
def test_psrr_matching_orientation(reverse):
    a = np.array([1., 2., 4.])
    if reverse:
        a = a[::-1]
    np.testing.assert_equal(paired_call('psrr_db', a, a).y, [0]*3)
    with pytest.raises(ValueError, match='same frequency axis'):
        paired_call('psrr_db', a, a[::-1])


@pytest.mark.parametrize('a,b,expected', [
    ([0, 1, 2], [0, 1.0005, 2], True),
    ([0, 1, 2], [0, 1.002, 2], False),
    ([2, 1, 0], [2, 1.0005, 0], True),
    ([0, 1, 1.1], [0, 1.0005, 1.1], False),
    ([0, 1], [0, 1, 2], False),
    ([1], [1], True), ([1], [1+1e-12], False),
])
@pytest.mark.parametrize('scale', [1e-12, 1., 1e12])
def test_shared_axis_same_grid(a, b, expected, scale):
    aa, bb = np.asarray(a)*scale, np.asarray(b)*scale
    saved = aa.copy()
    assert G._Axis(aa, 'test').same_grid(G._Axis(bb, 'test')) is expected
    np.testing.assert_equal(aa, saved)


@pytest.mark.parametrize('shape', [(), (2,), (2, 2)])
@pytest.mark.parametrize('bad', [1+2j, *BAD])
def test_nan_safe_transform_preserves_shape_and_fresh_storage(shape, bad):
    values = np.full(shape, bad, dtype=complex)
    saved = values.copy()
    result = G._nan_safe_transform(values, np.abs)
    assert result.shape == values.shape
    assert not np.shares_memory(values, result)
    np.testing.assert_equal(result, np.full(shape, np.nan if np.isnan(bad) else np.sqrt(5)))
    result[...] = 17
    np.testing.assert_equal(values, saved)


def test_nan_safe_transform_copies_identity_output():
    values = np.array([1., np.nan])
    result = G._nan_safe_transform(values, lambda x: x)
    result[0] = 9
    np.testing.assert_equal(values, [1, np.nan])


@pytest.mark.parametrize('reverse', [False, True])
def test_band_noise_figure_requires_exact_capture_grid(reverse):
    x = np.array([1., 2., 3.])
    if reverse:
        x = x[::-1]
    a = signal(np.full(3, 1e-8), x, 'frequency', 'noise', 'V/sqrt(Hz)')
    b = signal(np.full(3, 2e-8), x.copy(), 'frequency', 'noise', 'V/sqrt(Hz)')
    assert np.isfinite(M.band_noise_figure(a, b, 1, 3, 1000, 290))
    b.x[1] = np.nextafter(b.x[1], np.inf)
    with pytest.raises(ValueError, match='same .noise run'):
        M.band_noise_figure(a, b, 1, 3, 1000, 290)


def test_integrated_noise_real_nan_convention():
    s = signal([1., np.nan, 1.], [1, 2, 3], 'frequency', 'noise', 'V/sqrt(Hz)')
    value, diag = captured(lambda: M.integrated_noise(s, 1, 3))
    assert np.isnan(value) and [w.code for w in diag] == ['nan_density']
    with pytest.raises(ValueError):
        M.integrated_noise(signal([1+0j]*3, [1, 2, 3], 'frequency'), 1, 3)


@pytest.mark.parametrize('other,expected', [
    ([-1e308, 1e308], True),
    ([-9e307, 9e307], False),
    ([1e308, -1e308], False),
])
def test_extreme_finite_shared_grid(other, expected):
    with np.errstate(over='ignore'):
        a = G._Axis([-1e308, 1e308], 'test')
        b = G._Axis(other, 'test')
        assert a.same_grid(b) is expected
        assert b.same_grid(a) is expected


@pytest.mark.parametrize('other,expected', [
    ([-1e308, 1e308], True),
    ([-9e307, 9e307], False),
    ([1e308, -1e308], False),
])
def test_extreme_finite_adc_grid(other, expected):
    with np.errstate(over='ignore'):
        if expected:
            result = paired_call('extract_code_transitions', [-1e308, 1e308], other)
            np.testing.assert_equal(result.y, [1])
        else:
            message = 'strictly increasing axis' if other[0] > other[1] else 'share.*axis'
            with pytest.raises(ValueError, match=message):
                paired_call('extract_code_transitions', [-1e308, 1e308], other)


@pytest.mark.parametrize('name', ['overshoot', 'undershoot'])
@pytest.mark.parametrize('falling', [False, True])
@pytest.mark.parametrize('imaginary_nan', [False, True])
@pytest.mark.parametrize('bad_index', [1, 2])
def test_step_extrema_consumed_nan(name, falling, imaginary_nan, bad_index):
    # Both samples belong to overshoot's whole trace / undershoot's post-target
    # interval (the target is reached at index zero for undershoot).
    values = [1., 1.25, .5, 1.]
    if falling:
        values = [1-y for y in values]
    values[bad_index] = complex(values[bad_index], np.nan) if imaginary_nan else np.nan
    result, diag = captured(lambda: getattr(M, name)(
        signal(values), initial=int(falling), target=int(not falling)))
    assert np.isnan(result)
    assert [w.code for w in diag] == ['nan_waveform']
    assert diag[0].details['nan_sample_count'] == 1


@pytest.mark.parametrize('falling', [False, True])
@pytest.mark.parametrize('imaginary_nan', [False, True])
def test_undershoot_pre_target_nan_keeps_result_and_warning(falling, imaginary_nan):
    values = [0., 1., .5, 1.]
    if falling:
        values = [1-y for y in values]
    values[0] = complex(values[0], np.nan) if imaginary_nan else np.nan
    result, diag = captured(lambda: M.undershoot(
        signal(values), initial=int(falling), target=int(not falling)))
    assert result == .5
    assert [w.code for w in diag] == ['nan_waveform']


@pytest.mark.parametrize('name,expected', [('overshoot', .25), ('undershoot', .5)])
@pytest.mark.parametrize('falling', [False, True])
def test_step_extrema_finite_complex_control(name, expected, falling):
    values = np.array([0., 1.25, .5, 1.])
    if falling:
        values = 1-values
    # Preserve the existing real projection of a complex extremum, not magnitude.
    values = values + 2j
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = getattr(M, name)(signal(values), initial=int(falling),
                                  target=int(not falling))
    assert result == expected
    assert [w.category.__name__ for w in caught] == ['ComplexWarning']
