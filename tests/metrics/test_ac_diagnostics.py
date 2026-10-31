"""TASK-266: all 19 W-AC ledger rows, with numeric paths left intact."""
import math
import warnings

import numpy as np
import pytest

from silk.metrics import ac
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def response(y, x=None, name='response', unit='V/V'):
    if x is None:
        x = np.arange(1, len(y) + 1)
    return Signal(name, 'voltage', 'frequency', unit,
                  np.asarray(x, dtype=float), np.asarray(y, dtype=complex))


# Exactly the warning-needed inventory assigned to TASK-266. The two stopband
# rows already have their diagnostics from TASK-256; retain exactly one each.
LEDGER = [
    ('passband_ripple_db', (response([1, np.nan, 1]), 3), 'nan_magnitude'),
    ('stopband_attenuation_db', (response([0, 0, 0]), 2), 'no_passband_reference'),
    ('stopband_attenuation_db', (response([1, np.nan, 1]), 2), 'nan_magnitude'),
    ('q_from_peak', (response([0, 0, 0]),), 'no_positive_peak'),
    ('q_from_peak', (response([0, 1, 0], [1, 10, 100]),), 'half_power_span_unresolved'),
    ('q_from_peak', (response([1, np.nan, 1]),), 'nan_magnitude'),
    ('gain_bw_product', (response([1, np.nan, 1]),), 'nan_magnitude'),
    ('gain_bw_product', (response([0, 1, 1]),), 'no_dc_reference'),
    ('gain_bw_product', (response([1, 1, 1]),), 'upper_rolloff_not_observed'),
    ('dc_gain_dB', (response([np.nan, 1, 1]),), 'nan_magnitude'),
    ('peak_gain_dB', (response([1, np.nan, 1]),), 'nan_magnitude'),
    ('gain_at_freq', (response([1, np.nan, 1]), 2), 'nan_magnitude'),
    ('phase_at_freq', (response([1, complex(1, np.nan), 1]), 2), 'nan_response'),
    ('gain_margin', (response([2, 2, 2]),), 'no_phase_crossing_gain_at_least_one'),
    ('cmrr_dB', (response([0, 1, 1]), response([0, 1, 1])), 'no_differential_reference'),
    ('cmrr_dB', (response([np.nan, 1, 1]), response([1, 1, 1])), 'nan_magnitude'),
    ('group_delay', (response([1, np.nan, 1]),), 'nan_magnitude'),
    ('psrr_db', (response([1, np.nan, 1]), response([.1, .1, .1])), 'nan_magnitude'),
    ('gain_margins', (response([1, np.nan, 1]),), 'nan_magnitude'),
]


def assert_result(name, result, args):
    if name == 'gain_margins':
        assert type(result) is list and len(result) == 1
        assert type(result[0]) is tuple and len(result[0]) == 2
        assert all(type(v) is float and math.isnan(v) for v in result[0])
    elif name in ('group_delay', 'psrr_db'):
        assert type(result) is Signal
        kind = 'group_delay' if name == 'group_delay' else 'psrr'
        assert result.type == kind
        assert result.name == f'{kind}({",".join(arg.name for arg in args)})'
        assert result.xlabel == 'frequency'
        assert result.ylabel == ('s' if name == 'group_delay' else 'dB')
        np.testing.assert_array_equal(result.x, args[0].x)
        np.testing.assert_array_equal(
            result.y, [np.nan] * 3 if name == 'group_delay' else [20, np.nan, 20])
    else:
        assert type(result) is float and math.isnan(result)


@pytest.mark.parametrize('name,args,code', LEDGER)
@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_ledger_warning_policy_and_preserved_results(name, args, code, policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning) as error:
                getattr(ac, name)(*args)
            diagnostic = error.value
        else:
            result = getattr(ac, name)(*args)
            assert_result(name, result, args)
            if policy == 'ignore':
                assert caught == []
                return
            assert len(caught) == 1
            diagnostic = caught[0].message
    assert type(diagnostic) is MetricWarning
    assert diagnostic.metric == name
    assert diagnostic.code == code
    assert diagnostic.message and diagnostic.suggestion
    assert any(k.endswith('frequency_min_hz') for k in diagnostic.details)
    assert any(k.endswith('sample_count') for k in diagnostic.details)
    assert any(k.endswith('magnitude_unit') for k in diagnostic.details)
    assert all(v is None or isinstance(v, (str, float, int, bool))
               for v in diagnostic.details.values())


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('name,args', [
    ('dc_gain_dB', ()), ('gain_at_freq', (1.2,)), ('phase_at_freq', (1.2,)),
])
def test_point_warnings_describe_only_the_selected_sample(name, args, reverse):
    x, y = np.array([1, 2, 3]), np.array([complex(1, np.nan), 7, 8])
    if reverse:
        x, y = x[::-1], y[::-1]
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(ac, name)(response(y, x, unit='A/A'), *args))
    assert len(caught) == 1
    d = caught[0].message.details
    assert d['sample_count'] == d['nan_sample_count'] == 1
    assert d['frequency_min_hz'] == d['frequency_max_hz'] == 1
    assert d['nan_frequency_min_hz'] == d['nan_frequency_max_hz'] == 1
    assert d['magnitude_min'] is None and d['magnitude_max'] is None
    assert d['magnitude_unit'] == 'A/A'
    if args:
        assert d['requested_frequency_hz'] == 1.2
    if name == 'phase_at_freq':
        assert d['response_real'] == 1
        assert math.isnan(d['response_imag'])


@pytest.mark.parametrize('reverse', [False, True])
def test_unconsumed_nan_points_and_passband_are_quiet(reverse):
    x, y = np.array([1, 2, 3]), np.array([2, 2, np.nan])
    if reverse:
        x, y = x[::-1], y[::-1]
    sig = response(y, x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert ac.dc_gain_dB(sig) == 20 * np.log10(2)
        assert ac.gain_at_freq(sig, 1.2) == 2
        assert ac.phase_at_freq(sig, 1.2) == 0
        assert ac.cmrr_dB(sig, sig) == 0
        assert ac.cmrr_dB(sig, sig, 1.2) == 0
        assert ac.passband_ripple_db(sig, 2) == 0
    assert caught == []


def test_passband_warning_excludes_unconsumed_samples():
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(ac.passband_ripple_db(
            response([1, np.nan, np.nan, 100], [1, 2, 30, 40]), 2))
    d = caught[0].message.details
    assert d['sample_count'] == 2 and d['nan_sample_count'] == 1
    assert d['frequency_min_hz'] == 1 and d['frequency_max_hz'] == 2
    assert d['magnitude_min'] == d['magnitude_max'] == 1


@pytest.mark.parametrize('diff,common,code', [
    (np.nan, 0, 'nan_magnitude'), (0, np.nan, 'nan_magnitude'),
    (1, np.nan, 'nan_magnitude'), (0, 0, 'no_differential_reference'),
])
@pytest.mark.parametrize('freq', [None, 1.5])
def test_cmrr_distinguishes_contamination_and_zero_reference(diff, common, code, freq):
    with pytest.warns(MetricWarning) as caught:
        result = ac.cmrr_dB(response([7, diff], [3, 1], unit='A/A'),
                            response([common, 8], [1.2, 4], unit='V/V'), freq)
    assert math.isnan(result) and len(caught) == 1
    w = caught[0].message
    assert w.code == code
    d = w.details
    assert d['differential_sample_count'] == d['common_mode_sample_count'] == 1
    assert d['differential_frequency_min_hz'] == d['differential_frequency_max_hz'] == 1
    assert d['common_mode_frequency_min_hz'] == d['common_mode_frequency_max_hz'] == 1.2
    assert d['differential_magnitude_unit'] == 'A/A'
    assert d['common_mode_magnitude_unit'] == 'V/V'
    np.testing.assert_equal(d['differential_magnitude'], diff)
    np.testing.assert_equal(d['common_mode_magnitude'], common)


@pytest.mark.parametrize('supply,output,code,nans,zeros', [
    ([1, 0, 1, 0], [.1, 0, 0, 1], 'no_supply_reference', 0, 1),
    ([1, np.nan, 1, 0], [.1, .1, 0, 1], 'nan_magnitude', 1, 0),
    ([1, 1, 1, 0], [.1, np.nan, 0, 1], 'nan_magnitude', 1, 0),
    ([1, 0, 1, np.nan], [.1, 0, 0, np.nan], 'nan_magnitude', 1, 1),
])
@pytest.mark.parametrize('reverse', [False, True])
def test_psrr_preserves_finite_samples_floor_and_all_signal_fields(
        supply, output, code, nans, zeros, reverse):
    # Explicit expected values include the existing relative denominator floor
    # and the valid zero-supply/nonzero-output -inf limit.
    expected = [20, np.nan, 20 * np.log10(1 / np.finfo(float).eps),
                np.nan if nans and zeros else -np.inf]
    x = [1, 10, 100, 1000]
    if reverse:
        supply, output, expected, x = supply[::-1], output[::-1], expected[::-1], x[::-1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = ac.psrr_db(response(supply, x, 'supply', 'V'),
                            response(output, x, 'output', 'V'))
    assert len(caught) == 1
    w = caught[0].message
    assert w.code == code and w.metric == 'psrr_db'
    assert w.details['nan_pair_count'] == nans
    assert w.details['zero_pair_count'] == zeros
    assert w.details['supply_sample_count'] == w.details['output_sample_count'] == 4
    if zeros:
        assert w.details['zero_pair_frequency_min_hz'] == 10
        assert w.details['zero_pair_frequency_max_hz'] == 10
    assert result.name == 'psrr(supply,output)' and result.type == 'psrr'
    assert result.xlabel == 'frequency' and result.ylabel == 'dB'
    np.testing.assert_array_equal(result.x, x)
    np.testing.assert_array_equal(result.y, expected)


@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_gain_margin_inherits_one_originating_warning(policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning) as error:
                ac.gain_margin(response([1, np.nan, .1]))
            assert error.value.metric == 'gain_margins'
        else:
            assert math.isnan(ac.gain_margin(response([1, np.nan, .1])))
            assert len(caught) == (1 if policy == 'always' else 0)
            if caught:
                assert caught[0].message.metric == 'gain_margins'
                assert caught[0].message.code == 'nan_magnitude'


@pytest.mark.parametrize('reverse', [False, True])
def test_gbw_is_dc_relative_and_observes_endpoint(reverse):
    # The peak-relative bandwidth is 10 Hz; GBW's DC-relative bandwidth is
    # 100 Hz, giving 200 Hz*V/V. reference='peak' pins the peak-relative
    # definition explicitly (TASK-326 changed bandwidth_3dB's default to
    # 'dc', which would otherwise agree with gain_bw_product's own 100 Hz
    # here and defeat the point of this test).
    x, y = [1, 10, 100, 1000], [2, 10, 2, .1]
    if reverse:
        x, y = x[::-1], y[::-1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert ac.gain_bw_product(response(y, x)) == 200
        assert ac.bandwidth_3dB(response(y, x), reference='peak') == 10
        assert ac.bandwidth_3dB(response(y, x), reference='dc') == 100
    assert caught == []
    y = [2, 10, 2, 2 / np.sqrt(2)]
    if reverse:
        y = y[::-1]
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(ac.gain_bw_product(response(y, x)))
    assert len(caught) == 1
    w = caught[0].message
    assert w.metric == 'gain_bw_product' and 'DC-relative' in w.message
    assert w.details['threshold_magnitude'] == 2 / np.sqrt(2)
    assert w.details['endpoint_magnitude'] == 2 / np.sqrt(2)
    assert w.details['dc_magnitude'] == 2
    assert w.details['frequency_max_hz'] == 1000


def test_all_named_functions_have_quiet_successful_controls():
    sig = response([10, 8, 1], [1, 10, 100])
    # Unwrapped -90, -180, -270 degrees; exact crossover at 10 Hz.
    loop = response([-2j, -0.5, .1j], [1, 10, 100])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert ac.dc_gain_dB(sig) == 20
        assert ac.peak_gain_dB(sig) == 20
        assert ac.gain_at_freq(sig, 10) == 8
        assert ac.phase_at_freq(loop, 1) == -90
        assert ac.passband_ripple_db(sig, 10) == 20 * np.log10(10 / 8)
        assert ac.stopband_attenuation_db(sig, 100) == 20
        assert ac.q_from_peak(response([0, 1, 1, 0])) == 2
        assert ac.gain_bw_product(sig) == 100
        assert ac.cmrr_dB(sig, response([1, 1, 1], sig.x)) == 20
        assert ac.gain_margin(loop) == -20 * np.log10(.5)
        assert ac.gain_margins(loop) == [(10.0, -20 * np.log10(.5))]
        np.testing.assert_array_equal(ac.group_delay(sig).y, [0, 0, 0])
        np.testing.assert_array_equal(ac.psrr_db(sig, sig).y, [0, 0, 0])
    assert caught == []


def test_valid_limits_and_empty_margins_have_no_metric_warning():
    zero = response([0, 0, 0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        # Existing NumPy log10(0) RuntimeWarnings are not this warning family.
        with np.errstate(divide='ignore'):
            assert ac.dc_gain_dB(zero) == -math.inf
            assert ac.peak_gain_dB(zero) == -math.inf
            assert ac.cmrr_dB(zero, response([1, 1, 1])) == -math.inf
        assert ac.gain_at_freq(zero, 2) == 0
        assert ac.phase_at_freq(zero, 2) == 0
        assert ac.cmrr_dB(response([1, 1, 1]), zero) == math.inf
        assert ac.stopband_attenuation_db(response([1, 0, 0]), 2) == math.inf
        assert ac.gain_margins(response([2, 2, 2])) == []
        assert ac.gain_margin(response([.1, .1, .1])) == math.inf
        notch = response([-1j, complex(-0.0, -0.0), 1j])
        assert ac.gain_margins(notch) == [(2.0, math.inf)]
        assert ac.gain_margin(notch) == math.inf
    assert caught == []


@pytest.mark.parametrize('name', [
    'peak_gain_dB', 'gain_bw_product', 'q_from_peak', 'group_delay', 'gain_margins',
])
@pytest.mark.parametrize('all_nan', [False, True])
def test_whole_sweep_nan_details_include_counts_ranges_and_units(name, all_nan):
    y = [np.nan] * 3 if all_nan else [2, complex(1, np.nan), 4]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        getattr(ac, name)(response(y, [100, 10, 1], unit='A/A'))
    assert len(caught) == 1
    w = caught[0].message
    assert w.metric == name and w.code == 'nan_magnitude'
    d = w.details
    assert d['sample_count'] == 3
    assert d['nan_sample_count'] == (3 if all_nan else 1)
    assert d['frequency_min_hz'] == 1 and d['frequency_max_hz'] == 100
    assert d['nan_frequency_min_hz'] == (1 if all_nan else 10)
    assert d['nan_frequency_max_hz'] == (100 if all_nan else 10)
    assert d['magnitude_min'] == (None if all_nan else 2)
    assert d['magnitude_max'] == (None if all_nan else 4)
    assert d['magnitude_unit'] == 'A/A'
    if name == 'q_from_peak':
        assert math.isnan(d['peak_magnitude'])
        assert math.isnan(d['threshold_magnitude'])
        assert d['half_power_sample_count'] == 0


def test_unresolved_q_reports_the_measured_threshold_and_count():
    with pytest.warns(MetricWarning) as caught:
        ac.q_from_peak(response([0, 1, 0], [1, 10, 100]))
    w = caught[0].message
    assert w.code == 'half_power_span_unresolved'
    assert w.details['peak_magnitude'] == 1
    assert w.details['threshold_magnitude'] == 1 / np.sqrt(2)
    assert w.details['half_power_sample_count'] == 1
    assert 'denser' in w.suggestion


@pytest.mark.parametrize('reverse', [False, True])
def test_no_dc_reference_is_the_lowest_frequency_not_the_peak(reverse):
    x, y = [1, 10, 100], [0, 10, 1]
    if reverse:
        x, y = x[::-1], y[::-1]
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(ac.gain_bw_product(response(y, x)))
    w = caught[0].message
    assert w.code == 'no_dc_reference'
    assert w.details['dc_magnitude'] == 0
    assert w.details['dc_frequency_hz'] == 1
    assert w.details['magnitude_max'] == 10


@pytest.mark.parametrize('reverse', [False, True])
def test_no_phase_crossing_at_unity_endpoint_is_unavailable(reverse):
    x, y = [1, 10, 100], [2, 2, 1]
    if reverse:
        x, y = x[::-1], y[::-1]
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(ac.gain_margin(response(y, x)))
    assert len(caught) == 1
    w = caught[0].message
    assert w.metric == 'gain_margin'
    assert w.code == 'no_phase_crossing_gain_at_least_one'
    assert w.details['endpoint_magnitude'] == 1
    assert w.details['frequency_max_hz'] == 100
    assert 'not established' in w.suggestion


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('x', [[0, 100, 200], [1, 100, 200]])
@pytest.mark.parametrize('magnitudes,expected', [
    ([2, 1, .5], 100), ([.5, 1, 2], 100),
    ([1, .5, .25], None), ([1, 2, 4], None),
    ([2, 1.5, 1], 200), ([.25, .5, 1], 200),
])
def test_exact_unity_sample_uses_measured_frequency(x, magnitudes, expected, reverse):
    # Reconstructing a sampled crossing through logs loses exactness and
    # can manufacture a second crossing or an invalid log at DC.
    expected = x[0] if expected is None else expected
    y = -1j * np.asarray(magnitudes)
    if reverse:
        x, y = x[::-1], y[::-1]
    sig = response(y, x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert ac.unity_gain_freq(sig) == expected
        margin = ac.phase_margin(sig)
        if expected == 0:
            assert math.isnan(margin)
        else:
            assert margin == 90
    assert caught == []


@pytest.mark.parametrize('reverse', [False, True])
def test_exact_unity_after_dc_preserves_sampled_phase(reverse):
    x, y = [0, 100, 200], [2, -1j, -.5j]
    if reverse:
        x, y = x[::-1], y[::-1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        sig = response(y, x)
        assert ac.unity_gain_freq(sig) == 100
        assert ac.phase_margin(sig) == 90
    assert caught == []
