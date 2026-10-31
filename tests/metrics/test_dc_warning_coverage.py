"""Remaining DC diagnostic coverage (TASK-271); no new numerical policy."""
import math
import warnings

import numpy as np
import pytest

from silk.metrics.converter import (
    dnl,
    extract_code_transitions,
    inl,
    is_monotonic,
    missing_codes,
)
from silk.metrics.rf import (
    p1db,
)
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def power_signal(inputs, outputs):
    return Signal('sweep', 'power', 'power', 'dBm',
                  np.array(inputs, dtype=float), np.array(outputs, dtype=float))


def transitions(voltages):
    return Signal('transfer', 'voltage', 'code', 'V',
                  np.arange(10, 10 + len(voltages), dtype=float),
                  np.array(voltages, dtype=float))


def capture(voltages, codes):
    time = np.arange(len(codes), dtype=float) * 1e-6
    return (Signal('ramp', 'voltage', 'time', 'V', time, np.array(voltages)),
            Signal('codes', 'code', 'time', '', time, np.array(codes)))


def one_warning(caught, metric, code):
    assert len(caught) == 1
    warning = caught[0].message
    assert isinstance(warning, MetricWarning)
    assert warning.metric == metric
    assert warning.code == code
    assert warning.suggestion
    assert all(value is None or isinstance(value, (str, int, float, bool))
               for value in warning.details.values())
    return warning


@pytest.mark.parametrize('inputs,outputs,in_count,out_count,in_range,out_range', [
    ([-20, np.nan, 0], [0, 10, 20], 1, 0, (-20, 0), (0, 20)),
    ([-20, -10, 0], [0, np.nan, 20], 0, 1, (-20, 0), (0, 20)),
    ([np.nan] * 3, [np.nan] * 3, 3, 3, (None, None), (None, None)),
    # The existing guard consumes the full sweep, including after compression.
    ([-20, -10, 0], [0, 8, np.nan], 0, 1, (-20, 0), (0, 8)),
])
def test_contaminated_power_sweep(inputs, outputs, in_count, out_count,
                                  in_range, out_range):
    if np.any(np.isnan(inputs)):
        with pytest.raises(ValueError, match='p1db:.*axis.*nonfinite.*needs'):
            p1db(power_signal(inputs, outputs))
        return
    with pytest.warns(MetricWarning) as caught:
        result = p1db(power_signal(inputs, outputs))
    assert type(result) is float and math.isnan(result)
    warning = one_warning(caught, 'p1db', 'nan_power_sweep')
    details = warning.details
    assert details['sample_count'] == 3
    assert details['nan_input_power_count'] == in_count
    assert details['nan_output_power_count'] == out_count
    assert (details['input_power_min_dbm'], details['input_power_max_dbm']) == in_range
    assert (details['output_power_min_dbm'], details['output_power_max_dbm']) == out_range
    assert 'consumed' in warning.message and 'dBm' in warning.message
    assert 'source' in warning.suggestion
    assert 'extend' not in warning.suggestion


@pytest.mark.parametrize('inputs,outputs,gain', [
    ([-20, -10, 0], [0, 10, 20], 20),
    ([-20, -10, 0], [-20, -10, 0], 0),
    ([0, 0, 0], [0, 0, 0], 0),
])
def test_no_compression_observed_is_limited_to_measured_range(inputs, outputs, gain):
    with pytest.warns(MetricWarning) as caught:
        result = p1db(power_signal(inputs, outputs))
    assert type(result) is float and math.isnan(result)
    warning = one_warning(caught, 'p1db', 'no_compression_observed')
    assert warning.details == {
        'signal_name': 'sweep', 'sample_count': 3,
        'input_power_min_dbm': min(inputs), 'input_power_max_dbm': max(inputs),
        'output_power_min_dbm': min(outputs), 'output_power_max_dbm': max(outputs),
        'nan_input_power_count': 0, 'nan_output_power_count': 0,
        'reference_gain_db': gain, 'compression_threshold_db': 1.0,
    }
    assert 'No 1 dB compression' in warning.message
    assert 'observed in the measured input-power range' in warning.message
    assert 'outside this range is not established' in warning.suggestion


@pytest.mark.parametrize('metric', [dnl, missing_codes])
def test_one_interior_transition_names_both_affected_widths(metric):
    with pytest.warns(MetricWarning) as caught:
        result = metric(transitions([0, 1, np.nan, 3, 4]))
    warning = one_warning(caught, metric.__name__, 'nan_transition_voltage')
    details = warning.details
    assert details['code'] == 12
    assert details['transition_index'] == 2
    assert details['transition_count'] == 5
    assert details['nan_transition_count'] == 1
    assert (details['code_min'], details['code_max']) == (10, 14)
    assert details['transition_voltage_unit'] == 'V'
    assert (details['transition_voltage_min_v'], details['transition_voltage_max_v']) == (0, 4)
    assert details['affected_width_count'] == 2
    assert (details['left_width_from_code'], details['left_width_to_code']) == (11, 12)
    assert (details['right_width_from_code'], details['right_width_to_code']) == (12, 13)
    assert 'code 11 to code 12' in warning.message
    assert 'code 12 to code 13' in warning.message
    assert 'every DNL interval' not in warning.message
    if metric is dnl:
        np.testing.assert_array_equal(result.dnl_lsb, [0, np.nan, np.nan, 0])
    else:
        np.testing.assert_array_equal(result, [np.nan])


@pytest.mark.parametrize('index,side,from_code,to_code', [(0, 'right', 10, 11),
                                                       (4, 'left', 13, 14)])
@pytest.mark.parametrize('metric', [dnl, missing_codes])
def test_endpoint_warning_distinguishes_width_from_shared_lsb(metric, index, side,
                                                             from_code, to_code):
    values = np.arange(5, dtype=float)
    values[index] = np.nan
    with pytest.warns(MetricWarning) as caught:
        metric(transitions(values))
    warning = one_warning(caught, metric.__name__, 'nan_transition_voltage')
    assert warning.details['affected_width_count'] == 1
    assert warning.details[f'{side}_width_from_code'] == from_code
    assert warning.details[f'{side}_width_to_code'] == to_code
    assert 'shared LSB' in warning.message
    assert 'every DNL interval' in warning.message


@pytest.mark.parametrize('method', ['endpoint', 'best_fit'])
def test_inl_warnings_identify_each_missing_voltage_without_selecting_fit_policy(method):
    with pytest.warns(MetricWarning) as caught:
        result = inl(transitions([0, np.nan, 2, np.nan, 4]), method=method)
    assert len(caught) == 2
    assert result.method == method
    assert result.inl_lsb.shape == (5,)
    for item, code in zip(caught, (11, 13)):
        warning = item.message
        assert warning.metric == 'inl'
        assert warning.code == 'nan_transition_voltage'
        assert warning.details['code'] == code
        assert warning.details['nan_transition_count'] == 2
        assert warning.details['transition_voltage_unit'] == 'V'
        assert 'depending on it' in warning.message
        assert warning.suggestion


def test_selected_nan_voltage_names_sample_time_and_code():
    with pytest.warns(MetricWarning) as caught:
        result = extract_code_transitions(*capture([0, np.nan, 2, 3], [0, 1, 2, 3]))
    warning = one_warning(caught, 'extract_code_transitions', 'nan_transition_voltage')
    assert warning.details == {
        'code': 1, 'transition_index': 0, 'transition_count': 3,
        'transition_voltage_unit': 'V', 'sample_index': 1,
        'sample_time_s': 1e-6, 'sample_count': 4,
        'time_min_s': 0, 'time_max_s': 3e-6,
    }
    assert result.xlabel == 'code' and result.ylabel == 'V'
    np.testing.assert_array_equal(result.x, [1, 2, 3])
    np.testing.assert_array_equal(result.y, [np.nan, 2, 3])


def test_one_selected_sample_can_supply_multiple_transition_codes():
    with pytest.warns(MetricWarning) as caught:
        result = extract_code_transitions(*capture([0, np.nan, 3], [0, 2, 3]))
    assert len(caught) == 2
    assert [item.message.details['code'] for item in caught] == [1, 2]
    assert all(item.message.details['sample_index'] == 1 for item in caught)
    np.testing.assert_array_equal(result.y, [np.nan, np.nan, 3])


@pytest.mark.parametrize('measure,origin,code', [
    (lambda: p1db(power_signal([-20, -10], [0, 10])),
     'p1db', 'no_compression_observed'),
    (lambda: p1db(power_signal([-20, -10], [0, np.nan])),
     'p1db', 'nan_power_sweep'),
    (lambda: dnl(transitions([0, np.nan, 2])), 'dnl', 'nan_transition_voltage'),
    (lambda: inl(transitions([0, np.nan, 2]), method='best_fit'),
     'inl', 'nan_transition_voltage'),
    (lambda: missing_codes(transitions([0, np.nan, 2])),
     'missing_codes', 'nan_transition_voltage'),
    (lambda: extract_code_transitions(*capture([0, np.nan, 2], [0, 1, 2])),
     'extract_code_transitions', 'nan_transition_voltage'),
])
def test_new_diagnostics_obey_warning_policy_in_composed_calls(measure, origin, code):
    def composed():
        return measure()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore', MetricWarning)
        composed()
    assert not caught
    with warnings.catch_warnings():
        warnings.simplefilter('error', MetricWarning)
        with pytest.raises(MetricWarning) as raised:
            composed()
    assert raised.value.metric == origin
    assert raised.value.code == code


def test_successful_measurements_and_intentional_quiet_cases_emit_nothing():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert p1db(power_signal([-20, -10, 0], [0, 10, 18])) == -5
        assert p1db(power_signal([-20, -10], [0, 9])) == -10
        valid = transitions([0, 1, 1, 3])
        assert np.isfinite(inl(valid).max_abs_lsb)
        assert np.isfinite(inl(valid, method='best_fit').max_abs_lsb)
        assert dnl(valid).dnl_lsb[1] == -1
        np.testing.assert_array_equal(missing_codes(valid), [11])
        assert is_monotonic(valid) is False
        assert is_monotonic(transitions([0, np.nan, 2])) is False
        ideal = transitions([0, 1, 2])
        assert inl(ideal).max_abs_lsb == 0
        assert dnl(ideal).max_abs_lsb == 0
        assert is_monotonic(ideal) is True
        assert missing_codes(ideal).size == 0
        # Only voltages actually selected as transitions are consumed.
        result = extract_code_transitions(*capture([np.nan, 1, np.nan, 2, np.nan],
                                                   [0, 1, 1, 2, 2]))
        np.testing.assert_array_equal(result.y, [1, 2])
        for voltages, codes in [([np.nan], [0]), ([np.nan] * 3, [2] * 3)]:
            assert extract_code_transitions(*capture(voltages, codes)).y.size == 0
    assert not caught
