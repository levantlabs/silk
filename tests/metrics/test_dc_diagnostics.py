"""Unavailable transfer measurements and malformed DC requests (TASK-261)."""
import math
import warnings

import numpy as np
import pytest

from silk.circuit.spec_report import SpecReport
from silk.metrics.converter import (
    INL,
    DNL,
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


def code_signal(voltages):
    return Signal('transfer', 'voltage', 'code', 'V',
                  np.arange(len(voltages), dtype=float), np.array(voltages, dtype=float))


def assert_constant_warning(caught, metric, voltage):
    assert len(caught) == 1
    warning = caught[0].message
    assert warning.metric == metric
    assert warning.code == 'constant_transfer_curve'
    assert warning.details['transition_count'] == 3
    assert warning.details['transition_voltage_v'] == voltage
    assert warning.details['lsb_v'] == 0.0
    assert 'constant' in warning.message
    assert 'LSB' in warning.message
    assert warning.suggestion


@pytest.mark.parametrize('method', ['endpoint', 'best_fit'])
@pytest.mark.parametrize('voltage', [0.0, 1.0, -0.25])
def test_constant_inl(method, voltage):
    with pytest.warns(MetricWarning) as caught:
        result = inl(code_signal([voltage] * 3), method=method)
    assert isinstance(result, INL)
    assert result.method == method
    np.testing.assert_array_equal(result.inl_lsb, [np.nan] * 3)
    assert math.isnan(result.max_abs_lsb)
    assert_constant_warning(caught, 'inl', voltage)


@pytest.mark.parametrize('metric', [dnl, missing_codes])
@pytest.mark.parametrize('voltage', [0.0, 1.0, -0.25])
def test_constant_dnl_and_delegation(metric, voltage):
    with pytest.warns(MetricWarning) as caught:
        result = metric(code_signal([voltage] * 3))
    if metric is dnl:
        assert isinstance(result, DNL)
        assert math.isnan(result.max_abs_lsb)
        np.testing.assert_array_equal(result.dnl_lsb, [np.nan] * 2)
    else:
        np.testing.assert_array_equal(result, [np.nan])
    assert_constant_warning(caught, 'dnl', voltage)


@pytest.mark.parametrize('method', ['endpoint', 'best_fit'])
def test_constant_scalar_extraction_fails_specs(method):
    # Nonzero best-fit constants used to produce finite values that pass.
    # Field extraction is also the contract future scalar wrappers inherit.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        value = inl(code_signal([1.0] * 3), method=method).max_abs_lsb
    for op, target in [('<', 100.0), ('<=', 100.0), ('>', -1.0),
                       ('>=', -1.0), ('==', value)]:
        report = SpecReport('dc', 'adc', {'inl': value})
        assert not report.check('INL limit', 'inl', op, target).passed


def test_dnl_scalar_extraction_fails_specs():
    with pytest.warns(MetricWarning) as caught:
        value = dnl(code_signal([1.0] * 3)).max_abs_lsb
    assert_constant_warning(caught, 'dnl', 1.0)
    for op in ('<', '<=', '>', '>=', '=='):
        report = SpecReport('dc', 'adc', {'dnl': value})
        assert not report.check('DNL limit', 'dnl', op, 1.0).passed


@pytest.mark.parametrize('metric', [inl, dnl, missing_codes])
def test_warning_policy(metric):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        metric(code_signal([1.0] * 3))
    with warnings.catch_warnings():
        warnings.simplefilter('error', MetricWarning)
        with pytest.raises(MetricWarning, match='constant'):
            metric(code_signal([1.0] * 3))


@pytest.mark.parametrize('metric', [inl, dnl, missing_codes])
def test_nonconstant_zero_endpoint_span_is_not_called_constant(metric):
    with pytest.warns(MetricWarning) as caught:
        result = metric(code_signal([0.0, 1.0, 0.0]))
    assert len(caught) == 1
    warning = caught[0].message
    assert warning.code == 'zero_derived_lsb'
    assert 'constant' not in warning.message
    assert warning.details['lsb_v'] == 0.0
    if metric is missing_codes:
        np.testing.assert_array_equal(result, [np.nan])
    else:
        assert math.isnan(result.max_abs_lsb)


@pytest.mark.parametrize('method', ['endpoint', 'best_fit'])
@pytest.mark.parametrize('index', [0, 2, 4])
def test_nan_transition_preserves_shared_reference_semantics(method, index):
    v = np.arange(5, dtype=float)
    v[index] = np.nan
    sig = code_signal(v)
    sig.x = np.arange(10, 15, dtype=float)
    with pytest.warns(MetricWarning) as caught:
        result = inl(sig, method=method)
    np.testing.assert_array_equal(result.inl_lsb, [np.nan] * 5)
    assert math.isnan(result.max_abs_lsb)
    assert_nan_code(caught, 'inl', index + 10)
    with pytest.warns(MetricWarning) as caught:
        result = dnl(sig)
    expected = [0.0, np.nan, np.nan, 0.0] if index == 2 else [np.nan] * 4
    np.testing.assert_array_equal(result.dnl_lsb, expected)
    assert math.isnan(result.max_abs_lsb)
    assert_nan_code(caught, 'dnl', index + 10)
    with pytest.warns(MetricWarning) as caught:
        np.testing.assert_array_equal(missing_codes(sig), [np.nan])
    assert_nan_code(caught, 'missing_codes', index + 10)


def assert_nan_code(caught, metric, code):
    assert len(caught) == 1
    warning = caught[0].message
    assert warning.metric == metric
    assert warning.code == 'nan_transition_voltage'
    assert warning.details['code'] == code
    assert f'code {code:g}' in warning.message
    assert warning.suggestion


def test_real_zero_width_bin_and_finite_fits_stay_measurable():
    sig = code_signal([0.0, 1.0, 1.0, 3.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert dnl(sig).dnl_lsb[1] == -1.0
        np.testing.assert_array_equal(missing_codes(sig), [1.0])
        assert math.isfinite(inl(sig).max_abs_lsb)
        assert math.isfinite(inl(sig, method='best_fit').max_abs_lsb)
    assert not caught


@pytest.mark.parametrize('metric', [inl, dnl, missing_codes, is_monotonic])
def test_wrong_axis_and_too_few_codes_name_metric(metric):
    sig = code_signal([0.0, 1.0])
    sig.xlabel = 'wrong'
    with pytest.raises(ValueError, match=rf"{metric.__name__}:.*xlabel='code'.*'wrong'"):
        metric(sig)
    with pytest.raises(ValueError, match=rf'{metric.__name__}:.*at least 2 codes.*1'):
        metric(code_signal([0.0]))


@pytest.mark.parametrize('metric', [inl, dnl, missing_codes, is_monotonic])
def test_nan_code_is_caller_error_even_with_nan_voltage(metric):
    sig = code_signal([0.0, np.nan, 2.0])
    sig.x[1] = np.nan
    with pytest.raises(ValueError, match=rf'{metric.__name__}:.*x-axis.*nonfinite.*index 1.*needs'):
        metric(sig)


@pytest.mark.parametrize('method', ['endpoint', 'best_fit'])
def test_inl_rejects_nan_code_before_fitting(method):
    sig = code_signal([0.0, 1.0, 2.0])
    sig.x[1] = np.nan
    with pytest.raises(ValueError, match=r'inl:.*x-axis.*nonfinite.*index 1.*needs'):
        inl(sig, method=method)


@pytest.mark.parametrize('metric', [inl, dnl, missing_codes, is_monotonic])
def test_code_voltage_length_mismatch(metric):
    sig = code_signal([0.0, 1.0, 2.0])
    sig.y = sig.y[:2]
    with pytest.raises(ValueError, match=rf'{metric.__name__}:.*same length.*3.*2'):
        metric(sig)


def test_bad_method_names_request_even_with_degenerate_data():
    for v in ([1.0] * 3, [0.0, np.nan, 2.0]):
        with pytest.raises(ValueError, match=r"inl:.*'endpoint'.*'best_fit'.*'bogus'"):
            inl(code_signal(v), method='bogus')


@pytest.mark.parametrize('wrong', ['v_in', 'code'])
def test_transient_axis_label_names_metric_and_argument(wrong):
    sigs = {name: Signal(name, 'voltage', 'time', 'V',
                         np.arange(3.0), np.arange(3.0))
            for name in ('v_in', 'code')}
    sigs[wrong].xlabel = 'wrong'
    with pytest.raises(ValueError, match=rf"extract_code_transitions:.*{wrong}.*'time'.*'wrong'"):
        extract_code_transitions(**sigs)


@pytest.mark.parametrize('axis', [[0.0, 2.0, 4.0], [0.0, 1.0]])
def test_transient_axis_mismatch_is_actionable(axis):
    v = Signal('v', 'voltage', 'time', 'V', np.arange(3.0), np.arange(3.0))
    code = Signal('c', 'code', 'time', '', np.array(axis), np.arange(len(axis)))
    with pytest.raises(ValueError, match=r'extract_code_transitions:.*same time axis.*v_in.*code'):
        extract_code_transitions(v, code)


@pytest.mark.parametrize('axis', [[0.0, -10.0, -20.0], [-20.0, 0.0, -10.0]])
def test_descending_power_coordinates_are_caller_error(axis):
    p_in = np.asarray(axis)
    sig = Signal('power', 'power', 'power', 'dBm', p_in, p_in + 20.0)
    with pytest.raises(ValueError, match=r'p1db:.*x-axis.*index.*needs.*ascending'):
        p1db(sig)
