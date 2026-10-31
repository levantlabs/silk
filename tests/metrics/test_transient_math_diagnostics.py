"""Waveform math retains consumed NaNs and complex information."""
import hashlib
import json
import math
import warnings

import numpy as np
import pytest

from silk.circuit.spec_report import SpecReport
import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal
from silk.signals.schema import export_metric_schema


def signal(y, *, x=None, name='waveform'):
    return Signal(name, 'voltage', 'time', 'V',
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x),
                  np.asarray(y))


@pytest.mark.parametrize('y,goal,kind', [
    ([1, np.nan, 1], [1, 1, 1], 'rms'),
    ([complex(1, np.nan)] * 3, [1 + 0j] * 3, 'rms'),
], ids=['real_skips_nan', 'complex_nan'])
def test_recorded_false_pass_fails_spec(y, goal, kind):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = M.error_norm(signal(y), signal(goal), kind=kind)
    report = SpecReport('tran', 'dut', {'error': value})
    assert not report.check('upper', 'error', '<', .01).passed
    assert type(value) is float and math.isnan(value)
    for op in ('<=', '>', '>=', '=='):
        assert not SpecReport('tran', 'dut', {'error': value}).check(
            'limit', 'error', op, .01).passed
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert isinstance(diagnostic, MetricWarning)
    assert diagnostic.metric == 'error_norm'
    assert diagnostic.code == 'nan_waveform'
    assert 'consumed data contains NaNs' in diagnostic.message
    assert 'source result' in diagnostic.suggestion
    assert diagnostic.details['nan_sample_count'] == np.count_nonzero(np.isnan(y))


@pytest.mark.parametrize('kind', ['max', 'rms'])
@pytest.mark.parametrize('y,goal,count', [
    ([np.nan, 1, 1], [0, 1, 1], 1),
    ([np.nan, 1, 1], [0, 0, 0], 1),
    ([1, 1, 1], [1, np.nan, 1], 1),
    ([np.nan, 1, np.nan], [np.nan, np.nan, 1], 3),
    ([complex(np.inf, np.nan), 1, 1], [0, 1, 1], 1),
])
def test_nan_precedes_zero_reference_exclusion(kind, y, goal, count):
    with pytest.warns(MetricWarning) as caught:
        value = M.error_norm(signal(y, name='measured'),
                             signal(goal, name='reference'), kind=kind)
    assert math.isnan(value)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.code == 'nan_waveform'
    assert diagnostic.details['sample_count'] == 3
    assert diagnostic.details['nan_sample_count'] == count
    assert diagnostic.details['sig_name'] == 'measured'
    assert diagnostic.details['goal_name'] == 'reference'
    assert all(type(v) in (str, float, int, type(None))
               for v in diagnostic.details.values())


@pytest.mark.parametrize('kind,expected', [('max', 2), ('rms', 2)])
def test_real_signed_values_are_preserved(kind, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = M.error_norm(signal([-1, 1]), signal([1, -1]), kind=kind)
    assert value == expected
    assert caught == []


@pytest.mark.parametrize('kind', ['max', 'rms'])
def test_error_norm_only_consumes_interpolated_reference_region(kind):
    reference = signal([np.nan, 1, 1, np.nan], x=[-1, 0, 1, 2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert M.error_norm(signal([1, 1], x=[.25, .75]), reference, kind=kind) == 0
    assert caught == []
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.error_norm(signal([1, 1], x=[-.5, .5]), reference, kind=kind))
    assert caught[0].message.details['nan_sample_count'] == 1


@pytest.mark.parametrize('policy', ['ignore', 'error'])
def test_warning_policy_preserves_standard_behavior(policy):
    with warnings.catch_warnings():
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning, match='error_norm'):
                M.error_norm(signal([1, np.nan]), signal([1, 1]), kind='rms')
        else:
            assert math.isnan(M.error_norm(signal([1, np.nan]), signal([1, 1]), kind='rms'))


TRANSFORMS = [
    ('running_integral', lambda s: M.running_integral(s)),
    ('lshift', lambda s: M.lshift(s, 1)),
    ('clip', lambda s: M.window(s, 1, 2)),
    ('flip', lambda s: M.flip(s)),
    ('sample', lambda s: M.sample(s, n=4)),
    ('convolve', lambda s: M.convolve(s, s)),
    ('cross_correlation', lambda s: M.cross_correlation(s, s)),
]


@pytest.mark.parametrize('name,transform', TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_transforms_preserve_imaginary_nan(name, transform):
    s = signal([complex(1, np.nan)] * 4)
    original = s.y.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = transform(s)
    assert isinstance(out, Signal)
    assert np.iscomplexobj(out.y)
    consumed = out.y[1:] if name == 'running_integral' else out.y
    assert np.all(np.isnan(consumed))
    assert np.all(np.isfinite(out.x))
    assert not any(issubclass(w.category, np.exceptions.ComplexWarning) for w in caught)
    np.testing.assert_array_equal(s.y, original)


@pytest.mark.parametrize('name,transform', TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_finite_complex_transforms(name, transform):
    s = signal([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
    expected = {
        'running_integral': ([0, 1, 2, 3], [0, 2 + 3j, 6 + 8j, 12 + 15j]),
        'lshift': ([-1, 0, 1, 2], s.y),
        'clip': ([1, 2], [3 + 4j, 5 + 6j]),
        'flip': ([0, 1, 2, 3], s.y[::-1]),
        'sample': ([0, 1, 2, 3], s.y),
        'convolve': (np.arange(7), [-3 + 4j, -10 + 20j, -21 + 56j,
                                   -36 + 120j, -33 + 164j, -26 + 164j, -15 + 112j]),
        'cross_correlation': (np.arange(-3, 4), [23 + 6j, 70 + 8j, 133 + 6j,
                                               204, 133 - 6j, 70 - 8j, 23 - 6j]),
    }
    before = (s.x.copy(), s.y.copy())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = transform(s)
    assert caught == []
    assert isinstance(out, Signal)
    assert np.iscomplexobj(out.y)
    np.testing.assert_allclose(out.x, expected[name][0])
    np.testing.assert_allclose(out.y, expected[name][1])
    np.testing.assert_array_equal(s.x, before[0])
    np.testing.assert_array_equal(s.y, before[1])


def test_sample_interpolates_both_components_and_retains_endpoint_clamping():
    out = M.sample(signal([1 + 2j, 3 + 6j]), x_new=[-1, 0, .5, 1, 2])
    np.testing.assert_array_equal(out.y, [1 + 2j, 1 + 2j, 2 + 4j, 3 + 6j, 3 + 6j])


def test_clip_and_sample_do_not_consume_nan_outside_their_window():
    s = signal([complex(1, np.nan), 1 + 2j, 3 + 4j, complex(1, np.nan)])
    np.testing.assert_array_equal(M.window(s, 1, 2).y, [1 + 2j, 3 + 4j])
    np.testing.assert_array_equal(M.sample(s, x_new=[1.25, 1.75]).y, [1.5 + 2.5j, 2.5 + 3.5j])
    assert M.window(s, 5, 6).y.size == 0


def test_metric_schema_bytes_unchanged():
    canonical = json.dumps(export_metric_schema(), sort_keys=True,
                           separators=(',', ':'), allow_nan=False)
    # TASK-223 deliberately added propagation_delay.out_edge; digest bumped.
    # TASK-288 deliberately corrected integrated_noise's unit; digest bumped again.
    # TASK-326 deliberately added bandwidth_3dB's reference={'dc','peak'}
    # param (default 'dc', BREAKING); digest bumped again.
    # TASK-364 replaced x_when with crossing (threshold, start/end); bumped again.
    assert hashlib.sha256(canonical.encode()).hexdigest() == (
        # TASK-226.22: required scalar eye threshold; centering default True.
        'f47ca45cb4a0666deb6a955dfe495e17888bfa63308be96cb15e27966e65c040'  # TASK-417 jitter; TASK-418 long_term_jitter units/reference/phase/threshold/edge/start/end; TASK-412 removed input_referred_offset, kickback, hold_droop; TASK-379 variance; TASK-386 relative_error_norm; TASK-378 interpolate; TASK-399 static_errors wrappers; TASK-385 error_norm mean_abs + start/end; TASK-377 minimum/maximum + start/end; TASK-233 eight wrappers; TASK-323.3 final_value; TASK-364 crossing; TASK-369 evm_percent; TASK-365 rise/fall/slew edge params; TASK-363 propagation_delay; TASK-366 period/frequency; TASK-226.20 crossing_jitter_rms; TASK-367 average_duty, pulse_width, on/off_time; TASK-438 hysteresis n_rising/n_falling/start/end
    )


@pytest.mark.parametrize('kind', ['unknown', None])
def test_invalid_kind_is_not_hidden_by_nan(kind):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match='kind'):
            M.error_norm(signal([np.nan, 1]), signal([1, 1]), kind=kind)
    assert caught == []


@pytest.mark.parametrize('goal', [
    [complex(1, np.nan), 1],
    [complex(np.inf, np.nan), 1],
])
def test_reference_imaginary_nan_survives_magnitude_conversion(goal):
    with pytest.warns(MetricWarning) as caught:
        value = M.error_norm(signal([1, 1]), signal(goal))
    assert math.isnan(value)
    assert len(caught) == 1
    assert caught[0].message.details['goal_nan_sample_count'] == 1


@pytest.mark.parametrize('name,transform', TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
def test_real_transform_dtype_and_shape(name, transform, dtype):
    s = signal(np.asarray([1, 2, 3, 4], dtype=dtype))
    expected = {
        'running_integral': [0, 1.5, 4, 7.5],
        'lshift': [1, 2, 3, 4],
        'clip': [2, 3],
        'flip': [4, 3, 2, 1],
        'sample': [1, 2, 3, 4],
        'convolve': [1, 4, 10, 20, 25, 24, 16],
        'cross_correlation': [4, 11, 20, 30, 20, 11, 4],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = transform(s)
    assert caught == []
    assert result.y.dtype == np.dtype(float)
    np.testing.assert_array_equal(result.y, expected[name])


def test_complex_running_integral_single_sample_retains_initial_zero():
    out = M.running_integral(signal([1 + 2j]))
    np.testing.assert_array_equal(out.x, [0])
    np.testing.assert_array_equal(out.y, [0j])


@pytest.mark.parametrize('name', ['convolve', 'cross_correlation'])
@pytest.mark.parametrize('complex_role', ['a', 'b'])
def test_binary_transforms_preserve_either_complex_operand(name, complex_role):
    a = signal([1, 2], x=[2, 2.5])
    b = signal([3, 4, 5], x=[5, 5.5, 6])
    if complex_role == 'a':
        a = signal(a.y * 1j, x=a.x)
    else:
        b = signal(b.y * 1j, x=b.x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = getattr(M, name)(a, b)
    assert caught == []
    if name == 'convolve':
        np.testing.assert_array_equal(out.x, [7, 7.5, 8, 8.5])
        np.testing.assert_array_equal(out.y, [1.5j, 5j, 6.5j, 5j])
    else:
        # a.x[0]=2, b.x[0]=5 -> lag axis carries a.x[0]-b.x[0] = -3.
        np.testing.assert_array_equal(out.x, [-4, -3.5, -3, -2.5])
        sign = 1 if complex_role == 'a' else -1
        np.testing.assert_array_equal(out.y, sign * np.asarray([5j, 14j, 11j, 6j]))


@pytest.mark.parametrize('kind', ['max', 'rms'])
def test_reference_nan_at_unused_endpoints_is_not_consumed(kind):
    reference = signal([np.nan, 1, 2, np.nan], x=[-1, 0, 1, 2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = M.error_norm(signal([1, 2], x=[0, 1]), reference, kind=kind)
    assert value == 0
    assert caught == []


@pytest.mark.parametrize('kind', ['max', 'rms'])
def test_reference_nan_counts_compared_samples_after_resampling(kind):
    reference = signal([1, np.nan, 1], x=[0, 1, 2])
    measured = signal([1] * 5, x=[0, .5, 1, 1.5, 2])
    with pytest.warns(MetricWarning) as caught:
        value = M.error_norm(measured, reference, kind=kind)
    assert math.isnan(value)
    assert caught[0].message.details['sample_count'] == 5
    assert caught[0].message.details['nan_sample_count'] == 3


@pytest.mark.parametrize('kind,expected', [
    ('max', 2.5), ('rms', math.sqrt(3.25)),
])
def test_complex_error_norm_interpolates_goal_then_measures_difference(kind, expected):
    # The interpolated goal is [.5j, -.5j], so error magnitudes are [.5, 2.5].
    # Either magnitude-first alternative loses the phase-dependent difference.
    measured = signal([1j, 2j], x=[.25, .75])
    reference = signal([1j, -1j], x=[0, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = M.error_norm(measured, reference, kind=kind)
    assert type(value) is float
    assert value == pytest.approx(expected)
    assert caught == []
