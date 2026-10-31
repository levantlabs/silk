"""TASK-256: unavailable AC measurements cannot pass a numeric spec."""
import hashlib
import json
import math
import warnings

import numpy as np
import pytest

from silk.circuit.spec_report import SpecReport
from silk.metrics import ac
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal
from silk.signals.schema import export_metric_schema


def response(y, x=(1, 2, 3), name='response'):
    return Signal(name, 'voltage', 'frequency', 'V/V',
                  np.asarray(x, dtype=float), np.asarray(y, dtype=complex))


@pytest.mark.parametrize('name,y,x,args,op,target', [
    ('passband_ripple_db', [1, 0, 1], [1, 2, 3], (3,), '<', 1),
    ('stopband_attenuation_db', [0, np.nan, np.nan], [1, 2, 3], (2,), '<=', 0),
    ('q_from_peak', [2, 2, 2], [1, 10, 100], (), '<', 1),
])
def test_ledger_false_passes_fail_spec_check(name, y, x, args, op, target):
    # Assert the actual report verdict first: all three assertions fail on base.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        value = getattr(ac, name)(response(y, x), *args)
    report = SpecReport('ac', 'dut', {name: value})
    assert not report.check('ledger regression', name, op, target).passed
    assert type(value) is float and math.isnan(value)
    for comparison in ('<', '<=', '>', '>=', '=='):
        report = SpecReport('ac', 'dut', {name: value})
        assert not report.check('unavailable', name, comparison, 0).passed


@pytest.mark.parametrize('y,code', [
    ([0, 0, 0], 'no_passband_reference'),
    ([1, 0, 1], 'zero_passband_magnitude'),
    ([0, 1, 0], 'zero_passband_magnitude'),
])
def test_zero_passband_explains_missing_reference_or_unbounded_ripple(y, code):
    with pytest.warns(MetricWarning) as caught:
        value = ac.passband_ripple_db(response(y + [np.nan], [1, 2, 3, 4]), 3)
    assert type(value) is float and math.isnan(value)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == 'passband_ripple_db' and diagnostic.code == code
    assert diagnostic.details['band_max_hz'] == 3
    assert diagnostic.details['sample_count'] == 3
    assert diagnostic.details['zero_sample_count'] == y.count(0)
    assert diagnostic.details['frequency_min_hz'] == 1
    assert diagnostic.details['frequency_max_hz'] == 3
    assert diagnostic.details['magnitude_unit'] == 'V/V'
    assert diagnostic.message and diagnostic.suggestion


@pytest.mark.parametrize('y,code', [
    ([0, np.nan, np.nan], 'nan_magnitude'),
    ([np.nan, 0, 0], 'nan_magnitude'),
    ([0, complex(1, np.nan), 1], 'nan_magnitude'),
    ([0, 1, 1], 'no_passband_reference'),
    ([0, 0, 0], 'no_passband_reference'),
])
def test_stopband_nan_precedes_missing_reference(y, code):
    with pytest.warns(MetricWarning) as caught:
        value = ac.stopband_attenuation_db(response(y), 2)
    assert type(value) is float and math.isnan(value)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == 'stopband_attenuation_db'
    assert diagnostic.code == code
    assert diagnostic.details['stopband_min_hz'] == 2
    assert diagnostic.details['frequency_min_hz'] == 1
    assert diagnostic.details['frequency_max_hz'] == 3
    assert diagnostic.details['sample_count'] == 3
    assert diagnostic.details['magnitude_unit'] == 'V/V'
    if code == 'nan_magnitude':
        assert diagnostic.details['nan_sample_count'] >= 1
        assert 'contains NaNs' in diagnostic.message
        assert 'source result' in diagnostic.suggestion
    else:
        assert diagnostic.details['passband_peak_magnitude'] == 0
        assert 'reference' in diagnostic.message


@pytest.mark.parametrize('y,lower,upper', [
    ([2, 2, 2], True, True),
    ([2, 2, 0], True, False),
    ([0, 2, 2], False, True),
    ([1, np.sqrt(2), 1], True, True),
])
def test_half_power_band_must_not_reach_either_sweep_edge(y, lower, upper):
    with pytest.warns(MetricWarning) as caught:
        value = ac.q_from_peak(response(y, [1, 10, 100]))
    assert math.isnan(value) and len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == 'q_from_peak'
    assert diagnostic.code == 'half_power_band_unbounded'
    assert diagnostic.details['lower_edge_above_threshold'] == lower
    assert diagnostic.details['upper_edge_above_threshold'] == upper
    assert diagnostic.details['threshold_magnitude'] == pytest.approx(max(y) / np.sqrt(2))
    assert diagnostic.details['frequency_min_hz'] == 1
    assert diagnostic.details['frequency_max_hz'] == 100
    assert diagnostic.suggestion


@pytest.mark.parametrize('name', ['gain_at_freq', 'phase_at_freq', 'cmrr_dB'])
@pytest.mark.parametrize('freq', [float('nan'), float('inf'), -float('inf')])
def test_point_queries_reject_nonfinite_requested_frequency(name, freq):
    sig = response([1, 2, 3])
    args = (sig, sig) if name == 'cmrr_dB' else (sig,)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError) as error:
            getattr(ac, name)(*args, freq=freq)
    message = str(error.value)
    assert name in message and 'freq' in message
    assert str(freq) in message and 'finite' in message
    assert not caught


@pytest.mark.parametrize('name', ['gain_at_freq', 'phase_at_freq', 'cmrr_dB'])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('freq', [0, 100, np.nextafter(1., -np.inf),
                                  np.nextafter(10., np.inf)])
def test_point_queries_strictly_outside_coverage(name, reverse, freq):
    sig = response([2 + 2j, 1 + 1j], [1, 10])
    if reverse:
        sig = response(sig.y[::-1], sig.x[::-1])
    args = (sig, sig) if name == 'cmrr_dB' else (sig,)
    with pytest.warns(MetricWarning) as caught:
        value = getattr(ac, name)(*args, freq=freq)
    assert type(value) is float and math.isnan(value)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == name and diagnostic.code == 'frequency_outside_sweep'
    assert diagnostic.details['requested_frequency_hz'] == freq
    assert 'outside' in diagnostic.message and diagnostic.suggestion
    prefix = 'vdiff_' if name == 'cmrr_dB' else ''
    assert diagnostic.details[prefix + 'frequency_min_hz'] == 1
    assert diagnostic.details[prefix + 'frequency_max_hz'] == 10


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('freq,index', [(1, 0), (10, 1), (4, 0), (7, 1)])
def test_in_range_points_keep_nearest_sample_values(reverse, freq, index):
    sig = response([2 + 0j, 1j], [1, 10])
    common = response([.1j, .2], [1, 10])
    if reverse:
        sig = response(sig.y[::-1], sig.x[::-1])
        common = response(common.y[::-1], common.x[::-1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert ac.gain_at_freq(sig, freq, interpolate=False) == [2., 1.][index]
        assert ac.phase_at_freq(sig, freq, interpolate=False) == [0., 90.][index]
        assert ac.cmrr_dB(sig, common, freq) == pytest.approx(
            20 * np.log10([20., 5.][index]))
    assert not caught


@pytest.mark.parametrize('freq', [1, 10])
@pytest.mark.parametrize('narrow_slot', ['vdiff', 'vcm'])
def test_cmrr_requires_coverage_in_both_signals(freq, narrow_slot):
    signals = dict(vdiff=response([10, 10], [1, 10]),
                   vcm=response([1, 1], [1, 10]))
    signals[narrow_slot] = response([1, 1], [2, 9])
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(ac.cmrr_dB(**signals, freq=freq))
    diagnostic = caught[0].message
    assert diagnostic.details[narrow_slot + '_frequency_min_hz'] == 2
    assert diagnostic.details[narrow_slot + '_frequency_max_hz'] == 9


def test_valid_controls_and_unconsumed_samples_do_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert ac.passband_ripple_db(response([1, 1, np.nan]), 2) == 0
        assert ac.stopband_attenuation_db(response([1, 0, 0]), 2) == math.inf
        assert ac.stopband_attenuation_db(response([1, 1, 1]), 2) == 0
        assert ac.q_from_peak(response([0, 1, 1, 0], [1, 2, 3, 4])) == 2
        sig = response([2, np.nan, 1])
        assert ac.gain_at_freq(sig, 1) == 2
        assert ac.phase_at_freq(sig, 1) == 0
        assert ac.cmrr_dB(sig, sig, 1) == 0
        assert ac.cmrr_dB(response([10, 10], [1, 10]),
                          response([1, 1], [2, 9])) == 20
    assert not caught


def test_q_distinguishes_invalid_axis_from_descending_resonance():
    with pytest.raises(ValueError, match='axis.*repeats.*needs'):
        ac.q_from_peak(response([2, 2, 2], [1, 1, 1]))
    assert ac.q_from_peak(response([0, 1, 1, 0], [4, 3, 2, 1])) == 2.


@pytest.mark.parametrize('name,args', [
    ('passband_ripple_db', (response([1, 0, 1]), 3)),
    ('stopband_attenuation_db', (response([0, 1, 1]), 2)),
    ('q_from_peak', (response([2, 2, 2]),)),
    ('gain_at_freq', (response([1, 1, 1]), 4)),
    ('phase_at_freq', (response([1, 1, 1]), 4)),
    ('cmrr_dB', (response([1, 1, 1]), response([1, 1, 1]), 4)),
])
def test_warning_policy_preserves_float_or_raises_explicitly(name, args):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        assert math.isnan(getattr(ac, name)(*args))
    with warnings.catch_warnings():
        warnings.simplefilter('error', MetricWarning)
        with pytest.raises(MetricWarning) as caught:
            getattr(ac, name)(*args)
    assert caught.value.metric == name


@pytest.mark.parametrize('name,edge,found', [
    ('passband_ripple_db', 0, 0), ('passband_ripple_db', 1, 1),
    ('stopband_attenuation_db', 0, 0), ('stopband_attenuation_db', 4, 0),
])
def test_malformed_bands_keep_requested_found_and_available_context(name, edge, found):
    with pytest.raises(ValueError) as caught:
        getattr(ac, name)(response([1, 1, 1]), edge)
    message = str(caught.value)
    assert name in message and f'{edge:g}' in message
    assert f'contains {found} of 3' in message
    assert '1 Hz .. 3 Hz' in message


@pytest.mark.parametrize('other_x', [[2, 3, 4], [1, 2, 4], [1, 2]])
def test_psrr_axis_mismatch_has_both_available_grids(other_x):
    with pytest.raises(ValueError) as caught:
        ac.psrr_db(response([1, 1, 1]), response(np.ones(len(other_x)), other_x))
    message = str(caught.value)
    assert 'psrr_db' in message and 'same frequency axis' in message
    assert 'v_supply' in message and 'v_out' in message
    assert '3 points' in message and f'{len(other_x)} points' in message
    assert '1 Hz .. 3 Hz' in message
    assert f'{min(other_x):g} Hz .. {max(other_x):g} Hz' in message


def test_metric_schema_stays_byte_identical():
    payload = json.dumps(export_metric_schema(), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode()
    # TASK-223 deliberately added propagation_delay.out_edge; digest bumped.
    # TASK-288 deliberately corrected integrated_noise's unit; digest bumped again.
    # TASK-326 deliberately added bandwidth_3dB's reference={'dc','peak'}
    # param (default 'dc', BREAKING); digest bumped again.
    # TASK-364 replaced x_when with crossing (threshold, start/end); bumped again.
    # TASK-377 renamed minimum/maximum and the ten windowed primitives' start/end; bumped again.
    assert hashlib.sha256(payload).hexdigest() == (
        # TASK-226.22: required scalar eye threshold; centering default True.
        'f47ca45cb4a0666deb6a955dfe495e17888bfa63308be96cb15e27966e65c040'  # TASK-417 jitter; TASK-418 long_term_jitter units/reference/phase/threshold/edge/start/end; TASK-412 removed input_referred_offset, kickback, hold_droop; TASK-379 variance; TASK-386 relative_error_norm; TASK-399 static_errors wrappers; TASK-385 error_norm mean_abs + start/end; TASK-377 windowed_min/max -> minimum/maximum, tstart/tstop -> start/end; TASK-233 eight wrappers; TASK-323.3 final_value; TASK-364 crossing; TASK-369 evm_percent; TASK-365 rise/fall/slew edge params; TASK-363 propagation_delay; TASK-366 period/frequency; TASK-226.20 crossing_jitter_rms; TASK-367 average_duty, pulse_width, on/off_time; TASK-438 hysteresis n_rising/n_falling/start/end
    )
