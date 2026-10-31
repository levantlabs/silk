"""Unavailable edge/decision measurements explain their consumed data once."""
import hashlib
import json
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal
from silk.signals.schema import export_metric_schema


def signal(y, *, x=None, name='waveform', xlabel='time', ylabel='V'):
    return Signal(name, 'voltage', xlabel, ylabel,
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float),
                  np.asarray(y))


FLAT = signal([0., 0., 0.])
RISE = signal([0., .5, 1.])
FALL = signal([1., .5, 0.])
NAN = signal([0., np.nan, 1.])

# Every assigned warning-needed ledger branch, including both signal roles
# of missing decisions and both interpolation routes through hysteresis.
CASES = [
    ('rise_time', [FLAT], {}, 'zero_swing'),
    ('rise_time', [FALL], {}, 'missing_crossing'),
    ('rise_time', [signal([1., 0., .5])], {}, 'missing_crossing'),
    ('rise_time', [NAN], {}, 'nan_waveform'),
    ('fall_time', [FLAT], {}, 'zero_swing'),
    ('fall_time', [RISE], {}, 'missing_crossing'),
    ('fall_time', [signal([0., 1., .5])], {}, 'missing_crossing'),
    ('fall_time', [NAN], {}, 'nan_waveform'),
    ('settling_time', [FLAT], {'initial': 0, 'target': 1}, 'not_settled'),
    ('propagation_delay', [FLAT, RISE], {}, 'missing_crossing'),
    ('propagation_delay', [RISE, FLAT], {}, 'missing_crossing'),
    ('propagation_delay', [NAN, RISE], {}, 'nan_waveform'),
    ('average_duty', [FLAT], {}, 'insufficient_crossings'),
    ('average_duty', [RISE], {}, 'insufficient_crossings'),
    ('average_duty', [NAN], {}, 'nan_waveform'),
    ('average_period', [FLAT], {}, 'insufficient_crossings'),
    ('average_period', [RISE], {}, 'insufficient_crossings'),
    ('average_period', [NAN], {}, 'nan_waveform'),
    ('pulse_width', [FLAT], {}, 'insufficient_crossings'),
    ('pulse_width', [RISE], {}, 'insufficient_crossings'),
    ('pulse_width', [NAN], {}, 'nan_waveform'),
    ('hysteresis', [RISE, FLAT, FALL, FALL], {}, 'missing_crossing'),
    ('hysteresis', [RISE, RISE, FALL, FLAT], {}, 'missing_crossing'),
    ('hysteresis', [RISE, NAN, FALL, FALL], {}, 'nan_waveform'),
    ('hysteresis', [NAN, RISE, FALL, FALL], {}, 'nan_waveform'),
    ('hysteresis', [RISE, RISE, NAN, FALL], {}, 'nan_waveform'),
    ('crossing', [FLAT, .5], {'edge': 'any'}, 'insufficient_crossings'),
    ('crossing', [RISE, .5], {'n': 2, 'edge': 'any'}, 'insufficient_crossings'),
    ('crossing', [NAN, .5], {'edge': 'any'}, 'nan_waveform'),
]


@pytest.mark.parametrize('name,args,kwargs,code', CASES)
@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_unavailable_branch_and_warning_policy(name, args, kwargs, code, policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning) as exc:
                getattr(M, name)(*args, **kwargs)
            diagnostic = exc.value
        else:
            result = getattr(M, name)(*args, **kwargs)
            assert type(result) is float and math.isnan(result)
            assert len(caught) == (1 if policy == 'always' else 0)
            if policy == 'ignore':
                return
            diagnostic = caught[0].message
    assert isinstance(diagnostic, MetricWarning)
    assert diagnostic.metric == name
    assert diagnostic.code == code
    assert diagnostic.suggestion
    details = diagnostic.details
    assert details['sample_count'] > 0
    assert details['signal_name'] == 'waveform'
    assert all(type(v) in (str, float, int, type(None)) for v in details.values())
    if name == 'crossing':
        assert details['x_unit'] == 's'
        assert details['x_start'] == 0 and details['x_stop'] == 2
    else:
        assert 0 <= details['time_start_s'] <= details['time_stop_s'] <= 2
    if code == 'nan_waveform':
        assert details['nan_sample_count'] == 1
        assert 'missing' not in diagnostic.message.lower()
    else:
        assert details['value_unit'] == 'V'


@pytest.mark.parametrize('name,y,fraction,edge,start', [
    ('rise_time', [1., .5, 0.], 'low', 'rising', None),
    ('rise_time', [1., 0., .5], 'high', 'rising', 1),
    ('fall_time', [0., .5, 1.], 'high', 'falling', None),
    ('fall_time', [0., 1., .5], 'low', 'falling', 1),
])
def test_missing_edge_threshold_and_search_scope(name, y, fraction, edge, start):
    with pytest.warns(MetricWarning) as caught:
        getattr(M, name)(signal(y, ylabel='A'))
    d = caught[0].message.details
    assert d['threshold_role'] == fraction
    assert d['threshold'] == pytest.approx(.1 if fraction == 'low' else .9)
    assert d['threshold_fraction'] == pytest.approx(.1 if fraction == 'low' else .9)
    assert d['value_unit'] == 'A' and d['edge'] == edge
    assert d['sample_count'] == 3  # Both thresholds consume the whole trace.
    if start is not None:
        assert d['search_start_s'] == start
        assert d['search_sample_count'] == 2


@pytest.mark.parametrize('name', ['rise_time', 'fall_time', 'average_duty'])
@pytest.mark.parametrize('y,count', [([np.nan], 1), ([np.nan, np.nan, np.nan], 3),
                                     ([0., 1., np.nan], 1)])
def test_nan_threshold_is_not_reported_as_a_physically_absent_edge(name, y, count):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert math.isnan(getattr(M, name)(signal(y)))
    assert len(caught) == 1
    d = caught[0].message
    assert isinstance(d, MetricWarning)
    assert d.code == 'nan_waveform'
    assert d.details['sample_count'] == len(y)
    assert d.details['nan_sample_count'] == count
    assert d.details['time_start_s'] == 0
    assert d.details['time_stop_s'] == len(y) - 1


@pytest.mark.parametrize('name,args,kwargs,role,edge,level', [
    ('propagation_delay', [FLAT, RISE], {'in_threshold': .25, 'out_threshold': .25}, 'sig_in', 'rising', .25),
    ('propagation_delay', [RISE, FLAT], {'in_threshold': .25, 'out_threshold': .25, 'in_edge': 'any'}, 'sig_out', 'any', .25),
    ('propagation_delay', [RISE, FALL], {'in_edge': 'falling'}, 'sig_in', 'falling', .5),
    ('propagation_delay', [FLAT, RISE], {'in_threshold': .2, 'out_threshold': .8}, 'sig_in', 'rising', .2),
    ('propagation_delay', [RISE, FLAT], {'in_threshold': .2, 'out_threshold': .8}, 'sig_out', 'rising', .8),
    ('hysteresis', [RISE, FLAT, FALL, FALL], {'out_threshold': .3}, 'sig_out_rising', 'rising', .3),
    ('hysteresis', [RISE, RISE, FALL, FLAT], {}, 'sig_out_falling', 'falling', .5),
])
def test_missing_decision_identifies_signal_and_requested_threshold(name, args, kwargs, role, edge, level):
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(getattr(M, name)(*args, **kwargs))
    assert len(caught) == 1
    d = caught[0].message.details
    assert d['signal_role'] == role and d['edge'] == edge
    assert d['threshold'] == level
    assert (d['sample_count'], d['time_start_s'], d['time_stop_s']) == (3, 0, 2)


# TASK-367: pulse_width counts complete pulses; a missing rising or falling
# crossing is one case, too few pulses for the requested n.
@pytest.mark.parametrize('sig', [FLAT, RISE])
def test_missing_pulse_reports_requested_threshold_and_count(sig):
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.pulse_width(sig, threshold=.3))
    assert len(caught) == 1
    d = caught[0].message.details
    assert d['threshold'] == .3 and d['polarity'] == 'positive'
    assert d['pulse_count'] == 0 and d['requested_pulse'] == 1
    assert (d['sample_count'], d['time_start_s'], d['time_stop_s']) == (3, 0, 2)


@pytest.mark.parametrize('y,count', [([0., 0., 0.], 0), ([0., 1., 0.], 1)])
def test_period_reports_observed_and_required_crossing_counts(y, count):
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.average_period(signal(y), .5))
    d = caught[0].message.details
    assert d['crossing_count'] == count and d['required_crossing_count'] == 2
    assert d['threshold'] == .5 and d['value_unit'] == 'V'


def test_settling_final_out_of_band_diagnostic_preserves_window_scope():
    with pytest.warns(MetricWarning) as caught:
        result = M.settling_time(signal([np.nan, 1., 0.]), initial=0, target=1, tol=.1, start=1)
    assert math.isnan(result) and len(caught) == 1
    d = caught[0].message
    assert d.code == 'not_settled'
    assert d.details == {
        'signal_name': 'waveform', 'sample_count': 2,
        'time_start_s': 1., 'time_stop_s': 2., 'requested_start_s': 1,
        'target': 1, 'band': .1, 'final_value': 0., 'value_unit': 'V',
    }


@pytest.mark.parametrize('side', ['rising', 'falling'])
@pytest.mark.parametrize('at_sample', [False, True])
def test_hysteresis_diagnoses_only_selected_interpolation_samples(side, at_sample):
    # At threshold=.5, both directions select t=1 exactly when at_sample.
    threshold = .5
    rising = signal([0., .5, 1.]) if at_sample else signal([0., 1., 1.])
    falling = signal([1., .5, 0.]) if at_sample else signal([1., 0., 0.])
    contaminated = signal([np.nan, np.nan, np.nan])
    args = [contaminated if side == 'rising' else RISE, rising,
            contaminated if side == 'falling' else FALL, falling]
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.hysteresis(*args, out_threshold=threshold))
    assert len(caught) == 1
    d = caught[0].message.details
    assert d['signal_role'] == 'sig_in_' + side
    assert d['purpose'] == 'decision_interpolation'
    assert d['sample_count'] == (1 if at_sample else 2)
    assert d['nan_sample_count'] == d['sample_count']
    assert d['time_start_s'] == (1 if at_sample else 0)
    assert d['time_stop_s'] == 1


def test_hysteresis_exact_samples_ignore_nan_neighbors():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = M.hysteresis(signal([np.nan, 2., np.nan]), RISE,
                              signal([np.nan, 1., np.nan]), FALL)
    assert result == 1.


@pytest.mark.parametrize('y,code,observation', [
    ([0., 0., 0.], 'zero_swing', 'swing is zero'),
    ([0., np.nan, 1.], 'nan_waveform', 'consumed data contains NaNs'),
    ([1., .5, 0.], 'missing_crossing', 'low threshold'),
    ([1., 0., .5], 'missing_crossing', 'high threshold'),
])
@pytest.mark.parametrize('name', ['rise_time', 'slew_rate'])
@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_rising_edge_warning_keeps_public_name_and_observation(
    y, code, observation, name, policy,
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning) as exc:
                getattr(M, name)(signal(y))
            diagnostic = exc.value
        else:
            assert math.isnan(getattr(M, name)(signal(y)))
            assert len(caught) == (1 if policy == 'always' else 0)
            if policy == 'ignore':
                return
            diagnostic = caught[0].message
    assert diagnostic.metric == name
    assert diagnostic.code == code
    assert observation in diagnostic.message
    assert diagnostic.details['sample_count'] == 3
    assert diagnostic.details['time_start_s'] == 0
    assert diagnostic.details['time_stop_s'] == 2
    assert diagnostic.suggestion
    if code == 'zero_swing':
        assert diagnostic.details['value_min'] == 0
        assert diagnostic.details['value_max'] == 0
        assert diagnostic.details['value_unit'] == 'V'


@pytest.mark.parametrize('name,args,kwargs,expected', [
    ('rise_time', [RISE], {}, 1.6),
    ('fall_time', [FALL], {}, 1.6),
    ('slew_rate', [RISE], {}, .5),
    ('settling_time', [RISE], {'initial': 0, 'target': 1}, 2.),
    ('settling_time', [FLAT], {'initial': 1, 'target': 0}, 0.),
    ('propagation_delay', [RISE, RISE], {}, 0.),
    ('propagation_delay', [RISE, signal([0., .5, 1.], x=[1., 2., 3.])], {}, 1.),
    ('average_duty', [signal([0., 1., 0., 1., 0.])], {}, .5),
    # TASK-367: RISE holds no full cycle, so average_duty is nan with a
    # warning (the old partial-cycle fallback is gone); on_time answers.
    ('on_time', [RISE], {}, .5),
    ('on_time', [RISE], {'threshold': 2}, 0.),
    ('on_time', [RISE], {'threshold': -1}, 2.),
    ('average_period', [signal([0., 1., 0., 1., 0.])], {}, 2.),
    ('pulse_width', [signal([0., 1., 0.])], {}, 1.),
    ('hysteresis', [RISE, RISE, FALL, FALL], {}, 0.),
    ('propagation_delay', [RISE, RISE], {'in_threshold': .2, 'out_threshold': .8}, 1.2),
    ('crossing', [RISE, .5], {'edge': 'any'}, 1.),
    ('crossing', [signal([0., 1., 0., 1., 0.]), .5], {'n': 3, 'edge': 'any'}, 2.5),
])
def test_successful_values_shapes_zero_and_infinity_stay_quiet(name, args, kwargs, expected):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = getattr(M, name)(*args, **kwargs)
    assert type(result) is float
    assert result == pytest.approx(expected)


@pytest.mark.parametrize('nan_data', [False, True])
@pytest.mark.parametrize('xlabel,unit', [('frequency', 'Hz'), ('time', 's'), ('V', 'V')])
def test_crossing_reports_native_axis_units_and_complex_magnitude(nan_data, xlabel, unit):
    y = [0j, complex(1, np.nan) if nan_data else 1j, 2j]
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(M.crossing(signal(y, x=[10., 20., 30.], xlabel=xlabel).mag(), 1., n=2, edge='any'))
    d = caught[0].message
    assert d.code == ('nan_waveform' if nan_data else 'insufficient_crossings')
    assert d.details['x_unit'] == unit
    assert d.details['x_start'] == 10 and d.details['x_stop'] == 30
    assert d.details['requested_crossing'] == 2
    if nan_data:
        assert d.details['nan_x_start'] == d.details['nan_x_stop'] == 20
    else:
        assert d.details['crossing_count'] == 1
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert M.crossing(signal(y, x=[10., 20., 30.], xlabel=xlabel).mag(), 1., edge='any') == 20


@pytest.mark.parametrize('name,args,kwargs', [
    ('rise_time', [NAN], {'low': .9, 'high': .1}),
    ('fall_time', [NAN], {'low': -1}),
    ('propagation_delay', [NAN, NAN], {'in_edge': 'typo'}),
    ('propagation_delay', [NAN, NAN], {'in_n': 0}),
    ('propagation_delay', [NAN, NAN], {'out_n': 1.5}),
    ('propagation_delay', [NAN, NAN], {'start': 5.}),
    ('crossing', [NAN, .5], {'n': 0}),
    ('crossing', [NAN, .5], {'edge': 'typo'}),
    ('settling_time', [NAN], {'initial': 0, 'target': 1, 'start': 10}),
    ('propagation_delay', [NAN, NAN], {'out_threshold': 'typo'}),
])
def test_existing_caller_errors_precede_diagnostics(name, args, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError):
            getattr(M, name)(*args, **kwargs)


def test_metric_schema_canonical_digest_is_unchanged():
    canonical = json.dumps(export_metric_schema(), sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode()
    # TASK-223 deliberately added propagation_delay.out_edge; digest bumped.
    # TASK-288 deliberately corrected integrated_noise's unit; digest bumped again.
    # TASK-326 deliberately added bandwidth_3dB's reference={'dc','peak'}
    # param (default 'dc', BREAKING); digest bumped again.
    # TASK-364 replaced x_when with crossing (threshold optional, start/end
    # added, edge default rising); digest bumped again.
    assert hashlib.sha256(canonical).hexdigest() == (
        # TASK-226.22: required scalar eye threshold; centering default True.
        'f47ca45cb4a0666deb6a955dfe495e17888bfa63308be96cb15e27966e65c040'  # TASK-417 jitter; TASK-418 long_term_jitter units/reference/phase/threshold/edge/start/end; TASK-412 removed input_referred_offset, kickback, hold_droop; TASK-379 variance; TASK-386 relative_error_norm; TASK-378 interpolate; TASK-399 static_errors wrappers; TASK-385 error_norm mean_abs + start/end; TASK-377 minimum/maximum + start/end; TASK-233 eight wrappers; TASK-323.3 final_value; TASK-364 crossing; TASK-369 evm_percent; TASK-365 rise/fall/slew edge params; TASK-363 propagation_delay; TASK-366 period/frequency; TASK-226.20 crossing_jitter_rms; TASK-367 average_duty, pulse_width, on/off_time; TASK-438 hysteresis n_rising/n_falling/start/end
    )
