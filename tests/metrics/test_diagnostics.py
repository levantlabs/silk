"""Unavailable AC measurements retain floats and explain their source condition."""
import ast
import math
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest

from silk.circuit.spec_report import SpecReport
from silk.metrics import bandwidth_3dB, phase_margin, unity_gain_freq
from silk.metrics import diagnostics
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


FREQUENCIES = np.array([1e3, 1e4, 1e5, 1e6])
ABOVE = [100, 50, 20, 2]
BELOW = [.5, .3, .1, .01]
NAN = [10, complex(2, float('nan')), .5, .1]
UNAVAILABLE = [
    (unity_gain_freq, ABOVE, 'no_unity_crossing_gain_above_one'),
    (unity_gain_freq, BELOW, 'no_unity_crossing_gain_below_one'),
    (unity_gain_freq, NAN, 'nan_magnitude'),
    (bandwidth_3dB, NAN, 'nan_magnitude'),
    (bandwidth_3dB, [2, 2, 2, 2], 'upper_rolloff_not_observed'),
]


def response(magnitudes, frequencies=FREQUENCIES, unit='V/V'):
    return Signal('response', 'voltage', 'frequency', unit,
                  np.asarray(frequencies), np.asarray(magnitudes, dtype=complex))


@pytest.mark.parametrize('metric,magnitudes,code', UNAVAILABLE)
def test_unavailable_payload(metric, magnitudes, code):
    with pytest.warns(MetricWarning) as caught:
        value = metric(response(magnitudes))
    assert type(value) is float
    assert math.isnan(value)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == metric.__name__
    assert diagnostic.code == code
    assert diagnostic.message and diagnostic.suggestion
    details = diagnostic.details
    assert details['frequency_min_hz'] == 1e3
    assert details['frequency_max_hz'] == 1e6
    assert details['magnitude_unit'] == 'V/V'
    assert details['signal_name'] == 'response'
    assert details['sample_count'] == 4
    assert all(type(value) in (str, float, int, type(None))
               for value in details.values())
    text = str(diagnostic)
    for fact in ('frequency_min_hz=1000.0', 'frequency_max_hz=1000000.0',
                 "magnitude_unit='V/V'", diagnostic.message, diagnostic.suggestion):
        assert fact in text
    if code == 'nan_magnitude':
        assert details['nan_sample_count'] == 1
        assert details['nan_frequency_min_hz'] == 1e4
        assert details['nan_frequency_max_hz'] == 1e4
        assert details['magnitude_min'] == .1
        assert details['magnitude_max'] == 10
        assert 'contains NaNs' in diagnostic.message
        assert 'source result' in diagnostic.suggestion
    else:
        assert details['nan_sample_count'] == 0
        assert details['magnitude_min'] == min(magnitudes)
        assert details['magnitude_max'] == max(magnitudes)
        if code.endswith('above_one'):
            assert 'above 1 throughout' in diagnostic.message
            assert 'not established' in diagnostic.suggestion
        elif code.endswith('below_one'):
            assert 'below 1 throughout' in diagnostic.message
            assert 'coverage, transfer normalization, and operating point' in diagnostic.suggestion
        else:
            assert details['threshold_magnitude'] == pytest.approx(2 / np.sqrt(2))
            assert details['endpoint_magnitude'] == 2
            assert 'at/above' in diagnostic.message


@pytest.mark.parametrize('magnitudes', [ABOVE, BELOW])
def test_absent_crossing_fails_all_five_spec_operators(magnitudes):
    # Design Appendix A1: 0.0 falsely passed the upper-bound spec.
    with pytest.warns(MetricWarning):
        value = unity_gain_freq(response(magnitudes))
    report = SpecReport('ac', 'amp', {'ugf': value})
    assert type(report.metrics['ugf']) is float
    for op in ('<', '<=', '>', '>=', '=='):
        assert not report.check('spec', 'ugf', op, 100e6).passed


@pytest.mark.parametrize('magnitudes,expected', [
    ([10, 2, .5, .1], 46415.888336127835),
    ([1, .5, .2, .1], 1e3),
    ([10, 5, 2, 1], 1e6),
    ([.1, .5, 2, 10], 21544.346900318837),
    ([1, 1, 1, 1], 1e3),
])
def test_successful_crossings_are_unchanged_and_quiet(magnitudes, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = unity_gain_freq(response(magnitudes))
    assert type(value) is float
    assert value == pytest.approx(expected, rel=1e-12)
    assert caught == []


@pytest.mark.parametrize('magnitude', [2, float('nan')])
def test_one_sample_remains_a_caller_error_without_diagnostic(magnitude):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(ValueError, match=(
                '^unity_gain_freq: need at least 2 frequency points, got 1$')):
            unity_gain_freq(response([magnitude], [1e3]))
    assert caught == []


@pytest.mark.parametrize('metric,magnitudes,code', UNAVAILABLE)
def test_ignored_warning_preserves_numeric_result(metric, magnitudes, code):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore', MetricWarning)
        value = metric(response(magnitudes))
    assert type(value) is float and math.isnan(value)
    assert caught == []


@pytest.mark.parametrize('metric,magnitudes,code', UNAVAILABLE)
def test_warning_as_error_is_explicit_caller_policy(metric, magnitudes, code):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=MetricWarning)
        with pytest.raises(MetricWarning) as caught:
            metric(response(magnitudes))
    assert caught.value.metric == metric.__name__
    assert caught.value.code == code


@pytest.mark.parametrize('magnitudes,code', [
    (ABOVE, 'no_unity_crossing_gain_above_one'),
    (BELOW, 'no_unity_crossing_gain_below_one'),
    (NAN, 'nan_magnitude'),
])
def test_phase_margin_reports_its_originating_diagnostic_once(magnitudes, code):
    with pytest.warns(MetricWarning) as caught:
        value = phase_margin(response(magnitudes))
    assert type(value) is float and math.isnan(value)
    assert len(caught) == 1
    assert caught[0].message.metric == 'unity_gain_freq'
    assert caught[0].message.code == code


@pytest.mark.parametrize('metric', [unity_gain_freq, bandwidth_3dB])
def test_all_nan_has_no_invented_magnitude_bounds(metric):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = metric(response([float('nan')] * 4))
    assert math.isnan(value)
    assert len(caught) == 1  # no numpy all-NaN reduction warnings
    diagnostic = caught[0].message
    assert isinstance(diagnostic, MetricWarning)
    assert diagnostic.code == 'nan_magnitude'
    assert diagnostic.details['magnitude_min'] is None
    assert diagnostic.details['magnitude_max'] is None
    assert diagnostic.details['nan_sample_count'] == 4
    assert diagnostic.details['nan_frequency_min_hz'] == 1e3
    assert diagnostic.details['nan_frequency_max_hz'] == 1e6


@pytest.mark.parametrize('descending', [False, True])
@pytest.mark.parametrize('endpoint', [2, 2 / np.sqrt(2)])
def test_bandwidth_endpoint_threshold_and_units_are_observed(descending, endpoint):
    f = FREQUENCIES[::-1] if descending else FREQUENCIES
    magnitudes = [2, 2, 2, endpoint]
    if descending:
        magnitudes = magnitudes[::-1]
    with pytest.warns(MetricWarning) as caught:
        assert math.isnan(bandwidth_3dB(response(magnitudes, f, unit='A/A')))
    details = caught[0].message.details
    assert details['frequency_min_hz'] == 1e3
    assert details['frequency_max_hz'] == 1e6
    assert details['endpoint_magnitude'] == endpoint
    assert details['threshold_magnitude'] == 2 / np.sqrt(2)
    assert details['magnitude_unit'] == 'A/A'


@pytest.mark.parametrize('descending', [False, True])
def test_measured_bandwidth_keeps_its_grid_value_without_warning(descending):
    f, mag = FREQUENCIES, np.array([10, 8, 7, 1])
    if descending:
        f, mag = f[::-1], mag[::-1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        value = bandwidth_3dB(response(mag, f))
    assert type(value) is float and value == 1e4
    assert caught == []


def test_metric_warning_output_is_none_by_default_and_prefixes_text():
    """``emit()`` never passes ``output`` (TASK-337): the plain form is
    unchanged. Attribution is added only by module_test.py's evaluation
    loop, which builds a second MetricWarning carrying it explicitly."""
    plain = MetricWarning(metric='bandwidth_3dB', code='nan_magnitude',
                          message='msg', details={'k': 1})
    assert plain.output is None
    assert str(plain).startswith('bandwidth_3dB: msg')

    attributed = MetricWarning(metric='bandwidth_3dB', code='nan_magnitude',
                               message='msg', details={'k': 1},
                               output='bw_closed')
    assert attributed.output == 'bw_closed'
    assert str(attributed).startswith("output 'bw_closed' (bandwidth_3dB): msg")


def test_metric_warning_record_round_trips_through_dict():
    from silk.metrics.diagnostics import MetricWarningRecord

    warning = MetricWarning(
        metric='bandwidth_3dB', code='nan_magnitude', message='msg',
        details={'k': 1}, suggestion='try again', output='bw_closed',
    )
    record = MetricWarningRecord.from_warning(warning)
    assert record == MetricWarningRecord(
        metric='bandwidth_3dB', code='nan_magnitude', message='msg',
        details={'k': 1}, suggestion='try again', output='bw_closed',
    )
    as_dict = record.to_dict()
    assert as_dict == {
        'metric': 'bandwidth_3dB', 'code': 'nan_magnitude', 'message': 'msg',
        'details': {'k': 1}, 'suggestion': 'try again', 'output': 'bw_closed',
    }
    assert MetricWarningRecord.from_dict(as_dict) == record


def test_diagnostics_remains_a_stdlib_leaf():
    tree = ast.parse(Path(diagnostics.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split('.')[0] in sys.stdlib_module_names
                       for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert node.module.split('.')[0] in sys.stdlib_module_names


def test_canonical_metric_schema_is_unchanged():
    # A fresh interpreter excludes temporary metrics registered by other tests.
    result = subprocess.run([sys.executable, '-c', """
import hashlib
import json
from pathlib import Path
import silk
assert Path(silk.__file__).resolve().is_relative_to(Path.cwd())
from silk.signals.schema import export_metric_schema
canonical = json.dumps(export_metric_schema(), sort_keys=True,
                       separators=(',', ':'), allow_nan=False)
print(hashlib.sha256(canonical.encode()).hexdigest())
"""], check=True, capture_output=True, text=True)
    # TASK-223 added propagation_delay.out_edge and TASK-210 removed phase_noise_at_offset;
    # TASK-288 corrected integrated_noise's unit; TASK-326 added
    # bandwidth_3dB's reference={'dc','peak'} param (default 'dc',
    # BREAKING); digest bumped each time.
    # TASK-226.22: required scalar eye threshold; centering default True.
    assert result.stdout.strip() == 'f47ca45cb4a0666deb6a955dfe495e17888bfa63308be96cb15e27966e65c040'  # TASK-417 jitter; TASK-418 long_term_jitter units/reference/phase/threshold/edge/start/end; TASK-412 removed input_referred_offset, kickback, hold_droop; TASK-379 variance; TASK-386 relative_error_norm; TASK-378 interpolate; TASK-399 static_errors wrappers; TASK-385 error_norm mean_abs + start/end; TASK-377 minimum/maximum + start/end; TASK-233 eight wrappers; TASK-323.3 final_value; TASK-364 crossing; TASK-369 evm_percent; TASK-365 rise/fall/slew edge params; TASK-363 propagation_delay; TASK-366 period/frequency; TASK-226.20 crossing_jitter_rms; TASK-367 average_duty, pulse_width, on/off_time; TASK-438 hysteresis n_rising/n_falling/start/end


def _notch_loop_gain(f):
    # TASK-295 finding A1 / TASK-309: same loop gain used in
    # docs/archive/reviews/task-295-headline-findings-recheck.py, section
    # A1 — three 0 dB crossings (a notch dips the loop gain back below
    # unity and it recovers above 1 again before rolling off for good).
    return 1000 / (1 + 1j * f / 1e3) / (1 + 1j * f / 3e6) ** 2 * (
        1 - 0.995 * np.exp(-((np.log10(f) - np.log10(3e4)) / 0.08) ** 2))


@pytest.fixture
def notch_loop_orderings():
    f = np.logspace(1, 7, 1201)
    y = _notch_loop_gain(f)
    rng = np.random.RandomState(0)
    shuffled = rng.permutation(len(f))
    return {
        'ascending': response(y, f),
        'descending': response(y[::-1], f[::-1]),
        'shuffled': response(y[shuffled], f[shuffled]),
    }


def test_ugf_and_pm_are_ordering_independent_on_a_notch_loop(notch_loop_orderings):
    # TASK-309 AC#1: ascending and descending orderings of the
    # same notch-loop sweep used to disagree (29.2 kHz/90.8 deg ascending
    # vs. 915 kHz/56.2 deg descending) because both metrics searched in
    # ARRAY order instead of frequency order.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', MetricWarning)
        results = {
            name: (unity_gain_freq(sig), phase_margin(sig))
            for name, sig in notch_loop_orderings.items() if name != 'shuffled'
        }
    ascending = results['ascending']
    assert ascending[0] == pytest.approx(29150.25320105878, rel=1e-9)
    assert ascending[1] == pytest.approx(90.84880625529462, rel=1e-9)
    for name in ('descending',):
        assert results[name][0] == pytest.approx(ascending[0], rel=1e-12)
        assert results[name][1] == pytest.approx(ascending[1], rel=1e-12)


@pytest.mark.parametrize('metric', [unity_gain_freq, phase_margin])
def test_shuffled_frequency_axis_is_a_caller_error(metric, notch_loop_orderings):
    with pytest.raises(ValueError, match='axis.*needs.*only rises or only falls'):
        metric(notch_loop_orderings['shuffled'])


def test_multiple_crossings_keep_the_lowest_and_warn_with_every_margin(notch_loop_orderings):
    # TASK-309 AC#2: three crossings on the notch loop; the lowest-
    # frequency one is returned, and the warning names every crossing
    # frequency and the phase margin measured there.
    sig = notch_loop_orderings['ascending']
    with pytest.warns(MetricWarning) as caught:
        value = unity_gain_freq(sig)
    assert len(caught) == 1
    diagnostic = caught[0].message
    assert diagnostic.metric == 'unity_gain_freq'
    assert diagnostic.code == 'multiple_unity_gain_crossings'
    details = diagnostic.details
    assert details['crossing_count'] == 3
    freqs = details['crossing_frequencies_hz']
    pms = details['phase_margins_deg']
    assert len(freqs) == 3 and len(pms) == 3
    assert freqs == sorted(freqs)
    assert value == freqs[0]
    assert freqs[0] == pytest.approx(29150.25320105878, rel=1e-9)
    assert freqs[1] == pytest.approx(30910.374907822872, rel=1e-9)
    assert freqs[2] == pytest.approx(914918.3215430453, rel=1e-9)
    # the reported margins are the actual phase margin at each crossing,
    # not all copies of the returned (lowest-frequency) one
    assert len(set(round(pm, 6) for pm in pms)) == 3


def test_single_crossing_notch_loop_variant_emits_no_multi_crossing_warning():
    # A single-crossing sweep must stay quiet (AC#2's negative case).
    f = np.logspace(1, 7, 1201)
    y = 1000 / (1 + 1j * f / 1e3) / (1 + 1j * f / 3e6) ** 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        unity_gain_freq(response(y, f))
        phase_margin(response(y, f))
    assert caught == []
