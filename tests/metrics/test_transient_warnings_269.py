"""Clock warnings explain unavailable observations without changing measurements."""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def signal(y, *, x=None):
    return Signal('clock', 'voltage', 'time', 'V',
                  np.arange(len(y), dtype=float) if x is None else np.asarray(x),
                  np.asarray(y, dtype=float))


def clock():
    t = np.arange(65, dtype=float)
    return signal((t % 8 >= 4).astype(float), x=t)


NAN_CASES = [
    ('crossings', {}, 'crossings', 'array'),
    # TASK-417: the jitter pair reads edges through the period-family core,
    # which names the invoking metric (as cycle_frequency does).
    ('period_jitter', {}, 'period_jitter', 'scalar'),
    ('cycle_to_cycle_jitter', {}, 'cycle_to_cycle_jitter', 'scalar'),
    ('on_time', {'threshold': .5}, 'on_time', 'scalar'),
    ('off_time', {'threshold': .5}, 'on_time', 'scalar'),
    ('cycle_frequency', {}, 'cycle_frequency', 'signal'),
    ('tie', {}, 'tie', 'signal'),
    ('long_term_jitter', {}, 'long_term_jitter', 'scalar'),
    ('allan_deviation', {}, 'allan_deviation', 'signal'),
]
EDGE_CASES = [
    ('period_jitter', 3, 'scalar'),
    ('cycle_to_cycle_jitter', 4, 'scalar'),
    # cycle_frequency (was inst_frequency) left this table at TASK-366: it
    # shares the period-family core, which reports insufficient_crossings;
    # tests/metrics/test_period_family_366.py covers it.
    # TASK-418: two edges always lie on the fitted line, so TIE needs three.
    ('tie', 3, 'signal'),
    ('long_term_jitter', 3, 'scalar'),
    ('allan_deviation', 4, 'signal'),
]


def assert_unavailable(result, shape, size=1):
    if shape == 'scalar':
        assert type(result) is float and math.isnan(result)
    elif shape == 'array':
        assert isinstance(result, np.ndarray)
        assert result.shape == (size,)
        assert np.isnan(result).all()
    else:
        assert isinstance(result, Signal)
        assert result.x.shape == result.y.shape == (size,)
        assert np.isnan(result.x).all() and np.isnan(result.y).all()


def diagnostic(caught, origin, code, sig):
    assert len(caught) == 1  # Always-filtered nested calls must not re-explain.
    d = caught[0].message
    assert isinstance(d, MetricWarning)
    assert d.metric == origin
    assert d.code == code
    assert d.message and d.suggestion
    assert d.details['signal_name'] == sig.name
    assert d.details['sample_count'] == len(sig.y)
    assert d.details['time_start_s'] == sig.x[0]
    assert d.details['time_stop_s'] == sig.x[-1]
    assert all(type(v) in (str, float, int, type(None)) for v in d.details.values())
    for text in (origin, code, d.message, d.suggestion, 'time_start_s', 'sample_count'):
        assert text in str(d)
    return d


@pytest.mark.parametrize('name,kwargs,origin,shape', NAN_CASES)
@pytest.mark.parametrize('y', [[0, np.nan, 1, 0, np.nan], [np.nan] * 5])
def test_nan_observations_keep_shape_and_one_origin(name, kwargs, origin, shape, y):
    sig = signal(y, x=np.arange(5) * 2e-9 + 1e-9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = getattr(M, name)(sig, **kwargs)
    assert_unavailable(result, shape)
    d = diagnostic(caught, origin, 'nan_waveform', sig)
    mask = np.isnan(sig.y)
    assert d.details['nan_sample_count'] == mask.sum()
    assert d.details['nan_time_start_s'] == sig.x[mask][0]
    assert d.details['nan_time_stop_s'] == sig.x[mask][-1]
    assert 'contains NaNs' in d.message
    assert 'source result' in d.suggestion


@pytest.mark.parametrize('name,required,shape,edge_count', [
    (name, required, shape, count)
    for name, required, shape in EDGE_CASES for count in range(required)
])
def test_insufficient_observed_edges(name, required, shape, edge_count):
    y = [0] * 9 if edge_count == 0 else [0, 1] * edge_count + [0]
    sig = signal(y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = getattr(M, name)(sig)
    assert_unavailable(result, shape, size=0)
    d = diagnostic(caught, name, 'insufficient_edges', sig)
    assert d.details['edge_count'] == edge_count
    assert d.details['required_edge_count'] == required
    assert 'longer record' in d.suggestion


@pytest.mark.parametrize('name', ['period_jitter', 'cycle_to_cycle_jitter'])
@pytest.mark.parametrize('edge', ['rising', 'falling', 'any'])
@pytest.mark.parametrize('threshold', [None, .5, 2])
def test_jitter_edge_request_context(name, edge, threshold):
    sig = signal([0, 1, 0])
    expected_count = 0 if threshold == 2 else (2 if edge == 'any' else 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = getattr(M, name)(sig, edge=edge, threshold=threshold)
    assert math.isnan(result)
    assert caught[0].message.details['edge'] == edge
    assert caught[0].message.details['edge_count'] == expected_count
    assert len(caught) == 1


@pytest.mark.parametrize('threshold', [None, .5])
@pytest.mark.parametrize('edge', ['rising', 'falling', 'any'])
def test_crossings_enumeration_distinguishes_zero_events_from_unknown(threshold, edge):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        empty = M.crossings(signal([0, 0, 0]), threshold=threshold, edge=edge)
    assert empty.shape == (0,)
    assert caught == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.crossings(signal([0, 1, 0, np.nan]), threshold=threshold, edge=edge)
    assert_unavailable(result, 'array')
    assert len(caught) == 1
    assert caught[0].message.details['sample_count'] == 4


WARNING_CASES = [
    (name, [0, np.nan, 1], kwargs, origin, shape, 1, 'nan_waveform')
    for name, kwargs, origin, shape in NAN_CASES
] + [
    (name, [0, 0, 0], {}, name, shape, 0, 'insufficient_edges')
    for name, required, shape in EDGE_CASES
]


@pytest.mark.parametrize('name,y,kwargs,origin,shape,size,code', WARNING_CASES)
@pytest.mark.parametrize('policy', ['ignore', 'error'])
def test_clock_warning_filters(name, y, kwargs, origin, shape, size, code, policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning) as exc:
                getattr(M, name)(signal(y), **kwargs)
            assert exc.value.metric == origin
            assert exc.value.code == code
            return
        result = getattr(M, name)(signal(y), **kwargs)
    assert caught == []
    assert_unavailable(result, shape, size=size)


def test_successful_clock_measurements_and_valid_zeros_stay_quiet():
    sig = clock()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        np.testing.assert_array_equal(M.crossings(sig), np.arange(3.5, 64, 8))
        assert M.period_jitter(sig) == 0.0
        assert M.cycle_to_cycle_jitter(sig) == 0.0
        assert M.long_term_jitter(sig) == 0.0
        for result, metadata, x, y in [
            (M.cycle_frequency(sig), ('cycle_frequency(clock)', 'frequency', 'time', 'Hz'),
             np.arange(7.5, 60, 8), np.full(7, .125)),
            (M.tie(sig), ('tie(clock)', 'time_error', 'time', 's'),
             np.arange(3.5, 64, 8), np.zeros(8)),
            (M.allan_deviation(sig), ('allan(clock)', 'allan_dev', 'tau', ''),
             [8], [0]),
        ]:
            assert (result.name, result.type, result.xlabel, result.ylabel) == metadata
            np.testing.assert_array_equal(result.x, x)
            np.testing.assert_array_equal(result.y, y)
        assert M.on_time(sig, .5) == M.off_time(sig, .5) == 32.0
        assert M.on_time(sig, 2) == M.off_time(sig, -1) == 0.0


def test_nonzero_jitter_and_frequency_numerics_stay_quiet():
    sig = signal([0, 1, 0, 1, 0, 1, 0, 1], x=[0, 2, 3, 5, 7, 11, 14, 20])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # Rising edges 1, 4, 9, 17: periods 3, 5, 8 (TASK-417: RMS about
        # the mean, ddof=0; c2c is the RMS of the differences 2, 3).
        assert M.period_jitter(sig) == pytest.approx(math.sqrt(38 / 9))
        assert M.cycle_to_cycle_jitter(sig) == pytest.approx(math.sqrt(6.5))
        result = M.cycle_frequency(sig)
        np.testing.assert_array_equal(result.x, [2.5, 6.5, 13])
        np.testing.assert_allclose(result.y, [1 / 3, 1 / 5, 1 / 8])


@pytest.mark.parametrize('name,kwargs', [
    ('crossings', {'edge': 'bad'}), ('period_jitter', {'edge': 'bad'}),
    ('cycle_to_cycle_jitter', {'edge': 'bad'}), ('tie', {'ideal_period': 0}),
    ('long_term_jitter', {'ideal_period': -1}), ('allan_deviation', {'taus': [0]}),
])
def test_caller_errors_remain_ahead_of_warnings(name, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError):
            getattr(M, name)(signal([0, np.nan, 1]), **kwargs)


def test_nan_parameter_transport_does_not_gain_an_invented_waveform_reason():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert np.isnan(M.tie(clock(), ideal_period=np.nan).y).all()
        assert math.isnan(M.long_term_jitter(clock(), ideal_period=np.nan))


def assert_eye_placeholders(result):
    for name in ('height', 'width'):
        assert math.isnan(getattr(result, name))


@pytest.mark.parametrize('value', [0., np.nan, np.inf])
@pytest.mark.parametrize('n_traces', [None, 2])
def test_eye_refusal_warns_once_and_preserves_original_fold(value, n_traces):
    sig = signal(np.full(65, value))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        original = M.eye_diagram(sig, period=8, n_traces=n_traces)
    assert caught == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.eye_diagram(sig, period=8, n_traces=n_traces, auto_center=True)
    d = diagnostic(caught, 'eye_diagram', 'centering_unavailable', sig)
    assert d.message == 'Centering could not be determined from the supplied data.'
    assert d.details['period_s'] == 8
    assert d.details['full_period_count'] == 8
    assert d.details['nonfinite_sample_count'] == (0 if value == 0 else 65)
    assert 'auto_center=False' in d.suggestion
    assert math.isnan(result.sampling_instant_s)
    assert_eye_placeholders(result)
    assert len(result.traces) == len(original.traces) == (n_traces or 8)
    for folded, expected in zip(result.traces, original.traces):
        assert (folded.name, folded.type, folded.xlabel, folded.ylabel) == (
            expected.name, expected.type, expected.xlabel, expected.ylabel)
        assert folded.x.dtype == expected.x.dtype
        assert folded.y.dtype == expected.y.dtype
        assert folded.x.tobytes() == expected.x.tobytes()
        assert folded.y.tobytes() == expected.y.tobytes()


@pytest.mark.parametrize('value', [0., np.nan])
@pytest.mark.parametrize('policy', ['ignore', 'error'])
def test_eye_warning_filters(value, policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning, match='eye_diagram'):
                M.eye_diagram(signal(np.full(65, value)), period=8, auto_center=True)
            return
        result = M.eye_diagram(signal(np.full(65, value)), period=8, auto_center=True)
    assert caught == []
    assert math.isnan(result.sampling_instant_s)
    assert len(result.traces) == 8


def test_successful_eye_centering_has_no_warning_for_placeholder_nans():
    # Alternating plateaus with transitions at .1-.2 UI, centre at .65 UI.
    cycles = np.arange(-2, 23)
    knots = (cycles[:, None] + [.1, .2]).ravel()
    values = np.repeat(np.where(cycles % 2, 1., -1.), 2)
    values[::2] *= -1
    t = np.r_[0., knots[(knots > 0) & (knots < 20)], 20.]
    sig = signal(np.interp(t, knots, values), x=t)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = M.eye_diagram(sig, period=1., auto_center=True, threshold=0.)
    assert result.sampling_instant_s == pytest.approx(.65, abs=1 / 1024)
    assert len(result.traces) == 19
    assert result.height == pytest.approx(2.)
    assert result.width == pytest.approx(1.)


def test_eye_analysis_uses_full_input_even_when_output_count_is_limited():
    sig = clock()
    sig.y[-1] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = M.eye_diagram(sig, period=8, n_traces=2, auto_center=True)
    d = diagnostic(caught, 'eye_diagram', 'centering_unavailable', sig)
    assert d.details['nonfinite_sample_count'] == 1
    assert all(np.isfinite(trace.y).all() for trace in result.traces)
    assert math.isnan(result.sampling_instant_s)


@pytest.mark.parametrize('kwargs', [
    {'period': 0}, {'period': 100}, {'period': 8, 'n_traces': 100},
])
def test_eye_request_errors_precede_centering_diagnostics(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError):
            M.eye_diagram(signal(np.full(65, np.nan)), auto_center=True, **kwargs)
