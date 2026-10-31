"""Average-voltage crossing alignment and unchanged original-sample folding.

Band/maximin assertions were intentionally retired with TASK-226.17.
"""

import math
import warnings

import numpy as np
import pytest

from silk.metrics import eye_diagram
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def signal(t, y):
    return Signal('rx', 'voltage', 'time', 'V', np.asarray(t), np.asarray(y))


def ringing(levels, *, rare=False, counts=None):
    """The archived skew/ringing channel, with an optional 2% intermediate level.

    The analyzer never receives the transmitted symbols or level count. Seed 5
    and 400 vertices/UI reproduce the independent review's channel fixture.
    """
    levels = np.asarray(levels)
    rng = np.random.default_rng(5)
    if rare:
        counts = np.full(len(levels), 588 // (len(levels) - 1))
        counts[1] = 12
    if counts is not None:
        symbols = np.repeat(levels, counts)
        rng.shuffle(symbols)
    else:
        symbols = rng.choice(levels, 600)
    osr = 400
    t = np.arange(len(symbols) * osr) / osr
    ideal = np.repeat(symbols, osr)
    for i in range(1, len(symbols)):
        delay = round((0.05 + 0.30 * symbols[i] / levels.max()) * osr)
        ideal[i * osr:i * osr + delay] = symbols[i - 1]
    wn, damping, dt = 2 * np.pi * 1.4, 0.35, 1 / osr
    y = np.empty_like(ideal, dtype=float)
    position, velocity = float(ideal[0]), 0.0
    for i, target in enumerate(ideal):
        velocity += (wn * wn * (target - position)
                     - 2 * damping * wn * velocity) * dt
        position += velocity * dt
        y[i] = position
    return signal(t, y)


def trapezoidal_nrz(*, phase=0.0, periods=20, ui=1.0, origin=0.0):
    """Transitions at .1-.2 UI; average-voltage crossings are at .15 UI.

    The midpoint is .65 UI. Samples are vertices of the actual continuous
    piecewise-linear signal; additional subdivisions must not move this centre.
    """
    times = np.arange(-2, periods + 3)[:, None] + np.array([0.1, 0.2])
    values = np.repeat(np.where(np.arange(-2, periods + 3) % 2, 1., -1.), 2)
    values[::2] *= -1
    times = times.ravel() + phase
    keep = (times > 0) & (times < periods)
    t = np.r_[0., times[keep], float(periods)]
    y = np.interp(t, times, values)
    return signal(origin + t * ui, y)


def circular_distance(a, b, period=1.0):
    return abs((a - b + period / 2) % period - period / 2)


def assert_same_fold(a, b):
    assert len(a.traces) == len(b.traces)
    for left, right in zip(a.traces, b.traces):
        assert left.name == right.name
        np.testing.assert_array_equal(left.x, right.x)
        assert left.y.dtype == right.y.dtype
        assert left.y.tobytes() == right.y.tobytes()


def test_nrz_plateau_is_centred_in_returned_original_samples():
    sig = trapezoidal_nrz()
    result = eye_diagram(sig, period=1., auto_center=True, threshold=0.)
    assert result.sampling_instant_s == pytest.approx(.65, rel=0, abs=2e-12)
    assert len(result.traces) == 19
    shift = (result.sampling_instant_s - .5) % 1
    recovered_t, recovered_y = [], []
    for k, trace in enumerate(result.traces):
        assert np.all(np.diff(trace.x) > 0)
        assert np.all((trace.x >= 0) & (trace.x < 1))
        recovered_t.extend(trace.x + shift + k)
        recovered_y.extend(trace.y)
        take = (sig.x >= shift + k) & (sig.x < shift + k + 1)
        np.testing.assert_array_equal(trace.x, sig.x[take] - (shift + k))
        assert trace.y.tobytes() == sig.y[take].tobytes()
    take = (sig.x >= shift) & (sig.x < shift + len(result.traces))
    np.testing.assert_allclose(recovered_t, sig.x[take], rtol=0, atol=4e-15)
    np.testing.assert_array_equal(recovered_y, sig.y[take])
    assert result.height == pytest.approx(2.)
    assert result.width == pytest.approx(1.)


def test_explicit_count_must_fit_after_shifting():
    sig = trapezoidal_nrz(periods=4)
    with pytest.raises(ValueError, match=r'(?i)(shift|centr).*(3|three)'):
        eye_diagram(sig, period=1., auto_center=True, n_traces=4)
    result = eye_diagram(sig, period=1., auto_center=True, n_traces=2)
    assert len(result.traces) == 2


def test_explicit_false_preserves_the_original_fold():
    sig = trapezoidal_nrz()
    assert_same_fold(eye_diagram(sig, period=1., n_traces=5),
                     eye_diagram(sig, period=1., n_traces=5, auto_center=False))


def held_pam(levels, periods=40):
    levels = np.asarray(levels, dtype=float)
    cycle = np.arange(-2, periods + 3)
    target = levels[cycle % len(levels)]
    previous = levels[(cycle - 1) % len(levels)]
    vertices = (cycle[:, None] + np.array([.1, .2])).ravel()
    values = np.column_stack((previous, target)).ravel()
    t = np.r_[0., vertices[(vertices > 0) & (vertices < periods)], float(periods)]
    return signal(t, np.interp(t, vertices, values))


@pytest.mark.parametrize('origin', [0., .037, .113, .249, .381, .507, .619, .777, .923])
def test_equivalent_origins_keep_the_absolute_sampling_lattice(origin):
    # Exactly ten repetitions of the SAME two-UI population at each origin.
    # Include the actual PL endpoint values rather than extrapolating a crop.
    full = trapezoidal_nrz(periods=24)
    start, stop = 2 + origin, 22 + origin
    t = np.r_[start, full.x[(full.x > start) & (full.x < stop)], stop]
    sig = signal(t, np.interp(t, full.x, full.y))
    result = eye_diagram(sig, period=1., auto_center=True)
    assert circular_distance(start + result.sampling_instant_s, .65) <= 2e-12
    expected_relative = (.65 - origin) % 1.
    assert circular_distance(result.sampling_instant_s, expected_relative) <= 2e-12


@pytest.mark.parametrize('ui, origin', [(1e-9, 0.), (1e-6, .01), (1., 1000.)])
def test_time_units_and_nonzero_origin(ui, origin):
    sig = trapezoidal_nrz(ui=ui, origin=origin)
    result = eye_diagram(sig, period=ui, auto_center=True)
    assert result.sampling_instant_s / ui == pytest.approx(.65, rel=0, abs=2e-12)


@pytest.mark.parametrize('shift', [.35, .7])
def test_plateau_wraps_across_zero(shift):
    sig = trapezoidal_nrz(phase=shift)
    result = eye_diagram(sig, period=1., auto_center=True)
    assert circular_distance(result.sampling_instant_s, (.65 + shift) % 1.) < 2e-12


@pytest.mark.parametrize('axis', [[0., 1., np.nan, 3.],
                                 [0., 1., np.inf, 3.],
                                 [0., 1., 1., 3.], [0., 2., 1., 3.]])
def test_malformed_time_axis_is_an_actionable_caller_error(axis):
    with pytest.raises(ValueError, match='eye_diagram:.*x-axis.*needs.*increasing'):
        eye_diagram(signal(axis, [0., 1., 0., 1.]), period=1., auto_center=True)


def test_ordinary_signal_plot_uses_the_shifted_coordinates():
    from matplotlib import pyplot as plt

    sig = trapezoidal_nrz()
    result = eye_diagram(sig, period=1., auto_center=True, n_traces=3)
    fig, ax = plt.subplots()
    try:
        for trace in result.traces:
            trace.plot(ax=ax)
        assert len(ax.lines) == 3
        for line, trace in zip(ax.lines, result.traces):
            np.testing.assert_array_equal(line.get_xdata(), trace.x)
            np.testing.assert_array_equal(line.get_ydata(), trace.y)
        assert np.any(np.asarray(result.traces[0].x) < .1)
    finally:
        plt.close(fig)


def test_float32_values_and_signed_zero_are_preserved_in_shifted_windows():
    sig = trapezoidal_nrz()
    y = sig.y.astype(np.float32)
    # Subdivide a crossing at zero, preserving its continuous waveform.
    t = np.sort(np.r_[sig.x, .15, 2.15])
    y = np.interp(t, sig.x, y).astype(np.float32)
    y[np.argmin(abs(t - .15))] = np.float32(-0.)
    y[np.argmin(abs(t - 2.15))] = np.float32(-0.)
    sig = signal(t, y)
    result = eye_diagram(sig, period=1., auto_center=True)
    assert math.isfinite(result.sampling_instant_s)
    shift = (result.sampling_instant_s - .5) % 1.
    for k, trace in enumerate(result.traces):
        take = (t >= shift + k) & (t < shift + k + 1)
        assert trace.y.dtype == np.dtype('float32')
        assert trace.y.tobytes() == y[take].tobytes()
    assert any(np.any((trace.y == 0) & np.signbit(trace.y))
               for trace in result.traces)


@pytest.mark.parametrize('offset, gain', [(1., 1e-12), (1., 1e-14), (1000., 1e-8)])
def test_dc_offset_does_not_erase_resolvable_small_voltage_changes(offset, gain):
    sig = trapezoidal_nrz()
    moved = signal(sig.x, offset + gain * sig.y)
    assert np.ptp(moved.y) > 0
    with np.errstate(divide='raise', invalid='raise'):
        result = eye_diagram(moved, period=1., auto_center=True)
    assert result.sampling_instant_s == pytest.approx(.65, rel=0, abs=2e-12)


def balanced_record(t, y):
    """Append a voltage-inverted copy so the PL time average is exactly zero."""
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    assert y[-1] == -y[0]
    return signal(np.r_[t, t[-1] + t[1:]], np.r_[y, -y[1:]])


def test_threshold_is_time_average_not_vertex_mean_or_range_midpoint():
    # Area = 1 + 6 = 7, duration = 4, average = 7/4. Crossing at 7/8.
    sig = signal([0., 1., 4.], [0., 2., 2.])
    result = eye_diagram(sig, period=1., auto_center=True)
    assert result.sampling_instant_s == pytest.approx(.375, rel=0, abs=1e-14)


def test_one_crossing_can_establish_a_phase():
    with pytest.warns(MetricWarning) as caught:
        result = eye_diagram(signal([0., 4.5], [-1., 1.]), period=1., auto_center=True)
    assert len(caught) == 1 and caught[0].message.code == 'centering_sensitive'
    assert result.sampling_instant_s == pytest.approx(.75, rel=0, abs=1e-14)
    # Interpolation is for analysis only: do not insert the event or endpoints.
    assert all(len(trace.x) == 0 for trace in result.traces)


def test_normal_evenly_spaced_pam4_has_analytic_crossing_phase():
    # Mean = .9 V; rising .6 -> 1.2 and falling 1.8 -> 0 both cross at .15 UI.
    result = eye_diagram(held_pam([0., .6, 1.2, 1.8]), period=1., auto_center=True)
    assert result.sampling_instant_s == pytest.approx(.65, rel=0, abs=2e-14)


def test_circular_mean_keeps_wraparound_crossings_adjacent():
    # Average zero; crossings straddle phase zero.
    sig = balanced_record([0., .9, 1., 2., 2.1, 3.9, 4.],
                          [-1., -1., 1., 1., -1., -1., 1.])
    # Events .95, 2.05, 3.95 and their inverted copies. Equal weight means
    # angle atan2(-sin(.1*pi), 3*cos(.1*pi)), then half a UI.
    expected = .5 - math.atan(math.tan(.1 * math.pi) / 3) / (2 * math.pi)
    result = eye_diagram(sig, period=1., auto_center=True)
    assert result.sampling_instant_s == pytest.approx(expected, rel=0, abs=1e-14)
    assert .48 < result.sampling_instant_s < .5


@pytest.mark.parametrize('t,y,expected,sensitive', [
    # Exact vertex at .25 counts once alongside crossings at 1.5 and 2.75.
    ([0., .125, .25, .375, 1.25, 1.75, 2.5, 3., 4.],
     [-1., -1., 0., 1., 1., -1., -1., 1., 1.], 0., False),
    # A threshold plateau contributes its midpoint (.25), not its edges.
    ([0., .125, .375, .5, 1.25, 1.75, 2.5, 3., 4.],
     [-1., 0., 0., 1., 1., -1., -1., 1., 1.], 0., False),
    # A same-side vertex touch at 3.25 contributes nothing.
    ([0., .125, .25, .375, 1.25, 1.75, 2.5, 3., 3.25, 3.5, 4.],
     [-1., -1., 0., 1., 1., -1., -1., 1., 0., 1., 1.], 0., True),
    # A same-side threshold plateau also contributes nothing.
    ([0., .125, .25, .375, 1.25, 1.75, 2.5, 3., 3.125, 3.375, 3.5, 4.],
     [-1., -1., 0., 1., 1., -1., -1., 1., 0., 0., 1., 1.], 0., False),
    # Threshold endpoints have no two-sided context; the join at 4 is a touch.
    ([0., .125, .25, .5, 4.], [0., -1., 0., 1., 0.], .75, False),
    # All three ringing events in one UI count, rising AND falling.
    ([0., .1, .2, .3, .4, 4.], [-1., -1., 1., -1., 1., 1.], .75, False),
], ids=['vertex', 'plateau', 'vertex_touch', 'plateau_touch', 'endpoints', 'ringing'])
def test_every_genuine_event_counts_once(t, y, expected, sensitive):
    sig = balanced_record(t, y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = eye_diagram(sig, period=1., auto_center=True)
    assert [w.message.code for w in caught] == (['centering_sensitive'] if sensitive else [])
    assert circular_distance(result.sampling_instant_s, expected) < 2e-14


@pytest.mark.parametrize('n_traces', [None, 2])
@pytest.mark.parametrize('kind', ['constant', 'nan', 'inf', 'cancel'])
def test_unavailable_centering_warns_once_and_preserves_original_fold(kind, n_traces):
    sig = signal(np.arange(9.) / 2, np.where(np.arange(9) % 2, 1., -1.))
    if kind == 'constant':
        sig.y[:] = 1.
    elif kind != 'cancel':
        sig.y[-1] = float(kind)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = eye_diagram(sig, period=1., n_traces=n_traces, auto_center=True)
    assert len(caught) == 1
    assert isinstance(caught[0].message, MetricWarning)
    assert caught[0].message.code == 'centering_unavailable'
    assert math.isnan(result.sampling_instant_s)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        original = eye_diagram(sig, period=1., n_traces=n_traces)
    assert_same_fold(result, original)


def test_weakly_concentrated_crossings_are_not_a_quality_refusal():
    # Six events spaced 1 s apart nearly cancel for a ~2 s supplied UI.
    # |z| ~ pi*2e-8 is small but well above the numerical-zero guard.
    delta = 1e-8
    period = 2. - 4 * delta
    sig = signal(np.arange(7.), np.where(np.arange(7) % 2, 1., -1.))
    with pytest.warns(MetricWarning) as caught:
        result = eye_diagram(sig, period=period, auto_center=True)
    assert len(caught) == 1 and caught[0].message.code == 'centering_sensitive'
    # Geometric sum of the six phasors gives centre = 3 - period.
    # Cancellation amplifies floating-point angle error by ~1/|z|.
    assert result.sampling_instant_s == pytest.approx(3 - period, rel=0, abs=1e-8)


def test_full_record_phase_is_independent_of_output_limit():
    # First pair crosses at .15; later pairs at .4. Analysing just the returned
    # first two windows would select .65 instead of the whole-record result.
    t = [0., .1, .2, 1.1, 1.2, 2., 2.3, 2.5, 3.3, 3.5, 4.]
    sig = signal(t, [-1., -1., 1., 1., -1., -1., -1., 1., 1., -1., -1.])
    full = eye_diagram(sig, period=1., auto_center=True)
    limited = eye_diagram(sig, period=1., n_traces=1, auto_center=True)
    assert full.sampling_instant_s == pytest.approx(.775, rel=0, abs=1e-14)
    assert limited.sampling_instant_s == full.sampling_instant_s


def test_nonuniform_subdivision_does_not_reweight_time_or_add_samples():
    original = signal([0., 1., 4.], [0., 2., 2.])
    t = np.unique(np.r_[original.x, np.linspace(.01, .7, 101), 2.3, 3.9])
    refined = signal(t, np.interp(t, original.x, original.y))
    a = eye_diagram(original, period=1., auto_center=True)
    b = eye_diagram(refined, period=1., auto_center=True)
    assert a.sampling_instant_s == pytest.approx(.375, rel=0, abs=1e-14)
    assert b.sampling_instant_s == pytest.approx(a.sampling_instant_s, rel=0, abs=1e-14)
    shift = (b.sampling_instant_s - .5) % 1.
    for k, trace in enumerate(b.traces):
        take = (t >= shift + k) & (t < shift + k + 1)
        np.testing.assert_array_equal(trace.x, t[take] - (shift + k))
        assert trace.y.tobytes() == refined.y[take].tobytes()


@pytest.mark.parametrize('gain,offset', [(2.5, 7.), (-3., 1.), (1e-6, 0.), (1e6, -1e8)])
def test_voltage_gain_offset_and_polarity_preserve_alignment(gain, offset):
    sig = signal([0., 1., 4.], gain * np.array([0., 2., 2.]) + offset)
    result = eye_diagram(sig, period=1., auto_center=True)
    assert result.sampling_instant_s == pytest.approx(.375, rel=0, abs=2e-14)


@pytest.mark.parametrize('counts,sensitive', [
    (None, False), ([150, 150, 150, 150], False), ([196, 12, 196, 196], True),
])
def test_difficult_pam_is_crossing_characterization_not_opening_optimization(counts, sensitive, record_property):
    sig = ringing([0., 1.75, 1., 2.], counts=counts)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = eye_diagram(sig, period=1., auto_center=True, n_traces=2)
    assert [w.message.code for w in caught] == (['centering_sensitive'] if sensitive else [])
    assert math.isfinite(result.sampling_instant_s)
    record_property('crossing_center_ui', result.sampling_instant_s)


@pytest.mark.parametrize('delta,available', [(1e-14, False), (1e-12, True)])
def test_numerical_zero_guard_is_not_a_concentration_cutoff(delta, available):
    # The resultant magnitude is ~2*pi*delta, bracketing the 1e-12 guard.
    sig = signal(np.arange(7.), np.where(np.arange(7) % 2, 1., -1.))
    period = 2. - 4 * delta
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = eye_diagram(sig, period=period, auto_center=True)
    assert math.isfinite(result.sampling_instant_s) == available
    assert len(caught) == 1
    assert caught[0].message.code == ('centering_sensitive' if available else 'centering_unavailable')
    if not available:
        assert caught[0].message.code == 'centering_unavailable'
        assert_same_fold(result, eye_diagram(sig, period=period))


def test_centered_boundaries_are_half_open_without_synthesizing_endpoints():
    # A periodic triangle with dyadic vertices has crossings at .75 UI,
    # giving centre .25 and shift .75, exactly on original sample vertices.
    t = np.arange(0., 8.25, .25)
    sig = signal(t, 1 - 4 * abs(((t - .25) / 2) % 1 - .5))
    result = eye_diagram(sig, period=1., auto_center=True)
    assert result.sampling_instant_s == pytest.approx(.25, rel=0, abs=1e-14)
    shift = (result.sampling_instant_s - .5) % 1
    assert shift == .75
    for k, trace in enumerate(result.traces):
        take = (t >= shift + k) & (t < shift + k + 1)
        np.testing.assert_array_equal(trace.x, t[take] - (shift + k))
        assert trace.y.tobytes() == sig.y[take].tobytes()
        assert len(trace.x) == 4
        assert trace.x[0] == 0
        assert trace.x[-1] < 1


@pytest.mark.parametrize('t,y,period,match', [
    ([0., 1., 2.], [0., 1.], 1., 'matching 1-D'),
    ([0., 1., 2.], [0., 1j, 0.], 1., 'real voltage'),
    ([0., 1., 2.], [0., 1., 0.], float('inf'), 'finite'),
])
def test_centering_caller_errors_remain_errors(t, y, period, match):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match=match):
            eye_diagram(signal(t, y), period=period, auto_center=True)


def pam3_at_mean(*, noise=0., seed=0):
    """A 1 V mean on a PAM3 level; true symbol centres lie at half a UI."""
    levels = np.array([0., 1., 2., 1.])
    cycles = np.arange(-2, 163)
    knots = (cycles[:, None] + [-.125, .125]).ravel()
    values = np.column_stack((levels[(cycles - 1) % 4],
                              levels[cycles % 4])).ravel()
    t = np.arange(160 * 128 + 1) / 128
    y = np.interp(t, knots, values)
    y += np.random.default_rng(seed).normal(0., 2 * noise, len(y))
    return signal(t, y)


@pytest.mark.parametrize('n_traces', [None, 2])
def test_sensitive_center_warns_without_changing_phase_or_samples(n_traces):
    sig = pam3_at_mean()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = eye_diagram(sig, period=1., auto_center=True, n_traces=n_traces)
    assert len(caught) == 1
    d = caught[0].message
    assert isinstance(d, MetricWarning)
    assert d.metric == 'eye_diagram' and d.code == 'centering_sensitive'
    assert 'threshold' in d.message and 'Inspect' in d.suggestion
    assert d.details['signal_name'] == sig.name
    assert d.details['period_s'] == 1.
    assert d.details['sample_count'] == len(sig.y)
    assert d.details['threshold_offset'] == pytest.approx(.1)
    assert d.details['magnitude_unit'] == 'V'
    assert d.details['max_phase_shift_ui'] == pytest.approx(.5)
    assert d.details['undefined_probe_count'] == 0
    assert all(type(v) in (str, float, int, type(None)) for v in d.details.values())
    # Both probes select .5 UI while nominal stays at 0: comparing only the
    # probes would incorrectly call this stable.
    assert d.details['lower_center_s'] == pytest.approx(.5)
    assert d.details['upper_center_s'] == pytest.approx(.5)
    assert result.sampling_instant_s == 0.
    assert len(result.traces) == (n_traces or 159)
    for k, trace in enumerate(result.traces):
        start = .5 + k
        take = (sig.x >= start) & (sig.x < start + 1.)
        np.testing.assert_array_equal(trace.x, sig.x[take] - start)
        assert trace.y.tobytes() == sig.y[take].tobytes()
    for name in ('height', 'width'):
        assert math.isnan(getattr(result, name))


@pytest.mark.parametrize('noise', [.001, .01])
@pytest.mark.parametrize('seed', [0, 7, 29])
def test_noisy_threshold_sensitivity_does_not_require_exact_plateaus(noise, seed):
    with pytest.warns(MetricWarning) as caught:
        result = eye_diagram(pam3_at_mean(noise=noise, seed=seed),
                             period=1., n_traces=2, auto_center=True)
    assert len(caught) == 1
    assert caught[0].message.code == 'centering_sensitive'
    assert caught[0].message.details['max_phase_shift_ui'] > .45
    assert math.isfinite(result.sampling_instant_s)


@pytest.mark.parametrize('ui,origin,gain,offset', [
    (1., 0., 1., 0.), (.125, 32., 2.5, 7.), (8., 1024., -2., 3.),
])
def test_sensitivity_is_relative_to_ui_and_voltage_range(ui, origin, gain, offset):
    sig = pam3_at_mean()
    transformed = signal(origin + sig.x * ui, offset + gain * sig.y)
    with pytest.warns(MetricWarning) as caught:
        result = eye_diagram(transformed, period=ui, n_traces=2, auto_center=True)
    assert len(caught) == 1 and caught[0].message.code == 'centering_sensitive'
    assert caught[0].message.details['max_phase_shift_ui'] == pytest.approx(.5)
    assert result.sampling_instant_s == 0.


def test_undefined_probe_warns_but_does_not_refuse_the_nominal_center():
    # Mean .984375 V. The upper probe is above every sample: no crossings.
    sig = signal([0., .25, 8.], [0., 1., 1.])
    with pytest.warns(MetricWarning) as caught:
        result = eye_diagram(sig, period=1., auto_center=True)
    assert len(caught) == 1
    d = caught[0].message
    assert d.code == 'centering_sensitive'
    assert d.details['undefined_probe_count'] == 1
    assert d.details['upper_center_s'] is None
    assert result.sampling_instant_s == pytest.approx(.74609375)


def test_small_phase_changes_across_zero_do_not_warn_about_sensitivity():
    # Mean 1.75 V, crossing .875 s, UI 1.75 s: nominal center is exactly 0.
    # Probes select .05 s and 1.70 s: nearby circularly, NOT 1.65 s apart.
    # The shifted fold fits just one window: geometry is unavailable, but
    # the nearby circular phases must not be flagged as sensitive centering.
    with pytest.warns(MetricWarning, match='insufficient_eye_samples') as caught:
        result = eye_diagram(signal([0., 1., 4.], [0., 2., 2.]),
                             period=1.75, auto_center=True, threshold=1.)
    assert len(caught) == 1
    assert result.sampling_instant_s == 0.


@pytest.mark.parametrize('levels', [(0., 1.8), (0., .6, 1.2, 1.8)])
def test_approved_rc_examples_do_not_gain_sensitivity_warnings(levels):
    from silk.examples.eye_centering import UI, rc_waveform

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = eye_diagram(rc_waveform(levels), period=UI, auto_center=True)
    assert math.isfinite(result.sampling_instant_s)


def test_sensitive_warning_filters_preserve_results_or_raise():
    sig = pam3_at_mean()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore', MetricWarning)
        result = eye_diagram(sig, period=1., auto_center=True)
    assert caught == [] and result.sampling_instant_s == 0.
    assert len(result.traces) == 159
    with warnings.catch_warnings():
        warnings.simplefilter('error', MetricWarning)
        with pytest.raises(MetricWarning, match='centering_sensitive'):
            eye_diagram(sig, period=1., auto_center=True)


def test_disabled_centering_does_not_probe_or_warn_on_sensitive_waveform():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = eye_diagram(pam3_at_mean(), period=1.)
    assert math.isnan(result.sampling_instant_s)
    assert len(result.traces) == 160


def test_shifted_capacity_error_precedes_sensitivity_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='centering shifts'):
            eye_diagram(pam3_at_mean(), period=1., n_traces=160, auto_center=True)


@pytest.mark.parametrize('unit,gain', [('V', 1.), ('mV', 1000.)])
def test_sensitivity_warning_uses_the_input_amplitude_units(unit, gain):
    original = pam3_at_mean()
    sig = Signal('rx', 'voltage', 'time', unit, original.x, gain * original.y)
    with pytest.warns(MetricWarning) as caught:
        eye_diagram(sig, period=1., auto_center=True)
    d = caught[0].message.details
    assert d['magnitude_unit'] == unit
    assert d['threshold_offset'] == pytest.approx(.1 * gain)


@pytest.mark.parametrize('ramp_duration,sensitive', [(1.75, False), (2.25, True)])
def test_only_substantial_threshold_motion_warns(ramp_duration, sensitive):
    # A single linear ramp shifts by .05 * ramp_duration seconds per probe:
    # .0875 UI stays below the advisory limit; .1125 UI exceeds it.
    sig = signal([0., ramp_duration, 8.], [0., 1., 1.])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = eye_diagram(sig, period=1., auto_center=True)
    assert math.isfinite(result.sampling_instant_s)
    expected = ['centering_sensitive'] if sensitive else []
    assert [w.message.code for w in caught] == expected
