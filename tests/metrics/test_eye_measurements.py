"""Geometric measurements at the existing eye centre, not symbol recovery."""

import math
import warnings

import numpy as np
import pytest

from silk.metrics import eye_diagram, eye_height, eye_width
from silk.metrics.diagnostics import MetricWarning
from silk.metrics.eye import EyeResult
from tests.metrics.test_eye_centering import (
    assert_same_fold, held_pam, pam3_at_mean, signal, trapezoidal_nrz,
)


@pytest.fixture(scope='module')
def review_cases():
    """Reproduce the owner's review; symbols are an oracle, never metric input."""
    ui, samples_per_ui = 1e-9, 50

    def waveform(symbols, levels):
        t = np.arange(len(symbols) * samples_per_ui) / samples_per_ui * ui
        y = np.repeat(np.asarray(levels)[symbols], samples_per_ui)
        y = np.convolve(np.r_[np.full(5, y[0]), y], np.ones(5) / 5, 'same')[5:]
        return t, y

    def rc(y, tau_ui):
        decay = np.exp(-1 / (samples_per_ui * tau_ui))
        out = np.empty_like(y)
        state = y[0]
        for i, value in enumerate(y):
            state = decay * state + (1 - decay) * value
            out[i] = state
        return out

    bits = np.random.default_rng(1).integers(0, 2, 200)
    pam = np.random.default_rng(5).integers(0, 4, 400)
    biased = (np.random.default_rng(9).random(200) < .8).astype(int)
    t, y = waveform(bits, [0., 1.])
    tp, yp = waveform(pam, [0., 1/3, 2/3, 1.])
    tu, yu = waveform(pam, [0., .2, .6, 1.])
    tb, yb = waveform(biased, [0., 1.])
    noise = np.random.default_rng(7)
    return [
        ('clean', t, y, bits, [.5]),
        ('rc_05', t, rc(y, .5), bits, [.5]),
        ('rc_06', t, rc(y, .6), bits, [.5]),
        ('rc_10', t, rc(y, 1.), bits, [.5]),
        ('noise_015', t, y + .15 * noise.standard_normal(len(y)), bits, [.5]),
        ('noise_020', t, y + .20 * noise.standard_normal(len(y)), bits, [.5]),
        ('closed', t, y + .25 * noise.standard_normal(len(y)), bits, [.5]),
        ('biased', tb, rc(yb, .6), biased, [.5]),
        ('pam4', tp, yp, pam, [1/6, .5, 5/6]),
        ('uneven', tu, yu, pam, [.1, .4, .8]),
        ('pam4_noise', tp, yp + .03 * noise.standard_normal(len(yp)), pam, [1/6, .5, 5/6]),
    ]


@pytest.mark.parametrize('case_index', [0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
def test_threshold_height_matches_sent_symbol_openings(review_cases, case_index):
    name, t, y, symbols, thresholds = review_cases[case_index]
    sig = signal(t, y)
    ui = 1e-9
    fold = eye_diagram(sig, period=ui, auto_center=True)
    origin = t[0] + (fold.sampling_instant_s - ui / 2) % ui
    centres = origin + (np.arange(len(fold.traces)) + .5) * ui
    observations = np.interp(centres, t, y)
    sent = symbols[np.floor((centres - t[0]) / ui).astype(int)]
    for i, threshold in enumerate(thresholds):
        expected = observations[sent == i+1].min() - observations[sent == i].max()
        assert expected > 0, name
        result = eye_diagram(sig, period=ui, auto_center=True, threshold=threshold)
        assert result.height == pytest.approx(expected, abs=1e-12), name
        assert result.heights == pytest.approx((expected,), abs=1e-12)
        assert result.levels == (threshold,)
        assert eye_height(sig, period=ui, threshold=threshold) == pytest.approx(expected, abs=1e-12)
        assert result.sampling_instant_s == fold.sampling_instant_s
        assert_same_fold(result, fold)


def test_closed_eye_reports_local_spacing_not_population_overlap(review_cases):
    _, t, y, symbols, _ = review_cases[6]
    sig = signal(t, y)
    result = eye_diagram(sig, period=1e-9, auto_center=True, threshold=.5)
    centres = result.sampling_instant_s + np.arange(len(symbols)) * 1e-9
    centres = centres[centres <= t[-1]]
    samples = np.interp(centres, t, y)
    sent = symbols[np.floor(centres / 1e-9).astype(int)]
    assert samples[sent == 1].min() - samples[sent == 0].max() == pytest.approx(-.0985914553)
    assert result.height == pytest.approx(.027, abs=.001)
    assert 0 < result.height < .05  # > 0 alone would incorrectly pass a spec.


def test_without_measurement_threshold_only_fold_and_center_are_populated():
    result = eye_diagram(trapezoidal_nrz(), period=1., auto_center=True)
    assert result.sampling_instant_s == pytest.approx(.65)
    assert result.traces
    assert math.isnan(result.height) and math.isnan(result.width)
    assert result.levels == result.heights == result.widths == ()


@pytest.mark.parametrize('measure', [eye_height, eye_width])
def test_scalar_measurements_require_an_explicit_threshold(measure):
    with pytest.raises(TypeError, match='threshold'):
        measure(trapezoidal_nrz(), period=1.)


def test_width_uses_supplied_threshold_not_gap_midpoint():
    sig = held_pam([0., 1.])
    result = eye_diagram(sig, period=1., auto_center=True, threshold=.25)
    assert result.height == pytest.approx(1.)
    assert result.levels == (.25,)
    assert result.width == pytest.approx(.95)
    assert eye_width(sig, period=1., threshold=.25) == pytest.approx(.95)


def test_threshold_on_a_trace_is_zero_clearance():
    result = eye_diagram(held_pam([0., 1., 2., 3.]), period=1.,
                         auto_center=True, threshold=1.)
    assert result.height == result.width == 0.
    assert result.heights == result.widths == (0.,)


@pytest.mark.parametrize('threshold', [-.1, 1.1])
def test_threshold_without_both_boundaries_warns_and_returns_nan(threshold):
    with pytest.warns(MetricWarning, match='eye_threshold_unbracketed') as caught:
        result = eye_diagram(held_pam([0., 1.]), period=1.,
                             auto_center=True, threshold=threshold)
    assert len(caught) == 1
    assert math.isnan(result.height) and math.isnan(result.width)
    assert result.levels == (threshold,)
    assert len(result.heights) == len(result.widths) == 1
    assert caught[0].message.details['threshold'] == threshold


@pytest.mark.parametrize('threshold', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_threshold_is_a_caller_error(threshold):
    with pytest.raises(ValueError, match='threshold.*finite'):
        eye_diagram(trapezoidal_nrz(), period=1., auto_center=True, threshold=threshold)


@pytest.mark.parametrize('threshold', [[0.], np.array([0.]), 1j, 'not a number', None])
@pytest.mark.parametrize('measure', [eye_height, eye_width])
def test_scalar_threshold_validation_names_the_public_metric(threshold, measure):
    with pytest.raises(ValueError, match=f'{measure.__name__}: threshold.*finite real scalar'):
        measure(trapezoidal_nrz(), period=1., threshold=threshold)


def test_sparse_nrz_has_one_full_width_opening_without_adding_samples():
    sig = signal([0., .1, .2, 4.1, 4.2, 8.], [1., 1., -1., -1., 1., 1.])
    result = eye_diagram(sig, period=1., auto_center=True, threshold=0.)
    assert result.levels == pytest.approx((0.,))
    assert result.heights == pytest.approx((2.,))
    assert result.widths == pytest.approx((1.,))
    assert result.height == pytest.approx(2.)
    assert result.width == pytest.approx(1.)
    # Several windows contain no vertices at all, but the original continuous
    # PL waveform still defines their centre values without extrapolation.
    assert any(len(t.x) == 0 for t in result.traces)
    origin = (result.sampling_instant_s - .5) % 1.
    for k, trace in enumerate(result.traces):
        start = origin + k
        take = (sig.x >= start) & (sig.x < start + 1.)
        np.testing.assert_array_equal(trace.x, sig.x[take] - start)
        assert trace.y.tobytes() == sig.y[take].tobytes()


def test_pam4_caller_selects_three_eyes_with_independent_analytic_widths():
    results = [eye_diagram(held_pam([0., 1., 2., 3.]), period=1.,
                           auto_center=True, threshold=threshold)
               for threshold in (.5, 1.5, 2.5)]
    assert [r.sampling_instant_s for r in results] == pytest.approx([.65] * 3)
    assert [r.levels[0] for r in results] == pytest.approx((.5, 1.5, 2.5))
    assert [r.height for r in results] == pytest.approx((1., 1., 1.))
    # The rising ramp crosses each midpoint at .15 UI. The 3 -> 0 ramp
    # crosses the outer midpoints at .1 + 5/60 and .1 + 1/60 UI.
    assert [r.width for r in results] == pytest.approx((29 / 30, 1., 29 / 30))
    assert min(r.height for r in results) == pytest.approx(1.)
    assert min(r.width for r in results) == pytest.approx(29 / 30)


@pytest.mark.parametrize('plateau, bottom_width', [(False, .75), (True, .70)])
def test_width_includes_same_side_touches_and_plateau_ends(plateau, bottom_width):
    sig = held_pam([0., 1., 2., 3.])
    times, values = [], []
    for k in range(40):
        if k % 4 not in (0, 3):
            continue
        base, contact = (0., .5) if k % 4 == 0 else (3., 2.5)
        times.extend(k + np.array([.35, .4, .45, .50]))
        values.extend([base, contact, contact if plateau else base, base])
    t = np.sort(np.r_[sig.x, times])
    y = np.interp(t, sig.x, sig.y)
    y[np.searchsorted(t, times)] = values
    result = eye_diagram(signal(t, y), period=1., auto_center=True, threshold=.5)
    assert result.sampling_instant_s == pytest.approx(.65)
    assert result.heights == pytest.approx((1.,))
    assert result.widths[0] == pytest.approx(bottom_width)
    assert eye_width(signal(t, y), period=1., threshold=1.5) == pytest.approx(1.)


@pytest.mark.parametrize('ui, origin, gain, offset', [
    (1e-9, 0., .6, .7), (1e-6, .01, -2., 10.), (1., 1000., 1., 0.),
])
def test_measurement_units_offsets_polarity_and_order(ui, origin, gain, offset):
    sig = held_pam([0., 1., 2., 3.])
    scaled = signal(origin + sig.x * ui, sig.y * gain + offset)
    thresholds = sorted(offset + gain * np.array([.5, 1.5, 2.5]))
    results = [eye_diagram(scaled, period=ui, auto_center=True, threshold=v)
               for v in thresholds]
    assert [r.levels[0] for r in results] == pytest.approx(thresholds)
    assert [r.height for r in results] == pytest.approx((abs(gain),) * 3)
    np.testing.assert_allclose(np.array([r.width for r in results]) / ui,
                               [29 / 30, 1., 29 / 30], atol=1e-9)


def test_measurement_population_obeys_n_traces_but_center_uses_full_input():
    sig = held_pam([0., 1., 2., 3.])
    full = eye_diagram(sig, period=1., auto_center=True, threshold=.5)
    limited = eye_diagram(sig, period=1., auto_center=True, n_traces=2, threshold=.5)
    assert limited.sampling_instant_s == full.sampling_instant_s
    assert len(limited.traces) == 2
    assert limited.levels == pytest.approx((.5,))
    assert limited.heights == pytest.approx((1.,))
    assert limited.widths == pytest.approx((29 / 30,))


@pytest.mark.parametrize('levels', [(0., 1.8), (0., .6, 1.2, 1.8)])
@pytest.mark.parametrize('seed', range(20))
def test_explicit_thresholds_on_rc_examples_with_moderate_noise(levels, seed):
    from silk.examples.eye_centering import UI, rc_waveform
    sig = rc_waveform(levels)
    y = sig.y + np.random.default_rng(seed).normal(0., .02 * np.ptp(sig.y), len(sig.y))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for low, high in zip(levels[:-1], levels[1:]):
            result = eye_diagram(signal(sig.x, y), period=UI, auto_center=True,
                                 threshold=(low + high) / 2)
            assert len(result.heights) == len(result.widths) == len(result.levels) == 1
            assert 0. < result.width <= UI
            assert result.height > 0.


def test_subdivision_of_linear_segments_does_not_change_measurements():
    sig = held_pam([0., 1., 2., 3.])
    # Deliberately nonuniform subdivision of every segment, including seams.
    t = np.sort(np.r_[sig.x, sig.x[:-1] + .137 * np.diff(sig.x),
                      sig.x[:-1] + .893 * np.diff(sig.x)])
    dense = signal(t, np.interp(t, sig.x, sig.y))
    before = eye_diagram(sig, period=1., auto_center=True, threshold=1.5)
    after = eye_diagram(dense, period=1., auto_center=True, threshold=1.5)
    assert after.sampling_instant_s == pytest.approx(before.sampling_instant_s)
    assert after.levels == pytest.approx(before.levels)
    assert after.heights == pytest.approx(before.heights)
    assert after.widths == pytest.approx(before.widths)


@pytest.mark.parametrize('policy', ['always', 'ignore'])
def test_sensitive_center_withholds_measurements_independently_of_warning_policy(policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        result = eye_diagram(pam3_at_mean(), period=1., auto_center=True, threshold=.5)
    assert math.isfinite(result.sampling_instant_s)
    assert len(result.traces) > 0
    assert math.isnan(result.height) and math.isnan(result.width)
    assert result.levels == result.heights == result.widths == ()
    assert len(caught) == (1 if policy == 'always' else 0)
    if caught:
        assert caught[0].message.code == 'centering_sensitive'


def test_unavailable_and_disabled_centers_keep_empty_measurements():
    with pytest.warns(MetricWarning, match='centering_unavailable') as caught:
        unavailable = eye_diagram(signal([0., 10.], [1., 1.]), period=1.,
                                  auto_center=True, threshold=0.)
    assert len(caught) == 1
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        disabled = eye_diagram(trapezoidal_nrz(), period=1., threshold=0.)
    for result in (unavailable, disabled):
        assert math.isnan(result.height) and math.isnan(result.width)
        assert result.levels == result.heights == result.widths == ()


@pytest.mark.parametrize('policy', ['always', 'ignore', 'error'])
def test_one_center_observation_warns_without_discarding_fold_or_phase(policy):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(policy, MetricWarning)
        if policy == 'error':
            with pytest.raises(MetricWarning, match='eye_diagram.*insufficient_eye_samples'):
                eye_diagram(trapezoidal_nrz(), period=1., n_traces=1,
                            auto_center=True, threshold=0.)
            return
        result = eye_diagram(trapezoidal_nrz(), period=1., n_traces=1,
                             auto_center=True, threshold=0.)
    assert result.sampling_instant_s == pytest.approx(.65)
    assert len(result.traces) == 1
    assert math.isnan(result.height) and math.isnan(result.width)
    assert result.levels == result.heights == result.widths == ()
    assert len(caught) == (1 if policy == 'always' else 0)
    if caught:
        assert caught[0].message.code == 'insufficient_eye_samples'
        assert caught[0].message.details['center_sample_count'] == 1
        assert caught[0].message.suggestion


def test_explicit_threshold_keeps_small_gap_regardless_of_larger_gaps():
    from silk.metrics._eye import _measure_eye
    # Centre observations at .5, 1.5, ...; gaps .125, .25, .5, 1.
    x = np.arange(6.)
    y = np.array([0., 0., .25, .5, 1.25, 2.5])
    levels, heights, widths = _measure_eye(x, y, 1., 0., 5, .0625)
    assert heights == pytest.approx((.125,))
    assert levels == (.0625,)
    assert len(widths) == 1


def test_observed_zero_gap_is_not_a_decoded_closed_eye_claim():
    from silk.metrics._eye import _measure_eye
    assert _measure_eye(np.arange(4.), np.ones(4), 1., 0., 3, 1.) == ((1.,), (0.,), (0.,))
    # Global transitions still define the centre; restricting the returned
    # population to two identical plateau centres leaves zero observed gap.
    sig = held_pam([0., 0., 3., 3.])
    result = eye_diagram(sig, period=1., n_traces=2, auto_center=True, threshold=0.)
    assert result.sampling_instant_s == pytest.approx(.65)
    assert result.height == result.width == 0.
    assert result.levels == result.heights == result.widths == (0.,)


def test_result_positional_construction():
    old = EyeResult(1., 2., [], 6.)
    assert (old.height, old.width, old.sampling_instant_s) == (1., 2., 6.)
    assert old.levels == old.heights == old.widths == ()


def test_longdouble_input_measures_on_float_copy_but_preserves_original_samples():
    base = trapezoidal_nrz()
    sig = signal(base.x.astype(np.longdouble), base.y.astype(np.longdouble))
    original = (sig.x.tobytes(), sig.y.tobytes())
    result = eye_diagram(sig, period=1., auto_center=True, threshold=0.)
    assert result.height == pytest.approx(2.)
    assert result.width == pytest.approx(1.)
    assert (sig.x.tobytes(), sig.y.tobytes()) == original
    origin = float(sig.x[0]) + (result.sampling_instant_s - .5) % 1.
    for k, trace in enumerate(result.traces):
        start = origin + k
        take = (sig.x >= start) & (sig.x < start + 1.)
        assert trace.y.dtype == sig.y.dtype
        assert trace.y.tobytes() == sig.y[take].tobytes()
        np.testing.assert_array_equal(trace.x, sig.x[take] - start)


@pytest.mark.parametrize('low, step', [(1., np.spacing(1.)), (1e16, 2.)])
def test_small_opening_on_dc_offset_with_representable_threshold(low, step):
    high = low + 2 * step
    base = trapezoidal_nrz()
    sig = signal(base.x, np.where(base.y > 0, high, low))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = eye_diagram(sig, period=1., auto_center=True, threshold=low + step)
    assert result.height == pytest.approx(high - low, rel=1e-12, abs=0.)
    assert result.width == pytest.approx(1.)
