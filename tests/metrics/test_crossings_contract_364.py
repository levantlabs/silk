"""TASK-364: crossing and crossings share one core.

Owner rulings 2026-10-01: the return type never depends on an argument.
``crossing`` returns ONE float (n=k the k-th, n=-k the k-th from the end;
n=0/None raise), works on any 1-D Signal and is a registered metric.
``crossings`` ALWAYS returns an array and has no ``n``. Both take
edge 'rising'/'falling'/'any' (default 'rising') and a start/end window.
``x_when`` is removed with no alias; ``level`` is gone in favour of
``threshold``.
"""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def tran(t, y):
    return Signal('v', 'voltage', 'time', 'V', np.asarray(t, dtype=float),
                  np.asarray(y, dtype=float))


def wave():
    t = np.linspace(0, 1, 201)
    return tran(t, np.sin(2 * np.pi * 3 * t + 0.3) + 0.1 * t)


def triangle():
    """Piecewise-linear 0 -> 1 -> 0 ... with period 2: rising 0.5-crossings
    at exactly 0.5, 2.5, 4.5, 6.5; falling at 1.5, 3.5, 5.5, 7.5."""
    t = np.arange(9, dtype=float)
    return tran(t, t % 2)


def ac_response():
    f = np.logspace(0, 6, 121)
    return Signal('g', 'voltage', 'frequency', 'V', f, 1 / (1 + 1j * f / 1e3))


# Outputs of the pre-TASK-364 crossings/x_when, captured before the change.
PINNED = {
    (None, 'rising'): [0.3183025854907653, 0.6498761738014855, 0.9814479150054445],
    (None, 'falling'): [0.14896610936192373, 0.4840789130986046, 0.8191914992332855],
    (None, 'any'): [0.14896610936192373, 0.3183025854907653, 0.4840789130986046,
                    0.6498761738014855, 0.8191914992332855, 0.9814479150054445],
    (0.3, 'rising'): [0.0002514094457965877, 0.33176226617967997,
                      0.6632758441935749, 0.9947966657789664],
    (0.3, 'falling'): [0.13533372571575358, 0.4705126459447848, 0.8056755182650607],
    (0.3, 'any'): [0.0002514094457965877, 0.13533372571575358, 0.33176226617967997,
                   0.4705126459447848, 0.6632758441935749, 0.8056755182650607,
                   0.9947966657789664],
}
EDGES = ['rising', 'falling', 'any']


class TestPinnedOutputs:
    @pytest.mark.parametrize('key', sorted(PINNED, key=str))
    def test_crossings_unchanged(self, key):
        threshold, edge = key
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            got = M.crossings(wave(), threshold=threshold, edge=edge)
        assert got.tolist() == PINNED[key]

    @pytest.mark.parametrize('edge', EDGES)
    def test_crossing_matches_old_x_when(self, edge):
        expected = PINNED[(0.3, edge)]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            for n, value in enumerate(expected, start=1):
                assert M.crossing(wave(), 0.3, edge=edge, n=n) == value

    def test_old_x_when_default_is_edge_any(self):
        # x_when(wave(), 0.0) relied on edge='any'; the migration passes it.
        assert M.crossing(wave(), 0.0, edge='any') == 0.1515566014141747

    def test_complex_magnitude_unchanged(self):
        with pytest.raises(ValueError, match='crossing.*complex'):
            M.crossing(ac_response(), 0.5, edge='falling')
        assert M.crossing(ac_response().mag(), 0.5, edge='falling') == 1734.4777917377453


class TestReturnTypes:
    def test_crossing_default_edge_is_rising(self):
        assert M.crossing(triangle(), 0.5) == 0.5
        assert M.crossing(triangle()) == 0.5  # threshold=None: swing midpoint

    def test_crossing_always_returns_float(self):
        assert type(M.crossing(triangle(), 0.5)) is float
        assert type(M.crossing(triangle(), 0.5, n=np.int64(2))) is float

    def test_crossings_always_returns_array(self):
        for edge in EDGES:
            assert isinstance(M.crossings(triangle(), edge=edge), np.ndarray)
        assert M.crossings(triangle(), threshold=0.5, start=2.6, end=2.9).shape == (0,)

    def test_crossings_has_no_n(self):
        with pytest.raises(TypeError):
            M.crossings(triangle(), n=1)

    def test_x_when_is_removed(self):
        import silk.metrics.waveform as T
        assert not hasattr(M, 'x_when')
        assert not hasattr(T, 'x_when')

    def test_level_is_removed(self):
        with pytest.raises(TypeError, match='level'):
            M.crossing(triangle(), level=0.5)

    def test_registration(self):
        from silk.signals.schema import _REGISTRY
        assert 'crossing' in _REGISTRY
        assert 'crossings' not in _REGISTRY
        assert 'x_when' not in _REGISTRY


class TestOccurrence:
    @pytest.mark.parametrize('edge,expected', [
        ('rising', [0.5, 2.5, 4.5, 6.5]),
        ('falling', [1.5, 3.5, 5.5, 7.5]),
        ('any', [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]),
    ])
    def test_each_edge_every_n(self, edge, expected):
        assert M.crossings(triangle(), threshold=0.5, edge=edge).tolist() == expected
        for k, value in enumerate(expected, start=1):
            assert M.crossing(triangle(), 0.5, edge=edge, n=k) == value
            assert M.crossing(triangle(), 0.5, edge=edge, n=-k) == expected[-k]

    def test_negative_n_counts_from_the_end(self):
        assert M.crossing(triangle(), 0.5, n=-1) == 6.5
        assert M.crossing(triangle(), 0.5, edge='falling', n=-2) == 5.5

    @pytest.mark.parametrize('n', [5, -5, 100])
    def test_beyond_count_is_nan_with_warning(self, n):
        with pytest.warns(MetricWarning) as caught:
            got = M.crossing(triangle(), 0.5, edge='rising', n=n)
        assert type(got) is float and math.isnan(got)
        assert len(caught) == 1
        d = caught[0].message
        assert d.metric == 'crossing' and d.code == 'insufficient_crossings'
        assert d.details['crossing_count'] == 4
        assert d.details['requested_crossing'] == n

    @pytest.mark.parametrize('n', [0, None, 1.0, True, '1'])
    def test_invalid_n_raises(self, n):
        with pytest.raises(ValueError, match='crossing: n must be a nonzero integer'):
            M.crossing(triangle(), 0.5, n=n)

    @pytest.mark.parametrize('fn', ['crossings', 'crossing'])
    def test_invalid_edge_raises(self, fn):
        with pytest.raises(ValueError, match=f"{fn}: edge must be"):
            getattr(M, fn)(triangle(), threshold=0.5, edge='all')


class TestWindow:
    def test_start_only(self):
        assert M.crossings(triangle(), threshold=0.5, start=2.0).tolist() == [2.5, 4.5, 6.5]
        assert M.crossing(triangle(), 0.5, start=2.0) == 2.5

    def test_end_only(self):
        assert M.crossings(triangle(), threshold=0.5, end=5.0).tolist() == [0.5, 2.5, 4.5]
        assert M.crossing(triangle(), 0.5, n=-1, end=5.0) == 4.5

    def test_both(self):
        assert M.crossings(triangle(), threshold=0.5, edge='any',
                           start=2.0, end=6.0).tolist() == [2.5, 3.5, 4.5, 5.5]
        assert M.crossing(triangle(), 0.5, edge='any', start=2.0, end=6.0, n=-1) == 5.5

    def test_bounds_between_samples_cut_the_segment(self):
        # The 2.5 crossing lies inside the 2..3 segment: a bound on either
        # side of it inside that same segment decides it, not the samples.
        assert M.crossings(triangle(), threshold=0.5, start=2.4).tolist() == [2.5, 4.5, 6.5]
        assert M.crossings(triangle(), threshold=0.5, start=2.6).tolist() == [4.5, 6.5]
        assert M.crossings(triangle(), threshold=0.5, end=2.6).tolist() == [0.5, 2.5]
        assert M.crossings(triangle(), threshold=0.5, end=2.4).tolist() == [0.5]
        assert M.crossings(triangle(), threshold=0.5, start=2.4, end=2.6).tolist() == [2.5]
        assert M.crossings(triangle(), threshold=0.5, start=2.6, end=2.9).tolist() == []
        assert M.crossing(triangle(), 0.5, start=2.4) == 2.5
        assert M.crossing(triangle(), 0.5, start=2.6) == 4.5

    def test_bound_on_the_crossing_is_inclusive(self):
        assert M.crossings(triangle(), threshold=0.5, start=2.5, end=2.5).tolist() == [2.5]

    def test_auto_threshold_uses_the_window_swing(self):
        # A start-up excursion to 5 V outside the window must not set the
        # midpoint: inside [2, 8] the swing is 0..1, so threshold = 0.5.
        t = np.arange(9, dtype=float)
        y = t % 2
        y[0] = 5.0
        sig = tran(t, y)
        assert M.crossings(sig, start=2.0).tolist() == [2.5, 4.5, 6.5]
        assert M.crossing(sig, start=2.0) == 2.5
        # Without a window the 5 V excursion moves the midpoint to 2.5,
        # which the 0..1 wave never reaches on a rising edge.
        assert M.crossings(sig).tolist() == []

    def test_auto_threshold_inside_one_segment(self):
        # A window narrower than a sample interval still has a swing (the
        # interpolated bounds): 0.2..0.8 on the 0..1 ramp has midpoint 0.5.
        assert M.crossings(triangle(), start=0.2, end=0.8).tolist() == [0.5]

    def test_window_ignores_nan_outside_it(self):
        t = np.arange(9, dtype=float)
        y = t % 2
        y[8] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert M.crossings(tran(t, y), threshold=0.5, end=5.0).tolist() == [0.5, 2.5, 4.5]
            assert M.crossing(tran(t, y), 0.5, n=-1, end=5.0) == 4.5

    def test_nan_inside_window_is_unknown(self):
        t = np.arange(9, dtype=float)
        y = t % 2
        y[3] = np.nan
        with pytest.warns(MetricWarning):
            got = M.crossings(tran(t, y), threshold=0.5, start=2.0)
        assert got.shape == (1,) and np.isnan(got).all()
        with pytest.warns(MetricWarning):
            assert math.isnan(M.crossing(tran(t, y), 0.5, start=2.0))

    @pytest.mark.parametrize('kwargs,part', [
        ({'start': 3.0, 'end': 2.0}, 'require end >= start'),
        ({'start': 9.5}, 'does not overlap the trace'),
        ({'end': -1.0}, 'does not overlap the trace'),
        ({'start': float('nan')}, 'start must be a finite number'),
        ({'end': 'x'}, 'end must be a finite number'),
    ])
    @pytest.mark.parametrize('fn', ['crossings', 'crossing'])
    def test_malformed_window_raises(self, fn, kwargs, part):
        with pytest.raises(ValueError, match=part) as info:
            getattr(M, fn)(triangle(), threshold=0.5, **kwargs)
        assert str(info.value).startswith(fn + ':')
        assert 'trace spans [0.0, 8.0]' in str(info.value)

    def test_crossing_window_on_frequency_axis(self):
        assert M.crossing(ac_response().mag(), 0.5, edge='falling', start=1e3) == 1734.4777917377453
        with pytest.warns(MetricWarning):
            assert math.isnan(M.crossing(ac_response().mag(), 0.5, edge='falling', end=1e3))


class TestSchema:
    def test_crossing_parameters(self):
        from silk.signals.schema import metric_schema
        params = {p['name']: p for p in metric_schema(M.crossing)['parameters']}
        assert list(params) == ['threshold', 'edge', 'n', 'start', 'end']
        assert params['threshold']['unit'] == '[y]'
        assert params['threshold']['required'] is False
        assert params['start']['unit'] == params['end']['unit'] == '[x]'
        assert params['edge']['default'] == 'rising'
        assert params['edge']['choices'] == ['rising', 'falling', 'any']
        assert params['n']['default'] == 1
