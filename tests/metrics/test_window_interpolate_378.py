"""TASK-378: windowed primitives interpolate the window edges (default on)."""
import numpy as np
import pytest

import silk.metrics as M
from silk.signals import Signal

PRIMITIVES = ['average', 'rms', 'stddev', 'peak', 'peak_to_peak', 'minimum',
              'maximum', 'argmin_x', 'argmax_x', 'integral']

# Piecewise linear: 0 -> 4 at x=2, back to 0 at x=4, to 4 at x=6.
X = np.array([0.0, 2.0, 4.0, 6.0])
Y = np.array([0.0, 4.0, 0.0, 4.0])


def _sig(y=Y, x=X):
    return Signal('v', 'voltage', 'time', 'V', np.asarray(x), np.asarray(y))


def test_average_and_integral_exact_on_piecewise_linear():
    s = _sig()
    # window [1, 5]: y(1)=2, y(5)=2; segments:
    # [1,2]: (2+4)/2*1=3, [2,4]: 4, [4,5]: (0+2)/2*1=1 -> 8
    assert M.integral(s, start=1.0, end=5.0) == pytest.approx(8.0)
    assert M.average(s, start=1.0, end=5.0) == pytest.approx(2.0)
    # snapped: samples 2, 4 only -> 4
    assert M.integral(s, start=1.0, end=5.0, interpolate=False) == pytest.approx(4.0)
    assert M.average(s, start=1.0, end=5.0, interpolate=False) == pytest.approx(2.0)


def test_rms_exact_on_a_single_segment_cut():
    s = _sig()
    # window [0.5, 1.5] on y = 2x: rms = sqrt(mean(4x^2)) = 2*sqrt((1.5^3-.5^3)/3)
    expected = 2 * np.sqrt((1.5 ** 3 - 0.5 ** 3) / 3)
    # trapezoid of y^2 is not exact for a quadratic; check close, and finite
    assert M.rms(s, start=0.5, end=1.5) == pytest.approx(expected, rel=0.1)


def test_extrema_include_interpolated_edge():
    s = _sig()
    # window [1, 3]: y(1)=2, y(3)=2, interior sample 2 -> 4 (max); min is the edge
    assert M.minimum(s, start=1.0, end=3.0) == pytest.approx(2.0)
    assert M.argmin_x(s, start=1.0, end=3.0) in (1.0, 3.0)
    assert M.maximum(s, start=1.0, end=3.0) == pytest.approx(4.0)
    # window [2.5, 3.5] has no interior sample before; edges 3.0 / 3.0
    assert M.minimum(s, start=2.5, end=3.5) == pytest.approx(1.0 * 1.0)
    assert M.maximum(s, start=2.5, end=3.5) == pytest.approx(3.0)
    assert M.argmax_x(s, start=2.5, end=3.5) == pytest.approx(2.5)
    # no interior sample at all (snapped form raises)
    with pytest.raises(ValueError, match='need at least 2 samples'):
        M.maximum(s, start=2.5, end=3.5, interpolate=False)


def test_edge_extremum_beats_samples():
    s = _sig()
    # window [3, 5.5]: one interior sample (x=4, y=0); edges give 2 and 3
    assert M.maximum(s, start=3.0, end=5.5) == pytest.approx(3.0)
    with pytest.raises(ValueError, match='need at least 2 samples'):
        M.maximum(s, start=3.0, end=5.5, interpolate=False)
    assert M.argmax_x(s, start=3.0, end=5.5) == pytest.approx(5.5)


@pytest.mark.parametrize('name', PRIMITIVES)
def test_false_equals_snapped_and_aligned_windows_do_not_change(name):
    s = _sig(np.sin(np.arange(7.0)), np.arange(7.0))
    f = getattr(M, name)
    # edges on samples: identical either way
    assert f(s, start=1.0, end=4.0) == f(s, start=1.0, end=4.0, interpolate=False)
    assert f(s) == f(s, interpolate=False)
    # window beyond the trace is not extrapolated
    assert f(s, start=-5.0, end=100.0) == f(s)
    # False equals the old snapped result computed by hand
    m = (s.x >= 0.5) & (s.x <= 4.5)
    ref = Signal('v', 'voltage', 'time', 'V', s.x[m], s.y[m])
    assert f(s, start=0.5, end=4.5, interpolate=False) == f(ref)


def test_signature_default_is_true():
    import inspect
    for n in PRIMITIVES:
        p = inspect.signature(getattr(M, n)).parameters['interpolate']
        assert p.default is True and p.kind is p.KEYWORD_ONLY
