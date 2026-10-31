"""final_value (TASK-323.3) and settling_time's optional target."""
import math
import warnings

import numpy as np
import pytest

import silk.metrics as M
from silk.metrics.diagnostics import MetricWarning
from silk.signals.base import Signal


def sig(y, x=None, xlabel='time', ylabel='V'):
    y = np.asarray(y)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    return Signal('s', 'voltage', xlabel, ylabel, x, y)


def test_tran_returns_last_sample():
    assert M.final_value(sig([0.0, 0.5, 1.25])) == 1.25


def test_dc_sweep_end_point():
    s = sig([1.0, 2.0, 4.0], x=[0.0, 0.9, 1.8], xlabel='voltage')
    assert M.final_value(s) == 4.0


def test_one_sample_op_point():
    assert M.final_value(sig([0.7], x=[0.0])) == 0.7


def test_complex_trace_preserves_value_or_explicit_magnitude():
    s = sig([1 + 0j, 3 + 4j], xlabel='frequency')
    assert M.final_value(s) == 3+4j
    assert M.final_value(s.mag()) == pytest.approx(5.0)


def test_matches_value_at_last_x():
    s = sig([0.3, -2.0, 7.5])
    assert M.final_value(s) == M.value_at(s, s.x[-1])


def test_empty_trace_raises():
    with pytest.raises(ValueError, match='final_value'):
        M.final_value(sig([]))


def test_nan_last_sample_returns_nan_with_warning():
    with pytest.warns(MetricWarning) as caught:
        value = M.final_value(sig([1.0, 2.0, np.nan]))
    assert math.isnan(value)
    assert caught[0].message.metric == 'value_at'


def _step():
    t = np.linspace(0, 10, 101)
    y = 0.95 * (1 - np.exp(-t))
    return sig(y, x=t)


def test_settling_time_default_target_is_final_value():
    s = _step()
    default = M.settling_time(s, initial=0.0, tol=0.01)
    explicit = M.settling_time(s, initial=0.0, target=float(s.y[-1]), tol=0.01)
    assert default == explicit and not math.isnan(default)


def test_settling_time_explicit_target_unchanged():
    t = np.linspace(0, 10, 101)
    y = 1 - np.exp(-t)
    s = sig(y, x=t)
    assert 4.5 < M.settling_time(s, initial=0.0, target=1.0, tol=0.01) < 4.8
    with pytest.raises(ValueError, match='zero-swing'):
        M.settling_time(s, initial=1.0, target=1.0)


def test_settling_time_default_target_zero_swing_is_nan():
    # A measured target equal to initial is degenerate data (decision-1):
    # nan with a MetricWarning, unlike an explicit equal target, which raises.
    with pytest.warns(MetricWarning, match='initial level'):
        v = M.settling_time(sig([1.5, 1.5, 1.5]), initial=1.5)
    assert math.isnan(v)


def test_settling_time_default_target_nan_trace_is_nan():
    with pytest.warns(MetricWarning):
        v = M.settling_time(sig([0.0, np.nan, 1.0]), initial=0.0)
    assert math.isnan(v)
