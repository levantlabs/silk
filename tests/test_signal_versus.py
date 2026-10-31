"""``Signal.against`` — re-parametrise one Signal against another (TASK-344)."""

import numpy as np
import pytest

from silk.signals.base import Signal


def sig(name, y, x=None):
    y = np.asarray(y)
    x = np.arange(y.size, dtype=float) if x is None else np.asarray(x)
    return Signal(name, 'voltage', 'v(vg)', name, x, y)


class FakeNetlist:
    def __init__(self, signals):
        self.sim_results_loaded = False
        self._signals = signals

    def load_signal(self, name, selector='V'):
        return self._signals[name]


class FakeBench:
    def __init__(self, signals):
        self.test_name = 'tb'
        self.netlist = FakeNetlist(signals)


def test_ascending_x_keeps_order_and_labels():
    a = sig('a', [10, 20, 30])
    b = sig('b', [1, 2, 3])
    r = a.against(b)
    np.testing.assert_array_equal(r.x, [1, 2, 3])
    np.testing.assert_array_equal(r.y, [10, 20, 30])
    assert r.xlabel == 'b'
    assert r.name == 'a vs b'
    assert a.against(b, name='q').name == 'q'


def test_strictly_decreasing_x_is_flipped_to_ascending():
    a = sig('a', [10, 20, 30])
    b = sig('b', [3, 2, 1])
    r = a.against(b)
    np.testing.assert_array_equal(r.x, [1, 2, 3])
    np.testing.assert_array_equal(r.y, [30, 20, 10])
    assert r.interpolate([1.5]).y[0] == pytest.approx(25.0)


@pytest.mark.parametrize('xv, bad', [
    ([1, 3, 2, 4], 1),
    ([1, 1, 2, 3], 0),
    ([3, 2, 2, 1], 1),
    ([3, 4, 2, 1], 1),
])
def test_nonmonotonic_raises_naming_index(xv, bad):
    a = sig('a', [0, 0, 0, 0])
    with pytest.raises(ValueError, match=rf'samples {bad} and {bad + 1}'):
        a.against(sig('b', xv))


def test_allow_nonmonotonic_keeps_sample_order():
    a = sig('a', [10, 20, 30, 40])
    r = a.against(sig('b', [1, 3, 2, 4]), allow_nonmonotonic=True)
    np.testing.assert_array_equal(r.x, [1, 3, 2, 4])
    np.testing.assert_array_equal(r.y, [10, 20, 30, 40])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match='same sweep'):
        sig('a', [1, 2, 3]).against(sig('b', [1, 2]))


def test_complex_raises():
    with pytest.raises(ValueError, match='complex'):
        sig('a', [1, 2]).against(sig('b', np.array([1, 2]) * 1j))


def test_non_signal_raises():
    with pytest.raises(TypeError):
        sig('a', [1, 2]).against([1, 2])


def test_single_sample_ok():
    r = sig('a', [5]).against(sig('b', [7]))
    assert r.x[0] == 7 and r.y[0] == 5


def test_deferred_on_unbound_probes_resolves_after_run():
    a = sig('a', [10, 20, 30])
    b = sig('b', [3, 2, 1])
    bench = FakeBench({'a': a, 'b': b})
    pa = Signal._unbound('a', 'V', bench)
    pb = Signal._unbound('b', 'V', bench)
    r = (pa / pb).against(pb)
    assert r._is_unbound()
    bench.netlist.sim_results_loaded = True
    eager = (a / b).against(b)
    np.testing.assert_allclose(r.x, eager.x)
    np.testing.assert_allclose(r.y, eager.y)


def test_versus_is_the_name_and_against_is_a_silent_alias():
    a, b = sig('a', [1, 2, 3]), sig('b', [10, 20, 30])
    r = a.versus(b)
    assert list(r.x) == [10, 20, 30] and list(r.y) == [1, 2, 3]
    assert r.name == 'a vs b' and r.xlabel == 'b'
    assert Signal.against is Signal.versus
    assert list(a.against(b).x) == list(r.x)


def test_versus_defers_on_unbound_probes_and_refuses_bad_x():
    with pytest.raises(TypeError, match='versus'):
        sig('a', [1, 2]).versus([1, 2])
    with pytest.raises(ValueError, match='versus'):
        sig('a', [1, 2, 3]).versus(sig('b', [1, 3, 2]))
