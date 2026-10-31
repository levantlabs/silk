"""Scalar read types, initial-value boundaries and electrical read semantics."""
from fractions import Fraction

import numpy as np
import pytest

from silk.simulation._analysis_context import analysis_scope
from silk.exceptions import SimulationError
from silk.simulation._behavioral_state import SharedState
from silk.simulation._behavioral_views import _PinValue, _BusValue
from tests.test_behavioral_bus_shapes import _spec


def pin(state, quantity='v', index=0, port='x'):
    return _PinValue(state, 'model', port, quantity, index)


@pytest.mark.parametrize('quantity', ['v', 'i'])
@pytest.mark.parametrize('initial', [np.array([3., 4.]), np.array([3.], dtype=np.longdouble),
                                     np.array([1., 2., 3.])])
@pytest.mark.parametrize('empty', [False, True])
def test_vector_initial_operand_is_consumed(quantity, initial, empty):
    state = SharedState(2, {('model', 'x'): 'net'})
    value = pin(state, quantity, slice(None))
    value.initial_value = initial
    value._write([5., 6.])
    index = np.array([], dtype=int) if empty else slice(None)
    expected_shape = (0,) if empty else (2,)
    compatible = initial.size == 1 or (initial.size == 2 and not empty)
    with analysis_scope('transient'):
        if not compatible:
            with pytest.raises(ValueError, match='broadcast|shape'):
                value[index]
        else:
            result = value[index]
            assert result.shape == expected_shape
            assert result.dtype == (np.result_type(float, initial.dtype)
                                    if quantity == 'v' else np.dtype(float))
            np.testing.assert_array_equal(result, [] if empty else [5., 6.])
    assert not state._filled
    assert not state._kcl_currents
    assert state.read_events == ({('model', 'x', 'v', 0)} if quantity == 'v' else set())


def test_vector_bad_initial_shape_precedes_kcl_fill():
    state = SharedState(2, {('model', 'x'): 'net'})
    value = pin(state, 'i', slice(None))
    value.initial_value = [1., 2., 3.]
    with analysis_scope('transient'), pytest.raises(ValueError, match='broadcast|shape'):
        value[:]
    assert not state.read_events
    assert not state._filled
    assert not state._currents['model', 'x'].assigned.any()


@pytest.mark.parametrize('quantity', ['v', 'i'])
@pytest.mark.parametrize('initial', [0, True, 2., np.int16(3), np.float32(4),
                                     np.array(5.), -0., np.nan, np.inf, 1 << 100])
@pytest.mark.parametrize('index', [slice(None), (), Ellipsis])
def test_numeric_initial_scalar_type(quantity, initial, index):
    state = SharedState(2, {('model', 'x'): 'net'})
    value = pin(state, quantity)
    value.initial_value = initial
    with analysis_scope('transient'):
        result = value.previous_value[index]
        assert type(result) is np.float64
        np.testing.assert_equal(result, np.float64(initial))
        assert np.signbit(result) == np.signbit(np.float64(initial))
        assert type(float(value.previous_value)) is float
        assert np.asarray(value.previous_value, dtype=np.float32).dtype == np.float32


class BadArray:
    def __array__(self, *args, **kwargs):
        raise ValueError('cannot convert')


class FloatOnly:
    def __float__(self):
        return 1.0


@pytest.mark.parametrize('bad', ['1.5', b'1', 1+0j, None, object(),
                                 np.array([1], dtype=object), BadArray(),
                                 np.timedelta64(1, 's'), np.array([np.timedelta64(1, 's')]),
                                 FloatOnly(), np.array([Fraction(1, 3)], dtype=object)])
@pytest.mark.parametrize('kind', ['pin', 'delay', 'bus', 'bus-delay'])
@pytest.mark.parametrize('quantity', ['v', 'i'])
def test_initial_refusal_is_contextual_and_atomic(bad, kind, quantity):
    spec = _spec(range(8, 10))
    state = SharedState(2, {('model', name): name for name in ['x', *spec.expanded_names()]})
    with analysis_scope('transient'):
        value = (_BusValue(state, 'model', spec, spec.expanded_names(), quantity, 0) if kind.startswith('bus')
                 else pin(state, quantity))
        if 'delay' in kind:
            value = value.previous_value
        value.initial_value = 7
        before = state.initial_values.copy()
        with pytest.raises(TypeError) as exc:
            value.initial_value = bad
        message = str(exc.value)
        assert 'model' in message
        assert ('code' if kind.startswith('bus') else 'x') in message
        assert repr(bad) in message
        assert state.initial_values == before


def test_big_integer_initial_and_object_array_are_distinct():
    state = SharedState(1, {('model', 'x'): 'net'})
    value = pin(state)
    value.initial_value = 1 << 100
    with analysis_scope('transient'):
        assert value[()] == np.float64(1 << 100)
        with pytest.raises(TypeError):
            value.initial_value = np.array([1 << 100], dtype=object)
        assert value.initial_value == 1 << 100


@pytest.mark.parametrize('index', [slice(None), (), Ellipsis])
def test_initial_conversion_does_not_clamp_overflow(index):
    state = SharedState(1, {('model', 'x'): 'net'})
    value = pin(state)
    value.initial_value = 1 << 10000
    with analysis_scope('transient'), pytest.raises(OverflowError):
        value[index]


@pytest.mark.skipif(np.finfo(np.longdouble).max <= np.finfo(float).max,
                    reason='requires a wider longdouble exponent range')
@pytest.mark.parametrize('index', [slice(None), (), Ellipsis])
@pytest.mark.parametrize('shape', [None, (), (1,)])
def test_wide_real_initial_uses_numpy_float64_conversion(index, shape):
    import warnings
    state = SharedState(1, {('model', 'x'): 'net'})
    value = pin(state)
    value.initial_value = np.longdouble('1e4000')
    if shape is not None:
        value.initial_value = np.array(value.initial_value).reshape(shape)
    with analysis_scope('transient'), np.errstate(over='warn'):
        with warnings.catch_warnings(record=True) as expected:
            warnings.simplefilter('always')
            reference = np.float64(np.asarray(value.initial_value).reshape(())[()])
        with warnings.catch_warnings(record=True) as actual:
            warnings.simplefilter('always')
            result = value[index]
        assert type(result) is np.float64
        assert result == reference == np.inf
        assert [str(w.message) for w in actual] == [str(w.message) for w in expected]


@pytest.mark.parametrize('index', [None, True, False])
@pytest.mark.parametrize('initial', [np.int16(7), np.longdouble(np.finfo(np.longdouble).max)])
@pytest.mark.parametrize('quantity', ['v', 'i'])
def test_non_scalar_index_preserves_initial_dtype_and_range(index, initial, quantity):
    import warnings
    state = SharedState(1, {('model', 'x'): 'net'})
    value = pin(state, quantity)
    value.initial_value = initial
    expected = np.asarray([initial]).reshape(())[index]
    with analysis_scope('transient'), warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        reader = value.previous_value if quantity == 'i' else value
        result = reader[index]
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        np.testing.assert_array_equal(result, expected)
        result[...] = 0
        assert value.initial_value == initial


@pytest.mark.parametrize('quantity', ['v', 'i'])
@pytest.mark.parametrize('written', [False, True])
def test_context_check_precedes_all_read_bookkeeping(quantity, written):
    state = SharedState(1, {('model', 'x'): 'net'})
    if written:
        getattr(state, 'write_voltage' if quantity == 'v' else 'write_current')('model', 'x', 2., 0)
    before = (dict(state._voltages), set(state.read_events), dict(state._filled))
    with pytest.raises(SimulationError, match='only defined during a behavioral run'):
        pin(state, quantity)[()]
    assert before == (state._voltages, state.read_events, state._filled)


@pytest.mark.parametrize('quantity', ['v', 'i'])
@pytest.mark.parametrize('case', ['written', 'unwritten', 'ground', 'fallback', 'delayed'])
@pytest.mark.parametrize('index', [slice(None), (), Ellipsis, None])
def test_sample_value_and_shape_matrix(quantity, case, index):
    state = SharedState(3, {('model', 'x'): '0' if case == 'ground' else 'net'})
    value = pin(state, quantity, 1)
    value.initial_value = -0.
    if case in ('written', 'fallback', 'delayed'):
        getattr(state, 'write_voltage' if quantity == 'v' else 'write_current')(
            'model', 'x', 4., 0 if case != 'written' else 1)
    with analysis_scope('transient'):
        reader = value.previous_value if case == 'delayed' else value
        result = reader[index]
        expected = (4. if case in ('written', 'delayed') or
                    (case == 'fallback' and quantity == 'v') else 0.)
        if index is None:
            assert result.shape == (1,)
            assert result.dtype == (np.dtype(int) if case == 'ground' and quantity == 'v'
                                    else np.dtype(float))
            result[0] = 99
            assert reader[()] == expected
        else:
            assert type(result) is np.float64
            assert result == expected


def test_bad_scalar_index_precedes_wide_initial_conversion():
    import warnings
    state = SharedState(1, {('model', 'x'): 'net'})
    value = pin(state)
    value.initial_value = np.longdouble(np.finfo(np.longdouble).max)
    with analysis_scope('transient'), warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        with pytest.raises(IndexError, match='too many indices'):
            value[0]


@pytest.mark.parametrize('initial', [np.array([3.]), np.array([[3.]]), np.array([]), [1., 2.]])
def test_sample_initial_shapes_keep_reshape_rules(initial):
    state = SharedState(1, {('model', 'x'): 'net'})
    value = pin(state)
    value.initial_value = initial
    with analysis_scope('transient'):
        if np.size(initial) == 1:
            assert type(value[()]) is np.float64
            assert value[()] == 3
        else:
            with pytest.raises(ValueError, match='reshape'):
                value[()]


@pytest.mark.parametrize('bits', [range(8, 9), range(8, 11)])
def test_bus_selection_keeps_unselected_bits_untouched(bits):
    spec = _spec(bits)
    state = SharedState(2, {('model', name): name for name in spec.expanded_names()})
    for quantity in ('v', 'i'):
        value = _BusValue(state, 'model', spec, spec.expanded_names(), quantity, 0)
        with analysis_scope('transient'):
            empty = value[np.array([], dtype=int)]
            assert empty.shape == (0,)
            assert not state.read_events and not state._filled
            result = value[:1]
            assert result.dtype == np.float64
            assert result.shape == (1,)
        assert all(name == 'code[8]' for name in state._voltages)
        assert all(not q.assigned.any() for (path, name), q in state._currents.items()
                   if name != 'code[8]')
        state.read_events.clear()


@pytest.mark.parametrize('outcome', ['success', 'failure', 'cancel'])
def test_retained_reads_after_engine_exit(tmp_path, outcome):
    from tests.test_behavioral_ports import top, block
    from silk.simulation.behavioral import BehavioralEngine
    saved = []
    def model(p, t, q):
        p.vout.v = 3
        saved.extend([p.vout.v, p.vin.v, p.vout.i, p.vin.i])
        if outcome == 'failure':
            raise ValueError('model failed')
        if outcome == 'cancel':
            raise KeyboardInterrupt()
    mod = top(tmp_path)
    mod.add_module(block(tmp_path, 'model', model, execution='sample'), 'model',
                   nets={'vin': 'in', 'vout': 'out'})
    if outcome == 'success':
        BehavioralEngine().run(mod, 1, 1)
    else:
        with pytest.raises(ValueError if outcome == 'failure' else KeyboardInterrupt):
            BehavioralEngine().run(mod, 1, 1)
    for value in saved:
        state = value._state
        before = (len(state._voltages), set(state.read_events), len(state._filled))
        with pytest.raises(SimulationError, match='only defined during a behavioral run'):
            value[()]
        assert before == (len(state._voltages), state.read_events, len(state._filled))


@pytest.mark.parametrize('kind', ['pin', 'delay', 'bus', 'bus-delay'])
@pytest.mark.parametrize('quantity', ['v', 'i'])
@pytest.mark.parametrize('index', [slice(None), (), Ellipsis])
def test_fraction_initials_are_real_and_stored_unchanged(kind, quantity, index):
    spec = _spec(range(8, 10))
    state = SharedState(2, {('model', name): name for name in ['x', *spec.expanded_names()]})
    initial = Fraction(1, 3)
    with analysis_scope('transient'):
        value = (_BusValue(state, 'model', spec, spec.expanded_names(), quantity, 0) if kind.startswith('bus')
                 else pin(state, quantity))
        setter = value.previous_value if 'delay' in kind else value
        setter.initial_value = initial
        assert all(stored is initial for stored in state.initial_values.values())
        # Currents use authored initials only before the first time point.
        reader = (value.previous_value if quantity == 'i' or 'delay' in kind else value)
        result = reader._read(index)
        if kind.startswith('bus'):
            assert result.shape == (2,)
            assert result.dtype == np.float64
            np.testing.assert_array_equal(result, [np.float64(initial)] * 2)
        else:
            assert type(result) is np.float64
            assert result == np.float64(initial)
        for name in spec.expanded_names() if kind.startswith('bus') else ['x']:
            assert type(pin(state, quantity, port=name).previous_value._read(index)) is np.float64


@pytest.mark.parametrize('quantity', ['v', 'i'])
@pytest.mark.parametrize('bus', [False, True])
def test_dc_ignores_fraction_initial(quantity, bus):
    spec = _spec(range(8, 10))
    state = SharedState(1, {('model', name): name for name in ['x', *spec.expanded_names()]})
    value = _BusValue(state, 'model', spec, spec.expanded_names(), quantity, 0) if bus else pin(state, quantity)
    initial = Fraction(2, 3)
    with analysis_scope('dc'):
        value.initial_value = initial
        np.testing.assert_array_equal(value._read(), [0., 0.] if bus else np.float64(0))
        assert all(stored is initial for stored in state.initial_values.values())


def test_nested_context_and_initials_are_live():
    state = SharedState(1, {('model', 'x'): 'net'})
    value = pin(state)
    value.initial_value = 2
    with analysis_scope('transient'):
        assert value[()] == 2
        delayed = value.previous_value
        with analysis_scope('dc'):
            assert value[()] == 0
            with pytest.raises(SimulationError, match='previous_value is unavailable'):
                delayed[()]
        value.initial_value = 3
        assert value[()] == delayed[()] == 3


def test_bus_initial_arrays_broadcast_and_invalid_shape_is_atomic():
    spec = _spec(range(8, 10))
    state = SharedState(2, {('model', name): name for name in spec.expanded_names()})
    value = _BusValue(state, 'model', spec, spec.expanded_names(), 'v', 0)
    value.initial_value = [2., 3.]
    np.testing.assert_array_equal(value.initial_value, [2., 3.])
    before = state.initial_values.copy()
    with pytest.raises(ValueError, match='broadcast'):
        value.initial_value = [1., 2., 3.]
    assert state.initial_values == before
    value.initial_value = np.array([True])
    np.testing.assert_array_equal(value.initial_value, [True, True])
