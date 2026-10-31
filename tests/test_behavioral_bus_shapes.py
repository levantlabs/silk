"""Bus shapes for the ports view: stacking, splitting and declared bit numbers (design 3.1, 3.2)."""
import numpy as np
import pytest

from silk.circuit.bus_spec import BusSpec
from silk.circuit.circuit_elements import PortDir
from silk.exceptions import CircuitError


def _spec(bits, naming='bracketed'):
    from silk.circuit.bus_spec import indices_from_bits
    return BusSpec(name='code', indices=indices_from_bits(bits), direction=PortDir.IN,
                   role=None, domain=None, naming=naming)


def _views():
    from silk.simulation import _behavioral_views
    return _behavioral_views


@pytest.mark.parametrize('bits,naming,names', [
    (3, 'bracketed', ['code[0]', 'code[1]', 'code[2]']),
    (range(8, 11), 'bracketed', ['code[8]', 'code[9]', 'code[10]']),
    (range(8, 11), 'underscore', ['code_8', 'code_9', 'code_10']),
])
def test_vector_stack_follows_expanded_names(bits, naming, names):
    spec = _spec(bits, naming)
    values = {name: np.full(4, float(k)) for k, name in enumerate(names)}
    stacked = _views().stack_bus(spec, spec.expanded_names(), values, execution='vector')
    assert stacked.shape == (3, 4)
    np.testing.assert_array_equal(stacked[:, 0], [0.0, 1.0, 2.0])


def test_sample_stack_is_one_value_per_bit():
    spec = _spec(range(8, 11))
    values = {'code[8]': 0.0, 'code[9]': 1.0, 'code[10]': 2.0}
    stacked = _views().stack_bus(spec, spec.expanded_names(), values, execution='sample')
    assert stacked.shape == (3,)
    np.testing.assert_array_equal(stacked, [0.0, 1.0, 2.0])


def test_width_one_bus_keeps_its_bus_axis():
    spec = _spec(1)
    assert _views().stack_bus(spec, spec.expanded_names(), {'code[0]': np.zeros(5)}, execution='vector').shape == (1, 5)
    assert _views().stack_bus(spec, spec.expanded_names(), {'code[0]': 0.5}, execution='sample').shape == (1,)


@pytest.mark.parametrize('execution,good,bad', [
    ('vector', (3, 4), (2, 4)),
    ('vector', (3, 4), (4, 3)),
    ('sample', (3,), (2,)),
])
def test_split_checks_the_first_axis(execution, good, bad):
    spec = _spec(range(8, 11))
    split = _views().split_bus(spec, spec.expanded_names(), np.arange(np.prod(good), dtype=float).reshape(good),
                               execution=execution, n=4)
    assert list(split) == ['code[8]', 'code[9]', 'code[10]']
    with pytest.raises(ValueError, match=r"'code'.*3"):
        _views().split_bus(spec, spec.expanded_names(), np.zeros(bad), execution=execution, n=4)


def test_split_width_one_bus():
    spec = _spec(1)
    split = _views().split_bus(spec, spec.expanded_names(), np.ones((1, 5)), execution='vector', n=5)
    np.testing.assert_array_equal(split['code[0]'], np.ones(5))
    with pytest.raises(ValueError):
        _views().split_bus(spec, spec.expanded_names(), np.ones(5), execution='vector', n=5)
    assert _views().split_bus(spec, spec.expanded_names(), np.array([0.5]), execution='sample', n=5)['code[0]'] == 0.5


def test_bit_subscripts_name_declared_numbers():
    spec = _spec(range(8, 11))
    assert _views().bus_bit(spec, spec.expanded_names(), 8) == 'code[8]'
    assert _views().bus_bit(spec, spec.expanded_names(), 10) == 'code[10]'
    with pytest.raises(CircuitError, match='declared indices'):
        _views().bus_bit(spec, spec.expanded_names(), 0)
    with pytest.raises(CircuitError):
        _views().bus_bit(spec, spec.expanded_names(), -1)
    spec = _spec(range(8, 11), 'underscore')
    assert _views().bus_bit(spec, spec.expanded_names(), 9) == 'code_9'
