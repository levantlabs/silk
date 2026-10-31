"""Field calibration composes and validates all writes before committing."""
import pytest

from silk.circuit.variable import Variable
from silk.exceptions import ConfigurationError, NotSupportedError, VariationError
from silk.specs import CalibrationBinding, Measurement, Spec
from silk.specs.reconciler import reconcile
from silk.simulation.variation import Normal, Uniform


class Block:
    module_name = 'block'
    net_dict = {}
    elem_dict = {}

    def __init__(self, **params):
        self._behavioral_params = params


def binding(block, param, value, field=None, source='spice'):
    spec = Spec(name=f'{param}_{field}', module=block, metric='value', op='<=', target=10)
    spec.calibration = CalibrationBinding(block, param, field=field)
    spec.record_measurement(Measurement(source=source, value=value))
    return spec


def test_numeric_fields_compose_without_losing_either():
    old = Normal(0, 1)
    block = Block(offset=old)
    reconcile(block, [binding(block, 'offset', 2, 'nominal'), binding(block, 'offset', 3, 'sigma')])
    new = block._behavioral_params['offset']
    assert isinstance(new, Normal) and new is not old
    assert (new.nominal, new.sigma) == (2, 3)
    assert (old.nominal, old.sigma) == (0, 1)


def test_uniform_joint_valid_update():
    block = Block(offset=Uniform(0, 1))
    reconcile(block, [binding(block, 'offset', 2, 'low'), binding(block, 'offset', 3, 'high')])
    assert block._behavioral_params['offset'].resolve().fields == (2, 3)


def test_joint_invalid_update_leaves_all_targets_unchanged():
    old = Uniform(0, 1)
    sigma = Variable('sigma', value=1, bounds=(0, 10))
    block = Block(offset=old, normal=Normal(0, sigma), gain=1)
    specs = [binding(block, 'gain', 2), binding(block, 'normal', 2, 'sigma'),
             binding(block, 'offset', 3, 'low'), binding(block, 'offset', 2, 'high')]
    with pytest.raises(VariationError):
        reconcile(block, specs)
    assert block._behavioral_params['offset'] is old
    assert block._behavioral_params['gain'] == 1 and sigma.value == 1


def test_variable_field_retains_identity():
    sigma = Variable('sigma', value=1, bounds=(0, 10))
    old = Normal(0, sigma)
    block = Block(offset=old)
    reconcile(block, [binding(block, 'offset', 2, 'sigma')])
    assert block._behavioral_params['offset'] is old
    assert old.sigma is sigma and sigma.value == 2


@pytest.mark.parametrize('value', [-1, 11])
def test_invalid_sigma_or_variable_bounds_is_atomic(value):
    sigma = Variable('sigma', value=1, bounds=(-2, 10))
    block = Block(offset=Normal(0, sigma), gain=1)
    with pytest.raises((VariationError, ValueError)):
        reconcile(block, [binding(block, 'gain', 2), binding(block, 'offset', value, 'sigma')])
    assert sigma.value == 1 and block._behavioral_params['gain'] == 1


def test_aliases_of_one_variable_refuse_even_equal_values():
    shared = Variable('shared', value=1, bounds=(0, 10))
    block = Block(a=Normal(0, shared), b=Normal(0, shared))
    with pytest.raises(ConfigurationError, match='Variable'):
        reconcile(block, [binding(block, 'a', 2, 'sigma'), binding(block, 'b', 2, 'sigma')])
    assert shared.value == 1


def test_expression_field_refuses():
    variable = Variable('sigma', value=1)
    block = Block(offset=Normal(0, variable * 2))
    with pytest.raises(NotSupportedError):
        reconcile(block, [binding(block, 'offset', 3, 'sigma')])
    assert variable.value == 1


@pytest.mark.parametrize('field', [None, 'low'])
def test_missing_or_wrong_field_lists_valid_fields(field):
    block = Block(offset=Normal(0, 1))
    with pytest.raises(VariationError, match='nominal.*sigma'):
        reconcile(block, [binding(block, 'offset', 2, field)])


def test_filtered_binding_does_not_validate_target():
    block = Block(offset=Normal(0, 1))
    report = reconcile(block, [binding(block, 'offset', -2, 'low', source='behavioral')])
    assert not report.calibrations_applied
    assert block._behavioral_params['offset'].sigma == 1


def test_two_bindings_same_variable_field_refuse_even_equal_values():
    sigma = Variable('sigma', value=1)
    block = Block(offset=Normal(0, sigma))
    with pytest.raises(ConfigurationError, match='Variable'):
        reconcile(block, [binding(block, 'offset', 2, 'sigma'), binding(block, 'offset', 2, 'sigma')])
    assert sigma.value == 1


def test_unrelated_unresolved_declaration_does_not_block_plain_calibration():
    block = Block(gain=1, offset=Normal(0, Variable('unset')))
    reconcile(block, [binding(block, 'gain', 2)])
    assert block._behavioral_params['gain'] == 2


def test_shared_variable_validates_unbound_sibling_declaration(pdk, tmp_path):
    from silk import Module
    shared = Variable('shared', value=1)
    root = Module('sky130', str(tmp_path), 'root.cir', pdk_details=pdk, module_name='root')
    a = Module('sky130', str(tmp_path), 'a.cir', pdk_details=pdk, module_name='a')
    b = Module('sky130', str(tmp_path), 'b.cir', pdk_details=pdk, module_name='b')
    def model_1(ports, t, params):
        return
    a.set_behavioral_model(model_1, params={'offset': Normal(0, shared)}, execution='vector')
    def model_2(ports, t, params):
        return
    b.set_behavioral_model(model_2, params={'offset': Uniform(shared, 2)}, execution='vector')
    root.add_module(a, 'a', nets={})
    root.add_module(b, 'b', nets={})
    with pytest.raises(VariationError):
        reconcile(root, [binding(a, 'offset', 3, 'sigma')])
    assert shared.value == 1


@pytest.mark.parametrize('expression', [False, True])
def test_plain_parameter_keeps_scalar_replacement_contract(expression):
    variable = Variable('gain', value=1, bounds=(0, 1))
    target = variable * 2 if expression else variable
    block = Block(gain=target)
    reconcile(block, [binding(block, 'gain', 3)])
    assert block._behavioral_params['gain'] == 3
    assert variable.value == 1


def test_equal_plain_parameter_bindings_still_coalesce():
    variable = Variable('gain', value=1)
    block = Block(gain=variable)
    reconcile(block, [binding(block, 'gain', 3), binding(block, 'gain', 3)])
    assert block._behavioral_params['gain'] == 3
    assert variable.value == 1


@pytest.mark.parametrize('value', [float('inf'), -float('inf')])
def test_signed_infinity_sigma_still_refuses_atomically(value):
    old = Normal(0, 1)
    block = Block(offset=old, gain=1)
    with pytest.raises(VariationError, match='finite'):
        reconcile(block, [binding(block, 'gain', 2),
                          binding(block, 'offset', value, 'sigma')])
    assert block._behavioral_params['offset'] is old
    assert block._behavioral_params['gain'] == 1
