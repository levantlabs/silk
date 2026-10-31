"""Tests for arithmetic on Variable — derived expressions lazily
evaluate from their leaf operands."""

import pytest
from silk.circuit.variable import _VarExpr
from silk.circuit.variable import Variable


class TestBasicArithmetic:

    def test_rmul_scalar_by_var(self):
        w = Variable('w', value=2.0)
        expr = 3 * w
        assert isinstance(expr, _VarExpr)
        assert expr.value == 6.0

    def test_mul_var_by_scalar(self):
        w = Variable('w', value=2.0)
        expr = w * 3
        assert expr.value == 6.0

    def test_add(self):
        a = Variable('a', value=2.0)
        b = Variable('b', value=3.0)
        expr = a + b
        assert expr.value == 5.0

    def test_radd_scalar(self):
        a = Variable('a', value=2.0)
        expr = 10 + a
        assert expr.value == 12.0

    def test_sub(self):
        a = Variable('a', value=10.0)
        b = Variable('b', value=3.0)
        expr = a - b
        assert expr.value == 7.0

    def test_rsub_scalar(self):
        a = Variable('a', value=3.0)
        expr = 10 - a
        assert expr.value == 7.0

    def test_truediv(self):
        a = Variable('a', value=10.0)
        expr = a / 2
        assert expr.value == 5.0

    def test_rtruediv(self):
        a = Variable('a', value=5.0)
        expr = 20 / a
        assert expr.value == 4.0

    def test_floordiv(self):
        a = Variable('a', value=10)
        expr = a // 3
        assert expr.value == 3

    def test_neg(self):
        a = Variable('a', value=5.0)
        expr = -a
        assert expr.value == -5.0


class TestChaining:

    def test_chained_mul_add(self):
        a = Variable('a', value=2.0)
        b = Variable('b', value=3.0)
        expr = 2 * a + b       # 4 + 3 = 7
        assert expr.value == 7.0

    def test_deep_chain(self):
        a = Variable('a', value=4.0)
        b = Variable('b', value=2.0)
        expr = (a + b) * 3 - a / 2     # 6*3 - 2 = 16
        assert expr.value == 16.0

    def test_expression_of_expressions(self):
        a = Variable('a', value=2.0)
        b = Variable('b', value=3.0)
        ab = a * b
        sum_expr = ab + a       # 6 + 2 = 8
        assert sum_expr.value == 8.0


class TestLazyEvaluation:

    def test_leaf_update_propagates(self):
        """Core invariant: updating the leaf's .value changes the
        expression's .value on the next read."""
        w = Variable('w', value=2.0)
        big = 2 * w
        assert big.value == 4.0
        w.value = 5.0
        assert big.value == 10.0

    def test_leaf_update_through_deep_expr(self):
        a = Variable('a', value=2.0)
        b = Variable('b', value=3.0)
        expr = (a + b) * 2
        assert expr.value == 10.0
        a.value = 7.0
        assert expr.value == 20.0        # (7+3)*2
        b.value = 0.0
        assert expr.value == 14.0        # (7+0)*2

    def test_expression_has_no_settable_value(self):
        """Expressions are derived — .value is read-only."""
        a = Variable('a', value=2.0)
        expr = 2 * a
        with pytest.raises(AttributeError):
            expr.value = 99.0


class TestIntegerSemantics:

    def test_integer_mult_stays_integer(self):
        m = Variable('m', value=4)
        expr = 2 * m
        assert expr.value == 8
        assert isinstance(expr.value, int)

    def test_floordiv_stays_integer(self):
        m = Variable('m', value=8)
        expr = m // 2
        assert expr.value == 4
        assert isinstance(expr.value, int)


class TestMissingValue:

    def test_unset_leaf_raises(self):
        a = Variable('a', value=None)
        expr = 2 * a
        with pytest.raises(ValueError, match='no value'):
            _ = expr.value


class TestIntegrationWithSpice:
    """Expressions must flow through _render_param — devices should
    render the computed numeric value, not the expression repr."""

    def test_device_param_accepts_expression(self, pdk):
        m = Variable('m', value=4)
        mn = pdk.nmos1p8('mn1', d='vo', g='vi', s='gnd', b='gnd',
                         w=1, l=0.18, m=2 * m)
        spice = mn.to_spice()
        # m=8 should appear (as '8' or '8m'-like SI form — '8' is expected)
        assert 'm=8' in spice

    def test_device_param_expression_updates(self, pdk):
        m = Variable('m', value=4)
        mn = pdk.nmos1p8('mn1', d='vo', g='vi', s='gnd', b='gnd',
                         w=1, l=0.18, m=2 * m)
        assert 'm=8' in mn.to_spice()
        m.value = 5
        assert 'm=10' in mn.to_spice()


def test_variable_lives_in_circuit_variable_module():
    import silk
    import silk.circuit.variable as v
    assert silk.Variable is v.Variable
    assert silk.circuit.Variable is v.Variable
    for name in ('_VarExpr', '_VarArithmeticMixin'):
        assert hasattr(v, name)
    import silk.circuit._module_base as mb
    for name in ('Variable', 'VarType', 'Distribution', '_VarExpr', '_VarArithmeticMixin'):
        assert not hasattr(mb, name)


OLD = 'Module' + 'Variable'


def test_old_names_are_removed():
    import importlib
    import silk
    import silk.circuit
    import silk.circuit.modules
    import silk.circuit._module_base
    import silk.optimize
    for mod in (silk, silk.circuit, silk.circuit.modules,
                silk.circuit._module_base):
        assert not hasattr(mod, OLD)
    assert not hasattr(silk.optimize, 'Variable')
    for name in ('Variable', 'VarType', 'Distribution'):
        assert not hasattr(silk.circuit.modules, name)
    for stmt in ('from silk import ' + OLD,
                 'from silk.circuit import ' + OLD,
                 'from silk.circuit.modules import ' + OLD,
                 'from silk.optimize import Variable'):
        with pytest.raises(ImportError):
            exec(stmt, {})


def test_expression_repr_reads_as_arithmetic():
    from silk.circuit.variable import Variable
    w, wa, wb = Variable('w', 1.0), Variable('w_a', 1.0), Variable('w_b', 2.0)
    w_in = Variable('w_in', 2.0)
    assert repr(2 * w_in) == '2*w_in'
    assert repr((wa + wb) / 2) == '(w_a + w_b)/2'
    assert repr(-w) == '-w'
    assert repr(-(wa + wb)) == '-(w_a + w_b)'
    assert repr(wa - (wb - w)) == 'w_a - (w_b - w)'
    assert repr(wa * wb + w) == 'w_a*w_b + w'
    assert repr(wa * (wb + w)) == 'w_a*(w_b + w)'
    assert repr(w // 2) == 'w//2'


def test_unset_variable_error_names_the_variable():
    import pytest
    from silk.circuit.variable import Variable
    a, b = Variable('w_a', 1.0), Variable('w_unset')
    with pytest.raises(ValueError, match="'w_unset'"):
        (2 * b).value
    with pytest.raises(ValueError, match="'w_unset'"):
        ((a + 1) * (a - b)).value


@pytest.mark.parametrize('form', [
    'dict_key', 'dict_value', 'list', 'tuple', 'set', 'frozenset',
    'array', 'structured_array', 'structured_scalar',
])
def test_leaf_variables_in_authored_containers(form):
    import numpy as np
    from silk.circuit.variable import _leaf_variables
    var = Variable('unset')
    expr = 1 / var
    structured = np.empty(1, dtype=[('number', 'f8'), ('object', object)])
    structured['object'][0] = expr
    array = np.empty(1, dtype=object)
    array[0] = expr
    values = {
        'dict_key': {expr: 0}, 'dict_value': {'x': expr},
        'list': [expr], 'tuple': (expr,), 'set': {expr},
        'frozenset': frozenset([expr]), 'array': array,
        'structured_array': structured, 'structured_scalar': structured[0],
    }
    assert _leaf_variables(values[form]) == [var]


def test_leaf_variables_cycles_repeats_and_declarations_do_not_resolve(monkeypatch):
    import numpy as np
    from silk.circuit.variable import _leaf_variables
    from silk.simulation.variation import Normal, Uniform
    a, b, c, d = [Variable(n) for n in 'abcd']
    normal, uniform = Normal(a * 2, b), Uniform(c, d / 0)

    def refuse(*args, **kwargs):
        raise AssertionError('discovery must not resolve or draw')

    monkeypatch.setattr(Normal, 'resolve', refuse)
    monkeypatch.setattr(Uniform, 'resolve', refuse)
    monkeypatch.setattr(np.random, 'default_rng', refuse)
    values = [normal, uniform, a]
    array = np.empty(2, dtype=object)
    array[0], array[1] = values, array
    values.extend([values, array])
    assert _leaf_variables(values) == [a, b, c, d, a]
    assert _leaf_variables(np.zeros(2)) == []
    assert _leaf_variables(np.zeros(1, dtype=[('x', 'f8')])[0]) == []
