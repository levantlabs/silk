"""Operator recipes on unbound Signals (TASK-215.3).

See ``docs/probes.md`` for declaring expressions before a run. Builds on the
unbound state (``silk/signals/base.py``): if either operand of ``+``, ``-``,
``*``, ``/``, ``**`` or unary ``-`` (and the reflected scalar forms) is
unbound, the operator returns a new unbound Signal recording a *recipe*
``(op, *operands)`` — ``op`` a module-level ``operator.*`` builtin, never a
lambda — instead of computing. Resolving the recipe (``resolve()``) resolves
each Signal operand recursively and calls ``op`` on the results, so the
actual arithmetic is defined exactly once, in TASK-214's bound
implementations; this file checks the deferred path always lands on the
same numbers the eager path already produces.

The fake bench here is a small, self-contained copy of
``tests/test_signal_unbound.py``'s: a probe only needs a bench exposing
``netlist.sim_results_loaded`` and ``netlist.load_signal(name, selector)``.
"""

import copy
import operator
import pickle

import numpy as np
import pytest

from silk.signals.base import Signal

ATOL = 1e-12


# ---------------------------------------------------------------------------
# Fake bench: a name -> bound-Signal table, answering load_signal by name.
# ---------------------------------------------------------------------------

class FakeNetlist:
    def __init__(self, signals):
        self.sim_results_loaded = False
        self._signals = dict(signals)
        self.calls = []

    def load_signal(self, name, selector='V'):
        self.calls.append((name, selector))
        if not self.sim_results_loaded:
            raise AssertionError('load_signal called before results exist')
        return self._signals[name]


class FakeBench:
    def __init__(self, signals, name='tb'):
        self.test_name = name
        self.netlist = FakeNetlist(signals)


def make_bound(name, type_='voltage', xlabel='time', x=None, y=None):
    if x is None:
        x = np.linspace(0.0, 1e-6, 5)
    if y is None:
        y = np.arange(1.0, 6.0)
    return Signal(name, type_, xlabel, name, np.asarray(x), np.asarray(y))


def bench_with(*bound_signals, name='tb'):
    return FakeBench({s.name: s for s in bound_signals}, name=name)


def probe(bench, name, selector='V'):
    return Signal._unbound(name, selector, bench)


def run(bench):
    bench.netlist.sim_results_loaded = True


def _walk_ops(signal):
    """Yield every recipe callable reachable from *signal* (itself included)."""
    recipe = signal.__dict__.get('_recipe')
    if recipe is None:
        return
    op, *operands = recipe
    yield op
    for operand in operands:
        if isinstance(operand, Signal):
            yield from _walk_ops(operand)


# ---------------------------------------------------------------------------
# Signal <op> scalar, and the reflected scalar <op> Signal forms
# ---------------------------------------------------------------------------

#: Real scalars only — TASK-214 deliberately excludes complex scalar
#: operands (``_is_real_scalar``); complex dtype is exercised through
#: complex Y-DATA in :meth:`TestSignalScalar.test_complex_dtype_preserved`.
SCALARS = [2, 2.0, -3.5]

#: (label, eager-callable) — the callable is test-local, not part of a
#: recipe, so a lambda here is fine; only what silk *records* must avoid one.
SCALAR_OPS = [
    ('add', lambda sig, s: sig + s),
    ('radd', lambda sig, s: s + sig),
    ('sub', lambda sig, s: sig - s),
    ('rsub', lambda sig, s: s - sig),
    ('mul', lambda sig, s: sig * s),
    ('rmul', lambda sig, s: s * sig),
    ('truediv', lambda sig, s: sig / s),
    ('rtruediv', lambda sig, s: s / sig),
]


class TestSignalScalar:

    @pytest.mark.parametrize('label,fn', SCALAR_OPS, ids=[o[0] for o in SCALAR_OPS])
    @pytest.mark.parametrize('scalar', SCALARS)
    def test_unbound_operand_defers_and_resolves(self, label, fn, scalar):
        vp = make_bound('vp', y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        bench = bench_with(vp)
        p = probe(bench, 'vp')

        eager = fn(vp, scalar)
        lazy = fn(p, scalar)

        assert lazy._is_unbound()
        assert 'y' not in lazy.__dict__
        assert lazy.type == eager.type

        run(bench)
        resolved = lazy.resolve()
        assert resolved.bound is True
        np.testing.assert_allclose(resolved.y, eager.y, atol=ATOL)
        np.testing.assert_allclose(lazy.y, eager.y, atol=ATOL)     # __getattr__ path too
        assert lazy.xlabel == eager.xlabel

    def test_complex_dtype_preserved(self):
        """Complex Y-DATA (an AC-sweep Signal) stays complex through a
        chain of deferred, real-scalar operators."""
        y = np.array([1 + 2j, -1j, 3.0, 0.0, 2 - 2j])
        vp = make_bound('vp', y=y)
        bench = bench_with(vp)
        p = probe(bench, 'vp')
        eager = (vp * 2.0) + 3
        lazy = (p * 2.0) + 3
        run(bench)
        assert lazy.y.dtype == eager.y.dtype == np.complex128
        np.testing.assert_allclose(lazy.y, eager.y, atol=ATOL)


# ---------------------------------------------------------------------------
# Signal <op> Signal, with either/both operands unbound
# ---------------------------------------------------------------------------

#: Signal-Signal ops: add/sub need the SAME type to succeed eagerly (as
#: today); mul/truediv compose types and accept anything.
SIGNAL_SIGNAL_OPS = [
    ('add', lambda a, b: a + b, 'voltage', 'voltage'),
    ('sub', lambda a, b: a - b, 'voltage', 'voltage'),
    ('mul', lambda a, b: a * b, 'voltage', 'current'),
    ('truediv', lambda a, b: a / b, 'voltage', 'current'),
]


class TestSignalSignal:

    @pytest.mark.parametrize(
        'label,fn,ta,tb', SIGNAL_SIGNAL_OPS, ids=[o[0] for o in SIGNAL_SIGNAL_OPS]
    )
    @pytest.mark.parametrize('unbound_side', ['left', 'right', 'both'])
    def test_resolves_to_the_eager_result(self, label, fn, ta, tb, unbound_side):
        va = make_bound('va', type_=ta, y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        vb = make_bound('vb', type_=tb, y=np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        bench = bench_with(va, vb)
        pa = probe(bench, 'va', 'V' if ta == 'voltage' else 'I')
        pb = probe(bench, 'vb', 'V' if tb == 'voltage' else 'I')

        eager = fn(va, vb)
        if unbound_side == 'left':
            lazy = fn(pa, vb)
        elif unbound_side == 'right':
            lazy = fn(va, pb)
        else:
            lazy = fn(pa, pb)

        assert lazy._is_unbound()
        assert lazy.type == eager.type

        run(bench)
        np.testing.assert_allclose(lazy.y, eager.y, atol=ATOL)
        assert lazy.name == eager.name

    def test_chained_expression_resolves_leaf_by_leaf(self):
        """vcm = (vp + vm) / 2 — the design-doc example."""
        vp = make_bound('vp', y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        vm = make_bound('vm', y=np.array([-1.0, -2.0, -3.0, -4.0, -5.0]))
        bench = bench_with(vp, vm)
        pp, pm = probe(bench, 'vp'), probe(bench, 'vm')

        eager = (vp + vm) / 2
        vcm = (pp + pm) / 2
        assert vcm._is_unbound()

        run(bench)
        np.testing.assert_allclose(vcm.y, eager.y, atol=ATOL)
        assert vcm.type == eager.type == 'voltage'


class TestPower:

    @pytest.mark.parametrize('k', [2, 3, 0.5, -1])
    def test_unbound_signal_pow_scalar(self, k):
        vp = make_bound('vp', y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        bench = bench_with(vp)
        p = probe(bench, 'vp')

        eager = vp ** k
        lazy = p ** k
        assert lazy._is_unbound()
        assert lazy.type == eager.type == f'voltage^{k}'

        run(bench)
        np.testing.assert_allclose(lazy.y, eager.y, atol=ATOL)

    def test_signal_pow_signal_still_refused_when_unbound(self):
        bench = bench_with(make_bound('vp'), make_bound('vq'))
        p = probe(bench, 'vp')
        q = probe(bench, 'vq')
        with pytest.raises(TypeError, match='Signal \\*\\* Signal is not supported'):
            p ** q


class TestNegation:

    def test_unbound_signal_neg(self):
        vp = make_bound('vp', y=np.array([1.0, -2.0, 3.0, -4.0, 5.0]))
        bench = bench_with(vp)
        p = probe(bench, 'vp')

        eager = -vp
        lazy = -p
        assert lazy._is_unbound()
        assert lazy.type == eager.type

        run(bench)
        np.testing.assert_allclose(lazy.y, eager.y, atol=ATOL)

    def test_double_negation_through_a_chain(self):
        vp = make_bound('vp', y=np.array([1.0, -2.0, 3.0, -4.0, 5.0]))
        bench = bench_with(vp)
        p = probe(bench, 'vp')

        eager = -(-vp + 1)
        lazy = -(-p + 1)
        run(bench)
        np.testing.assert_allclose(lazy.y, eager.y, atol=ATOL)


# ---------------------------------------------------------------------------
# Declaration-time type refusal (design section 4)
# ---------------------------------------------------------------------------

class TestDeclarationTimeTypeCheck:

    def test_voltage_plus_current_raises_at_declaration(self):
        bench = bench_with(make_bound('vp', type_='voltage'),
                            make_bound('ivs', type_='current'))
        vp = probe(bench, 'vp', 'V')
        ivs = probe(bench, 'ivs', 'I')
        with pytest.raises(ValueError) as exc:
            vp + ivs
        assert "voltage" in str(exc.value) and "current" in str(exc.value)
        # never touched the (not-yet-run) bench to decide this
        assert bench.netlist.calls == []

    def test_voltage_minus_current_raises_at_declaration(self):
        bench = bench_with(make_bound('vp', type_='voltage'),
                            make_bound('ivs', type_='current'))
        vp = probe(bench, 'vp', 'V')
        ivs = probe(bench, 'ivs', 'I')
        with pytest.raises(ValueError, match='different types'):
            vp - ivs

    def test_message_matches_the_bound_check_exactly(self):
        """Same text as today's Signal._check_types, bound-Signal side."""
        a = make_bound('a', type_='voltage')
        b = make_bound('b', type_='current')
        try:
            a + b
            pytest.fail('bound check did not raise')
        except ValueError as bound_exc:
            bound_message = str(bound_exc)

        bench = bench_with(make_bound('vp', type_='voltage'),
                            make_bound('ivs', type_='current'))
        vp = probe(bench, 'vp', 'V')
        ivs = probe(bench, 'ivs', 'I')
        with pytest.raises(ValueError) as unbound_exc:
            vp + ivs
        assert str(unbound_exc.value) == bound_message

    def test_two_voltages_do_not_raise(self):
        bench = bench_with(make_bound('vp', type_='voltage'),
                            make_bound('vm', type_='voltage'))
        vp, vm = probe(bench, 'vp', 'V'), probe(bench, 'vm', 'V')
        result = vp + vm            # must not raise
        assert result._is_unbound()

    def test_two_currents_do_not_raise(self):
        bench = bench_with(make_bound('i1', type_='current'),
                            make_bound('i2', type_='current'))
        i1, i2 = probe(bench, 'i1', 'I'), probe(bench, 'i2', 'I')
        result = i1 - i2            # must not raise
        assert result._is_unbound()

    def test_reflected_scalar_forms_never_trip_the_type_check(self):
        bench = bench_with(make_bound('ivs', type_='current'))
        ivs = probe(bench, 'ivs', 'I')
        result = 5 + ivs             # scalar has no declared type: never refused
        assert result._is_unbound()

    def test_two_non_none_differing_types_raise_at_declaration(self):
        """The declaration-time check is the bound ``_check_types`` rule
        exactly (TASK-215.11): any two non-``None`` declared types that
        differ raise, not just the voltage/current pair. 'phase' and 'dB'
        are outside the old voltage/current vocabulary but are still two
        distinct, non-None declared strings, and a bound Signal carrying
        them already raises today (pinned in
        tests/test_signal_transforms_deferred.py's
        test_bound_phase_plus_voltage_raises_today-style check) — so the
        deferred equivalent must raise at declaration too, not defer."""
        bench = bench_with(make_bound('a', type_='phase'),
                            make_bound('b', type_='dB'))
        pa = probe(bench, 'a', 'V')
        pb = probe(bench, 'b', 'V')
        pa.type = 'phase'
        pb.type = 'dB'
        with pytest.raises(ValueError, match='different types'):
            pa + pb

    def test_declared_type_none_defers(self):
        """A future transform's data-dependent (``None``) declared type
        (TASK-215.4) must defer, not raise, regardless of the other
        operand's type; simulated here by hand-setting ``.type`` on a
        probe, since 215.3 has no transform decorator yet to produce
        one."""
        vp = make_bound('vp', type_='voltage', y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        vq = make_bound('vq', type_='voltage', y=np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        bench = bench_with(vp, vq)
        pp = probe(bench, 'vp', 'V')
        pq = probe(bench, 'vq', 'V')
        pp.type = None               # simulates a data-dependent transform result

        result = pp + pq             # must not raise: one side is undecidable
        assert result._is_unbound()

        run(bench)
        # Resolution runs the REAL bound check against the loaded data's
        # actual (non-None) types, so a genuinely compatible pair still
        # resolves correctly even though the declared type deferred.
        np.testing.assert_allclose(result.y, (vp + vq).y, atol=ATOL)


# ---------------------------------------------------------------------------
# Copying and pickling a recipe Signal; no lambda anywhere in it
# ---------------------------------------------------------------------------

class TestCopyingAndPicklingRecipes:

    def _expr(self):
        vp = make_bound('vp')
        vm = make_bound('vm')
        bench = bench_with(vp, vm)
        pp, pm = probe(bench, 'vp'), probe(bench, 'vm')
        return bench, (pp + pm) / 2 - 1

    def test_no_lambda_anywhere_in_the_recipe(self):
        _, expr = self._expr()
        ops = list(_walk_ops(expr))
        assert ops, 'expected at least one recorded op'
        for op in ops:
            assert op.__name__ != '<lambda>'
            assert op.__module__ in ('operator', '_operator'), (
                f'{op!r} is not a module-level operator builtin'
            )

    def test_deepcopy_of_a_recipe_signal_works(self):
        bench, expr = self._expr()
        clone = copy.deepcopy(expr)
        assert clone is not expr
        assert clone._is_unbound()

        run(bench)
        np.testing.assert_allclose(clone.y, expr.y, atol=ATOL)

    def test_deepcopy_shares_the_leaf_bench_weakrefs(self):
        """A deep-copied recipe's leaf probes still point at the ORIGINAL
        bench (weakrefs are shared atomically, per TASK-215.2); rebinding
        replicas to a copied bench is TASK-215.8's job, not this one."""
        bench, expr = self._expr()
        clone = copy.deepcopy(expr)

        def leaves(sig):
            recipe = sig.__dict__.get('_recipe')
            if recipe is None:
                yield sig
                return
            _, *operands = recipe
            for operand in operands:
                if isinstance(operand, Signal):
                    yield from leaves(operand)

        for leaf in leaves(clone):
            assert leaf.__dict__['_bench']() is bench

    def test_pickle_of_a_recipe_signal_is_detached(self):
        bench, expr = self._expr()
        run(bench)

        restored = pickle.loads(pickle.dumps(expr))
        assert restored is not expr
        assert restored._is_unbound()
        assert restored.bound is False           # detached: no bench survives a pickle

        from silk.exceptions import UnboundSignalError
        with pytest.raises(UnboundSignalError, match='detached'):
            restored.y

    def test_pickling_does_not_disturb_the_original(self):
        bench, expr = self._expr()
        run(bench)
        pickle.dumps(expr)
        np.testing.assert_allclose(expr.y, ((make_bound('vp') + make_bound('vm')) / 2 - 1).y, atol=ATOL)


def test_unknown_power_and_known_arithmetic_metadata():
    vp = make_bound('vp')
    bench = bench_with(vp)
    p = probe(bench, 'vp')
    assert (p.dbm() ** 2).type is None
    assert (p ** 2).type == 'voltage^2'
    assert (p * p).type == 'voltage*voltage'
    assert (p / p).type == 'voltage/voltage'
    for exponent in (p, 1j, [2]):
        with pytest.raises(TypeError, match='exponent must be a real number'):
            p.dbm() ** exponent


def test_bound_unknown_power_preserves_unknown_type():
    unknown = make_bound('u', type_=None, y=[2., 3., 4., 5., 6.])
    squared = unknown ** 2
    np.testing.assert_allclose(squared.y, [4., 9., 16., 25., 36.])
    assert squared.type is None


def test_mixed_deferred_unknown_power_preserves_evaluated_type():
    known = make_bound('vp', y=[1., 2., 3., 4., 5.])
    unknown = make_bound('u', type_=None, y=[2., 3., 4., 5., 6.])
    bench = bench_with(known)
    expr = (probe(bench, 'vp') * unknown) ** 2
    assert expr.type is None
    run(bench)
    np.testing.assert_allclose(expr.y, [4., 36., 144., 400., 900.])
    assert expr.type is None
    assert expr.resolve().type is None


# Each case was accepted by tolerant comparison or numpy broadcasting.
GRID_MISMATCHES = [
    ([0., 1e-9, 2e-9], [9e-9, 10e-9, 11e-9]),
    ([0., 1e-9, 2e-9], [0., 2e-9, 4e-9]),
    ([0., 1., 2.], [0., np.nextafter(1., 2.), 2.]),
    ([1e9, 2e9], [1e9 + 9000., 2e9 + 9000.]),
    ([0.], [0., 0., 0.]),
    ([0., 0., 0.], [0.]),
]


@pytest.mark.parametrize('operation', [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize('deferred', [False, True])
@pytest.mark.parametrize('left_x,right_x', GRID_MISMATCHES)
def test_arithmetic_refuses_inexact_grids(operation, deferred, left_x, right_x):
    left = make_bound('left', x=left_x, y=np.ones(len(left_x)))
    right = make_bound('right', x=right_x, y=np.ones(len(right_x)))
    if deferred:
        bench = bench_with(left, right)
        expr = operation(probe(bench, 'left'), probe(bench, 'right'))
        assert bench.netlist.calls == []
        run(bench)
    with pytest.raises(ValueError) as exc:
        if deferred:
            _ = expr.y
        else:
            operation(left, right)
    message = str(exc.value)
    assert 'X-values' in message
    assert 'left' in message and 'right' in message
    assert str(len(left_x)) in message and str(len(right_x)) in message
    assert 'silk.metrics.sample' in message
    assert len(message) < 700
    if len(left_x) == len(right_x):
        index = next(i for i, (a, b) in enumerate(zip(left_x, right_x)) if a != b)
        assert f'index {index}' in message
        assert repr(float(left_x[index])) in message and repr(float(right_x[index])) in message
    else:
        assert 'shape' in message and 'index' not in message


@pytest.mark.parametrize('operation,expected', [
    (operator.add, [6., 6., 6.]), (operator.sub, [2., 2., 2.]),
    (operator.mul, [8., 8., 8.]), (operator.truediv, [2., 2., 2.]),
])
@pytest.mark.parametrize('axis,label', [([2, 1, 0], 'time'),
                                        ([1, 2, 3], 'frequency'), ([1], 'time')])
@pytest.mark.parametrize('deferred', [False, True])
def test_arithmetic_accepts_exact_copied_grids(operation, expected, axis, label, deferred):
    left = make_bound('left', xlabel=label, x=np.array(axis, dtype=int),
                      y=np.full(len(axis), 4.))
    right = make_bound('right', xlabel=label, x=np.array(axis, dtype=float),
                       y=np.full(len(axis), 2.))
    if deferred:
        bench = bench_with(left, right)
        result = operation(probe(bench, 'left'), probe(bench, 'right'))
        assert bench.netlist.calls == []
        run(bench)
    else:
        result = operation(left, right)
    np.testing.assert_array_equal(result.y, expected[:len(axis)])
    np.testing.assert_array_equal(result.x, axis)
