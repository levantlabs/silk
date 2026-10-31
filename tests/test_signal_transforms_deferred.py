"""Deferred transforms on unbound Signals (TASK-215.4).

See ``docs/probes.md`` for declaring expressions before a run. Builds on the
unbound state and operator recipes (``silk/signals/base.py``).
Every Signal-to-Signal transform (``mag``, ``real``, ``imag``, ``phase_*``,
``shift``, ``window``, ``flip``, ``interpolate``, ``derivative``,
``integral``, ``fft``, ``dbm``, ``db10``, ``db20``, ``as_int``) is wrapped by
ONE decorator (``_deferrable``): called on an unbound receiver, it returns a
NEW unbound Signal recording ``(method_name, receiver, args, kwargs)``
instead of running the transform — the receiver **expression** is retained
(``slew = vcm.derivative()`` keeps ``vcm``), so ``resolve()`` resolves the
receiver first and then calls the method by name on it, and the transform
math is defined exactly once, in the bound implementation this file never
duplicates.

The fake bench here is a small, self-contained copy of
``tests/test_signal_unbound.py``'s: a probe only needs a bench exposing
``netlist.sim_results_loaded`` and ``netlist.load_signal(name, selector)``.
"""

import copy
import pickle
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from silk.signals.base import Signal, _DETACHED_BENCH

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


def test_wide_as_int_warns_only_when_implicitly_evaluated():
    from silk.metrics.diagnostics import MetricWarning
    words = [1 << 64, (1 << 53) + 1]
    x = np.array([0., 1.])
    bits = tuple(Signal(f'd[{i}]', 'voltage', 'time', 'V', x,
                        np.array([(word >> i) & 1 for word in words], dtype=float))
                 for i in range(65))
    bench = bench_with(Signal._make_bus_handle('d', bits))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', MetricWarning)
        decoded = probe(bench, 'd').as_int()
        assert decoded._is_unbound()
        assert not caught
        assert not bench.netlist.calls
        run(bench)
        np.testing.assert_array_equal(decoded.y, [float(word) for word in words])
    assert len(caught) == 1
    assert caught[0].message.metric == 'as_int'
    assert caught[0].message.code == 'binary_code_precision_loss'
    assert caught[0].message.details['rounded_sample_count'] == 1


class FakeBench:
    def __init__(self, signals, name='tb'):
        self.test_name = name
        self.netlist = FakeNetlist(signals)


def bench_with(*bound_signals, name='tb'):
    return FakeBench({s.name: s for s in bound_signals}, name=name)


def probe(bench, name, selector='V'):
    return Signal._unbound(name, selector, bench)


def run(bench):
    bench.netlist.sim_results_loaded = True


def make_bound(name, type_='voltage', xlabel='time', x=None, y=None):
    if x is None:
        x = np.linspace(0.0, 1e-6, 6)
    if y is None:
        y = np.arange(1.0, 7.0)
    return Signal(name, type_, xlabel, name, np.asarray(x), np.asarray(y))


#: Complex Y-data exercised alongside real data for every transform that
#: accepts it (everything except ``interpolate``, TASK-114's complex
#: refusal, and ``db10``, which needs a positive-real power signal).
COMPLEX_Y = np.array([1 + 2j, -0.5 + 3j, 2 - 1j, 0.1 + 0.1j, -3 + 0j, 1 - 4j])

REAL_Y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

#: (method, args, kwargs) tried against both REAL_Y and COMPLEX_Y.
TRANSFORM_CALLS = [
    ('mag', (), {}),
    ('real', (), {}),
    ('imag', (), {}),
    ('phase_rad', (), {}),
    ('phase_deg', (), {}),
    ('phase_rad_unwrapped', (), {}),
    ('phase_deg_unwrapped', (), {}),
    ('db20', (), {}),
    ('shift', (2e-7,), {}),
    ('window', (1e-7, 5e-7), {}),
    ('flip', (), {}),
    ('derivative', (), {}),
    ('integral', (), {}),
    ('dbm', (), {}),
]


# ---------------------------------------------------------------------------
# Every transform on an unbound Signal returns unbound; resolved equals the
# eager call on the bound Signal, to 1e-12.
# ---------------------------------------------------------------------------

class TestTransformsResolveToEager:

    @pytest.mark.parametrize('method,args,kwargs', TRANSFORM_CALLS)
    def test_real_signal(self, method, args, kwargs):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        deferred = getattr(p, method)(*args, **kwargs)
        assert deferred._is_unbound()

        run(bench)
        eager = getattr(vp, method)(*args, **kwargs)
        np.testing.assert_allclose(deferred.y, eager.y, atol=ATOL)
        np.testing.assert_allclose(deferred.x, eager.x, atol=ATOL)

    @pytest.mark.parametrize('method,args,kwargs', TRANSFORM_CALLS)
    def test_complex_signal(self, method, args, kwargs):
        vp = make_bound('vp', y=COMPLEX_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        deferred = getattr(p, method)(*args, **kwargs)
        assert deferred._is_unbound()

        run(bench)
        eager = getattr(vp, method)(*args, **kwargs)
        np.testing.assert_allclose(deferred.y, eager.y, atol=ATOL)

    def test_db10(self):
        # db10 wants a positive-real power-like signal (log of a negative
        # or complex value is a bound-implementation concern, not this
        # task's).
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        deferred = p.db10()
        assert deferred._is_unbound()

        run(bench)
        eager = vp.db10()
        np.testing.assert_allclose(deferred.y, eager.y, atol=ATOL)

    def test_interpolate(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        x_new = np.linspace(0.0, 1e-6, 11)

        deferred = p.interpolate(x_new)
        assert deferred._is_unbound()

        run(bench)
        eager = vp.interpolate(x_new)
        np.testing.assert_allclose(deferred.y, eager.y, atol=ATOL)

    def test_fft(self):
        n = 32
        t = np.linspace(0.0, (n - 1) * 1e-9, n)
        y = np.sin(2 * np.pi * 5e7 * t)
        vp = make_bound('vp', x=t, y=y)
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        deferred = p.fft(tclk=1e-9, tstart=0.0, n_samples=n)
        assert deferred._is_unbound()

        run(bench)
        eager = vp.fft(tclk=1e-9, tstart=0.0, n_samples=n)
        np.testing.assert_allclose(deferred.y, eager.y, atol=ATOL)
        np.testing.assert_allclose(deferred.x, eager.x, atol=ATOL)

    def test_as_int(self):
        bit_x = np.linspace(0.0, 1e-6, 5)
        bit0 = Signal('bit0', 'voltage', 'time', 'bit0', bit_x,
                       np.array([0.0, 1.0, 1.0, 0.0, 1.0]))
        bit1 = Signal('bit1', 'voltage', 'time', 'bit1', bit_x,
                       np.array([1.0, 1.0, 0.0, 0.0, 1.0]))
        bus = Signal._make_bus_handle('bus', (bit0, bit1))
        bench = bench_with(bus)
        p = probe(bench, 'bus', 'V')

        deferred = p.as_int()
        assert deferred._is_unbound()
        assert deferred.type == 'count'

        run(bench)
        eager = bus.as_int()
        np.testing.assert_allclose(deferred.y, eager.y, atol=ATOL)


# ---------------------------------------------------------------------------
# Declared type after a transform (design section 4).
# ---------------------------------------------------------------------------

class TestDeclaredType:

    @pytest.mark.parametrize('method', [
        'phase_rad', 'phase_deg', 'phase_rad_unwrapped', 'phase_deg_unwrapped',
    ])
    def test_phase_methods_declare_phase(self, method):
        vp = make_bound('vp', y=COMPLEX_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        assert getattr(p, method)().type == 'phase'

    def test_db20_declares_dB(self):
        vp = make_bound('vp', y=COMPLEX_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        assert p.db20().type == 'dB'

    def test_db10_declares_dB(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        assert p.db10().type == 'dB'

    @pytest.mark.parametrize('method,args,kwargs', [
        ('mag', (), {}),
        ('real', (), {}),
        ('imag', (), {}),
        ('shift', (1e-7,), {}),
        ('window', (0.0, 1.0), {}),
        ('flip', (), {}),
        ('interpolate', (np.linspace(0.0, 1e-6, 3),), {}),
        ('derivative', (), {}),
        ('integral', (), {}),
    ])
    def test_type_preserving_methods_keep_receiver_type(self, method, args, kwargs):
        # The probe's declared type comes from the selector ('I' ->
        # 'current'), not from introspecting the eventual bound data, so
        # the fake bench's bound signal is typed 'current' too, to match.
        vp = make_bound('vp', type_='current', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'I')
        assert getattr(p, method)(*args, **kwargs).type == 'current'

    def test_fft_declares_none(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        assert p.fft(tclk=1e-9, tstart=0.0, n_samples=4).type is None

    def test_dbm_declares_none(self):
        vp = make_bound('vp', y=COMPLEX_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        assert p.dbm().type is None

    def test_as_int_declares_count(self):
        bit_x = np.linspace(0.0, 1e-6, 5)
        bit0 = Signal('bit0', 'voltage', 'time', 'bit0', bit_x,
                       np.array([0.0, 1.0, 1.0, 0.0, 1.0]))
        bus = Signal._make_bus_handle('bus', (bit0,))
        bench = bench_with(bus)
        p = probe(bench, 'bus', 'V')
        assert p.as_int().type == 'count'


# ---------------------------------------------------------------------------
# The recipe retains the RECEIVER EXPRESSION (not a snapshot).
# ---------------------------------------------------------------------------

class TestReceiverRetention:

    def test_slew_records_vcm_as_receiver_and_resolves_through_it(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        vm = make_bound('vm', y=np.zeros(6))
        bench = bench_with(vp, vm)
        p_vp = probe(bench, 'vp', 'V')
        p_vm = probe(bench, 'vm', 'V')

        vcm = (p_vp + p_vm) / 2
        slew = vcm.derivative()

        assert slew._recipe[0] == 'derivative'
        assert slew._recipe[1] is vcm      # the expression, retained exactly
        assert slew._recipe[2] == ()
        assert slew._recipe[3] == {}

        run(bench)
        eager_vcm = (vp + vm) / 2
        eager_slew = eager_vcm.derivative()
        np.testing.assert_allclose(slew.y, eager_slew.y, atol=ATOL)

    def test_vcm_window_matches_eager(self):
        """add_metric('vcm_ac', peak_to_peak, sig=vcm.window(1e-6, 5e-6))
        shape from the design doc's TARGET snippet (AC-8)."""
        vp = make_bound('vp', y=REAL_Y.copy())
        vm = make_bound('vm', y=np.zeros(6))
        bench = bench_with(vp, vm)
        p_vp = probe(bench, 'vp', 'V')
        p_vm = probe(bench, 'vm', 'V')

        vcm = (p_vp + p_vm) / 2
        windowed = vcm.window(1e-7, 5e-7)

        run(bench)
        eager_vcm = (vp + vm) / 2
        eager_windowed = eager_vcm.window(1e-7, 5e-7)
        np.testing.assert_allclose(windowed.y, eager_windowed.y, atol=ATOL)
        np.testing.assert_allclose(windowed.x, eager_windowed.x, atol=ATOL)


# ---------------------------------------------------------------------------
# Argument validation moves from the call line to resolution.
# ---------------------------------------------------------------------------

class TestArgumentValidationDeferred:

    def test_window_bad_range_does_not_raise_at_declaration(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        deferred = p.window(5.0, 1.0)      # x2 <= x1: must not raise here
        assert deferred._is_unbound()

        run(bench)
        with pytest.raises(ValueError, match='require x2 > x1'):
            deferred.y

    def test_interpolate_complex_does_not_raise_at_declaration(self):
        vp = make_bound('vp', y=COMPLEX_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        deferred = p.interpolate(np.linspace(0.0, 1e-6, 3))  # must not raise
        assert deferred._is_unbound()

        run(bench)
        with pytest.raises(ValueError, match='complex'):
            deferred.y


# ---------------------------------------------------------------------------
# vp.phase_deg() + vp: refused AT DECLARATION (TASK-215.11), matching
# today's bound behaviour immediately rather than only at resolution — the
# declaration-time check is the bound _check_types rule exactly (any two
# non-None declared types that differ), not a voltage/current vocabulary.
# ---------------------------------------------------------------------------

class TestPhaseVoltageRefusal:

    def test_bound_phase_plus_voltage_raises_today(self):
        """Pin the PRE-EXISTING (pre-TASK-215) bound-bound behaviour: a
        BOUND phase Signal added to a BOUND voltage Signal already raises
        via _check_types, entirely independent of anything this task
        adds. This is the baseline the deferred case below is compared
        against."""
        vp = make_bound('vp', y=COMPLEX_Y.copy())
        with pytest.raises(ValueError, match='different types'):
            vp.phase_deg() + vp

    def test_deferred_phase_plus_voltage_raises_at_declaration(self):
        """The same combination, built from unbound probes: the line
        itself now raises (TASK-215.11). The declaration-time check
        (``Signal._check_declared_types``) is the bound ``_check_types``
        rule exactly — any two non-``None`` declared types that differ —
        so 'phase' vs 'voltage' is refused the moment the expression is
        written, with the identical message the pinned bound-bound check
        above raises, not deferred to resolution."""
        vp = make_bound('vp', y=COMPLEX_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        with pytest.raises(ValueError, match='different types'):
            p.phase_deg() + p

    def test_deferred_voltage_plus_dB_raises_at_declaration(self):
        """(vp + vp.db20()): 'voltage' vs 'dB', two non-None declared
        types that differ — raises at declaration with the exact
        _check_types text, same rule as the phase/voltage case above."""
        vp = make_bound('vp', y=COMPLEX_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        with pytest.raises(ValueError, match='different types'):
            p + p.db20()

    def test_deferred_phase_plus_phase_accepted_and_resolves(self):
        """(vp.phase_deg() + i.phase_deg()): both declare 'phase' —
        accepted at declaration (types are non-None but equal, so the
        exact-match rule does not fire) and resolves to the eager value,
        confirming resolution behaviour is unchanged by TASK-215.11."""
        vp = make_bound('vp', type_='voltage', y=COMPLEX_Y.copy())
        iv = make_bound('i', type_='current', y=COMPLEX_Y.copy() * 0.5)
        bench = bench_with(vp, iv)
        p_vp = probe(bench, 'vp', 'V')
        p_i = probe(bench, 'i', 'I')

        combo = p_vp.phase_deg() + p_i.phase_deg()
        assert combo._is_unbound()

        run(bench)
        eager = vp.phase_deg() + iv.phase_deg()
        np.testing.assert_allclose(combo.y, eager.y, atol=ATOL)

    def test_deferred_none_typed_operand_defers_at_declaration(self):
        """A ``None``-typed deferred operand (``vp.fft(...)``, whose
        TASK-215.4 declared type is ``None`` regardless of the receiver)
        added to a typed operand must not raise at declaration: one side
        is undecided, so the check defers to resolution exactly as it
        does for a hand-set ``None`` on an unbound probe
        (test_signal_recipes.py's own None-defers test)."""
        n = 8
        t = np.linspace(0.0, (n - 1) * 1e-9, n)
        y = np.sin(2 * np.pi * 5e7 * t)
        vp = make_bound('vp', x=t, y=y)
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')

        spectrum = p.fft(tclk=1e-9, tstart=0.0, n_samples=n)
        assert spectrum.type is None

        expr = spectrum + p          # must NOT raise: 'None' defers
        assert expr._is_unbound()


# ---------------------------------------------------------------------------
# plot() and other multi-attribute readers resolve exactly once.
# ---------------------------------------------------------------------------

class TestPlotResolvesOnce:

    def test_plot_unbound_leaf_resolves_once(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        run(bench)

        import matplotlib.pyplot as plt
        ax = plt.gca()
        p.plot(ax=ax)
        assert bench.netlist.calls == [('vp', 'V')]
        plt.close('all')

    def test_plot_deferred_transform_resolves_once(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        run(bench)

        import matplotlib.pyplot as plt
        ax = plt.gca()
        p.derivative().plot(ax=ax)
        assert bench.netlist.calls == [('vp', 'V')]
        plt.close('all')

    def test_plot_deferred_operator_recipe_resolves_each_leaf_once(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        vm = make_bound('vm', y=np.zeros(6))
        bench = bench_with(vp, vm)
        p_vp = probe(bench, 'vp', 'V')
        p_vm = probe(bench, 'vm', 'V')
        run(bench)

        import matplotlib.pyplot as plt
        ax = plt.gca()
        vcm = (p_vp + p_vm) / 2
        vcm.plot(ax=ax)
        assert sorted(bench.netlist.calls) == [('vm', 'V'), ('vp', 'V')]
        plt.close('all')

    def test_plot_bound_signal_unchanged(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        import matplotlib.pyplot as plt
        ax = plt.gca()
        result_ax = vp.plot(ax=ax)
        assert result_ax is ax
        plt.close('all')


# ---------------------------------------------------------------------------
# Copying and pickling a transform recipe.
# ---------------------------------------------------------------------------

class TestCopyingAndPickling:

    def test_deepcopy_transform_recipe(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        d = p.derivative()

        dc = copy.deepcopy(d)
        assert dc is not d
        assert dc._recipe[0] == 'derivative'
        # The receiver is deep-copied (its own object)...
        assert dc._recipe[1] is not p
        # ...but shares the ORIGINAL weak bench reference (copy semantics,
        # design section 3.0 — TASK-215.8 rebinds bench replicas, not this
        # task).
        assert dc._recipe[1].__dict__['_bench'] is p.__dict__['_bench']

        run(bench)
        np.testing.assert_allclose(dc.y, d.y, atol=ATOL)

    def test_deepcopy_transform_recipe_no_lambda(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        vm = make_bound('vm', y=np.zeros(6))
        bench = bench_with(vp, vm)
        p_vp = probe(bench, 'vp', 'V')
        p_vm = probe(bench, 'vm', 'V')
        slew = ((p_vp + p_vm) / 2).derivative()

        dc = copy.deepcopy(slew)

        def walk(sig, ops):
            recipe = sig.__dict__.get('_recipe')
            if recipe is None:
                return
            head = recipe[0]
            ops.append(head)
            if isinstance(head, str):
                _, receiver, args, kwargs = recipe
                nested = [receiver, *args, *kwargs.values()]
            else:
                nested = list(recipe[1:])
            for item in nested:
                if isinstance(item, Signal):
                    walk(item, ops)

        ops = []
        walk(dc, ops)
        assert ops, 'expected at least one recorded op'
        for op in ops:
            if isinstance(op, str):
                continue
            assert getattr(op, '__module__', None) in ('_operator', 'operator'), op
            assert not callable(op) or op.__name__ != '<lambda>'

    def test_pickle_transform_recipe_detaches(self):
        vp = make_bound('vp', y=REAL_Y.copy())
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        d = p.derivative()

        restored = pickle.loads(pickle.dumps(d))
        assert restored.bound is False
        # The nested receiver (leaf probe) is detached too.
        assert restored._recipe[1].__dict__['_bench'] is _DETACHED_BENCH
        with pytest.raises(Exception, match='detached'):
            restored.y

        # The original is undisturbed.
        run(bench)
        np.testing.assert_allclose(d.y, vp.derivative().y, atol=ATOL)


class TestInterpolateDescending600:
    """TASK-600: deferred interpolate on a descending axis, with
    independently computed expected values (not eager-vs-deferred)."""

    def test_descending_deferred(self):
        vp = make_bound('vp', x=np.array([1.8, 0.9, 0.0]),
                        y=np.array([4.0, 2.0, 1.0]))
        bench = bench_with(vp)
        p = probe(bench, 'vp', 'V')
        targets = [0.9, 1.35, 0.45, -1.0, 2.0]

        deferred = p.interpolate(targets)
        assert deferred._is_unbound()

        run(bench)
        np.testing.assert_allclose(deferred.y, [2.0, 3.0, 1.5, 1.0, 4.0])
        np.testing.assert_allclose(vp.interpolate(targets).y,
                                   [2.0, 3.0, 1.5, 1.0, 4.0])


@pytest.mark.parametrize('operation', ['mul', 'truediv'])
@pytest.mark.parametrize('sides', ['left', 'right', 'both', 'mixed_left', 'mixed_right'])
@pytest.mark.parametrize('transform', ['fft', 'dbm'])
def test_unknown_transform_arithmetic(operation, sides, transform):
    import operator
    # Non-DC impulse bins have squared magnitude 1; DC is suppressed by fft.
    bound = make_bound('vp', x=np.arange(4.) * 1e-9, y=[1., 0., 0., 0.])
    if transform == 'dbm':
        bound = make_bound('vp', y=[1., 2., 3., 4., 5., 6.])
    bench = bench_with(bound)
    p = probe(bench, 'vp')
    unknown = p.fft(tclk=1e-9, n_samples=4).window(1., 5e8) if transform == 'fft' else p.dbm()
    values = np.ones(2) if transform == 'fft' else 10 * np.log10(np.arange(1., 7.) ** 2 / .05)
    # Reuse the stored FFT grid: rounded decimal bins can differ by one ULP.
    axis = (bound.fft(tclk=1e-9, n_samples=4).window(1., 5e8).x
            if transform == 'fft' else bound.x)
    known = Signal('known', 'voltage', 'frequency' if transform == 'fft' else 'time',
                   'V', axis, np.full(len(axis), 2.))
    bench.netlist._signals['known'] = known
    k = probe(bench, 'known')
    left, right = {
        'left': (unknown, k), 'right': (k, unknown), 'both': (unknown, unknown),
        'mixed_left': (unknown, known), 'mixed_right': (known, unknown),
    }[sides]
    expr = getattr(operator, operation)(left, right)
    assert expr.type is None
    assert bench.netlist.calls == []
    run(bench)
    a = values if sides in ('left', 'both', 'mixed_left') else 2.
    b = values if sides in ('right', 'both', 'mixed_right') else 2.
    expected = a * b if operation == 'mul' else a / b
    np.testing.assert_allclose(expr.y, expected)
    assert expr.type is None
    if transform == 'fft' and sides == 'both' and operation == 'truediv':
        assert expr.resolve().type == 'power/power'
    np.testing.assert_allclose(expr.x, axis)
    eager = getattr(operator, operation)(left.resolve(), right.resolve())
    assert expr.resolve().type == eager.type


def test_fft_chain_implicit_plot_and_fresh_results():
    import matplotlib.pyplot as plt
    bound = make_bound('vp', x=np.arange(4.) * 1e-9, y=[1., 0., 0., 0.])
    bench = bench_with(bound)
    s = probe(bench, 'vp').fft(tclk=1e-9, n_samples=4).window(1., 5e8)
    expr = (2 * s / 4) ** 2 * (8 / s)
    assert expr.type is None
    assert bench.netlist.calls == []
    run(bench)
    np.testing.assert_allclose(expr.y, [2., 2.])
    fig, ax = plt.subplots()
    try:
        expr.plot(ax=ax)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [2., 2.])
    finally:
        plt.close(fig)
    bench.netlist._signals['vp'] = make_bound('vp', x=bound.x, y=[2., 0., 0., 0.])
    np.testing.assert_allclose(expr.y, [8., 8.])


@pytest.mark.parametrize('mismatch', ['label', 'grid'])
def test_fft_arithmetic_axis_refusals_remain_deferred(mismatch):
    a = make_bound('a', x=np.arange(4.) * 1e-9, y=[1., 0., 0., 0.])
    b = make_bound('b', x=np.arange(4.) * 2e-9, y=[1., 0., 0., 0.])
    bench = bench_with(a, b)
    p, q = probe(bench, 'a'), probe(bench, 'b')
    s = p.fft(tclk=1e-9, n_samples=4)
    other = q if mismatch == 'label' else q.fft(tclk=2e-9, n_samples=4)
    expr = s / other
    run(bench)
    with pytest.raises(ValueError):
        expr.y


def test_transformed_ratio_registered_metric(tmp_path, monkeypatch):
    from tests.test_object_probes import block
    from silk.metrics import rms
    tb = block(tmp_path, 'dut').create_test('read')
    tb.add_module_under_test({'out': 'out'})
    bound = make_bound('vp', x=np.arange(4.) * 1e-9, y=[1., 0., 0., 0.])
    monkeypatch.setattr(tb.netlist, 'load_signal', lambda *a, **kw: bound)
    tb.netlist.add_net('vp')
    p = tb.probe('vp')
    spectrum = p.fft(tclk=1e-9, n_samples=4).window(1., 5e8)
    ratio = spectrum / spectrum
    tb.add_metric('ratio_rms', rms, sig=ratio)
    tb.netlist.sim_results_loaded = True
    assert tb._metrics[-1]['extractor'](tb.netlist) == pytest.approx(1.)


def test_fft_deferred_interpolates_requested_instants():
    from silk.metrics.diagnostics import MetricWarning
    x = np.arange(9.)
    bound = make_bound('vp', x=x, y=x*x)
    bench = bench_with(bound)
    deferred = probe(bench, 'vp').fft(tclk=1, tstart=.25, n_samples=8)
    assert deferred._is_unbound()
    run(bench)
    with pytest.warns(MetricWarning, match='interpolated_samples'):
        resolved = deferred.resolve()
    # Piecewise-linear values at quarter steps of the quadratic's stored grid.
    samples = np.array([.25, 1.75, 5.25, 10.75, 18.25, 27.75, 39.25, 52.75])
    expected = np.abs(np.fft.rfft(samples - samples.mean()))**2
    np.testing.assert_allclose(resolved.y, expected, atol=1e-12)
    np.testing.assert_array_equal(resolved.x, np.arange(5.)/8)


@pytest.mark.parametrize('keyword', [False, True])
def test_transform_bound_includes_signal_argument_without_running_it(keyword, monkeypatch):
    first = bench_with(make_bound('a'))
    second = bench_with(make_bound('b'))
    a, b = probe(first, 'a'), probe(second, 'b')
    derived = a.versus(x_signal=b) if keyword else a.versus(b)
    assert derived.bound is False
    run(first)
    assert derived.bound is False
    run(second)

    def refuse(*args, **kwargs):
        pytest.fail('bound evaluated a transform or loaded its inputs')

    monkeypatch.setattr(Signal, 'versus', refuse)
    monkeypatch.setattr(Signal, 'resolve', refuse)
    assert derived.bound is True
    assert first.netlist.calls == second.netlist.calls == []
    assert 'y' not in derived.__dict__
    second.netlist.sim_results_loaded = False
    assert derived.bound is False


def test_bound_does_not_prevalidate_transform_arguments():
    bench = bench_with(make_bound('a'))
    derived = probe(bench, 'a').window(1., 0.)
    run(bench)
    assert derived.bound is True
    with pytest.raises(ValueError):
        _ = derived.y
