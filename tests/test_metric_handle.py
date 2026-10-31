"""Metric arithmetic consumes reported values, including after composition."""
import copy
import math
import pickle
from unittest.mock import Mock

import numpy as np
import pytest

from silk import sources
from silk.circuit import Module
from silk.circuit.metric_handle import MetricHandle
from silk.metrics import average
from silk.signals import Signal
from silk.simulation import Transient
from tests._env import needs_ngspice


def constant(ckt):
    return 4.0


@pytest.fixture
def tb(pdk, tmp_path):
    dut = Module('PDK_skywater130', str(tmp_path), 'dut.cir',
                 pdk_details=pdk, module_name='dut')
    dut.define_module_ports({})
    bench = dut.create_test('measure')
    bench.add_module_under_test({})
    bench.add_sim_type(Transient(step=1e-9, stop=10e-9))
    return bench


def evaluate(tb):
    # Synthetic loaded evidence: exercise the production evaluator/spec tail.
    tb.netlist._raw = None
    tb.netlist.sim_results_loaded = True
    return tb.evaluate()


def test_arithmetic_reads_each_extractor_once_and_reports_specs(tb):
    up_fn, dn_fn = Mock(return_value=4.), Mock(return_value=6.)
    up = tb.add_metric('up', up_fn)
    dn = tb.add_metric('dn', dn_fn)
    mismatch = tb.add_metric('mismatch', 100 * abs(up - dn) / ((up + dn) / 2))
    twice = tb.add_metric('twice', mismatch * 2)
    tb.add_spec('matched', 'mismatch', '<', 50)
    with pytest.raises(RuntimeError, match='run.*first'):
        _ = mismatch.value
    report = evaluate(tb)
    assert report.metrics == {'up': 4., 'dn': 6., 'mismatch': 40., 'twice': 80.}
    assert report.passed
    assert mismatch.value == 40
    assert twice.value == 80
    up_fn.assert_called_once_with(tb.netlist)
    dn_fn.assert_called_once_with(tb.netlist)
    up_fn.return_value = 6.
    evaluate(tb)
    assert mismatch.value == 0
    assert up_fn.call_count == dn_fn.call_count == 2


@pytest.mark.parametrize('build,expected', [
    (lambda h: h + 2, 6), (lambda h: 2 + h, 6),
    (lambda h: h - 2, 2), (lambda h: 2 - h, -2),
    (lambda h: h * 2, 8), (lambda h: 2 * h, 8),
    (lambda h: h / 2, 2), (lambda h: 2 / h, .5),
    (lambda h: h ** 2, 16), (lambda h: 2 ** h, 16),
    (lambda h: abs(-h), 4),
])
def test_numeric_operators(tb, build, expected):
    h = tb.add_metric('base', constant)
    result = tb.add_metric('derived', build(h))
    assert evaluate(tb).metrics['derived'] == expected
    assert result.value == expected


def test_guardrails_and_builder_forwarding(tb):
    h = tb.add_metric('base', constant)
    assert isinstance(h, MetricHandle)
    assert h == h
    assert h != h + 0
    assert {h: 'key'}[h] == 'key'
    assert h in [h]
    with pytest.raises(TypeError, match='run.*first.*value'):
        bool(h)
    with pytest.raises(TypeError):
        _ = h < 3
    with pytest.raises(TypeError):
        _ = h + '3'
    assert h.add_source(sources.VoltageSource('v', positive='x', negative='0', dc=1)) is tb
    assert h.add_sim_type(Transient(step=1e-9, stop=20e-9)) is tb
    assert h.add_custom_line('* forwarded') is tb
    second = h.add_metric('second', h + 1)
    spec = second.add_spec('second high', 'second', '>', 4)
    assert spec.name == 'second high'
    assert evaluate(tb).passed


def test_cross_bench_and_extra_arguments_refused(tb):
    other = copy.deepcopy(tb)
    h = tb.add_metric('base', constant)
    foreign = other.add_metric('foreign', constant)
    with pytest.raises(ValueError, match='same testbench'):
        _ = h + foreign
    with pytest.raises(ValueError, match='this testbench'):
        tb.add_metric('bad', foreign)
    with pytest.raises(TypeError, match='no signal'):
        tb.add_metric('bad', h + 1, scale=2)


@pytest.mark.parametrize('build', [lambda h: h + 1, lambda h: h ** 0, lambda h: 0 ** h])
def test_nan_propagates_and_fails_spec(tb, build):
    h = tb.add_metric('unknown', Mock(return_value=math.nan))
    derived = tb.add_metric('derived', build(h))
    tb.add_spec('finite', 'derived', '<', 100)
    report = evaluate(tb)
    assert math.isnan(report.metrics['derived'])
    assert math.isnan(derived.value)
    assert not report.passed


def test_zero_denominator_returns_nan_and_fails_spec(tb):
    num = tb.add_metric('num', Mock(return_value=1.0))
    den = tb.add_metric('den', Mock(return_value=0.0))
    ratio = tb.add_metric('ratio', num / den)
    tb.add_spec('ratio small', 'ratio', '<', 100)
    report = evaluate(tb)
    assert math.isnan(report.metrics['ratio'])
    assert math.isnan(ratio.value)
    assert not report.passed


def test_failed_input_does_not_remeasure(tb):
    fn = Mock(side_effect=ValueError('bad data'))
    h = tb.add_metric('bad', fn)
    result = tb.add_metric('derived', h * 2)
    report = evaluate(tb)
    assert 'input metric' in report.metrics['derived']
    with pytest.raises(ValueError, match='failed'):
        _ = result.value
    fn.assert_called_once()


def test_pickle_and_deepcopy_handle_and_bench(tb):
    h = tb.add_metric('base', average,
                      sig=Signal('s', 'V', 'time', 'V', np.array([0., 1.]), np.array([2., 6.])))
    derived = tb.add_metric('derived', h * 3)
    evaluate(tb)
    restored = pickle.loads(pickle.dumps(derived))
    assert restored.value == 12
    assert evaluate(restored._bench).metrics['derived'] == 12
    copied = copy.deepcopy(derived)
    with pytest.raises(RuntimeError, match='run.*first'):
        _ = copied.value
    assert evaluate(copied._bench).metrics['derived'] == 12
    assert derived.value == 12


def test_explicit_module_copy_reads_its_own_report(tb, pdk, tmp_path):
    h = tb.add_metric('base', average, sig=tb.probe('x'))
    tb.add_metric('derived', h / 2)
    parent = Module('PDK_skywater130', str(tmp_path), 'parent.cir', pdk_details=pdk)
    placed = parent.add_module(copy.deepcopy(tb.module), 'child', nets={})
    copied = placed.subcircuit._tests['measure']
    copied.netlist.load_signal = Mock(return_value=Signal(
        'x', 'V', 'time', 'V', np.array([0., 1.]), np.array([8., 12.])))
    assert evaluate(copied).metrics == {'base': 10., 'derived': 5.}
    with pytest.raises(RuntimeError, match='run.*first'):
        _ = h.value


def test_all_registration_forms_return_handles(tb):
    callable_handle = tb.add_metric('callable', constant)
    plain_handle = tb.add_metric('plain', _plain_mean,
                                 sig=Signal('s', 'V', 'time', 'V', [0., 1.], [2., 2.]))
    declarative_handle = tb.add_metric('declarative', type='average', signal='x')
    tb.netlist.load_signal = Mock(return_value=Signal('x', 'V', 'time', 'V', [0., 1.], [6., 6.]))
    total = tb.add_metric('total', callable_handle + plain_handle + declarative_handle)
    assert evaluate(tb).metrics['total'] == 12
    assert total.value == 12
    assert all(isinstance(h, MetricHandle) for h in (callable_handle, plain_handle, declarative_handle))


def _plain_mean(sig):
    return float(np.mean(sig.y))


def _corner_value(ckt):
    # The corner the run was requested at (its label). This deck has no PDK
    # device, so since TASK-406 its evidence corner is none, not the label.
    return {'tt': 2., 'ff': 4.}[ckt._last_sim_label]


@pytest.mark.slow
@needs_ngspice
def test_run_corner_values_and_cache_hits(tb):
    tb.add_source(sources.VoltageSource('v', positive='x', negative='0', dc=1))
    h = tb.add_metric('base', _corner_value)
    derived = tb.add_metric('derived', h * 3)
    tb._cache_enabled = True
    assert tb.run('tt').metrics['derived'] == derived.value == 6
    assert tb.run('ff').metrics['derived'] == derived.value == 12
    # A cached earlier corner restores the handle's value without evaluation.
    assert tb.run('tt').metrics['derived'] == derived.value == 6
    reports = tb.run_corners(['tt', 'ff'])
    assert reports['ff'].metrics['derived'] == derived.value == 12


def test_handle_name_and_add_spec_accepts_handle(tb):
    up = tb.add_metric('up', Mock(return_value=4.))
    dn = tb.add_metric('dn', Mock(return_value=6.))
    assert up.name == 'up'
    with pytest.raises(ValueError, match='add_metric'):
        (up + dn).name
    spec = tb.add_spec('by handle', up, '>', 1.0)
    tb.add_spec('by name', 'up', '>', 1.0)
    report = evaluate(tb)
    assert report.passed
    assert [r.value for r in report.results] == [4., 4.]
    assert report.metrics[up] == report.metrics['up'] == 4.
    assert up in report.metrics and report.metrics.get(dn) == 6.
    assert spec.metric == 'up'


def test_add_spec_rejects_foreign_handle_unregistered_and_other_types(tb):
    up = tb.add_metric('up', Mock(return_value=4.))
    other = tb.module.create_test('other')
    with pytest.raises(ValueError, match='different testbench'):
        other.add_spec('x', up, '>', 1.0)
    with pytest.raises(ValueError, match='add_metric'):
        tb.add_spec('x', up + 1, '>', 1.0)
    with pytest.raises(TypeError, match='MetricHandle'):
        tb.add_spec('x', 3, '>', 1.0)


@pytest.mark.parametrize('first', ['callable', 'decorated', 'plain', 'declarative', 'derived'])
@pytest.mark.parametrize('second', ['callable', 'decorated', 'plain', 'declarative', 'derived'])
def test_duplicate_metric_forms_refuse_before_binding(tb, first, second):
    seed = tb.add_metric('seed', constant)
    sig = tb.probe('x')
    calls = []

    def extract(ckt):
        calls.append(ckt)
        return 4.0

    recipes = {
        'callable': ((extract,), {}),
        'decorated': ((average,), {'sig': sig}),
        'plain': ((_plain_mean,), {'sig': sig}),
        'declarative': ((), {'type': 'average', 'signal': 'x'}),
        'derived': ((seed * 2,), {}),
    }
    args, kwargs = recipes[first]
    tb.add_metric('gain', *args, **kwargs)
    declarations = list(tb._metrics)
    probes = list(tb._probe_registry)
    args, kwargs = recipes[second]
    with pytest.raises(ValueError, match="measure.*gain.*distinct name"):
        tb.add_metric('gain', *args, **kwargs)
    assert tb._metrics == declarations
    assert tb._probe_registry == probes
    assert calls == []
    # Duplicate refusal takes precedence over recipe validation/binding.
    with pytest.raises(ValueError, match='distinct name'):
        tb.add_metric('gain', type='not_a_metric', signal='unbound')


def test_metric_names_are_exact_and_local_to_authored_bench(tb):
    tb.add_metric('gain', constant)
    tb.add_metric('Gain', constant)
    tb.add_metric('gain ', constant)
    other = tb.module.create_test('other')
    other.add_module_under_test({})
    other.add_sim_type(Transient(step=1e-9, stop=10e-9))
    other.add_metric('gain', constant)
    assert evaluate(tb).metrics == {'gain': 4., 'Gain': 4., 'gain ': 4.}
    assert evaluate(other).metrics == {'gain': 4.}


def test_selected_metric_duplicate_uses_authored_declarations(tb, pdk, tmp_path):
    top = Module('PDK_skywater130', str(tmp_path), 'top.cir', pdk_details=pdk)
    top.add_module(tb.module, 'one', nets={})
    top.add_module(tb.module, 'two', nets={})
    one = top.get_test('measure', instance='one')
    two = top.get_test('measure', instance='two')
    one.add_metric('gain', constant)
    for bench in (tb, one, two):
        with pytest.raises(ValueError, match='measure.*gain.*distinct name'):
            bench.add_metric('gain', constant)
    assert [m['name'] for m in tb._metrics] == ['gain']


@pytest.mark.parametrize('num_type,den_type', [
    (float, float), (np.float64, np.float64),
    (float, np.float64), (np.float64, float),
    (np.float32, np.float16), (np.longdouble, np.longdouble),
    (int, np.int64),
])
@pytest.mark.parametrize('zero', [0.0, -0.0])
@pytest.mark.parametrize('mode', ['ignore', 'warn', 'raise'])
def test_scalar_zero_division_is_unavailable(tb, num_type, den_type, zero, mode):
    num_fn = Mock(return_value=num_type(1))
    den_fn = Mock(return_value=den_type(zero))
    num = tb.add_metric('num', num_fn)
    den = tb.add_metric('den', den_fn)
    expressions = [num / den, 1 / den, num / 0, num / -0.0,
                   num / (den + den), (num / den) * 2 + 1]
    for index, expression in enumerate(expressions):
        name = f'ratio_{index}'
        tb.add_metric(name, expression)
        for op in ('<', '<=', '>', '>=', '=='):
            tb.add_spec(f'{name} {op}', name, op, 10)
    with np.errstate(divide=mode, invalid=mode):
        report = evaluate(tb)
        for index, expression in enumerate(expressions):
            assert math.isnan(report.metrics[f'ratio_{index}'])
            assert math.isnan(expression.value)
    assert not report.passed
    assert all(not result.passed for result in report.results)
    num_fn.assert_called_once_with(tb.netlist)
    den_fn.assert_called_once_with(tb.netlist)


@pytest.mark.parametrize('num,den,expected', [
    (np.float64(6), np.float64(2), 3),
    (np.float32(6), 2., 3), (6., np.float32(2), 3),
    (np.float64(math.inf), np.float64(2), math.inf),
    (np.float64(-math.inf), 2., -math.inf),
    (1., np.float64(math.inf), 0),
    (np.float64(math.nan), 2., math.nan),
    (1., np.float64(math.nan), math.nan),
])
def test_scalar_division_retains_valid_results(tb, num, den, expected):
    numerator = tb.add_metric('num', Mock(return_value=num))
    denominator = tb.add_metric('den', Mock(return_value=den))
    ratio = tb.add_metric('ratio', numerator / denominator)
    with np.errstate(all='raise'):
        report = evaluate(tb)
        if math.isnan(expected):
            assert math.isnan(report.metrics['ratio'])
            assert math.isnan(ratio.value)
        else:
            assert report.metrics['ratio'] == ratio.value == expected


@pytest.mark.parametrize('zero', [0.0, np.float64(0)])
@pytest.mark.parametrize('mode', ['ignore', 'raise'])
def test_array_numerator_keeps_division_shape_and_error_mode(tb, zero, mode):
    numerator = tb.add_metric('num', Mock(return_value=np.array([1., -1.])))
    denominator = tb.add_metric('den', Mock(return_value=zero))
    expression = numerator / denominator
    ratio = tb.add_metric('ratio', expression)
    with np.errstate(divide=mode):
        report = evaluate(tb)
        if mode == 'raise':
            assert 'divide by zero' in report.metrics['ratio']
            with pytest.raises(FloatingPointError, match='divide by zero'):
                _ = expression.value
            with pytest.raises(ValueError, match='divide by zero'):
                _ = ratio.value
        else:
            for value in (report.metrics['ratio'], expression.value, ratio.value):
                assert isinstance(value, np.ndarray)
                assert value.shape == (2,)
                np.testing.assert_array_equal(value, [math.inf, -math.inf])
