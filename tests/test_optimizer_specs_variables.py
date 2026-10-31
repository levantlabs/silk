"""Optimizer limits follow declarations and variable values follow samples."""
from types import SimpleNamespace

import pytest

from silk import Variable
from silk.optimize import Optimizer, VariableSpace
from tests._env import needs_ngspice


def make_optimizer(specs=(), metrics=None):
    var = Variable('bias', value=2.0, bounds=(1.0, 3.0))
    test = SimpleNamespace(_specs=list(specs),
                           run=lambda **kw: {'metrics': dict(metrics or {})})
    return Optimizer(test, VariableSpace({'sample_bias': var})), var


def spec(op='>', target=2.0, name='gain limit', metric='gain'):
    return dict(name=name, metric=metric, op=op, target=target)


def test_variable_objective_constraint_and_training_use_sample():
    opt, var = make_optimizer(metrics={'sample_bias': -100})
    opt.add_objective(var).add_constraint(var, min=1.5)
    low = opt.evaluate({'sample_bias': 1.0})
    best = opt.evaluate({'sample_bias': 2.0})
    opt.evaluate({'sample_bias': 3.0})
    assert low['feasible'] is False
    assert opt.best is best
    assert [opt._record_params(r)['sample_bias'] for r in opt.history] == [1.0, 2.0, 3.0]
    assert opt.summary()['objectives'] == [
        {'name': 'sample_bias', 'goal': 'minimize', 'value_source': 'variable'}]
    assert opt.feasible({}, params={'sample_bias': 2.0})


def test_variable_nan_is_not_best():
    opt, var = make_optimizer()
    opt.add_objective(var)
    bad = opt.evaluate({'sample_bias': float('nan')})
    good = opt.evaluate({'sample_bias': 2.0})
    assert bad['status'] == 'success'
    assert opt.best is good


def test_foreign_variable_rejected():
    opt, var = make_optimizer()
    foreign = Variable('bias', value=2.0, bounds=(1.0, 3.0))
    for method in (opt.add_objective, opt.add_constraint):
        with pytest.raises(ValueError, match='VariableSpace'):
            method(foreign)


@pytest.mark.parametrize('op,value,expected', [
    ('>', 2, False), ('>=', 2, True), ('<', 2, False),
    ('<=', 2, True), ('==', 2, True), ('==', 3, False),
    ('>', float('nan'), False),
])
def test_specs_constrain_by_default(op, value, expected):
    opt, _ = make_optimizer([spec(op)], {'gain': value})
    assert opt.evaluate({'sample_bias': 2})['feasible'] is expected


def test_explicit_override_replaces_all_specs_on_metric():
    opt, _ = make_optimizer([spec('>', 2), spec('<', 2.5, 'upper')], {'gain': 3})
    opt.add_constraint('gain', min=1)
    record = opt.evaluate({'sample_bias': 2})
    assert record['feasible'] is True
    limits = record['constraints']
    assert len(limits) == 1
    assert limits[0]['source'] == 'explicit constraint'
    assert limits[0]['overrides'] == ['gain limit', 'upper']
    assert opt.summary()['constraints'] == limits


@pytest.mark.parametrize('op,target,amount,tightened', [
    ('>', 60, 3, 63), ('>=', 60, 3, 63),
    ('<', .005, .001, .004), ('<=', .005, .001, .004),
])
def test_margin_direction_and_provenance(op, target, amount, tightened):
    opt, _ = make_optimizer([spec(op, target)])
    assert opt.add_margin('gain limit', amount) is opt
    limit = opt.summary()['constraints'][0]
    assert limit['target'] == pytest.approx(tightened)
    assert limit['op'] == op
    assert limit['source'] == 'spec+margin'
    assert limit['spec_name'] == 'gain limit'
    assert limit['margin'] == amount
    assert opt.feasible({'gain': tightened}) is (op in ('>=', '<='))


@pytest.mark.parametrize('amount', [-1, float('nan'), float('inf')])
def test_invalid_margin_rejected(amount):
    opt, _ = make_optimizer([spec()])
    with pytest.raises(ValueError, match='nonnegative|finite'):
        opt.add_margin('gain limit', amount)


def test_unknown_and_equality_margin_errors():
    opt, _ = make_optimizer([spec('==')])
    with pytest.raises(ValueError, match='Unknown.*gain limit'):
        opt.add_margin('missing', 1)
    with pytest.raises(ValueError, match='=='):
        opt.add_margin('gain limit', 1)


def test_history_keeps_limit_snapshot():
    opt, _ = make_optimizer([spec()], {'gain': 3})
    old = opt.evaluate({'sample_bias': 2})
    opt.add_margin('gain limit', 2)
    new = opt.evaluate({'sample_bias': 2})
    assert old['constraints'][0]['source'] == 'spec'
    assert old['feasible'] is True
    assert new['feasible'] is False


def test_margin_override_order_and_multiple_specs():
    opt, _ = make_optimizer([spec('>', 1), spec('<', 3, 'upper')])
    opt.add_margin('gain limit', .5).add_margin('upper', .5)
    assert opt.feasible({'gain': 2})
    assert not opt.feasible({'gain': 2.6})
    opt.add_constraint('gain', max=4)
    opt.add_margin('upper', 2)
    assert opt.feasible({'gain': 3.5})
    assert opt.summary()['constraints'][0]['source'] == 'explicit constraint'


def test_missing_variable_constraint_does_not_read_live_value():
    opt, var = make_optimizer()
    opt.add_constraint(var, min=1)
    assert not opt.feasible({})


def test_repeated_margin_replaces_and_ambiguous_name_rejected():
    opt, _ = make_optimizer([spec()])
    opt.add_margin('gain limit', 1).add_margin('gain limit', 2)
    assert opt.summary()['constraints'][0]['target'] == 4
    opt.test._specs.append(spec('<', 10))
    with pytest.raises(ValueError, match='ambiguous'):
        opt.add_margin('gain limit', 1)


def test_variable_constraint_does_not_override_same_named_metric_spec():
    opt, var = make_optimizer([spec(metric='sample_bias')], {'sample_bias': 1})
    opt.add_constraint(var, min=1)
    assert opt.evaluate({'sample_bias': 2})['feasible'] is False
    assert len(opt.summary()['constraints']) == 2


@needs_ngspice
@pytest.mark.parametrize('mode', ['evaluate', 'sweep', 'prepared_batch'])
def test_real_testbench_specs_variables_and_margin(tmp_path, mode):
    from tests.test_optimizer_parallel import _build_variable_lowpass_tb

    _, tb, resistance = _build_variable_lowpass_tb(str(tmp_path), 'limits')
    opt = Optimizer(tb, VariableSpace([resistance]))
    opt.add_objective(resistance).add_constraint(resistance, min=500)
    # Tighten the fixture's deliberately permissive measured spec to > 0 V.
    opt.add_margin('settles', 1e9)
    if mode == 'sweep':
        records = opt.sweep(3, method='grid', n_workers=1)
    elif mode == 'prepared_batch':
        records = opt._evaluate_points(
            [{'r': r} for r in (200, 10100, 20000)], 'tt', resolved=1)
        opt.history.extend(records)
    else:
        records = [opt.evaluate({'r': r}) for r in (200, 10100, 20000)]
    assert all(r['status'] == 'success' for r in records)
    assert [r['feasible'] for r in records] == [False, True, True]
    assert opt.best is records[1]
    assert resistance.value == 20000
    for record in records:
        assert record['constraints'][0]['source'] == 'spec+margin'
        assert record['constraints'][0]['target'] == 0
    assert [r['applied_params']['r'] for r in records] == [200, 10100, 20000]


def test_variable_uses_applied_sample_after_resolution_snapping():
    var = Variable('bias', value=1, bounds=(1, 3), resolution=1)
    test = SimpleNamespace(run=lambda **kw: {'metrics': {'actual': var.value}})
    opt = Optimizer(test, VariableSpace([var])).add_objective(var)
    opt.add_constraint(var, max=1.8)
    record = opt.evaluate({'bias': 1.6})
    assert record['metrics']['actual'] == 2
    assert record['feasible'] is False
    assert record['params'] == {'bias': 1.6}
    assert record['applied_params'] == {'bias': 2}
    assert [r['applied_params']['bias'] for r in opt.successes] == [2]


def test_legacy_history_records_remain_readable():
    opt, var = make_optimizer()
    opt.add_objective(var)
    old = dict(id=0, params={'sample_bias': 2}, corner='tt',
               status='success', metrics={}, feasible=True)
    opt.history.append(old)
    assert opt.best is old
    assert opt._record_params(old) == {'sample_bias': 2}
    assert old.get('constraints') is None  # historical limits are unknown


def test_best_rechecks_explicit_limits_and_preserves_history(monkeypatch):
    from copy import deepcopy

    opt, var = make_optimizer()
    opt.test.run = lambda **kw: {'metrics': {'voltage': var.value * 2 - 2}}
    opt.add_objective('voltage').add_constraint('voltage', min=1)
    low = opt.evaluate({'sample_bias': 1.5})  # 1 V
    high = opt.evaluate({'sample_bias': 3})  # 4 V
    assert opt.best is low
    opt.add_constraint('voltage', min=3)
    assert opt.best is high
    middle = opt.evaluate({'sample_bias': 2.5})  # 3 V
    assert opt.best is middle
    opt.add_constraint('voltage', min=5)
    assert opt.best is None
    excluded = opt.evaluate({'sample_bias': 2})  # 2 V, historically infeasible
    opt.add_constraint('voltage', min=2)
    frozen = deepcopy(opt.history)
    monkeypatch.setattr(opt.test, 'run', lambda **kw: pytest.fail('selection ran test'))
    monkeypatch.setattr(opt.space, 'apply', lambda *a: pytest.fail('selection applied params'))
    assert opt.best is excluded
    assert excluded['feasible'] is False
    assert opt.summary()['best'] is excluded
    assert opt.summary()['feasible_count'] == 3
    assert opt.history == frozen
    assert opt.best is excluded


@pytest.mark.parametrize('op,expected', [
    ('>', False), ('>=', True), ('<', False), ('<=', True), ('==', True),
])
def test_best_uses_changed_specs_and_override(op, expected):
    opt, _ = make_optimizer(metrics={'gain': 2})
    opt.add_objective('gain')
    record = opt.evaluate({'sample_bias': 2})
    opt.test._specs.append(spec(op, 2))
    assert opt.best is (record if expected else None)
    opt.add_constraint('gain', min=2, max=2)
    assert opt.best is record


def test_best_follows_changed_margin_and_spec_target():
    opt, _ = make_optimizer([spec('>=', 2)], {'gain': 3})
    opt.add_objective('gain')
    record = opt.evaluate({'sample_bias': 2})
    opt.add_margin('gain limit', 2)
    assert opt.best is None
    opt.add_margin('gain limit', 1)
    assert opt.best is record
    opt.test._specs[0]['target'] = 3
    assert opt.best is None
    opt.add_constraint('gain', min=3)
    opt.add_margin('gain limit', 10)
    assert opt.best is record


@pytest.mark.parametrize('legacy', [False, True])
def test_best_variable_limits_use_recorded_alias_not_live_value(legacy):
    opt, var = make_optimizer()
    opt.add_objective(var).add_constraint(var, min=2)
    low = opt.evaluate({'sample_bias': 1})
    high = opt.evaluate({'sample_bias': 3})
    if legacy:
        for record in opt.history:
            del record['applied_params']
    else:
        # Requested values must not override the applied sample.
        low['params']['sample_bias'] = 3
        high['params']['sample_bias'] = 1
    var.value = 1
    opt.add_constraint(var, min=1, max=2)
    assert opt.best is low
    assert low['feasible'] is False
    opt.add_constraint(var, min=2)
    assert opt.best is high


@pytest.mark.parametrize('source', ['metric', 'variable'])
@pytest.mark.parametrize('sample', [{}, {'value': None}, {'value': 'unknown'},
                                    {'value': float('nan')}])
def test_best_rejects_missing_new_constraint_values(source, sample):
    opt, var = make_optimizer(metrics={'gain': 2})
    opt.add_objective('gain')
    record = opt.evaluate({'sample_bias': 2})
    assert opt.best is record
    if source == 'variable':
        record['applied_params'] = {'sample_bias': sample['value']} if sample else {}
        opt.add_constraint(var, min=1)
    else:
        record['metrics'].update(sample)
        opt.add_constraint('value', min=1)
    assert opt.best is None
    assert record['feasible'] is True


def test_best_resolves_current_constraints_once(monkeypatch):
    from unittest.mock import Mock

    opt, var = make_optimizer()
    opt.add_objective(var)
    records = [opt.evaluate({'sample_bias': v}) for v in (1, 2, 3)]
    resolve = Mock(wraps=opt._effective_constraints)
    monkeypatch.setattr(opt, '_effective_constraints', resolve)
    assert opt.best is records[0]
    resolve.assert_called_once_with()
    opt._objectives.clear()
    resolve.reset_mock()
    assert opt.best is None
    resolve.assert_not_called()


def test_best_new_complex_constraint_keeps_refusal():
    opt, _ = make_optimizer(metrics={'gain': 2, 'phase': 1+2j})
    opt.add_objective('gain')
    opt.evaluate({'sample_bias': 2})
    opt.add_constraint('phase', min=0)
    with pytest.raises(ValueError, match='complex'):
        _ = opt.best
