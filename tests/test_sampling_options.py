"""Seeded public sampling contract; baseline recorded before TASK-501 source edits."""
import copy
from contextlib import nullcontext
import json
import math
import pickle
import random
from pathlib import Path

import pytest
from silk import Variable, load_pdk
from silk.optimize import VariableSpace, Optimizer


@pytest.fixture(autouse=True)
def seeded_random():
    state = random.getstate()
    random.seed(7)
    yield
    random.setstate(state)


def make_variable(kind, name="x"):
    kwargs = {"bounds": (1, 10)}
    if kind == "log":
        kwargs["scale"] = "log"
    elif kind == "integer":
        kwargs["var_type"] = "integer"
    elif kind == "categorical":
        kwargs = {"var_type": "categorical", "categories": ["a", "b", "c"]}
    elif kind in ("resolution", "unenforced", "log_resolution"):
        kwargs["resolution"] = 2
        if kind == "unenforced":
            kwargs["enforce_resolution"] = False
        if kind == "log_resolution":
            kwargs["scale"] = "log"
    return Variable(name, **kwargs)


BASELINE = json.loads(Path(__file__).with_name("sampling_options_baseline.json").read_text())


# TASK-501 deterministic changes relative to its original fixture.
# TASK-505 seeded stochastic pins are updated in the JSON baseline.
CHANGED_PINS = {'variable/log/grid': [1, 2.154434690031884, 4.641588833612779, 10],
 'variable/resolution/halton': [5, 3, 7, 1],
 'variable/resolution/grid': [1, 3, 5, 9],
 'variable/log_resolution/halton': [3, 1, 5, 1],
 'variable/log_resolution/grid': [1, 3, 5, 9],
 'space/log/grid': [1, 2.154434690031884, 4.641588833612779, 10],
 'space/log_resolution/halton': [3, 1, 5, 1],
 # TASK-506 categorical deterministic methods.
 'variable/categorical/grid': ['a', 'b', 'c'],
 'variable/categorical/halton': ['b', 'a', 'c', 'a'],
 'space/categorical/halton': ['b', 'a', 'c', 'a']}


@pytest.mark.parametrize("key", BASELINE)
def test_seeded_sampling_pin(key):
    api, kind, method = key.split("/")
    var = make_variable(kind)
    if method == 'log_grid':
        with pytest.raises(ValueError, match="scale='log'"):
            var.sample(4, method)
        return
    if api == "variable":
        got = var.sample(4, method)
    else:
        got = [p["x"] for p in VariableSpace([var]).sample(4, method)]
    expected = CHANGED_PINS.get(key, BASELINE[key])
    assert len(got) == len(expected)
    if (kind in ("categorical", "integer")
            or (kind in ("resolution", "log_resolution") and api == "space")):
        assert got == expected
    else:
        assert got == pytest.approx(expected)
    if kind == "integer":
        assert all(type(v) is int for v in got)


@pytest.mark.parametrize("method", ["random", "lhs", "halton", "grid"])
def test_group_sampling_pin(method):
    x = Variable("x", bounds=(0, 1))
    y = Variable("y", bounds=(0, 1))
    pts = VariableSpace([x, y], groups=[[x, y]]).sample(4, method)
    assert len(pts) == 4
    if method == "grid":
        assert pts == [{"x": v, "y": v} for v in [0, 1 / 3, 2 / 3, 1]]
    if method == "lhs":
        assert [int(p["x"] * 4) for p in pts] == [2, 0, 1, 3]
        assert [int(p["y"] * 4) for p in pts] == [2, 0, 1, 3]
    if method == "halton":
        assert pts == [{"x": v, "y": v} for v in [.5, .25, .75, .125]]


def test_value_physical_helpers_and_identity():
    x = Variable("x", value=3, bounds=(1, 10), resolution=2)
    assert repr(x) == "Variable(name='x', value=3)"
    with pytest.warns(UserWarning):
        x.value = 4
    assert x.value == 5
    assert x.clip(42) == 9
    assert x.validate(4) and not x.validate(11)
    space = VariableSpace([x])
    space.apply({"x": 6})
    assert x.value == 5
    assert space.to_bounds() == [(1, 10)]
    assert space.to_list({"x": 7}) == [7]
    assert space.from_list([7]) == {"x": 7}
    assert space.clip({"x": 42}) == {"x": 9}
    assert space.validate({"x": 3})
    assert copy.copy(x) is x and copy.deepcopy(x) is x
    a, b = pickle.loads(pickle.dumps([x, x]))
    assert a is b and a is not x and a.value == 5
    with pytest.raises(ValueError):
        x.value = 11
    assert x.value == 5


@pytest.mark.parametrize("accessor", [True, False])
def test_pdk_factories(accessor):
    pdk = load_pdk("sky130")
    factory = pdk.nmos1p8.w.variable if accessor else lambda **kw: pdk.nmos1p8.variable("w", **kw)
    var = factory(name="w", value=1)
    assert var.bounds == (0.42, None)
    assert var.resolution == .005
    assert var.value == 1


@pytest.mark.parametrize("method", ["random", "lhs", "halton", "grid"])
def test_optimizer_sweep_sampling_pin(method):
    var = make_variable("linear")
    class Bench:
        def run(self, corner="tt"):
            return {"metrics": {"observed": var.value}}
    opt = Optimizer(Bench(), VariableSpace([var]))
    records = opt.sweep(4, method=method)
    want = BASELINE["space/linear/" + method]
    assert [r["params"]["x"] for r in records] == pytest.approx(want)
    assert [r["metrics"]["observed"] for r in records] == pytest.approx(want)


@pytest.mark.parametrize('field,valid', [('var_type', ['continuous', 'integer', 'categorical']), ('scale', ['linear', 'log'])])
@pytest.mark.parametrize('bad', ['CONTINUOUS', 'LOG', 'log_uniform', 'log-uniform', '', None, 1, []])
def test_options_reject_unknown_on_construction_and_assignment(field, valid, bad):
    with pytest.raises(ValueError) as exc:
        Variable('x', **{field: bad})
    assert all(choice in str(exc.value) for choice in valid)
    var = Variable('x')
    old = getattr(var, field)
    with pytest.raises(ValueError):
        setattr(var, field, bad)
    assert getattr(var, field) == old


@pytest.mark.parametrize('api', ['variable', 'space'])
@pytest.mark.parametrize('n', [0, 4])
@pytest.mark.parametrize('method', ['typo', 'GRID', 'log_grid', 'sobol'])
def test_method_validation_including_empty(api, n, method):
    var = Variable('x', bounds=(1, 10))
    target = var if api == 'variable' else VariableSpace([var])
    with pytest.raises(ValueError) as exc:
        target.sample(n, method)
    if method == 'log_grid':
        assert "scale='log'" in str(exc.value) and "method='grid'" in str(exc.value)


@pytest.mark.parametrize('bounds', [None, (None, 10), (1, None), (0, 10), (-1, 10), (10, 1), (1, math.inf), (math.nan, 10), (1, math.nan)])
@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
@pytest.mark.parametrize('resolution', [None, 2])
def test_log_requires_positive_finite_sampling_bounds(bounds, method, resolution):
    var = Variable('x', bounds=bounds, scale='log', resolution=resolution)
    for target in (var, VariableSpace([var])):
        with pytest.raises(ValueError, match='x'):
            target.sample(4, method)


def test_incompatible_log_options_and_assignment():
    with pytest.raises(ValueError):
        Variable('x', scale='log', var_type='categorical', categories=['a'])
    var = Variable('x')
    var.scale = 'log'
    with pytest.raises(ValueError):
        var.var_type = 'categorical'
    assert var.scale == 'log' and var.var_type == 'continuous'
    var.scale = 'linear'
    var.var_type = 'categorical'
    with pytest.raises(ValueError):
        var.scale = 'log'


@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
@pytest.mark.parametrize('api', ['variable', 'space'])
@pytest.mark.parametrize('kind', ['continuous', 'integer', 'resolution', 'unenforced'])
def test_log_coordinates_then_projection(api, method, kind):
    kwargs = {'bounds': (1, 1000), 'scale': 'log'}
    if kind == 'integer':
        kwargs['var_type'] = 'integer'
    if kind in ('resolution', 'unenforced'):
        kwargs.update(resolution=2, enforce_resolution=kind != 'unenforced')
    var = Variable('x', **kwargs)
    def draw(v):
        return v.sample(4, method) if api == 'variable' else [p['x'] for p in VariableSpace([v]).sample(4, method)]
    random.seed(7)
    raw = draw(Variable('x', bounds=(1, 1000), scale='log'))
    random.seed(7)
    got = draw(var)
    if kind == 'integer':
        assert got == [int(round(v)) for v in raw]
        assert all(type(v) is int for v in got)
    elif kind == 'resolution':
        assert got == [1 + 2 * min(499, max(0, round((v - 1) / 2))) for v in raw]
    else:
        assert got == pytest.approx(raw)
    if method == 'grid':
        assert raw == pytest.approx([1, 10, 100, 1000])
    if method == 'halton':
        assert raw == pytest.approx([1000**u for u in [.5, .25, .75, .125]])
    if method == 'lhs':
        assert sorted(int(math.log(v, 1000) * 4) for v in raw) == [0, 1, 2, 3]


@pytest.mark.parametrize('api', ['variable', 'space'])
def test_log_grid_cardinality_and_endpoints(api):
    var = Variable('x', bounds=(1, 10), scale='log', resolution=2)
    def draw(n):
        return var.sample(n, 'grid') if api == 'variable' else [p['x'] for p in VariableSpace([var]).sample(n, 'grid')]
    assert draw(4) == [1, 3, 5, 9]
    assert draw(1) == [1]
    assert draw(0) == []
    assert draw(8) == [1, 1, 1, 3, 3, 5, 7, 9]


@pytest.mark.parametrize('method', ['lhs', 'halton'])
def test_group_log_strata(method):
    x = Variable('x', bounds=(1, 1000), scale='log')
    y = Variable('y', bounds=(1, 1000), scale='log', resolution=1, enforce_resolution=False)
    pts = VariableSpace([x, y], groups=[[x, y]]).sample(4, method)
    if method == 'lhs':
        assert [int(math.log(p['x'], 1000) * 4) for p in pts] == [2, 0, 1, 3]
        assert [int(math.log(p['y'], 1000) * 4) for p in pts] == [2, 0, 1, 3]
    else:
        assert all(p['x'] == p['y'] for p in pts)


def test_removed_distribution_and_exports():
    import importlib
    with pytest.raises(TypeError, match='distribution'):
        Variable('x', distribution='normal')
    for path in ('silk', 'silk.circuit', 'silk.optimize.parameter_space'):
        mod = importlib.import_module(path)
        for name in ('VarType', 'Distribution'):
            assert not hasattr(mod, name)
            with pytest.raises(ImportError):
                exec(f'from {path} import {name}', {})


@pytest.mark.parametrize('accessor', [True, False])
def test_factory_scale_and_variable_accessor(accessor):
    pdk = load_pdk('sky130')
    factory = pdk.nmos1p8.w.variable if accessor else lambda **kw: pdk.nmos1p8.variable('w', **kw)
    var = factory(name='w', value=1, bounds=(1, 10), scale='log')
    class Bench:
        def run(self, corner='tt'):
            return {'metrics': {'w': var.value}}
    opt = Optimizer(Bench(), VariableSpace([var]))
    got = opt.space.variables()
    assert list(got) == ['w'] and got['w'] is var
    assert (var.bounds, var.var_type, var.scale, var.value) == ((1, 10), 'continuous', 'log', 1)
    assert [p['params']['w'] for p in opt.sweep(2, method='grid')] == [1, 10]
    opt.space.apply({'w': 3})
    assert var.value == 3 and opt.space.to_bounds() == [(1, 10)]
    assert opt.space.to_list({'w': 3}) == [3]
    assert opt.space.from_list([3]) == {'w': 3}


@pytest.mark.parametrize('n,expected', [(0, []), (1, [1]), (4, [1, 3, 5]), (8, [1, 3, 5])])
def test_linear_resolution_small_domain(n, expected):
    var = Variable('x', bounds=(1, 6.5), resolution=2)
    with pytest.warns(UserWarning) if n > 3 else nullcontext():
        assert var.sample(n, 'grid') == expected
    with pytest.warns(UserWarning) if n > 3 else nullcontext():
        assert [p['x'] for p in VariableSpace([var]).sample(n, 'grid')] == expected


@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
@pytest.mark.parametrize('resolution', [None, .5])
def test_singleton_log_domain(method, resolution):
    var = Variable('x', value=2, bounds=(2, 2), scale='log', resolution=resolution)
    assert var.sample(4, method) == [2] * 4
    assert VariableSpace([var]).sample(4, method) == [{'x': 2}] * 4


def test_clearing_value_and_reassigning_valid_options():
    var = Variable('x', value=2, bounds=(1, 10))
    var.value = None
    assert var.value is None
    var.var_type = 'integer'
    var.scale = 'log'
    assert var.sample(4, 'grid') == [1, 2, 5, 10]
    var.scale = 'linear'
    var.resolution = 2
    var.value = 3
    assert var.sample(4, 'grid') == [1, 3, 5, 9]


def test_categorical_apply_rejection_is_atomic_and_units_round_trip():
    x = Variable('x', value=2, bounds=(1, 10), scale='log')
    c = Variable('c', value='a', var_type='categorical', categories=['a', 'b'])
    space = VariableSpace([x, c])
    with pytest.raises(ValueError, match='categories'):
        space.apply({'x': 3, 'c': 'missing'})
    assert x.value == 2 and c.value == 'a'
    assert space.to_bounds() == [(1, 10), (0, 1)]
    point = {'x': 3, 'c': 'b'}
    assert space.to_list(point) == [3, 'b']
    assert space.from_list([3, 'b']) == point
    space.apply(point)
    assert x.value == 3 and c.value == 'b'


@pytest.mark.parametrize('scale', ['linear', 'log'])
@pytest.mark.parametrize('enforce_resolution', [True, False])
@pytest.mark.parametrize('resolution,bounds', [
    (1.5, (1, 10)), (2, (1.5, 10.5)),
    (float('nan'), (1, 10)), (float('inf'), (1, 10)),
    (2, (float('nan'), 10)), (2, (float('inf'), 10)),
])
def test_integer_grid_ruling_rejects_invalid_construction(scale, enforce_resolution,
                                                        resolution, bounds):
    with pytest.raises(ValueError) as exc:
        Variable('count', var_type='integer', scale=scale, resolution=resolution,
                 bounds=bounds, enforce_resolution=enforce_resolution)
    message = str(exc.value)
    assert 'count' in message and repr(resolution) in message and repr(bounds) in message
    assert 'integer resolution' in message and "var_type='continuous'" in message


@pytest.mark.parametrize('scale', ['linear', 'log'])
@pytest.mark.parametrize('field,value', [('resolution', 1.5), ('bounds', (1.5, 10))])
def test_integer_grid_ruling_assignment_is_atomic(scale, field, value):
    var = Variable('count', value=3, bounds=(1, 10), resolution=2,
                   var_type='integer', scale=scale)
    old = getattr(var, field)
    with pytest.raises(ValueError, match='count'):
        setattr(var, field, value)
    assert getattr(var, field) == old and var.value == 3


@pytest.mark.parametrize('kwargs', [dict(resolution=1.5, bounds=(1, 10)),
                                  dict(resolution=2, bounds=(1.5, 10))])
def test_integer_grid_ruling_changing_type_validates_existing_grid(kwargs):
    var = Variable('count', **kwargs)
    with pytest.raises(ValueError, match='count'):
        var.var_type = 'integer'
    assert var.var_type == 'continuous'
    var.resolution = None
    var.var_type = 'integer'
    assert var.var_type == 'integer'


@pytest.mark.parametrize('bounds', [None, (None, 10)])
@pytest.mark.parametrize('scale', ['linear', 'log'])
@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
def test_integer_grid_open_lower_bound_checked_at_sampling(bounds, scale, method):
    var = Variable('count', bounds=bounds, resolution=2, var_type='integer', scale=scale)
    for target in (var, VariableSpace([var])):
        with pytest.raises(ValueError) as exc:
            target.sample(4, method)
        message = str(exc.value)
        assert 'count' in message and 'resolution=2' in message
        assert repr(bounds) in message and "var_type='continuous'" in message
    var.bounds = (1, 10)
    assert all(type(v) is int for v in var.sample(4, method))


@pytest.mark.parametrize('scale', ['linear', 'log'])
@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
def test_integer_grid_accepts_integral_floats_and_returns_ints(scale, method):
    var = Variable('count', bounds=(1.0, 10.5), resolution=2.0,
                   var_type='integer', scale=scale)
    shortened = scale == 'linear' and method == 'grid'
    with pytest.warns(UserWarning) if shortened else nullcontext():
        draws = var.sample(8, method)
    with pytest.warns(UserWarning) if shortened else nullcontext():
        points = VariableSpace([var]).sample(8, method)
    assert all(type(v) is int and v in [1, 3, 5, 7, 9] for v in draws)
    assert all(type(p['count']) is int and p['count'] in [1, 3, 5, 7, 9] for p in points)
    var.resolution = 1.0
    var.bounds = (2.0, 6.0)
    assert var.sample(2, 'grid') == [2, 6]


def test_log_integer_resolution_projects_once_in_physical_units():
    # sqrt(5) is nearer grid point 3 than 1. Rounding it to 2 before grid
    # projection would create a tie and incorrectly select 1.
    var = Variable('count', bounds=(1, 5), var_type='integer', resolution=2, scale='log')
    assert var.sample(4, 'halton') == [3, 1, 3, 1]
    assert [p['count'] for p in VariableSpace([var]).sample(4, 'halton')] == [3, 1, 3, 1]


# TASK-505: frozen before replacing the random engine.
DETERMINISTIC_PINS = {'variable/linear/halton': [5.5, 3.25, 7.75, 2.125],
 'variable/linear/grid': [1.0, 4.0, 7.0, 10.0],
 'variable/log/halton': [3.1622776601683795,
                         1.778279410038923,
                         5.623413251903492,
                         1.333521432163324],
 'variable/log/grid': [1, 2.154434690031884, 4.641588833612779, 10],
 'variable/integer/halton': [6, 3, 8, 2],
 'variable/integer/grid': [1, 4, 7, 10],
 'variable/resolution/halton': [5, 3, 7, 1],
 'variable/resolution/grid': [1, 3, 5, 9],
 'variable/unenforced/halton': [5.5, 3.25, 7.75, 2.125],
 'variable/unenforced/grid': [1.0, 4.0, 7.0, 10.0],
 'variable/log_resolution/halton': [3, 1, 5, 1],
 'variable/log_resolution/grid': [1, 3, 5, 9],
 'space/linear/halton': [5.5, 3.25, 7.75, 2.125],
 'space/linear/grid': [1.0, 4.0, 7.0, 10.0],
 'space/log/halton': [3.1622776601683795,
                      1.778279410038923,
                      5.623413251903492,
                      1.333521432163324],
 'space/log/grid': [1, 2.154434690031884, 4.641588833612779, 10],
 'space/integer/halton': [6, 3, 8, 2],
 'space/integer/grid': [1, 4, 7, 10],
 'space/resolution/halton': [5, 3, 7, 1],
 'space/resolution/grid': [1, 3, 5, 9],
 'space/unenforced/halton': [5.5, 3.25, 7.75, 2.125],
 'space/unenforced/grid': [1.0, 4.0, 7.0, 10.0],
 'space/log_resolution/halton': [3, 1, 5, 1],
 'space/log_resolution/grid': [1, 3, 5, 9]}

@pytest.mark.parametrize("key,expected", DETERMINISTIC_PINS.items())
def test_numeric_halton_grid_pre_seed_contract(key, expected):
    api, kind, method = key.split("/")
    var = make_variable(kind)
    got = var.sample(4, method) if api == "variable" else [p["x"] for p in VariableSpace([var]).sample(4, method)]
    assert got == expected


def test_categorical_space_grid_pre_seed_contract():
    assert VariableSpace([make_variable("categorical")]).sample(4, "grid") == [{"x": "a"}, {"x": "b"}, {"x": "c"}]


@pytest.mark.parametrize('api', ['variable', 'space', 'sweep'])
@pytest.mark.parametrize('method', ['random', 'lhs'])
@pytest.mark.parametrize('kind', ['linear', 'log', 'integer', 'categorical', 'resolution', 'log_resolution'])
def test_seed_reproducibility(api, method, kind):
    var = make_variable(kind)
    space = VariableSpace([var])
    class Bench:
        def run(self, corner='tt'):
            return {'metrics': {'observed': var.value}}
    opt = Optimizer(Bench(), space)
    def draw(**kwargs):
        if api == 'variable':
            return var.sample(32, method, **kwargs)
        if api == 'space':
            return space.sample(32, method, **kwargs)
        return [r['params'] for r in opt.sweep(32, method, **kwargs)]
    assert draw() == draw(seed=0) == draw(seed=0)
    assert draw(seed=17) == draw(seed=17)
    assert draw(seed=17) != draw(seed=18)
    assert draw(seed=None) != draw(seed=None)
    if api == 'sweep':
        assert draw(seed=17) == space.sample(32, method, seed=17)


@pytest.mark.parametrize('api', ['variable', 'space', 'sweep'])
@pytest.mark.parametrize('bad', [True, False, 1.5, '7', [], object()])
@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
def test_seed_rejects_non_integer_even_for_empty_draw(api, bad, method):
    var = make_variable('linear')
    target = var if api == 'variable' else VariableSpace([var])
    if api == 'sweep':
        target = Optimizer(object(), target)
    sample = target.sweep if api == 'sweep' else target.sample
    with pytest.raises(TypeError, match='seed'):
        sample(0, method, seed=bad)


@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
def test_sampling_does_not_touch_global_random_states(method):
    import numpy as np
    py_before = random.getstate()
    np_before = pickle.dumps(np.random.get_state())
    x, c = make_variable('linear'), make_variable('categorical', 'c')
    x.sample(4, method, seed=71)
    c.sample(4, method, seed=None)
    VariableSpace([x, c]).sample(4, method, seed=71)
    assert random.getstate() == py_before
    assert pickle.dumps(np.random.get_state()) == np_before


@pytest.mark.parametrize('method', ['random', 'lhs'])
def test_shared_generator_groups_and_independent_columns(method, monkeypatch):
    import numpy as np
    original = np.random.default_rng
    seeds = []
    def tracked(seed):
        seeds.append(seed)
        return original(seed)
    monkeypatch.setattr(np.random, 'default_rng', tracked)
    x, y, z = [Variable(k, bounds=(0, 1)) for k in ('x', 'y', 'z')]
    space = VariableSpace([x, y, z, make_variable('categorical', 'c')], groups=[[x, y]])
    pts = space.sample(32, method, seed=23)
    assert seeds == [23]
    assert pts == space.sample(32, method, seed=23)
    assert pts != space.sample(32, method, seed=24)
    assert any(p['x'] != p['y'] for p in pts)
    if method == 'lhs':
        strata = lambda name: [int(p[name] * 32) for p in pts]
        assert strata('x') == strata('y')
        assert strata('x') != strata('z')
        assert sorted(strata('x')) == list(range(32))


@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
def test_categorical_sampling_preserves_python_objects(method):
    token = object()
    choices = [token, 1, 'one', ('tuple', 2)]
    var = Variable('c', var_type='categorical', categories=choices)
    for target in (var, VariableSpace([var])):
        got = target.sample(32, method, seed=3)
        vals = got if target is var else [p['c'] for p in got]
        assert all(any(v is c for c in choices) for v in vals)


def test_sampling_seed_is_keyword_only():
    import inspect
    for method in (Variable.sample, VariableSpace.sample, Optimizer.sweep):
        p = inspect.signature(method).parameters['seed']
        assert p.default == 0 and p.kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize('api', ['variable', 'space', 'sweep'])
def test_negative_seed_is_rejected(api):
    var = make_variable('linear')
    space = VariableSpace([var])
    sample = var.sample if api == 'variable' else space.sample
    if api == 'sweep':
        sample = Optimizer(object(), space).sweep
    with pytest.raises(ValueError, match='non-negative'):
        sample(4, seed=-1)


@pytest.mark.parametrize('api', ['variable', 'space'])
@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
def test_fresh_entropy_is_passed_to_one_generator(api, method, monkeypatch):
    import numpy as np
    original = np.random.default_rng
    seeds = []
    def tracked(seed):
        seeds.append(seed)
        return original(seed)
    monkeypatch.setattr(np.random, 'default_rng', tracked)
    var = make_variable('categorical')
    target = var if api == 'variable' else VariableSpace([var])
    target.sample(4, method, seed=None)
    assert seeds == [None]


@pytest.mark.parametrize('method', ['halton', 'grid'])
def test_numeric_deterministic_methods_ignore_seed(method):
    for kind in ('linear', 'log', 'integer', 'resolution', 'log_resolution'):
        var = make_variable(kind)
        for target in (var, VariableSpace([var])):
            assert target.sample(4, method, seed=7) == target.sample(4, method, seed=99)


@pytest.mark.parametrize('n', [0, 1, 2, 5, 8])
@pytest.mark.parametrize('api', ['variable', 'space'])
@pytest.mark.parametrize('seed', [0, 17, None])
def test_categorical_grid_enumerates_declared_order(api, n, seed):
    categories = ['e', 'c', 'a', 'd', 'b']
    var = Variable('c', var_type='categorical', categories=categories)
    if api == 'variable':
        got = var.sample(n, 'grid', seed=seed)
    else:
        got = [p['c'] for p in VariableSpace([var]).sample(n, 'grid', seed=seed)]
    assert got == (categories if n else [])


@pytest.mark.parametrize('n', [0, 1, 2, 5, 8])
@pytest.mark.parametrize('api', ['variable', 'space'])
@pytest.mark.parametrize('seed', [0, 17, None])
def test_categorical_halton_direct_mapping(api, n, seed):
    var = Variable('c', var_type='categorical', categories=list('abcde'))
    # Base-2 prefix maps to repeated d and misses e at n=5: no balancing.
    expected = list('cbdadbea')[:n]
    if api == 'variable':
        got = var.sample(n, 'halton', seed=seed)
    else:
        got = [p['c'] for p in VariableSpace([var]).sample(n, 'halton', seed=seed)]
    assert got == expected


def test_categorical_halton_uses_group_prime_and_matches_discrete_numeric():
    x = Variable('x', bounds=(0, 1))
    c = Variable('c', var_type='categorical', categories=list('abcde'))
    d = Variable('d', bounds=(0, 4), resolution=1)
    pts = VariableSpace([x, c, d], groups=[[x], [c, d]]).sample(6, 'halton')
    # Second group uses base 3: 1/3, 2/3, 1/9, 4/9, 7/9, 2/9.
    assert [p['c'] for p in pts] == list('bdacdb')
    assert [p['d'] for p in pts] == [1, 3, 0, 2, 3, 1]
    assert [p['x'] for p in pts] == [.5, .25, .75, .125, .625, .375]


@pytest.mark.parametrize('categories', [None, []])
def test_categorical_space_halton_requires_categories(categories):
    var = Variable('c', var_type='categorical', categories=categories)
    with pytest.raises(ValueError, match="Variable 'c' has no categories defined"):
        VariableSpace([var]).sample(1, 'halton')


@pytest.mark.parametrize('bounds,resolution,expected', [
    ((0, 0.9999999995), 1, [0]),
    ((0, 0.9999999995e-12), 1e-12, [0]),
    ((0.42, 0.435), .005, [.42, .425, .43, .435]),
    ((0.1, 0.30000000000000004), .1, [.1, .2, .30000000000000004]),
    ((0.123456, 0.123457), .01, [.123456]),
    ((0.123456, 0.143456), .01, [.123456, .1335, .143456]),
    ((1., 1. + 8 * 2**-52), 2**-52,
     [1. + k * 2**-52 for k in range(9)]),
])
def test_resolution_domain_endpoints_survive_apply(bounds, resolution, expected):
    import warnings
    var = Variable('x', bounds=bounds, resolution=resolution)
    space = VariableSpace([var])
    for target in (var, space):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            got = target.sample(20, 'grid')
        vals = got if target is var else [p['x'] for p in got]
        assert vals == expected
        for value in vals:
            assert var.validate(value)
            space.apply({'x': value})
            assert var.value == value
            assert var.validate(var.value)


@pytest.mark.parametrize('bounds,expected', [((.5, 4.5), [1, 2, 3, 4]),
                                            ((1.2, 4.8), [2, 3, 4])])
@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
@pytest.mark.parametrize('scale', ['linear', 'log'])
def test_fractional_integer_sampling_and_stored_values(bounds, expected, method, scale):
    import warnings
    var = Variable('m', bounds=bounds, var_type='integer', scale=scale)
    space = VariableSpace([var])
    for target in (var, space):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            got = target.sample(10, method)
        vals = got if target is var else [p['m'] for p in got]
        if method == 'grid' and scale == 'linear':
            assert vals == expected
            assert len(caught) == 1
        else:
            assert len(vals) == 10 and not caught
        for v in vals:
            assert type(v) is int and v in expected
            space.apply({'m': v})
            assert var.value == v


@pytest.mark.parametrize('bounds', [(.1, .9), (1.2, 1.8)])
def test_empty_integer_domain_rejected_atomically(bounds):
    with pytest.raises(ValueError, match='count.*no integer'):
        Variable('count', bounds=bounds, var_type='integer')
    var = Variable('count', bounds=(1, 4), var_type='integer')
    with pytest.raises(ValueError, match='count.*no integer'):
        var.bounds = bounds
    assert var.bounds == (1, 4)
    continuous = Variable('count', bounds=bounds)
    with pytest.raises(ValueError, match='count.*no integer'):
        continuous.var_type = 'integer'
    assert continuous.var_type == 'continuous'


@pytest.mark.parametrize('kwargs,domain', [
    (dict(bounds=(1e-12, 2e-12), resolution=.25e-12),
     [1e-12, 1.25e-12, 1.5e-12, 1.75e-12, 2e-12]),
    (dict(bounds=(1, 5), var_type='integer'), [1, 2, 3, 4, 5]),
    (dict(bounds=(1, 6.5), var_type='integer', resolution=2), [1, 3, 5]),
    (dict(bounds=(1, 5), var_type='integer', resolution=2,
          enforce_resolution=False), [1, 2, 3, 4, 5]),
    (dict(bounds=(2, 2), resolution=.5), [2]),
])
@pytest.mark.parametrize('n', [0, 1, 2, 3, 5, 10])
def test_linear_grid_counts_and_warning_contract(kwargs, domain, n):
    import warnings
    var = Variable('C', **kwargs)
    for target in (var, VariableSpace([var])):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            got = target.sample(n, 'grid')
        vals = got if target is var else [p['C'] for p in got]
        assert len(vals) == min(n, len(domain))
        assert len(set(vals)) == len(vals)
        assert all(v in domain for v in vals)
        if n:
            assert vals[0] == domain[0]
        if 1 < n < len(domain):
            assert vals[-1] == domain[-1]
        assert len(caught) == int(n > len(domain))
        if caught:
            msg = str(caught[0].message)
            assert caught[0].category is UserWarning
            assert all(s in msg for s in ['C', str(n), str(len(domain)), 'bounds'])


@pytest.mark.parametrize('api', ['variable', 'space', 'sweep'])
def test_shortened_grid_warning_callsite_and_filtering(api):
    import inspect
    import warnings
    var = Variable('C', bounds=(1e-12, 2e-12), resolution=.25e-12)
    space = VariableSpace([var])
    class Bench:
        def run(self, corner='tt'):
            return {'metrics': {'observed': var.value}}
    opt = Optimizer(Bench(), space)
    call = var.sample if api == 'variable' else space.sample
    if api == 'sweep':
        call = opt.sweep
    for _ in range(2):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            line = inspect.currentframe().f_lineno + 1
            call(10, 'grid')
        assert len(caught) == 1
        assert caught[0].filename == __file__ and caught[0].lineno == line
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        old_history = len(opt.history)
        with pytest.raises(UserWarning, match='C.*10.*5'):
            call(10, 'grid')
        assert len(opt.history) == old_history
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('ignore', UserWarning)
        call(10, 'grid')
    assert not caught


def test_grid_product_warns_once_per_shortened_variable():
    import warnings
    x = Variable('x', bounds=(1, 2), var_type='integer')
    y = Variable('y', bounds=(1, 3), resolution=1)
    c = Variable('c', var_type='categorical', categories=list('abcd'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pts = VariableSpace([x, y, c]).sample(10, 'grid')
    assert len(pts) == 24
    assert len(caught) == 2
    assert [p['c'] for p in pts[:4]] == list('abcd')


@pytest.mark.parametrize('kwargs,expected', [
    (dict(bounds=(1e-12, 2e-12), resolution=.25e-12),
     [1e-12, 1.25e-12, 1.5e-12, 1.75e-12, 2e-12]),
    (dict(bounds=(1.2, 4.8), var_type='integer'), [2, 3, 4]),
    (dict(bounds=(.123456, .143456), resolution=.01),
     [.123456, .1335, .143456]),
])
def test_optimizer_grid_does_not_spend_failure_budget(kwargs, expected):
    var = Variable('x', **kwargs)
    class Bench:
        def run(self, corner='tt'):
            assert var.validate(var.value)
            return {'metrics': {'stored': var.value}}
    opt = Optimizer(Bench(), VariableSpace([var]))
    with pytest.warns(UserWarning):
        records = opt.sweep(10, 'grid', max_failures=1)
    assert [r['params']['x'] for r in records] == expected
    assert [r['metrics']['stored'] for r in records] == expected
    assert all(r['status'] == 'success' for r in records)


def test_integer_endpoint_stays_integral_and_continuous_endpoint_exact():
    var = Variable('m', bounds=(1, math.nextafter(3., math.inf)),
                   resolution=1, var_type='integer')
    with pytest.warns(UserWarning):
        assert var.sample(10, 'grid') == [1, 2, 3]
    # The old arithmetic puts its last point one ULP above the upper bound.
    var = Variable('x', bounds=(.01, .9))
    for target in (var, VariableSpace([var])):
        got = target.sample(4, 'grid')
        vals = got if target is var else [p['x'] for p in got]
        assert vals[0] == .01 and vals[-1] == .9
        assert all(var.validate(v) for v in vals)


def test_removed_step_constructor_is_not_a_tombstone():
    import inspect
    assert 'step' not in inspect.signature(Variable).parameters
    with pytest.raises(TypeError, match="unexpected keyword argument 'step'"):
        Variable('C', bounds=(1e-12, 2e-12), step=.25e-12)


@pytest.mark.parametrize('api', ['variable', 'space'])
@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
@pytest.mark.parametrize('scale', ['linear', 'log'])
@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 10])
def test_integer_fractional_interval_counts_seeds_and_repeats(api, method, scale, n):
    import warnings
    var = Variable('m', bounds=(1.2, 4.8), var_type='integer', scale=scale)
    target = var if api == 'variable' else VariableSpace([var])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        got = target.sample(n, method, seed=17)
    vals = got if api == 'variable' else [p['m'] for p in got]
    assert all(type(v) is int and v in {2, 3, 4} for v in vals)
    if method == 'grid' and scale == 'linear':
        assert vals == {0: [], 1: [2], 2: [2, 4]}.get(n, [2, 3, 4])
        assert len(caught) == int(n > 3)
    else:
        assert len(vals) == n and not caught
        if n > 3:
            assert len(set(vals)) < n
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        assert got == target.sample(n, method, seed=17)


def test_terminal_resolution_index_normalizes_below_hi_drift():
    hi = math.nextafter(.3, math.inf)
    assert .3 < hi
    var = Variable('x', bounds=(0, hi), resolution=.3)
    for target in (var, VariableSpace([var])):
        got = target.sample(2, 'grid')
        vals = got if target is var else [p['x'] for p in got]
        assert vals == [0, hi]
        space = VariableSpace([var])
        for v in vals:
            space.apply({'x': v})
            assert var.value == v


def test_small_grid_request_does_not_allocate_entire_integer_domain():
    import tracemalloc
    var = Variable('m', bounds=(0, 100_000), var_type='integer')
    tracemalloc.start()
    try:
        assert var.sample(5, 'grid') == [0, 25_000, 50_000, 75_000, 100_000]
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


@pytest.mark.parametrize('method', ['grid', 'random', 'lhs', 'halton'])
@pytest.mark.parametrize('scale', ['linear', 'log'])
def test_fractional_integer_sweep_stores_valid_draws(method, scale):
    import warnings
    var = Variable('m', bounds=(1.2, 4.8), var_type='integer', scale=scale)
    class Bench:
        def run(self, corner='tt'):
            assert type(var.value) is int and var.value in {2, 3, 4}
            return {'metrics': {'stored': var.value}}
    opt = Optimizer(Bench(), VariableSpace([var]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        records = opt.sweep(10, method, max_failures=1, seed=17)
    assert len(records) == (3 if method == 'grid' and scale == 'linear' else 10)
    assert len(caught) == int(method == 'grid' and scale == 'linear')
    for r in records:
        assert r['status'] == 'success'
        assert r['metrics']['stored'] == r['params']['m']


@pytest.mark.parametrize('method', ['random', 'lhs', 'halton', 'grid'])
@pytest.mark.parametrize('scale', ['linear', 'log'])
@pytest.mark.parametrize('bounds,resolution,domain', [
    ((1., 1.9999999995), 1., [1.]),
    ((1e-12, 1.9999999995e-12), 1e-12, [1e-12]),
    ((.123456, .123457), .01, [.123456]),
])
def test_projected_samples_and_assignment_keep_near_bound_domain(method, scale, bounds,
                                                                 resolution, domain):
    import warnings
    var = Variable('x', bounds=bounds, resolution=resolution, scale=scale)
    space = VariableSpace([var])
    for target in (var, space):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            got = target.sample(10, method)
        vals = got if target is var else [p['x'] for p in got]
        assert len(vals) == (1 if method == 'grid' and scale == 'linear' else 10)
        assert len(caught) == int(method == 'grid' and scale == 'linear')
        for v in vals:
            assert v in domain and var.validate(v)
            space.apply({'x': v})
            assert var.value == v
