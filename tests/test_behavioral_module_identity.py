"""Behavioral models participate in definition naming, not DB structure identity."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from silk.circuit.modules import Module
from silk.circuit.circuit_elements import register_module_definition as _register_module_definition, _module_structure_hash
from silk.elements import elements
from silk.pdk import load_pdk
from silk.exceptions import CircuitError


def register_module_definition(*args, **kwargs):
    return _register_module_definition(*args, validate=True, **kwargs)


def factory():
    def model(ports, t, params):
        ports.y.v = ports.a.v * params['gain']
    return model


def other_model(ports, t, params):
    ports.y.v = ports.a.v + params['gain']


def block(tmp_path, *, model=None, params=None, circuit=True, variants=False):
    mod = Module('sky130', str(tmp_path), 'amp.cir', pdk_details=load_pdk('sky130'), module_name='amp', variants=variants)
    mod.define_module_ports({'a': 'in', 'y': 'out'})
    if circuit:
        mod.add(elements.resistor('r1', p='a', n='y', value=1000))
    if model is not None:
        mod.set_behavioral_model(model, params=params, execution='vector')
    return mod


@pytest.mark.parametrize('circuit', [True, False])
@pytest.mark.parametrize('difference', ['code', 'params', 'absent'])
def test_behavioral_difference_rejects_same_name(tmp_path, circuit, difference):
    first = block(tmp_path, model=factory(), params={'gain': 2}, circuit=circuit)
    second = block(tmp_path, model=other_model if difference == 'code' else
                   None if difference == 'absent' else factory(),
                   params={'gain': 3 if difference == 'params' else 2}, circuit=circuit)
    registry = {}
    register_module_definition(registry, first)
    with pytest.raises(CircuitError, match="amp.*behavioral model differs"):
        register_module_definition(registry, second)


@pytest.mark.parametrize('circuit', [True, False])
def test_variants_separate_models_and_dedup_factory_rebuilds(tmp_path, circuit):
    registry = {}
    names = []
    for model, gain in [(factory(), 2), (factory(), 3), (other_model, 2), (factory(), 2)]:
        mod = block(tmp_path, model=model, params={'gain': gain}, circuit=circuit, variants=True)
        names.append(register_module_definition(registry, mod)[0])
        assert mod.module_name == 'amp'
    assert len(set(names)) == 3
    assert names[0] == names[3]
    assert len(registry) == 3


def test_db_structure_identity_still_conflates_behavioral_parameters(tmp_path):
    first = block(tmp_path, model=factory(), params={'gain': 2})
    second = block(tmp_path, model=factory(), params={'gain': 5})
    assert _module_structure_hash(first) == _module_structure_hash(second)


def test_closure_values_are_deliberately_not_compared(tmp_path):
    def captured(gain):
        def model(ports, t, params):
            ports.y.v = ports.a.v * gain
        return model
    registry = {}
    for gain in (2, 5):
        register_module_definition(registry, block(tmp_path, model=captured(gain)))
    assert len(registry) == 1


def test_variant_names_stable_across_processes(tmp_path):
    script = '''
import sys
from tests.test_behavioral_module_identity import block, factory
from silk.circuit.circuit_elements import register_module_definition
m = block(sys.argv[1], model=factory(), params=dict(set([('gain', 2), ('offset', 0)])), variants=True)
print(register_module_definition({}, m)[0])
'''
    names = []
    for seed in ('1', '42'):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        names.append(subprocess.check_output([sys.executable, '-c', script, str(tmp_path)], env=env, text=True).strip())
    assert names[0] == names[1]


def test_public_placement_clash_names_both_spellings(tmp_path):
    first = block(tmp_path, model=factory(), params={'gain': 2})
    second = block(tmp_path, model=factory(), params={'gain': 3})
    second.define_module_name('AMP')
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=load_pdk('sky130'))
    top.add_module(first, 'one', a='input', y='output')
    top.add_module(second, 'two', a='input', y='output2')
    with pytest.raises(CircuitError, match="xone.*xtwo.*AMP.*behavioral model differs.*amp"):
        top.generate_netlist()


def test_equal_nested_parameters_dedup_and_changed_array_clashes(tmp_path):
    import numpy as np
    registry = {}
    blocks = []  # Keep every registered source alive through the collision check.
    for params in ({'gain': 2, 'data': [np.array([1., 2.]), {'flags': {'x', 'y'}}]},
                   {'data': [np.array([1., 2.]), {'flags': {'y', 'x'}}], 'gain': 2}):
        blocks.append(block(tmp_path, model=factory(), params=params))
        register_module_definition(registry, blocks[-1])
    assert len(registry) == 1
    with pytest.raises(CircuitError, match='behavioral model differs'):
        register_module_definition(registry, block(tmp_path, model=factory(),
            params={'gain': 2, 'data': [np.array([1., 3.]), {'flags': {'x', 'y'}}]}))


def test_unsupported_identity_fails_actionably(tmp_path):
    with pytest.raises(CircuitError, match='cannot compare behavioral model identity.*explicit params'):
        register_module_definition({}, block(tmp_path, model=factory(),
                                             params={'opaque': object()}, variants=True))


def test_forced_variant_collision_still_checks_model(tmp_path, monkeypatch):
    import silk.circuit.circuit_elements as ce
    monkeypatch.setattr(ce, 'definition_hash', lambda *args: 'f' * 64)
    registry = {}
    first = block(tmp_path, model=factory(), params={'gain': 2}, variants=True)
    register_module_definition(registry, first)
    with pytest.raises(CircuitError, match='behavioral model differs'):
        register_module_definition(registry, block(tmp_path, model=factory(), params={'gain': 3}, variants=True))


@pytest.mark.parametrize('variants', [True, False])
def test_wrapper_identity_includes_descendant_model(tmp_path, variants):
    registry = {}
    names = []
    wrappers = []
    for gain in (2, 5, 2):
        wrapper = Module('sky130', str(tmp_path), 'wrapper.cir',
                         pdk_details=load_pdk('sky130'), module_name='wrapper', variants=variants)
        wrapper.define_module_ports({'a': 'in', 'y': 'out'})
        wrapper.add_module(block(tmp_path, model=factory(), params={'gain': gain}),
                           'child', a='a', y='y')
        wrappers.append(wrapper)
        if gain == 5 and not variants:
            with pytest.raises(CircuitError, match='behavioral model differs'):
                register_module_definition(registry, wrapper)
        else:
            names.append(register_module_definition(registry, wrapper)[0])
    if variants:
        assert names[0] != names[1]
        assert names[0] == names[2]
    else:
        assert names[0] == names[1]


def test_structured_array_field_names_are_model_parameters(tmp_path):
    import numpy as np
    registry = {}
    first = block(tmp_path, model=factory(),
                  params={'table': np.array([(1,)], dtype=[('gain', 'i4')])})
    register_module_definition(registry, first)
    with pytest.raises(CircuitError, match='behavioral model differs'):
        register_module_definition(registry, block(tmp_path, model=factory(),
            params={'table': np.array([(1,)], dtype=[('offset', 'i4')])}))


@pytest.mark.parametrize(('circuit', 'legacy_name'), [
    (True, 'amp_17d4870b'), (False, 'amp_174a491f'),
])
def test_model_free_variant_suffix_is_unchanged(tmp_path, circuit, legacy_name):
    # Names measured before TASK-484, independently of the naming helper.
    mod = block(tmp_path, circuit=circuit, variants=True)
    assert register_module_definition({}, mod)[0] == legacy_name


def partial_factory(k, *, nested=False, offset=0):
    from functools import partial

    def model(offset, ports, t, params, *, k):
        ports.y.v = offset + ports.a.v * k

    inner = partial(model, offset, k=99 if nested else k)
    if not nested:
        return inner
    # A custom attribute prevents Python from flattening this wrapper itself.
    inner.description = 'inner model'
    outer = partial(inner, k=k)
    assert isinstance(outer.func, partial)
    return outer


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('variants', [False, True])
def test_partial_factory_rebuilds_dedup_in_parent(tmp_path, nested, variants):
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=load_pdk('sky130'))
    for name in ('one', 'two'):
        top.add_module(block(tmp_path, model=partial_factory(3, nested=nested),
                             variants=variants), name, a='input', y='output')
    assert len(top.module_dict) == 1


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('difference', ['keyword', 'positional'])
def test_partial_bound_values_are_definition_identity(tmp_path, nested, difference):
    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=load_pdk('sky130'))
    top.add_module(block(tmp_path, model=partial_factory(3, nested=nested)),
                   'one', a='input', y='output')
    other = partial_factory(4 if difference == 'keyword' else 3,
                            nested=nested, offset=1 if difference == 'positional' else 0)
    top.add_module(block(tmp_path, model=other), 'two', a='input', y='output')
    with pytest.raises(CircuitError, match='behavioral model differs'):
        top.generate_netlist()


@pytest.mark.parametrize('variants', [False, True])
def test_model_source_line_changes_do_not_change_identity(tmp_path, variants):
    source = '''def model(ports, t, params):
    def scale(value):
        return value * 3
    ports.y.v = scale(ports.a.v)
'''
    models = []
    for blank_lines in (0, 10):
        scope = {}
        exec(compile('\n' * blank_lines + source, '<model>', 'exec'), scope)
        models.append(scope['model'])
    assert models[0].__code__.co_firstlineno != models[1].__code__.co_firstlineno
    registry = {}
    names = [register_module_definition(registry, block(tmp_path, model=model,
                                                        variants=variants))[0]
             for model in models]
    assert names[0] == names[1]
    assert len(registry) == 1


def test_unknown_params_are_refused_only_when_default_names_meet(tmp_path):
    registry = {}
    first = block(tmp_path, model=factory(), params={'opaque': object()})
    second = block(tmp_path, model=factory(), params={'opaque': object()})
    register_module_definition(registry, first)
    with pytest.raises(CircuitError, match='unsupported identity value object'):
        register_module_definition(registry, second)


@pytest.mark.parametrize('variants', [False, True])
def test_nested_and_flat_partials_share_effective_bindings(tmp_path, variants):
    from functools import partial

    def model(left, right, ports, t, params, *, k):
        ports.y.v = left - right + ports.a.v * k

    inner = partial(model, 2, k=99)
    inner.description = 'keep nested'
    nested = partial(inner, 9, k=3)
    flat = partial(model, 2, 9, k=3)
    registry = {}
    blocks = [block(tmp_path, model=fn, variants=variants) for fn in (nested, flat)]
    names = [register_module_definition(registry, mod)[0] for mod in blocks]
    assert names[0] == names[1]
    assert len(registry) == 1
    reversed_args = block(tmp_path, model=partial(model, 9, 2, k=3), variants=variants)
    if variants:
        assert register_module_definition(registry, reversed_args)[0] != names[0]
    else:
        with pytest.raises(CircuitError, match='behavioral model differs'):
            register_module_definition(registry, reversed_args)


def test_definition_source_ids_belong_to_live_placed_modules(tmp_path):
    import gc
    import weakref

    top = Module('sky130', str(tmp_path), 'top.cir', pdk_details=load_pdk('sky130'))
    sources = []
    try:
        for instance, definition in (('one', 'amp'), ('two', 'amp'), ('three', 'other')):
            inline_block = block(tmp_path, model=partial_factory(3))
            inline_block.define_module_name(definition)
            sources.append(weakref.ref(inline_block))
            top.add_module(inline_block, instance, a='input', y='output')
            del inline_block
        gc.collect()

        # Do not depend on the allocator actually reusing a freed object's id.
        # The original objects must stay alive, including equal factory builds.
        assert all(source() is not None for source in sources)
        placed_ids = {id(entry['device']) for entry in top.elem_dict.values()}
        assert {id(source()) for source in sources} <= placed_ids
        definitions = top.module_dict
        assert len(definitions) == 2
        for entry in definitions.values():
            assert entry['_source_ids']
            assert entry['_source_ids'] <= placed_ids
    finally:
        top.close()
