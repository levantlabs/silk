"""Calling-contract defaults, identity, copies and replay for electrical-port models."""
import copy
import pickle

import pytest

from silk import PortDir
from silk.circuit._behavioral_identity import _own_model_key
from silk.circuit.circuit_elements import (
    _module_structure_hash, register_module_definition as _register)
from silk.circuit.modules import Module
from silk.exceptions import CircuitError, MonteCarloError, NotSupportedError
from silk.simulation.variation import Normal
from tests._variation_bench import chip

def register_module_definition(registry, module):
    return _register(registry, module, validate=True)


FORMS = {'default-vector': {}, 'ports-vector': {'execution': 'vector'},
         'ports-sample': {'execution': 'sample'}}


def gain_model(ports, t, params):
    ports.y.v = ports.a.v * params.gain


def _module(pdk, tmp_path, *, variants=False):
    mod = Module('PDK_skywater130', str(tmp_path), 'amp.cir', pdk_details=pdk,
                 module_name='amp', variants=variants)
    mod.define_module_ports({'a': PortDir.IN, 'y': PortDir.OUT})
    return mod


def _block(pdk, tmp_path, form, *, variants=False):
    mod = _module(pdk, tmp_path, variants=variants)
    mod.set_behavioral_model(gain_model, params={'gain': 2.0}, **FORMS[form])
    return mod


def _fields(mod):
    return mod._behavioral_form, mod._behavioral_execution


def test_a_new_module_and_an_omitted_keyword_use_ports_vector(pdk, tmp_path):
    mod = _module(pdk, tmp_path)
    assert _fields(mod) == ('ports', 'vector')
    mod.set_behavioral_model(gain_model, params={'gain': 2.0})
    assert _fields(mod) == ('ports', 'vector')


@pytest.mark.parametrize('execution', ['vector', 'sample'])
def test_the_keyword_selects_the_ports_form_and_its_contract(pdk, tmp_path, execution):
    mod = _module(pdk, tmp_path)
    mod.set_behavioral_model(gain_model, params={'gain': 2.0}, execution=execution)
    assert _fields(mod) == ('ports', execution)


def test_reattaching_without_the_keyword_restores_vector_execution(pdk, tmp_path):
    mod = _block(pdk, tmp_path, 'ports-sample')
    mod.set_behavioral_model(gain_model, params={'gain': 2.0})
    assert _fields(mod) == ('ports', 'vector')


@pytest.mark.parametrize('bad', ['Vector', 'scalar', 'samples', None, 1])
def test_any_other_contract_raises_before_anything_changes(pdk, tmp_path, bad):
    fresh = _module(pdk, tmp_path / 'fresh')
    with pytest.raises(ValueError, match="'vector' or 'sample'"):
        fresh.set_behavioral_model(gain_model, params={'gain': 2.0}, execution=bad)
    assert fresh._behavioral_model is None and _fields(fresh) == ('ports', 'vector')

    attached = _block(pdk, tmp_path / 'attached', 'ports-sample')
    params = attached._behavioral_params

    def other(ports, t, params):
        pass

    with pytest.raises(ValueError):
        attached.set_behavioral_model(other, params={'gain': 3.0}, execution=bad)
    assert attached._behavioral_model is gain_model
    assert attached._behavioral_params is params
    assert _fields(attached) == ('ports', 'sample')


def test_bad_params_change_nothing(pdk, tmp_path):
    mod = _block(pdk, tmp_path, 'default-vector')
    with pytest.raises(NotSupportedError):
        mod.set_behavioral_model(gain_model, params={'gain': {'nested': Normal(0, 1)}},
                                 execution='sample')
    assert _fields(mod) == ('ports', 'vector') and mod._behavioral_params == {'gain': 2.0}


def test_execution_is_keyword_only(pdk, tmp_path):
    with pytest.raises(TypeError):
        _module(pdk, tmp_path).set_behavioral_model(gain_model, {'gain': 2.0}, 'sample')


def test_signature_defaults_to_vector():
    import inspect
    sig = inspect.signature(Module.set_behavioral_model)
    assert str(sig) == "(self, fn, params=None, *, execution='vector')"


def test_default_and_explicit_vector_share_identity(pdk, tmp_path):
    keys = {form: _own_model_key(_block(pdk, tmp_path / form, form)) for form in FORMS}
    assert keys['default-vector'] == keys['ports-vector']
    assert keys['ports-vector'] != keys['ports-sample']


def test_definition_naming_separates_execution_contracts(pdk, tmp_path):
    registry = {}
    vector = _block(pdk, tmp_path / 'vector', 'ports-vector')
    register_module_definition(registry, vector)
    with pytest.raises(CircuitError, match='amp.*behavioral model differs'):
        register_module_definition(registry, _block(pdk, tmp_path / 'sample', 'ports-sample'))

    registry = {}
    names = [register_module_definition(
        registry, _block(pdk, tmp_path / f'v{k}', form, variants=True))[0]
        for k, form in enumerate([*FORMS, 'default-vector'])]
    assert names[0] == names[1] == names[3]
    assert names[2] != names[0]


def test_spice_identity_is_unchanged(pdk, tmp_path):
    hashes = {_module_structure_hash(_block(pdk, tmp_path / form, form)) for form in FORMS}
    assert len(hashes) == 1


@pytest.mark.parametrize('how', ['copy', 'deepcopy', 'pickle'])
@pytest.mark.parametrize('form', list(FORMS))
def test_copies_keep_form_and_contract(pdk, tmp_path, how, form):
    mod = _block(pdk, tmp_path, form)
    clone = {'copy': copy.copy, 'deepcopy': copy.deepcopy,
             'pickle': lambda m: pickle.loads(pickle.dumps(m))}[how](mod)
    assert _fields(clone) == _fields(mod)


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('change', ['same', 'execution'])
def test_replay_refuses_a_change_of_execution_contract_only(pdk, tmp_path, monkeypatch, change):
    """mc.rerun drives the real replay hook; a refusal comes before any
    prepare, simulate or evaluate, and the sample evidence is kept."""
    from silk.simulation._replay import experiment_signature

    top, tb = chip(pdk, tmp_path, {'offset': Normal(0.0, 1e-3)})
    block = top.elem_dict['c0']['device']
    model, params = block._behavioral_model, block._behavioral_params
    if change == 'execution':
        def model(ports, t, params):
            ports.y.v = ports.a.v + params.offset
        block.set_behavioral_model(model, params=params, execution='vector')
    mc = tb.set_monte_carlo(n=2).run()
    key = list(mc.keys())[0]
    recorded = experiment_signature(tb, mc.settings, {})
    recorded.require_valid()

    kwargs = {'same': {},
              'execution': {'execution': 'sample'}}[change]
    block.set_behavioral_model(model, params=params, **kwargs)
    experiment_signature(tb, mc.settings, {}).require_valid()
    if change == 'same':
        assert mc.rerun(key).metrics == pytest.approx(mc[key].metrics)
        return

    calls = []
    for name in ('_prepare_run', '_simulate_run', '_evaluate_run'):
        monkeypatch.setattr(tb, name, lambda *a, _name=name, **k: calls.append(_name))
    before = _files(tmp_path)
    with pytest.raises(MonteCarloError, match='changed.*model'):
        mc.rerun(key)
    assert calls == []
    assert _files(tmp_path) == before
