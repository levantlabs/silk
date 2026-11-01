"""Catalog-driven fingers and derived mismatch multiplicity contracts."""
import copy
from pathlib import Path

import pytest

from silk import load_pdk
from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.exceptions import CircuitError, NotSupportedError

FAMILIES = ('nmos1p8', 'nmos1p8_lvt', 'nmos3p3_native', 'nmos5v',
            'nmos5v_native', 'pmos1p8', 'pmos1p8_lvt', 'pmos1p8_hvt', 'pmos5v')


@pytest.fixture
def pdk():
    return load_pdk('sky130')


def device(pdk, family='nmos1p8', name='mn', **params):
    pins = {p: ('d' if p == 'd' else 'g' if p == 'g' else '0')
            for p in getattr(pdk, family)._pin_names}
    return getattr(pdk, family)(name, **pins, **params)


def deck(mod, **kwargs):
    return Path(mod.generate_netlist(**kwargs)).read_text()


def bench(tmp_path, pdk, **params):
    mod = Module('sky130', str(tmp_path), 'fingers', pdk)
    mod.add(device(pdk, **params))
    return mod


@pytest.mark.parametrize('family', FAMILIES)
def test_optional_fingers_and_derived_multiplicity(pdk, family):
    old = device(pdk, family).to_spice()
    assert ' nf=' not in old and ' mult=' not in old
    explicit = device(pdk, family, nf=1).to_spice()
    assert explicit == old.rstrip() + ' nf=1\n'
    multi = device(pdk, family, nf=2, m=4).to_spice()
    assert ' m=4 ' in multi and multi.endswith(' nf=2 mult=4\n')
    with pytest.raises(TypeError, match='mult'):
        device(pdk, family, mult=1)


@pytest.mark.parametrize('value', [0, -1, 1.5, float('inf'), float('nan'), True, False])
def test_invalid_fingers_fail_at_emission(pdk, value):
    dev = device(pdk, nf=value)
    with pytest.raises(CircuitError, match='nf'):
        dev.to_spice()


def test_live_variable_mutation_and_literal(pdk):
    m = Variable('parallel', value=2)
    dev = device(pdk, m=m)
    assert dev.to_spice().endswith(' mult=2\n')
    m.value = 4
    assert dev.to_spice().endswith(' mult=4\n')
    dev.params['nf'] = 1
    assert dev.to_spice().endswith(' nf=1 mult=4\n')
    dev.nf = 2
    assert dev.to_spice().endswith(' nf=2 mult=4\n')
    assert ' nf={fingers}' in device(pdk, nf='{fingers}').to_spice()
    assert device(pdk).nf == 1


@pytest.mark.parametrize('family', ['nmos_esd', 'pmos_esd', 'nmos20v',
                                    'nmos20v_native', 'nmos20v_zvt',
                                    'nmos20v_iso', 'pmos20v'])
def test_other_families_stay_unsupported(pdk, family):
    for param in ('nf', 'mult'):
        with pytest.raises(TypeError, match=param):
            device(pdk, family, **{param: 2})


@pytest.mark.parametrize('values, tail', [([1, 1, .15], 'm=1 w=1 l=0.15'),
                                         ([4, 1, .15], 'm=4 w=1 l=0.15 mult=4'),
                                         ([1, 1, .15, 2], 'm=1 w=1 l=0.15 nf=2')])
def test_legacy_positional_lists(tmp_path, pdk, values, tail):
    mod = Module('sky130', str(tmp_path), 'legacy', pdk)
    mod.add_element(pdk.nmos1p8, 'mn', nets={'d': 'd', 'g': 'g', 's': '0', 'b': '0'}, parameters=values)
    assert next(s for s in deck(mod).splitlines() if s.startswith('XMmn ')).endswith(tail)


@pytest.mark.parametrize('explicit', [None, 0, 1])
def test_conditional_option_and_caller_precedence(tmp_path, pdk, explicit):
    mod = bench(tmp_path, pdk, nf=1)
    if explicit is not None:
        mod.set_simulator_options(wnflag=explicit)
    out = deck(mod)
    assert out.count('wnflag=') == 1
    assert f'wnflag={1 if explicit is None else explicit}' in out
    mod.elem_dict['mn']['device'].params.pop('nf')
    if explicit is None:
        assert 'wnflag=' not in deck(mod)


@pytest.mark.parametrize('value', [2, '{fingers}', Variable('fingers', value=2)])
def test_xyce_refuses_unprovable_or_multifinger(tmp_path, pdk, value):
    mod = bench(tmp_path, pdk, nf=value)
    mod.simulator = 'xyce'
    with pytest.raises(NotSupportedError, match='nf.*[Xx]yce|[Xx]yce.*nf'):
        deck(mod)


@pytest.mark.parametrize('value', [None, 1])
def test_xyce_single_finger_remains_supported(tmp_path, pdk, value):
    mod = bench(tmp_path, pdk, **({} if value is None else {'nf': value}))
    mod.simulator = 'xyce'
    assert 'wnflag' not in deck(mod)


@pytest.mark.parametrize('reverse', [False, True])
def test_ammeter_forwarding_independent_of_instance_order(tmp_path, pdk, reverse):
    mod = Module('sky130', str(tmp_path), 'amm', pdk)
    devices = [device(pdk, name='a'), device(pdk, name='b', nf=2, m=4)]
    for dev in devices[::-1] if reverse else devices:
        mod.add(dev)
    from silk.simulation import Transient
    mod.add_sim_type(Transient(step=1e-9, stop=1e-8))
    mod.save_device_currents = True
    mod.terminal_current_method = 'ammeter_wrapper'
    out = deck(mod)
    header = next(s for s in out.splitlines() if s.startswith('.subckt amm_'))
    assert 'nf=1' in header and 'mult=1' in header
    inner = next(s for s in out.splitlines() if s.startswith('XMdev '))
    assert 'nf={nf}' in inner and 'mult={mult}' in inner
    a = next(s for s in out.splitlines() if s.startswith('XMa '))
    b = next(s for s in out.splitlines() if s.startswith('XMb '))
    assert 'nf=' not in a and 'mult=' not in a
    assert b.endswith('nf=2 mult=4')


def test_schema_preserves_optional_semantics(pdk):
    from silk.elements.schema import export_schema
    entry = export_schema(pdk, include_metadata=False)['nmos1p8']
    nf = next(p for p in entry['parameters'] if p['name'] == 'nf')
    assert nf['emit_if_supplied'] is True
    assert entry['derivedParams'] == {'mult': {'from': 'm', 'omit_when': 1}}
    assert 'param_nf' not in entry['code_template']
    assert 'param_mult' not in entry['code_template']


@pytest.mark.parametrize('api', ['factory', 'legacy'])
def test_nested_requirements_wrappers_and_effective_values(tmp_path, pdk, api):
    from silk import Port
    from silk.simulation import Transient
    child = Module('sky130', str(tmp_path / 'child'), 'child', pdk, module_name='child')
    child.define_module_ports({'d': Port.analog_output(), 'g': Port.analog_input()})
    if api == 'factory':
        child.add(device(pdk, nf=2, m=4))
    else:
        child.add_element(pdk.nmos1p8, 'mn', nets={'d': 'd', 'g': 'g', 's': '0', 'b': '0'},
                          parameters={'nf': Variable('f', value=2), 'm': Variable('m', value=4)})
    top = bench(tmp_path / 'top', pdk)
    top.add_module(child, 'u', d='d', g='g')
    top.add_sim_type(Transient(step=1e-9, stop=1e-8))
    top.save_device_currents = True
    top.terminal_current_method = 'ammeter_wrapper'
    out = deck(top)
    assert out.count('wnflag=1') == 1
    assert 'nf={nf} mult={mult}' in out
    top.simulator = 'xyce'
    with pytest.raises(NotSupportedError, match='nf'):
        deck(top)
    # A previous parent export must not leak its accumulator or backend.
    assert 'nf=2 mult=4' in child.generate_netlist()


@pytest.mark.parametrize('api', ['factory', 'legacy'])
def test_fault_native_overrides_retain_presence_and_validation(pdk, api):
    from types import SimpleNamespace
    from silk.circuit.spice_netlister import _derive_faulted_instance, _derive_faulted_pdk_device
    fault = SimpleNamespace(name=None, nets={}, params={'nf': 2, 'm': 4})
    if api == 'factory':
        source = device(pdk)
        emit = lambda f: _derive_faulted_instance(source, f).to_spice()
    else:
        source = copy.deepcopy(pdk._devices['nmos1p8'])
        for p, net in zip(source.pin_list, ('d', 'g', '0', '0')):
            source.pin_list[p].net = net
        emit = lambda f: pdk.generate_netlist_entry('mn', {'device': _derive_faulted_pdk_device(source, f)})
    assert emit(fault).endswith('nf=2 mult=4\n')
    fault.params['nf'] = 0
    with pytest.raises(CircuitError, match='nf'):
        emit(fault)


def test_legacy_update_can_omit_optional_trailing_parameter(tmp_path, pdk):
    mod = Module('sky130', str(tmp_path), 'legacy', pdk)
    mod.add_element(pdk.nmos1p8, 'mn', nets={'d': 'd', 'g': 'g', 's': '0', 'b': '0'})
    params = list(mod.elem_dict['mn']['device'].param_list)
    mod._element_mgr.update_element('mn', params, [4, 1, .15])
    assert 'm=4 w=1 l=0.15 mult=4' in deck(mod)


@pytest.mark.parametrize('derived', [
    {'m': {'from': 'm', 'omit_when': 1}},
    {'extra': {'from': 'missing', 'omit_when': 1}},
    {'bad-name': {'from': 'm', 'omit_when': 1}},
    {'extra': {'from': 'm'}},
])
def test_invalid_derived_catalog_rejected_at_load(tmp_path, derived):
    import yaml
    from silk.pdk import BasePDK
    from silk.exceptions import PDKError
    spec = {'demo': {'PDKDevice': 'demo', 'pinList': {'p': 'plus'},
                     'paramList': {'m': {'minval': 1, 'maxval': -1}},
                     'derivedParams': derived}}
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(PDKError, match='derivedParams'):
        BasePDK(str(path), library_location='')


def test_derived_metadata_is_generic_and_uses_live_values(tmp_path):
    import yaml
    from silk.pdk import BasePDK
    spec = {'demo': {'PDKDevice': 'demo', 'pinList': {'p': 'plus'},
                     'paramList': {'copies': {'minval': 1, 'maxval': -1}},
                     'derivedParams': {'area_scale': {'from': 'copies', 'omit_when': 1}}}}
    path = tmp_path / 'devices.yaml'
    path.write_text(yaml.safe_dump(spec))
    pdk = BasePDK(str(path), library_location='')
    dev = pdk.demo('x', p='0')
    assert dev.to_spice() == 'XMx 0 demo copies=1\n'
    dev.copies = 3
    assert dev.to_spice() == 'XMx 0 demo copies=3 area_scale=3\n'


def test_caller_option_precedence_is_case_insensitive(tmp_path, pdk):
    mod = bench(tmp_path, pdk, nf=2)
    mod.set_simulator_options(WNFLAG=0)
    out = deck(mod)
    assert 'wnflag=1' not in out
    assert 'WNFLAG=0' in out


def test_legacy_readonly_derived_parameter_refusal(tmp_path, pdk):
    mod = Module('sky130', str(tmp_path), 'legacy', pdk)
    with pytest.raises(TypeError, match='mult'):
        mod.add_element(pdk.nmos1p8, 'mn', nets={'d': 'd', 'g': 'g', 's': '0', 'b': '0'},
                        parameters={'mult': 4})


def test_derived_source_cannot_itself_be_omitted(tmp_path):
    import yaml
    from silk.pdk import BasePDK
    from silk.exceptions import PDKError
    spec = {'demo': {'PDKDevice': 'demo', 'pinList': {'p': 'plus'},
                     'paramList': {'copies': {'minval': 1, 'maxval': -1,
                                              'emit_if_supplied': True}},
                     'derivedParams': {'scale': {'from': 'copies', 'omit_when': 1}}}}
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(PDKError, match='derivedParams.*source'):
        BasePDK(str(path), library_location='')


def test_strict_optional_parameter_accepts_string_catalog_bounds(tmp_path):
    import yaml
    from silk.pdk import BasePDK
    spec = {'demo': {'PDKDevice': 'demo', 'pinList': {'p': 'plus'},
                     'paramList': {'count': {'minval': '1', 'maxval': '-1',
                                            'default': 1, 'strict': True,
                                            'emit_if_supplied': True}}}}
    path = tmp_path / 'devices.yaml'
    path.write_text(yaml.safe_dump(spec))
    pdk = BasePDK(str(path), library_location='')
    assert pdk.demo('x', p='0', count=2).to_spice() == 'XMx 0 demo count=2\n'


@pytest.mark.parametrize('nf', [None, 1, 2])
def test_generated_circuit_preserves_omission_and_live_multiplier(tmp_path, nf):
    from silk.codegen.generator import Generator
    params = {'m': {'variable': 'copies'}, 'w': 4, 'l': .15}
    if nf is not None:
        params['nf'] = nf
    spec = {'circuit_name': 'fingers', 'pdk': 'sky130', 'output_dir': str(tmp_path),
            'ports': {'d': {'dir': 'inout'}, 'g': {'dir': 'inout'}},
            'variables': [{'name': 'copies', 'value': 4}],
            'devices': [{'type': 'nmos1p8', 'instance': 'mn',
                         'nets': {'d': 'd', 'g': 'g', 's': '0', 'b': '0'},
                         'params': params}]}
    namespace = {}
    exec(compile(Generator().generate(spec), '<generated>', 'exec'), namespace)
    mod = namespace['mod']
    text = mod.generate_netlist()
    assert (' nf=' in text) == (nf is not None)
    assert 'mult=4' in text
    namespace['copies'].value = 2
    assert 'mult=2' in mod.generate_netlist()


def test_omitted_fingers_are_not_missing_required_lint_parameters(tmp_path, pdk):
    from silk.lint.rules.structural import MissingParamRule
    mod = bench(tmp_path, pdk)
    assert list(MissingParamRule().check(mod)) == []


def test_removing_required_parameter_remains_visible_to_lint(tmp_path, pdk):
    from silk.lint.rules.structural import MissingParamRule
    mod = bench(tmp_path, pdk)
    dev = mod.elem_dict['mn']['device']
    dev.params.pop('w')
    assert dev.w is None
    findings = list(MissingParamRule().check(mod))
    assert len(findings) == 1 and "'w'" in findings[0].message


@pytest.mark.parametrize('nf, m', [('{1 + 1}', '{2 + 2}'), ("'1 + 1'", "'2 + 2'")])
def test_ammeter_forwarding_preserves_spaced_literal_expressions(tmp_path, pdk, nf, m):
    from silk.simulation import Transient
    mod = bench(tmp_path, pdk, nf=nf, m=m)
    mod.add(device(pdk, name='second'))
    mod.add_sim_type(Transient(step=1e-9, stop=1e-8))
    mod.save_device_currents = True
    mod.terminal_current_method = 'ammeter_wrapper'
    out = deck(mod)
    assert ('XMdev d_i g_i s_i b_i sky130_fd_pr__nfet_01v8 '
            'm={m} w={w} l={l} nf={nf} mult={mult}\n') in out
    assert ('XMmn d g 0 0 amm_sky130_fd_pr__nfet_01v8 '
            f'm={m} w=1 l=0.15 nf={nf} mult={m}\n') in out
    header = next(s for s in out.splitlines() if s.startswith('.subckt amm_'))
    assert header.endswith(f'm={m} w=1 l=0.15 nf=1 mult=1')


@pytest.mark.parametrize('api', ['factory', 'legacy'])
def test_derived_destination_refused_on_attribute_assignment(pdk, api):
    dev = device(pdk) if api == 'factory' else copy.deepcopy(pdk._devices['nmos1p8'])
    with pytest.raises(TypeError, match='mult.*derived'):
        dev.mult = 8


@pytest.mark.parametrize('api', ['factory', 'legacy'])
@pytest.mark.parametrize('operation', ['item', 'update', 'setdefault', 'union', 'replace'])
def test_derived_destination_refused_on_mapping_assignment(pdk, api, operation):
    dev = device(pdk) if api == 'factory' else copy.deepcopy(pdk._devices['nmos1p8'])
    attr = 'params' if api == 'factory' else 'param_list'
    values = getattr(dev, attr)
    with pytest.raises(TypeError, match='mult.*derived'):
        if operation == 'item':
            values['mult'] = 8
        elif operation == 'update':
            values.update(mult=8)
        elif operation == 'setdefault':
            values.setdefault('mult', 8)
        elif operation == 'union':
            values |= {'mult': 8}
        else:
            setattr(dev, attr, {**values, 'mult': 8})


@pytest.mark.parametrize('api', ['factory', 'legacy'])
@pytest.mark.parametrize('raw', ['mapping', 'attribute'])
def test_emission_refuses_bypassed_derived_assignment(pdk, api, raw):
    dev = device(pdk) if api == 'factory' else copy.deepcopy(pdk._devices['nmos1p8'])
    if raw == 'attribute':
        vars(dev)['mult'] = 8
    else:
        values = dev.params if api == 'factory' else dev.param_list
        dict.__setitem__(values, 'mult', 8)
    if api == 'legacy':
        for p, net in zip(dev.pin_list, ('d', 'g', '0', '0')):
            dev.pin_list[p].net = net
    with pytest.raises(TypeError, match='mult.*derived'):
        if api == 'factory':
            dev.to_spice()
        else:
            pdk.generate_netlist_entry('mn', {'device': dev})


@pytest.mark.parametrize('api', ['factory', 'legacy', 'factory_variable', 'legacy_variable'])
@pytest.mark.parametrize('scalar', ['int64', 'float32', 'float64', 'fraction'])
def test_real_scalar_one_omits_derived_multiplicity(pdk, api, scalar):
    import numpy as np
    from fractions import Fraction
    value = Fraction(1) if scalar == 'fraction' else getattr(np, scalar)(1)
    if api.endswith('_variable'):
        value = Variable('copies', value=value)
    if api.startswith('factory'):
        text = device(pdk, m=value).to_spice()
    else:
        dev = copy.deepcopy(pdk._devices['nmos1p8'])
        dev.param_list['m'].value = value
        for p, net in zip(dev.pin_list, ('d', 'g', '0', '0')):
            dev.pin_list[p].net = net
        text = pdk.generate_netlist_entry('mn', {'device': dev})
    assert ' mult=' not in text


@pytest.mark.parametrize('api', ['factory', 'legacy'])
@pytest.mark.parametrize('pdk_name,family,param', [
    ('sky130', 'nmos1p8', 'w'),
    ('sky130', 'nmos1p8', 'm'),
    ('gf180mcu', 'nmos3p3', 'w'),
    ('sky130', 'res_hpo', 'w'),
])
def test_missing_required_parameter_refused(pdk_name, family, param, api):
    pdk = load_pdk(pdk_name)
    if api == 'factory':
        dev = device(pdk, family=family)
        dev.params.pop(param)
        emit = dev.to_spice
    else:
        dev = copy.deepcopy(pdk._devices[family])
        for pin in dev.pin_list.values():
            pin.net = '0'
        dev.param_list.pop(param)
        emit = lambda: pdk.generate_netlist_entry('mn', {'device': dev})
    with pytest.raises(CircuitError, match=f'mn.*{param}.*required.*restore'):
        emit()


@pytest.mark.parametrize('api', ['factory', 'legacy'])
def test_missing_optional_parameter_remains_allowed(pdk, api):
    if api == 'factory':
        dev = device(pdk, nf=2)
        dev.params.pop('nf')
        text = dev.to_spice()
    else:
        dev = copy.deepcopy(pdk._devices['nmos1p8'])
        for pin in dev.pin_list.values():
            pin.net = '0'
        dev.param_list.pop('nf')
        text = pdk.generate_netlist_entry('mn', {'device': dev})
    assert ' nf=' not in text and ' mult=' not in text


@pytest.mark.parametrize('api', ['factory', 'legacy'])
@pytest.mark.parametrize('transport', ['deepcopy', 'pickle'])
def test_derived_mapping_guard_survives_copy_and_pickle(pdk, api, transport):
    import pickle
    dev = device(pdk, m=4) if api == 'factory' else copy.deepcopy(pdk._devices['nmos1p8'])
    if api == 'legacy':
        dev.param_list['m'].value = 4
    restored = copy.deepcopy(dev) if transport == 'deepcopy' else pickle.loads(pickle.dumps(dev))
    values = restored.params if api == 'factory' else restored.param_list
    with pytest.raises(TypeError, match='mult.*derived'):
        values['mult'] = 8
    if api == 'legacy':
        for p, net in zip(restored.pin_list, ('d', 'g', '0', '0')):
            restored.pin_list[p].net = net
        text = pdk.generate_netlist_entry('mn', {'device': restored})
    else:
        text = restored.to_spice()
    assert text.endswith('mult=4\n')
