"""Execute helper imports in real packages, then inspect composed SPICE decks."""
import os
import importlib
import re
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from silk.codegen.testbench_generator import (
    TestbenchSpec, TestbenchSpecError, generate_testbenches,
)
from silk.circuit.circuit_elements import CircuitError
from silk.exceptions import UndeclaredNetError
from silk.simulation import Transient


def _ref(name='bias1', path='.biasgen', var='mod', nets=None):
    return {'instance_name': name,
            'module_ref': {'import_path': path, 'module_var': var},
            'nets': {'out': 'out', 'vss': '0'} if nets is None else nets}


def _spec():
    return {'circuit_name': 'chip', 'circuit_builder': '_build_dut',
            'tests': [{'name': 'case', 'dut_portmap': {'out': 'out', 'vss': '0'}}]}


@pytest.fixture
def package(tmp_path, monkeypatch):
    name = 'testbench_helper_package'
    root = tmp_path / name
    root.mkdir()
    (root / '__init__.py').write_text('')
    (root / 'common.py').write_text('''from pathlib import Path
from silk import Module, load_pdk
from silk.elements import elements

ROOT = Path(__file__).parent / 'output'

def make_block(name, pdk=None):
    mod = Module('sky130', str(ROOT), name + '.cir',
                 pdk_details=pdk or load_pdk('sky130'), module_name=name)
    mod.define_module_ports({'out': 'output', 'vss': 'ground'})
    return mod

def make_helper(name):
    leaf = make_block('leaf')
    leaf.add(elements.resistor('r', p='out', n='vss', value=1000))
    mod = make_block(name)
    mod.add_element(leaf, 'nested', nets={'out': 'out', 'vss': 'vss'})
    return mod

def _build_dut(pdk):
    return make_block('dut_block', pdk)
''')
    monkeypatch.syspath_prepend(str(tmp_path))

    def write_helper(path='.biasgen', var='mod', model='biasgen'):
        parts = path.lstrip('.').split('.')
        directory = root
        for part in parts[:-1]:
            directory /= part
            directory.mkdir(exist_ok=True)
            (directory / '__init__.py').touch()
        file = directory / (parts[-1] + '.py')
        file.write_text(f'from {name}.common import make_helper\n'
                        f'{var} = make_helper({model!r})\n')

    def execute(spec):
        code = generate_testbenches(spec)
        (root / 'generated.py').write_text(
            'from .common import _build_dut\n' + code)
        importlib.invalidate_caches()
        generated = importlib.import_module(name + '.generated')
        return generated, code

    yield write_helper, execute
    for key in list(sys.modules):
        if key == name or key.startswith(name + '.'):
            del sys.modules[key]


def _deck(tb):
    tb_netlist_deck_path = tb.netlist.generate_netlist()
    return Path(tb_netlist_deck_path).read_text()


@pytest.mark.parametrize('direct_spec', [False, True])
def test_test_and_rig_helpers_execute_shared_imports(package, direct_spec):
    write_helper, execute = package
    write_helper()
    spec = _spec()
    spec['rigs'] = [{'name': 'bias', 'submodules': [_ref('rig_bias')]}]
    test = spec['tests'][0]
    test['applied_rigs'] = ['bias']
    test['submodules'] = [_ref('local_bias', nets={'out': 'aux', 'vss': '0'})]
    spec['tests'].append({'name': 'other', 'dut_portmap': test['dut_portmap'],
                          'applied_rigs': ['bias']})
    generated, code = execute(TestbenchSpec.from_dict(spec) if direct_spec else spec)
    dut = generated.build_chip_with_tests(None)
    assert code.count('from .biasgen import mod as mod_biasgen') == 1
    for test_name in ('case', 'other'):
        tb = dut._tests[test_name]
        deck = _deck(tb)
        assert 'xrig_bias out 0 biasgen' in deck
        assert 'xnested out vss leaf' in deck
        assert 'xdut out 0 dut_block' in deck
        assert deck.count('.subckt biasgen ') == 1
        assert deck.count('.subckt leaf ') == 1
        assert ('xlocal_bias aux 0 biasgen' in deck) == (test_name == 'case')
        assert tb.module is dut
    assert dut._tests['case'].netlist is not dut._tests['other'].netlist


def test_mixed_rig_sources_and_helpers_execute(package):
    write_helper, execute = package
    write_helper()
    spec = _spec()
    spec['rigs'] = [{'name': 'supply', 'sources': [
        {'type': 'VoltageSource', 'instance': 'supply', 'positive': 'out',
         'negative': '0', 'dc': 1.8}], 'submodules': [_ref()]}]
    spec['tests'][0]['applied_rigs'] = ['supply']
    generated, _ = execute(spec)
    tb = generated.build_chip_with_tests(None)._tests['case']
    assert tb.netlist.sig_sources_dict['supply'].dc == 1.8
    assert 'xbias1 out 0 biasgen' in _deck(tb)


def test_aliases_avoid_distinct_references_and_combined_bindings(package):
    write_helper, execute = package
    references = [('.a.b', 'mod'), ('.a_b', 'mod'),
                  ('.case', '_attach'), ('.rig', '_apply_supply'),
                  ('.tests', 'build_chip_with'), ('.dut', '_build'),
                  ('.pdk', 'load')]
    spec = _spec()
    spec['rigs'] = [{'name': 'supply'}]
    spec['tests'][0]['applied_rigs'] = ['supply']
    helpers = spec['tests'][0]['submodules'] = []
    for index, (path, var) in enumerate(references):
        write_helper(path, var, f'helper{index}')
        helpers.append(_ref(f'helper{index}', path, var))
    generated, code = execute(spec)
    dut = generated.build_chip_with_tests(None)
    tb = dut._tests['case']
    deck = _deck(tb)
    for index in range(len(references)):
        assert f'xhelper{index} out 0 helper{index}' in deck
    assert 'from .a_b import mod as mod_a_b_1' in code
    assert 'from .case import _attach as _attach_case_1' in code
    assert 'from .rig import _apply_supply as _apply_supply_rig_1' in code
    assert 'from .tests import build_chip_with as build_chip_with_tests_1' in code
    assert 'from .dut import _build as _build_dut_1' in code
    assert 'from .pdk import load as load_pdk_1' in code
    assert callable(generated._build_dut)
    assert callable(generated._apply_supply_rig)
    assert callable(generated._attach_case)


@pytest.mark.parametrize('location', ['test', 'rig'])
@pytest.mark.parametrize('failure', ['unknown', 'missing', 'strict'])
def test_generated_wiring_errors_execute_existing_checks(package, location, failure):
    write_helper, execute = package
    write_helper()
    spec = _spec()
    nets = {'out': 'out', 'vss': '0'}
    if failure == 'unknown':
        nets['typo_port'] = 'out'
    elif failure == 'missing':
        del nets['vss']
    else:
        nets['out'] = 'typo_net'
    group = spec['tests'][0]
    if location == 'rig':
        spec['rigs'] = [{'name': 'bias'}]
        group['applied_rigs'] = ['bias']
        group = spec['rigs'][0]
    group['submodules'] = [_ref(nets=nets)]
    generated, _ = execute(spec)
    dut = generated._build_dut(None)
    if failure == 'strict':
        dut.strict_nets = True
        # Public create_test callback only declares legitimate testbench nets.
        create_test = dut.create_test

        def declared_test(name):
            tb = create_test(name)
            tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
            tb.netlist.add_net('out')
            tb.netlist.add_net('0')
            return tb

        dut.create_test = declared_test
        with pytest.raises(UndeclaredNetError, match='typo_net'):
            generated._attach_case(dut)
    elif failure == 'unknown':
        with pytest.raises(CircuitError, match='typo_port.*bias1'):
            generated._attach_case(dut)
    else:
        generated._attach_case(dut)
        tb = dut._tests['case']
        with pytest.raises(CircuitError, match='bias1.*vss'):
            _deck(tb)
        assert not Path(os.path.join(tb.netlist.dir_folder, tb.netlist.netlistname)).exists()


@pytest.mark.parametrize('collision', ['local', 'rig_local', 'rig_rig', 'repeated_rig', 'dut'])
def test_duplicate_instance_names_name_both_contexts(collision):
    spec = _spec()
    spec['rigs'] = [{'name': 'one', 'submodules': [_ref()]},
                    {'name': 'two', 'submodules': [_ref()]}]
    test = spec['tests'][0]
    if collision == 'local':
        test['submodules'] = [_ref(), _ref()]
    elif collision == 'rig_local':
        test['applied_rigs'] = ['one']
        test['submodules'] = [_ref()]
    elif collision == 'rig_rig':
        test['applied_rigs'] = ['one', 'two']
    elif collision == 'repeated_rig':
        test['applied_rigs'] = ['one', 'one']
    else:
        test['submodules'] = [_ref('dut')]
    name = 'dut' if collision == 'dut' else 'bias1'
    with pytest.raises(TestbenchSpecError, match=f"test 'case'.*{name}.*collides"):
        generate_testbenches(spec)


@pytest.mark.parametrize('collision', ['rig_rig', 'rig_local', 'overlapping_range'])
def test_arrayed_helper_collision_on_expanded_names(collision):
    """Task 291 AC#3: collisions are checked on the EXPANDED names, not the
    pre-expansion base instance_name."""
    spec = _spec()
    array_ref = _ref()
    array_ref['array'] = 2
    array_ref['naming'] = 'bracketed'
    if collision == 'rig_rig':
        spec['rigs'] = [{'name': 'one', 'submodules': [array_ref]},
                        {'name': 'two', 'submodules': [array_ref]}]
        spec['tests'][0]['applied_rigs'] = ['one', 'two']
        expanded = 'bias1[0]'
    elif collision == 'rig_local':
        local_array = _ref()
        local_array['array'] = 1
        local_array['naming'] = 'bracketed'
        spec['rigs'] = [{'name': 'one', 'submodules': [array_ref]}]
        spec['tests'][0]['applied_rigs'] = ['one']
        spec['tests'][0]['submodules'] = [local_array]
        expanded = 'bias1[0]'
    else:
        other = _ref()
        other['array'] = [1, 3]
        other['naming'] = 'bracketed'
        spec['rigs'] = [{'name': 'one', 'submodules': [array_ref]},
                        {'name': 'two', 'submodules': [other]}]
        spec['tests'][0]['applied_rigs'] = ['one', 'two']
        expanded = 'bias1[1]'
    with pytest.raises(TestbenchSpecError, match=f"test 'case'.*{re.escape(expanded)}.*collides"):
        generate_testbenches(spec)


@pytest.mark.parametrize('naming,expected', [
    ('bracketed', ['u<0>', 'u<1>', 'u<2>']),
    ('underscore', ['u_0', 'u_1', 'u_2']),
])
def test_arrayed_helper_executes_and_places_n_instances(package, naming, expected):
    """Task 291 AC#2: an arrayed helper runs the generated code (not just
    string-matched) and places one instance per expanded index, with $i
    net templates expanded per instance."""
    write_helper, execute = package
    write_helper()
    spec = _spec()
    spec['rigs'] = [{'name': 'bias', 'submodules': [
        {'instance_name': 'u',
         'module_ref': {'import_path': '.biasgen', 'module_var': 'mod'},
         'nets': {'out': 'net$i', 'vss': '0'},
         'array': 3, 'naming': naming},
    ]}]
    spec['tests'][0]['applied_rigs'] = ['bias']
    generated, code = execute(spec)
    assert 'for i in range(3):' in code
    dut = generated.build_chip_with_tests(None)
    tb = dut._tests['case']
    deck = _deck(tb)
    for i, name in enumerate(expected):
        assert f'x{name} net{i} 0 biasgen' in deck
    assert deck.count('.subckt biasgen ') == 1


def test_arrayed_helper_range_and_index_template_arithmetic(package):
    """Task 291 AC#2: [a, b) range arrays and $(i+-N) arithmetic expand
    identically to circuit-level submodule arrays."""
    write_helper, execute = package
    write_helper()
    spec = _spec()
    spec['rigs'] = [{'name': 'bias', 'submodules': [
        {'instance_name': 'u',
         'module_ref': {'import_path': '.biasgen', 'module_var': 'mod'},
         'nets': {'out': 'net$(i+1)', 'vss': '0'},
         'array': [1, 3], 'naming': 'bracketed'},
    ]}]
    spec['tests'][0]['applied_rigs'] = ['bias']
    generated, _ = execute(spec)
    dut = generated.build_chip_with_tests(None)
    tb = dut._tests['case']
    deck = _deck(tb)
    assert 'xu<1> net2 0 biasgen' in deck
    assert 'xu<2> net3 0 biasgen' in deck
    assert deck.count('.subckt biasgen ') == 1
    assert 'xu<0>' not in deck


def test_arrayed_helper_error_text_matches_circuit_generator():
    """Task 291 AC#1: helper array/naming errors reuse
    generator._parse_array_and_naming, so the wording matches circuit-level
    submodule errors exactly (modulo the label prefix)."""
    from silk.codegen.generator import _parse_array_and_naming

    try:
        _parse_array_and_naming({'array': -1}, 'label')
    except ValueError as exc:
        circuit_message = str(exc).split('label: ', 1)[1]

    spec = _spec()
    spec['tests'][0]['submodules'] = [{**_ref(), 'array': -1}]
    with pytest.raises(TestbenchSpecError) as exc:
        generate_testbenches(spec)
    assert circuit_message in str(exc.value)


@pytest.mark.parametrize('location', ['tests', 'rigs'])
@pytest.mark.parametrize('field,value,match', [
    ('instance_name', None, 'instance_name'),
    ('instance_name', 'bad-name', 'identifier'),
    ('module_ref', None, 'module_ref'),
    ('module_ref', {}, 'import_path'),
    ('module_ref', {'import_path': '.biasgen'}, 'module_var'),
    ('module_ref', {'import_path': '.biasgen; evil()', 'module_var': 'mod'}, 'import_path'),
    ('module_ref', {'import_path': '', 'module_var': 'mod'}, 'import_path'),
    ('module_ref', {'import_path': '.biasgen', 'module_var': 'class'}, 'module_var'),
    ('nets', [], 'nets'),
    ('nets', {'out': 3}, 'out'),
    ('array', 0, "'array' must be > 0"),
    ('array', [3, 1], "'array' list must be"),
    ('naming', 'diagonal', "'naming' must be"),
    ('parameters', {'gain': 2}, 'unsupported'),
])
def test_malformed_helper_specs_are_named(location, field, value, match):
    spec = _spec()
    spec['rigs'] = [{'name': 'bias'}]
    entry = _ref()
    entry[field] = value
    spec[location][0]['submodules'] = [entry]
    with pytest.raises(TestbenchSpecError, match=match) as exc:
        generate_testbenches(spec)
    assert ('test \'case\'' if location == 'tests' else 'rig \'bias\'') in str(exc.value)


@pytest.mark.parametrize('location', ['tests', 'rigs'])
@pytest.mark.parametrize('entries', [None, {}, [None], ['biasgen']])
def test_malformed_helper_list_is_spec_error(location, entries):
    spec = _spec()
    spec['rigs'] = [{'name': 'bias'}]
    spec[location][0]['submodules'] = entries
    with pytest.raises(TestbenchSpecError, match='submodules'):
        generate_testbenches(spec)


def test_omitted_and_empty_helpers_preserve_original_output(package):
    _, execute = package
    spec = _spec()
    spec['rigs'] = [{'name': 'empty'}, {'name': 'supply', 'sources': [
        {'type': 'VoltageSource', 'instance': 'supply', 'positive': 'out',
         'negative': '0', 'dc': 1.8}]}]
    spec['tests'][0]['applied_rigs'] = ['empty', 'supply']
    expected = '''from silk import sources

def _apply_empty_rig(tb):
    pass

def _apply_supply_rig(tb):
    tb.add(sources.vdc('supply', positive='out', negative='0', value=1.8))

def _attach_case(mod):
    tb = mod.create_test('case')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    _apply_empty_rig(tb)
    _apply_supply_rig(tb)

def build_chip_with_tests(pdk):
    mod = _build_dut(pdk)
    _attach_case(mod)
    return mod
'''
    assert generate_testbenches(spec) == expected
    empty = deepcopy(spec)
    for group in empty['rigs'] + empty['tests']:
        group['submodules'] = []
    assert generate_testbenches(empty) == expected
    assert generate_testbenches(TestbenchSpec.from_dict(empty)) == expected
    generated, _ = execute(empty)
    tb = generated.build_chip_with_tests(None)._tests['case']
    assert 'xdut out 0 dut_block' in _deck(tb)
    assert set(tb.netlist.elem_dict) == {'dut'}
