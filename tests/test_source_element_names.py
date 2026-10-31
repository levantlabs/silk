"""Public saved-entry names must match the cards actually emitted (TASK-522)."""
from pathlib import Path
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.codegen.generator import _source_entry_to_spec
from silk.codegen.testbench_generator import TestbenchSpecError, _source_add_line
from silk.exceptions import NotSupportedError
from silk.simulation import _ngspice_version


@pytest.fixture
def measured_ngspice_version(monkeypatch):
    """Compare deck structure without discovering an installed simulator."""
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')

# Literal fixtures also drive the real-ngspice probe inventory in the slow file.
KINDS = [
    ('VoltageSource', 'V', True, {'positive': 'out', 'negative': '0', 'dc': .2}),
    ('CurrentSource', 'I', True, {'positive': 'out', 'negative': '0', 'dc': .2}),
    ('VCVS', 'E', True, {'out_pos': 'out', 'out_neg': '0', 'ctrl_pos': 'control', 'ctrl_neg': '0', 'gain': 2}),
    ('VCCS', 'G', False, {'out_pos': 'out', 'out_neg': '0', 'ctrl_pos': 'control', 'ctrl_neg': '0', 'gm': .002}),
    ('CCVS', 'H', True, {'out_pos': 'out', 'out_neg': '0', 'ctrl_vname': 'control', 'transresistance': 2}),
    ('CCCS', 'F', False, {'out_pos': 'out', 'out_neg': '0', 'ctrl_vname': 'control', 'gain': 2}),
    ('BehavioralVoltage', 'B', True, {'positive': 'out', 'negative': '0', 'expression': '.2'}),
    ('BehavioralCurrent', 'B', False, {'positive': 'out', 'negative': '0', 'expression': '.2'}),
]
FULL_NAMES = [
    ('VoltageControlledVoltageSource', *KINDS[2][1:]),
    ('VoltageControlledCurrentSource', *KINDS[3][1:]),
    ('CurrentControlledVoltageSource', *KINDS[4][1:]),
    ('CurrentControlledCurrentSource', *KINDS[5][1:]),
]
WAVES = [
    ('constant', {}),
    ('pwl', {'pwl': [0, .1, 1e-6, .3]}),
    ('pwl-waveform', {'waveform': {'type': 'PWL', 'params': {'points': [0, .1, 1e-6, .3]}}}),
    ('pulse', {'waveform': {'type': 'Pulse', 'params': {'v1': .1, 'v2': .3, 'tr': 1e-8, 'tf': 1e-8, 'pw': 1e-6, 'per': 2e-6}}}),
    ('sine', {'waveform': {'type': 'Sine', 'params': {'vampl': .1, 'freq': 1e3}}}),
    ('exponential', {'waveform': {'type': 'Exponential', 'params': {'v1': .1, 'v2': .3, 'td1': 1e-6, 'tau1': 1e-7, 'td2': 2e-6, 'tau2': 1e-7}}}),
    ('sffm', {'waveform': {'type': 'SFFM', 'params': {'vo': .1, 'va': .1, 'fc': 1e3, 'mdi': 2, 'fs': 10}}}),
    ('am', {'waveform': {'type': 'AM', 'params': {'va': .1, 'vo': .1, 'mf': 1e3, 'fc': 1e6}}}),
    ('trnoise', {'waveform': {'type': 'TransientNoise', 'params': {'na': .01, 'nt': 1e-7}}}),
    ('trrandom', {'waveform': {'type': 'RandomSource', 'params': {'type': 1, 'ts': 1e-7, 'param1': .1, 'param2': .1}}}),
    ('bitpattern', {'waveform': {'type': 'BitPattern', 'params': {'bits': '01', 'v1': .1, 'v2': .3, 'ui': 1e-6, 'tr': 1e-8, 'tf': 1e-8}}}),
    ('prbs', {'waveform': {'type': 'PRBS', 'params': {'order': 7, 'length': 4, 'v1': .1, 'v2': .3, 'ui': 1e-6, 'tr': 1e-8, 'tf': 1e-8}}}),
]
NAMINGS = [({}, ['Stim'], ['Stim']),
           ({'array': [2, 4]}, ['Stim[2]', 'Stim[3]'], ['Stim<2>', 'Stim<3>']),
           ({'array': [2, 4], 'naming': 'underscore'}, ['Stim_2', 'Stim_3'], ['Stim_2', 'Stim_3'])]
CONTROL = {'type': 'VoltageSource', 'instance': 'control', 'positive': 'control', 'negative': '0', 'dc': .4}


def emitted_module(tmp_path, entry):
    """Use the actual testbench source emitter, including its factory dispatch."""
    mod = Module(None, str(tmp_path), 'names.cir', None)
    specs = [_source_entry_to_spec(e, 'test') for e in (CONTROL, entry)]
    ns = {'tb': mod, 'sources': sources}
    exec(_source_add_line(CONTROL, specs), ns)
    exec(_source_add_line(entry, specs), ns)
    return mod


@pytest.mark.parametrize('kind,letter,readable,fields', KINDS + FULL_NAMES)
@pytest.mark.parametrize('array,names,components', NAMINGS)
def test_each_kind_reports_only_created_elements_and_matches_deck(tmp_path, kind, letter, readable, fields, array, names, components):
    from silk.codegen import source_element_names
    entry = {'type': kind, 'instance': 'Stim', **fields, **array}
    records = source_element_names(entry)
    assert isinstance(records, tuple)
    assert [r.designer_name for r in records] == names
    assert [r.current_probe for r in records] == [f'i({n})' if readable else None for n in names]
    assert [r.card_names for r in records] == [(letter + n,) for n in components]
    mod = emitted_module(tmp_path, entry)
    mod_deck_path = mod.generate_netlist()
    deck = Path(mod_deck_path).read_text()
    emitted = [line.split()[0] for line in deck.splitlines() if line.startswith(letter + 'Stim')]
    assert emitted == [card for r in records for card in r.card_names]


@pytest.mark.parametrize('kind,letter', [('VoltageSource', 'V'), ('CurrentSource', 'I')])
@pytest.mark.parametrize('wave,fields', WAVES)
@pytest.mark.parametrize('array,names,components', NAMINGS)
def test_every_independent_waveform_matches_emitted_cards(tmp_path, measured_ngspice_version, kind, letter, wave, fields, array, names, components):
    from silk.codegen import source_element_names
    entry = {'type': kind, 'instance': 'Stim', 'positive': 'out', 'negative': '0', 'dc': .2, **fields, **array}
    records = source_element_names(entry)
    suffix = '__silk_pwl' if wave in ('pwl', 'pwl-waveform') else '__silk_wave'
    expected = [(letter + n,) if wave == 'constant' else (letter + n + suffix, letter + n + '__silk_dc') for n in components]
    assert [r.card_names for r in records] == expected
    assert [r.current_probe for r in records] == [f'i({n})' for n in names]
    mod = emitted_module(tmp_path, entry)
    mod_deck_path = mod.generate_netlist()
    deck = Path(mod_deck_path).read_text()
    assert [line.split()[0] for line in deck.splitlines() if line.startswith(letter + 'Stim')] == [c for pair in expected for c in pair]


@pytest.mark.parametrize('array,names', [(1, ['Stim[0]']), (2, ['Stim[0]', 'Stim[1]'])])
def test_integer_arrays_exclude_the_bare_list_name(array, names):
    from silk.codegen import source_element_names
    assert [r.designer_name for r in source_element_names(dict(CONTROL, instance='Stim', array=array))] == names


@pytest.mark.parametrize('kind,letter', [('VoltageSource', 'V'), ('CurrentSource', 'I')])
def test_literal_colon_source_name_is_refused(kind, letter):
    from silk.circuit import source_element_names
    with pytest.raises(TestbenchSpecError, match='hierarchy levels'):
        source_element_names(dict(CONTROL, type=kind, instance='drive:part'))


def test_scalar_bracketed_name_and_negative_array_range(tmp_path):
    from silk.circuit import ElementNames, source_element_names
    entry = dict(CONTROL, instance='Stim[2]')
    assert source_element_names(entry) == (ElementNames('Stim[2]', 'i(Stim[2])', ('VStim<2>',)),)
    assert emitted_module(tmp_path, entry).sig_sources_dict['Stim[2]'].instance_name == 'VStim<2>'
    with pytest.raises(TestbenchSpecError, match='ASCII'):
        source_element_names(dict(CONTROL, instance='Stim', array=[-2, 0]))


@pytest.mark.parametrize('kind,letter,readable,fields', [KINDS[0], KINDS[2], KINDS[6]])
def test_array_terminal_templates_with_spaces_remain_valid(tmp_path, kind, letter, readable, fields):
    from silk.circuit import source_element_names
    pin = 'out_pos' if 'out_pos' in fields else 'positive'
    entry = {'type': kind, 'instance': 'Stim', **fields,
             'array': 2, pin: 'out[$( i + 1 )]'}
    assert [r.card_names for r in source_element_names(entry)] == [
        (letter + 'Stim<0>',), (letter + 'Stim<1>',)]
    mod = emitted_module(tmp_path, entry)
    mod_deck_path = mod.generate_netlist()
    deck = Path(mod_deck_path).read_text()
    assert f'{letter}Stim<0> out[1] 0 ' in deck
    assert f'{letter}Stim<1> out[2] 0 ' in deck


@pytest.mark.parametrize('kind,letter,readable,fields', [KINDS[2], KINDS[6]])
@pytest.mark.parametrize('name', ['bad-name', 'Stim[0]'])
def test_controlled_and_behavioral_scalar_names_follow_codegen_rules(kind, letter, readable, fields, name):
    from silk.circuit import source_element_names
    entry = {'type': kind, 'instance': name, **fields}
    with pytest.raises(TestbenchSpecError, match='identifier'):
        _source_add_line(entry)
    reason = 'ASCII' if name == 'bad-name' else 'identifier'
    with pytest.raises(TestbenchSpecError, match=reason):
        source_element_names(entry)


def test_names_do_not_construct_elements_or_mutate_input(monkeypatch):
    from silk.codegen import ElementNames, source_element_names
    def refuse(*args, **kwargs):
        raise AssertionError('Naming must not build sources or waveforms')
    monkeypatch.setattr(sources.SourceBase, '__init__', refuse)
    monkeypatch.setattr(sources.Sine, '__init__', refuse)
    entry = dict(CONTROL, waveform={'type': 'Sine', 'params': {'vampl': .1, 'freq': 1e3}}, array=1)
    before = deepcopy(entry)
    records = source_element_names(entry)
    assert records == (ElementNames('control[0]', 'i(control[0])', ('Vcontrol<0>__silk_wave', 'Vcontrol<0>__silk_dc')),)
    assert entry == before
    with pytest.raises(FrozenInstanceError):
        records[0].designer_name = 'changed'


@pytest.mark.parametrize('wave,fields', [row for row in WAVES if row[0] in ('am', 'sffm')])
@pytest.mark.parametrize('kind,letter', [('VoltageSource', 'V'), ('CurrentSource', 'I')])
def test_entry_names_need_no_simulator_discovery(monkeypatch, wave, fields, kind, letter):
    from silk.circuit import ElementNames, source_element_names
    def refuse():
        raise AssertionError('Naming must not discover a simulator')
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', refuse)
    entry = dict(CONTROL, type=kind, instance='Stim', **fields)
    assert source_element_names(entry) == (
        ElementNames('Stim', 'i(Stim)', (letter + 'Stim__silk_wave', letter + 'Stim__silk_dc')),)


@pytest.mark.parametrize('wave,fields', [row for row in WAVES if row[0] in ('am', 'sffm')])
@pytest.mark.parametrize('kind', ['VoltageSource', 'CurrentSource'])
def test_unknown_version_still_refuses_waveform_emission(tmp_path, monkeypatch, wave, fields, kind):
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: None)
    mod = emitted_module(tmp_path, dict(CONTROL, type=kind, instance='Stim', **fields))
    with pytest.raises(NotSupportedError, match='ngspice.*version unknown.*cannot emit'):
        mod.generate_netlist()


@pytest.mark.parametrize('changes', [
    {'array': 0}, {'array': -1}, {'array': []}, {'array': [2, 2]}, {'array': True},
    {'array': '2'}, {'naming': 'invalid'}, {'type': 'bogus'}, {'instance': None},
    {'instance': 'bad<0>'}, {'instance': ''}, {'instance': 'bad name'},
    {'waveform': {}}, {'waveform': {'type': 'UnknownWave'}},
    {'waveform': {'type': 'Sine', 'params': {}}}, {'pwl': []},
    {'pwl': [0, 1, 2]}, {'pwl': [0, 1, 1, 2], 'waveform': {'type': 'Sine', 'params': {'vampl': 1, 'freq': 1}}},
    {'dc': {'variable': 'level'}},
])
def test_invalid_entry_raises_testbench_spec_error(changes):
    from silk.codegen import source_element_names
    with pytest.raises(TestbenchSpecError):
        source_element_names(dict(CONTROL, **changes))


@pytest.mark.parametrize('entry', [None, [], 'source', {}, {'type': 'VoltageSource', 'instance': 'missing'}])
def test_nonentries_and_missing_fields_are_typed(entry):
    from silk.codegen import source_element_names
    with pytest.raises(TestbenchSpecError):
        source_element_names(entry)
