"""Built-element names report actual cards, preserving logical source probes."""
from pathlib import Path

import pytest

from silk import sources
from silk.circuit.emitted_names import emitted_instance_name
from silk.elements import elements
from silk.exceptions import SimulationError
from silk.simulation import _ngspice_version
from tests.test_source_element_names import KINDS, NAMINGS, WAVES, emitted_module, measured_ngspice_version


@pytest.mark.parametrize('kind,letter,readable,fields', KINDS)
@pytest.mark.parametrize('array,names,components', NAMINGS)
def test_built_sources_match_entry_names_and_emitted_deck(tmp_path, kind, letter, readable, fields, array, names, components):
    from silk.circuit import ElementNames, element_names, source_element_names
    entry = {'type': kind, 'instance': 'Stim', **fields, **array}
    mod = emitted_module(tmp_path, entry)
    records = tuple(element_names(mod.sig_sources_dict[name]) for name in names)
    assert records == source_element_names(entry)
    assert records == tuple(ElementNames(n, f'i({n})' if readable else None, (letter + c,)) for n, c in zip(names, components))
    mod_deck_path = mod.generate_netlist()
    deck = Path(mod_deck_path).read_text()
    for record in records:
        assert any(line.startswith(record.card_names[0] + ' ') for line in deck.splitlines())


@pytest.mark.parametrize('kind,letter', [('VoltageSource', 'V'), ('CurrentSource', 'I')])
@pytest.mark.parametrize('wave,fields', WAVES)
@pytest.mark.parametrize('array,names,components', NAMINGS)
def test_built_waveforms_report_all_cards_and_keep_legacy_instance_name(tmp_path, measured_ngspice_version, kind, letter, wave, fields, array, names, components):
    from silk.circuit import element_names, source_element_names
    entry = {'type': kind, 'instance': 'Stim', 'positive': 'out', 'negative': '0', 'dc': .2, **fields, **array}
    mod = emitted_module(tmp_path, entry)
    records = tuple(element_names(mod.sig_sources_dict[name]) for name in names)
    assert records == source_element_names(entry)
    mod_deck_path = mod.generate_netlist()
    deck = Path(mod_deck_path).read_text()
    for record, component in zip(records, components):
        src = mod.sig_sources_dict[record.designer_name]
        assert emitted_instance_name(src, src.name) == letter + component
        assert record.card_names == src.physical_instance_names()
        for card in record.card_names:
            assert any(line.startswith(card + ' ') for line in deck.splitlines())


def test_built_generic_waveform_reports_wave_suffix_without_rendering():
    from silk.circuit import ElementNames, element_names
    class Unrenderable:
        def to_spice(self):
            raise AssertionError('Names must not render a waveform')
    src = sources.VoltageSource('Stim[0]', 'out', '0', waveform=Unrenderable())
    assert element_names(src) == ElementNames('Stim[0]', 'i(Stim[0])', ('VStim<0>__silk_wave', 'VStim<0>__silk_dc'))


@pytest.mark.parametrize('wave,fields', [row for row in WAVES if row[0] in ('am', 'sffm')])
@pytest.mark.parametrize('kind,letter', [('VoltageSource', 'V'), ('CurrentSource', 'I')])
def test_built_names_need_no_simulator_discovery(tmp_path, monkeypatch, wave, fields, kind, letter):
    from silk.circuit import ElementNames, element_names
    def refuse():
        raise AssertionError('Naming must not discover a simulator')
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', refuse)
    entry = {'type': kind, 'instance': 'Stim', 'positive': 'out', 'negative': '0', **fields}
    mod = emitted_module(tmp_path, entry)
    assert element_names(mod.sig_sources_dict['Stim']) == ElementNames(
        'Stim', 'i(Stim)', (letter + 'Stim__silk_wave', letter + 'Stim__silk_dc'))


@pytest.mark.parametrize('kind,letter,readable,fields', [row for row in KINDS if row[2] and row[1] != 'I'])
def test_literal_colon_source_name_is_refused(tmp_path, kind, letter, readable, fields):
    from silk.exceptions import CircuitError
    with pytest.raises(CircuitError, match='hierarchy levels'):
        getattr(sources, kind)('drive:part', **fields)


@pytest.mark.parametrize('factory,letter', [(elements.resistor, 'R'), (elements.capacitor, 'C'), (elements.inductor, 'L')])
@pytest.mark.parametrize('name,component', [('part', 'part'), ('part[2]', 'part<2>'), ('part_2', 'part_2')])
def test_built_passive_cards_match_emission(factory, letter, name, component):
    from silk.circuit import element_names
    part = factory(name, p='out', n='0', value=1e-3)
    record = element_names(part)
    assert record.designer_name == name
    assert record.card_names == (letter + component,)
    assert part.to_spice().split()[0] == record.card_names[0]
    if letter == 'L':
        assert record.current_probe == f'i(L{component})'
    else:
        assert record.current_probe is None


def test_pdk_device_has_cards_but_no_branch_probe(pdk):
    from silk.circuit import element_names
    device = pdk.nmos1p8('M[2]', d='out', g='in', s='0', b='0', w=1, l=.15)
    record = element_names(device)
    assert record.designer_name == 'M[2]'
    assert record.card_names == (emitted_instance_name(device, device.name),)
    assert record.current_probe is None


@pytest.mark.parametrize('element', [None, {}, 'source'])
def test_unsupported_objects_raise_type_error(element):
    from silk.circuit import element_names
    with pytest.raises(TypeError, match='element'):
        element_names(element)
