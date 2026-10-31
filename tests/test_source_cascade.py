"""One additive DC contract across independent sources (TASK-535)."""
import numpy as np
import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.circuit.variable import Variable
from silk.exceptions import CircuitError, NotSupportedError
from silk.circuit.xyce_netlister import XyceNetlister


@pytest.mark.parametrize('cls,letter', [(sources.VoltageSource, 'V'), (sources.CurrentSource, 'I')])
@pytest.mark.parametrize('slot', ['pulse', 'sine', 'pwl', 'waveform'])
@pytest.mark.parametrize('dc', [0, .2, '200m'])
def test_every_waveform_has_separate_additive_dc_card(cls, letter, slot, dc):
    waves = dict(pulse=sources.Pulse(.5, 1, tr='1n', tf='1n', pw='1u', per='2u'),
                 sine=sources.Sine(.5, .3, 1e3),
                 pwl=sources.PWL([(0, .5), (1e-6, 1)]),
                 waveform=sources.PWL([(0, .5), (1e-6, 1)]))
    src = cls('stim', 'out', '0', dc=dc, ac=1, **{slot: waves[slot]})
    cards = [line.split() for line in src.to_spice().splitlines()]
    assert len(cards) == 2
    suffix = '__silk_pwl' if slot == 'pwl' else '__silk_wave'
    assert [c[0] for c in cards] == [letter + 'stim' + suffix, letter + 'stim__silk_dc']
    assert tuple(c[0] for c in cards) == src.physical_instance_names()
    assert 'DC' not in cards[0] and 'AC' not in cards[0]
    assert cards[1][3] == 'DC' and cards[1][-2:] == ['AC', '1']
    if letter == 'V':
        assert cards[0][1:3] == ['out', '__silk_src_Vstim']
        assert cards[1][1:3] == ['__silk_src_Vstim', '0']
    else:
        assert cards[0][1:3] == cards[1][1:3] == ['out', '0']
    np.testing.assert_allclose(src.to_numpy([0]), [.5 + (0 if dc == 0 else .2)])


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
def test_constant_variable_and_expression_samples_resolve_live(cls):
    value = Variable('level', value=.2)
    src = cls('bias', 'out', '0', dc=value)
    np.testing.assert_allclose(src.to_numpy([0, 1]), [.2, .2])
    src.dc = value * 2
    value.value = .3
    np.testing.assert_allclose(src.to_numpy([0]), [.6])


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
def test_multiple_waveforms_refuse_at_construction_and_emission(cls):
    wave = sources.Sine(0, 1, 1e3)
    with pytest.raises(CircuitError, match='stim.*pwl.*sine'):
        cls('stim', 'out', '0', pwl=[(0, 1), (1, 2)], sine=wave)
    src = cls('stim', 'out', '0', sine=wave)
    src.__dict__['pwl'] = sources.PWL([(0, 1), (1, 2)])
    with pytest.raises(CircuitError, match='stim.*pwl.*sine'):
        src.to_spice()


def test_placed_identity_update_refuses_without_partial_mutation(tmp_path, pdk):
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    src = sources.VoltageSource('stim', 'out', '0', dc=.2)
    mod.add_source(src)
    with pytest.raises(CircuitError, match='placed'):
        src.sine = sources.Sine(0, 1, 1e3)
    with pytest.raises(CircuitError, match='placed'):
        mod.update_source('stim', ['dc', 'sine'], [.9, sources.Sine(0, 1, 1e3)])
    assert src.dc == .2 and src.sine is None
    with pytest.raises(CircuitError, match='placed'):
        src.name = 'renamed'
    src.dc = .7
    assert src.physical_instance_names() == ('Vstim',)


@pytest.mark.parametrize('name,want', [('bias[0]', 'bias[0]'), ('a.b', 'a_db'),
                                      ('a:b', 'a_cb'), ('a_db', 'a_edb'),
                                      ('电源[2]', '电源[2]'), ('e\u0301', 'e\u0301'), ('é', 'é')])
def test_private_node_uses_readable_injective_designer_component(name, want):
    from silk.sources._layout import _escape_component
    assert _escape_component(name) == want
    if name not in ('bias[0]', 'a_db'):
        with pytest.raises(CircuitError, match='ASCII'):
            sources.VoltageSource(name, 'out', '0', sine=sources.Sine(0, 1, 1e3))
        return
    src = sources.VoltageSource(name, 'out', '0', sine=sources.Sine(0, 1, 1e3))
    assert src.to_spice().splitlines()[0].split()[2] == '__silk_src_V' + want


@pytest.mark.parametrize('net', ['__silk_x', '__SILK_src_test', 'xdut.__silk_x'])
def test_reserved_net_namespace_refuses_user_terminals(net):
    reason = 'hierarchy levels' if '.' in net else 'reserved'
    with pytest.raises(ValueError, match=reason):
        sources.VoltageSource('stim', net, '0')


def test_sine_default_preserves_positional_meaning_and_delay_phase():
    wave = sources.Sine(vampl=.3, freq=1e3, td=1e-6, phase=30)
    np.testing.assert_allclose(wave.to_numpy([0, .5e-6, 1e-6]), [.15, .15, .15])
    assert sources.Sine(.5, .3, 1e3).voffset == .5
    with pytest.raises(TypeError, match='vampl'):
        sources.Sine(freq=1e3)
    with pytest.raises(TypeError, match='freq'):
        sources.Sine(vampl=.3)


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
@pytest.mark.parametrize('phase', [30, '{phase}'])
def test_xyce_refuses_sffm_phase_even_with_nonzero_dc(cls, phase):
    src = cls('stim', 'out', '0', dc=.2, sffm=sources.SFFM(.5, .3, 1e3, 2, 10, phasec=phase))
    with pytest.raises(NotSupportedError, match='stim.*phase'):
        XyceNetlister(None).generate_sources({'stim': src})


def test_net_objects_and_renames_cannot_enter_private_namespace(tmp_path, pdk):
    from silk.circuit.net import Net
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    with pytest.raises(ValueError, match='reserved'):
        Net('__SiLk_src_Vstim', mod)
    net = mod.add_net('out')
    with pytest.raises(ValueError, match='reserved'):
        net.name = '__silk_src_Vstim'
    assert net.name == 'out'


def test_opaque_authored_reserved_tokens_refuse_but_include_boundary_is_external():
    from silk.circuit.spice_netlister import PySpiceNetlister
    n = PySpiceNetlister(None)
    with pytest.raises(CircuitError, match='reserved'):
        n.generate_custom_lines(['Rbad __silk_src_Vstim 0 1k'])
    assert '.include' in n.generate_custom_lines(['.include "__silk_external.lib"'])


def test_atomic_same_layout_waveform_replacement_and_pwl_normalization(tmp_path, pdk):
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    src = sources.VoltageSource('stim', 'out', '0', dc=.2, sine=sources.Sine(0, 1, 1))
    mod.add_source(src)
    pulse = sources.Pulse(.5, 1, tr=.1, tf=.1, pw=1, per=2)
    mod.update_source('stim', ['dc', 'sine', 'pulse'], [.3, None, pulse])
    assert src.sine is None and src.pulse is pulse and src.dc == .3
    assert src.physical_instance_names() == ('Vstim__silk_wave', 'Vstim__silk_dc')
    with pytest.raises(CircuitError, match='conflicting'):
        mod.update_source('stim', ['dc', 'sine'], [.9, sources.Sine(0, 1, 1)])
    assert src.dc == .3
    loose = sources.VoltageSource('loose', 'out', '0')
    loose.pwl = [(0, .5), (1, 1)]
    assert 'PWL(' in loose.to_spice()


@pytest.mark.parametrize('nested', [False, True])
def test_reserved_net_validation_repeats_at_emission(tmp_path, pdk, nested):
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    child = Module('sky130', str(tmp_path), 'child.cir', pdk_details=pdk, module_name='child')
    child.define_module_ports({})
    target = child if nested else mod
    target.add_source(sources.VoltageSource('stim', 'out', '0', sine=sources.Sine(0, 1, 1)))
    if nested:
        mod.add_module(child, 'child', nets={})
        target = mod.module_dict['child']['device']
    # Imported or directly mutated graph state must not bypass the boundary.
    net = target.net_dict['out']
    net.__dict__['name'] = '__SILK_src_Vstim'
    with pytest.raises(ValueError, match='reserved'):
        mod.generate_netlist()


def test_rejected_batch_terminal_update_keeps_bindings_and_values(tmp_path, pdk):
    mod = Module('sky130', str(tmp_path), 'top.cir', pdk_details=pdk)
    src = sources.VoltageSource('stim', 'out', '0', dc=.2)
    mod.add_source(src)
    before = {name: [(p.instance, p.port) for p in net.pins] for name, net in mod.net_dict.items()}
    with pytest.raises(ValueError, match='reserved'):
        mod.update_source('stim', ['positive', 'negative', 'dc'], ['other', '__silk_bad', .3])
    assert (src.positive, src.negative, src.dc) == ('out', '0', .2)
    assert {name: [(p.instance, p.port) for p in net.pins] for name, net in mod.net_dict.items()} == before


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
def test_strict_net_batch_is_atomic_and_success_can_rewire(tmp_path, cls):
    from silk.exceptions import UndeclaredNetError
    mod = Module(None, str(tmp_path), 'strict.cir', None, strict_nets=True)
    for name in ('a', 'b', '0'):
        mod.add_net(name)
    wave = sources.Sine(0, .1, 1e3)
    replacement = sources.Sine(0, .3, 2e3)
    src = cls('stim', 'a', '0', dc=.2, sine=wave)
    mod.add_source(src)
    pins = {name: list(net.pins) for name, net in mod.net_dict.items()}
    reverse = dict(mod._element_mgr._pin_to_net)
    inventory = src.physical_layout()
    with pytest.raises(UndeclaredNetError, match='undeclared'):
        mod.update_source('stim', ['positive', 'negative', 'dc', 'sine'],
                          ['b', 'undeclared', .9, replacement])
    assert (src.positive, src.negative, src.dc, src.sine) == ('a', '0', .2, wave)
    assert {name: list(net.pins) for name, net in mod.net_dict.items()} == pins
    assert mod._element_mgr._pin_to_net == reverse
    assert src.physical_layout() == inventory
    mod.update_source('stim', ['positive', 'negative', 'dc', 'sine'],
                      ['b', 'a', .9, replacement])
    assert (src.positive, src.negative, src.dc, src.sine) == ('b', 'a', .9, replacement)
    assert mod._element_mgr._pin_to_net == {('stim', 'positive'): 'b', ('stim', 'negative'): 'a'}
    assert [(p.instance, p.port) for p in mod.net_dict['b'].pins] == [('stim', 'positive')]
    assert [(p.instance, p.port) for p in mod.net_dict['a'].pins] == [('stim', 'negative')]
    assert not mod.net_dict['0'].pins
    assert src.physical_layout() == inventory


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
@pytest.mark.parametrize('slot', ['am', 'sffm', 'generic_am', 'generic_sffm'])
@pytest.mark.parametrize('version', [None, '41', 'unknown'])
def test_ngspice_modulated_sources_refuse_unverified_versions(monkeypatch, cls, slot, version):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: version)
    wave = sources.AM(.3, .5, 10e3, 100e3) if slot.endswith('am') else sources.SFFM(.5, .3, 100e3, 2, 10e3)
    src = cls('stim', 'out', '0', dc=.2,
              **{'waveform' if slot.startswith('generic') else slot: wave})
    with pytest.raises(NotSupportedError) as exc:
        PySpiceNetlister(None).generate_sources({'stim': src})
    message = str(exc.value)
    assert (version or 'unknown') in message
    assert 'stim' in message and type(wave).__name__ in message


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
@pytest.mark.parametrize('phase', [30, '{phase}', None, float('nan'), float('inf')])
def test_ngspice_42_refuses_ignored_or_unresolved_am_phase(monkeypatch, cls, phase):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    src = cls('stim', 'out', '0', am=sources.AM(.3, .5, 10e3, 100e3, phases=phase))
    with pytest.raises(NotSupportedError, match='42.*stim.*phase'):
        PySpiceNetlister(None).generate_sources({'stim': src})


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
def test_ngspice_42_keeps_measured_order_and_resolves_am_phase_each_emission(monkeypatch, cls):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '42')
    phase = Variable('phase', value=0)
    am = cls('amstim', 'out', '0', waveform=sources.AM(.3, .5, 10e3, 100e3, phases=phase))
    fm = cls('fmstim', 'out', '0', sffm=sources.SFFM(.5, .3, 100e3, 2, 10e3, 30, 40))
    netlister = PySpiceNetlister(None)
    deck = netlister.generate_sources({'amstim': am, 'fmstim': fm})
    assert 'AM(0.3 0.5 10000.0 100000.0 0)' in deck
    assert 'SFFM(0.5 0.3 100000.0 2 10000.0 30 40)' in deck
    phase.value = 30
    with pytest.raises(NotSupportedError, match='42.*amstim.*phase'):
        netlister.generate_sources({'amstim': am})
    phase.value = '0'
    assert 'AM(' in netlister.generate_sources({'amstim': am})


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
@pytest.mark.parametrize('slot', ['am', 'waveform'])
@pytest.mark.parametrize('version', ['42', '42+', '42.1', '43', '45.2', '46', '47', '47+', '48'])
def test_am_emits_selected_release_mapping(monkeypatch, cls, slot, version):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: version)
    src = cls('stim', 'out', '0', dc=.2,
              **{slot: sources.AM(.3, .5, 10e3, 100e3, td='200n')})
    deck = PySpiceNetlister(None).generate_sources({'stim': src})
    if version.startswith('42'):
        assert 'AM(0.3 0.5 10000.0 100000.0 200n)' in deck
    else:
        assert 'AM(0 0.15 0.3 10000.0 100000.0 200n 0 0)' in deck
    assert '__silk_dc' in deck and 'DC 0.2' in deck


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
def test_new_am_resolves_si_variables_expressions_and_phase(monkeypatch, cls):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '47')
    amplitude = Variable('amplitude', value=.3)
    wave = sources.AM(amplitude, '500m', '10k', '100k', phases=30)
    src = cls('stim', 'out', '0', am=wave)
    renderer = PySpiceNetlister(None)
    assert 'AM(0 0.15 0.3 10k 100k 0 30 0)' in renderer.generate_sources({'stim': src})
    amplitude.value = .6
    assert 'AM(0 0.3 0.6 10k 100k 0 30 0)' in renderer.generate_sources({'stim': src})
    wave.va, wave.vo = '{amplitude}', '{offset + 1}'
    assert 'AM(0 {(amplitude)*(offset + 1)} {amplitude} 10k 100k 0 30 0)' in renderer.generate_sources({'stim': src})


@pytest.mark.parametrize('version', ['42+', '42.1', '43', '45.2', '47', '48'])
def test_sffm_retains_refusal_with_waveform_reason(monkeypatch, version):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: version)
    src = sources.VoltageSource('stim', 'out', '0', sffm=sources.SFFM(.5, .3, 100e3, 2, 10e3))
    with pytest.raises(NotSupportedError, match='start SFFM at 0 and clamp the modulation depth'):
        PySpiceNetlister(None).generate_sources({'stim': src})


@pytest.mark.parametrize('target', ['waveform_class', 'source_class', 'waveform_instance', 'source_instance'])
@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
def test_am_mapping_preserves_custom_emitters(monkeypatch, cls, target):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '47')

    class CustomAM(sources.AM):
        def to_spice(self):
            return 'PWL(0 7 1u 7)'

    class CustomSource(cls):
        def to_spice(self):
            return 'Vcustom out 0 PWL(0 7 1u 7)\n'

    wave = (CustomAM if target == 'waveform_class' else sources.AM)(.3, .5, 1e4, 1e5)
    src = (CustomSource if target == 'source_class' else cls)('stim', 'out', '0', waveform=wave)
    if target == 'waveform_instance':
        wave.to_spice = lambda: 'PWL(0 7 1u 7)'
    if target == 'source_instance':
        src.to_spice = lambda: 'Vcustom out 0 PWL(0 7 1u 7)\n'
    authored = src.to_spice()
    assert authored in PySpiceNetlister(None).generate_sources({'stim': src})


def test_am_mapping_supports_inherited_builtin_emitters(monkeypatch):
    from silk.circuit.spice_netlister import PySpiceNetlister
    from silk.simulation import _ngspice_version
    monkeypatch.setattr(_ngspice_version, 'ngspice_version', lambda: '47')

    class InheritedAM(sources.AM):
        pass

    class InheritedSource(sources.VoltageSource):
        pass

    src = InheritedSource('stim', 'out', '0', am=InheritedAM(.3, .5, 1e4, 1e5))
    assert 'AM(0 0.15 0.3 10000.0 100000.0 0 0 0)' in PySpiceNetlister(None).generate_sources({'stim': src})


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
@pytest.mark.parametrize('reverse', [False, True])
def test_pwl_replacement_and_timing_batch_commits_to_replacement(tmp_path, cls, reverse):
    mod = Module(None, str(tmp_path), 'top.cir', None)
    old = sources.PWL([(0, 0), (1, 1)], td=1)
    src = cls('stim', 'out', '0', pwl=old)
    mod.add_source(src)
    replacement = sources.PWL([(0, 0), (1, 2)], td=5, r=0)
    changes = [('pwl', replacement), ('pwl_td', 2), ('pwl_r', None)]
    if reverse:
        changes.reverse()
    mod.update_source('stim', [key for key, _ in changes], [value for _, value in changes])
    assert src.pwl is replacement and old.td == 1
    assert (replacement.td, replacement.r) == (2, None)
    assert src.to_numpy([2.5])[0] == pytest.approx(1)
    assert 'td=2' in src.to_spice() and 'r=' not in src.to_spice()


@pytest.mark.parametrize('cls', [sources.VoltageSource, sources.CurrentSource])
@pytest.mark.parametrize('bad', ['reserved', 'undeclared', 'card', 'conflict'])
def test_rejected_pwl_timing_batch_keeps_nested_waveform_and_bindings(tmp_path, cls, bad):
    mod = Module(None, str(tmp_path), 'top.cir', None, strict_nets=True)
    for name in ('out', 'other', '0'):
        mod.add_net(name)
    wave = sources.PWL([(0, 0), (1, 1), (2, 0)], td=1, r=0)
    replacement = sources.PWL(wave.points, td=3, r=1)
    src = cls('stim', 'out', '0', dc=.2, pwl=wave)
    mod.add_source(src)
    before = src.to_spice()
    pins = {name: list(net.pins) for name, net in mod.net_dict.items()}
    reverse = dict(mod._element_mgr._pin_to_net)
    changes = dict(pwl_td=2, pwl_r=None, dc=.9, positive='other')
    if bad == 'reserved':
        changes['negative'] = '__silk_bad'
    elif bad == 'undeclared':
        changes['negative'] = 'missing'
    elif bad == 'card':
        changes['ac'] = '1\nBAD'
    else:
        changes['sine'] = sources.Sine(0, 1, 1)
    # Both existing and incoming shared waveforms must survive refusal untouched.
    for replace in (False, True):
        if replace:
            changes['pwl'] = replacement
        with pytest.raises((ValueError, CircuitError)):
            mod.update_source('stim', list(changes), list(changes.values()))
        assert src.pwl is wave
        assert (wave.td, wave.r, replacement.td, replacement.r) == (1, 0, 3, 1)
        assert (src.pwl_td, src.pwl_r, src.dc, src.positive, src.negative) == (1, 0, .2, 'out', '0')
        assert src.to_spice() == before
        assert {name: list(net.pins) for name, net in mod.net_dict.items()} == pins
        assert mod._element_mgr._pin_to_net == reverse
