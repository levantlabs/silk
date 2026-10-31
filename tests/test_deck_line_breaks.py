"""A line break inside a caller's string never becomes a new deck line (TASK-593).

silk writes each parameter value, source value, option, analysis field, path
and corner name into ONE card. A string holding any character
``str.splitlines()`` splits on used to end that card early, and the rest of
the string ran as deck text: an injected ``.control`` block executes under
``ngspice -b``, and ngspice's ``shell`` command runs a program. Every such
string is now refused with :class:`silk.exceptions.DeckTextError`, naming
where it was written, at construction where silk validates values and again
when the card is rendered.

Simulator-free throughout. No test here runs a deck, and no refused value is
ever written to disk.
"""
import os

import pytest

from silk import sources
from silk.elements import elements
from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.elements import passives
from silk.exceptions import ConfigurationError, DeckTextError
from silk.simulation import AC, Noise, Transient, analysis
from silk.utils.deck_text import (
    LINE_BREAKS, refuse_line_breaks, validate_corner_name,
)

#: A newline, a carriage return and a non-ASCII separator (U+2028).
BREAKS = ['\n', '\r', ' ']
PAYLOAD = '{}.control{}echo injected{}.endc'


def payload(brk, head='1k'):
    return head + PAYLOAD.format(brk, brk, brk)


def _module(pdk, tmp_path, tag='top', **kwargs):
    return Module('PDK_skywater130', str(tmp_path / tag), f'{tag}.cir',
                  pdk_details=pdk, **kwargs)


def _files(root):
    return {p for p in root.rglob('*') if p.is_file()}


def _no_deck_written(tmp_path, before=frozenset()):
    """No file of any kind appeared under *tmp_path* (not only decks)."""
    return _files(tmp_path) <= set(before)


@pytest.fixture
def no_subprocess(monkeypatch):
    """Fail if anything tries to launch a process (a simulator)."""
    import subprocess

    def refuse(*args, **kwargs):
        raise AssertionError(f'a process was launched: {args!r}')

    monkeypatch.setattr(subprocess, 'run', refuse)
    monkeypatch.setattr(subprocess, 'Popen', refuse)


class _Text:
    """A non-string value whose text form holds a line break."""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text

    __format__ = lambda self, spec: self.text


# ---------------------------------------------------------------------------
# The shared validator
# ---------------------------------------------------------------------------

def test_line_break_set_is_exactly_what_splitlines_splits_on():
    """The constant is derived from Python's own definition, not a guess."""
    split = {chr(c) for c in range(0x110000)
             if len(('a' + chr(c) + 'b').splitlines()) == 2}
    assert LINE_BREAKS == frozenset(split)


def test_error_is_a_configuration_error_and_a_value_error():
    assert issubclass(DeckTextError, ConfigurationError)
    assert issubclass(DeckTextError, ValueError)


@pytest.mark.parametrize('brk', sorted(LINE_BREAKS))
def test_refuse_line_breaks_refuses_every_separator(brk):
    with pytest.raises(DeckTextError, match="'thing'"):
        refuse_line_breaks('1k' + brk, who="'thing'")


@pytest.mark.parametrize('value', ['1k', '{rload}', 'rload*2', 'a b', 1.5, 3, None])
def test_refuse_line_breaks_passes_ordinary_values(value):
    assert refuse_line_breaks(value, who='x') is value


# ---------------------------------------------------------------------------
# AC1: device and element parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('brk', BREAKS)
def test_ideal_element_value_refused_at_construction(brk):
    with pytest.raises(DeckTextError, match=r"'r1'.*'value'"):
        elements.resistor('r1', p='a', n='0', value=payload(brk))


@pytest.mark.parametrize('brk', BREAKS)
def test_ideal_element_value_set_later_refused(brk):
    r = elements.resistor('r1', p='a', n='0', value='1k')
    with pytest.raises(DeckTextError, match=r"'r1'.*'value'"):
        r.value = payload(brk)


@pytest.mark.parametrize('brk', BREAKS)
def test_ideal_element_value_written_into_params_refused_at_render(brk):
    r = elements.resistor('r1', p='a', n='0', value='1k')
    r.params['value'] = payload(brk)
    with pytest.raises(DeckTextError, match=r"'r1'.*'value'"):
        r.to_spice()


@pytest.mark.parametrize('brk', BREAKS)
def test_pdk_device_parameter_refused_at_construction(pdk, brk):
    with pytest.raises(DeckTextError, match=r"'mn'.*'w'"):
        pdk.nmos1p8('mn', d='d', g='g', s='0', b='0', w=payload(brk, '1'), l=0.15)


@pytest.mark.parametrize('brk', BREAKS)
def test_pdk_device_parameter_set_later_refused(pdk, brk):
    mn = pdk.nmos1p8('mn', d='d', g='g', s='0', b='0', w=1, l=0.15)
    with pytest.raises(DeckTextError, match=r"'mn'.*'w'"):
        mn.w = payload(brk, '1')
    mn.params['w'] = payload(brk, '1')
    with pytest.raises(DeckTextError, match=r"'mn'.*'w'"):
        mn.to_spice()


@pytest.mark.parametrize('brk', BREAKS)
def test_legacy_add_element_parameter_refused(pdk, tmp_path, brk):
    m = _module(pdk, tmp_path)
    with pytest.warns(DeprecationWarning):
        try:
            m.add_element(pdk.nmos1p8, 'mn',
                          nets={'d': 'd', 'g': 'g', 's': '0', 'b': '0'},
                          parameters={'w': payload(brk, '1'), 'l': 0.15})
        except DeckTextError as exc:
            assert "'mn'" in str(exc) and "'w'" in str(exc)
            return
    with pytest.raises(DeckTextError, match=r"'mn'.*'w'"):
        m.generate_netlist()
    assert _no_deck_written(tmp_path)


@pytest.mark.parametrize('brk', BREAKS)
def test_passive_shim_refused_at_construction_and_alias(brk):
    with pytest.raises(DeckTextError, match=r"resistance"):
        passives.Resistor('r1', 'a', 'b', payload(brk))
    r = passives.Resistor('r1', 'a', 'b', '1k')
    with pytest.raises(DeckTextError, match=r"resistance"):
        r.resistance = payload(brk)


def test_variable_value_refused_at_render():
    from silk.circuit.variable import Variable
    v = Variable('rv', value='1k')
    r = elements.resistor('r1', p='a', n='0', value=v)
    assert r.to_spice() == 'Rr1 a 0 1k\n'
    v.value = payload('\n')
    with pytest.raises(DeckTextError, match=r"'r1'.*'value'"):
        r.to_spice()


def test_pdk_catalog_strings_refused_in_the_device_line_writer():
    from silk.utils._utils import render_pdk_device_line
    with pytest.raises(DeckTextError):
        render_pdk_device_line('XM', 'mn', ['d'], 'model\n.endc', [])
    with pytest.raises(DeckTextError, match=r"'mn'.*'w'"):
        render_pdk_device_line('XM', 'mn', ['d'], 'model', [('w', '1\n', 1.0)])


@pytest.mark.parametrize('value, card', [
    ('1k', 'Rr1 a 0 1k\n'),
    ('{rload}', 'Rr1 a 0 {rload}\n'),
    ("'rload*2'", "Rr1 a 0 'rload*2'\n"),
    (1000.0, 'Rr1 a 0 1k\n'),
])
def test_ordinary_ideal_values_emit_unchanged(value, card):
    assert elements.resistor('r1', p='a', n='0', value=value).to_spice() == card


def test_ordinary_pdk_values_emit_unchanged(pdk):
    mn = pdk.nmos1p8('mn', d='d', g='g', s='0', b='0', w='{wn}', l=0.15)
    assert mn.to_spice() == (
        'XMmn d g 0 0 sky130_fd_pr__nfet_01v8 m=1 w={wn} l=0.15\n')


# ---------------------------------------------------------------------------
# AC2: every other path that writes a caller string into a card
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('brk', BREAKS)
@pytest.mark.parametrize('field', ['dc', 'ac'])
def test_source_dc_ac_refused(brk, field):
    with pytest.raises(DeckTextError, match=rf"'vs'.*'{field}'"):
        sources.VoltageSource('vs', positive='a', negative='0',
                              **{field: payload(brk, '1')})
    vs = sources.VoltageSource('vs', positive='a', negative='0', dc=1)
    with pytest.raises(DeckTextError, match=rf"'vs'.*'{field}'"):
        setattr(vs, field, payload(brk, '1'))
    object.__setattr__(vs, field, payload(brk, '1'))
    with pytest.raises(DeckTextError, match=rf"'vs'.*'{field}'"):
        vs.to_spice()


@pytest.mark.parametrize('brk', BREAKS)
def test_waveform_fields_refused(brk):
    with pytest.raises(DeckTextError, match=r'Pulse.*v1'):
        sources.Pulse(v1=payload(brk, '0'), v2=1, td=0, tr='1n', tf='1n',
                      pw='5n', per='10n')
    sine = sources.Sine(voffset=0, vampl=1, freq='1Meg')
    with pytest.raises(DeckTextError, match=r'Sine.*vampl'):
        sine.vampl = payload(brk, '1')
    object.__setattr__(sine, 'vampl', payload(brk, '1'))
    with pytest.raises(DeckTextError, match=r'Sine.*vampl'):
        sine.to_spice()


@pytest.mark.parametrize('brk', BREAKS)
def test_value_with_trailing_break_is_refused_before_the_number_parser(brk):
    """The SPICE number parser strips its input, so '1n\\n' used to pass its
    check and be emitted unstripped. The raw string is now checked first."""
    with pytest.raises(DeckTextError, match=r'Pulse.*tr'):
        sources.Pulse(v1=0, v2=1, td=0, tr='1n' + brk, tf='1n', pw='5n', per='10n')
    with pytest.raises(DeckTextError):
        Transient(step='1n', stop='100n' + brk)


@pytest.mark.parametrize('brk', BREAKS)
def test_pwl_fields_refused(brk):
    from silk.circuit.variable import Variable
    with pytest.raises(DeckTextError, match=r'PWL.*td'):
        sources.PWL([(0, 0), (1e-9, 1)], td=payload(brk, '0'))
    t = Variable('tv', value='1n')
    pwl = sources.PWL([(0, 0), (t, 1)])
    t.value = '1n' + brk
    with pytest.raises(DeckTextError, match='PWL'):
        pwl.to_spice()


@pytest.mark.parametrize('brk', BREAKS)
def test_controlled_and_behavioral_sources_refused(brk):
    with pytest.raises(DeckTextError, match=r"'e1'.*'gain'"):
        sources.VoltageControlledVoltageSource(
            'e1', out_pos='o', out_neg='0', ctrl_pos='i', ctrl_neg='0',
            gain=payload(brk, '10'))
    with pytest.raises(DeckTextError, match=r"'b1'.*'expression'"):
        sources.BehavioralVoltage('b1', positive='o', negative='0',
                                  expression=payload(brk, 'v(i)'))
    with pytest.raises(DeckTextError, match=r"'f1'.*'ctrl_vname'"):
        sources.CurrentControlledCurrentSource(
            'f1', out_pos='o', out_neg='0', gain=2,
            ctrl_vname=payload(brk, 'vmeas'))
    with pytest.raises(DeckTextError, match=r"'g1'.*'gm'"):
        sources.VoltageControlledCurrentSource(
            'g1', out_pos='o', out_neg='0', ctrl_pos='i', ctrl_neg='0',
            gm=payload(brk, '1m'))
    with pytest.raises(DeckTextError, match=r"'h1'.*'transresistance'"):
        sources.CurrentControlledVoltageSource(
            'h1', out_pos='o', out_neg='0', ctrl_vname='vmeas',
            transresistance=payload(brk, '1k'))
    e = sources.VoltageControlledVoltageSource(
        'e1', out_pos='o', out_neg='0', ctrl_pos='i', ctrl_neg='0', gain=10)
    object.__setattr__(e, 'gain', payload(brk, '10'))
    with pytest.raises(DeckTextError, match=r"'e1'.*'gain'"):
        e.to_spice()


@pytest.mark.parametrize('brk', BREAKS)
def test_simulator_option_key_and_value_refused(pdk, tmp_path, brk):
    m = _module(pdk, tmp_path)
    with pytest.raises(DeckTextError, match='option'):
        m.set_simulator_options(**{'reltol': payload(brk, '1e-4')})
    with pytest.raises(DeckTextError, match='option'):
        m.set_simulator_options(**{payload(brk, 'gmin'): 1})
    from silk.simulators import load_simulator_config
    cfg = load_simulator_config('ngspice')
    with pytest.raises(DeckTextError, match='option'):
        cfg.emit_options_line({'reltol': payload(brk, '1e-4')})


@pytest.mark.parametrize('brk', BREAKS)
def test_analysis_fields_refused(brk):
    from silk.simulation.backend import EmitContext, SimulatorBackend
    backend = SimulatorBackend.get('ngspice')
    ctx = EmitContext(resolve_source=lambda n: 'V' + n)
    tran = Transient(step=1e-9, stop=1e-7)
    tran.stop = payload(brk, '1e-7')
    with pytest.raises(DeckTextError, match=r'Transient.*stop'):
        backend.render_analysis(tran, ctx)
    noise = Noise(output='out', ref='0', source='vin', variation='dec',
                  points=10, fstart=1, fstop=1e6)
    noise.output = payload(brk, 'out')
    with pytest.raises(DeckTextError, match=r'Noise.*output'):
        backend.render_analysis(noise, ctx)
    ac = AC(variation='dec', points=10, fstart=1, fstop=1e6)
    ac.variation = 'dec' + brk
    with pytest.raises(DeckTextError, match=r'AC.*variation'):
        backend.render_analysis(ac, ctx)


@pytest.mark.parametrize('brk', BREAKS)
def test_module_name_refused(pdk, tmp_path, brk):
    with pytest.raises(DeckTextError) as exc:
        _module(pdk, tmp_path, module_name='inv' + brk)
    assert type(exc.value) is DeckTextError
    m = _module(pdk, tmp_path, tag='sub', module_name='inv')
    m.define_module_ports({'a': PortDir.INOUT})
    m.add(elements.resistor('r1', p='a', n='0', value='1k'))
    m.module_name = 'inv' + brk
    with pytest.raises(DeckTextError) as exc:
        m.generate_netlist()
    assert type(exc.value) is DeckTextError


@pytest.mark.parametrize('brk', BREAKS)
def test_output_directory_refused(pdk, tmp_path, brk):
    with pytest.raises(DeckTextError, match='dir_folder'):
        Module('PDK_skywater130', str(tmp_path / ('out' + brk + 'x')),
               'top.cir', pdk_details=pdk)
    assert not any(brk in p.name for p in tmp_path.iterdir())


@pytest.mark.parametrize('brk', BREAKS)
def test_output_directory_set_later_refused_before_any_folder(pdk, tmp_path, brk):
    m = _module(pdk, tmp_path)
    m.add(elements.resistor('r1', p='a', n='0', value='1k'))
    m.add_sim_type(analysis.Op())
    bad = str(tmp_path / ('out' + brk + 'x'))
    m.dir_folder = bad
    m.netlistname = 'top.cir'
    with pytest.raises(DeckTextError, match='dir_folder'):
        m.generate_netlist()
    assert not any(brk in p.name for p in tmp_path.iterdir())


@pytest.mark.parametrize('brk', BREAKS)
def test_library_and_include_paths_refused(pdk, tmp_path, brk):
    m = _module(pdk, tmp_path)
    m.add(pdk.nmos1p8('mn', d='d', g='g', s='0', b='0', w=1, l=0.15))
    m.add_sim_type(analysis.Op())
    pdk.library_location = '/x/sky130.lib.spice' + payload(brk, '')
    with pytest.raises(DeckTextError, match='library'):
        m.generate_netlist()
    pdk.library_location = '/x/sky130.lib.spice'
    pdk.extra_includes = ['/x/extra.spice' + brk + '.endc']
    with pytest.raises(DeckTextError, match='include'):
        m.generate_netlist()
    assert _no_deck_written(tmp_path)


@pytest.mark.parametrize('brk', BREAKS)
def test_osdi_path_refused(pdk, brk):
    from silk.circuit.spice_netlister import PySpiceNetlister
    pdk.osdi_files = ['/x/psp.osdi' + brk + 'shell id']
    with pytest.raises(DeckTextError, match='osdi'):
        PySpiceNetlister(pdk)._osdi_lines()


@pytest.mark.parametrize('brk', BREAKS)
def test_save_argument_refused(brk):
    from silk.circuit.spice_netlister import _quote_save_argument
    assert _quote_save_argument('@m.xm1[id]') == '"@m.xm1[id]"'
    with pytest.raises(DeckTextError):
        _quote_save_argument('@x' + brk + 'shell id')


def test_ordinary_source_values_emit_unchanged():
    vs = sources.VoltageSource('vs', positive='a', negative='0', dc='{vdd}', ac=1)
    assert vs.to_spice() == 'Vvs a 0 DC {vdd} AC 1\n'
    e = sources.VoltageControlledVoltageSource(
        'e1', out_pos='o', out_neg='0', ctrl_pos='i', ctrl_neg='0', gain='{av}')
    assert e.to_spice() == 'Ee1 o 0 i 0 {av}\n'
    b = sources.BehavioralVoltage('b1', positive='o', negative='0',
                                  expression='v(i)*2')
    assert b.to_spice() == 'Bb1 o 0 V={v(i)*2}\n'
    p = sources.Pulse(v1=0, v2='{vh}', td=0, tr='1n', tf='1n', pw='5n', per='10n')
    assert p.to_spice() == 'PULSE(0 {vh} 0 1n 1n 5n 10n)'


# ---------------------------------------------------------------------------
# AC3: explicit opaque content stays verbatim
# ---------------------------------------------------------------------------

def test_custom_lines_stay_verbatim(pdk, tmp_path):
    m = _module(pdk, tmp_path)
    m.add(elements.resistor('r1', p='a', n='0', value='1k'))
    m.add_sim_type(analysis.Op())
    m.add_custom_line('.param a=1\n.param b=2')
    m.generate_netlist()
    deck = open(m._run_files.netlist_path).read()
    assert '.param a=1\n.param b=2\n' in deck


# ---------------------------------------------------------------------------
# AC7: corner names
# ---------------------------------------------------------------------------

BAD_CORNERS = ['tt\n.include /etc/hostname', 'tt\r', 'tt ', 'tt\t',
               'tt\x00', '.', '..']


@pytest.mark.parametrize('corner', BAD_CORNERS)
def test_validate_corner_name_refuses(corner):
    with pytest.raises(DeckTextError, match='corner'):
        validate_corner_name(corner)


@pytest.mark.parametrize('corner', ['tt', 'ff', 'mc_tt', 'tt ff', 'a.b',
                                    '...x', 'a/b', 'tt_mm'])
def test_validate_corner_name_passes_other_names(corner):
    """Only line breaks, control characters and the exact names '.' and
    '..' are refused (reviewer ruling). Spaces and separators are left alone:
    the output folder name is percent-encoded."""
    assert validate_corner_name(corner) == corner


@pytest.fixture
def bench(pdk, tmp_path):
    m = Module('PDK_skywater130', str(tmp_path / 'dut'), 'nfet.cir',
               pdk_details=pdk, module_name='nfet')
    m.define_module_ports({'d': PortDir.OUT, 'g': PortDir.IN,
                           's': PortDir.GROUND, 'b': PortDir.GROUND})
    m.add(pdk.nmos1p8('M1', d='d', g='g', s='s', b='b', w=1.0, l=0.15))
    tb = m.create_test('op')
    tb.add_module_under_test({'d': 'vd', 'g': 'vg', 's': '0', 'b': '0'})
    tb.add(sources.vdc('vd', positive='vd', negative='0', value=1.0))
    tb.add(sources.vdc('vg', positive='vg', negative='0', value=1.0))
    tb.add_sim_type(analysis.Op())
    tb.add_metric('one', lambda ckt: 1.0)
    return m, tb


@pytest.mark.parametrize('corners', [False, True])
@pytest.mark.parametrize('corner', BAD_CORNERS)
def test_corner_refused_through_generate_netlist(bench, tmp_path, corner, corners):
    """Both with the PDK's declared corners and with none declared (where
    any name used to be written as the .lib section)."""
    m, tb = bench
    if not corners:
        tb.netlist.pdk_details.corners = {}
    with pytest.raises(DeckTextError, match='corner'):
        tb.netlist.generate_netlist(corner=corner)
    assert _no_deck_written(tmp_path)


@pytest.mark.parametrize('corner', BAD_CORNERS)
def test_corner_refused_through_run_and_run_all_tests(bench, tmp_path, corner,
                                                     monkeypatch):
    m, tb = bench
    tb.netlist.pdk_details.corners = {}

    def no_simulator(*args, **kwargs):
        raise AssertionError('a simulator was reached')

    monkeypatch.setattr(type(tb), '_simulate_run', no_simulator)
    with pytest.raises(DeckTextError, match='corner'):
        tb.run(corner=corner)
    with pytest.raises(DeckTextError, match='corner'):
        m.run_all_tests(corner=corner)
    with pytest.raises(DeckTextError, match='corner'):
        tb.set_corner(corner)
    with pytest.raises(DeckTextError, match='corner'):
        tb.set_corners(['tt', corner])
    assert _no_deck_written(tmp_path)
    assert not any(p.is_dir() and p.name in ('.', '..', corner)
                   for p in tmp_path.rglob('*'))


@pytest.mark.parametrize('corner', BAD_CORNERS)
def test_corner_refused_in_lib_section_and_output_folder(bench, corner):
    """The last defences: the .lib line writer every deck uses (including the
    parameter preflight and Xyce decks) and the corner output folder."""
    m, tb = bench
    tb.netlist.pdk_details.corners = {}
    with pytest.raises(DeckTextError, match='corner'):
        tb.netlist.netlister.generate_pdk_include(corner=corner)
    with pytest.raises(DeckTextError, match='corner'):
        tb._output_paths(corner_subdir=corner)
    with pytest.raises(DeckTextError, match='corner'):
        tb._output_paths(corner_subdir=(corner, 'mc_0000'))


def test_ordinary_corner_still_writes_the_lib_section(bench):
    m, tb = bench
    text = tb.netlist.netlister.generate_pdk_include(corner='ss')
    assert f'.lib "{tb.netlist.pdk_details.library_location}" ss\n' in text
    tb_dir, _ = tb._output_paths(corner_subdir='ss')
    assert os.path.basename(tb_dir) == 'ss'


@pytest.mark.parametrize('corner', BAD_CORNERS)
def test_load_pdk_corner_refused(corner):
    from silk import load_pdk
    with pytest.raises(DeckTextError, match='corner'):
        load_pdk('sky130', corner=corner)


# ---------------------------------------------------------------------------
# The final text each emitter writes is checked
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('brk', BREAKS)
def test_object_whose_text_holds_a_break_is_refused_at_render(pdk, brk):
    r = elements.resistor('r1', p='a', n='0', value='1k')
    r.params['value'] = _Text(payload(brk))
    with pytest.raises(DeckTextError, match=r"'r1'.*'value'"):
        r.to_spice()
    mn = pdk.nmos1p8('mn', d='d', g='g', s='0', b='0', w=1, l=0.15)
    mn.params['w'] = _Text(payload(brk, '1'))
    with pytest.raises(DeckTextError, match=r"'mn'.*'w'"):
        mn.to_spice()
    e = sources.VoltageControlledVoltageSource(
        'e1', out_pos='o', out_neg='0', ctrl_pos='i', ctrl_neg='0', gain=10)
    object.__setattr__(e, 'gain', _Text(payload(brk, '10')))
    with pytest.raises(DeckTextError, match=r"'e1'.*'gain'"):
        e.to_spice()
    sine = sources.Sine(voffset=0, vampl=1, freq='1Meg')
    object.__setattr__(sine, 'vampl', _Text(payload(brk, '1')))
    with pytest.raises(DeckTextError, match=r'Sine.*vampl'):
        sine.to_spice()
    from silk.simulators import load_simulator_config
    with pytest.raises(DeckTextError, match='option'):
        load_simulator_config('ngspice').emit_options_line(
            {'reltol': _Text(payload(brk, '1e-4'))})


@pytest.mark.parametrize('brk', BREAKS)
def test_spice_prefix_refused_on_both_device_paths(brk):
    from silk.utils._utils import render_pdk_device_line
    with pytest.raises(DeckTextError, match='prefix'):
        render_pdk_device_line('XM' + brk + '.endc\n', 'mn', ['d'], 'model', [])
    r = elements.resistor('r1', p='a', n='0', value='1k')
    bad = type('BadResistor', (type(r),), {'_spice_prefix': 'R' + brk + '.endc '})
    r.__class__ = bad
    with pytest.raises(DeckTextError, match='prefix'):
        r.to_spice()


@pytest.mark.parametrize('brk', BREAKS)
def test_path_values_checked_as_written(pdk, tmp_path, brk):
    from pathlib import Path

    from silk.circuit.spice_netlister import PySpiceNetlister
    netlister = PySpiceNetlister(pdk)
    pdk.extra_includes = [Path('/x/extra' + brk + '.endc.spice')]
    with pytest.raises(DeckTextError, match='include'):
        netlister.generate_pdk_include('tt')
    pdk.extra_includes = [Path('/x/extra.spice')]
    assert '.include "/x/extra.spice"\n' in netlister.generate_pdk_include('tt')
    pdk.osdi_files = [Path('/x/psp' + brk + 'shell id.osdi')]
    with pytest.raises(DeckTextError, match='osdi'):
        netlister._osdi_lines()

    class Sample:
        WRAPPER_SECTION = 'mc_both'
        wrapper_path = Path('/x/mc' + brk + '.control/mc_both.lib.spice')
        section = 'mc'

    with pytest.raises(DeckTextError, match='wrapper'):
        netlister.generate_pdk_include('tt', sample=Sample())


def _statistical(**overrides):
    from silk.pdk.statistical import StatisticalModels
    fields = dict(mismatch_sections={'tt': 'tt_mm'}, mismatch_switch='mc_mm_switch',
                  record={}, process_section='mc', process_corners=('tt',),
                  process_switch='mc_pr_switch', process_includes=())
    fields.update(overrides)
    return StatisticalModels(**fields)


@pytest.mark.parametrize('brk', BREAKS)
def test_monte_carlo_wrapper_refused_before_it_is_written(tmp_path, brk):
    from pathlib import Path

    from silk.circuit.monte_carlo import write_both_wrapper
    good = tmp_path / 'mc_both.lib.spice'
    for stat, path in [
        (_statistical(mismatch_switch='mc_mm' + brk + '.control'), good),
        (_statistical(process_switch='mc_pr' + brk + '.control'), good),
        (_statistical(process_includes=('/x/p' + brk + '.spice',)), good),
        (_statistical(), Path(str(tmp_path / ('w' + brk + 'x')) + '/mc_both.lib.spice')),
    ]:
        with pytest.raises(DeckTextError):
            write_both_wrapper(path, stat)
    assert _no_deck_written(tmp_path)
    assert not any(p.is_dir() for p in tmp_path.iterdir())
    write_both_wrapper(good, _statistical())
    assert '.param mc_mm_switch=1\n.param mc_pr_switch=1\n' in good.read_text()


@pytest.mark.parametrize('brk', BREAKS)
def test_mismatch_off_switch_refused(brk):
    from types import SimpleNamespace

    from silk.circuit.modules_netlist import _mismatch_off_line
    module = SimpleNamespace(mismatch_enabled=False)

    def netlister(switch):
        return SimpleNamespace(
            _mc_sample=SimpleNamespace(variation='mismatch'),
            pdk_details=SimpleNamespace(
                statistical=_statistical(mismatch_switch=switch)))

    assert _mismatch_off_line(module, netlister('mc_mm_switch')) == \
        '.param mc_mm_switch=0\n'
    with pytest.raises(DeckTextError, match='mismatch switch'):
        _mismatch_off_line(module, netlister('mc_mm' + brk + '.control'))


@pytest.mark.parametrize('brk', BREAKS)
def test_showmod_fields_refused(brk):
    from silk.circuit.monte_carlo_draws import showmod_lines
    assert showmod_lines([('xdut.xm1', 'sky130', ('vth0', 'u0'))]) == [
        "showmod 'm.xdut.xm1.msky130' : vth0 u0"]
    with pytest.raises(DeckTextError, match='showmod'):
        showmod_lines([('xdut' + brk + 'shell id', 'm', ('vth0',))])
    with pytest.raises(DeckTextError, match='showmod'):
        showmod_lines([('xdut.xm1', 'm' + brk, ('vth0',))])
    with pytest.raises(DeckTextError, match='showmod'):
        showmod_lines([('xdut.xm1', 'm', ('vth0' + brk + 'shell id',))])


@pytest.mark.parametrize('field', ['model', 'param', 'internal'])
@pytest.mark.parametrize('brk', BREAKS)
def test_preflight_deck_refused_before_any_file_or_process(
        pdk, tmp_path, monkeypatch, no_subprocess, field, brk):
    import tempfile

    from silk.circuit.spice_netlister import PySpiceNetlister

    def no_tempdir(*args, **kwargs):
        raise AssertionError('a preflight folder was created')

    monkeypatch.setattr(tempfile, 'mkdtemp', no_tempdir)
    args = dict(model='sky130_fd_pr__nfet_01v8', param='gm', internal='msky130_fd_pr__nfet_01v8')
    args[field] = args[field] + brk + '.control\nshell id\n.endc'
    with pytest.raises(DeckTextError, match='preflight'):
        PySpiceNetlister(pdk)._ngspice_param_preflight_uncached(
            'ngspice', 'tt', args['model'], args['param'], 'n', args['internal'])


@pytest.mark.parametrize('brk', BREAKS)
def test_batch_save_line_refused(brk):
    from silk.circuit.spice_netlister import _quote_save_argument
    with pytest.raises(DeckTextError):
        _quote_save_argument(_Text('@x' + brk + 'shell id'))


# ---------------------------------------------------------------------------
# Xyce output paths are checked without running a simulator
# ---------------------------------------------------------------------------

def _xyce(pdk):
    from silk.circuit.xyce_netlister import XyceNetlister
    return XyceNetlister(pdk)


@pytest.mark.parametrize('brk', BREAKS)
@pytest.mark.parametrize('which', ['rawfile', 'noise', 'temperature'])
def test_xyce_output_paths_checked_as_written(pdk, tmp_path, brk, which):
    from silk.simulation.run_files import RunFiles
    ok = str(tmp_path)
    bad = str(tmp_path / ('o' + brk + '.control')) + '/out'
    files = RunFiles(
        directory=ok, netlist_path='',
        rawfile_path=bad if which == 'rawfile' else ok + '/rawfile.raw',
        noise_path=bad if which == 'noise' else ok + '/noise.csv',
        temperature_path=bad if which == 'temperature' else ok + '/t.csv')
    analyses = ({'noise': Noise(output='out', ref='0', source='vin',
                                variation='dec', points=10, fstart=1, fstop=1e6)}
                if which == 'noise' else
                {'tran': Transient(step=1e-9, stop=1e-7)})
    with pytest.raises(DeckTextError, match=f'Xyce {which} output path'):
        _xyce(pdk).generate_output_block(ok, analyses, run_files=files)


@pytest.mark.parametrize('brk', BREAKS)
def test_xyce_relative_output_directory_checked_after_abspath(
        pdk, tmp_path, monkeypatch, brk):
    cwd = tmp_path / ('cwd' + brk + '.control')
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with pytest.raises(DeckTextError, match='Xyce rawfile output path'):
        _xyce(pdk).generate_output_block(
            'out', {'tran': Transient(step=1e-9, stop=1e-7)})


def test_xyce_ordinary_output_block_unchanged(pdk, tmp_path):
    block = _xyce(pdk).generate_output_block(
        str(tmp_path), {'tran': Transient(step=1e-9, stop=1e-7)})
    assert f'.print tran format=raw file={tmp_path}/rawfile.raw' in block


# ---------------------------------------------------------------------------
# AC7 containment controls: separators and dots stay inside the test folder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('corner, folder', [
    ('/tmp/x', '%2Ftmp%2Fx'),
    ('a/b', 'a%2Fb'),
    ('../../escape', '..%2F..%2Fescape'),
    ('a/../b', 'a%2F..%2Fb'),
    ('a\\b', 'a%5Cb'),
    ('C:\\x', 'C%3A%5Cx'),
    ('a..b', 'a..b'),
    ('%2F', '%252F'),
])
def test_corner_folder_is_one_contained_component(bench, corner, folder):
    m, tb = bench
    base, _ = tb._output_paths()
    tb_dir, _ = tb._output_paths(corner_subdir=corner)
    assert os.path.dirname(tb_dir) == base
    assert os.path.basename(tb_dir) == folder
    tb_dir, _ = tb._output_paths(corner_subdir=(corner, 'mc_0000'))
    assert os.path.dirname(os.path.dirname(tb_dir)) == base


def test_long_corner_folder_is_bounded(bench):
    m, tb = bench
    base, _ = tb._output_paths()
    tb_dir, _ = tb._output_paths(corner_subdir='c' * 300)
    assert os.path.dirname(tb_dir) == base
    assert os.path.basename(tb_dir).startswith('test~')
    assert len(os.path.basename(tb_dir)) < 255


@pytest.mark.parametrize('later', ['..', '.', 'mc\n', 'mc\x00'])
def test_invalid_later_tuple_part_refused(bench, later):
    m, tb = bench
    with pytest.raises(DeckTextError, match='corner'):
        tb._output_paths(corner_subdir=('ss', later))


# ---------------------------------------------------------------------------
# Raw text is checked before conversion, the exact error type
# survives, and custom emitters keep their own text
# ---------------------------------------------------------------------------

def _exactly(exc_info):
    assert type(exc_info.value) is DeckTextError, type(exc_info.value)


@pytest.mark.parametrize('brk', BREAKS)
def test_pwl_points_checked_before_float(brk):
    for build in (lambda: sources.PWL([(0, 0), ('1e-9' + brk, 1)]),
                  lambda: sources.PWL([0, 0, 1e-9, '1' + brk]),
                  lambda: sources.PWL([(0, 0)]).add('1e-9', '1' + brk)):
        with pytest.raises(DeckTextError, match='PWL point') as exc:
            build()
        _exactly(exc)


@pytest.mark.parametrize('brk', BREAKS)
def test_analysis_counts_checked_before_parsing(brk):
    with pytest.raises(DeckTextError, match='AC.points') as exc:
        AC(variation='dec', points='10' + brk, fstart=1, fstop=1e6)
    _exactly(exc)
    with pytest.raises(DeckTextError, match='Noise.points') as exc:
        Noise(output='out', ref='0', source='vin', variation='dec',
              points='10' + brk, fstart=1, fstop=1e6)
    _exactly(exc)


@pytest.mark.parametrize('brk', BREAKS)
def test_pulse_edge_with_embedded_break_keeps_its_type(brk):
    from silk.circuit.variable import Variable
    with pytest.raises(DeckTextError, match='Pulse tr') as exc:
        sources.Pulse(v1=0, v2=1, td=0, tr='1' + brk + 'n', tf='1n',
                      pw='5n', per='10n')
    _exactly(exc)
    tf = Variable('tfv', value='1n')
    pulse = sources.Pulse(v1=0, v2=1, td=0, tr='1n', tf=tf, pw='5n', per='10n')
    tf.value = '1' + brk + 'n'
    with pytest.raises(DeckTextError, match='Pulse tf') as exc:
        pulse.to_spice()
    _exactly(exc)


@pytest.mark.parametrize('brk', BREAKS)
def test_initial_value_keeps_the_refusal_type(brk):
    from silk.circuit.variable import Variable
    td = Variable('tdv', value='0')
    pulse = sources.Pulse(v1=0, v2=1, td=td, tr='1n', tf='1n', pw='5n', per='10n')
    td.value = '0' + brk
    with pytest.raises(DeckTextError) as exc:
        pulse.initial_value()
    _exactly(exc)


@pytest.mark.parametrize('brk', BREAKS)
def test_backend_timing_checks_keep_the_refusal_type(monkeypatch, brk):
    from silk.circuit.variable import Variable
    from silk.sources import _timing
    phase = Variable('ph', value=0)
    sffm = sources.SFFM(vo=0, va=1, fc='1Meg', mdi=1, fs='1k', phasec=phase)
    vs = sources.VoltageSource('vs', positive='a', negative='0', sffm=sffm)
    phase.value = '0' + brk
    with pytest.raises(DeckTextError) as exc:
        _timing.validate_source_timing(vs, 'xyce')
    _exactly(exc)

    class OldOrder:
        version = '40'
        verified_waveforms_label = '40'
        waveforms_verified = True
        release = 40
        below_minimum = False
        minimum = 40

        def has(self, feature):
            return False

    monkeypatch.setattr(_timing, 'NgspiceCompatibility', OldOrder)
    phase.value = 0
    am = sources.AM(va=1, vo=0, mf='1k', fc='1Meg', phases=phase)
    va = sources.VoltageSource('va', positive='a', negative='0', am=am)
    phase.value = '0' + brk
    with pytest.raises(DeckTextError) as exc:
        _timing.validate_source_timing(va, 'ngspice')
    _exactly(exc)


@pytest.mark.parametrize('brk', BREAKS)
def test_passive_refusal_names_the_instance(brk):
    with pytest.raises(DeckTextError, match=r"Resistor 'r1' parameter 'resistance'") as exc:
        passives.Resistor('r1', 'a', 'b', brk)
    _exactly(exc)
    r = passives.Resistor('r1', 'a', 'b', '1k')
    with pytest.raises(DeckTextError, match=r"'r1'.*'resistance'") as exc:
        r.resistance = '1k' + brk
    _exactly(exc)


def test_custom_source_emitter_keeps_its_own_text():
    """A subclass with its own to_spice() owns its text: silk does not
    inspect it, at assignment or at emission (AC3)."""
    from silk.sources.classes import BehavioralVoltage, SourceBase

    class Block(SourceBase):
        _spice_prefix = 'X'
        _node_terminals = ()

        def __init__(self, name, text):
            super().__init__(name)
            self.deck_text = text

        def to_spice(self):
            return self.deck_text + '\n'

    block = Block('blk', '.param a=1\n.param b=2')
    assert block.to_spice() == '.param a=1\n.param b=2\n'

    class TwoLine(BehavioralVoltage):
        def to_spice(self):
            return f'* {self.expression}\n'

    two = TwoLine('b2', positive='o', negative='0', expression='a\nb')
    assert two.expression == 'a\nb'


def test_custom_waveform_emitter_keeps_its_own_text():
    class Burst(sources.Pulse):
        def to_spice(self):
            return self.v1

    burst = Burst(v1='PWL(0 0\n+ 1n 1)', v2=1, td=0, tr='1n', tf='1n',
                  pw='5n', per='10n')
    assert burst.to_spice() == 'PWL(0 0\n+ 1n 1)'


def test_metadata_on_a_builtin_source_is_not_card_text():
    vs = sources.VoltageSource('vs', positive='a', negative='0', dc=1)
    vs.note = 'bias for\nthe mirror'
    assert vs.to_spice() == 'Vvs a 0 DC 1\n'
    sine = sources.Sine(voffset=0, vampl=1, freq='1Meg')
    sine.note = 'a\nb'
    assert sine.to_spice() == 'SIN(0 1 1Meg 0 0)'


def test_generated_continuation_lines_still_emit():
    bits = sources.BitPattern('0110', v1=0, v2=1, ui='1n', tr='10p', tf='10p')
    assert '\n+ ' in bits.to_spice()


# ---------------------------------------------------------------------------
# Review round 2: the prepared-deck freshness check keeps the refusal's type
# ---------------------------------------------------------------------------

def _prepared_module(pdk, tmp_path, value):
    m = _module(pdk, tmp_path)
    r = elements.resistor('r1', p='a', n='0', value=value)
    m.add(r)
    m.add(sources.vdc('va', positive='a', negative='0', value=1.0))
    m.add_sim_type(analysis.Op())
    m.generate_netlist()
    path = m._run_files.netlist_path
    with open(path, 'rb') as f:
        return m, r, path, f.read()


@pytest.fixture
def no_launch(monkeypatch, no_subprocess):
    from silk.simulation.backend import SimulatorBackend

    def refuse(*args, **kwargs):
        raise AssertionError('the simulator backend was launched')

    for cls in SimulatorBackend._registry.values():
        monkeypatch.setattr(cls, 'run', refuse)


@pytest.mark.parametrize('brk', BREAKS)
@pytest.mark.parametrize('how', ['params', 'variable'])
def test_run_simulator_keeps_the_refusal_after_preparation(
        pdk, tmp_path, no_launch, brk, how):
    from silk.circuit.variable import Variable
    var = Variable('rv', value='1k')
    m, r, path, before = _prepared_module(
        pdk, tmp_path, var if how == 'variable' else '1k')
    bad = payload(brk)
    if how == 'variable':
        var.value = bad
    else:
        r.params['value'] = bad
    with pytest.raises(DeckTextError) as exc:
        m.run_simulator()
    _exactly(exc)
    assert "'r1'" in str(exc.value) and "'value'" in str(exc.value)
    assert repr(bad) in str(exc.value)
    with open(path, 'rb') as f:
        assert f.read() == before


def test_run_simulator_stale_deck_is_still_a_configuration_error(
        pdk, tmp_path, no_launch):
    from silk.exceptions import SimulationConfigurationError
    m, r, path, before = _prepared_module(pdk, tmp_path, '1k')
    r.params['value'] = '2k'
    with pytest.raises(SimulationConfigurationError) as exc:
        m.run_simulator()
    assert type(exc.value) is SimulationConfigurationError
    with open(path, 'rb') as f:
        assert f.read() == before
