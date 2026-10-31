"""TASK-352: a DC sweep or noise analysis names ONE source element.

``vin[1]`` (an element of the arrayed source ``vin``) is valid; the bare array
name ``vin`` and an unknown name are refused at generate time and, on the run
path, before any deck is written.
"""
import pytest

from silk.circuit.circuit_elements import PortDir
from silk.circuit.modules import Module
from silk.codegen.generator import Generator
from silk.codegen.testbench_generator import TestbenchSpecError, generate_testbenches
from silk.exceptions import SweepSourceError
from silk.simulation import analysis
from tests._env import HAVE_NGSPICE

VIN = {'type': 'VoltageSource', 'instance': 'vin', 'positive': 'n$i',
       'negative': '0', 'dc': 0.5, 'ac': 1, 'array': 2}
VS = {'type': 'VoltageSource', 'instance': 'vs', 'positive': 'a',
      'negative': '0', 'dc': 0.5, 'ac': 1}
R1 = {'type': 'resistor', 'instance': 'r1', 'nets': {'p': 'n0', 'n': '0'},
      'params': {'value': 1000.0}}
DC = {'kind': 'dc', 'start': 0, 'stop': 1, 'step': 0.1}
NOISE = {'kind': 'noise', 'output': 'n0', 'ref': '0', 'points': 5,
         'fstart': 1, 'fstop': 1e3}


def _spec(analysis_dict, sources=(VIN, VS), rig_sources=None):
    t = {'name': 't', 'dut_portmap': {'a': 'a'}, 'sources': list(sources),
         'ideal_elements': [R1], 'analyses': [analysis_dict]}
    spec = {'circuit_name': 'c', 'tests': [t], 'rigs': []}
    if rig_sources is not None:
        spec['rigs'] = [{'name': 'r', 'sources': rig_sources}]
        t['applied_rigs'] = ['r']
        t['sources'] = []
    return spec


@pytest.mark.parametrize('base', [DC, NOISE], ids=['dc', 'noise'])
def test_element_name_accepted(base):
    code = generate_testbenches(_spec(dict(base, source='vin[1]')))
    assert "'vin[1]'" in code


@pytest.mark.parametrize('base', [DC, NOISE], ids=['dc', 'noise'])
def test_bare_arrayed_name_refused(base):
    kind = base['kind']
    with pytest.raises(TestbenchSpecError) as e:
        generate_testbenches(_spec(dict(base, source='vin')))
    assert str(e.value) == (
        f"test 't' {kind} analysis: source 'vin' is an arrayed source, and a "
        f"sweep names one element. Valid element names: 'vin[0]', 'vin[1]'. "
        f"To move all elements together, use one source.")


@pytest.mark.parametrize('base', [DC, NOISE], ids=['dc', 'noise'])
def test_unknown_name_refused(base):
    kind = base['kind']
    with pytest.raises(TestbenchSpecError) as e:
        generate_testbenches(_spec(dict(base, source='nosuch')))
    temp = " ('temp' is also accepted.)" if kind == 'dc' else ''
    assert str(e.value) == (
        f"test 't' {kind} analysis: source 'nosuch' is not a source of this "
        f"test. Valid source names: 'vin[0]', 'vin[1]', 'vs'.{temp}")


def test_scalar_source_emitted_spelling_and_temp_unchanged():
    for src in ('vs', 'Vvs', 'VS'):
        generate_testbenches(_spec(dict(DC, source=src)))
    generate_testbenches(_spec(dict(DC, source='temp')))
    generate_testbenches(_spec(dict(NOISE, source='vs')))


def test_rig_sources_count():
    generate_testbenches(_spec(dict(DC, source='vin[0]'), rig_sources=[VIN]))
    with pytest.raises(TestbenchSpecError, match="arrayed source"):
        generate_testbenches(_spec(dict(DC, source='vin'), rig_sources=[VIN]))


def test_underscore_naming_elements():
    v = dict(VIN, naming='underscore')
    generate_testbenches(_spec(dict(DC, source='vin_1'), sources=(v, VS)))
    with pytest.raises(TestbenchSpecError, match="'vin_0', 'vin_1'"):
        generate_testbenches(_spec(dict(DC, source='vin'), sources=(v, VS)))


@pytest.mark.parametrize('blank', [None, '', '  '], ids=['absent', 'empty', 'blank'])
@pytest.mark.parametrize('base', [DC, NOISE], ids=['dc', 'noise'])
def test_missing_source_refused_testbench(base, blank):
    kind = base['kind']
    with pytest.raises(TestbenchSpecError) as e:
        generate_testbenches(_spec(dict(base) if blank is None
                                   else dict(base, source=blank)))
    temp = " ('temp' is also accepted.)" if kind == 'dc' else ''
    assert str(e.value) == (
        f"test 't' {kind} analysis: names no source. Valid source names: "
        f"'vin[0]', 'vin[1]', 'vs'.{temp}")


# --- circuit codegen ---------------------------------------------------------

def _circuit(sim):
    return {'circuit_name': 'c', 'devices': [], 'ideal_elements': [R1],
            'sources': [VIN, VS], 'simulation': sim}


def test_circuit_codegen_same_gap_same_fix():
    gen = Generator()
    assert "source='vin[1]'" in gen.generate(_circuit(dict(DC, source='vin[1]')))
    with pytest.raises(ValueError, match=r"dc analysis: source 'vin' is an arrayed"):
        gen.generate(_circuit(dict(DC, source='vin')))
    with pytest.raises(ValueError, match=r"noise analysis: source 'nosuch' is not"):
        gen.generate(_circuit(dict(NOISE, source='nosuch')))


@pytest.mark.parametrize('blank', [None, '', '  '], ids=['absent', 'empty', 'blank'])
@pytest.mark.parametrize('base', [DC, NOISE], ids=['dc', 'noise'])
def test_missing_source_refused_circuit(base, blank):
    temp = " ('temp' is also accepted.)" if base['kind'] == 'dc' else ''
    with pytest.raises(ValueError) as e:
        Generator().generate(_circuit(
            dict(base) if blank is None else dict(base, source=blank)))
    assert str(e.value) == (
        f"{base['kind']} analysis: names no source. Valid source names: "
        f"'vin[0]', 'vin[1]', 'vs'.{temp}")


# --- run path ------------------------------------------------------------------





def test_run_refuses_before_writing_a_deck(tmp_path):
    from silk import sources
    from silk.elements import elements
    dut = Module('dut', str(tmp_path), 'dut.cir', pdk_details=None,
                 module_name='dut')
    dut.define_module_ports({'a': PortDir.INOUT})
    tb = dut.create_test('t')
    tb.add_module_under_test({'a': 'a'})
    tb.add(*[sources.vdc(f'vin[{i}]', positive=f'n{i}', negative='0', value=0.5)
             for i in range(2)])
    tb.add(elements.resistor('r1', p='n0', n='0', value=1000.0))
    tb.add_sim_type(analysis.DC(source='vin', start=0, stop=1, step=0.1))
    with pytest.raises(SweepSourceError) as e:
        tb.run('tt')
    assert str(e.value) == (
        "test 't': DC(source='vin'): source 'vin' is an arrayed source, and a "
        "sweep names one element. Valid element names: 'vin[0]', 'vin[1]'. "
        "To move all elements together, use one source.")
    assert not list(tmp_path.rglob('*.cir'))

    tb.add_sim_type(analysis.DC(source='nosuch', start=0, stop=1, step=0.1))
    with pytest.raises(SweepSourceError) as e:
        tb.run('tt')
    assert str(e.value) == (
        "test 't': DC(source='nosuch'): source 'nosuch' is not a source of "
        "this test. Valid source names: 'vin[0]', 'vin[1]'. "
        "('temp' is also accepted.)")

    tb.add_sim_type(analysis.DC(source='vin[0]', start=0, stop=1, step=0.1))
    tb.add_sim_type(analysis.Noise('n0', '0', 'nosuch', 'dec', 5, 1, 1e3))
    with pytest.raises(SweepSourceError, match="Noise\\(source='nosuch'\\)"):
        tb.run('tt')


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_NGSPICE, reason='needs the ngspice binary')
@pytest.mark.parametrize('kind', ['dc', 'noise'])
def test_element_name_simulates(tmp_path, kind):
    base = DC if kind == 'dc' else NOISE
    ns = {}
    exec(generate_testbenches(_spec(dict(base, source='vin[1]'))), ns)
    dut = Module('dut', str(tmp_path), 'dut.cir', pdk_details=None,
                 module_name='dut')
    dut.define_module_ports({'a': PortDir.INOUT})
    ns['_attach_t'](dut)
    tb = dut._tests['t']
    tb.run('tt')     # SimulationAborted if the analysis card named no source
    if kind == 'dc':
        assert tb.netlist.load_signal('v(n1)').y[-1] == pytest.approx(1.0)
