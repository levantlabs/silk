"""Measure ngspice probe support and public logical currents for every kind."""
import numpy as np
import pytest

from silk import sources
from silk.circuit.modules import Module
from silk.codegen.generator import _source_entry_to_spec
from silk.codegen.testbench_generator import _source_add_line
from silk.elements import elements
from silk.exceptions import CircuitError, NotSupportedError, SimulationError
from silk.simulation import Op, Transient
from silk.simulation._ngspice_version import ngspice_version
from tests._env import needs_ngspice
from tests.test_source_element_names import CONTROL, KINDS, NAMINGS, WAVES

pytestmark = pytest.mark.slow


def source_bench(tmp_path, kind, fields, array, analysis=None, *, name='Stim'):
    dut = Module(None, str(tmp_path), 'dut.cir', None, module_name='dut')
    dut.define_module_ports({})
    dut.add(elements.resistor('unused', p='unused', n='0', value=1000))
    tb = dut.create_test('names')
    tb.add_module_under_test({})
    entry = {'type': kind, 'instance': name, **fields, **array}
    if array:
        pin = 'out_pos' if 'out_pos' in entry else 'positive'
        entry[pin] = 'out[$i]'
    specs = [_source_entry_to_spec(e, 'test') for e in (CONTROL, entry)]
    ns = {'tb': tb, 'sources': sources}
    exec(_source_add_line(CONTROL, specs), ns)
    exec(_source_add_line(entry, specs), ns)
    tb.add(elements.resistor('control_load', p='control', n='0', value=1000))
    for index, src in enumerate(tb.netlist.sig_sources_dict.values()):
        if src.name == 'control':
            continue
        out = getattr(src, 'positive', None) or src.out_pos
        tb.add(elements.resistor(f'load{index}', p=out, n='0', value=1000))
    tb.add_sim_type(analysis or Transient(step=1e-8, stop=2e-6))
    return tb, entry


@pytest.mark.parametrize('kind,letter,readable,fields', KINDS)
@pytest.mark.parametrize('analysis', ['tran', 'op'])
@needs_ngspice
def test_ngspice_kind_inventory(tmp_path, kind, letter, readable, fields, analysis):
    """B V= has a branch; G/F/B I= cannot be read by silk's i(name) path."""
    tb, _ = source_bench(tmp_path, kind, fields, {}, Op() if analysis == 'op' else None)
    current = tb.probe('i(Stim)', analysis=analysis)
    tb.probe('out', analysis=analysis)
    tb.run()
    raw = tb.netlist.get_signal_names(raw=True)
    print(f'{kind} ({analysis}): raw vectors={raw}')
    if readable:
        assert np.all(np.isfinite(current.y))
        assert np.any(np.abs(current.y) > 0)
        np.testing.assert_array_equal(current.y, tb.netlist.load_signal(letter + 'Stim', 'I', analysis=analysis).y)
    else:
        with pytest.raises(SimulationError, match='not found'):
            _ = current.y


CASES = [(kind, letter, readable, fields, 'constant') for kind, letter, readable, fields in KINDS]
CASES += [(kind, letter, True, {'positive': 'out', 'negative': '0', 'dc': .2, **fields}, wave)
          for kind, letter in [('VoltageSource', 'V'), ('CurrentSource', 'I')]
          for wave, fields in WAVES if wave != 'constant']


@pytest.mark.parametrize('kind,fields', [('VoltageSource', KINDS[0][3]),
                                       ('CurrentSource', KINDS[1][3])])
def test_literal_colon_source_is_refused_at_construction(tmp_path, kind, fields):
    # Authored colons cannot reach emission or simulation (TASK-553).
    with pytest.raises(CircuitError, match="invalid instance name 'Stim:part'"):
        source_bench(tmp_path, kind, fields, {}, name='Stim:part')


# SFFM refusal needs no binary; supported waveforms require actual readback.
@pytest.mark.parametrize('kind,letter,readable,fields,wave', [
    pytest.param(*case, marks=needs_ngspice if case[4] != 'sffm' else ())
    for case in CASES])
@pytest.mark.parametrize('array,names,components', NAMINGS)
@pytest.mark.parametrize('analysis', ['tran', 'op'])
def test_public_probe_matches_logical_card_current(tmp_path, kind, letter, readable, fields, wave, array, names, components, analysis):
    from silk.codegen import source_element_names
    from silk.circuit import element_names
    tb, entry = source_bench(tmp_path, kind, fields, array, Op() if analysis == 'op' else None)
    records = source_element_names(entry)
    assert records == tuple(element_names(tb.netlist.sig_sources_dict[n]) for n in names)
    if wave == 'sffm' and ngspice_version() != '42':
        with pytest.raises(NotSupportedError, match='only ngspice 42 is verified'):
            tb.netlist.generate_netlist()
        return
    probes = [tb.probe(r.current_probe, analysis=analysis) if r.current_probe else None for r in records]
    tb.run()
    for record, probe in zip(records, probes):
        if not readable:
            assert record.current_probe is None
            continue
        assert probe is not None
        measured = [tb.netlist.load_signal(c, 'I', analysis=analysis).y for c in record.card_names]
        # A series voltage source reads its waveform branch; parallel current
        # cards add. Their signed currents also agree with the resistor load.
        expected = sum(measured) if letter == 'I' else measured[0]
        np.testing.assert_allclose(probe.y, expected, atol=1e-12)
        src = tb.netlist.sig_sources_dict[record.designer_name]
        out = getattr(src, 'positive', None) or src.out_pos
        voltage = tb.netlist.load_signal(out, analysis=analysis).y
        np.testing.assert_allclose(probe.y, -voltage / 1000, atol=1e-9)
        assert np.any(np.abs(probe.y) > 0)


@pytest.mark.parametrize('name,component', [('coil', 'coil'), ('coil[2]', 'coil<2>'), ('coil_2', 'coil_2')])
@pytest.mark.parametrize('analysis', ['tran', 'op'])
@needs_ngspice
def test_built_inductor_probe_reads_the_emitted_branch(tmp_path, name, component, analysis):
    from silk.circuit import ElementNames, element_names
    tb, _ = source_bench(tmp_path, 'VoltageSource', KINDS[0][3], {},
                         Op() if analysis == 'op' else None)
    coil = elements.inductor(name, p='out', n='load', value=1e-3)
    tb.add(coil)
    tb.add(elements.resistor('series_load', p='load', n='0', value=1000))
    record = element_names(coil)
    assert record == ElementNames(name, f'i(L{component})', (f'L{component}',))
    current = tb.probe(record.current_probe, analysis=analysis)
    voltage = tb.probe('load', analysis=analysis)
    tb.run()
    np.testing.assert_allclose(current.y, tb.netlist.load_signal(record.card_names[0], 'I', analysis=analysis).y)
    np.testing.assert_allclose(current.y, voltage.y / 1000, atol=1e-12)
    assert np.any(np.abs(current.y) > 0)
