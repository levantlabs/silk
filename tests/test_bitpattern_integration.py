"""Finite source integration: executable declarations and recorded run identity."""
import copy
from contextlib import nullcontext
import hashlib
import json
import pickle
import sys
from types import ModuleType
from pathlib import Path

import numpy as np
import pytest

from silk import Module, Variable, load_pdk, sources
from silk.codegen.generator import Generator
from silk.codegen.testbench_generator import generate_testbenches
from silk.elements import elements
from silk.exceptions import SimulationConfigurationError, SimulationError
from silk.simulation import Transient
from silk.simulation.conditions import _identify_supply
from silk.sources.schema import export_source_schema
from tests._env import HAVE_XYCE, needs_ngspice


def _params(kind, **overrides):
    params = dict(v1=0.25, v2=1.25, ui=1e-6, tr=1e-7, tf=2e-7, td=3e-7)
    params.update(dict(bits='1010', repeat=2) if kind == 'bitpattern'
                  else dict(order=7, length=16, seed=3))
    params.update(overrides)
    return params


def _wave(kind, **overrides):
    cls = sources.BitPattern if kind == 'bitpattern' else sources.PRBS
    return cls(**_params(kind, **overrides))


def _module(tmp_path, **kwargs):
    return Module('sky130', str(tmp_path), 'bits.cir', load_pdk('sky130'), **kwargs)


def _entry(kind, quantity='VoltageSource', **overrides):
    entry = dict(type=quantity, instance='data', positive='out', negative='0',
                 waveform=dict(type=kind, params=_params(kind)))
    entry.update(overrides)
    return entry


def _exec(code, **namespace):
    exec(compile(code, '<generated-bits>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
@pytest.mark.parametrize('quantity', ['VoltageSource', 'CurrentSource'])
@pytest.mark.parametrize('placement', ['circuit', 'hierarchy', 'array'])
def test_executable_codegen_preserves_wave_and_variable(tmp_path, monkeypatch, kind, quantity, placement):
    # Execute all generated construction/emission; simulation is covered separately.
    monkeypatch.setattr(Module, 'simulate', lambda self, **kw: self.generate_netlist())
    entry = _entry(kind, quantity)
    entry['waveform']['params']['v2'] = {'variable': 'level', 'scale': 2}
    spec = dict(circuit_name='bits', pdk='sky130', output_dir=str(tmp_path),
                variables=[dict(name='level', value=0.625)], sources=[entry])
    if placement != 'circuit':
        spec['ports'] = {'out': {'dir': 'inout'}}
    if placement == 'array':
        entry['array'] = 2
    ns = _exec(Generator().generate(spec))
    mod = ns['ckt'] if placement == 'circuit' else ns['mod']
    if placement == 'hierarchy':
        imported = ModuleType('_generated_bit_driver')
        imported.mod = mod
        monkeypatch.setitem(sys.modules, imported.__name__, imported)
        parent_spec = dict(
            circuit_name='parent', pdk='sky130', output_dir=str(tmp_path / 'parent'),
            ports={'out': {'dir': 'inout'}}, submodules=[dict(
                instance_name='child', nets={'out': 'out'},
                module_ref=dict(import_path=imported.__name__, module_var='mod'))])
        parent = _exec(Generator().generate(parent_spec))['mod']
        mod = parent.elem_dict['child']['device']
    assert len(mod.sig_sources_dict) == (2 if placement == 'array' else 1)
    for src in mod.sig_sources_dict.values():
        assert src.waveform.stream == _wave(kind).stream
        np.testing.assert_allclose(src.waveform.to_numpy([0, 1e-6, 4e-6]),
                                   _wave(kind).to_numpy([0, 1e-6, 4e-6]))
    assert mod.variables() == [ns['level']]
    ns['level'].value = 1
    for src in mod.sig_sources_dict.values():
        assert max(src.waveform.to_numpy(np.arange(0, 16e-6, 0.5e-6))) == 2


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
@pytest.mark.parametrize('quantity', ['VoltageSource', 'CurrentSource'])
def test_executable_testbench_codegen(tmp_path, kind, quantity):
    dut = _module(tmp_path, module_name='dut')
    dut.define_module_ports({'out': 'inout'})
    dut.add(elements.resistor('load', p='out', n='0', value=1000))
    code = generate_testbenches(dict(circuit_name='bits', tests=[dict(
        name='stream', dut_portmap={'out': 'out'}, sources=[_entry(kind, quantity)])]))
    _exec(code)['_attach_stream'](dut)
    tb = dut._tests['stream']
    src = tb.netlist.sig_sources_dict['data']
    assert src.waveform.stream == _wave(kind).stream
    np.testing.assert_array_equal(src.waveform.to_numpy([0, 2e-6]),
                                  _wave(kind).to_numpy([0, 2e-6]))


@pytest.mark.parametrize('kind,field,bad', [
    ('bitpattern', 'bits', '101x'), ('bitpattern', 'bits', [1, True]),
    ('bitpattern', 'bits', [0, 1.0]), ('bitpattern', 'repeat', 1.5),
    ('bitpattern', 'repeat', True), ('prbs', 'order', 8),
    ('prbs', 'length', 0), ('prbs', 'seed', 127), ('prbs', 'seed', False),
    ('prbs', 'order', {'variable': 'level'}),
    ('bitpattern', 'bits', "__import__('os').environ.clear()"),
])
def test_generated_structural_input_is_rejected(tmp_path, kind, field, bad):
    entry = _entry(kind)
    entry['waveform']['params'][field] = bad
    spec = dict(circuit_name='bits', pdk='sky130', output_dir=str(tmp_path), sources=[entry])
    if isinstance(bad, dict):
        spec['variables'] = [dict(name='level', value=7)]
    with pytest.raises((ValueError, TypeError), match='bits|repeat|order|length|seed'):
        _exec(Generator().generate(spec))


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
def test_copy_pickle_and_placement_share_only_declared_variables(tmp_path, kind):
    level = Variable('level', 1.25)
    wave = _wave(kind, v2=level)
    mod = _module(tmp_path, module_name='driver')
    mod.define_module_ports({'out': 'inout'})
    mod.add(sources.VoltageSource('data', positive='out', negative='0', waveform=wave))
    clone = copy.deepcopy(mod)
    back = pickle.loads(pickle.dumps(mod))
    parent = _module(tmp_path / 'parent')
    parent.add_module(mod, 'child', out='out')
    placed = parent.elem_dict['child']['device']
    for other in (clone, back, placed):
        assert other.sig_sources_dict['data'].waveform.stream == wave.stream
    assert clone.variables()[0] is placed.variables()[0] is level
    assert back.variables()[0] is not level
    level.value = 2
    for other in (mod, clone, placed):
        assert max(other.sig_sources_dict['data'].waveform.to_numpy(np.arange(0, 16e-6, 0.5e-6))) == 2
    assert back.variables()[0].value == 1.25


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
@pytest.mark.parametrize('simulator', ['ngspice', 'xyce'])
@pytest.mark.parametrize('change', ['sequence', 'electrical', 'variable'])
def test_changed_source_refuses_prepared_deck(tmp_path, kind, simulator, change):
    level = Variable('level', 1.25)
    wave = _wave(kind, v2=level)
    mod = _module(tmp_path)
    mod.simulator = simulator
    mod.add(sources.VoltageSource('data', positive='out', negative='0', waveform=wave))
    mod.add_sim_type(Transient(1e-7, 20e-6))
    mod.generate_netlist()
    before = Path(mod._run_files.netlist_path).read_text()
    if change == 'sequence':
        if kind == 'bitpattern':
            wave.bits = '0101'
        else:
            wave.seed = 11
    elif change == 'electrical':
        wave.ui = 2e-6
    else:
        level.value = 2
    with pytest.raises(SimulationError, match='Circuit configuration changed'):
        mod.run_simulator()
    assert Path(mod._run_files.netlist_path).read_text() == before
    mod.generate_netlist()
    after = Path(mod._run_files.netlist_path).read_text()
    assert before != after
    assert wave.to_spice() in after


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
@pytest.mark.parametrize('quantity', ['VoltageSource', 'CurrentSource'])
def test_waveform_is_never_a_fixed_supply(tmp_path, kind, quantity):
    mod = _module(tmp_path)
    mod.add(getattr(sources, quantity)('data', positive='out', negative='0',
                                     dc=1.8, waveform=_wave(kind)))
    assert _identify_supply(mod)[1] is None
    mod.set_supply_voltage(1.8, source='data')
    with pytest.raises(SimulationConfigurationError, match='waveform|voltage source'):
        mod.generate_netlist()


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
@pytest.mark.parametrize('quantity', ['VoltageSource', 'CurrentSource'])
def test_schema_template_constructs_the_actual_waveform(kind, quantity):
    from silk.elements.schema import export_schema
    schema = export_source_schema()[quantity]
    entry = schema['waveforms'][kind]
    assert entry['kwarg'] == 'waveform'
    values = _params(kind)
    code = entry['code_template'].format(**{f'param_{k}': repr(v) for k, v in values.items()})
    wave = eval(code, {'sources': sources})
    assert wave.stream == _wave(kind).stream
    if kind == 'bitpattern':
        assert next(p for p in entry['parameters'] if p['name'] == 'bits')['kind'] == 'bits'
    catalog = export_schema(elements, include_metadata=False)
    name = 'vsource' if quantity == 'VoltageSource' else 'isource'
    catalog_entry = catalog[name]['waveforms'][kind]
    assert catalog_entry['parameters'] == entry['parameters']
    fields = {f'param_{k}': repr(v) for k, v in values.items()}
    fields.update(instance_name='data', net_p=repr('out'), net_n=repr('0'), param_dc=0, param_ac=0)
    src = _exec(catalog_entry['code_template'].format(**fields), sources=sources)['data']
    assert src.waveform.stream == wave.stream


def _replay(parameters):
    params = dict(parameters)
    kind = params.pop('kind')
    params.pop('convention', None)
    return (sources.BitPattern if kind == 'bitpattern' else sources.PRBS)(**params)


@pytest.mark.parametrize('kind', ['bitpattern', 'prbs'])
@pytest.mark.parametrize('simulator', [
    'behavioral',
    pytest.param('ngspice', marks=[pytest.mark.slow, needs_ngspice]),
    pytest.param('xyce', marks=[pytest.mark.slow, pytest.mark.requires_xyce,
                               pytest.mark.skipif(not HAVE_XYCE, reason='Xyce unavailable')]),
])
def test_report_profile_replay_and_swept_values(tmp_path, kind, simulator):
    from silk.metrics import final_value
    level = Variable('level', 1.25)
    wave = _wave(kind, v2=level)
    dut = _module(tmp_path, module_name='load', db_path=str(tmp_path / 'runs.db'))
    dut.define_module_ports({'out': 'inout'})
    dut.add(elements.resistor('load', p='out', n='0', value=1000))
    dut.simulator = simulator
    tb = dut.create_test('stream', parameters={'stimulus': wave.as_parameters()})
    tb.add_module_under_test({'out': 'out'})
    tb.add(sources.VoltageSource('data', positive='out', negative='0', waveform=wave))
    tb.add_sim_type(Transient(1e-7, 20e-6))
    tb.add_metric('last', final_value, sig=tb.probe('out'))
    branch = tb.probe('i(data)') if simulator != 'behavioral' else None
    if simulator != 'behavioral':
        tb.set_temperature(27)
    if simulator == 'ngspice':
        tb.netlist.set_simulator_options(seed=17)
    tb.add_spec('finite level', 'last', '>=', 0, key='last')
    assert tb.parameters['stimulus']['v2'] is level
    dut.create_circuit_version()
    reports = []
    for value in (1.25, 2):
        level.value = value
        with (pytest.warns(UserWarning, match='ignored R load')
              if simulator == 'behavioral' else nullcontext()):
            report = tb.run()
        reports.append(pickle.loads(pickle.dumps(report)))
        assert report.parameters['stimulus']['v2'] == value
        assert report.characterization_context.parameters == report.parameters
        dut.store_report(report)
        profile = dut.resolve_characterization_profile(context=report.characterization_context)
        assert dict(profile.parameters['stimulus']) == json.loads(json.dumps(report.parameters))['stimulus']
        replay = _replay(profile.parameters['stimulus'])
        doc = (Path(__file__).resolve().parents[1] / 'docs/api-reference.md').read_text()
        snippet = doc.split('To replay a saved stimulus', 1)[1].split('```python\n', 1)[1].split('```', 1)[0]
        documented = _exec(snippet, report=report, sources=sources)['replayed']
        assert documented.stream == replay.stream
        assert replay.stream == wave.stream
        signal = tb.netlist.load_signal('out')
        np.testing.assert_allclose(signal.y, replay.to_numpy(signal.x), atol=1e-12)
        assert report.provenance.seed == (17 if simulator == 'ngspice' else None)
        if branch is not None:
            np.testing.assert_allclose(branch.y, -wave.to_numpy(branch.x) / 1000, atol=2e-8)
            deck = Path(tb.netlist._run_files.netlist_path).read_bytes()
            assert report.provenance.deck_sha256 == hashlib.sha256(deck).hexdigest()
    assert reports[0].parameters['stimulus']['v2'] == 1.25
    assert reports[1].parameters['stimulus']['v2'] == 2
    assert len(dut.list_characterization_profiles()) == 2
    assert tb.parameters['stimulus']['v2'] is level
    # Literal fields are snapshots: changing the waveform requires a new declaration.
    snapshot = tb.parameters['stimulus']
    if kind == 'bitpattern':
        wave.bits = '0001'
        assert snapshot['bits'] == '1010'
    else:
        wave.seed = 11
        assert snapshot['seed'] == 3
    replacement = dut.create_test('updated', parameters={'stimulus': wave.as_parameters()})
    assert _replay(replacement.parameters['stimulus']).stream == wave.stream
    assert _replay(snapshot).stream != wave.stream
    dut.close()


@pytest.mark.parametrize('kind,field', [('bitpattern', 'bits'), ('prbs', 'seed')])
@pytest.mark.parametrize('entrypoint', ['circuit', 'testbench'])
def test_structural_strings_cannot_execute_code(tmp_path, kind, field, entrypoint):
    marker = tmp_path / 'executed'
    entry = _entry(kind)
    entry['waveform']['params'][field] = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    spec = dict(circuit_name='bits', pdk='sky130', output_dir=str(tmp_path), sources=[entry])
    with pytest.raises((TypeError, ValueError), match=field):
        if entrypoint == 'circuit':
            _exec(Generator().generate(spec))
        else:
            dut = _module(tmp_path, module_name='dut')
            dut.define_module_ports({'out': 'inout'})
            code = generate_testbenches(dict(circuit_name='bits', tests=[dict(
                name='stream', dut_portmap={'out': 'out'}, sources=[entry])]))
            _exec(code)['_attach_stream'](dut)
    assert not marker.exists()
