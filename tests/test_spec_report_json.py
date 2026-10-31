"""Standalone records preserve complete evidence without a live circuit."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from silk.circuit.spec_report import SpecReport, SpecResult
from silk.db.context import CharacterizationContext
from silk.metrics.diagnostics import MetricWarningRecord
from silk.simulation.provenance import RunProvenance
from silk.signals.rawfile import RawfileError
from silk.simulation.conditions import ConditionState, OperatingConditions, ConditionSweep

FIXTURES = Path(__file__).parent / 'fixtures/reports'


def complete_report():
    report = SpecReport(
        'bias', 'mirror', {
            'reference_current_A': np.float64(1e-5), 'gain': 'Error: missing signal',
            'first': float('nan'), 'second': float('nan'),
            'nan': float('nan'), 'positive': float('inf'), 'negative': -float('inf'),
            'complex': complex(-0.0, float('inf')),
            'literal': {'status': 'complex', 're': 1, 'im': 2},
        }, owner_path='$/left', parameters={'ratio': 2},
        characterization_context=CharacterizationContext(
            '$/left', 'bias', {'ratio': 2}, 'tt', 27, ConditionState.NONE),
        operating_conditions=OperatingConditions(
            'external', 'ngspice', 'tt', temperature_kind='fixed', temperature_c=27,
            temperature_basis='caller-declared', corner_basis='caller-declared',
            supply_kind='none', supply_basis='caller-declared'),
        corner='tt', rawfile_path='missing/runs/tt/bias.raw',
        metric_warnings={name: (MetricWarningRecord(
            'average', 'unavailable', f'{name} unavailable', {'value': float('nan')},
            'Inspect the signal.', name),) for name in ('first', 'second')},
        op_devices=[{'path': 'xdut.m1', 'name': 'm1', 'model': 'nmos', 'polarity': 'n',
                     'w_um': 4., 'l_um': .5, 'm': 1., 'declared_params': ('gm', 'id'),
                     'op_values': {'gm': .001, 'id': 1e-5}}],
        provenance=RunProvenance(
            run_origin='external-rawfile', silk_version='0.2.0', silk_git='fixture',
            python_version='3.12.3', numpy_version='2.5.3', simulator='ngspice',
            pdk_name='sky130', pdk_libraries=[{'file': 'models.lib', 'section': 'tt',
                'file_sha256': 'a' * 64, 'section_text_sha256': 'b' * 64}],
            corner='tt', corner_kind='fixed', temperature_c=27., temperature_kind='fixed',
            supply_kind='none', simulator_options={'reltol': .001}, save_list={'voltage': ['out']},
            deck_sha256='c' * 64, seed=42, seedinfo='fixture', timestamp_utc='2026-10-06T00:00:00Z'),
        behavioral_elements=('rload: ignored load',))
    report.report_key = '$/left:mirror.bias'
    report.metric_types = {key: None for key in report.metrics}
    for name in ('reference_current_A', 'first', 'second'):
        report.metric_types[name] = 'average'
    report.metric_errors = {'gain': {'message': 'missing signal', 'exception_type': 'ValueError',
                                      'available_signals': ['time', 'v(out)']}}
    report.storage = {'metrics_stored': 7, 'specs_stored': 2, 'skipped': {}, 'cached': False}
    report.check('current lower bound', 'reference_current_A', '>=', 1e-6,
                 spec_key='current_min', owner_path='$/left')
    report.check('current upper bound', 'reference_current_A', '<', 1e-6)
    return report


def test_complete_record_offline_roundtrip(tmp_path):
    report = complete_report()
    wire = report.to_dict()
    assert wire['format'] == 'silk.spec-report'
    assert wire['schema_version'] == 1
    assert wire['metrics']['nan'] == {'value': None, 'status': 'unavailable'}
    assert wire['metrics']['literal'] == {'status': 'literal', 'value': {
        'status': 'complex', 're': 1, 'im': 2}}
    path = tmp_path / 'record.json'
    report.save_json(path)
    assert json.loads(path.read_text()) == wire
    restored = SpecReport.load_json(path)
    assert restored.to_dict() == wire
    assert math.isnan(restored.metrics['nan'])
    assert restored.metrics['complex'].imag == float('inf')
    assert math.copysign(1, restored.metrics['complex'].real) == -1
    assert isinstance(restored.results[0], SpecResult)
    assert restored.characterization_context.voltage is ConditionState.NONE
    assert isinstance(restored.metric_warnings['first'], tuple)
    assert restored.provenance.timestamp_utc == '2026-10-06T00:00:00Z'
    assert restored.op_table() is not None
    assert not restored.passed
    assert 'FAIL' in restored.summary()
    wire['parameters']['ratio'] = 99
    assert restored.parameters == report.parameters == {'ratio': 2}


def test_manual_defaults_and_successful_error_string():
    report = SpecReport('t', 'm', {'text': 'Error: successful custom result'})
    wire = report.to_dict()
    assert wire['metric_types'] == {'text': None}
    assert wire['metric_errors'] == {}
    assert wire['report_key'] is None
    assert wire['passed'] is True
    assert SpecReport.from_dict(wire).metrics == report.metrics


@pytest.mark.parametrize('value', [np.array([1]), object(), {1: 2}])
def test_unsupported_export_preserves_destination(tmp_path, value):
    report = SpecReport('t', 'm', {'bad': value})
    path = tmp_path / 'report.json'
    path.write_text('keep me')
    with pytest.raises((TypeError, ValueError), match='metrics.bad'):
        report.save_json(path)
    assert path.read_text() == 'keep me'


def test_circular_export_has_field_path():
    cycle = []
    cycle.append(cycle)
    report = SpecReport('t', 'm', {'cycle': cycle})
    with pytest.raises(ValueError, match='metrics.cycle'):
        report.to_dict()


@pytest.mark.parametrize(('field', 'value'), [
    ('format', 'wrong'), ('schema_version', 2), ('schema_version', True),
    ('metrics', []), ('metric_types', {}), ('metric_errors', []),
    ('passed', True), ('report_key', 2), ('behavioral_elements', [2]),
    ('results', [{'name': 'bad'}]), ('op_devices', [{}]),
    ('operating_conditions', {}), ('provenance', {}),
])
def test_reject_malformed_field(field, value):
    wire = complete_report().to_dict()
    wire[field] = value
    with pytest.raises(ValueError, match=field):
        SpecReport.from_dict(wire)


def test_missing_required_fields_and_unknown_additive_keys():
    wire = complete_report().to_dict()
    for field in wire:
        bad = dict(wire)
        del bad[field]
        with pytest.raises(ValueError, match=field):
            SpecReport.from_dict(bad)
    wire['future'] = {'extra': True}
    assert 'future' not in SpecReport.from_dict(wire).to_dict()


def test_reader_preserves_checks_without_rechecking(monkeypatch):
    wire = complete_report().to_dict()
    def refuse(*args, **kwargs):
        raise AssertionError('must not recheck')
    monkeypatch.setattr(SpecReport, 'check', refuse)
    assert not SpecReport.from_dict(wire).passed
    wire['results'][0]['op'] = '!='
    with pytest.raises(ValueError, match=r'results\[0\].op'):
        SpecReport.from_dict(wire)


def test_load_errors_name_file_and_field(tmp_path):
    path = tmp_path / 'broken.json'
    path.write_text('{"format": "wrong"}')
    with pytest.raises(ValueError, match='broken.json.*format'):
        SpecReport.load_json(path)
    path.write_text('{broken')
    with pytest.raises(ValueError, match='broken.json'):
        SpecReport.load_json(path)


@pytest.mark.parametrize('conditions', [
    OperatingConditions('external', None, None),
    OperatingConditions('external', 'ngspice', 'none', temperature_kind='sweep',
                        temperature_sweep=ConditionSweep(-40, 125, 5),
                        temperature_basis='configured'),
    OperatingConditions('silk', 'ngspice', None, supply_kind='sweep',
                        supply_sweep=ConditionSweep(0, 2, 0.5),
                        supply_basis='configured', supply_source='rail'),
])
def test_unknown_and_swept_conditions(conditions):
    report = SpecReport('t', 'm', {}, operating_conditions=conditions)
    restored = SpecReport.from_dict(report.to_dict())
    assert restored.operating_conditions == conditions
    assert restored.characterization_context is None


@pytest.fixture
def bench(tmp_path):
    from silk import Module, sources
    from silk.elements import elements
    from silk.simulation import Transient
    mod = Module(None, str(tmp_path), 'dut.cir', None, module_name='mirror')
    mod.define_module_ports({'out': 'out', 'vss': 'ground'})
    mod.add(elements.resistor('r', p='out', n='vss', value=1000))
    tb = mod.create_test('bias')
    tb.add_module_under_test({'out': 'out', 'vss': '0'})
    tb.add_source(sources.VoltageSource('supply', positive='out', negative='0', dc=1.8))
    tb.add_sim_type(Transient(step=1e-6, stop=2e-6))
    yield tb
    mod.close()


RAW = Path(__file__).parent / 'fixtures/rawfiles/e_ng_tran.raw'


@pytest.mark.parametrize('prepared', [False, True])
def test_immediate_evaluate_metadata(bench, prepared, tmp_path):
    from silk.metrics import average
    def broken(c):
        raise LookupError('exact failure: Error: keep this')
    handle = bench.add_metric('declared', type='average', signal='out')
    bench.add_metric('function', average, sig=bench.probe('out'))
    bench.add_metric('custom', lambda c: complex(1, 2))
    bench.add_metric('derived', handle * 2)
    bench.add_metric('string', lambda c: 'Error: returned successfully')
    bench.add_metric('broken', broken)
    bench.add_metric('broken_again', broken)
    if prepared:
        assert Path(bench.prepare()).is_file()
    raw = tmp_path / 'external.raw'
    raw.write_bytes(RAW.read_bytes())
    report = bench.evaluate(rawfile=raw)
    raw.unlink()
    wire = report.to_dict()
    assert wire['report_key'] == 'mirror.bias'
    assert wire['metric_types'] == {'declared': 'average', 'function': 'average',
        'custom': None, 'derived': None, 'string': None, 'broken': None, 'broken_again': None}
    assert wire['metric_errors']['broken'] == {
        'message': 'exact failure: Error: keep this', 'exception_type': 'LookupError',
        'available_signals': ['time', 'v(clk)', 'v(out)', 'v(step)', 'v(tone)', 'i(v1)']}
    assert wire['metric_errors']['broken_again'] == wire['metric_errors']['broken']
    assert 'string' not in wire['metric_errors']
    assert wire['metrics']['broken'] == 'Error: exact failure: Error: keep this'
    assert wire['provenance']['run_origin'] == 'external-rawfile'
    assert wire['provenance']['simulator_binary'] is None
    assert (wire['provenance']['deck_sha256'] is not None) == prepared
    assert SpecReport.from_dict(wire).to_dict() == wire


@pytest.mark.parametrize('inventory', [None, []])
def test_error_inventory_failure_does_not_mask_extractor(bench, monkeypatch, inventory):
    def names():
        if inventory is None:
            raise RuntimeError('inventory unavailable')
        return inventory
    monkeypatch.setattr(bench.netlist, 'get_signal_names', names)
    def broken(c):
        raise ValueError('extractor failed')
    bench.add_metric('bad', broken)
    assert bench.evaluate(rawfile=RAW).to_dict()['metric_errors'] == {'bad': {
        'message': 'extractor failed', 'exception_type': 'ValueError',
        'available_signals': inventory}}


def test_expanded_type_and_error_metadata(bench, monkeypatch):
    from dataclasses import dataclass
    from silk.signals import schema as S
    monkeypatch.setattr(S, '_REGISTRY', S._REGISTRY.copy())
    @dataclass
    class Pair:
        height: float
        width: float
    @S.metric(domain='tran', unit='', outputs={'height': 'V', 'width': 's'})
    def pair(sig: S.Transient) -> Pair:
        return Pair(1., 2.)
    @S.metric(domain='tran', unit='', outputs={'height': 'V', 'width': 's'})
    def broken(sig: S.Transient) -> Pair:
        raise ValueError('pair failed')
    bench.add_metric('good', pair, sig=bench.probe('out'))
    bench.add_metric('bad', broken, sig=bench.probe('out'))
    report = bench.evaluate(rawfile=RAW)
    assert report.metric_types == {
        'good': 'pair', 'good.height': 'pair', 'good.width': 'pair',
        'bad': 'broken', 'bad.height': 'broken', 'bad.width': 'broken'}
    assert set(report.metric_errors) == {'bad', 'bad.height', 'bad.width'}
    assert all(e['message'] == 'pair failed' for e in report.metric_errors.values())
    report.metric_errors['bad']['available_signals'].append('mutation')
    assert 'mutation' not in report.metric_errors['bad.width']['available_signals']


def test_duplicate_placement_keys_survive_selection_and_failure(bench, tmp_path):
    from silk import Module
    top = Module(None, str(tmp_path), 'top.cir', None)
    try:
        top.add_module(bench.module, 'left', out='a', vss='0')
        top.add_module(bench.module, 'right', out='b', vss='0')
        entries = top.list_test_entries()
        reports = [entry.test.evaluate(rawfile=RAW) for entry in entries]
        assert [r.to_dict()['report_key'] for r in reports] == [
            '$/left:mirror.bias', '$/right:mirror.bias']
        assert [r.report_key for r in reports] == [entry.report_key for entry in entries]
        with pytest.raises(RawfileError):
            entries[1].test.evaluate(rawfile=tmp_path / 'absent.raw')
        assert reports[0].to_dict()['report_key'] == '$/left:mirror.bias'
        assert bench.evaluate(rawfile=RAW).report_key == 'mirror.bias'
        assert [r.to_dict()['owner_path'] for r in reports] == ['$/left', '$/right']
    finally:
        top.close()


def test_same_type_warning_outputs_survive_export(bench):
    from silk.metrics import rise_time
    from silk.signals.base import Signal
    sig = Signal('flat', 'voltage', 'time', 'voltage', np.arange(5.), np.ones(5))
    bench.add_metric('first', rise_time, sig=sig)
    bench.add_metric('second', rise_time, sig=sig)
    with pytest.warns(UserWarning, match='zero_swing'):
        report = bench.evaluate(rawfile=RAW)
    wire = report.to_dict()
    assert wire['metric_types'] == {'first': 'rise_time', 'second': 'rise_time'}
    assert list(wire['metric_warnings']) == ['first', 'second']
    for name in ('first', 'second'):
        assert wire['metric_warnings'][name][0]['output'] == name
        assert wire['metric_warnings'][name][0]['metric'] == 'rise_time'
    assert SpecReport.from_dict(wire).to_dict() == wire


@pytest.mark.parametrize('no_op', [False, True])
def test_consumer_golden_contract(no_op, tmp_path):
    from dataclasses import replace
    name = 'spec_report_v1_no_op_wire.json' if no_op else 'spec_report_v1_wire.json'
    wire = json.loads((FIXTURES / name).read_text())
    report = complete_report()
    # The frozen consumer fixture retains its historical provenance scheme.
    report.provenance = replace(report.provenance, scheme='silk.provenance/v2')
    if no_op:
        report.op_devices = None
        report._op_unavailable_reason = 'Device parameters were not captured.'
    assert report.to_dict() == wire
    loaded = SpecReport.load_json(FIXTURES / name)
    assert loaded.to_dict() == wire
    path = tmp_path / name
    loaded.save_json(path)
    assert json.loads(path.read_text()) == wire
    if no_op:
        from silk.exceptions import SimulationError
        with pytest.raises(SimulationError, match='Device parameters were not captured'):
            loaded.op_table()
    else:
        assert loaded.op_devices[0]['op_values']['gm'] == .001
    assert wire['metric_errors']['gain']['message'] == 'missing signal'
    assert wire['metrics']['gain'] == 'Error: missing signal'
    assert wire['metrics']['positive']['sign'] == '+'
    assert wire['metrics']['negative']['sign'] == '-'
    assert wire['metric_warnings']['first'][0]['output'] == 'first'
    assert wire['metric_warnings']['second'][0]['output'] == 'second'
    assert [r['passed'] for r in wire['results']] == [True, False]


def test_numpy_scalars_and_nested_tag_collisions():
    from silk._json_codec import encode_json
    metrics = {'integer': np.int64(7), 'boolean': np.bool_(True),
               'tuple': (np.float32(1), np.complex64(2 + 3j)),
               'literal': {'status': 'literal', 'value': {
                   'status': 'complex', 're': float('nan'), 'im': float('inf')}},
               'nonfinite_literal': {'value': None, 'status': 'unavailable'}}
    report = SpecReport('t', 'm', metrics)
    wire = report.to_dict()
    assert wire['metrics']['integer'] == 7
    assert wire['metrics']['boolean'] is True
    assert wire['metrics']['tuple'] == [1., {'status': 'complex', 're': 2., 'im': 3.}]
    restored = SpecReport.from_dict(wire)
    assert restored.to_dict() == wire
    assert isinstance(restored.metrics['literal'], dict)
    assert restored.metrics['nonfinite_literal'] == {'value': None, 'status': 'unavailable'}
    # The default codec must still treat a complex-shaped dictionary literally.
    assert encode_json({'status': 'complex', 're': 2., 'im': 3.}) == {
        'status': 'complex', 're': 2., 'im': 3.}


def test_provenance_uses_strict_numeric_boundary():
    report = SpecReport('t', 'm', {}, provenance=RunProvenance(
        simulator_options={'nested': {'z': complex(1, float('nan'))}}))
    wire = report.to_dict()
    assert wire['provenance']['simulator_options']['nested']['z'] == {
        'status': 'complex', 're': 1., 'im': {'value': None, 'status': 'unavailable'}}
    assert math.isnan(SpecReport.from_dict(wire).provenance.simulator_options['nested']['z'].imag)


@pytest.mark.parametrize('value', [object(), {1: 'bad key'}, np.array([1.])])
def test_provenance_unsupported_data_is_not_stringified(value):
    report = SpecReport('t', 'm', {}, provenance=RunProvenance(simulator_options={'bad': value}))
    with pytest.raises((TypeError, ValueError), match='provenance.simulator_options.bad'):
        report.to_dict()


@pytest.mark.parametrize(('field', 'value'), [('owner_path', 4), ('test_name', ''),
    ('parameters', []), ('temperature', None), ('voltage', {'state': 'sweep'})])
def test_malformed_context_errors_identify_file_and_field(field, value, tmp_path):
    wire = complete_report().to_dict()
    wire['characterization_context'][field] = value
    path = tmp_path / 'context.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError, match='context.json.*characterization_context'):
        SpecReport.load_json(path)


def test_loaded_condition_kinds_are_never_inferred():
    wire = complete_report().to_dict()
    wire['operating_conditions']['corner_kind'] = 'unknown'
    with pytest.raises(ValueError, match='operating_conditions.corner_kind'):
        SpecReport.from_dict(wire)


@pytest.mark.parametrize(('field', 'value'), [
    ('metrics_stored', 'all'), ('specs_stored', True), ('skipped', []), ('cached', 'false')])
def test_malformed_receipt(field, value):
    wire = complete_report().to_dict()
    wire['storage'][field] = value
    with pytest.raises(ValueError, match='storage.' + field):
        SpecReport.from_dict(wire)


def test_empty_capture_is_distinct_from_uncaptured():
    report = SpecReport('t', 'm', {}, op_devices=[])
    loaded = SpecReport.from_dict(report.to_dict())
    assert loaded.op_devices == []
    assert loaded.op_table() is not None


def test_unsupported_result_object_is_not_silently_dropped(bench, monkeypatch):
    from dataclasses import dataclass
    from silk.signals import schema as S
    monkeypatch.setattr(S, '_REGISTRY', S._REGISTRY.copy())
    @dataclass
    class Pair:
        height: float
    @S.metric(domain='tran', unit='', outputs={'height': 'V'})
    def pair(sig: S.Transient) -> Pair:
        return Pair(1.)
    bench.add_metric('eye', pair, sig=bench.probe('out'))
    report = bench.evaluate(rawfile=RAW)
    assert report.metrics['eye.height'] == 1.
    with pytest.raises(TypeError, match='metrics.eye: unsupported Pair'):
        report.to_dict()


def test_numeric_tags_are_restored_at_all_supported_nesting_levels():
    report = complete_report()
    report.storage['future'] = {'value': complex(1, float('inf'))}
    report.parameters['future'] = (float('nan'), {'status': 'complex', 're': 1, 'im': 2})
    report.op_devices[0]['op_values']['gds'] = float('nan')
    report._results[0].target = float('inf')
    report._results[0].value = float('nan')
    wire = report.to_dict()
    restored = SpecReport.from_dict(wire)
    assert restored.to_dict() == wire
    assert math.isnan(restored.results[0].value)
    assert math.isnan(restored.op_devices[0]['op_values']['gds'])
    assert restored.storage['future']['value'] == complex(1, float('inf'))


def test_loaded_data_is_detached_from_wire():
    wire = complete_report().to_dict()
    restored = SpecReport.from_dict(wire)
    wire['metric_errors']['gain']['available_signals'].append('later')
    wire['op_devices'][0]['op_values']['gm'] = 99
    wire['provenance']['simulator_options']['reltol'] = 1
    assert restored.metric_errors['gain']['available_signals'] == ['time', 'v(out)']
    assert restored.op_devices[0]['op_values']['gm'] == .001
    assert restored.provenance.simulator_options['reltol'] == .001
