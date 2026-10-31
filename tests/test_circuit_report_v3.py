"""Complete embedded evidence and the legacy CircuitReport decoder boundary."""
import json
from pathlib import Path

import pytest

from silk.circuit.report import CircuitReport, generate_report
from silk.circuit.spec_report import CornerResults, SpecReport
from silk import Module, load_pdk


def _top_with_instances(tmp_path):
    return Module('sky130', str(tmp_path), 'report.cir',
                  pdk_details=load_pdk('sky130'), module_name='top')

FIXTURES = Path(__file__).parent / 'fixtures' / 'reports'


def test_supplied_evidence_is_embedded_and_restored(tmp_path):
    sr = SpecReport.load_json(FIXTURES / 'spec_report_v1_wire.json')
    results = {'$/left:mirror.bias': CornerResults({'tt': sr}, failed={'ss': 'failed'}, skip_unavailable=True)}
    module = _top_with_instances(tmp_path)
    report = generate_report(module, test_results=results, auto_discover_config=False)
    assert len(report.op_tables) == 1
    assert len(report.metric_warnings) == 2
    assert len(report.provenance) == 1
    path = tmp_path / 'report.json'
    report.save_json(path)
    wire = json.loads(path.read_text())
    assert wire['schema_version'] == 5
    embedded = wire['test_results']['entries'][0]['result']['entries'][0]['result']
    assert embedded == json.loads((FIXTURES / 'spec_report_v1_wire.json').read_text())
    restored = CircuitReport.load_json(path)
    assert restored.test_results['$/left:mirror.bias'].failed == {'ss': 'failed'}
    restored.save_json(tmp_path / 'restored.json')
    assert json.loads((tmp_path / 'restored.json').read_text()) == wire


@pytest.mark.parametrize('kwargs,kind', [({}, None), ({'spec_reports': []}, 'records'),
                                        ({'test_results': {}}, 'tests'),
                                        ({'test_results': CornerResults({})}, 'corners')])
def test_absent_and_empty_evidence_never_run_tests(tmp_path, kwargs, kind):
    module = _top_with_instances(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('report generation must not run or discover tests')
    module.run_all_tests = forbidden
    module.list_tests = forbidden
    report = generate_report(module, auto_discover_config=False, **kwargs)
    path = tmp_path / 'report.json'
    report.save_json(path)
    evidence = json.loads(path.read_text())['test_results']
    if kind is None:
        assert evidence is None
    else:
        assert evidence['kind'] == kind
        assert evidence['entries'] == []


def test_spec_reports_preserve_duplicates_and_reconciliation(tmp_path):
    sr = SpecReport('unspecced', 'm', {'only': 4})
    module = _top_with_instances(tmp_path)
    before = module.reconcile_specs().as_dict()
    report = generate_report(module, spec_reports=[sr, sr], auto_discover_config=False)
    assert [r.metrics['only'] for r in report.test_results] == [4, 4]
    after = report.spec_reconciliation.as_dict()
    before.pop('timestamp')
    after.pop('timestamp')
    assert before == after
    with pytest.raises(ValueError, match='spec_reports.*test_results'):
        generate_report(module, spec_reports=[], test_results={})


@pytest.mark.parametrize('version', [None, 1, 2])
def test_legacy_decoding_unchanged_and_evidence_absent(tmp_path, version):
    report = generate_report(_top_with_instances(tmp_path), auto_discover_config=False)
    path = tmp_path / 'legacy.json'
    report.save_json(path)
    wire = json.loads(path.read_text())
    wire.pop('test_results')
    if version is None:
        wire.pop('schema_version')
    else:
        wire['schema_version'] = version
    literal = {'status': 'complex', 're': 1, 'im': 2}
    wire['stress_coverage'] = {'literal': literal, 'nan': {'value': None, 'status': 'unavailable'}}
    path.write_text(json.dumps(wire))
    loaded = CircuitReport.load_json(path)
    assert loaded.test_results is None
    assert loaded.stress_coverage['literal'] == literal
    if version == 2:
        import math
        assert math.isnan(loaded.stress_coverage['nan'])
    else:
        assert loaded.stress_coverage['nan'] == {'value': None, 'status': 'unavailable'}
    loaded.save_json(tmp_path / 'upgraded.json')
    assert json.loads((tmp_path / 'upgraded.json').read_text())['test_results'] is None


def test_whole_tree_codec_and_invalid_destination(tmp_path):
    report = generate_report(_top_with_instances(tmp_path), auto_discover_config=False)
    literal = {'status': 'complex', 're': 1, 'im': 2}
    report.stress_coverage = {'complex': complex(1, 2), 'literal': literal}
    report.test_results = [SpecReport('t', 'm', {'nested': [literal, complex(3, 4)]})]
    path = tmp_path / 'report.json'
    report.save_json(path)
    restored = CircuitReport.load_json(path)
    assert restored.stress_coverage == report.stress_coverage
    assert restored.test_results[0].metrics['nested'] == [literal, complex(3, 4)]
    old = path.read_bytes()
    report.test_results[0].metrics['bad'] = object()
    with pytest.raises(TypeError):
        report.save_json(path)
    assert path.read_bytes() == old


def test_frozen_v3_fixture(tmp_path):
    source = FIXTURES / 'circuit_report_v3_wire.json'
    report = CircuitReport.load_json(source)
    assert len(report.test_results) == 2
    assert report.test_results[0].op_devices is not None
    assert report.test_results[1].op_devices is None
    path = tmp_path / 'roundtrip.json'
    report.save_json(path)
    expected = json.loads(source.read_text())
    expected['schema_version'] = 5
    assert json.loads(path.read_text()) == expected


def test_nested_invalid_evidence_names_file_and_field(tmp_path):
    report = generate_report(_top_with_instances(tmp_path), spec_reports=[], auto_discover_config=False)
    path = tmp_path / 'invalid-v3.json'
    report.save_json(path)
    wire = json.loads(path.read_text())
    wire['test_results']['schema_version'] = 99
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError, match=r'invalid-v3.json.*test_results.*schema_version'):
        CircuitReport.load_json(path)


def test_nested_provenance_values_preserved_in_both_sections(tmp_path):
    from dataclasses import replace
    import math
    import numpy as np
    sr = SpecReport.load_json(FIXTURES / 'spec_report_v1_wire.json')
    literal = {'value': None, 'status': 'unavailable'}
    sr.provenance = replace(sr.provenance, simulator_options={
        'nested': [literal, np.complex128(complex(-0.0, float('inf')))]})
    report = generate_report(_top_with_instances(tmp_path), spec_reports=[sr], auto_discover_config=False)
    path = tmp_path / 'provenance.json'
    report.save_json(path)
    loaded = CircuitReport.load_json(path)
    for provenance in (loaded.provenance[0], loaded.test_results[0].provenance):
        values = provenance.simulator_options['nested']
        assert values[0] == literal
        assert math.copysign(1, values[1].real) == -1
        assert values[1].imag == float('inf')


@pytest.mark.parametrize('version', ['3', 3.0, True, {}, [], 0, -1])
def test_invalid_circuit_versions_have_file_field_errors(tmp_path, version):
    wire = json.loads((FIXTURES / 'circuit_report_v3_wire.json').read_text())
    wire['schema_version'] = version
    path = tmp_path / 'bad-version.json'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError, match=r'bad-version.json.*schema_version'):
        CircuitReport.load_json(path)


def test_invalid_circuit_root_has_file_error(tmp_path):
    path = tmp_path / 'bad-root.json'
    path.write_text('[]')
    with pytest.raises(ValueError, match=r'bad-root.json.*object'):
        CircuitReport.load_json(path)


def test_circular_provenance_reports_path_before_truncation(tmp_path):
    from silk.simulation.provenance import RunProvenance
    report = CircuitReport.load_json(FIXTURES / 'circuit_report_v3_wire.json')
    circular = {}
    circular['child'] = circular
    report.provenance = [RunProvenance(simulator_options=circular)]
    path = tmp_path / 'keep.json'
    path.write_text('keep')
    with pytest.raises(ValueError, match=r'provenance.*simulator_options.*circular'):
        report.save_json(path)
    assert path.read_text() == 'keep'


@pytest.mark.parametrize('version', [1, 2, 3])
def test_legacy_reconciliation_owner_is_unknown(tmp_path, version):
    from silk.specs import Spec, SpecOutcome, Measurement, SpecReconciliationReport
    module = _top_with_instances(tmp_path)
    report = generate_report(module)
    report.spec_reconciliation = SpecReconciliationReport(module, [SpecOutcome(
        Spec('gain', None, 'gain', '>=', 60), Measurement('user', 70), 'PASS')])
    path = tmp_path / 'legacy-row.json'
    report.save_json(path)
    wire = json.loads(path.read_text())
    wire['schema_version'] = version
    row = wire['spec_reconciliation']['outcomes'][0]
    for key in ('owner_path', 'report_key'):
        row.pop(key, None)
    row['spec'].pop('key', None)
    path.write_text(json.dumps(wire))
    restored = CircuitReport.load_json(path)
    row, = restored.spec_reconciliation.outcomes
    assert row.owner_path is None and row.report_key is None
    assert row.spec.key == 'gain'
    assert '$' not in restored._render_spec_reconciliation_html()
    module.close()
