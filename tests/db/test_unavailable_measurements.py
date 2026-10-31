"""Unavailable observations survive report storage, typed queries and restore."""

import json
import math

import pytest

from silk import Module
from silk.circuit.report import CircuitReport, generate_report
from silk.circuit.spec_report import SpecReport
from silk.db import Database
from silk.db.context import CharacterizationContext
from silk.exceptions import EvidenceMismatchError
from silk.specs import Measurement
from tests.db_snapshot_helpers import create_sealed_snapshot


def _build(pdk, tmp_path, db_path, *, mixed=False):
    mod = Module(
        'sky130', str(tmp_path / 'review'), 'review.cir', pdk_details=pdk,
        module_name='review', db_path=str(db_path),
    )
    mod.add_net('out')
    specs = [mod.declare_spec(
        'UGF upper bound', 'ugf', '<', 100e6, units='Hz', key='ugf'
    )]
    if mixed:
        for name in ('finite', 'zero', 'error', 'unmeasured'):
            specs.append(mod.declare_spec(name, name, '<', 100e6, key=name))
    mod.create_circuit_version()
    return mod, specs


def _report(specs, values, *, context=None):
    report = SpecReport(
        'ac', 'review', dict(zip((s.metric for s in specs), values)),
        corner='tt', characterization_context=context,
    )
    for spec in specs[:len(values)]:
        report.check(
            spec.name, spec.metric, spec.op, spec.target,
            spec_key=spec.key, owner_path='$',
        )
    return report


def _counts(mod):
    return [mod._tracker.db.conn.execute(
        f'SELECT COUNT(*) FROM {table}'
    ).fetchone()[0] for table in ('simulation_results', 'spec_measurements')]


def test_design_appendix_a2_nan_report_saves_and_restores(pdk, tmp_path):
    mod, (spec,) = _build(pdk, tmp_path, ':memory:')
    try:
        report = _report([spec], [float('nan')])
        spec.record_measurement(Measurement(
            source='spice', value=float('nan'), passes=False, corner='tt',
        ))
        mod.store_report(report)
        assert _counts(mod) == [2, 1]
        metric, = mod.query_results(parameter='ugf')
        assert math.isnan(metric.value)
        assert metric.value_text is None
        declaration, = mod.get_stored_specs()
        measurement, = declaration.measurements
        assert math.isnan(measurement.value)
        assert measurement.passes is False
        assert measurement.temperature is measurement.voltage is None
        assert metric.temperature is metric.voltage is None
        spec.measurements.clear()
        assert mod.restore_spec_measurements() == 1
        assert math.isnan(spec.measurements['spice@tt'].value)
        assert mod.reconcile_specs().outcomes[0].status == 'FAIL'
    finally:
        mod.close()


def test_mixed_report_round_trip_keeps_distinctions_and_profile(pdk, tmp_path):
    path = tmp_path / 'mixed.db'
    mod, specs = _build(pdk, tmp_path, path, mixed=True)
    context = CharacterizationContext(
        owner_path='$', test_name='ac', parameters={'load': 1e-12},
        corner='tt', temperature=27.0, voltage=1.8,
    )
    result = mod.store_report(_report(
        specs, [float('nan'), 42.0, 0.0, 'Error: missing signal'], context=context
    ))
    assert result['metrics_stored'] == 3
    assert result['specs_stored'] == 4
    assert set(result['skipped']) == {'error'}
    assert _counts(mod) == [7, 3]
    profile, = mod._tracker.db.list_characterization_profiles(mod._tracker.circuit_id)
    mod.close()

    rebuilt, specs = _build(pdk, tmp_path, path, mixed=True)
    try:
        rows = {r.parameter: r for r in rebuilt.query_results(profile_id=profile.profile_id)}
        assert math.isnan(rows['ugf'].value)
        assert rows['finite'].value == 42.0
        assert rows['zero'].value == 0.0
        # store_report's existing extractor-error channel is a compatibility
        # status row, with no numeric measurement fabricated for that spec.
        assert rows['spec_error'].value is None
        assert rows['spec_error'].value_text == 'fail (error=Error: missing signal < 100000000.0)'
        assert 'error' not in rows
        assert all((r.corner, r.temperature, r.voltage) == ('tt', 27.0, 1.8) for r in rows.values())
        records = {r.spec_key: r for r in rebuilt.get_stored_specs(profile_id=profile.profile_id)}
        assert records['error'].measurements == records['unmeasured'].measurements == ()
        assert math.isnan(records['ugf'].measurements[0].value)
        assert records['ugf'].measurements[0].profile_id == profile.profile_id
        assert rebuilt.restore_spec_measurements(
            profile_id=profile.profile_id, corner='tt', temperature=27.0, voltage=1.8
        ) == 3
        assert math.isnan(next(iter(specs[0].measurements.values())).value)
        assert [o.status for o in rebuilt.reconcile_specs().outcomes] == [
            'FAIL', 'PASS', 'PASS', 'UNMEASURED', 'UNMEASURED'
        ]
    finally:
        rebuilt.close()


def test_nan_mixed_report_still_rolls_back_on_write_failure(pdk, tmp_path, monkeypatch):
    mod, specs = _build(pdk, tmp_path, ':memory:', mixed=True)
    try:
        db = mod._tracker.db
        original = db.insert_spec_measurement
        def fail_after_insert(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError('injected after nan insert')
        monkeypatch.setattr(db, 'insert_spec_measurement', fail_after_insert)
        with pytest.raises(RuntimeError, match='after nan insert'):
            mod.store_report(_report(specs, [float('nan'), 42.0, 0.0, 'Error: missing']))
        assert _counts(mod) == [0, 0]
        assert all(not spec.measurements for spec in specs)
    finally:
        mod.close()


@pytest.mark.parametrize('op', ['<', '<=', '>', '>=', '=='])
def test_nan_cached_pass_validation_remains_enforced(pdk, tmp_path, op):
    mod = Module('sky130', str(tmp_path), 'cache.cir', pdk_details=pdk, db_path=':memory:')
    try:
        spec = mod.declare_spec('nan', 'ugf', op, 1.0, key='ugf')
        mod.create_circuit_version()
        report = _report([spec], [float('nan')])
        report.results[0].passed = True
        with pytest.raises(EvidenceMismatchError, match='cached|pass'):
            mod.store_report(report)
        assert _counts(mod) == [0, 0]
        report.results[0].passed = False
        mod.store_report(report)
        assert mod.get_stored_specs()[0].measurements[0].passes is False
        mod._tracker.db.conn.execute('UPDATE spec_measurements SET passes = 1')
        mod._tracker.db.conn.commit()
        with pytest.raises(EvidenceMismatchError, match='cached|pass'):
            mod.restore_spec_measurements()
    finally:
        mod.close()


def test_numeric_text_missing_and_context_nulls_are_distinct():
    with Database(':memory:') as db:
        version = create_sealed_snapshot(db)
        for parameter, kwargs in (
            ('nan', {'value': float('nan')}), ('zero', {'value': 0.0}),
            ('error', {'value_text': 'Error: missing signal'}),
            ('empty_text', {'value_text': ''}),
        ):
            db.add_simulation_result(version, parameter, **kwargs)
        rows = {r.parameter: r for r in db.get_simulation_results(version)}
        assert math.isnan(rows['nan'].value)
        assert rows['zero'].value == 0.0
        assert rows['error'].value is None
        assert rows['error'].value_text == 'Error: missing signal'
        assert rows['empty_text'].value is None
        assert rows['empty_text'].value_text == ''
        assert all(
            r.temperature is r.voltage is r.corner is r.profile_id is None
            for r in rows.values()
        )
        assert db.get_simulation_results(version, parameter='absent') == []
        with pytest.raises(ValueError, match='exactly one'):
            db.add_simulation_result(version, 'neither')
        with pytest.raises(ValueError, match='exactly one'):
            db.add_simulation_result(version, 'both', value=float('nan'), value_text='error')


def test_report_json_uses_explicit_unavailable_encoding(pdk, tmp_path):
    # TASK-274 deliberately replaces TASK-254 AC5's temporary bare-NaN pin.
    mod, (spec,) = _build(pdk, tmp_path, ':memory:')
    try:
        spec.record_measurement(Measurement(
            source='spice', value=float('nan'), passes=False, corner='tt', timestamp=1.0
        ))
        report = generate_report(mod)
        path = tmp_path / 'report.json'
        report.save_json(path)
        raw = path.read_text()
        def reject_constant(token):
            raise AssertionError(f'Non-standard JSON token: {token}')

        row = json.loads(raw, parse_constant=reject_constant)[
            'spec_reconciliation']['outcomes'][0]
        assert row['status'] == 'FAIL' and row['measurement'] is None
        assert row['latest_attempt']['kind'] == 'unavailable'
        assert row['latest_attempt']['reason']
        assert row['latest_attempt']['timestamp'] == 1.0
        assert row['last_successful_attempt'] is None
        restored = CircuitReport.load_json(path)
        assert restored.spec_reconciliation.outcomes[0].as_dict() == row
        assert math.isnan(spec.latest().value)  # legacy numeric API remains intact
    finally:
        mod.close()


def test_database_json_metadata_is_strict_and_restores_numeric_values(tmp_path):
    path = tmp_path / 'metadata.db'
    metadata = {'result': float('nan'), 'zero': 0.0, 'error': 'Error: missing signal',
                'limits': [float('inf'), -float('inf')],
                'literal': {'value': None, 'status': 'unavailable'}}
    with Database(str(path)) as db:
        version = create_sealed_snapshot(db, metadata=metadata)
        raw = db.conn.execute('SELECT metadata FROM circuit_versions WHERE version_id = ?',
                              (version,)).fetchone()[0]
        def reject(token):
            raise AssertionError(f'Non-standard JSON token: {token}')
        wire = json.loads(raw, parse_constant=reject)
        assert wire['result'] == {'value': None, 'status': 'unavailable'}
    with Database(str(path)) as db:
        restored = db.get_circuit_by_id(version)['metadata']
        assert math.isnan(restored.pop('result'))
        assert {k: restored[k] for k in metadata if k != 'result'} == {
            k: v for k, v in metadata.items() if k != 'result'}
        assert restored['run_count'] == 1


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_simulation_conditions_keep_existing_finite_only_validation(value):
    with pytest.raises(TypeError, match='JSON-compatible'):
        Database._canonical_simulation_conditions({'requested': value})


def test_simulation_conditions_keep_literal_tag_shaped_input():
    conditions = {'value': None, 'status': 'unavailable'}
    raw = Database._canonical_simulation_conditions(conditions)
    assert json.loads(raw) == conditions
    assert Database._decode_simulation_conditions(raw, result_id='run') == conditions
