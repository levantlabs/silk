"""Explicit aggregate save retains identity through database restoration."""
from dataclasses import replace

import pytest

from silk.db import Database
from silk.specs import Measurement
from tests.db.test_spec_restore import _build, _report
from tests.test_aggregate_measurement import aggregate


def test_explicit_save_reopen_restore(pdk, tmp_path):
    path = tmp_path / 'aggregate.db'
    module, gain, _ = _build(pdk, tmp_path, path)
    version = module.create_circuit_version()
    record = next(s for s in module.get_stored_specs() if s.spec_key == gain.key)
    a = aggregate()
    m = Measurement(source='spice', value=2, aggregate=a, sim_run_id=a.campaign_id)
    gain.record_measurement(m)
    module._tracker.db.insert_spec_measurement('aggregate', record.spec_id, 'spice', 2,
        aggregate=a, sim_run_id=a.campaign_id)
    module.close()
    with Database(path) as db:
        stored = next(s for s in db.get_stored_specs(version) if s.spec_key == gain.key)
        assert stored.measurements[0].aggregate == a
    rebuilt, restored_gain, _ = _build(pdk, tmp_path, path)
    assert rebuilt.create_circuit_version() == version
    rebuilt.restore_spec_measurements()
    assert restored_gain.latest().aggregate == a
    rebuilt.close()


@pytest.mark.parametrize('change', ['metric', 'run'])
def test_mismatched_identity_refuses_before_insert(pdk, tmp_path, change):
    module, gain, _ = _build(pdk, tmp_path, tmp_path / 'bad.db')
    module.create_circuit_version()
    record = next(s for s in module.get_stored_specs() if s.spec_key == gain.key)
    a = aggregate(metric='wrong' if change == 'metric' else 'gain_dB')
    db = module._tracker.db
    with pytest.raises(ValueError, match='metric|run|campaign'):
        db.insert_spec_measurement('bad', record.spec_id, 'spice', 2, aggregate=a,
            sim_run_id='wrong' if change == 'run' else a.campaign_id)
    assert db.conn.execute('SELECT count(*) FROM spec_measurements').fetchone()[0] == 0
    module.close()


def test_same_number_ordinary_report_does_not_borrow_aggregate(pdk, tmp_path):
    module, gain, _ = _build(pdk, tmp_path, tmp_path / 'ordinary.db')
    module.create_circuit_version()
    a = aggregate()
    gain.record_measurement(Measurement(source='spice', value=2,
        aggregate=a, sim_run_id=a.campaign_id))
    module.store_report(_report((gain, 2)), source='spice')
    record = next(s for s in module.get_stored_specs() if s.spec_key == gain.key)
    assert record.measurements[0].aggregate is None
    assert record.measurements[0].sim_run_id != a.campaign_id
    module.close()


def test_malformed_stored_aggregate_reports_measurement_id(pdk, tmp_path):
    path = tmp_path / 'corrupt.db'
    module, gain, _ = _build(pdk, tmp_path, path)
    module.create_circuit_version()
    record = next(s for s in module.get_stored_specs() if s.spec_key == gain.key)
    db = module._tracker.db
    db.insert_spec_measurement('broken-evidence', record.spec_id, 'spice', 2,
        aggregate=aggregate(), sim_run_id='campaign-1')
    db.conn.execute("UPDATE spec_measurements SET aggregate_json='{}'")
    db.conn.commit()
    module.close()
    from silk.exceptions import DatabaseCompatibilityError
    with pytest.raises(DatabaseCompatibilityError, match='broken-evidence.*aggregate'):
        Database(path)


def test_profile_restore_and_report_keep_explicit_aggregate(pdk, tmp_path):
    from tests.db.test_characterization_readback import _build_live, _context, _store_live_context
    from silk.circuit.report import generate_report, CircuitReport
    import time
    path = tmp_path / 'profile.db'
    context = _context()
    original, gain = _build_live(pdk, tmp_path, path)
    version, profile, _ = _store_live_context(original, gain, context)
    record = next(s for s in original.get_stored_specs() if s.spec_key == gain.key)
    a = aggregate()
    original._tracker.db.insert_spec_measurement('explicit-profile-aggregate', record.spec_id,
        'spice', 2, corner=context.corner, temperature=context.temperature,
        voltage=context.voltage, profile_id=profile.profile_id,
        timestamp=time.time()+1, aggregate=a, sim_run_id=a.campaign_id)
    original.close()
    rebuilt, rebuilt_gain = _build_live(pdk, tmp_path, path)
    assert rebuilt.create_circuit_version() == version
    rebuilt.restore_characterization_evidence(context=context)
    restored = next(iter(rebuilt_gain.measurements.values()))
    assert restored.aggregate == a
    report = generate_report(rebuilt, auto_discover_config=False)
    report.save_json(tmp_path / 'restored.json')
    loaded = CircuitReport.load_json(tmp_path / 'restored.json')
    assert loaded.spec_reconciliation.outcomes[0].measurement.aggregate == a
    rebuilt.close()


@pytest.mark.parametrize('field,bad', [('metric', 'unrelated_metric'),
                                       ('campaign_id', 'another_campaign')])
def test_live_readback_refuses_contradictory_aggregate(pdk, tmp_path, field, bad):
    import json
    from silk.specs.aggregate import encode_aggregate
    module, gain, _ = _build(pdk, tmp_path, tmp_path / 'live-corruption.db')
    version = module.create_circuit_version()
    record = next(s for s in module.get_stored_specs() if s.spec_key == gain.key)
    db = module._tracker.db
    try:
        db.insert_spec_measurement('changed-identity', record.spec_id, 'spice', 2,
                                   aggregate=aggregate(), sim_run_id='campaign-1')
        payload = encode_aggregate(aggregate())
        payload[field] = bad
        db.conn.execute('UPDATE spec_measurements SET aggregate_json=? WHERE measurement_id=?',
                        (json.dumps(payload), 'changed-identity'))
        with pytest.raises(ValueError, match='changed-identity.*aggregate.*' + field):
            db.get_stored_specs(version)
    finally:
        module.close()


@pytest.mark.parametrize('evidence', [None, aggregate()])
def test_live_readback_valid_and_ordinary_controls(pdk, tmp_path, evidence):
    module, gain, _ = _build(pdk, tmp_path, tmp_path / 'live-control.db')
    version = module.create_circuit_version()
    record = next(s for s in module.get_stored_specs() if s.spec_key == gain.key)
    try:
        db = module._tracker.db
        db.insert_spec_measurement('control', record.spec_id, 'spice', 2,
                                   aggregate=evidence, sim_run_id='campaign-1')
        stored = next(s for s in db.get_stored_specs(version) if s.spec_key == gain.key)
        assert stored.measurements[0].aggregate == evidence
    finally:
        module.close()
