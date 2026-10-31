"""Deck provenance stays optional in historical offline report records."""
import json
from pathlib import Path

import pytest

from silk.circuit.report import generate_report
from silk.circuit.spec_report import SpecReport
from silk.simulation.provenance import RunProvenance
from silk import Module, load_pdk

FIXTURE = Path(__file__).parents[1] / 'fixtures/reports/spec_report_v1_wire.json'


@pytest.mark.parametrize('scheme', ['silk.provenance/v1', 'silk.provenance/v2'])
@pytest.mark.parametrize('aggregate', [False, True])
def test_old_provenance_roundtrips_without_backfilling_launch_facts(tmp_path, scheme, aggregate):
    wire = json.loads(FIXTURE.read_text())
    wire['provenance']['scheme'] = scheme
    if scheme.endswith('/v1'):
        for name in ('corner_kind', 'supply_kind'):
            wire['provenance'].pop(name)
    restored = SpecReport.from_dict(wire)
    assert restored.provenance.normalized_deck_sha256 is None
    assert restored.provenance.producing_test is None
    assert restored.to_dict() == wire
    if aggregate:
        module = Module('sky130', str(tmp_path), 'test.cir', pdk_details=load_pdk('sky130'))
        report = generate_report(module, spec_reports=[restored], auto_discover_config=False)
        path = tmp_path / 'report.json'
        report.save_json(path)
        exported = json.loads(path.read_text())
        assert exported['provenance'][0] == wire['provenance']
        assert exported['test_results']['entries'][0] == wire


def test_current_producing_test_binding_roundtrips_without_a_verdict():
    provenance = RunProvenance(run_origin='silk-launch', simulator='ngspice',
        normalized_deck_sha256='silk.normalized-deck/v1:' + 'a' * 64,
        producing_test={'owner_path': '$/left', 'test_name': 'bias'}, launch_corner_label='ff')
    report = SpecReport('bias', 'm', {}, provenance=provenance)
    wire = report.to_dict()
    assert SpecReport.from_dict(wire).provenance == provenance
    assert 'deck_check' not in wire['provenance']


def test_malformed_producing_binding_keeps_field_diagnostic():
    report = SpecReport('bias', 'm', {}, provenance=RunProvenance(
        producing_test={'owner_path': '$', 'test_name': 7}))
    with pytest.raises(ValueError, match=r'provenance.producing_test.test_name'):
        SpecReport.from_dict(report.to_dict())


LEGACY_DIGESTS = json.loads((Path(__file__).parent / 'fixtures/legacy_provenance_digests.json').read_text())['records']


@pytest.mark.parametrize('frozen', LEGACY_DIGESTS, ids=['v1', 'v2'])
def test_historical_digest_matches_f83d966d_exactly(frozen):
    from dataclasses import replace
    record = RunProvenance.from_dict(frozen['record'])
    assert record.digest() == frozen['digest']
    assert record.to_dict() == frozen['record']
    # Producing facts belong to v3 even if a caller adds them to an old object.
    altered = replace(record, normalized_deck_sha256='silk.normalized-deck/v1:' + 'b' * 64,
                      producing_test={'owner_path': '$', 'test_name': 'bias'}, launch_corner_label='ss')
    assert altered.digest() == frozen['digest']
    assert altered.to_dict() == frozen['record']


@pytest.mark.parametrize('frozen', LEGACY_DIGESTS, ids=['v1', 'v2'])
def test_historical_provenance_load_restore_deduplicates_without_rewriting(tmp_path, frozen):
    from dataclasses import replace
    from silk.db import Database
    db = Database(str(tmp_path / 'history.db'))
    try:
        old_id = 'historical-record'
        record_json = json.dumps(frozen['record'], sort_keys=True, separators=(',', ':'))
        db.conn.execute('INSERT INTO run_provenance (provenance_id, scheme, digest, record_json) VALUES (?, ?, ?, ?)',
                        (old_id, frozen['record']['scheme'], frozen['digest'], record_json))
        db.conn.commit()
        before = [tuple(row) for row in db.conn.execute('SELECT * FROM run_provenance')]
        restored = db.get_run_provenance(old_id)
        assert db.store_run_provenance(restored) == old_id
        assert db.store_run_provenance(replace(restored, timestamp_utc='2030-01-01T00:00:00Z')) == old_id
        assert [tuple(row) for row in db.conn.execute('SELECT * FROM run_provenance')] == before
    finally:
        db.close()


@pytest.mark.parametrize('changes', [
    {'normalized_deck_sha256': 'silk.normalized-deck/v1:' + 'b' * 64},
    {'producing_test': {'owner_path': '$/right', 'test_name': 'bias'}},
    {'launch_corner_label': 'ss'},
])
def test_v3_digest_remains_sensitive_to_each_producing_fact(changes):
    from dataclasses import replace
    record = RunProvenance(normalized_deck_sha256='silk.normalized-deck/v1:' + 'a' * 64,
                           producing_test={'owner_path': '$/left', 'test_name': 'bias'},
                           launch_corner_label='tt')
    assert replace(record, **changes).digest() != record.digest()


@pytest.mark.parametrize('frozen', LEGACY_DIGESTS, ids=['v1', 'v2'])
def test_provenance_diff_crosses_schemes_without_missing_field_errors(frozen):
    from dataclasses import replace
    old = RunProvenance.from_dict(frozen['record'])
    producing_test = {'owner_path': '$', 'test_name': 'bias'}
    digest = 'silk.normalized-deck/v1:' + 'a' * 64
    new = replace(old, scheme='silk.provenance/v3', normalized_deck_sha256=digest,
                  producing_test=producing_test, launch_corner_label='ss')
    assert old.diff(new) == [
        ('scheme', old.scheme, new.scheme),
        ('normalized_deck_sha256', None, digest),
        ('producing_test', None, producing_test),
        ('launch_corner_label', None, 'ss'),
    ]
