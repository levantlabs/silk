"""Schema tests for net_current_sets/net_currents (EM-current persistence,
schema v2 — TASK-4 rebuilt these tables per design §6.2/§6.3)."""
import sqlite3
import uuid

import pytest

from silk.db.circuit_db import Database
from tests.db_snapshot_helpers import create_sealed_snapshot


def _mk_version(db):
    return create_sealed_snapshot(db, name='blk', version='1.0')


def _insert_set(db, version_id, *, corner='tt'):
    current_set_id = str(uuid.uuid4())
    db.conn.execute(
        """INSERT INTO net_current_sets
           (current_set_id, circuit_version_id, profile_id, state, corner,
            analysis, tstart, results_file, structure_scheme, structure_hash,
            corner_state, temperature_state, voltage_state)
           VALUES (?, ?, NULL, 'current', ?, 'transient', 0.0, 'x.raw',
                   'silk.structure/v1', 'H1', 'fixed', 'unknown', 'unknown')""",
        (current_set_id, version_id, corner),
    )
    db.conn.commit()
    return current_set_id


def _insert(db, current_set_id, *, net='mid', pin='', avg=1e-3):
    db.conn.execute(
        """INSERT INTO net_currents
           (current_id, current_set_id, net_name, pin_id, avg_a, rms_a,
            peak_a, source, current_vector)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'spice', NULL)""",
        (str(uuid.uuid4()), current_set_id, net, pin, avg, avg, avg),
    )
    db.conn.commit()


def test_net_currents_table_exists_with_columns():
    db = Database(':memory:')
    table_info = {
        row['name']: row
        for row in db.conn.execute("PRAGMA table_info(net_currents)")
    }
    cols = set(table_info)
    assert {
        'current_id', 'current_set_id', 'net_name', 'pin_id', 'avg_a',
        'rms_a', 'peak_a', 'source', 'current_vector', 'timestamp',
    } <= cols
    assert table_info['source']['notnull'] == 1
    assert table_info['avg_a']['notnull'] == 1
    # v1 columns moved to the net_current_sets parent — verify they are
    # really gone, not merely unchecked.
    assert 'circuit_version_id' not in cols
    assert 'corner' not in cols
    assert 'structure_hash' not in cols


def test_net_current_sets_table_exists_with_columns():
    db = Database(':memory:')
    cols = {
        row['name']
        for row in db.conn.execute("PRAGMA table_info(net_current_sets)")
    }
    assert {
        'current_set_id', 'circuit_version_id', 'profile_id', 'state',
        'supersedes_current_set_id', 'corner', 'temperature', 'voltage',
        'tool_name', 'analysis', 'tstart', 'results_file',
        'structure_scheme', 'structure_hash', 'selected_at', 'superseded_at',
    } <= cols


def test_net_currents_set_index_starts_with_current_set_id():
    db = Database(':memory:')
    index_names = {
        row['name'] for row in db.conn.execute("PRAGMA index_list(net_currents)")
    }
    assert 'idx_net_currents_set' in index_names
    columns = [
        row['name']
        for row in db.conn.execute("PRAGMA index_info(idx_net_currents_set)")
    ]
    assert columns[:1] == ['current_set_id']


def test_current_sets_scoped_and_unscoped_unique_indexes_exist():
    db = Database(':memory:')
    index_names = {
        row['name']
        for row in db.conn.execute("PRAGMA index_list(net_current_sets)")
    }
    assert 'idx_current_sets_scoped_current' in index_names
    assert 'idx_current_sets_unscoped_current' in index_names


def test_central_writer_rejects_missing_source_before_write():
    db = Database(':memory:')
    version_id = _mk_version(db)
    version = db._snapshot_row(version_id)

    with pytest.raises(ValueError, match='source'):
        db.replace_net_current_set(
            version_id,
            corner='tt',
            structure_scheme=version['structure_scheme'],
            structure_hash=version['structure_hash'],
            rows=[{'net_name': 'mid', 'avg_a': 1e-3}],
        )

    assert db.conn.execute(
        "SELECT COUNT(*) FROM net_currents"
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM net_current_sets"
    ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("avg_a", "corrupt"),
        ("avg_a", None),
        ("rms_a", False),
        ("peak_a", object()),
        ("current_vector", 17),
    ],
)
def test_central_writer_rejects_invalid_current_value_types(field, invalid):
    db = Database(':memory:')
    version_id = _mk_version(db)
    version = db._snapshot_row(version_id)
    row = {
        'net_name': 'mid',
        'pin_id': '',
        'avg_a': 1e-3,
        'rms_a': 1.1e-3,
        'peak_a': 1.5e-3,
        'source': 'spice',
        'current_vector': None,
    }
    row[field] = invalid

    with pytest.raises(TypeError, match=field):
        db.replace_net_current_set(
            version_id,
            corner='tt',
            structure_scheme=version['structure_scheme'],
            structure_hash=version['structure_hash'],
            rows=[row],
        )

    assert db.conn.execute(
        "SELECT COUNT(*) FROM net_currents"
    ).fetchone()[0] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM net_current_sets"
    ).fetchone()[0] == 0


def test_duplicate_aggregate_row_conflicts():
    """The UNIQUE must be real for aggregate rows (pin_id='') — the SQLite
    NULL-distinct trap is exactly what the sentinel design prevents."""
    db = Database(':memory:')
    v = _mk_version(db)
    current_set_id = _insert_set(db, v, corner='tt')
    _insert(db, current_set_id, pin='')
    with pytest.raises(sqlite3.IntegrityError):
        _insert(db, current_set_id, pin='')


def test_same_net_different_set_coexists():
    db = Database(':memory:')
    v = _mk_version(db)
    set_tt = _insert_set(db, v, corner='tt')
    set_ss = _insert_set(db, v, corner='ss')
    _insert(db, set_tt)
    _insert(db, set_ss)   # must not raise: different parent set
    n = db.conn.execute("SELECT COUNT(*) FROM net_currents").fetchone()[0]
    assert n == 2


def test_upsert_updates_in_place():
    db = Database(':memory:')
    v = _mk_version(db)
    current_set_id = _insert_set(db, v)
    _insert(db, current_set_id, avg=1e-3)
    db.conn.execute(
        """INSERT INTO net_currents
           (current_id, current_set_id, net_name, pin_id, avg_a, rms_a,
            peak_a, source, current_vector)
           VALUES (?, ?, 'mid', '', 2e-3, 2e-3, 2e-3, 'spice', NULL)
           ON CONFLICT(current_set_id, net_name, pin_id)
           DO UPDATE SET avg_a=excluded.avg_a, rms_a=excluded.rms_a,
                         peak_a=excluded.peak_a, source=excluded.source,
                         current_vector=excluded.current_vector,
                         timestamp=CURRENT_TIMESTAMP""",
        (str(uuid.uuid4()), current_set_id))
    db.conn.commit()
    rows = db.conn.execute("SELECT avg_a FROM net_currents").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(2e-3)


def test_source_check_constraint():
    db = Database(':memory:')
    v = _mk_version(db)
    current_set_id = _insert_set(db, v)
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            """INSERT INTO net_currents
               (current_id, current_set_id, net_name, pin_id, avg_a, source)
               VALUES (?, ?, 'mid', '', 1e-3, 'bogus')""",
            (str(uuid.uuid4()), current_set_id))


def test_null_pin_id_rejected():
    """pin_id has a DEFAULT of '', but an explicit NULL must still be
    rejected — the sentinel-not-NULL design only works if NULL can't sneak
    in and defeat the UNIQUE index's NULL-distinct semantics."""
    db = Database(':memory:')
    v = _mk_version(db)
    current_set_id = _insert_set(db, v)
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            """INSERT INTO net_currents
               (current_id, current_set_id, net_name, pin_id, avg_a, source)
               VALUES (?, ?, 'mid', NULL, 1e-3, 'spice')""",
            (str(uuid.uuid4()), current_set_id))


def test_null_avg_a_rejected():
    """avg_a is NOT NULL on the row itself (design §6.3) — unlike v1, where
    it was nullable at the DB layer and only enforced by the central
    writer."""
    db = Database(':memory:')
    v = _mk_version(db)
    current_set_id = _insert_set(db, v)
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            """INSERT INTO net_currents
               (current_id, current_set_id, net_name, pin_id, avg_a, source)
               VALUES (?, ?, 'mid', '', NULL, 'spice')""",
            (str(uuid.uuid4()), current_set_id))


def test_net_current_sets_null_corner_rejected():
    """corner moved from net_currents to net_current_sets in schema v2 but
    remains a required field of one selected current set."""
    db = Database(':memory:')
    v = _mk_version(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            """INSERT INTO net_current_sets
               (current_set_id, circuit_version_id, profile_id, state,
                corner, structure_scheme, structure_hash)
               VALUES (?, ?, NULL, 'current', NULL, 'silk.structure/v1', 'H1')""",
            (str(uuid.uuid4()), v))


def test_net_currents_row_requires_existing_current_set():
    """FOREIGN KEY (current_set_id) REFERENCES net_current_sets ... RESTRICT
    is real, not just a same-named column."""
    db = Database(':memory:')
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            """INSERT INTO net_currents
               (current_id, current_set_id, net_name, pin_id, avg_a, source)
               VALUES (?, ?, 'mid', '', 1e-3, 'spice')""",
            (str(uuid.uuid4()), str(uuid.uuid4())))


def test_current_set_deletion_restricted_while_rows_exist():
    db = Database(':memory:')
    v = _mk_version(db)
    current_set_id = _insert_set(db, v)
    _insert(db, current_set_id)
    with pytest.raises(sqlite3.IntegrityError):
        db.conn.execute(
            "DELETE FROM net_current_sets WHERE current_set_id = ?",
            (current_set_id,),
        )
