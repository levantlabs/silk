"""Aggregate schema migration is a validated copy, preserving the source."""
from pathlib import Path
import sqlite3

import pytest

from silk.db import Database
from silk.exceptions import DatabaseCompatibilityError


def source_db(tmp_path):
    source = tmp_path / 'v5.db'
    with sqlite3.connect(source) as conn:
        conn.executescript(Path(__file__).with_name('v5_schema.sql').read_text())
        conn.execute('CREATE TABLE extra_history (id TEXT PRIMARY KEY, value TEXT)')
        conn.execute("INSERT INTO extra_history VALUES ('kept', 'history')")
    return source


def test_copy_preserves_source_and_adds_nullable_column(tmp_path):
    from silk.db import migrate_v5_to_v6
    source = source_db(tmp_path)
    before = source.read_bytes()
    dest = tmp_path / 'v6.db'
    migrate_v5_to_v6(source, dest)
    assert source.read_bytes() == before
    from silk.db import migrate_v6_to_v7
    with Database(migrate_v6_to_v7(dest, tmp_path / 'current.db')) as db:
        columns = {r['name']: r for r in db.conn.execute('PRAGMA table_info(spec_measurements)')}
        assert columns['aggregate_json']['notnull'] == 0
        assert tuple(db.conn.execute('SELECT * FROM extra_history').fetchone()) == ('kept', 'history')
        assert db.conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_invalid_source_does_not_publish_destination(tmp_path):
    from silk.db import migrate_v5_to_v6
    source = source_db(tmp_path)
    with sqlite3.connect(source) as conn:
        conn.execute('DROP TABLE specs')
    before = source.read_bytes()
    dest = tmp_path / 'v6.db'
    with pytest.raises(DatabaseCompatibilityError):
        migrate_v5_to_v6(source, dest)
    assert source.read_bytes() == before
    assert not dest.exists()


def test_populated_copy_preserves_all_ids_profiles_seals_and_history(pdk, tmp_path):
    from tests.db.test_migration_v4_to_v5 import _populate
    from silk.db import migrate_v5_to_v6
    source = tmp_path / 'populated-v5.db'
    _populate(pdk, tmp_path, source)
    populated = source
    source = tmp_path / 'historical-v5.db'
    with sqlite3.connect(source) as conn:
        conn.executescript(Path(__file__).with_name('v5_schema.sql').read_text())
        conn.execute('DELETE FROM silk_metadata')
        conn.execute('ATTACH DATABASE ? AS populated', (str(populated),))
        tables = [r[0] for r in conn.execute("SELECT name FROM main.sqlite_master WHERE type='table'")]
        for table in tables:
            columns = ', '.join(f'"{r[1]}"' for r in conn.execute(f'PRAGMA main.table_info("{table}")'))
            conn.execute(f'INSERT INTO main."{table}" (rowid, {columns}) '
                         f'SELECT rowid, {columns} FROM populated."{table}"')
        conn.execute('UPDATE silk_metadata SET schema_version=5')
    before = source.read_bytes()
    dest = tmp_path / 'populated-v6.db'
    migrate_v5_to_v6(source, dest)
    assert source.read_bytes() == before
    with sqlite3.connect(source) as old, sqlite3.connect(dest) as new:
        for (table,) in old.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            columns = [r[1] for r in old.execute(f'PRAGMA table_info("{table}")')]
            if table == 'silk_metadata':
                columns.remove('schema_version')
            names = ', '.join(f'"{c}"' for c in columns)
            query = f'SELECT rowid, {names} FROM "{table}" ORDER BY rowid'
            assert old.execute(query).fetchall() == new.execute(query).fetchall(), table
        assert new.execute('SELECT count(*) FROM spec_measurements WHERE aggregate_json IS NOT NULL').fetchone()[0] == 0

    # Populate v6-only evidence before exercising the table rebuild to v7.
    import json
    from silk.specs.aggregate import encode_aggregate
    from tests.test_aggregate_measurement import aggregate
    from silk.db import migrate_v6_to_v7
    frozen = json.loads((Path(__file__).parent / 'fixtures/legacy_provenance_digests.json').read_text())['records'][0]
    with sqlite3.connect(dest) as conn:
        conn.execute('INSERT INTO run_provenance (provenance_id, scheme, digest, record_json) VALUES (?, ?, ?, ?)',
                     ('historical', frozen['record']['scheme'], frozen['digest'],
                      json.dumps(frozen['record'], sort_keys=True, separators=(',', ':'))))
        for table in ('simulation_results', 'spec_measurements', 'net_current_sets'):
            conn.execute(f"UPDATE {table} SET provenance_id='historical'")
        measurement, metric = conn.execute(
            'SELECT m.measurement_id, s.metric FROM spec_measurements m '
            'JOIN specs s ON s.spec_id=m.spec_id LIMIT 1').fetchone()
        value = aggregate(metric=metric, base_seed=7)
        conn.execute('UPDATE spec_measurements SET aggregate_json=?, sim_run_id=? WHERE measurement_id=?',
                     (json.dumps(encode_aggregate(value)), value.campaign_id, measurement))
        conn.execute('UPDATE spec_measurements SET rowid=rowid+50')
        conn.execute('CREATE INDEX history_unit ON simulation_results(unit)')
    v6_before = dest.read_bytes()
    final = tmp_path / 'populated-v7.db'
    migrate_v6_to_v7(dest, final)
    assert dest.read_bytes() == v6_before
    with sqlite3.connect(dest) as old, Database(final) as db:
        for (table,) in old.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            columns = [r[1] for r in old.execute(f'PRAGMA table_info("{table}")')]
            if table == 'silk_metadata':
                columns.remove('schema_version')
            names = ', '.join(f'"{c}"' for c in columns)
            query = f'SELECT rowid, {names} FROM "{table}" ORDER BY rowid'
            assert old.execute(query).fetchall() == [tuple(row) for row in db.conn.execute(query)], table
        assert db.get_run_provenance('historical').digest() == frozen['digest']
        assert db.conn.execute('SELECT count(*) FROM spec_measurements WHERE aggregate_json IS NOT NULL').fetchone()[0] == 1
        assert db.conn.execute("SELECT sql FROM sqlite_master WHERE name='history_unit'").fetchone()[0] == old.execute("SELECT sql FROM sqlite_master WHERE name='history_unit'").fetchone()[0]


def test_v6_to_v7_preserves_rows_and_widens_only_supply_state(tmp_path):
    from silk.db import migrate_v5_to_v6, migrate_v6_to_v7
    from silk.db.schema import _validate_database_version
    source = tmp_path / 'v6.db'
    migrate_v5_to_v6(source_db(tmp_path), source)
    before = source.read_bytes()
    dest = tmp_path / 'v7.db'
    migrate_v6_to_v7(source, dest)
    assert source.read_bytes() == before
    with sqlite3.connect(source) as old, Database(dest) as new:
        for (table,) in old.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            if table != 'silk_metadata':
                assert old.execute(f'SELECT rowid,* FROM "{table}"').fetchall() == [
                    tuple(row) for row in new.conn.execute(f'SELECT rowid,* FROM "{table}"')]
        assert new.conn.execute('PRAGMA foreign_key_check').fetchall() == []
        assert new.conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        for table in ('simulation_results', 'spec_measurements', 'net_current_sets'):
            sql = new.conn.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone()[0]
            assert "voltage_state IN ('fixed','none','unknown','sweep')" in sql
        _validate_database_version(old, path_label=str(source), schema_version=6)
    with pytest.raises(FileExistsError):
        migrate_v6_to_v7(source, dest)
    assert source.read_bytes() == before


def test_v6_to_v7_preserves_dependent_views_and_triggers(tmp_path):
    from silk.db import migrate_v5_to_v6, migrate_v6_to_v7
    source = tmp_path / 'v6.db'
    migrate_v5_to_v6(source_db(tmp_path), source)
    with sqlite3.connect(source) as conn:
        conn.execute('CREATE VIEW metric_history AS SELECT * FROM simulation_results')
        conn.execute('CREATE TRIGGER keep_view_readable AFTER INSERT ON extra_history '
                     'BEGIN SELECT count(*) FROM metric_history; END')
    before = source.read_bytes()
    dest = tmp_path / 'v7.db'
    migrate_v6_to_v7(source, dest)
    assert source.read_bytes() == before
    with Database(dest) as db:
        assert db.conn.execute('SELECT * FROM metric_history').fetchall() == []
        db.conn.execute("INSERT INTO extra_history VALUES ('next', 'works')")
        assert db.conn.execute('PRAGMA foreign_key_check').fetchall() == []


def _review_v6_source(pdk, tmp_path, *, fresh=False, quoted=False,
                      extra_constraints=False, check_matches=1):
    """Populate a real historical v6 shape without current-schema relabeling."""
    import json
    from silk.db import migrate_v5_to_v6
    from silk.specs.aggregate import encode_aggregate
    from tests.db.test_migration_v4_to_v5 import _populate
    from tests.db.test_characterization_readback import _current_row
    from tests.test_aggregate_measurement import aggregate

    populated = tmp_path / 'review-populated.db'
    version = _populate(pdk, tmp_path, populated)
    frozen = json.loads((Path(__file__).parent / 'fixtures/legacy_provenance_digests.json').read_text())['records'][0]
    with Database(populated) as db:
        row = db._snapshot_row(version)
        db.replace_net_current_set(
            version, corner='ff', structure_scheme=row['structure_scheme'],
            structure_hash=row['structure_hash'], rows=[_current_row(7e-3)])
        db.conn.execute('INSERT INTO run_provenance (provenance_id, scheme, digest, record_json) VALUES (?, ?, ?, ?)',
                        ('frozen', frozen['record']['scheme'], frozen['digest'],
                         json.dumps(frozen['record'], sort_keys=True, separators=(',', ':'))))
        for table in ('simulation_results', 'spec_measurements', 'net_current_sets'):
            db.conn.execute(f"UPDATE {table} SET provenance_id='frozen'")
        measurement, metric = db.conn.execute(
            'SELECT m.measurement_id, s.metric FROM spec_measurements m '
            'JOIN specs s ON s.spec_id=m.spec_id LIMIT 1').fetchone()
        value = aggregate(metric=metric, base_seed=7)
        db.conn.execute('UPDATE spec_measurements SET aggregate_json=?, sim_run_id=? WHERE measurement_id=?',
                        (json.dumps(encode_aggregate(value)), value.campaign_id, measurement))
        db.conn.execute('UPDATE spec_measurements SET rowid=rowid+50')
        db.conn.commit()

    template = tmp_path / 'review-template-v6.db'
    if fresh:
        with sqlite3.connect(template) as conn:
            # Existing historical DDL entry point, without another Database class.
            db = Database.__new__(Database)
            db.conn = conn
            db._txn_depth = 0
            db._create_tables(schema_version=6)
    else:
        migrate_v5_to_v6(source_db(tmp_path), template)
    source = tmp_path / 'review-v6.db'
    clause = "voltage_state IN ('fixed','none','unknown')"
    with sqlite3.connect(template) as schema, sqlite3.connect(source) as conn:
        for name, sql in schema.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
            if quoted and name == 'spec_measurements':
                quote = '[spec_measurements]' if fresh else '`spec_measurements`'
                sql = sql.replace('CREATE TABLE spec_measurements', 'CREATE TABLE ' + quote, 1)
            if name == 'simulation_results':
                if extra_constraints:
                    sql = sql.rstrip().removesuffix(')') + ', CHECK(value IS NULL OR value >= 0), UNIQUE(parameter, value))'
                if check_matches == 0:
                    sql = sql.replace(clause, "voltage_state IN ('none','fixed','unknown')")
                elif check_matches == 2:
                    sql = sql.rstrip().removesuffix(')') + ', CHECK(' + clause + '))'
            conn.execute(sql)
        for (sql,) in schema.execute("SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"):
            conn.execute(sql)
        conn.execute('ATTACH DATABASE ? AS populated', (str(populated),))
        for (table,) in conn.execute("SELECT name FROM populated.sqlite_master WHERE type='table'").fetchall():
            columns = ', '.join(f'"{r[1]}"' for r in conn.execute(f'PRAGMA main.table_info("{table}")'))
            conn.execute(f'INSERT INTO main."{table}" (rowid, {columns}) '
                         f'SELECT rowid, {columns} FROM populated."{table}"')
        conn.execute('UPDATE silk_metadata SET schema_version=6')
        conn.execute('CREATE INDEX review_results_unit ON simulation_results(unit)')
    return source, frozen


@pytest.mark.parametrize('fresh', [False, True], ids=['v5-copy', 'fresh-v6'])
def test_v7_quoted_tables_preserve_named_history_constraints_and_foreign_keys(pdk, tmp_path, fresh):
    from silk.db import migrate_v6_to_v7
    source, frozen = _review_v6_source(pdk, tmp_path, fresh=fresh, quoted=True,
                                       extra_constraints=True)
    before = source.read_bytes()
    dest = tmp_path / 'review-v7.db'
    with sqlite3.connect(source) as conn:
        columns = [r[1] for r in conn.execute('PRAGMA table_info(spec_measurements)')]
        assert (columns.index('aggregate_json') < columns.index('provenance_id')) is fresh
    migrate_v6_to_v7(source, dest)
    assert source.read_bytes() == before
    with sqlite3.connect(source) as old, Database(dest) as db:
        for (table,) in old.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            columns = [r[1] for r in old.execute(f'PRAGMA table_info("{table}")')]
            if table == 'silk_metadata':
                columns.remove('schema_version')
            names = ', '.join(f'"{c}"' for c in columns)
            query = f'SELECT rowid, {names} FROM "{table}" ORDER BY rowid'
            assert old.execute(query).fetchall() == [tuple(row) for row in db.conn.execute(query)], table
        for table in ('simulation_results', 'spec_measurements', 'net_current_sets'):
            old_sql = old.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone()[0]
            new_sql = db.conn.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone()[0]
            assert new_sql == old_sql.replace("voltage_state IN ('fixed','none','unknown')",
                                               "voltage_state IN ('fixed','none','unknown','sweep')")
        assert db.get_run_provenance('frozen').digest() == frozen['digest']
        assert db.conn.execute('SELECT count(*) FROM spec_measurements WHERE aggregate_json IS NOT NULL').fetchone()[0] == 1
        assert db.conn.execute("SELECT count(*) FROM net_current_sets WHERE state='superseded'").fetchone()[0] >= 1
        assert db.conn.execute('SELECT count(*) FROM net_currents').fetchone()[0] > 0
        assert db.conn.execute("SELECT count(*) FROM net_current_sets WHERE supersedes_current_set_id IS NOT NULL").fetchone()[0] >= 1
        assert 'net_current_sets' in {r[2] for r in db.conn.execute('PRAGMA foreign_key_list(net_current_sets)')}
        assert {r[2] for r in db.conn.execute('PRAGMA foreign_key_list(net_currents)')} == {'net_current_sets'}
        assert db.conn.execute('PRAGMA foreign_key_check').fetchall() == []
        assert db.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name GLOB '_silk_v6_*'").fetchall() == []
        assert db.conn.execute("SELECT sql FROM sqlite_master WHERE name='review_results_unit'").fetchone()[0] == old.execute("SELECT sql FROM sqlite_master WHERE name='review_results_unit'").fetchone()[0]
        version = db.conn.execute('SELECT version_id FROM circuit_versions LIMIT 1').fetchone()[0]
        parameter, value = db.conn.execute('SELECT parameter, value FROM simulation_results WHERE value IS NOT NULL LIMIT 1').fetchone()
        insert = ('INSERT INTO simulation_results (sim_id, circuit_version_id, source, parameter, value, '
                  'corner_state, temperature_state, voltage_state) '
                  "VALUES (?, ?, 'user', ?, ?, 'unknown', 'unknown', 'unknown')")
        with pytest.raises(sqlite3.IntegrityError, match='CHECK'):
            db.conn.execute(insert, ('negative', version, 'negative', -1))
        with pytest.raises(sqlite3.IntegrityError, match='UNIQUE'):
            db.conn.execute(insert, ('duplicate', version, parameter, value))


def test_v7_copy_does_not_fire_rebuilt_table_trigger(pdk, tmp_path):
    from silk.db import migrate_v6_to_v7
    source, _ = _review_v6_source(pdk, tmp_path)
    with sqlite3.connect(source) as conn:
        conn.execute('CREATE TABLE copied_audit (sim_id TEXT)')
        conn.execute('CREATE TRIGGER audit_metric AFTER INSERT ON simulation_results '
                     'BEGIN INSERT INTO copied_audit VALUES (NEW.sim_id); END')
    before = source.read_bytes()
    dest = tmp_path / 'trigger-v7.db'
    migrate_v6_to_v7(source, dest)
    assert source.read_bytes() == before
    with Database(dest) as db:
        assert db.conn.execute('SELECT * FROM copied_audit').fetchall() == []
        version = db.conn.execute('SELECT version_id FROM circuit_versions LIMIT 1').fetchone()[0]
        db.add_simulation_result(version, 'later', value=1)
        assert len(db.conn.execute('SELECT * FROM copied_audit').fetchall()) == 1


@pytest.mark.parametrize('matches', [0, 2])
def test_v7_refuses_nonunique_state_clause_without_publication(pdk, tmp_path, matches):
    from silk.db import migrate_v6_to_v7
    source, _ = _review_v6_source(pdk, tmp_path, check_matches=matches)
    before = source.read_bytes()
    dest = tmp_path / 'refused-v7.db'
    with pytest.raises(DatabaseCompatibilityError):
        migrate_v6_to_v7(source, dest)
    assert source.read_bytes() == before
    assert not dest.exists()
    assert list(tmp_path.glob('.refused-v7.db.migrating-*')) == []


def test_v7_refuses_private_name_collision_without_deleting_history(pdk, tmp_path):
    from silk.db import migrate_v6_to_v7
    source, _ = _review_v6_source(pdk, tmp_path)
    with sqlite3.connect(source) as conn:
        conn.execute('CREATE TABLE _silk_v6_spec_measurements (history TEXT)')
        conn.execute("INSERT INTO _silk_v6_spec_measurements VALUES ('do not delete')")
    before = source.read_bytes()
    dest = tmp_path / 'collision-v7.db'
    with pytest.raises(DatabaseCompatibilityError, match='_silk_v6_spec_measurements'):
        migrate_v6_to_v7(source, dest)
    assert source.read_bytes() == before
    assert not dest.exists()
    assert list(tmp_path.glob('.collision-v7.db.migrating-*')) == []


def test_v7_failure_after_rebuild_is_typed_and_unpublished(pdk, tmp_path, monkeypatch):
    from silk.db import migration
    source, _ = _review_v6_source(pdk, tmp_path)
    before = source.read_bytes()
    dest = tmp_path / 'failure-v7.db'
    validate = migration._validate_database_version
    def refuse_after_rebuild(conn, *, path_label, schema_version):
        validate(conn, path_label=path_label, schema_version=schema_version)
        if schema_version == 7:
            raise sqlite3.OperationalError('injected after rebuild')
    monkeypatch.setattr(migration, '_validate_database_version', refuse_after_rebuild)
    with pytest.raises(DatabaseCompatibilityError, match='injected after rebuild'):
        migration.migrate_v6_to_v7(source, dest)
    assert source.read_bytes() == before
    assert not dest.exists()
    assert list(tmp_path.glob('.failure-v7.db.migrating-*')) == []
