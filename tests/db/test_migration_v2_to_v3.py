"""The v3 copy changes only measurement nullability and never the source."""

import inspect
import math
from pathlib import Path
import sqlite3

import pytest

from silk.db import (
    Database,
    migrate_v1_to_v2,
    migrate_v2_to_v3,
    migrate_v3_to_v4,
    migrate_v4_to_v5,
    migrate_v5_to_v6,
    migrate_v6_to_v7,
)
from silk.db.context import CharacterizationContext
from silk.exceptions import DatabaseCompatibilityError, UnsupportedSchemaVersionError
from tests.db.test_migration_v1_to_v2 import _declared_shape
from tests.db.test_unavailable_measurements import _build, _report


@pytest.fixture
def v2_file(pdk, tmp_path):
    current = tmp_path / 'fixture-current.db'
    mod, specs = _build(pdk, tmp_path, current, mixed=True)
    context = CharacterizationContext(
        owner_path='$', test_name='ac', parameters={'load': 1e-12},
        corner='tt', temperature=27.0, voltage=1.8,
    )
    mod.store_report(_report(specs, [10.0, 42.0, 0.0, 'Error: missing'], context=context))
    version = mod._tracker.circuit_id
    # Numeric-only NaN rows already existed in v2.
    mod._tracker.db.add_simulation_result(version, 'metric_only_nan', value=float('nan'))
    mod.close()

    source = tmp_path / 'v2.db'
    with sqlite3.connect(source) as conn:
        conn.executescript(Path(__file__).with_name('v2_schema.sql').read_text())
        conn.execute('ATTACH DATABASE ? AS populated', (str(current),))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM main.sqlite_master WHERE type = 'table'"
        )]
        for table in tables:
            columns = ", ".join(
                f'"{r[1]}"' for r in conn.execute(f'PRAGMA main.table_info("{table}")')
            )
            # Named columns, not '*': the populating file is the current
            # (v4) shape, which has columns a v2 table does not.
            conn.execute(
                f'INSERT INTO main."{table}" (rowid, {columns}) '
                f'SELECT rowid, {columns} FROM populated."{table}"'
            )
        conn.execute('UPDATE silk_metadata SET schema_version = 2')
        # Non-contiguous rowids matter to latest-observation tie breaking.
        conn.execute('UPDATE spec_measurements SET rowid = rowid + 50')
        conn.execute('CREATE INDEX extra_measurement_notes ON spec_measurements(notes)')
    return source, version


def _rows(conn):
    return {table: conn.execute(f'SELECT rowid, * FROM "{table}" ORDER BY rowid').fetchall()
            for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_v2_open_rejected_read_only_with_actionable_migration(v2_file):
    source, _ = v2_file
    before = source.read_bytes()
    with pytest.raises(UnsupportedSchemaVersionError, match='migrate_v2_to_v3'):
        Database(source)
    assert source.read_bytes() == before


def test_copy_preserves_rows_ids_indexes_relationships_and_profile(v2_file, tmp_path, pdk):
    source, version = v2_file
    before = source.read_bytes()
    dest = tmp_path / 'v3.db'
    assert migrate_v2_to_v3(source, dest) == str(dest)
    assert source.read_bytes() == before
    with sqlite3.connect(source) as src, sqlite3.connect(dest) as dst:
        source_shape = _declared_shape(src)
        columns = source_shape['spec_measurements']['columns']
        source_shape['spec_measurements']['columns'] = tuple(
            (name, kind, False if name == 'value' else notnull)
            for name, kind, notnull in columns
        )
        assert _declared_shape(dst) == source_shape
        # Every rowid, UUID, fingerprint, profile descriptor, and value survives.
        old_rows, new_rows = _rows(src), _rows(dst)
        assert old_rows.pop('silk_metadata')[0][3] == 2
        assert new_rows.pop('silk_metadata')[0][3] == 3
        assert old_rows == new_rows
        for row in src.execute("SELECT name, sql FROM sqlite_master WHERE type = 'index'"):
            assert dst.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                (row[0],),
            ).fetchone() == (row[1],)
        assert dst.execute('PRAGMA foreign_key_check').fetchall() == []

    # Current silk opens v7; the v3 copy reaches it through the
    # next copy steps, keeping every profile and measurement.
    v4 = tmp_path / 'v4.db'
    migrate_v3_to_v4(dest, v4)
    current = tmp_path / 'v5.db'
    v5 = tmp_path / 'migration-v5.db'
    migrate_v4_to_v5(v4, v5)
    v6 = tmp_path / 'v6.db'
    migrate_v5_to_v6(v5, v6)
    migrate_v6_to_v7(v6, current)
    mod, specs = _build(pdk, tmp_path, current, mixed=True)
    try:
        assert mod._tracker.circuit_id == version
        profile, = mod._tracker.db.list_characterization_profiles(version)
        context = CharacterizationContext(
            owner_path='$', test_name='ac', parameters={'load': 1e-12},
            corner='tt', temperature=27.0, voltage=1.8,
        )
        mod.store_report(_report(specs, [float('nan')], context=context))
        assert len(mod._tracker.db.list_characterization_profiles(version)) == 1
        history = next(
            r for r in mod.get_stored_specs(profile_id=profile.profile_id)
            if r.spec_key == 'ugf'
        ).measurements
        assert [m.value for m in history[:-1]] == [10.0]
        assert math.isnan(history[-1].value)
        assert math.isnan(mod.query_results(parameter='metric_only_nan')[0].value)
        for spec in specs:
            spec.measurements.clear()
        assert mod.restore_spec_measurements(profile_id=profile.profile_id) == 3
        assert math.isnan(next(iter(specs[0].measurements.values())).value)
    finally:
        mod.close()


@pytest.mark.parametrize('kind', ['source', 'existing', 'symlink', 'hardlink', 'dangling'])
def test_never_replaces_source_or_existing_destination(v2_file, tmp_path, kind):
    source, _ = v2_file
    before = source.read_bytes()
    dest = tmp_path / 'destination.db'
    if kind == 'source':
        dest = source
    elif kind == 'existing':
        dest.write_bytes(b'keep this file')
    elif kind == 'symlink':
        dest.symlink_to(source)
    elif kind == 'hardlink':
        dest.hardlink_to(source)
    else:
        dest.symlink_to(tmp_path / 'missing.db')
    with pytest.raises(FileExistsError):
        migrate_v2_to_v3(source, dest)
    assert source.read_bytes() == before
    if kind == 'existing':
        assert dest.read_bytes() == b'keep this file'
    if kind == 'dangling':
        assert dest.is_symlink()
    assert not list(tmp_path.glob('.*.migrating-*'))


@pytest.mark.parametrize('failure', ['copy', 'schema', 'foreign_key', 'publication_race'])
def test_failed_migration_leaves_source_and_no_partial_destination(
    v2_file, tmp_path, monkeypatch, failure,
):
    import silk.db.migration as migration

    source, _ = v2_file
    before = source.read_bytes()
    dest = tmp_path / 'failed.db'
    original = migration._make_measurements_nullable

    def sabotage(conn):
        original(conn)
        if failure == 'copy':
            raise RuntimeError('injected copy failure')
        if failure == 'schema':
            conn.execute('DROP INDEX idx_current_sets_scoped_current')
        if failure == 'foreign_key':
            conn.execute("UPDATE spec_measurements SET spec_id = 'orphan'")
        if failure == 'publication_race':
            dest.write_bytes(b'concurrent owner')

    monkeypatch.setattr(migration, '_make_measurements_nullable', sabotage)
    with pytest.raises((RuntimeError, DatabaseCompatibilityError, FileExistsError)):
        migrate_v2_to_v3(source, dest)
    assert source.read_bytes() == before
    if failure == 'publication_race':
        assert dest.read_bytes() == b'concurrent owner'
    else:
        assert not dest.exists()
    assert not list(tmp_path.glob('.*.migrating-*'))


@pytest.mark.parametrize('version, notnull', [(2, False), (3, True)])
def test_metadata_and_actual_nullability_must_agree(v2_file, tmp_path, version, notnull):
    source, _ = v2_file
    if notnull:
        with sqlite3.connect(source) as conn:
            conn.execute('UPDATE silk_metadata SET schema_version = 3')
        before = source.read_bytes()
        # A v3 file is validated against the v3 shape by its copy step
        # (a normal open now refuses v3 on the version alone).
        with pytest.raises(DatabaseCompatibilityError, match='nullable'):
            migrate_v3_to_v4(source, tmp_path / 'rejected.db')
        assert not (tmp_path / 'rejected.db').exists()
    else:
        dest = tmp_path / 'v3.db'
        migrate_v2_to_v3(source, dest)
        source = dest
        with sqlite3.connect(source) as conn:
            conn.execute('UPDATE silk_metadata SET schema_version = 2')
        before = source.read_bytes()
        with pytest.raises(DatabaseCompatibilityError, match='NOT NULL'):
            migrate_v2_to_v3(source, tmp_path / 'rejected.db')
        assert not (tmp_path / 'rejected.db').exists()
    assert source.read_bytes() == before


def test_v1_migration_signature_remains_unchanged():
    assert str(inspect.signature(migrate_v1_to_v2)) == "(source_path, dest_path) -> 'str'"
