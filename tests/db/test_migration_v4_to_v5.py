"""migrate_v4_to_v5: condition states per row (TASK-409; design 2026-10-01
operating-conditions-none, §5.3 and §5.4).

The source is a real schema-v4 file: the frozen v4 DDL (``v4_schema.sql``,
dumped before v5 existed) populated with rows written by silk's own
writers. Today's NULL becomes unknown, never none, and every existing
scoped profile keeps its identity and its evidence its selection key.
"""

import json
from pathlib import Path
import sqlite3

import pytest

from silk.db import Database, migrate_v4_to_v5, migrate_v5_to_v6, migrate_v6_to_v7
from silk.db.context import CharacterizationContext
from silk.db.identity import (
    _canonicalize_characterization_profile,
    _decode_characterization_profile_parameters,
)
from silk.exceptions import DatabaseCompatibilityError, UnsupportedSchemaVersionError
from silk.simulation.conditions import (
    ConditionState,
    OperatingConditions,
    ConditionSweep,
)
from tests.db.test_characterization_readback import (
    _build_live,
    _current_row,
    _scoped_report,
)

_V4_SCHEMA = Path(__file__).with_name('v4_schema.sql')
_MARKER = 'silk_operating_conditions'
_EVIDENCE_TABLES = ('simulation_results', 'spec_measurements', 'net_current_sets')

#: Two PVT points under one profile (same owner/test/parameters).
_POINTS = (('tt', 27.0, 1.8), ('ss', 85.0, 1.6))


def _context(corner, temperature, voltage):
    return CharacterizationContext(
        owner_path='$/amp', test_name='load_step',
        parameters={'cload_f': 2e-12}, corner=corner,
        temperature=temperature, voltage=voltage)


def _populate(pdk, tmp_path, db_path):
    """Write scoped and unscoped evidence through silk's writers."""
    module, gain = _build_live(pdk, tmp_path, db_path)
    try:
        version = module.create_circuit_version()
        db = module._tracker.db
        row = db._snapshot_row(version)
        for index, point in enumerate(_POINTS):
            context = _context(*point)
            module.store_report(_scoped_report(gain, context, value=60.0 + index))
            db.replace_net_current_set(
                version, corner=point[0], temperature=point[1],
                voltage=point[2], structure_scheme=row['structure_scheme'],
                structure_hash=row['structure_hash'], context=context,
                rows=[_current_row(1e-3 * (index + 1))])
        # Unscoped evidence with a recorded snapshot (the marker).
        report = _scoped_report(gain, _context(*_POINTS[0]), value=70.0)
        report.characterization_context = None
        report.operating_conditions = OperatingConditions(
            'silk', 'ngspice', 'tt', temperature_kind='fixed',
            temperature_c=27.0, temperature_basis='configured-and-observed')
        module.store_report(report)
        # Unknown temperature/voltage, and a temperature-sweep row.
        db.add_simulation_result(version, 'unknown_pvt', value=1.0, corner='tt')
        db.add_simulation_result(version, 'no_corner', value=2.0)
        sweep = OperatingConditions(
            'silk', 'ngspice', 'tt', temperature_kind='sweep',
            temperature_sweep=ConditionSweep(-40.0, 125.0, 5.0))
        db.add_simulation_result(
            version, 'vth_sweep', value=0.4, corner='tt',
            temperature=ConditionState.SWEEP,
            conditions={_MARKER: sweep.to_dict()})
        db.replace_net_current_set(
            version, corner='ff', structure_scheme=row['structure_scheme'],
            structure_hash=row['structure_hash'], rows=[_current_row(5e-3)])
    finally:
        module.close()
    return version


def _strip_v5_marker_keys(conditions_json):
    """Make a marker look like one written before TASK-403's kinds."""
    if conditions_json is None:
        return None
    decoded = json.loads(conditions_json)
    marker = decoded.get(_MARKER)
    if not isinstance(marker, dict):
        return conditions_json
    for key in ('corner_kind', 'corner_basis', 'supply_kind'):
        marker.pop(key, None)
    return Database._canonical_simulation_conditions(decoded)


@pytest.fixture
def v4_file(pdk, tmp_path):
    populated = tmp_path / 'populated.db'
    version = _populate(pdk, tmp_path, populated)
    source = tmp_path / 'v4.db'
    with sqlite3.connect(source) as conn:
        conn.executescript(_V4_SCHEMA.read_text())
        conn.execute('ATTACH DATABASE ? AS populated', (str(populated),))
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM main.sqlite_master WHERE type = 'table'")]
        for table in tables:
            columns = ', '.join(
                f'"{r[1]}"' for r in conn.execute(f'PRAGMA main.table_info("{table}")'))
            conn.execute(
                f'INSERT INTO main."{table}" (rowid, {columns}) '
                f'SELECT rowid, {columns} FROM populated."{table}"')
        conn.execute('UPDATE silk_metadata SET schema_version = 4')
        # Pre-TASK-403 markers lack the v5 kinds; keep one current-shape
        # marker (written after TASK-403, still v4) to prove kept keys stay.
        rows = conn.execute(
            "SELECT sim_id, conditions FROM simulation_results "
            "WHERE conditions IS NOT NULL ORDER BY rowid").fetchall()
        kept = rows[0][0]
        for sim_id, conditions in rows[1:]:
            conn.execute(
                'UPDATE simulation_results SET conditions = ? WHERE sim_id = ?',
                (_strip_v5_marker_keys(conditions), sim_id))
        # Non-contiguous rowids and a caller index must both survive.
        conn.execute('UPDATE spec_measurements SET rowid = rowid + 50')
        conn.execute('CREATE INDEX extra_results_unit ON simulation_results(unit)')
    return source, version, kept


def _v4_selection(conn, version, profile_id, point):
    """The v4 exact PVT selection, run on the v4 file itself."""
    where = ('circuit_version_id = ? AND profile_id = ? AND corner = ? '
             'AND temperature = ? AND voltage = ?')
    params = (version, profile_id, *point)
    current = conn.execute(
        f"SELECT current_set_id FROM net_current_sets WHERE {where} "
        "AND state = 'current'", params).fetchall()
    metrics = conn.execute(
        f'SELECT sim_id FROM simulation_results WHERE {where}', params).fetchall()
    measurements = conn.execute(
        f'SELECT measurement_id FROM spec_measurements WHERE {where}',
        params).fetchall()
    return ({r[0] for r in current}, {r[0] for r in metrics},
            {r[0] for r in measurements})


def _table_rows(conn, table):
    return [tuple(row) for row in conn.execute(
        f'SELECT rowid, * FROM "{table}" ORDER BY rowid')]


def test_v4_open_is_refused_and_names_the_migration(v4_file):
    source, _version, _kept = v4_file
    before = source.read_bytes()
    with pytest.raises(UnsupportedSchemaVersionError, match='migrate_v4_to_v5'):
        Database(str(source))
    assert source.read_bytes() == before


def test_every_scoped_profile_keeps_its_identity_and_selection(v4_file, tmp_path):
    """Design §5.4, executable: same profile rows and hashes, and the same
    exact-PVT evidence selected after migration as the v4 query selects."""
    source, version, _kept = v4_file
    with sqlite3.connect(source) as conn:
        profiles_before = _table_rows(conn, 'characterization_profiles')
        assert profiles_before
        selections_before = {}
        for profile in conn.execute(
                'SELECT profile_id FROM characterization_profiles'):
            for point in _POINTS:
                selections_before[(profile[0], point)] = _v4_selection(
                    conn, version, profile[0], point)
    assert all(all(parts) for parts in selections_before.values())

    dest = tmp_path / 'v5.db'
    before = source.read_bytes()
    assert migrate_v4_to_v5(source, dest) == str(dest)
    assert source.read_bytes() == before

    with Database(migrate_v6_to_v7(migrate_v5_to_v6(dest, tmp_path / 'v6.db'), tmp_path / 'current.db')) as db:
        assert _table_rows(db.conn, 'characterization_profiles') == [
            tuple(row) for row in profiles_before]
        for row in db.conn.execute(
                'SELECT owner_path, test_name, parameters_json, profile_hash '
                'FROM characterization_profiles'):
            parameters, _entries = _decode_characterization_profile_parameters(
                row['parameters_json'])
            for point in (*_POINTS, (ConditionState.NONE, 27.0, ConditionState.NONE)):
                context = CharacterizationContext(
                    owner_path=row['owner_path'], test_name=row['test_name'],
                    parameters=parameters, corner=point[0],
                    temperature=point[1], voltage=point[2])
                fingerprint = _canonicalize_characterization_profile(context).fingerprint
                assert fingerprint.digest == row['profile_hash']

        for (profile_id, point), (current, metrics, measurements) in (
                selections_before.items()):
            evidence = db.select_characterization_evidence(
                version, profile_id=profile_id, corner=point[0],
                temperature=point[1], voltage=point[2])
            assert {evidence.current_set_id} == current
            assert {m.result_id for m in evidence.metrics} == metrics
            assert {m.measurement_id for s in evidence.specs
                    for m in s.measurements} == measurements
            assert (evidence.corner, evidence.temperature, evidence.voltage) == point
            by_context = db.select_characterization_evidence(
                version, context=_context(*point))
            assert by_context.current_set_id == evidence.current_set_id

        # Nothing new joins an old key: no migrated row is none.
        for table in _EVIDENCE_TABLES:
            for column in ('corner', 'temperature', 'voltage'):
                assert db.conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {column}_state = 'none'"
                ).fetchone()[0] == 0


def test_states_follow_the_migration_table(v4_file, tmp_path):
    """Design §5.3: non-NULL is fixed (value kept), NULL is unknown, and a
    NULL temperature recorded as a sweep is sweep."""
    source, version, _kept = v4_file
    dest = tmp_path / 'v5.db'
    migrate_v4_to_v5(source, dest)
    with sqlite3.connect(source) as src, Database(migrate_v6_to_v7(migrate_v5_to_v6(dest, tmp_path / 'v6.db'), tmp_path / 'current.db')) as db:
        for table in _EVIDENCE_TABLES:
            old = {tuple(r)[0]: tuple(r)[1:] for r in src.execute(
                f'SELECT rowid, corner, temperature, voltage FROM {table}')}
            new = {tuple(r)[0]: tuple(r)[1:] for r in db.conn.execute(
                f'SELECT rowid, corner, temperature, voltage, corner_state, '
                f'temperature_state, voltage_state FROM {table}')}
            assert set(old) == set(new)
            for rowid, values in old.items():
                assert new[rowid][:3] == values
                expected = tuple(
                    'fixed' if value is not None else 'unknown' for value in values)
                if table == 'simulation_results' and rowid == db.conn.execute(
                        "SELECT rowid FROM simulation_results "
                        "WHERE parameter = 'vth_sweep'").fetchone()[0]:
                    expected = ('fixed', 'sweep', 'unknown')
                assert new[rowid][3:] == expected, (table, rowid)

        by_parameter = {r.parameter: r for r in db.get_simulation_results(version)}
        assert by_parameter['vth_sweep'].temperature is ConditionState.SWEEP
        assert (by_parameter['unknown_pvt'].corner,
                by_parameter['unknown_pvt'].temperature,
                by_parameter['unknown_pvt'].voltage) == ('tt', None, None)
        assert by_parameter['no_corner'].corner is None
        origin, rows = db.get_net_current_set(version, 'ff')
        assert origin == version and rows[0]['temperature_state'] == 'unknown'


def test_markers_gain_the_derivable_kinds_and_keep_existing_keys(v4_file, tmp_path):
    """Design §5.2: an old marker gains corner_kind / supply_kind and a
    legacy corner basis; a marker that already has them, and conditions with
    no marker, are copied byte-for-byte."""
    source, _version, kept = v4_file
    dest = tmp_path / 'v5.db'
    migrate_v4_to_v5(source, dest)
    with sqlite3.connect(source) as src, sqlite3.connect(dest) as dst:
        old = dict(src.execute('SELECT sim_id, conditions FROM simulation_results'))
        new = dict(dst.execute('SELECT sim_id, conditions FROM simulation_results'))
    assert old.keys() == new.keys()
    rewritten = 0
    for sim_id, conditions in old.items():
        if sim_id == kept or conditions is None or _MARKER not in json.loads(conditions):
            assert new[sim_id] == conditions
            continue
        marker = json.loads(new[sim_id])[_MARKER]
        old_marker = json.loads(conditions)[_MARKER]
        assert {k: v for k, v in marker.items()
                if k not in ('corner_kind', 'corner_basis', 'supply_kind')} == old_marker
        assert marker['corner_kind'] == 'fixed'
        assert marker['corner_basis'] == 'legacy'
        assert marker['supply_kind'] == 'unknown'
        rewritten += 1
    assert rewritten >= 2
    kept_marker = json.loads(new[kept])[_MARKER]
    assert kept_marker['corner_basis'] == 'unknown'   # as written, not rewritten


def test_copy_keeps_rowids_indexes_and_every_other_table(v4_file, tmp_path):
    source, _version, _kept = v4_file
    dest = tmp_path / 'v5.db'
    migrate_v4_to_v5(source, dest)
    with sqlite3.connect(source) as src, sqlite3.connect(dest) as dst:
        tables = [r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")]
        for table in tables:
            old = _table_rows(src, table)
            new = _table_rows(dst, table)
            if table == 'silk_metadata':
                assert [r[:3] + r[4:] for r in old] == [r[:3] + r[4:] for r in new]
                assert (old[0][3], new[0][3]) == (4, 5)
                continue
            columns = [r[1] for r in src.execute(f'PRAGMA table_info("{table}")')]
            projected = dst.execute(
                f'SELECT rowid, {", ".join(columns)} FROM "{table}" ORDER BY rowid'
            ).fetchall()
            if table == 'simulation_results':
                conditions = columns.index('conditions') + 1
                old = [r[:conditions] + r[conditions + 1:] for r in old]
                projected = [r[:conditions] + r[conditions + 1:] for r in projected]
            assert projected == old, table
        assert dst.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'extra_results_unit'"
        ).fetchone() is not None
        # A migrated file declares the same tables as a fresh one.
        fresh = tmp_path / 'fresh.db'
        with sqlite3.connect(fresh) as connection:
            connection.executescript(Path(__file__).with_name('v5_schema.sql').read_text())
        with sqlite3.connect(fresh) as f:
            fresh_sql = dict(f.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table'"))
        assert dict(dst.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table'")) == fresh_sql
        assert dst.execute('PRAGMA foreign_key_check').fetchall() == []


def test_refusals_leave_no_destination(v4_file, tmp_path):
    source, _version, _kept = v4_file
    dest = tmp_path / 'v5.db'
    migrate_v4_to_v5(source, dest)
    with pytest.raises(FileExistsError):
        migrate_v4_to_v5(source, dest)
    # A v5 file is not a v4 source.
    with pytest.raises(UnsupportedSchemaVersionError):
        migrate_v4_to_v5(dest, tmp_path / 'again.db')
    assert not (tmp_path / 'again.db').exists()
    # A reshaped evidence table is refused rather than guessed at.
    odd = tmp_path / 'odd.db'
    odd.write_bytes(source.read_bytes())
    with sqlite3.connect(odd) as conn:
        conn.execute('ALTER TABLE spec_measurements ADD COLUMN corner_state TEXT')
    with pytest.raises(DatabaseCompatibilityError, match='spec_measurements'):
        migrate_v4_to_v5(odd, tmp_path / 'odd-v5.db')
    assert not (tmp_path / 'odd-v5.db').exists()
    assert [p.name for p in tmp_path.iterdir() if 'migrating' in p.name] == []
