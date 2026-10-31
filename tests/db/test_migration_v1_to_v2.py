"""Explicit v1-to-v2 copy migration (TASK-4, DB-02-SCHEMA).

Design authority: docs/superpowers/specs/2026-08-31-silk-db-characterization-
context-design.md §7 (explicit v1-to-v2 copy mapping) and §10 (proof
matrix). The v1 fixtures here are built through the genuine v1 write path
(``tests/db/_v1_fixture``), not hand-authored SQLite files.
"""

import hashlib
from pathlib import Path
import sqlite3
import uuid

import pytest

from silk.db.circuit_db import Database
from silk.db.migration import (
    migrate_v1_to_v2,
    migrate_v2_to_v3,
    migrate_v3_to_v4,
    migrate_v4_to_v5,
    migrate_v5_to_v6,
    migrate_v6_to_v7,
)
from silk.exceptions import (
    DatabaseCompatibilityError,
    EvidenceMismatchError,
    UnsupportedSchemaVersionError,
    UnsupportedFingerprintSchemeError,
)

from tests.db._v1_fixture.build import build_v1_database
from tests.db._v1_fixture.legacy_schema import (
    validate_database_connection as legacy_validate_database_connection,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _declared_shape(conn: sqlite3.Connection):
    """A structural fingerprint of every table: columns, foreign keys, and
    unique indexes — independent of row data."""
    tables = sorted(
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name != 'sqlite_sequence'"
        )
    )
    shape = {}
    for table in tables:
        columns = tuple(
            (row[1], row[2], bool(row[3]))  # name, type, notnull
            for row in conn.execute(f'PRAGMA table_info("{table}")')
        )
        fk_rows = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
        grouped = {}
        for row in fk_rows:
            grouped.setdefault(row[0], []).append(row)
        fks = set()
        for entries in grouped.values():
            entries.sort(key=lambda r: r[1])
            fks.add(
                (
                    tuple(e[3] for e in entries),
                    entries[0][2],
                    tuple(e[4] for e in entries),
                    entries[0][6],
                )
            )
        unique_indexes = set()
        for idx in conn.execute(f'PRAGMA index_list("{table}")').fetchall():
            if not idx[2]:
                continue
            cols = tuple(
                info[2]
                for info in conn.execute(f'PRAGMA index_info("{idx[1]}")')
            )
            unique_indexes.add(cols)
        shape[table] = {
            "columns": columns,
            "foreign_keys": fks,
            "unique_indexes": unique_indexes,
        }
    return shape


@pytest.fixture()
def v1_fixture(tmp_path):
    path = tmp_path / "v1.db"
    info = build_v1_database(str(path))
    return path, info


def test_migrated_shape_matches_clean_create(tmp_path, v1_fixture):
    src, _info = v1_fixture
    dst = tmp_path / "v2.db"
    migrate_v1_to_v2(str(src), str(dst))

    with sqlite3.connect(":memory:") as clean:
        clean.executescript(Path(__file__).with_name("v2_schema.sql").read_text())
        clean_shape = _declared_shape(clean)
    with sqlite3.connect(dst) as migrated:
        migrated_shape = _declared_shape(migrated)
        assert migrated.execute(
            "SELECT schema_version FROM silk_metadata"
        ).fetchone()[0] == 2

    assert migrated_shape == clean_shape


def test_source_byte_identical_after_migration(tmp_path, v1_fixture):
    src, _info = v1_fixture
    dst = tmp_path / "v2.db"
    before_hash = _sha256(src)
    migrate_v1_to_v2(str(src), str(dst))
    after_hash = _sha256(src)
    assert before_hash == after_hash


def test_ids_and_fingerprints_preserved(tmp_path, v1_fixture):
    src, info = v1_fixture
    dst = tmp_path / "v2.db"
    migrate_v1_to_v2(str(src), str(dst))

    src_conn = sqlite3.connect(src)
    src_conn.row_factory = sqlite3.Row
    dst_conn = sqlite3.connect(dst)
    dst_conn.row_factory = sqlite3.Row

    for version_id in (info["version_a"], info["version_b"]):
        source_row = src_conn.execute(
            "SELECT version_id, structure_scheme, structure_hash, "
            "spec_scheme, spec_set_hash FROM circuit_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        dest_row = dst_conn.execute(
            "SELECT version_id, structure_scheme, structure_hash, "
            "spec_scheme, spec_set_hash FROM circuit_versions WHERE version_id = ?",
            (version_id,),
        ).fetchone()
        assert tuple(source_row) == tuple(dest_row)

    # spec identity (definition_scheme/definition_hash) preserved exactly
    source_specs = {
        row["spec_id"]: tuple(row)
        for row in src_conn.execute(
            "SELECT spec_id, owner_path, spec_key, definition_scheme, "
            "definition_hash FROM specs"
        )
    }
    dest_specs = {
        row["spec_id"]: tuple(row)
        for row in dst_conn.execute(
            "SELECT spec_id, owner_path, spec_key, definition_scheme, "
            "definition_hash FROM specs"
        )
    }
    assert source_specs == dest_specs
    assert info["spec_child_a"] in dest_specs

    # net_currents current_id preserved exactly (UUIDs never reminted)
    source_current_ids = {row[0] for row in src_conn.execute(
        "SELECT current_id FROM net_currents"
    )}
    dest_current_ids = {row[0] for row in dst_conn.execute(
        "SELECT current_id FROM net_currents"
    )}
    assert source_current_ids == dest_current_ids

    # release_id preserved
    assert dst_conn.execute(
        "SELECT COUNT(*) FROM releases WHERE release_id = ?",
        (info["release_id"],),
    ).fetchone()[0] == 1

    src_conn.close()
    dst_conn.close()


def test_migrated_rows_are_unscoped(tmp_path, v1_fixture):
    src, _info = v1_fixture
    dst = tmp_path / "v2.db"
    migrate_v1_to_v2(str(src), str(dst))

    conn = sqlite3.connect(dst)
    assert conn.execute(
        "SELECT COUNT(*) FROM characterization_profiles"
    ).fetchone()[0] == 0

    total_sim = conn.execute(
        "SELECT COUNT(*) FROM simulation_results"
    ).fetchone()[0]
    unscoped_sim = conn.execute(
        "SELECT COUNT(*) FROM simulation_results WHERE profile_id IS NULL"
    ).fetchone()[0]
    assert total_sim > 0
    assert total_sim == unscoped_sim

    total_meas = conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0]
    unscoped_meas = conn.execute(
        "SELECT COUNT(*) FROM spec_measurements WHERE profile_id IS NULL"
    ).fetchone()[0]
    assert total_meas > 0
    assert total_meas == unscoped_meas

    total_sets = conn.execute(
        "SELECT COUNT(*) FROM net_current_sets"
    ).fetchone()[0]
    unscoped_sets = conn.execute(
        "SELECT COUNT(*) FROM net_current_sets "
        "WHERE profile_id IS NULL AND state = 'current'"
    ).fetchone()[0]
    assert total_sets > 0
    assert total_sets == unscoped_sets

    # spec_measurements.circuit_version_id was correctly derived via the
    # spec_id join, not left blank/mismatched.
    mismatched = conn.execute(
        """
        SELECT COUNT(*) FROM spec_measurements AS sm
        JOIN specs AS s ON s.spec_id = sm.spec_id
        WHERE s.circuit_version_id != sm.circuit_version_id
        """
    ).fetchone()[0]
    assert mismatched == 0


def test_migrated_database_is_usable_via_public_api(tmp_path, v1_fixture):
    """Structural validity is necessary but not sufficient — the migrated
    file must also work through the same public read paths a real caller
    uses (get_net_current_set), unchanged."""
    src, info = v1_fixture
    dst = tmp_path / "v2.db"
    migrate_v1_to_v2(str(src), str(dst))

    v3 = tmp_path / "v3.db"
    migrate_v2_to_v3(dst, v3)
    v4 = tmp_path / "v4.db"
    migrate_v3_to_v4(v3, v4)
    current = tmp_path / "v5.db"
    v5 = tmp_path / 'migration-v5.db'
    migrate_v4_to_v5(v4, v5)
    v6 = tmp_path / 'v6.db'
    migrate_v5_to_v6(v5, v6)
    migrate_v6_to_v7(v6, current)
    with Database(str(current)) as db:
        origin_id, rows = db.get_net_current_set(info["version_a"], "tt")
        assert origin_id == info["version_a"]
        by_pin = {row["pin_id"]: row for row in rows}
        assert by_pin[""]["net_name"] == "out_a"
        assert by_pin[""]["avg_a"] == pytest.approx(1e-3)
        assert by_pin[""]["corner"] == "tt"
        assert by_pin[""]["temperature"] == pytest.approx(27.0)

        circuit_row = db._snapshot_row(info["version_a"])
        assert circuit_row["structure_hash"] == info["structure_fp_a"].digest


def test_v1_open_rejects_without_mutation_and_points_to_migration(v1_fixture):
    src, _info = v1_fixture
    before = src.read_bytes()
    with pytest.raises(
        UnsupportedSchemaVersionError, match=r"migrate_v1_to_v2"
    ):
        Database(str(src))
    assert src.read_bytes() == before


def test_old_v1_code_rejects_v2_file(tmp_path, v1_fixture):
    """DB-01A version policy, checked symmetrically: the frozen v1 validator
    (a real snapshot of the old code, not a simulation of it) rejects a
    genuine migrated v2 file."""
    src, _info = v1_fixture
    dst = tmp_path / "v2.db"
    migrate_v1_to_v2(str(src), str(dst))

    conn = sqlite3.connect(dst)
    with pytest.raises(UnsupportedSchemaVersionError, match=r"schema version 2"):
        legacy_validate_database_connection(conn, path_label=str(dst))
    conn.close()


def test_destination_must_not_already_exist(tmp_path, v1_fixture):
    src, _info = v1_fixture
    dst = tmp_path / "v2.db"
    dst.write_bytes(b"pre-existing")
    with pytest.raises(FileExistsError):
        migrate_v1_to_v2(str(src), str(dst))
    assert dst.read_bytes() == b"pre-existing"


def test_atomicity_on_injected_failure(tmp_path, v1_fixture, monkeypatch):
    src, _info = v1_fixture
    dst = tmp_path / "v2.db"
    before = src.read_bytes()

    import silk.db.migration as migration_mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(migration_mod, "_copy_spec_measurements", boom)

    with pytest.raises(RuntimeError, match="injected migration failure"):
        migrate_v1_to_v2(str(src), str(dst))

    assert not dst.exists()
    assert not any(tmp_path.glob(".v2.db.migrating-*"))
    assert src.read_bytes() == before


def test_atomicity_on_foreign_key_violation(tmp_path, v1_fixture, monkeypatch):
    """A destination that fails PRAGMA foreign_key_check must also not be
    published — a second, independent atomicity trigger from step 8."""
    src, _info = v1_fixture
    dst = tmp_path / "v2.db"

    import silk.db.migration as migration_mod

    original = migration_mod._copy_net_currents

    def sabotage(src_conn, dest_conn):
        result = original(src_conn, dest_conn)
        # Insert a dangling reference: a net_currents row pointing at a
        # current_set_id that does not exist.
        dest_conn.execute(
            """
            INSERT INTO net_currents
                (current_id, current_set_id, net_name, pin_id, avg_a, source)
            VALUES (?, ?, 'sabotage', '', 1e-3, 'user')
            """,
            (str(uuid.uuid4()), str(uuid.uuid4())),
        )
        return result

    monkeypatch.setattr(migration_mod, "_copy_net_currents", sabotage)

    # SQLite's own deferred FK enforcement (PRAGMA defer_foreign_keys) can
    # already raise IntegrityError at commit time, before our explicit
    # PRAGMA foreign_key_check belt-and-braces step ever runs; either
    # failure must still leave no published destination.
    with pytest.raises((DatabaseCompatibilityError, sqlite3.IntegrityError)):
        migrate_v1_to_v2(str(src), str(dst))

    assert not dst.exists()


def _tamper_group_field(path, *, corner, field, value, skip_first=True):
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT current_id FROM net_currents WHERE corner = ? ORDER BY rowid",
        (corner,),
    ).fetchall()
    assert len(rows) >= 2, "fixture must have >=2 rows in this corner group"
    target_id = rows[1 if skip_first else 0][0]
    conn.execute(
        f"UPDATE net_currents SET {field} = ? WHERE current_id = ?",
        (value, target_id),
    )
    conn.commit()
    conn.close()


def test_corrupt_group_abort_is_actionable_and_atomic(tmp_path, v1_fixture):
    src, info = v1_fixture
    dst = tmp_path / "v2.db"

    # version_a/corner='tt' has 2 rows (aggregate + pin) — tamper the
    # second row's temperature to desynchronize the group's shared context.
    _tamper_group_field(
        src, corner="tt", field="temperature", value=999.0
    )

    before = src.read_bytes()
    with pytest.raises(EvidenceMismatchError) as excinfo:
        migrate_v1_to_v2(str(src), str(dst))

    message = str(excinfo.value)
    assert info["version_a"] in message
    assert "'tt'" in message
    assert "temperature" in message
    assert not dst.exists()
    assert src.read_bytes() == before


def test_corrupt_group_abort_names_every_offending_group(tmp_path, v1_fixture):
    """Two independently corrupted groups must BOTH be named in one error —
    the abort is not first-bad-row-wins (reviewer NOTE, accepted design §7
    step 5). version_a/'tt' has 2 rows (cross-row temperature divergence);
    version_b/'tt' has only 1 row, so it is corrupted via the
    unsupported-structure-scheme path instead, which fires regardless of
    group size — together they prove both corruption paths are aggregated,
    not just the first one found."""
    src, info = v1_fixture
    dst = tmp_path / "v2.db"

    _tamper_group_field(src, corner="tt", field="temperature", value=999.0)

    conn = sqlite3.connect(src)
    conn.execute(
        "UPDATE net_currents SET structure_scheme = 'silk.structure/v9' "
        "WHERE circuit_version_id = ? AND corner = 'tt'",
        (info["version_b"],),
    )
    conn.commit()
    conn.close()

    with pytest.raises(EvidenceMismatchError) as excinfo:
        migrate_v1_to_v2(str(src), str(dst))

    message = str(excinfo.value)
    assert info["version_a"] in message
    assert "temperature" in message
    assert info["version_b"] in message
    assert "structure_scheme" in message
    assert not dst.exists()


def test_corrupt_group_unsupported_structure_scheme_named(tmp_path, v1_fixture):
    src, info = v1_fixture
    dst = tmp_path / "v2.db"

    conn = sqlite3.connect(src)
    conn.execute(
        "UPDATE net_currents SET structure_scheme = 'silk.structure/v9' "
        "WHERE corner = 'ss'"
    )
    conn.commit()
    conn.close()

    with pytest.raises(EvidenceMismatchError, match="structure_scheme"):
        migrate_v1_to_v2(str(src), str(dst))
    assert not dst.exists()


@pytest.mark.parametrize('historical_scheme', ['silk.structure/v1', 'silk.structure/v2'])
def test_schema_copy_preserves_legacy_structure_identity_without_authorizing_it(
    tmp_path, v1_fixture, historical_scheme,
):
    src, info = v1_fixture
    # Preserve the old scheme even when digest bytes match a current fixture.
    # The copy upgrades table shape, never evidence identity.
    with sqlite3.connect(src) as conn:
        for table, column in (
            ('circuit_versions', 'structure_scheme'),
            ('modules', 'module_hash_scheme'),
            ('net_currents', 'structure_scheme'),
        ):
            conn.execute(f"UPDATE {table} SET {column} = ?", (historical_scheme,))
    before = _sha256(src)
    dst = tmp_path / 'historical-schema-v2.db'
    migrate_v1_to_v2(str(src), str(dst))
    assert _sha256(src) == before
    v3 = tmp_path / "v3.db"
    migrate_v2_to_v3(dst, v3)
    v4 = tmp_path / "v4.db"
    migrate_v3_to_v4(v3, v4)
    current = tmp_path / "v5.db"
    v5 = tmp_path / 'migration-v5.db'
    migrate_v4_to_v5(v4, v5)
    v6 = tmp_path / 'v6.db'
    migrate_v5_to_v6(v5, v6)
    migrate_v6_to_v7(v6, current)
    with Database(str(current)) as db:
        for table, column in (
            ('circuit_versions', 'structure_scheme'),
            ('modules', 'module_hash_scheme'),
            ('net_current_sets', 'structure_scheme'),
        ):
            assert {row[0] for row in db.conn.execute(
                f'SELECT {column} FROM {table}'
            )} == {historical_scheme}
        assert db.conn.execute(
            'SELECT structure_hash FROM circuit_versions WHERE version_id = ?',
            (info['version_a'],),
        ).fetchone()[0] == info['structure_fp_a'].digest
        stored = list(db.conn.iterdump())
        for read in (
            lambda: db.get_simulation_results(info['version_a']),
            lambda: db.get_stored_specs(info['version_a']),
            lambda: db.get_net_current_set(info['version_a'], 'tt'),
        ):
            with pytest.raises(UnsupportedFingerprintSchemeError, match='rebuild'):
                read()
        assert list(db.conn.iterdump()) == stored
