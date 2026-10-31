"""The public ``silk.db`` file-format and compatibility boundary."""

import sqlite3

import pytest

from silk.db import (
    Database,
    DatabaseError,
    DatabaseCompatibilityError,
    EvidenceError,
    EvidenceMismatchError,
    EvidenceNotFoundError,
    LockedVersionError,
    UnsupportedFingerprintSchemeError,
    UnsupportedSchemaVersionError,
)
from silk.exceptions import CircuitError
from silk.db.schema import (
    DATABASE_FORMAT,
    DATABASE_SCHEMA_VERSION,
    required_columns,
)
from tests.db_snapshot_helpers import create_sealed_snapshot


def _create_metadata_database(path, *, schema_version, format_id=DATABASE_FORMAT):
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE silk_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                format_id TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                creator_version TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO silk_metadata
                (singleton, format_id, schema_version, creator_version)
            VALUES (1, ?, ?, 'fixture')
            """,
            (format_id, schema_version),
        )


def _assert_rejected_byte_identically(path, error_type, *, match):
    before = path.read_bytes()
    with pytest.raises(error_type, match=match):
        Database(path)
    assert path.read_bytes() == before


def _metadata_row(db):
    return db.conn.execute(
        """
        SELECT singleton, format_id, schema_version, created_at, creator_version
        FROM silk_metadata
        """
    ).fetchone()


def test_database_errors_have_typed_hierarchy():
    assert issubclass(DatabaseError, CircuitError)
    assert issubclass(DatabaseCompatibilityError, DatabaseError)
    assert issubclass(UnsupportedSchemaVersionError, DatabaseCompatibilityError)
    assert issubclass(
        UnsupportedFingerprintSchemeError,
        DatabaseCompatibilityError,
    )
    assert issubclass(EvidenceError, DatabaseError)
    assert issubclass(EvidenceNotFoundError, EvidenceError)
    assert issubclass(EvidenceMismatchError, EvidenceError)
    assert issubclass(LockedVersionError, DatabaseError)


def test_new_file_initializes_schema_v7(tmp_path):
    path = tmp_path / "new.db"
    assert not path.exists()

    with Database(path) as db:
        row = _metadata_row(db)
        assert row["singleton"] == 1
        assert row["format_id"] == DATABASE_FORMAT == "silk.db"
        assert row["schema_version"] == DATABASE_SCHEMA_VERSION == 7
        assert row["created_at"]

        tables = {
            record[0]
            for record in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        current = required_columns(DATABASE_SCHEMA_VERSION)
        assert set(current) <= tables
        for table, required in current.items():
            actual = {
                record[1]
                for record in db.conn.execute(f'PRAGMA table_info("{table}")')
            }
            assert required == actual, table


def test_zero_byte_file_initializes_schema_v5(tmp_path):
    path = tmp_path / "empty.db"
    path.touch()
    assert path.stat().st_size == 0

    with Database(path) as db:
        row = _metadata_row(db)
        assert (row["format_id"], row["schema_version"]) == (
            DATABASE_FORMAT,
            DATABASE_SCHEMA_VERSION,
        )


def test_failed_initialization_rolls_back_and_can_retry(tmp_path, monkeypatch):
    path = tmp_path / "retry.db"

    def fail_after_first_ddl(db, **kwargs):
        db.conn.execute("CREATE TABLE partial_initialization (id INTEGER)")
        raise RuntimeError("injected initialization failure")

    with monkeypatch.context() as patch:
        patch.setattr(Database, "_create_tables", fail_after_first_ddl)
        with pytest.raises(RuntimeError, match="injected initialization failure"):
            Database(path)

    with Database(path) as db:
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "partial_initialization" not in tables
        assert _metadata_row(db)["schema_version"] == DATABASE_SCHEMA_VERSION


def test_current_file_reopens_without_data_loss(tmp_path):
    path = tmp_path / "current.db"
    with Database(path) as db:
        version_id = create_sealed_snapshot(
            db, name="amplifier", version="1.0"
        )

    with Database(path) as reopened:
        assert reopened.get_circuit_by_id(version_id)["circuit_name"] == "amplifier"
        assert _metadata_row(reopened)["schema_version"] == DATABASE_SCHEMA_VERSION


def test_newer_schema_is_rejected_byte_identically(tmp_path):
    path = tmp_path / "newer.db"
    _create_metadata_database(path, schema_version=DATABASE_SCHEMA_VERSION + 1)

    _assert_rejected_byte_identically(
        path,
        UnsupportedSchemaVersionError,
        match=r"schema version 8.*supports schema version 7",
    )


def test_older_schema_is_rejected_byte_identically(tmp_path):
    path = tmp_path / "older.db"
    _create_metadata_database(path, schema_version=DATABASE_SCHEMA_VERSION - 1)

    _assert_rejected_byte_identically(
        path,
        UnsupportedSchemaVersionError,
        match=r"schema version 6.*supports schema version 7"
              r".*silk\.db\.migration\.migrate_v6_to_v7",
    )


def test_v1_file_is_rejected_with_migration_pointer(tmp_path):
    """AC5: a normal current open of a v1 file rejects without mutation and
    directs the caller to the explicit copy migration; it never upgrades
    implicitly."""
    path = tmp_path / "v1-pointer.db"
    _create_metadata_database(path, schema_version=1)

    _assert_rejected_byte_identically(
        path,
        UnsupportedSchemaVersionError,
        match=r"schema version 1.*supports schema version 7"
              r".*silk\.db\.migration\.migrate_v1_to_v2",
    )


def test_unversioned_legacy_silk_file_is_rejected_byte_identically(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE circuit_versions (version_id TEXT PRIMARY KEY)")

    _assert_rejected_byte_identically(
        path,
        DatabaseCompatibilityError,
        match=r"unversioned legacy silk database",
    )


def test_unrelated_sqlite_file_is_rejected_byte_identically(tmp_path):
    path = tmp_path / "unrelated.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE notes (body TEXT)")

    _assert_rejected_byte_identically(
        path,
        DatabaseCompatibilityError,
        match=r"not a silk database",
    )


def test_corrupt_file_is_rejected_byte_identically(tmp_path):
    path = tmp_path / "corrupt.db"
    path.write_bytes(b"this is not a SQLite database\n")

    _assert_rejected_byte_identically(
        path,
        DatabaseCompatibilityError,
        match=r"not a readable SQLite database",
    )


def test_wrong_format_identifier_is_rejected_byte_identically(tmp_path):
    path = tmp_path / "wrong-format.db"
    _create_metadata_database(
        path,
        schema_version=DATABASE_SCHEMA_VERSION,
        format_id="another.application",
    )

    _assert_rejected_byte_identically(
        path,
        DatabaseCompatibilityError,
        match=r"format identifier.*another\.application.*expected.*silk\.db",
    )


def test_current_metadata_with_missing_required_column_is_rejected(tmp_path):
    path = tmp_path / "malformed-current.db"
    _create_metadata_database(path, schema_version=DATABASE_SCHEMA_VERSION)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE circuit_versions (version_id TEXT PRIMARY KEY)")

    _assert_rejected_byte_identically(
        path,
        DatabaseCompatibilityError,
        match=r"circuit_versions.*missing required columns.*circuit_name",
    )


def test_memory_database_initializes_normally():
    with Database(":memory:") as db:
        row = _metadata_row(db)
        assert (row["format_id"], row["schema_version"]) == (
            DATABASE_FORMAT,
            DATABASE_SCHEMA_VERSION,
        )
