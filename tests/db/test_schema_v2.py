"""V2 columns and relationships retained in databases v3 and v4.

Checks the required physical columns of the schema-v2 tables.
"""

import sqlite3

import pytest

from silk.db import Database
from silk.db.identity import (
    PROFILE_FINGERPRINT_SCHEME,
    STRUCTURE_FINGERPRINT_SCHEME,
    _fingerprints_equal,
    _require_fingerprint_scheme,
)
from silk.db.models import Fingerprint
from silk.db.schema import (
    DATABASE_SCHEMA_VERSION,
    _validate_database_version,
    required_columns,
    validate_database_connection,
)
from silk.exceptions import (
    DatabaseCompatibilityError,
    UnsupportedFingerprintSchemeError,
)


def test_schema_version_is_7():
    assert DATABASE_SCHEMA_VERSION == 7


def test_clean_create_declares_exactly_the_v2_columns():
    """Current creation keeps the exact v2 column sets, with only the v4
    (TASK-306) provenance and v5 (TASK-409) condition-state additions."""
    with Database(":memory:") as db:
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        current = required_columns(DATABASE_SCHEMA_VERSION)
        assert set(current) <= tables
        for table, required in current.items():
            actual = {
                row[1] for row in db.conn.execute(f'PRAGMA table_info("{table}")')
            }
            assert required == actual, table


def test_clean_create_passes_current_validation():
    """A freshly created database passes ``validate_database_connection``
    (already implied by ``Database.__init__``, asserted directly here too)."""
    with Database(":memory:") as db:
        validate_database_connection(db.conn, path_label=":memory:")


def test_new_tables_present():
    with Database(":memory:") as db:
        tables = {
            row[0]
            for row in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "characterization_profiles" in tables
        assert "net_current_sets" in tables


@pytest.mark.parametrize(
    "table,constraint_sql_fragment",
    [
        (
            "characterization_profiles",
            "FOREIGN KEY (circuit_version_id) "
            "REFERENCES circuit_versions(version_id) ON DELETE RESTRICT",
        ),
        (
            "net_current_sets",
            "FOREIGN KEY (circuit_version_id, profile_id) "
            "REFERENCES characterization_profiles(circuit_version_id, profile_id) "
            "ON DELETE RESTRICT",
        ),
        (
            "net_currents",
            "FOREIGN KEY (current_set_id) "
            "REFERENCES net_current_sets(current_set_id) ON DELETE RESTRICT",
        ),
        (
            "simulation_results",
            "FOREIGN KEY (circuit_version_id, profile_id) "
            "REFERENCES characterization_profiles(circuit_version_id, profile_id) "
            "ON DELETE RESTRICT",
        ),
        (
            "spec_measurements",
            "FOREIGN KEY (circuit_version_id, spec_id) "
            "REFERENCES specs(circuit_version_id, spec_id) ON DELETE CASCADE",
        ),
    ],
)
def test_lifecycle_fk_declared_in_ddl(table, constraint_sql_fragment):
    """Smoke-check that the literal DDL really declares each lifecycle FK
    (design §6) — a second signal alongside the introspection-based
    validation test below."""
    with Database(":memory:") as db:
        sql = db.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()[0]
        normalized = " ".join(sql.split())
        assert " ".join(constraint_sql_fragment.split()) in normalized


def test_foreign_key_check_clean_on_populated_database():
    from tests.db_snapshot_helpers import create_sealed_snapshot

    with Database(":memory:") as db:
        create_sealed_snapshot(db, name="amp", version="1.0")
        violations = db.conn.execute("PRAGMA foreign_key_check").fetchall()
        assert violations == []


# --- Same-column-impostor rejection: REQUIRED_COLUMNS alone is not enough --


def _impostor_connection_missing_fks():
    """Build an in-memory DB with the right TABLE/COLUMN names but with the
    lifecycle FKs and partial unique indexes silently omitted — the exact
    shape REQUIRED_COLUMNS-only validation would wrongly accept."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE silk_metadata (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            format_id TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            creator_version TEXT
        );
        INSERT INTO silk_metadata (singleton, format_id, schema_version)
        VALUES (1, 'silk.db', 3);

        CREATE TABLE circuit_versions (
            version_id TEXT PRIMARY KEY, circuit_name TEXT NOT NULL,
            version TEXT NOT NULL, stage TEXT NOT NULL, state TEXT NOT NULL,
            is_locked BOOLEAN, parent_version_id TEXT, created_at TIMESTAMP,
            created_by TEXT, description TEXT, netlist_path TEXT,
            metadata TEXT, simulation_summary TEXT, last_summary_update TIMESTAMP,
            structure_scheme TEXT NOT NULL, structure_hash TEXT NOT NULL,
            spec_scheme TEXT NOT NULL, spec_set_hash TEXT NOT NULL
        );
        CREATE TABLE devices (
            device_id TEXT PRIMARY KEY, circuit_version_id TEXT NOT NULL,
            name TEXT NOT NULL, type TEXT NOT NULL, model TEXT, properties TEXT
        );
        CREATE TABLE nets (
            net_id TEXT PRIMARY KEY, circuit_version_id TEXT NOT NULL,
            name TEXT NOT NULL, net_type TEXT, properties TEXT
        );
        CREATE TABLE parasitics (
            parasitic_id TEXT PRIMARY KEY, circuit_version_id TEXT NOT NULL,
            net_id TEXT NOT NULL, parasitic_type TEXT, value REAL, unit TEXT,
            from_node TEXT, to_node TEXT, extraction_tool TEXT, timestamp TIMESTAMP
        );
        CREATE TABLE releases (
            release_id TEXT PRIMARY KEY, release_name TEXT UNIQUE NOT NULL,
            stage TEXT NOT NULL, created_at TIMESTAMP, created_by TEXT,
            notes TEXT, status TEXT
        );
        CREATE TABLE release_contents (
            release_id TEXT NOT NULL, circuit_version_id TEXT NOT NULL,
            PRIMARY KEY (release_id, circuit_version_id)
        );
        CREATE TABLE modules (
            module_id TEXT PRIMARY KEY, module_name TEXT NOT NULL,
            module_path TEXT NOT NULL, module_hash_scheme TEXT NOT NULL,
            module_hash TEXT NOT NULL, device_count INTEGER, structure TEXT,
            created_at TIMESTAMP
        );
        CREATE TABLE circuit_modules (
            circuit_version_id TEXT NOT NULL, module_id TEXT NOT NULL,
            instance_name TEXT, position_in_hierarchy INTEGER,
            PRIMARY KEY (circuit_version_id, module_id, instance_name)
        );

        -- characterization_profiles: right columns, NO unique constraints,
        -- NO foreign key. A REQUIRED_COLUMNS-only check would accept this.
        CREATE TABLE characterization_profiles (
            profile_id TEXT PRIMARY KEY, circuit_version_id TEXT NOT NULL,
            owner_path TEXT NOT NULL, test_name TEXT NOT NULL,
            parameters_json TEXT NOT NULL, profile_scheme TEXT NOT NULL,
            profile_hash TEXT NOT NULL, created_at TIMESTAMP
        );

        CREATE TABLE simulation_results (
            sim_id TEXT PRIMARY KEY, circuit_version_id TEXT NOT NULL,
            object_id TEXT, object_type TEXT, source TEXT NOT NULL,
            sim_type TEXT, corner TEXT, temperature REAL, voltage REAL,
            parameter TEXT NOT NULL, value REAL, value_text TEXT, unit TEXT,
            tool_name TEXT, timestamp TIMESTAMP, conditions TEXT,
            profile_id TEXT
        );

        -- net_current_sets: right columns, but the two named partial unique
        -- indexes are omitted and there is no FK to net_currents' child.
        CREATE TABLE net_current_sets (
            current_set_id TEXT PRIMARY KEY, circuit_version_id TEXT NOT NULL,
            profile_id TEXT, state TEXT NOT NULL,
            supersedes_current_set_id TEXT UNIQUE, corner TEXT NOT NULL,
            temperature REAL, voltage REAL, tool_name TEXT, analysis TEXT,
            tstart REAL, results_file TEXT, structure_scheme TEXT NOT NULL,
            structure_hash TEXT NOT NULL, selected_at TIMESTAMP,
            superseded_at TIMESTAMP
        );

        CREATE TABLE net_currents (
            current_id TEXT PRIMARY KEY, current_set_id TEXT NOT NULL,
            net_name TEXT NOT NULL, pin_id TEXT NOT NULL DEFAULT '',
            avg_a REAL NOT NULL, rms_a REAL, peak_a REAL,
            source TEXT NOT NULL, current_vector TEXT, timestamp TIMESTAMP
        );

        CREATE TABLE specs (
            spec_id TEXT PRIMARY KEY, circuit_version_id TEXT NOT NULL,
            owner_path TEXT NOT NULL, spec_key TEXT NOT NULL, name TEXT NOT NULL,
            metric TEXT NOT NULL, op TEXT NOT NULL, target REAL NOT NULL,
            units TEXT, description TEXT, parent_id TEXT, declared_at TEXT,
            definition_scheme TEXT NOT NULL, definition_hash TEXT NOT NULL,
            created_at TIMESTAMP
        );

        CREATE TABLE spec_measurements (
            measurement_id TEXT PRIMARY KEY, circuit_version_id TEXT NOT NULL,
            spec_id TEXT NOT NULL, profile_id TEXT, source TEXT NOT NULL,
            value REAL NOT NULL, corner TEXT, temperature REAL, voltage REAL,
            sim_type TEXT, tool_name TEXT, timestamp REAL, sim_run_id TEXT,
            notes TEXT, passes INTEGER
        );
        """
    )
    return conn


def test_impostor_missing_lifecycle_fks_is_rejected():
    conn = _impostor_connection_missing_fks()
    with pytest.raises(DatabaseCompatibilityError, match="foreign key"):
        # The v3-labelled impostor is checked against the v3 shape it claims
        # (a normal open now requires v4 and would refuse on the version).
        _validate_database_version(conn, path_label="impostor", schema_version=3)


@pytest.mark.parametrize(
    "index_name",
    ["idx_current_sets_scoped_current", "idx_current_sets_unscoped_current"],
)
def test_impostor_missing_partial_unique_index_is_rejected(index_name):
    """A database with every lifecycle FK intact but one named partial
    unique index dropped must still be rejected — isolates the index check
    from the FK check above (both are required, independently)."""
    with Database(":memory:") as db:
        db.conn.execute(f"DROP INDEX {index_name}")
        with pytest.raises(DatabaseCompatibilityError, match=index_name):
            validate_database_connection(db.conn, path_label="impostor")


# --- silk.profile/v1 scheme registration is domain-specific -----------------


def test_profile_scheme_is_accepted_when_explicitly_expected():
    fp = Fingerprint(PROFILE_FINGERPRINT_SCHEME, "0" * 64)
    _require_fingerprint_scheme(fp, PROFILE_FINGERPRINT_SCHEME)  # does not raise


def test_profile_scheme_never_compares_equal_to_structure_scheme():
    """Registering silk.profile/v1 as globally supported must never let a
    profile digest be compared as a structure digest, or vice versa."""
    profile_fp = Fingerprint(PROFILE_FINGERPRINT_SCHEME, "a" * 64)
    structure_fp = Fingerprint(STRUCTURE_FINGERPRINT_SCHEME, "a" * 64)

    with pytest.raises(UnsupportedFingerprintSchemeError):
        _fingerprints_equal(
            profile_fp, structure_fp, expected_scheme=STRUCTURE_FINGERPRINT_SCHEME
        )
    with pytest.raises(UnsupportedFingerprintSchemeError):
        _fingerprints_equal(
            profile_fp, structure_fp, expected_scheme=PROFILE_FINGERPRINT_SCHEME
        )
    with pytest.raises(UnsupportedFingerprintSchemeError):
        _require_fingerprint_scheme(profile_fp, STRUCTURE_FINGERPRINT_SCHEME)
