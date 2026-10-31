"""Versioned SQLite file-format policy for :mod:`silk.db`.

Existing non-empty files are inspected through a read-only connection before
``Database`` opens them normally.  Keep this module free of schema mutation:
table creation remains an explicit new-destination operation in
``circuit_db.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping

from silk.exceptions import (
    DatabaseCompatibilityError,
    UnsupportedSchemaVersionError,
)


DATABASE_FORMAT = "silk.db"
DATABASE_SCHEMA_VERSION = 1

# DB-01 freezes schema v1 only after all of its behavior-owning tasks land.
# Until then, each task updates this registry in lockstep with the columns it
# owns; an earlier development v1 file is rejected, never silently altered.
REQUIRED_COLUMNS: Mapping[str, frozenset[str]] = {
    "silk_metadata": frozenset(
        {
            "singleton",
            "format_id",
            "schema_version",
            "created_at",
            "creator_version",
        }
    ),
    "circuit_versions": frozenset(
        {
            "version_id",
            "circuit_name",
            "version",
            "stage",
            "state",
            "is_locked",
            "parent_version_id",
            "created_at",
            "created_by",
            "description",
            "netlist_path",
            "metadata",
            "simulation_summary",
            "last_summary_update",
            "structure_scheme",
            "structure_hash",
            "spec_scheme",
            "spec_set_hash",
        }
    ),
    "devices": frozenset(
        {"device_id", "circuit_version_id", "name", "type", "model", "properties"}
    ),
    "nets": frozenset(
        {"net_id", "circuit_version_id", "name", "net_type", "properties"}
    ),
    "simulation_results": frozenset(
        {
            "sim_id",
            "circuit_version_id",
            "object_id",
            "object_type",
            "source",
            "sim_type",
            "corner",
            "temperature",
            "voltage",
            "parameter",
            "value",
            "value_text",
            "unit",
            "tool_name",
            "timestamp",
            "conditions",
        }
    ),
    "parasitics": frozenset(
        {
            "parasitic_id",
            "circuit_version_id",
            "net_id",
            "parasitic_type",
            "value",
            "unit",
            "from_node",
            "to_node",
            "extraction_tool",
            "timestamp",
        }
    ),
    "net_currents": frozenset(
        {
            "current_id",
            "circuit_version_id",
            "net_name",
            "pin_id",
            "avg_a",
            "rms_a",
            "peak_a",
            "source",
            "current_vector",
            "analysis",
            "tstart",
            "results_file",
            "corner",
            "temperature",
            "voltage",
            "tool_name",
            "structure_scheme",
            "structure_hash",
            "timestamp",
        }
    ),
    "releases": frozenset(
        {
            "release_id",
            "release_name",
            "stage",
            "created_at",
            "created_by",
            "notes",
            "status",
        }
    ),
    "release_contents": frozenset({"release_id", "circuit_version_id"}),
    "modules": frozenset(
        {
            "module_id",
            "module_name",
            "module_path",
            "module_hash_scheme",
            "module_hash",
            "device_count",
            "structure",
            "created_at",
        }
    ),
    "circuit_modules": frozenset(
        {
            "circuit_version_id",
            "module_id",
            "instance_name",
            "position_in_hierarchy",
        }
    ),
    "specs": frozenset(
        {
            "spec_id",
            "circuit_version_id",
            "owner_path",
            "spec_key",
            "name",
            "metric",
            "op",
            "target",
            "units",
            "description",
            "parent_id",
            "declared_at",
            "definition_scheme",
            "definition_hash",
            "created_at",
        }
    ),
    "spec_measurements": frozenset(
        {
            "measurement_id",
            "spec_id",
            "source",
            "value",
            "corner",
            "temperature",
            "voltage",
            "sim_type",
            "tool_name",
            "timestamp",
            "sim_run_id",
            "notes",
            "passes",
        }
    ),
}


_PRESERVE_GUIDANCE = (
    "Preserve the original file and create a new schema-v1 database; "
    "silk 0.2 does not migrate database files in place."
)


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    # ``table`` comes only from the static REQUIRED_COLUMNS registry.
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _validate_required_schema(
    conn: sqlite3.Connection,
    *,
    path_label: str,
) -> None:
    tables = _table_names(conn)
    for table, required in REQUIRED_COLUMNS.items():
        if table not in tables:
            raise DatabaseCompatibilityError(
                f"Database '{path_label}' reports silk schema version "
                f"{DATABASE_SCHEMA_VERSION} but is missing required table "
                f"'{table}'. {_PRESERVE_GUIDANCE}"
            )
        missing = sorted(required - _columns(conn, table))
        if missing:
            raise DatabaseCompatibilityError(
                f"Database '{path_label}' table '{table}' is missing required "
                f"columns: {', '.join(missing)}. {_PRESERVE_GUIDANCE}"
            )


def _read_and_validate_metadata(
    conn: sqlite3.Connection,
    *,
    path_label: str,
) -> None:
    tables = _table_names(conn)
    if "silk_metadata" not in tables:
        if "circuit_versions" in tables:
            kind = "an unversioned legacy silk database"
        else:
            kind = "not a silk database"
        raise DatabaseCompatibilityError(
            f"Database '{path_label}' is {kind}: required table "
            f"'silk_metadata' is absent. {_PRESERVE_GUIDANCE}"
        )

    missing_metadata = sorted(
        REQUIRED_COLUMNS["silk_metadata"] - _columns(conn, "silk_metadata")
    )
    if missing_metadata:
        raise DatabaseCompatibilityError(
            f"Database '{path_label}' table 'silk_metadata' is missing required "
            f"columns: {', '.join(missing_metadata)}. {_PRESERVE_GUIDANCE}"
        )

    rows = conn.execute(
        """
        SELECT singleton, format_id, schema_version
        FROM silk_metadata
        """
    ).fetchall()
    if len(rows) != 1 or rows[0][0] != 1:
        raise DatabaseCompatibilityError(
            f"Database '{path_label}' has invalid silk_metadata: expected one "
            f"singleton row with singleton=1. {_PRESERVE_GUIDANCE}"
        )

    format_id = rows[0][1]
    schema_version = rows[0][2]
    if format_id != DATABASE_FORMAT:
        raise DatabaseCompatibilityError(
            f"Database '{path_label}' has format identifier {format_id!r}; "
            f"expected {DATABASE_FORMAT!r}. {_PRESERVE_GUIDANCE}"
        )
    if schema_version != DATABASE_SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Database '{path_label}' uses silk schema version "
            f"{schema_version!r}; this release supports schema version "
            f"{DATABASE_SCHEMA_VERSION}. {_PRESERVE_GUIDANCE}"
        )


def validate_database_connection(
    conn: sqlite3.Connection,
    *,
    path_label: str,
) -> None:
    """Validate format identity, version, and the required v1 shape."""

    try:
        _read_and_validate_metadata(conn, path_label=path_label)
        _validate_required_schema(conn, path_label=path_label)
    except (DatabaseCompatibilityError, UnsupportedSchemaVersionError):
        raise
    except sqlite3.DatabaseError as exc:
        raise DatabaseCompatibilityError(
            f"Database '{path_label}' is not a readable SQLite database: "
            f"{exc}. {_PRESERVE_GUIDANCE}"
        ) from exc


def inspect_existing_database(path: str | Path) -> None:
    """Read-only preflight for an existing, non-empty database file."""

    resolved = Path(path).resolve()
    uri = f"{resolved.as_uri()}?mode=ro"
    conn = None
    try:
        conn = sqlite3.connect(uri, uri=True)
        validate_database_connection(conn, path_label=str(resolved))
    except (DatabaseCompatibilityError, UnsupportedSchemaVersionError):
        raise
    except sqlite3.DatabaseError as exc:
        raise DatabaseCompatibilityError(
            f"Database '{resolved}' is not a readable SQLite database: "
            f"{exc}. {_PRESERVE_GUIDANCE}"
        ) from exc
    finally:
        if conn is not None:
            conn.close()


__all__ = [
    "DATABASE_FORMAT",
    "DATABASE_SCHEMA_VERSION",
    "REQUIRED_COLUMNS",
    "inspect_existing_database",
    "validate_database_connection",
]
