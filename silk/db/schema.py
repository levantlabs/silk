"""Versioned SQLite file-format policy for :mod:`silk.db`.

Existing non-empty files are inspected through a read-only connection before
``Database`` opens them normally.  Keep this module free of schema mutation:
table creation remains an explicit new-destination operation in
``circuit_db.py``. Upgrades use the explicit v1-to-v2, v2-to-v3, v3-to-v4 and
v4-to-v5, v5-to-v6 and v6-to-v7 copy migrations in ``silk.db.migration``, never an in-place
migration.

v7 also permits supply sweeps, with voltage=NULL; profile states stay fixed/none.

v5 stores the state of each operating condition per row:
``corner_state``, ``temperature_state`` and ``voltage_state`` on
``simulation_results``, ``spec_measurements`` and ``net_current_sets``, one of
``'fixed'`` (the value column is set), ``'none'`` (provably does not apply),
``'unknown'`` and, for temperature, ``'sweep'`` (the value column is NULL).
A scoped row has every fact fixed or none. ``net_current_sets.corner`` is
nullable, and its two current-selection unique indexes key on the states.

v4 adds the ``run_provenance`` table (one content-addressed row
per distinct ``silk.simulation.provenance`` record) and a nullable ``provenance_id``
foreign key on ``simulation_results``, ``spec_measurements`` and
``net_current_sets``. NULL means no provenance was recorded.

In v3, a spec_measurements row with value IS NULL is an evaluated but
unavailable observation: typed readback/restore returns float('nan'). No row
means no observation. A simulation_results row decodes to nan only when both
value and value_text are NULL; text/error rows retain value=None and their
text. Public writers still require a numeric value (including nan) or text.
NULL context fields retain None. Metric schema format 2.0 is independent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from silk.exceptions import (
    DatabaseCompatibilityError,
    UnsupportedSchemaVersionError,
)


DATABASE_FORMAT = "silk.db"
DATABASE_SCHEMA_VERSION = 7

# Schema v4 (TASK-306) adds the run_provenance table and a nullable
# provenance_id column on the three measurement-type tables; everything else
# is the v3 shape. Versions 2 and 3 are validated only as explicit migration
# sources/destinations.
_V4_ADDED_COLUMNS: Mapping[str, frozenset[str]] = {
    "run_provenance": frozenset(
        {"provenance_id", "scheme", "digest", "record_json", "created_at"}
    ),
    "simulation_results": frozenset({"provenance_id"}),
    "spec_measurements": frozenset({"provenance_id"}),
    "net_current_sets": frozenset({"provenance_id"}),
}

# Schema v5 (TASK-409): the per-row condition states.
_V5_STATE_COLUMNS = frozenset({"corner_state", "temperature_state", "voltage_state"})
_V5_ADDED_COLUMNS: Mapping[str, frozenset[str]] = {
    "simulation_results": _V5_STATE_COLUMNS,
    "spec_measurements": _V5_STATE_COLUMNS,
    "net_current_sets": _V5_STATE_COLUMNS,
}

# Required-column sets describe the physical schema. Unchanged v1 tables
# keep their v1 entries verbatim. V3 keeps all v2 columns and changes
# only spec_measurements.value nullability, validated separately below.
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
            "profile_id",
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
    "characterization_profiles": frozenset(
        {
            "profile_id",
            "circuit_version_id",
            "owner_path",
            "test_name",
            "parameters_json",
            "profile_scheme",
            "profile_hash",
            "created_at",
        }
    ),
    "net_current_sets": frozenset(
        {
            "current_set_id",
            "circuit_version_id",
            "profile_id",
            "state",
            "supersedes_current_set_id",
            "corner",
            "temperature",
            "voltage",
            "tool_name",
            "analysis",
            "tstart",
            "results_file",
            "structure_scheme",
            "structure_hash",
            "selected_at",
            "superseded_at",
        }
    ),
    "net_currents": frozenset(
        {
            "current_id",
            "current_set_id",
            "net_name",
            "pin_id",
            "avg_a",
            "rms_a",
            "peak_a",
            "source",
            "current_vector",
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
            "circuit_version_id",
            "spec_id",
            "profile_id",
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


def required_columns(schema_version: int) -> Mapping[str, frozenset[str]]:
    """The required-column map for one schema version (2, 3, 4 or 5)."""
    merged = dict(REQUIRED_COLUMNS)
    if schema_version >= 6:
        merged["spec_measurements"] = merged["spec_measurements"] | {"aggregate_json"}
    for since, added_columns in ((4, _V4_ADDED_COLUMNS), (5, _V5_ADDED_COLUMNS)):
        if schema_version >= since:
            for table, added in added_columns.items():
                merged[table] = merged.get(table, frozenset()) | added
    return merged


_PRESERVE_GUIDANCE = (
    "Preserve the original file and create a new schema-v7 database; "
    "silk does not migrate database files in place."
)

_V1_TO_V2_GUIDANCE = (
    "A schema-v1 database is never upgraded implicitly. Create a distinct, "
    "non-existing destination file with "
    "silk.db.migration.migrate_v1_to_v2(source_path, dest_path); the source "
    "file is left byte-identical. Then migrate that v2 copy to v3, and the "
    "v3 copy to v4, the v4 copy to v5, the v5 copy to v6, and the v6 copy to v7."
)


_V2_TO_V3_GUIDANCE = (
    "A schema-v2 database is never upgraded implicitly. Create a distinct, "
    "non-existing destination file with "
    "silk.db.migration.migrate_v2_to_v3(source_path, dest_path); the source "
    "file is left byte-identical. Then migrate that v3 copy to v4, and the "
    "v4 copy to v5, the v5 copy to v6, and the v6 copy to v7."
)


_V3_TO_V4_GUIDANCE = (
    "A schema-v3 database is never upgraded implicitly. Create a distinct, "
    "non-existing destination file with "
    "silk.db.migration.migrate_v3_to_v4(source_path, dest_path); the source "
    "file is left byte-identical. Then migrate that v4 copy to v5, the v5 copy to v6, and the v6 copy to v7."
)


_V4_TO_V5_GUIDANCE = (
    "A schema-v4 database is never upgraded implicitly. Create a distinct, "
    "non-existing destination file with "
    "silk.db.migration.migrate_v4_to_v5(source_path, dest_path); the source "
    "file is left byte-identical. Existing NULL corner/temperature/voltage "
    "values become 'unknown', never 'none'. Then migrate that v5 copy to v6 and the v6 copy to v7."
)


# Lifecycle-critical composite/simple foreign keys that a same-column
# impostor (right column names, no real constraints) would not have. Each
# entry: (from_columns, to_table, to_columns, on_delete_action).
_V2_FOREIGN_KEYS: Mapping[str, Tuple[Tuple[Tuple[str, ...], str, Tuple[str, ...], str], ...]] = {
    "characterization_profiles": (
        (("circuit_version_id",), "circuit_versions", ("version_id",), "RESTRICT"),
    ),
    "net_current_sets": (
        (("circuit_version_id",), "circuit_versions", ("version_id",), "RESTRICT"),
        (
            ("circuit_version_id", "profile_id"),
            "characterization_profiles",
            ("circuit_version_id", "profile_id"),
            "RESTRICT",
        ),
        (
            ("supersedes_current_set_id",),
            "net_current_sets",
            ("current_set_id",),
            "RESTRICT",
        ),
    ),
    "net_currents": (
        (("current_set_id",), "net_current_sets", ("current_set_id",), "RESTRICT"),
    ),
    "simulation_results": (
        (
            ("circuit_version_id", "profile_id"),
            "characterization_profiles",
            ("circuit_version_id", "profile_id"),
            "RESTRICT",
        ),
    ),
    "spec_measurements": (
        (
            ("circuit_version_id", "spec_id"),
            "specs",
            ("circuit_version_id", "spec_id"),
            "CASCADE",
        ),
        (
            ("circuit_version_id", "profile_id"),
            "characterization_profiles",
            ("circuit_version_id", "profile_id"),
            "RESTRICT",
        ),
    ),
}

# A unique constraint/index must exist over exactly these columns (in this
# order) on this table. This is the FK-target/idempotence backbone; it does
# not enumerate every v1 index, only the ones DB-02 lifecycle safety depends
# on.
_V2_UNIQUE_COLUMN_SETS: Mapping[str, Tuple[Tuple[str, ...], ...]] = {
    "characterization_profiles": (
        ("circuit_version_id", "profile_scheme", "profile_hash"),
        ("circuit_version_id", "profile_id"),
    ),
    "net_current_sets": (
        ("circuit_version_id", "current_set_id"),
        ("supersedes_current_set_id",),
    ),
    "net_currents": (("current_set_id", "net_name", "pin_id"),),
    "specs": (("circuit_version_id", "spec_id"),),
}

# Named partial-unique indexes that enforce "at most one current row per
# selection key" and must exist by name, be UNIQUE, cover exactly these
# columns, and carry a WHERE predicate containing these normalized
# substrings (checked against whitespace-collapsed, upper-cased SQL text).
_V2_NAMED_PARTIAL_UNIQUE_INDEXES: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    (
        "idx_current_sets_scoped_current",
        "net_current_sets",
        (
            "circuit_version_id",
            "profile_id",
            "corner",
            "temperature",
            "voltage",
        ),
        ("STATE = 'CURRENT'", "PROFILE_ID IS NOT NULL"),
    ),
    (
        "idx_current_sets_unscoped_current",
        "net_current_sets",
        ("circuit_version_id", "corner"),
        ("STATE = 'CURRENT'", "PROFILE_ID IS NULL"),
    ),
)


# Schema v5 (TASK-409): the same two indexes key on the condition states. A
# None column is an expression (``ifnull(value, '')``, so that the NULL of a
# none or unknown fact still collides), checked through the SQL text.
_V5_NAMED_PARTIAL_UNIQUE_INDEXES: Tuple[Tuple[str, str, Tuple[Optional[str], ...], Tuple[str, ...]], ...] = (
    (
        "idx_current_sets_scoped_current",
        "net_current_sets",
        (
            "circuit_version_id",
            "profile_id",
            "corner_state",
            None,
            "temperature_state",
            None,
            "voltage_state",
            None,
        ),
        (
            "STATE = 'CURRENT'",
            "PROFILE_ID IS NOT NULL",
            "IFNULL(CORNER, '')",
            "IFNULL(TEMPERATURE, '')",
            "IFNULL(VOLTAGE, '')",
        ),
    ),
    (
        "idx_current_sets_unscoped_current",
        "net_current_sets",
        ("circuit_version_id", "corner_state", None),
        ("STATE = 'CURRENT'", "PROFILE_ID IS NULL", "IFNULL(CORNER, '')"),
    ),
)


# Schema v4: each provenance link must be a real foreign key.
_V4_FOREIGN_KEYS: Mapping[str, Tuple[Tuple[Tuple[str, ...], str, Tuple[str, ...], str], ...]] = {
    table: ((("provenance_id",), "run_provenance", ("provenance_id",), "RESTRICT"),)
    for table in ("simulation_results", "spec_measurements", "net_current_sets")
}
_V4_UNIQUE_COLUMN_SETS: Mapping[str, Tuple[Tuple[str, ...], ...]] = {
    "run_provenance": (("scheme", "digest"),),
}


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


def _foreign_keys(conn: sqlite3.Connection, table: str):
    """Return this table's declared foreign keys as composite tuples.

    ``PRAGMA foreign_key_list`` emits one row per referencing column, sharing
    an ``id`` across the columns of one composite constraint; this groups
    them back into ``(from_columns, to_table, to_columns, on_delete)``
    ordered by ``seq`` so composite column order is preserved.
    """
    rows = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
    grouped: dict = {}
    for row in rows:
        # row layout: id, seq, table, from, to, on_update, on_delete, match
        grouped.setdefault(row[0], []).append(row)
    result = []
    for entries in grouped.values():
        entries.sort(key=lambda r: r[1])
        result.append(
            (
                tuple(entry[3] for entry in entries),
                entries[0][2],
                tuple(entry[4] for entry in entries),
                entries[0][6],
            )
        )
    return result


def _unique_index_column_sets(conn: sqlite3.Connection, table: str) -> set:
    """Return the set of exact column tuples covered by a UNIQUE index or
    table-level UNIQUE constraint on ``table`` (auto-named or not)."""
    result = set()
    for idx in conn.execute(f'PRAGMA index_list("{table}")').fetchall():
        # idx layout: seq, name, unique, origin, partial
        if not idx[2]:
            continue
        name = idx[1]
        cols = tuple(
            info[2]
            for info in conn.execute(f'PRAGMA index_info("{name}")').fetchall()
        )
        result.add(cols)
    return result


def _index_sql(conn: sqlite3.Connection, name: str) -> Optional[str]:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    ).fetchone()
    return row[0] if row is not None else None


def _validate_v2_relationships(
    conn: sqlite3.Connection,
    *,
    path_label: str,
    schema_version: int = 3,
) -> None:
    """Verify the lifecycle-critical FK graph and unique indexes.

    ``REQUIRED_COLUMNS`` alone accepts a same-column impostor: a table with
    the right column names but no real constraints. This walks the actual
    declared foreign keys and unique indexes so a corrupt or hand-crafted v2
    file is rejected before any behavior task trusts it.
    """
    foreign_keys = {t: tuple(v) for t, v in _V2_FOREIGN_KEYS.items()}
    unique_sets = {t: tuple(v) for t, v in _V2_UNIQUE_COLUMN_SETS.items()}
    if schema_version >= 4:
        for t, v in _V4_FOREIGN_KEYS.items():
            foreign_keys[t] = foreign_keys.get(t, ()) + v
        for t, v in _V4_UNIQUE_COLUMN_SETS.items():
            unique_sets[t] = unique_sets.get(t, ()) + v
    for table, expected_fks in foreign_keys.items():
        actual_fks = set(_foreign_keys(conn, table))
        for expected in expected_fks:
            if expected not in actual_fks:
                from_cols, to_table, to_cols, on_delete = expected
                raise DatabaseCompatibilityError(
                    f"Database '{path_label}' table '{table}' is missing the "
                    f"required foreign key {', '.join(from_cols)} -> "
                    f"{to_table}({', '.join(to_cols)}) ON DELETE {on_delete}. "
                    f"{_PRESERVE_GUIDANCE}"
                )

    for table, expected_unique_sets in unique_sets.items():
        actual = _unique_index_column_sets(conn, table)
        for expected in expected_unique_sets:
            if expected not in actual:
                raise DatabaseCompatibilityError(
                    f"Database '{path_label}' table '{table}' is missing a "
                    f"required UNIQUE constraint over columns "
                    f"({', '.join(expected)}). {_PRESERVE_GUIDANCE}"
                )

    named_indexes = (
        _V5_NAMED_PARTIAL_UNIQUE_INDEXES
        if schema_version >= 5
        else _V2_NAMED_PARTIAL_UNIQUE_INDEXES
    )
    for name, table, expected_cols, required_substrings in named_indexes:
        index_rows = conn.execute(
            f'PRAGMA index_list("{table}")'
        ).fetchall()
        matching = [row for row in index_rows if row[1] == name]
        if not matching or not matching[0][2]:
            raise DatabaseCompatibilityError(
                f"Database '{path_label}' table '{table}' is missing the "
                f"required unique partial index '{name}'. {_PRESERVE_GUIDANCE}"
            )
        cols = tuple(
            info[2]
            for info in conn.execute(f'PRAGMA index_info("{name}")').fetchall()
        )
        if cols != expected_cols:
            raise DatabaseCompatibilityError(
                f"Database '{path_label}' index '{name}' on table '{table}' "
                f"covers columns {cols} but requires {expected_cols}. "
                f"{_PRESERVE_GUIDANCE}"
            )
        sql_text = _index_sql(conn, name)
        normalized = " ".join((sql_text or "").split()).upper()
        missing_substrings = [
            substring
            for substring in required_substrings
            if substring not in normalized
        ]
        if missing_substrings:
            raise DatabaseCompatibilityError(
                f"Database '{path_label}' index '{name}' on table '{table}' "
                f"is missing its required partial predicate terms "
                f"{missing_substrings}. {_PRESERVE_GUIDANCE}"
            )


# Each v5 evidence table must carry these CHECKs (whitespace-collapsed,
# upper-cased), or a same-column impostor could store a state that disagrees
# with its value.
_V5_REQUIRED_CHECKS: Tuple[str, ...] = tuple(
    f"CHECK(({column}_STATE = 'FIXED') = ({column} IS NOT NULL))"
    for column in ("CORNER", "TEMPERATURE", "VOLTAGE")
) + (
    "CORNER_STATE IN ('FIXED','NONE','UNKNOWN')",
    "TEMPERATURE_STATE IN ('FIXED','NONE','UNKNOWN','SWEEP')",
    "VOLTAGE_STATE IN ('FIXED','NONE','UNKNOWN')",
    "CHECK(PROFILE_ID IS NULL OR (CORNER_STATE IN ('FIXED','NONE') AND",
)


def _validate_v5_state_checks(conn: sqlite3.Connection, *, path_label: str, schema_version: int) -> None:
    for table in _V5_ADDED_COLUMNS:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        normalized = " ".join((row[0] if row else "").split()).upper()
        checks = _V5_REQUIRED_CHECKS
        if schema_version >= 7:
            checks = tuple(check.replace("VOLTAGE_STATE IN ('FIXED','NONE','UNKNOWN')",
                                         "VOLTAGE_STATE IN ('FIXED','NONE','UNKNOWN','SWEEP')")
                           for check in checks)
        missing = [check for check in checks if check not in normalized]
        if missing:
            raise DatabaseCompatibilityError(
                f"Database '{path_label}' table '{table}' is missing the "
                f"schema-v5 condition-state constraints {missing}. "
                f"{_PRESERVE_GUIDANCE}"
            )


def _validate_required_schema(
    conn: sqlite3.Connection,
    *,
    path_label: str,
    schema_version: int,
) -> None:
    tables = _table_names(conn)
    for table, required in required_columns(schema_version).items():
        if table not in tables:
            raise DatabaseCompatibilityError(
                f"Database '{path_label}' reports silk schema version "
                f"{schema_version} but is missing required table "
                f"'{table}'. {_PRESERVE_GUIDANCE}"
            )
        missing = sorted(required - _columns(conn, table))
        if missing:
            raise DatabaseCompatibilityError(
                f"Database '{path_label}' table '{table}' is missing required "
                f"columns: {', '.join(missing)}. {_PRESERVE_GUIDANCE}"
            )
    _validate_v2_relationships(
        conn, path_label=path_label, schema_version=schema_version)
    if schema_version >= 5:
        _validate_v5_state_checks(conn, path_label=path_label, schema_version=schema_version)
    if schema_version >= 6:
        from silk.specs.aggregate import decode_aggregate, validate_attribution
        import json
        for row in conn.execute(
                'SELECT m.measurement_id, m.aggregate_json, s.metric, m.sim_run_id '
                'FROM spec_measurements m JOIN specs s ON m.spec_id=s.spec_id '
                'WHERE m.aggregate_json IS NOT NULL'):
            try:
                aggregate = decode_aggregate(json.loads(row[1]))
                if aggregate is None:
                    raise ValueError('ordinary observations require SQL NULL')
                validate_attribution(aggregate, row[2], row[3])
            except (TypeError, ValueError) as exc:
                raise DatabaseCompatibilityError(
                    f"Database '{path_label}' measurement {row[0]!r}: {exc}") from exc
    value_column = next(
        row for row in conn.execute('PRAGMA table_info("spec_measurements")')
        if row[1] == "value"
    )
    if bool(value_column[3]) != (schema_version == 2):
        constraint = "NOT NULL" if schema_version == 2 else "nullable"
        raise DatabaseCompatibilityError(
            f"Database '{path_label}' schema version {schema_version} requires "
            f"spec_measurements.value to be {constraint}. {_PRESERVE_GUIDANCE}"
        )


def _read_and_validate_metadata(
    conn: sqlite3.Connection,
    *,
    path_label: str,
    schema_version: int,
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
    actual_version = rows[0][2]
    if format_id != DATABASE_FORMAT:
        raise DatabaseCompatibilityError(
            f"Database '{path_label}' has format identifier {format_id!r}; "
            f"expected {DATABASE_FORMAT!r}. {_PRESERVE_GUIDANCE}"
        )
    if actual_version != schema_version:
        extra = ""
        if actual_version == 1:
            extra = f" {_V1_TO_V2_GUIDANCE}"
        elif actual_version == 2:
            extra = f" {_V2_TO_V3_GUIDANCE}"
        elif actual_version == 3:
            extra = f" {_V3_TO_V4_GUIDANCE}"
        elif actual_version == 5:
            extra = " Use silk.db.migration.migrate_v5_to_v6(source_path, dest_path); the source is left byte-identical."
        elif actual_version == 4:
            extra = f" {_V4_TO_V5_GUIDANCE}"
        if actual_version == 6 and schema_version == 7:
            extra = " Use silk.db.migration.migrate_v6_to_v7(source_path, dest_path); the source is left byte-identical."
        raise UnsupportedSchemaVersionError(
            f"Database '{path_label}' uses silk schema version "
            f"{actual_version!r}; this release supports schema version "
            f"{schema_version}. {_PRESERVE_GUIDANCE}{extra}"
        )


def validate_database_connection(
    conn: sqlite3.Connection,
    *,
    path_label: str,
) -> None:
    """Validate format identity, version, and the required current shape."""
    _validate_database_version(
        conn, path_label=path_label, schema_version=DATABASE_SCHEMA_VERSION
    )


def _validate_database_version(
    conn: sqlite3.Connection, *, path_label: str, schema_version: int
) -> None:
    """Validate historical versions only for explicit migrations; normal opens require v7."""
    if schema_version not in (2, 3, 4, 5, 6, 7):
        raise ValueError("Only database versions 2 to 7 share this schema")

    try:
        _read_and_validate_metadata(
            conn, path_label=path_label, schema_version=schema_version
        )
        _validate_required_schema(
            conn, path_label=path_label, schema_version=schema_version
        )
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
    "required_columns",
    "inspect_existing_database",
    "validate_database_connection",
]
