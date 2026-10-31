"""Explicit schema v1-to-v2, v2-to-v3, v3-to-v4 and v4-to-v5 copy migrations.

The v1-to-v2 copy introduces characterization profiles.
The v2-to-v3 copy changes only measurement nullability.
The v3-to-v4 copy only adds the run_provenance table and a
nullable provenance_id column on simulation_results, spec_measurements and
net_current_sets; every existing row keeps provenance_id NULL (no record is
ever invented for evidence that predates provenance).
The v4-to-v5 copy adds the per-row condition states: a non-NULL
corner/temperature/voltage becomes ``'fixed'`` and a NULL one ``'unknown'``,
never ``'none'`` (none is a proof made at launch, and a v4 row keeps no deck
to re-prove it from). A NULL temperature on a ``simulation_results`` row
whose recorded operating conditions say ``temperature_kind == 'sweep'``
becomes ``'sweep'``.
Normal ``Database`` opens reject both older versions without mutation
(``silk.db.schema``) and point here instead.

Guarantees:

- The source file is opened read-only and is never mutated. Its bytes are
  identical before and after the call.
- The destination is a distinct, caller-selected path that must not already
  exist. It is built at a private temporary path and atomically published without overwriting
  a destination, only after the copy passes its target-version validation and
  ``PRAGMA foreign_key_check``; a failed migration leaves no partial file at
  the destination path.
- Existing UUIDs (``version_id``, ``device_id``, ``spec_id``,
  ``measurement_id``, ``current_id``, ...) and every existing fingerprint
  scheme/digest pair are copied byte-for-byte, never recomputed.
- In v1-to-v2, no ``characterization_profiles`` rows are created — v1 carries no
  authoritative owner/test/parameter descriptor. Migrated
  ``simulation_results``/``spec_measurements``/``net_current_sets`` rows are
  therefore explicitly unscoped (``profile_id IS NULL``), never given an
  invented test identity.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Tuple

from silk.db.circuit_db import Database
from silk.db.identity import STRUCTURE_FINGERPRINT_SCHEME
from silk.db.schema import (
    DATABASE_FORMAT,
    _validate_database_version,
    validate_database_connection,
)
from silk.exceptions import DatabaseCompatibilityError, EvidenceMismatchError


_V1_SCHEMA_VERSION = 1

# The v1 physical shape this migration accepts as its source. Frozen here
# (not imported from ``silk.db.schema``, whose current version is v3) because
# accepting a v1 file is this module's own, permanent contract.
_V1_REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "silk_metadata": (
        "singleton", "format_id", "schema_version", "created_at",
        "creator_version",
    ),
    "circuit_versions": (
        "version_id", "circuit_name", "version", "stage", "state",
        "is_locked", "parent_version_id", "created_at", "created_by",
        "description", "netlist_path", "metadata", "simulation_summary",
        "last_summary_update", "structure_scheme", "structure_hash",
        "spec_scheme", "spec_set_hash",
    ),
    "devices": (
        "device_id", "circuit_version_id", "name", "type", "model",
        "properties",
    ),
    "nets": ("net_id", "circuit_version_id", "name", "net_type", "properties"),
    "simulation_results": (
        "sim_id", "circuit_version_id", "object_id", "object_type", "source",
        "sim_type", "corner", "temperature", "voltage", "parameter", "value",
        "value_text", "unit", "tool_name", "timestamp", "conditions",
    ),
    "parasitics": (
        "parasitic_id", "circuit_version_id", "net_id", "parasitic_type",
        "value", "unit", "from_node", "to_node", "extraction_tool",
        "timestamp",
    ),
    "net_currents": (
        "current_id", "circuit_version_id", "net_name", "pin_id", "avg_a",
        "rms_a", "peak_a", "source", "current_vector", "analysis", "tstart",
        "results_file", "corner", "temperature", "voltage", "tool_name",
        "structure_scheme", "structure_hash", "timestamp",
    ),
    "releases": (
        "release_id", "release_name", "stage", "created_at", "created_by",
        "notes", "status",
    ),
    "release_contents": ("release_id", "circuit_version_id"),
    "modules": (
        "module_id", "module_name", "module_path", "module_hash_scheme",
        "module_hash", "device_count", "structure", "created_at",
    ),
    "circuit_modules": (
        "circuit_version_id", "module_id", "instance_name",
        "position_in_hierarchy",
    ),
    "specs": (
        "spec_id", "circuit_version_id", "owner_path", "spec_key", "name",
        "metric", "op", "target", "units", "description", "parent_id",
        "declared_at", "definition_scheme", "definition_hash", "created_at",
    ),
    "spec_measurements": (
        "measurement_id", "spec_id", "source", "value", "corner",
        "temperature", "voltage", "sim_type", "tool_name", "timestamp",
        "sim_run_id", "notes", "passes",
    ),
}

# Verbatim tables: identical column set in v1 and v2, copied unchanged.
_VERBATIM_TABLES: Tuple[str, ...] = (
    "circuit_versions",
    "devices",
    "nets",
    "parasitics",
    "releases",
    "release_contents",
    "modules",
    "circuit_modules",
    "specs",
)

_SIMULATION_RESULTS_COLUMNS = _V1_REQUIRED_COLUMNS["simulation_results"]

_NET_CURRENT_GROUP_CONTEXT_FIELDS = (
    "analysis",
    "tstart",
    "results_file",
    "temperature",
    "voltage",
    "tool_name",
    "structure_scheme",
    "structure_hash",
)


def _validate_v1_source(conn: sqlite3.Connection, *, path_label: str) -> None:
    """Read-only preflight: confirm ``conn`` is a genuine schema-v1 file."""
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    if "silk_metadata" not in tables:
        raise DatabaseCompatibilityError(
            f"Migration source '{path_label}' is not a silk database: "
            "required table 'silk_metadata' is absent."
        )
    metadata_columns = {
        row[1] for row in conn.execute('PRAGMA table_info("silk_metadata")')
    }
    missing = sorted(set(_V1_REQUIRED_COLUMNS["silk_metadata"]) - metadata_columns)
    if missing:
        raise DatabaseCompatibilityError(
            f"Migration source '{path_label}' table 'silk_metadata' is "
            f"missing required columns: {', '.join(missing)}."
        )
    rows = conn.execute(
        "SELECT singleton, format_id, schema_version FROM silk_metadata"
    ).fetchall()
    if len(rows) != 1 or rows[0][0] != 1:
        raise DatabaseCompatibilityError(
            f"Migration source '{path_label}' has invalid silk_metadata: "
            "expected one singleton row with singleton=1."
        )
    format_id, schema_version = rows[0][1], rows[0][2]
    if format_id != DATABASE_FORMAT:
        raise DatabaseCompatibilityError(
            f"Migration source '{path_label}' has format identifier "
            f"{format_id!r}; expected {DATABASE_FORMAT!r}."
        )
    if schema_version != _V1_SCHEMA_VERSION:
        raise DatabaseCompatibilityError(
            f"Migration source '{path_label}' uses silk schema version "
            f"{schema_version!r}; migrate_v1_to_v2 only accepts schema "
            f"version {_V1_SCHEMA_VERSION}."
        )
    for table, required in _V1_REQUIRED_COLUMNS.items():
        if table not in tables:
            raise DatabaseCompatibilityError(
                f"Migration source '{path_label}' is missing required v1 "
                f"table '{table}'."
            )
        actual = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        missing = sorted(set(required) - actual)
        if missing:
            raise DatabaseCompatibilityError(
                f"Migration source '{path_label}' table '{table}' is "
                f"missing required v1 columns: {', '.join(missing)}."
            )


def _copy_verbatim(
    src_conn: sqlite3.Connection, dest_conn: sqlite3.Connection, table: str
) -> int:
    columns = _V1_REQUIRED_COLUMNS[table]
    select_sql = f'SELECT {", ".join(columns)} FROM "{table}"'
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = (
        f'INSERT INTO "{table}" ({", ".join(columns)}) VALUES ({placeholders})'
    )
    rows = src_conn.execute(select_sql).fetchall()
    for row in rows:
        dest_conn.execute(insert_sql, tuple(row[c] for c in columns))
    return len(rows)


def _copy_simulation_results(
    src_conn: sqlite3.Connection, dest_conn: sqlite3.Connection
) -> int:
    columns = _SIMULATION_RESULTS_COLUMNS
    rows = src_conn.execute(
        f'SELECT {", ".join(columns)} FROM simulation_results'
    ).fetchall()
    insert_columns = columns + ("profile_id",)
    placeholders = ", ".join("?" for _ in insert_columns)
    insert_sql = (
        "INSERT INTO simulation_results "
        f'({", ".join(insert_columns)}) VALUES ({placeholders})'
    )
    for row in rows:
        dest_conn.execute(
            insert_sql, tuple(row[c] for c in columns) + (None,)
        )
    return len(rows)


def _copy_spec_measurements(
    src_conn: sqlite3.Connection, dest_conn: sqlite3.Connection
) -> int:
    total = src_conn.execute(
        "SELECT COUNT(*) FROM spec_measurements"
    ).fetchone()[0]
    rows = src_conn.execute(
        """
        SELECT sm.measurement_id, s.circuit_version_id, sm.spec_id,
               sm.source, sm.value, sm.corner, sm.temperature, sm.voltage,
               sm.sim_type, sm.tool_name, sm.timestamp, sm.sim_run_id,
               sm.notes, sm.passes
        FROM spec_measurements AS sm
        JOIN specs AS s ON s.spec_id = sm.spec_id
        """
    ).fetchall()
    if len(rows) != total:
        raise EvidenceMismatchError(
            "v1-to-v2 migration found spec_measurements rows whose spec_id "
            "does not join to any specs row (orphaned evidence); migration "
            "aborted."
        )
    insert_sql = """
        INSERT INTO spec_measurements
            (measurement_id, circuit_version_id, spec_id, profile_id,
             source, value, corner, temperature, voltage, sim_type,
             tool_name, timestamp, sim_run_id, notes, passes)
        VALUES (?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    for row in rows:
        dest_conn.execute(
            insert_sql,
            (
                row["measurement_id"],
                row["circuit_version_id"],
                row["spec_id"],
                row["source"],
                row["value"],
                row["corner"],
                row["temperature"],
                row["voltage"],
                row["sim_type"],
                row["tool_name"],
                row["timestamp"],
                row["sim_run_id"],
                row["notes"],
                row["passes"],
            ),
        )
    return len(rows)


def _grouped_net_currents(src_conn: sqlite3.Connection):
    """Group v1 ``net_currents`` rows by ``(circuit_version_id, corner)`` and
    verify each group shares one consistent operating context. Every offending group and its inconsistent field(s) are
    collected and reported together — a corrupt migration aborts atomically
    and actionably, not on the first bad row found."""
    rows = src_conn.execute(
        f'SELECT {", ".join(_V1_REQUIRED_COLUMNS["net_currents"])} '
        "FROM net_currents ORDER BY circuit_version_id, corner, rowid"
    ).fetchall()

    groups: Dict[Tuple[str, str], list] = {}
    for row in rows:
        groups.setdefault((row["circuit_version_id"], row["corner"]), []).append(
            row
        )

    problems = []
    for (version_id, corner), group_rows in groups.items():
        first = group_rows[0]
        bad_fields = set()
        for field in _NET_CURRENT_GROUP_CONTEXT_FIELDS:
            expected = first[field]
            for row in group_rows[1:]:
                if row[field] != expected:
                    bad_fields.add(field)
                    break
        # Schema migration copies historical evidence without authorizing
        # its use as a current live-design identity.
        if first["structure_scheme"] not in (
            "silk.structure/v1", "silk.structure/v2", STRUCTURE_FINGERPRINT_SCHEME,
        ):
            bad_fields.add(
                f"structure_scheme (unsupported: {first['structure_scheme']!r})"
            )
        if bad_fields:
            problems.append((version_id, corner, sorted(bad_fields)))

    if problems:
        detail = "; ".join(
            f"(circuit_version_id={version_id!r}, corner={corner!r}) "
            f"inconsistent field(s): {fields}"
            for version_id, corner, fields in problems
        )
        raise EvidenceMismatchError(
            "v1-to-v2 migration found inconsistent net_currents evidence "
            f"groups — {detail}. Migration aborted atomically; no "
            "destination database was written."
        )

    return groups


def _copy_net_currents(
    src_conn: sqlite3.Connection, dest_conn: sqlite3.Connection
) -> int:
    groups = _grouped_net_currents(src_conn)
    total_rows = 0
    for (version_id, corner), group_rows in groups.items():
        first = group_rows[0]
        timestamps = [row["timestamp"] for row in group_rows if row["timestamp"] is not None]
        new_set_id = str(uuid.uuid4())
        if timestamps:
            selected_at = max(timestamps)
            dest_conn.execute(
                """
                INSERT INTO net_current_sets
                    (current_set_id, circuit_version_id, profile_id, state,
                     corner, temperature, voltage, tool_name, analysis,
                     tstart, results_file, structure_scheme, structure_hash,
                     selected_at)
                VALUES (?, ?, NULL, 'current', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_set_id, version_id, corner, first["temperature"],
                    first["voltage"], first["tool_name"], first["analysis"],
                    first["tstart"], first["results_file"],
                    first["structure_scheme"], first["structure_hash"],
                    selected_at,
                ),
            )
        else:
            dest_conn.execute(
                """
                INSERT INTO net_current_sets
                    (current_set_id, circuit_version_id, profile_id, state,
                     corner, temperature, voltage, tool_name, analysis,
                     tstart, results_file, structure_scheme, structure_hash)
                VALUES (?, ?, NULL, 'current', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_set_id, version_id, corner, first["temperature"],
                    first["voltage"], first["tool_name"], first["analysis"],
                    first["tstart"], first["results_file"],
                    first["structure_scheme"], first["structure_hash"],
                ),
            )
        for row in group_rows:
            dest_conn.execute(
                """
                INSERT INTO net_currents
                    (current_id, current_set_id, net_name, pin_id, avg_a,
                     rms_a, peak_a, source, current_vector, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                """,
                (
                    row["current_id"], new_set_id, row["net_name"],
                    row["pin_id"], row["avg_a"], row["rms_a"], row["peak_a"],
                    row["source"], row["current_vector"], row["timestamp"],
                ),
            )
            total_rows += 1
    return total_rows


class _V2MigrationDatabase(Database):
    """New destinations for the permanent v1-to-v2 API, never normal opens."""

    def _initialize_schema(self):
        super()._initialize_schema(schema_version=2)


def _publish_copy(tmp_dest: Path, dest: Path) -> None:
    # Both paths share a directory/filesystem. Linking publishes the complete
    # file atomically and refuses an existing destination, including a racer.
    os.link(tmp_dest, dest)
    tmp_dest.unlink()


def migrate_v1_to_v2(source_path, dest_path) -> str:
    """Copy a schema-v1 database into a new schema-v2 database.

    The source is opened read-only and is never modified. The destination must
    be a distinct path that does not exist yet, in an existing directory. The
    copy is validated in a private temporary file and published atomically
    without overwriting, so a failure leaves no partial file at the destination.

    Existing simulation results and spec measurements are copied as unscoped
    (no characterization profile), and the v1 net-current rows are grouped into
    one unscoped current set per circuit version and corner. No
    characterization profiles are created, because v1 holds no authoritative
    descriptor.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path of the v1 database file.
    dest_path : str or pathlib.Path
        Path of the v2 database to create.

    Returns
    -------
    str
        ``str(dest_path)``.

    Raises
    ------
    silk.exceptions.DatabaseCompatibilityError
        The source is missing, empty or not a schema-v1 silk database, the
        destination directory does not exist, the copy is not consistent, or the
        source changed during migration.
    FileExistsError
        The destination already exists.

    See Also
    --------
    silk.db.migrate_v2_to_v3 : The next step.
    """
    source = Path(source_path)
    dest = Path(dest_path)

    if not source.exists() or source.stat().st_size == 0:
        raise DatabaseCompatibilityError(
            f"Migration source '{source}' does not exist or is empty."
        )
    if os.path.lexists(dest):
        raise FileExistsError(
            f"Migration destination '{dest}' already exists; "
            "migrate_v1_to_v2 requires a distinct, non-existing destination "
            "path."
        )
    if not dest.parent.exists():
        raise DatabaseCompatibilityError(
            f"Migration destination directory '{dest.parent}' does not "
            "exist."
        )

    source_resolved = source.resolve()
    source_uri = f"{source_resolved.as_uri()}?mode=ro"
    src_conn = sqlite3.connect(source_uri, uri=True)
    src_conn.row_factory = sqlite3.Row

    tmp_dest = dest.parent / f".{dest.name}.migrating-{uuid.uuid4().hex}"
    dest_db = None
    try:
        _validate_v1_source(src_conn, path_label=str(source_resolved))
        before_bytes = source.read_bytes()

        if tmp_dest.exists():
            tmp_dest.unlink()

        dest_db = _V2MigrationDatabase(str(tmp_dest))
        with dest_db.transaction():
            dest_db.conn.execute("PRAGMA defer_foreign_keys = ON")
            for table in _VERBATIM_TABLES:
                _copy_verbatim(src_conn, dest_db.conn, table)
            _copy_simulation_results(src_conn, dest_db.conn)
            _copy_spec_measurements(src_conn, dest_db.conn)
            _copy_net_currents(src_conn, dest_db.conn)
            # §7 step 4: no characterization_profiles rows are ever created
            # by this migration — v1 carries no authoritative descriptor.

        violations = dest_db.conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise DatabaseCompatibilityError(
                "v1-to-v2 migration produced a foreign-key-inconsistent "
                f"destination: {list(violations)}."
            )
        _validate_database_version(
            dest_db.conn, path_label=str(tmp_dest.resolve()), schema_version=2
        )
        profile_count = dest_db.conn.execute(
            "SELECT COUNT(*) FROM characterization_profiles"
        ).fetchone()[0]
        if profile_count != 0:
            raise DatabaseCompatibilityError(
                "v1-to-v2 migration unexpectedly created "
                f"{profile_count} characterization_profiles row(s)."
            )

        dest_db.close()
        dest_db = None

        after_bytes = source.read_bytes()
        if before_bytes != after_bytes:
            raise DatabaseCompatibilityError(
                f"Migration source '{source}' was modified during "
                "migration; the destination was not published."
            )

        _publish_copy(tmp_dest, dest)
    except BaseException:
        if dest_db is not None:
            dest_db.close()
        if tmp_dest.exists():
            tmp_dest.unlink()
        raise
    finally:
        src_conn.close()

    return str(dest)


def _make_measurements_nullable(conn: sqlite3.Connection) -> None:
    """Rebuild the single changed table inside the unpublished v2 backup."""
    objects = conn.execute(
        "SELECT sql FROM sqlite_master WHERE tbl_name = 'spec_measurements' "
        "AND type IN ('index', 'trigger') AND sql IS NOT NULL"
    ).fetchall()
    Database._create_spec_measurements_table(
        conn, schema_version=3, table_name="_silk_spec_measurements_v3"
    )
    old_columns = conn.execute(
        "PRAGMA table_info(spec_measurements)"
    ).fetchall()
    new_columns = conn.execute(
        "PRAGMA table_info(_silk_spec_measurements_v3)"
    ).fetchall()
    expected_columns = [
        (*row[:3], 0, *row[4:]) if row[1] == "value" else tuple(row)
        for row in old_columns
    ]
    if expected_columns != [tuple(row) for row in new_columns]:
        raise DatabaseCompatibilityError(
            "v2-to-v3 migration requires the standard spec_measurements "
            "columns; refusing to reshape additional columns or constraints."
        )
    columns = ", ".join(f'"{row[1]}"' for row in old_columns)
    conn.execute(
        f"INSERT INTO _silk_spec_measurements_v3 (rowid, {columns}) "
        f"SELECT rowid, {columns} FROM spec_measurements"
    )
    conn.execute("DROP TABLE spec_measurements")
    conn.execute(
        "ALTER TABLE _silk_spec_measurements_v3 RENAME TO spec_measurements"
    )
    for (sql,) in objects:
        conn.execute(sql)


def migrate_v2_to_v3(source_path: str | Path, dest_path: str | Path) -> str:
    """Copy a schema-v2 database into a new schema-v3 database.

    The source is opened read-only and is never modified. The destination must
    be a distinct path that does not exist yet, in an existing directory. The
    copy is validated in a private temporary file and published atomically
    without overwriting, so a failure leaves no partial file at the destination.

    All rows (including rowids), profile identities, foreign keys and indexes
    survive. Only ``spec_measurements.value`` loses its NOT NULL constraint, so
    an unavailable measurement can be stored.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path of the v2 database file.
    dest_path : str or pathlib.Path
        Path of the v3 database to create.

    Returns
    -------
    str
        ``str(dest_path)``.

    Raises
    ------
    silk.exceptions.UnsupportedSchemaVersionError
        The source is a silk database of another schema version.
    silk.exceptions.DatabaseCompatibilityError
        The source is missing, empty or not a readable silk database, the
        destination directory does not exist, the copy is not consistent, or the
        source changed during migration.
    FileExistsError
        The destination already exists.

    See Also
    --------
    silk.db.migrate_v3_to_v4 : The next step.
    """
    source = Path(source_path)
    dest = Path(dest_path)
    if not source.exists() or source.stat().st_size == 0:
        raise DatabaseCompatibilityError(
            f"Migration source '{source}' does not exist or is empty."
        )
    if os.path.lexists(dest):
        raise FileExistsError(
            f"Migration destination '{dest}' already exists; "
            "migrate_v2_to_v3 requires a distinct, non-existing destination path."
        )
    if not dest.parent.exists():
        raise DatabaseCompatibilityError(
            f"Migration destination directory '{dest.parent}' does not exist."
        )

    before_bytes = source.read_bytes()
    src_conn = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    tmp_dest = None
    dest_conn = None
    try:
        src_conn.execute("BEGIN")
        _validate_database_version(
            src_conn, path_label=str(source.resolve()), schema_version=2
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{dest.name}.migrating-", dir=dest.parent
        )
        os.close(fd)
        tmp_dest = Path(tmp_name)
        dest_conn = sqlite3.connect(tmp_dest)
        src_conn.backup(dest_conn)
        # SQLite's table-rebuild procedure needs FK enforcement disabled
        # outside the transaction; the full graph is checked before publish.
        dest_conn.execute("PRAGMA foreign_keys = OFF")
        with dest_conn:
            _make_measurements_nullable(dest_conn)
            dest_conn.execute("UPDATE silk_metadata SET schema_version = 3")
        dest_conn.execute("PRAGMA foreign_keys = ON")
        _validate_database_version(
            dest_conn, path_label=str(tmp_dest), schema_version=3)
        violations = dest_conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise DatabaseCompatibilityError(
                "v2-to-v3 migration produced a foreign-key-inconsistent "
                f"destination: {violations}."
            )
        dest_conn.close()
        dest_conn = None
        if source.read_bytes() != before_bytes:
            raise DatabaseCompatibilityError(
                f"Migration source '{source}' was modified during migration; "
                "the destination was not published."
            )
        _publish_copy(tmp_dest, dest)
    finally:
        if dest_conn is not None:
            dest_conn.close()
        src_conn.close()
        if tmp_dest is not None and tmp_dest.exists():
            tmp_dest.unlink()
    return str(dest)


def migrate_v3_to_v4(source_path: str | Path, dest_path: str | Path) -> str:
    """Copy a schema-v3 database into a new schema-v4 database with run provenance.

    The source is opened read-only and is never modified. The destination must
    be a distinct path that does not exist yet, in an existing directory. The
    copy is validated in a private temporary file and published atomically
    without overwriting, so a failure leaves no partial file at the destination.

    All rows (including rowids), identities, foreign keys and indexes survive.
    The copy gains an empty ``run_provenance`` table and a nullable
    ``provenance_id`` column on ``simulation_results``, ``spec_measurements``
    and ``net_current_sets``, NULL on every existing row.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path of the v3 database file.
    dest_path : str or pathlib.Path
        Path of the v4 database to create.

    Returns
    -------
    str
        ``str(dest_path)``.

    Raises
    ------
    silk.exceptions.UnsupportedSchemaVersionError
        The source is a silk database of another schema version.
    silk.exceptions.DatabaseCompatibilityError
        The source is missing, empty or not a readable silk database, the
        destination directory does not exist, the copy is not consistent, or the
        source changed during migration.
    FileExistsError
        The destination already exists.

    See Also
    --------
    silk.db.migrate_v4_to_v5 : The next step.
    """
    from silk.db.circuit_db import _PROVENANCE_COLUMN_SQL, _PROVENANCE_TABLES

    source = Path(source_path)
    dest = Path(dest_path)
    if not source.exists() or source.stat().st_size == 0:
        raise DatabaseCompatibilityError(
            f"Migration source '{source}' does not exist or is empty."
        )
    if os.path.lexists(dest):
        raise FileExistsError(
            f"Migration destination '{dest}' already exists; "
            "migrate_v3_to_v4 requires a distinct, non-existing destination path."
        )
    if not dest.parent.exists():
        raise DatabaseCompatibilityError(
            f"Migration destination directory '{dest.parent}' does not exist."
        )

    before_bytes = source.read_bytes()
    src_conn = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    tmp_dest = None
    dest_conn = None
    try:
        src_conn.execute("BEGIN")
        _validate_database_version(
            src_conn, path_label=str(source.resolve()), schema_version=3
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{dest.name}.migrating-", dir=dest.parent
        )
        os.close(fd)
        tmp_dest = Path(tmp_name)
        dest_conn = sqlite3.connect(tmp_dest)
        src_conn.backup(dest_conn)
        with dest_conn:
            Database._create_run_provenance_table(dest_conn)
            for table in _PROVENANCE_TABLES:
                dest_conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {_PROVENANCE_COLUMN_SQL}")
            dest_conn.execute("UPDATE silk_metadata SET schema_version = 4")
        dest_conn.execute("PRAGMA foreign_keys = ON")
        _validate_database_version(
            dest_conn, path_label=str(tmp_dest), schema_version=4)
        violations = dest_conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise DatabaseCompatibilityError(
                "v3-to-v4 migration produced a foreign-key-inconsistent "
                f"destination: {violations}."
            )
        dest_conn.close()
        dest_conn = None
        if source.read_bytes() != before_bytes:
            raise DatabaseCompatibilityError(
                f"Migration source '{source}' was modified during migration; "
                "the destination was not published."
            )
        _publish_copy(tmp_dest, dest)
    finally:
        if dest_conn is not None:
            dest_conn.close()
        src_conn.close()
        if tmp_dest is not None and tmp_dest.exists():
            tmp_dest.unlink()
    return str(dest)


# --- v4 -> v5 (TASK-409) ----------------------------------------------------

_V5_STATE_TABLES = ("simulation_results", "spec_measurements", "net_current_sets")
_V5_STATE_COLUMNS = ("corner_state", "temperature_state", "voltage_state")
_OPERATING_CONDITIONS_KEY = "silk_operating_conditions"


def _v4_conditions_marker(conditions_json):
    """The decoded ``silk_operating_conditions`` marker of a v4 row, or None."""
    if conditions_json is None:
        return None
    try:
        decoded = json.loads(conditions_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    marker = decoded.get(_OPERATING_CONDITIONS_KEY)
    return marker if isinstance(marker, dict) else None


def _v5_conditions_json(conditions_json):
    """Add the v5 marker keys derivable from a v4 marker (design §5.2).

    ``corner_kind`` / ``supply_kind`` are ``'fixed'`` when the marker's value
    is set, else ``'unknown'``; ``corner_basis`` is ``'legacy'`` for a fixed
    corner (never recorded before v5) and ``'unknown'`` otherwise. Keys a
    marker already has are kept, and text without a marker is copied
    byte-for-byte.
    """
    marker = _v4_conditions_marker(conditions_json)
    if marker is None:
        return conditions_json
    added = {}
    if "corner_kind" not in marker:
        added["corner_kind"] = (
            "fixed" if marker.get("corner") is not None else "unknown"
        )
    if "corner_basis" not in marker:
        corner_kind = marker.get("corner_kind", added.get("corner_kind"))
        added["corner_basis"] = "legacy" if corner_kind == "fixed" else "unknown"
    if "supply_kind" not in marker:
        added["supply_kind"] = (
            "fixed" if marker.get("supply_voltage_v") is not None else "unknown"
        )
    if not added:
        return conditions_json
    decoded = json.loads(conditions_json)
    decoded[_OPERATING_CONDITIONS_KEY] = {**marker, **added}
    return Database._canonical_simulation_conditions(decoded)


def _v5_states(table, row):
    """The §5.3 states of one v4 row: NULL is unknown, never none."""
    corner_state = "fixed" if row["corner"] is not None else "unknown"
    if row["temperature"] is not None:
        temperature_state = "fixed"
    else:
        marker = (
            _v4_conditions_marker(row["conditions"])
            if table == "simulation_results"
            else None
        )
        temperature_state = (
            "sweep"
            if marker is not None and marker.get("temperature_kind") == "sweep"
            else "unknown"
        )
    voltage_state = "fixed" if row["voltage"] is not None else "unknown"
    return corner_state, temperature_state, voltage_state


def _copy_table_v4_to_v5(src_conn, dest_conn, table) -> None:
    source_columns = [
        r[1] for r in src_conn.execute(f'PRAGMA table_info("{table}")')
    ]
    dest_columns = [
        r[1] for r in dest_conn.execute(f'PRAGMA table_info("{table}")')
    ]
    added = _V5_STATE_COLUMNS if table in _V5_STATE_TABLES else ()
    if set(dest_columns) != set(source_columns) | set(added) or (
        set(added) & set(source_columns)
    ):
        raise DatabaseCompatibilityError(
            f"v4-to-v5 migration requires the standard '{table}' columns; "
            "refusing to reshape additional or missing columns."
        )
    insert_columns = ["rowid", *source_columns, *added]
    insert_sql = (
        f'INSERT INTO "{table}" ('
        + ", ".join(f'"{c}"' for c in insert_columns)
        + ") VALUES ("
        + ", ".join("?" for _ in insert_columns)
        + ")"
    )
    quoted = ", ".join(f'"{c}"' for c in source_columns)
    for row in src_conn.execute(
        f'SELECT rowid AS "_silk_rowid", {quoted} FROM "{table}" ORDER BY rowid'
    ):
        values = [row["_silk_rowid"], *(row[c] for c in source_columns)]
        if added:
            values.extend(_v5_states(table, row))
            if table == "simulation_results":
                index = insert_columns.index("conditions")
                values[index] = _v5_conditions_json(row["conditions"])
        dest_conn.execute(insert_sql, values)


def migrate_v4_to_v5(source_path: str | Path, dest_path: str | Path) -> str:
    """Copy a schema-v4 database into a new schema-v5 database with condition states.

    The source is opened read-only and is never modified. The destination must
    be a distinct path that does not exist yet, in an existing directory. The
    copy is validated in a private temporary file and published atomically
    without overwriting, so a failure leaves no partial file at the destination.

    The destination is built as a fresh v5 database, so its table definitions
    equal a new file's. Every row is copied with its rowid.
    ``simulation_results``, ``spec_measurements`` and ``net_current_sets`` gain
    ``corner_state``, ``temperature_state`` and ``voltage_state``: a non-NULL
    value becomes ``'fixed'`` (value kept), NULL becomes ``'unknown'``, and a
    NULL temperature on a simulation result recorded as a temperature sweep
    becomes ``'sweep'``. Nothing becomes ``'none'``. Each recorded
    ``silk_operating_conditions`` marker gains the ``corner_kind``,
    ``corner_basis`` and ``supply_kind`` keys derivable from it.
    Characterization profiles and every other table are copied unchanged, so
    every profile keeps its identity and its evidence its selection key.
    Indexes and triggers the source adds beyond silk's own are recreated.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path of the v4 database file.
    dest_path : str or pathlib.Path
        Path of the v5 database to create.

    Returns
    -------
    str
        ``str(dest_path)``.

    Raises
    ------
    silk.exceptions.UnsupportedSchemaVersionError
        The source is a silk database of another schema version.
    silk.exceptions.DatabaseCompatibilityError
        The source is missing, empty or not a readable silk database, the
        destination directory does not exist, the copy is not consistent, or the
        source changed during migration.
    FileExistsError
        The destination already exists.

    See Also
    --------
    silk.db.migrate_v5_to_v6 : The next step.
    """
    source = Path(source_path)
    dest = Path(dest_path)
    if not source.exists() or source.stat().st_size == 0:
        raise DatabaseCompatibilityError(
            f"Migration source '{source}' does not exist or is empty."
        )
    if os.path.lexists(dest):
        raise FileExistsError(
            f"Migration destination '{dest}' already exists; "
            "migrate_v4_to_v5 requires a distinct, non-existing destination path."
        )
    if not dest.parent.exists():
        raise DatabaseCompatibilityError(
            f"Migration destination directory '{dest.parent}' does not exist."
        )

    before_bytes = source.read_bytes()
    src_conn = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    src_conn.row_factory = sqlite3.Row
    tmp_dest = None
    dest_db = None
    try:
        src_conn.execute("BEGIN")
        _validate_database_version(
            src_conn, path_label=str(source.resolve()), schema_version=4
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{dest.name}.migrating-", dir=dest.parent
        )
        os.close(fd)
        tmp_dest = Path(tmp_name)
        dest_db = _V5MigrationDatabase(str(tmp_dest))
        dest_conn = dest_db.conn
        dest_tables = {
            r[0] for r in dest_conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        source_objects = src_conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        ).fetchall()
        with dest_db.transaction():
            dest_conn.execute("PRAGMA defer_foreign_keys = ON")
            metadata = src_conn.execute(
                "SELECT created_at, creator_version FROM silk_metadata"
            ).fetchone()
            dest_conn.execute(
                "UPDATE silk_metadata SET created_at = ?, creator_version = ?",
                (metadata["created_at"], metadata["creator_version"]),
            )
            for obj in source_objects:
                if obj["type"] != "table" or obj["name"] == "silk_metadata":
                    continue
                if obj["name"] not in dest_tables:
                    # A table silk does not define: carry it over unchanged.
                    dest_conn.execute(obj["sql"])
                _copy_table_v4_to_v5(src_conn, dest_conn, obj["name"])
            dest_objects = {
                r[0] for r in dest_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('index', 'trigger')")
            }
            for obj in source_objects:
                if obj["type"] in ("index", "trigger") and obj["name"] not in dest_objects:
                    dest_conn.execute(obj["sql"])
        _validate_database_version(
            dest_conn, path_label=str(tmp_dest), schema_version=5)
        violations = dest_conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise DatabaseCompatibilityError(
                "v4-to-v5 migration produced a foreign-key-inconsistent "
                f"destination: {[tuple(v) for v in violations]}."
            )
        dest_db.close()
        dest_db = None
        if source.read_bytes() != before_bytes:
            raise DatabaseCompatibilityError(
                f"Migration source '{source}' was modified during migration; "
                "the destination was not published."
            )
        _publish_copy(tmp_dest, dest)
    finally:
        if dest_db is not None:
            dest_db.close()
        src_conn.close()
        if tmp_dest is not None and tmp_dest.exists():
            tmp_dest.unlink()
    return str(dest)


__all__ = [
    "migrate_v1_to_v2",
    "migrate_v2_to_v3",
    "migrate_v3_to_v4",
    "migrate_v4_to_v5",
    "migrate_v5_to_v6",
    "migrate_v6_to_v7",
]


def migrate_v5_to_v6(source_path: str | Path, dest_path: str | Path) -> str:
    """Copy a schema-v5 database into a new schema-v6 database with aggregate evidence.

    The source is opened read-only and is never modified. The destination must
    be a distinct path that does not exist yet, in an existing directory. The
    copy is validated in a private temporary file and published atomically
    without overwriting, so a failure leaves no partial file at the destination.

    Ids, profiles, seals, conditions, provenance and history are unchanged. The
    only schema change is a nullable ``aggregate_json`` column on
    ``spec_measurements``, which stores the Monte Carlo attribution of an
    aggregate measurement. The destination is checked for integrity and
    foreign-key consistency before publication.

    Parameters
    ----------
    source_path : str or pathlib.Path
        Path of the v5 database file.
    dest_path : str or pathlib.Path
        Path of the v6 database to create.

    Returns
    -------
    str
        ``str(dest_path)``.

    Raises
    ------
    silk.exceptions.UnsupportedSchemaVersionError
        The source is a silk database of another schema version.
    silk.exceptions.DatabaseCompatibilityError
        The source is missing or empty, the destination directory does not
        exist, the copy fails its integrity or foreign-key check, or the source
        changed during migration.
    FileExistsError
        The destination already exists.
    """
    return _migrate_condition_copy(source_path, dest_path, source_version=5)


def migrate_v6_to_v7(source_path: str | Path, dest_path: str | Path) -> str:
    """Copy a v6 database to v7, permitting unscoped supply-sweep evidence.

    Only the voltage-state CHECK on the three evidence tables is widened.
    All historical rows, provenance and unknown tables are preserved; the
    source is unchanged and an existing destination is never overwritten.
    """
    return _migrate_condition_copy(source_path, dest_path, source_version=6)


def _migrate_condition_copy(source_path, dest_path, *, source_version):
    """Validated copy/publication shared by the v5 and v6 upgrades."""
    source, dest = Path(source_path), Path(dest_path)
    if not source.exists() or source.stat().st_size == 0:
        raise DatabaseCompatibilityError(f'Migration source {str(source)!r} is missing or empty')
    if os.path.lexists(dest):
        raise FileExistsError(f'Migration destination {str(dest)!r} already exists')
    if not dest.parent.exists():
        raise DatabaseCompatibilityError(f'Destination directory {str(dest.parent)!r} does not exist')
    before = source.read_bytes()
    src = sqlite3.connect(f'{source.resolve().as_uri()}?mode=ro', uri=True)
    temporary = None
    target = None
    try:
        _validate_database_version(src, path_label=str(source), schema_version=source_version)
        fd, name = tempfile.mkstemp(prefix=f'.{dest.name}.migrating-', dir=dest.parent)
        os.close(fd)
        temporary = Path(name)
        target = sqlite3.connect(temporary)
        src.backup(target)
        if source_version == 6:
            target.execute('PRAGMA foreign_keys=OFF')
            # Rebuilds temporarily remove the referenced table. Keep dependent
            # views/triggers intact instead of revalidating them mid-rename.
            target.execute('PRAGMA legacy_alter_table=ON')
        with target:
            if source_version == 5:
                target.execute('ALTER TABLE spec_measurements ADD COLUMN aggregate_json TEXT')
            else:
                # Rebuild only affected tables; retain their original columns,
                # foreign keys, extra constraints, rowids, indexes and triggers.
                import re
                for table in ('simulation_results', 'spec_measurements', 'net_current_sets'):
                    sql = target.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone()[0]
                    attachments = target.execute(
                        "SELECT sql FROM sqlite_master WHERE tbl_name=? "
                        "AND type IN ('index','trigger') AND sql IS NOT NULL", (table,)).fetchall()
                    sql, count = re.subn(
                        r"(?i)voltage_state\s+IN\s*\(\s*'fixed'\s*,\s*'none'\s*,\s*'unknown'\s*\)",
                        "voltage_state IN ('fixed','none','unknown','sweep')", sql)
                    if count != 1:
                        raise DatabaseCompatibilityError(
                            f'Cannot widen the voltage-state CHECK on {table!r}')
                    temporary_table = '_silk_v6_' + table
                    if target.execute(
                            'SELECT 1 FROM sqlite_master WHERE name=? COLLATE NOCASE',
                            (temporary_table,)).fetchone():
                        raise DatabaseCompatibilityError(
                            f'Cannot rebuild {table!r}: private name {temporary_table!r} '
                            'already exists; no destination was published.')
                    columns = ', '.join('"' + row[1] + '"' for row in
                                        target.execute(f'PRAGMA table_info("{table}")'))
                    # Retain the original CREATE's identifier spelling and every
                    # other constraint. Only the contracted state clause changes.
                    target.execute(f'ALTER TABLE "{table}" RENAME TO "{temporary_table}"')
                    target.execute(sql)
                    target.execute(f'INSERT INTO "{table}" (rowid, {columns}) '
                                   f'SELECT rowid, {columns} FROM "{temporary_table}"')
                    target.execute(f'DROP TABLE "{temporary_table}"')
                    for (statement,) in attachments:
                        target.execute(statement)
            target.execute('UPDATE silk_metadata SET schema_version=?', (source_version + 1,))
        target.execute('PRAGMA legacy_alter_table=OFF')
        _validate_database_version(target, path_label=str(temporary), schema_version=source_version + 1)
        if target.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
            raise DatabaseCompatibilityError('Aggregate migration failed destination integrity check')
        if target.execute('PRAGMA foreign_key_check').fetchall():
            raise DatabaseCompatibilityError('Aggregate migration failed destination foreign-key check')
        target.close()
        target = None
        if source.read_bytes() != before:
            raise DatabaseCompatibilityError('Migration source changed; destination was not published')
        _publish_copy(temporary, dest)
    except sqlite3.DatabaseError as exc:
        raise DatabaseCompatibilityError(
            f'Condition migration failed; destination was not published: {exc}') from exc
    finally:
        if target is not None:
            target.close()
        src.close()
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return str(dest)


class _V5MigrationDatabase(Database):
    def _initialize_schema(self):
        super()._initialize_schema(schema_version=5)
